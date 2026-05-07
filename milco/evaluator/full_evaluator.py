"""Full-corpus evaluator for sparse bi-encoder retrieval.

1. Encodes all queries once into a sparse ``[n_q, V_en + V_mul]`` matrix.
2. Streams the corpus in document batches; each batch is encoded, converted
   to scipy CSR, and scored against all queries via ``Q @ D.T``. A per-query
   top-K heap is updated incrementally so we never materialise the full
   ``[n_q, n_d]`` score matrix.
3. Reduces top-Ks across ranks (when distributed) and feeds run_results
   into ``ir_measures``.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse
import torch
import torch.distributed as dist
import ir_measures
from ir_measures import nDCG, R
from tqdm import tqdm

from ..data.evaluation_data import BaseEvaluationDataset, MultiLanguageEvaluationDataset
from ..utils import master_print


def _torch_sparse_to_csr(sp: torch.Tensor, vocab_size: int) -> scipy.sparse.csr_matrix:
    """Convert a 2-D torch sparse COO tensor on CPU to a scipy CSR matrix.

    Coalesces first so duplicate (row, col) entries are summed (relevant for
    echo: repeated input_ids are scatter-added).
    """
    sp = sp.coalesce()
    rows, cols = sp.indices()
    rows = rows.cpu().numpy()
    cols = cols.cpu().numpy()
    vals = sp.values().float().cpu().numpy()
    n_rows = sp.size(0)
    return scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(n_rows, vocab_size))


class FullCorpusEvaluator:
    """Exhaustive sparse retrieval evaluator (no BM25 candidate stage)."""

    def __init__(
        self,
        model,
        batch_size: int = 64,
        query_max_length: int = 50,
        passage_max_length: int = 512,
        top_k: int = 1000,
        metrics: Optional[List[Any]] = None,
    ):
        self.model = model
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.top_k = top_k
        self.metrics = metrics or [nDCG @ 5, nDCG @ 10, nDCG @ 20, R @ 100, R @ 1000]

        self.vocab_size = (
            self.model.config.en_vocab_size + self.model.config.m_vocab_size
        )

    def _encode_to_csr(
        self, texts: List[str], batch_size: int, max_length: int, desc: str
    ) -> scipy.sparse.csr_matrix:
        """Encode texts into a CSR matrix of shape ``[len(texts), V_en + V_mul]``."""
        rows = []
        for start in tqdm(range(0, len(texts), batch_size), desc=desc):
            chunk = texts[start : start + batch_size]
            sp = self.model.encode_text(
                chunk, batch_size=batch_size, max_length=max_length
            )
            rows.append(_torch_sparse_to_csr(sp, self.vocab_size))
        if not rows:
            return scipy.sparse.csr_matrix((0, self.vocab_size), dtype=np.float32)
        return scipy.sparse.vstack(rows, format="csr")

    @staticmethod
    def _merge_topk(
        top_scores: np.ndarray,
        top_ids: np.ndarray,
        new_scores: np.ndarray,
        new_ids: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge a new ``[n_q, b]`` score block into the running top-K.

        Returns updated ``(top_scores, top_ids)`` of shape ``[n_q, k]``.
        """
        combined_scores = np.concatenate([top_scores, new_scores], axis=1)
        combined_ids = np.concatenate([top_ids, new_ids], axis=1)
        if combined_scores.shape[1] <= k:
            return combined_scores, combined_ids
        # argpartition is O(n) per row; argsort the partition for stable ordering
        part = np.argpartition(-combined_scores, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(combined_scores.shape[0])[:, None]
        return combined_scores[rows, part], combined_ids[rows, part]

    def evaluate(
        self, eval_dataset: BaseEvaluationDataset, return_run_results: bool = False
    ) -> Dict[str, float]:
        if not isinstance(eval_dataset, BaseEvaluationDataset):
            raise TypeError(
                f"Dataset must inherit from BaseEvaluationDataset. Got {type(eval_dataset)}"
            )

        query_ids = list(eval_dataset.queries.keys())
        query_texts = [eval_dataset.queries[qid] for qid in query_ids]
        n_q = len(query_ids)

        master_print(f"Encoding {n_q} queries...")
        Q = self._encode_to_csr(
            query_texts, self.batch_size, self.query_max_length, "Encoding queries"
        )

        doc_ids = list(eval_dataset.documents.keys())
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            chunk = math.ceil(len(doc_ids) / world_size)
            shard_doc_ids = doc_ids[rank * chunk : (rank + 1) * chunk]
            master_print(
                f"Distributed: {len(doc_ids)} docs sharded across {world_size} ranks "
                f"(rank {rank}: {len(shard_doc_ids)} docs)"
            )
        else:
            shard_doc_ids = doc_ids

        # Per-query running top-K. ``top_idx`` indexes into ``shard_doc_ids``;
        # we map to string doc_ids only at the end.
        k = self.top_k
        top_scores = np.full((n_q, k), -np.inf, dtype=np.float32)
        top_idx = np.full((n_q, k), -1, dtype=np.int64)

        b = self.batch_size
        n_batches = math.ceil(len(shard_doc_ids) / b)
        for bi in tqdm(range(n_batches), desc="Streaming docs"):
            start = bi * b
            end = min(start + b, len(shard_doc_ids))
            batch_ids = shard_doc_ids[start:end]
            batch_texts = [eval_dataset.documents[did] for did in batch_ids]
            D = self._encode_to_csr(
                batch_texts, b, self.passage_max_length, f"Batch {bi}"
            )
            # Q: [n_q, V] CSR; D.T: [V, b] CSC. Result is [n_q, b]; small in b
            # so it's safe to densify.
            scores = (Q @ D.T).toarray().astype(np.float32, copy=False)
            new_idx = np.broadcast_to(
                np.arange(start, end, dtype=np.int64)[None, :], (n_q, end - start)
            )
            top_scores, top_idx = self._merge_topk(
                top_scores, top_idx, scores, new_idx, k
            )

        # Convert local shard indices to string doc_ids
        local_doc_ids = np.array(shard_doc_ids, dtype=object)
        # Guard against -1 placeholders (when shard had fewer than k docs)
        valid = top_idx >= 0
        local_top_doc_ids = np.where(
            valid, np.where(valid, local_doc_ids[np.clip(top_idx, 0, None)], ""), ""
        )
        local_top_scores = np.where(valid, top_scores, -np.inf)

        if dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_scores = [None] * world_size
            gathered_ids = [None] * world_size
            dist.all_gather_object(gathered_scores, local_top_scores)
            dist.all_gather_object(gathered_ids, local_top_doc_ids)
            merged_scores = np.concatenate(gathered_scores, axis=1)
            merged_ids = np.concatenate(gathered_ids, axis=1)
        else:
            merged_scores = local_top_scores
            merged_ids = local_top_doc_ids

        # Final top-K per query across all ranks
        final_k = min(k, merged_scores.shape[1])
        part = np.argpartition(-merged_scores, kth=final_k - 1, axis=1)[:, :final_k]
        rows = np.arange(n_q)[:, None]
        final_scores = merged_scores[rows, part]
        final_ids = merged_ids[rows, part]

        run_results: Dict[str, Dict[str, float]] = {qid: {} for qid in query_ids}
        for qi, qid in enumerate(query_ids):
            for j in range(final_k):
                did = final_ids[qi, j]
                s = final_scores[qi, j]
                if did == "" or not np.isfinite(s):
                    continue
                run_results[qid][str(did)] = float(s)

        metrics = ir_measures.calc_aggregate(
            self.metrics, eval_dataset.qrels, run_results
        )
        metrics = {str(m): metrics[m] for m in metrics}
        if return_run_results:
            return metrics, run_results
        return metrics

    def evaluate_multi_language(
        self,
        multi_lang_dataset: MultiLanguageEvaluationDataset,
        languages: Optional[List[str]] = None,
        unload_after_each: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        if languages is None:
            languages = multi_lang_dataset.get_languages()

        all_metrics: Dict[str, Dict[str, float]] = {}
        for language in languages:
            master_print(f"\n=== Evaluating {language.upper()} (full corpus) ===")
            try:
                dataset = multi_lang_dataset.get_dataset(language)
                metrics = self.evaluate(dataset)
                all_metrics[language] = metrics
                master_print(f"Results for {language.upper()}:")
                for name, value in metrics.items():
                    master_print(f"  {name}: {value:.4f}")
                if unload_after_each:
                    dataset = None
                    import gc

                    gc.collect()
            except Exception as e:
                master_print(f"Error evaluating {language}: {e}")
                import traceback

                traceback.print_exc()
                all_metrics[language] = {}

        if len(languages) > 1:
            avg = self._compute_average_metrics(all_metrics)
            all_metrics["average"] = avg
            master_print(f"\n=== AVERAGE ACROSS {len(languages)} LANGUAGES ===")
            for name, value in avg.items():
                master_print(f"  {name}: {value:.4f}")

        return all_metrics

    @staticmethod
    def _compute_average_metrics(
        all_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        if not all_metrics:
            return {}
        names = set()
        for m in all_metrics.values():
            names.update(m.keys())
        avg = {}
        for name in names:
            values = [m.get(name, 0.0) for m in all_metrics.values() if m]
            if values:
                avg[name] = sum(values) / len(values)
        return avg

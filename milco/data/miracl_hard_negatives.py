from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from ..utils import master_print
from .evaluation_data import BaseEvaluationDataset


class MIRACLHardNegativesDataset(BaseEvaluationDataset):
    """
    Evaluation dataset for ``mteb/MIRACLRetrievalHardNegatives``.

    The corpus is pre-pruned to relevant docs + mined hard negatives across all
    queries, so it is small enough for full-corpus scoring with
    ``FullCorpusEvaluator``.
    """

    prompt = "Given a question, retrieve Wikipedia passages that answer the question."
    dataset_name = "mteb/MIRACLRetrievalHardNegatives"

    def __init__(
        self,
        language: str,
        use_prompt: bool = False,
        split: str = "dev",
        top_k_candidates: int = 100,
    ):
        super().__init__()
        self.language = language
        self.split = split
        self.top_k_candidates = top_k_candidates
        self.use_prompt = use_prompt

        self.queries: Dict[str, str] = {}
        self.documents: Dict[str, str] = {}
        self.qrels: Dict[str, Dict[str, int]] = {}
        self._candidates: Dict[str, List[str]] = defaultdict(list)

        self._load_queries()
        self._load_corpus()
        self._load_qrels()

    def _load_queries(self) -> None:
        master_print(f"Loading {self.dataset_name} queries for {self.language}")
        ds = load_dataset(
            self.dataset_name,
            f"{self.language}-queries",
            split=self.split,
        )
        for item in tqdm(ds, desc="Loading queries"):
            qid = str(item["id"])
            query = item["text"].lower()
            if self.use_prompt:
                query = f"Instruct: {self.prompt}\nQuery: {query}"
            self.queries[qid] = query
        master_print(f"Loaded {len(self.queries)} queries")

    def _load_corpus(self) -> None:
        master_print(f"Loading {self.dataset_name} corpus for {self.language}")
        ds = load_dataset(
            self.dataset_name,
            f"{self.language}-corpus",
            split=self.split,
        )
        for item in tqdm(ds, desc="Loading corpus"):
            did = str(item["id"])
            title = item.get("title") or ""
            text = item.get("text") or ""
            self.documents[did] = f"{title} {text}".strip().lower()
        master_print(f"Loaded {len(self.documents)} documents")

    def _load_qrels(self) -> None:
        master_print(
            f"Loading {self.dataset_name} qrels for {self.language} ({self.split})"
        )
        ds = load_dataset(
            self.dataset_name,
            f"{self.language}-qrels",
            split=self.split,
        )
        for item in tqdm(ds, desc="Loading qrels"):
            qid = str(item["query-id"])
            did = str(item["corpus-id"])
            score = int(item["score"])
            self.qrels.setdefault(qid, {})[did] = score
            self._candidates[qid].append(did)
        master_print(f"Loaded qrels for {len(self.qrels)} queries")

    def get_candidate_ids(self, query_id: str) -> List[Tuple[str, Optional[int], Optional[float]]]:
        cands = self._candidates.get(query_id, [])[: self.top_k_candidates]
        return [(did, None, None) for did in cands]

    def get_candidate_documents(self, query_id: str) -> Dict[str, str]:
        return {did: self.documents[did] for did, _, _ in self.get_candidate_ids(query_id)}

    def get_all_candidate_ids(self, query_ids: Optional[List[str]] = None) -> set:
        if query_ids is None:
            query_ids = list(self.queries.keys())
        all_ids = set()
        for qid in query_ids:
            all_ids.update(did for did, _, _ in self.get_candidate_ids(qid))
        return all_ids

    def get_statistics(self) -> Dict[str, Any]:
        total_candidates = sum(len(self.get_candidate_ids(qid)) for qid in self.queries)
        avg_candidates = total_candidates / len(self.queries) if self.queries else 0
        total_relevant = sum(
            sum(1 for rel in qrels.values() if rel > 0) for qrels in self.qrels.values()
        )
        avg_relevant = total_relevant / len(self.qrels) if self.qrels else 0
        return {
            "language": self.language,
            "split": self.split,
            "num_queries": len(self.queries),
            "num_documents": len(self.documents),
            "num_qrels_entries": sum(len(q) for q in self.qrels.values()),
            "total_candidates": total_candidates,
            "avg_candidates_per_query": avg_candidates,
            "total_relevant_docs": total_relevant,
            "avg_relevant_per_query": avg_relevant,
            "top_k_candidates": self.top_k_candidates,
        }

from transformers import PreTrainedModel, AutoModel, AutoModelForMaskedLM, AutoConfig, AutoTokenizer
from transformers.activations import GELUActivation
import torch.nn.functional as F
from transformers.utils import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, List
from ..config import MILCOConfig


logger = logging.get_logger(__name__)

PAD_TO_MULTIPLE_OF = 32  # tensor-core alignment for inference batches


class MILCOModel(PreTrainedModel):
    """Multilingual Sparse Retrieval model.

    Encodes multilingual text into sparse representations aligned with an English
    vocabulary space.  Supports two training modes (alignment and distillation) and
    provides a clean inference API (``encode_text``, ``encode_query``,
    ``encode_document``) that returns 2-D sparse COO tensors.

    Architecture
    ------------
    - ``m_model``: multilingual encoder (e.g. XLM-R, mBERT, Qwen3)
    - ``projector``: linear projection from multilingual hidden size → English hidden size
    - ``mlm_head``: MLM head from the LSR encoder; maps projected states to English vocab logits
    - ``en_lsr``: frozen English LSR encoder (alignment training only; deleted in ContrastiveMILCOModel)
    """

    config_class = MILCOConfig
    base_model_prefix = "milco"
    _tied_weights_keys = ["mlm_head.predictions.decoder.bias"]

    # ------------------------------------------------------------------
    # Sub-model construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_sub_model(auto_cls, checkpoint):
        """Instantiate a sub-model from a pretrained checkpoint."""
        return auto_cls.from_pretrained(checkpoint, trust_remote_code=True)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, config: MILCOConfig):
        super().__init__(config)

        self.m_model = self._init_sub_model(AutoModel, config.multilingual_encoder_checkpoint)

        lsr_for_head = self._init_sub_model(AutoModelForMaskedLM, config.lsr_encoder_checkpoint)
        self.mlm_head = lsr_for_head.cls if hasattr(lsr_for_head, "cls") else lsr_for_head.lm_head

        self.en_lsr = self._init_sub_model(AutoModelForMaskedLM, config.lsr_encoder_checkpoint)

        if config.en_hidden_size is None:
            config.en_hidden_size = self.en_lsr.config.hidden_size
        if config.m_hidden_size is None:
            config.m_hidden_size = self.m_model.config.hidden_size

        # --- projection and scoring layers ---
        self.projector = nn.Linear(config.m_hidden_size, config.en_hidden_size)
        self.activation = GELUActivation()
        self.loss_fn = nn.MSELoss(reduction="mean")

        self.post_init()
        self.freeze_params(self.en_lsr)

        if self.config.echo:
            self.echo = nn.Linear(config.m_hidden_size, 1)
            self.scale = nn.Parameter(torch.tensor([1.0]))

    # ------------------------------------------------------------------
    # Tokenizer properties (lazy, cached on first access)
    # ------------------------------------------------------------------

    @property
    def en_tokenizer(self):
        if not hasattr(self, "_en_tokenizer"):
            self._en_tokenizer = AutoTokenizer.from_pretrained(
                self.config.lsr_encoder_checkpoint, trust_remote_code=True
            )
        return self._en_tokenizer

    @property
    def m_tokenizer(self):
        if not hasattr(self, "_m_tokenizer"):
            self._m_tokenizer = AutoTokenizer.from_pretrained(
                self.config.multilingual_encoder_checkpoint, trust_remote_code=True
            )
        return self._m_tokenizer

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def get_vocab(self) -> Dict[int, str]:
        """Return a combined id→token mapping for both vocabularies.

        English tokens are prefixed with ``e_``, multilingual tokens with ``m_``.
        Multilingual token IDs are offset by ``en_vocab_size``.
        """
        en_id2term = {idx: f"e_{term}" for term, idx in self.en_tokenizer.vocab.items()}
        m_id2term = {
            idx + self.config.en_vocab_size: f"m_{term}"
            for term, idx in self.m_tokenizer.vocab.items()
        }
        return en_id2term | m_id2term

    def encode_query(self, texts, **kwargs):
        """Encode query texts into sparse representations. Alias for ``encode_text``."""
        return self.encode_text(texts, **kwargs)

    def encode_document(self, texts, **kwargs):
        """Encode document texts into sparse representations. Alias for ``encode_text``."""
        return self.encode_text(texts, **kwargs)

    def encode_text(
        self,
        texts,
        batch_size: int = 32,
        max_length: int = None,
        return_dict: bool = False,
    ):
        """Encode texts into sparse retrieval representations.

        Args:
            texts: A string or list of strings.
            batch_size: Number of texts per forward pass.
            max_length: Maximum token length (defaults to tokenizer ``model_max_length``).
            return_dict: If ``True``, returns a list of ``{token: weight}`` dicts.

        Returns:
            A 2-D sparse COO tensor of shape ``(N, vocab_size)``, or a list of dicts
            when ``return_dict=True``.
        """
        if isinstance(texts, str):
            texts = [texts]
        if max_length is None:
            max_length = self.m_tokenizer.model_max_length
        device = next(self.parameters()).device
        reps = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.m_tokenizer(
                batch,
                padding=True,
                truncation=True,
                pad_to_multiple_of=PAD_TO_MULTIPLE_OF,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                output = self.encode(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
            if isinstance(output, tuple):
                sparse_rep = self._build_dual_view_sparse(output)
            else:
                sparse_rep = output.to_sparse()
            reps.append(sparse_rep.cpu())
        combined = _sparse_row_cat_2d(reps)
        assert combined.size(0) == len(texts)
        if return_dict:
            return self._sparse_to_dicts(combined)
        return combined

    def _build_dual_view_sparse(self, output):
        """Combine pivot-view (English vocab) and source-view (native tokens) into one sparse tensor.

        The result has shape ``(batch, en_vocab_size + m_vocab_size)``.  English-side
        term weights occupy the first ``en_vocab_size`` columns; native-token weights
        occupy the remaining columns (offset by ``en_vocab_size``).
        """
        pivot_view, (source_ids, source_weights) = output
        batch_size = source_ids.size(0)
        vocab_size = self.config.en_vocab_size + self.config.m_vocab_size

        pivot_rows, pivot_cols = pivot_view.nonzero(as_tuple=True)
        pivot_vals = pivot_view[pivot_rows, pivot_cols].contiguous()

        source_rows = (
            torch.arange(batch_size, device=source_ids.device)
            .repeat_interleave(source_ids.size(1))
        )
        source_cols = source_ids.view(-1)
        source_vals = source_weights.view(-1)
        nonzero_mask = source_vals > 0
        source_rows = source_rows[nonzero_mask]
        source_cols = source_cols[nonzero_mask] + self.config.en_vocab_size
        source_vals = source_vals[nonzero_mask]

        rows = torch.cat([pivot_rows, source_rows])
        cols = torch.cat([pivot_cols, source_cols])
        values = torch.cat([pivot_vals, source_vals])
        indices = torch.stack([rows, cols], dim=0)
        return torch.sparse_coo_tensor(indices, values, (batch_size, vocab_size))

    def _sparse_to_dicts(self, sparse_tensor) -> List[Dict[str, float]]:
        """Convert a sparse ``[N, vocab]`` tensor to a list of ``{token: weight}`` dicts."""
        id2term = self.get_vocab()
        sparse_tensor = sparse_tensor.coalesce()
        indices = sparse_tensor.indices()
        values = sparse_tensor.values()
        rows, cols = indices[0], indices[1]
        result: List[List] = [[] for _ in range(sparse_tensor.size(0))]
        for i in range(len(values)):
            row = rows[i].item()
            col = cols[i].item()
            weight = values[i].item()
            token = id2term.get(col, f"unk_{col}")
            result[row].append((token, weight))
        return [
            dict(sorted(pairs, key=lambda x: x[1], reverse=True))
            for pairs in result
        ]

    # ------------------------------------------------------------------
    # Training 
    # ------------------------------------------------------------------

    def encode(self, input_ids, attention_mask):
        """Core encoding forward pass used during training and inference."""
        hidden_states_m = self.m_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        hidden_states_e = self.activation(self.projector(hidden_states_m))
        logits = self.mlm_head(hidden_states_e) * attention_mask.unsqueeze(-1)
        pivot_view = torch.log1p(torch.relu(logits.max(dim=1).values))
        if not self.config.echo:
            return pivot_view
        token_scores = torch.relu(self.echo(hidden_states_m).squeeze(-1))
        vals = token_scores * attention_mask * self.scale
        return pivot_view, (input_ids, vals)

    def score_pairs(self, queries, passages):
        q_reps = self.encode(queries["input_ids"], queries["attention_mask"])
        p_reps = self.encode(passages["input_ids"], passages["attention_mask"])
        if not self.config.echo:
            return (q_reps * p_reps).sum(dim=1)
        q_pivot, q_source = q_reps
        p_pivot, p_source = p_reps
        score_row = (q_pivot * p_pivot).sum(dim=1)
        matching_mask = (q_source[0].unsqueeze(-1) == p_source[0].unsqueeze(1)).float()
        score_col = ((q_source[1].unsqueeze(-1) * p_source[1].unsqueeze(1)) * matching_mask).sum(-1).sum(-1)
        return score_row + score_col

    def freeze_params(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def compute_alignment_loss(self, reps_a, reps_b):
        mask = (reps_a > 0) | (reps_b > 0)
        return F.mse_loss(reps_a[mask], reps_b[mask])

    def forward(self, en_inputs, me_inputs, m_inputs, **kwargs) -> Dict[str, torch.Tensor]:
        """Alignment training forward pass.

        Encodes the non-English passage (``m_inputs``) and its English counterpart
        (``me_inputs``) through the multilingual encoder, then minimises the distance
        to the frozen English LSR encoder output (``en_inputs``).
        """
        m_h = self.activation(self.projector(
            self.m_model(input_ids=m_inputs["input_ids"], attention_mask=m_inputs["attention_mask"]).last_hidden_state
        ))
        m_reps = (self.mlm_head(m_h) * m_inputs["attention_mask"].unsqueeze(-1)).max(dim=1).values

        me_h = self.activation(self.projector(
            self.m_model(input_ids=me_inputs["input_ids"], attention_mask=me_inputs["attention_mask"]).last_hidden_state
        ))
        me_reps = (self.mlm_head(me_h) * me_inputs["attention_mask"].unsqueeze(-1)).max(dim=1).values

        with torch.no_grad():
            en_logits = self.en_lsr(
                input_ids=en_inputs["input_ids"],
                attention_mask=en_inputs["attention_mask"],
                return_dict=True,
            ).logits * en_inputs["attention_mask"].unsqueeze(-1)
            en_reps = en_logits.max(dim=1).values

        loss = self.compute_alignment_loss(m_reps, en_reps) + self.compute_alignment_loss(me_reps, en_reps)
        student_length = ((m_reps > 0).float().sum(-1).mean() + (me_reps > 0).float().sum(-1).mean()) / 2
        teacher_length = (en_reps > 0).float().sum(-1).mean()
        return {"loss": loss, "student_length": student_length, "teacher_length": teacher_length}

    def save_pretrained(self, save_directory: str, **kwargs):
        kwargs["safe_serialization"] = False
        super().save_pretrained(save_directory, **kwargs)


class ContrastiveMILCOModel(MILCOModel):
    """MILCOModel variant for contrastive and distillation training.
    """

    def __init__(self, config: MILCOConfig):
        super().__init__(config)
        del self.en_lsr

    def score_passages(self, q_reps, p_reps, num_docs):
        """Compute per-query passage scores directly, shape [B_q, num_docs]."""
        if not self.config.echo:
            B_q = q_reps.size(0)
            return (q_reps.unsqueeze(1) * p_reps.view(B_q, num_docs, -1)).sum(-1)
        q_pivot, (q_ids, q_vals) = q_reps
        p_pivot, (p_ids, p_vals) = p_reps
        B_q, S1, S2 = q_pivot.size(0), q_ids.size(1), p_ids.size(1)
        score_pivot = (q_pivot.unsqueeze(1) * p_pivot.view(B_q, num_docs, -1)).sum(-1)
        matching = (q_ids.view(B_q, 1, S1, 1) == p_ids.view(B_q, num_docs, 1, S2)).float()
        score_source = (q_vals.view(B_q, 1, S1, 1) * p_vals.view(B_q, num_docs, 1, S2) * matching).sum(-1).sum(-1)
        return score_pivot + score_source

    def l1(self, reps):
        return torch.abs(reps).sum(dim=1).mean()

    def flops(self, reps):
        return (torch.abs(reps).mean(dim=0) ** 2).sum()

    def sparsity_loss(self, q_reps, p_reps):
        sparse_reg_func = self.l1 if self.config.sparse_reg == "l1" else self.flops
        q = q_reps[0] if self.config.echo else q_reps
        p = p_reps[0] if self.config.echo else p_reps
        return self.config.lambda_q * sparse_reg_func(q) + self.config.lambda_d * sparse_reg_func(p)

    def cal_length(self, q_reps, p_reps):
        q = q_reps[0] if self.config.echo else q_reps
        p = p_reps[0] if self.config.echo else p_reps
        return (q > 0).sum(dim=1).float().mean(), (p > 0).sum(dim=1).float().mean()

    def forward(self, queries, passages, teacher_scores):
        q_reps = self.encode(queries["input_ids"], queries["attention_mask"])
        p_reps = self.encode(passages["input_ids"], passages["attention_mask"])

        q_len = queries["input_ids"].size(0)
        num_docs = passages["input_ids"].size(0) // q_len
        student_scores = self.score_passages(q_reps, p_reps, num_docs)
        with torch.no_grad():
            teacher_scores = torch.as_tensor(teacher_scores, device=queries["input_ids"].device).view(q_len, -1)
        distillation_loss = F.kl_div(
            torch.log_softmax(student_scores, dim=-1),
            torch.softmax(teacher_scores, dim=-1),
            reduction="batchmean",
        )

        l1_loss = self.sparsity_loss(q_reps, p_reps)
        q_length, d_length = self.cal_length(q_reps, p_reps)
        return {
            "loss": distillation_loss + l1_loss,
            "l1_loss": l1_loss.detach(),
            "distillation_loss": distillation_loss.detach(),
            "q_length": q_length,
            "d_length": d_length,
        }

# ---------------------------------------------------------------------------
# Sparse tensor utility
# ---------------------------------------------------------------------------

def _sparse_row_cat_2d(tensors):
    """Concatenate 2-D COO sparse tensors along rows (dim 0)."""
    assert tensors, "empty list"
    if len(tensors) == 1:
        return tensors[0]
    tensors = [t.coalesce() for t in tensors]
    assert all(t.layout == torch.sparse_coo and t.dim() == 2 for t in tensors)
    ncols = tensors[0].size(1)
    assert all(t.size(1) == ncols for t in tensors)
    total_rows = sum(t.size(0) for t in tensors)
    device = tensors[0].device
    ind_parts, val_parts = [], []
    row_offset = 0
    for t in tensors:
        idx = t._indices()
        if row_offset:
            idx = idx.clone()
            idx[0] += row_offset
        ind_parts.append(idx)
        val_parts.append(t._values())
        row_offset += t.size(0)
    indices = torch.cat(ind_parts, dim=1)
    values = torch.cat(val_parts, dim=0)
    return torch.sparse_coo_tensor(indices, values, (total_rows, ncols), device=device).coalesce()

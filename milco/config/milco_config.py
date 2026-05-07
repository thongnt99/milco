from dataclasses import dataclass, field
from typing import Optional, List
from transformers import PretrainedConfig, AutoConfig


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    lsr_encoder_checkpoint: str = field(
        default="naver/splade-v3",
        metadata={"help": "Checkpoint for the LSR encoder"}
    )
    multilingual_encoder_checkpoint: str = field(
        default="BAAI/bge-m3-unsupervised",
        metadata={"help": "Checkpoint for the multilingual encoder"}
    )
    echo: bool = field(
        default=False,
        metadata={"help": "Enable echo head: adds a per-token scoring layer (self.echo + self.scale) that produces a source-view representation alongside the pivot view."}
    )
    prompt: Optional[str] = field(
        default=None,
        metadata={"help": "Optional prompt prepended to inputs (e.g. for instruction-tuned LLMs)."}
    )
    model_type: str = field(
        default="bert",
        metadata={"help": "Model variant identifier (passed as model_variant in MILCOConfig)."}
    )
    lambda_q: float = field(
        default=0.0,
        metadata={"help": "Query sparsity regularization weight."}
    )
    lambda_d: float = field(
        default=0.0,
        metadata={"help": "Document sparsity regularization weight."}
    )
    sparse_reg: str = field(
        default="l1",
        metadata={"help": "Sparsity regularizer: 'l1' or 'flops'."}
    )
    pretrained_alignment_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to a pre-trained alignment checkpoint to warm-start from."}
    )


@dataclass
class DataArguments:
    """Arguments for data loading and preprocessing."""
    train_datasets: List[str] = field(
        default=("mmarco",),
        metadata={"help": "Dataset identifiers to load for training (e.g. mmarco, msmarco, wikipedia)."}
    )
    use_prompt: bool = field(
        default=False,
        metadata={"help": "Prepend task prompt to queries during evaluation."}
    )
    training_type: str = field(
        default="alignment",
        metadata={"help": "Training mode: 'alignment' or 'distillation'."}
    )
    train_group_size: int = field(
        default=4,
        metadata={"help": "Total passages per query (1 positive + negatives) for distillation training."}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum token length for tokenization."}
    )
    query_max_length: int = field(
        default=64,
        metadata={"help": "Maximum query token length."}
    )
    passage_max_length: int = field(
        default=512,
        metadata={"help": "Maximum passage token length."}
    )
    dynamic_length: bool = field(
        default=False,
        metadata={"help": "Use dynamic (batch-level) padding instead of fixed max_length."}
    )
    eval_languages: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Language codes to evaluate on (e.g. ['en', 'fr']). None skips evaluation."}
    )
    eval_split: str = field(
        default="dev",
        metadata={"help": "Dataset split to evaluate on: 'train', 'dev', or 'test'."}
    )
    eval_top_k_candidates: int = field(
        default=1000,
        metadata={"help": "Maximum candidates to re-rank per query during evaluation."}
    )
    neg_sampling_strategy: str = field(
        default="random",
        metadata={"help": "Negative sampling strategy for distillation: 'random' or 'stratified'."}
    )
    false_neg_threshold: float = field(
        default=0.0,
        metadata={"help": "Promote negatives with teacher score above this threshold to positives. 0.0 disables."}
    )

class MILCOConfig(PretrainedConfig):
    """Configuration for the MILCO (Multilingual Sparse Retrieval) model.

    Sub-model vocab and hidden sizes are populated on first init and cached in
    config.json so they survive ``save_pretrained`` / ``from_pretrained`` round-trips.
    """

    model_type = "milco"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lsr_encoder_checkpoint = kwargs.get("lsr_encoder_checkpoint", "bert-base-uncased")
        self.multilingual_encoder_checkpoint = kwargs.get("multilingual_encoder_checkpoint", "bert-base-uncased")
        self.model_variant = kwargs.get("model_variant", "bert")

        # Sub-model metadata — populated on first init and cached in config.json.
        self.en_vocab_size = kwargs.get("en_vocab_size", None)
        self.en_hidden_size = kwargs.get("en_hidden_size", None)
        self.m_vocab_size = kwargs.get("m_vocab_size", None)
        self.m_hidden_size = kwargs.get("m_hidden_size", None)


        if self.en_vocab_size is None:
            en_cfg = AutoConfig.from_pretrained(self.lsr_encoder_checkpoint, trust_remote_code=True)
            self.en_vocab_size = en_cfg.vocab_size
        if self.m_vocab_size is None:
            m_cfg = AutoConfig.from_pretrained(self.multilingual_encoder_checkpoint, trust_remote_code=True)
            self.m_vocab_size = m_cfg.vocab_size

        # Training hyperparameters
        self.echo = kwargs.get("echo", False)
        self.lambda_q = kwargs.get("lambda_q", 0.0)
        self.lambda_d = kwargs.get("lambda_d", 0.0)
        self.sparse_reg = kwargs.get("sparse_reg", "l1")


def create_config_from_args(model_args: ModelArguments) -> MILCOConfig:
    """Create an MILCOConfig from parsed ModelArguments."""
    return MILCOConfig(
        lsr_encoder_checkpoint=model_args.lsr_encoder_checkpoint,
        multilingual_encoder_checkpoint=model_args.multilingual_encoder_checkpoint,
        echo=model_args.echo,
        lambda_q=model_args.lambda_q,
        lambda_d=model_args.lambda_d,
        sparse_reg=model_args.sparse_reg,
        model_variant=model_args.model_type,
    )

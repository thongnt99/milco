"""
MiLCO: Multilingual Connector for Sparse Retrieval

A framework for training multilingual sparse retrieval models via alignment
and knowledge distillation.
"""

__version__ = "0.1.0"

from .config import MILCOConfig, ModelArguments, DataArguments, create_config_from_args
from .data import DataProcessor, prepare_alignment_datasets, prepare_distillation_datasets
from .trainer import MilcoTrainer, train_from_args, test_from_args
from .utils import master_print

__all__ = [
    "MILCOConfig",
    "ModelArguments",
    "DataArguments",
    "create_config_from_args",
    "DataProcessor",
    "prepare_alignment_datasets",
    "prepare_distillation_datasets",
    "MilcoTrainer",
    "train_from_args",
    "test_from_args",
    "master_print",
]

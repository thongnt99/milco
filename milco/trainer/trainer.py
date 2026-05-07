import os
import sys
import getpass
import json
from contextlib import nullcontext
from typing import Optional, Dict, List
import random
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import psutil
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    AutoTokenizer,
)

from ..config import ModelArguments, DataArguments, MILCOConfig, create_config_from_args
from ..model.milco import MILCOModel, ContrastiveMILCOModel
from ..data.processing import DataProcessor
from ..data import (
    prepare_alignment_datasets,
    prepare_distillation_datasets,
)
from ..data.evaluation_data import MultiLanguageEvaluationDataset
from ..data.miracl_hard_negatives import MIRACLHardNegativesDataset
from ..data.collator import PaddingCollator, ContrastiveCollator
from ..evaluator import FullCorpusEvaluator
from ..logging import WandbPredictionProgressCallback
from ..utils import master_print


EVAL_DATASET_NAME = "miracl_hard_negatives"
EVAL_LANGUAGES = ["ar", "bn", "de", "en", "es", "fa", "fi", "fr", "hi", "id", "ja", "ko", "ru", "sw", "te", "th", "zh", "yo"]
EVAL_SPLIT = "dev"


def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"Current memory usage: {mem_mb:.2f} MB")


class MilcoTrainer(Trainer):
    """Custom HuggingFace Trainer for MiLCO models with integrated evaluation."""

    def __init__(self, model_args: ModelArguments, data_args: DataArguments, **kwargs):
        self.model_args = model_args
        self.data_args = data_args
        self.customed_log = defaultdict(lambda: 0.0)

        os.environ.setdefault("WANDB_PROJECT", "milco")
        os.environ.setdefault("WANDB_ENTITY", "omai-research")

        if "args" in kwargs:
            if not kwargs["args"].run_name or kwargs["args"].run_name == kwargs["args"].output_dir:
                cmd = " ".join(sys.argv)
                username = getpass.getuser()
                kwargs["args"].run_name = f"{username}-{cmd}"
            kwargs["args"].remove_unused_columns = False

        self.config = create_config_from_args(model_args)
        self.data_processor = DataProcessor(self.config, data_args)

        if self.data_args.training_type == "alignment":
            model = MILCOModel(self.config)
            data_collator = PaddingCollator(self.data_args, self.data_processor)
            train_dataset = prepare_alignment_datasets(self.data_args.train_datasets)
        elif self.data_args.training_type == "distillation":
            model = ContrastiveMILCOModel(self.config)
            data_collator = ContrastiveCollator(self.data_args, self.data_processor)
            train_dataset = prepare_distillation_datasets(
                self.data_args,
                self.data_args.train_datasets,
            )
        else:
            raise ValueError(
                f"Unknown training_type '{self.data_args.training_type}'. "
                "Expected 'alignment' or 'distillation'."
            )

        print(f"Number of training samples: {len(train_dataset)}")

        if model_args.pretrained_alignment_checkpoint:
            print(f"Loading aligned checkpoint from: {model_args.pretrained_alignment_checkpoint}")
            checkpoint = MILCOModel.from_pretrained(model_args.pretrained_alignment_checkpoint)
            result = model.load_state_dict(checkpoint.state_dict(), strict=False)
            print(f"Alignment load: {len(result.missing_keys)} missing, {len(result.unexpected_keys)} unexpected")
            if result.missing_keys:
                print(f"  missing[:10]: {result.missing_keys[:10]}")
            if result.unexpected_keys:
                print(f"  unexpected[:10]: {result.unexpected_keys[:10]}")
            del checkpoint

        eval_dataset = None
        print("Evaluation setup:", self.data_args.eval_languages)
        if self.data_args.eval_languages:
            eval_dataset = {
                EVAL_DATASET_NAME: self._create_multi_language_eval_dataset(self.data_args.eval_languages)
            }

        print_memory_usage()

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            **kwargs,
        )

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        if self.control.should_log:
            log = {}
            steps_since_last_log = max(1, self.state.global_step - self._globalstep_last_logged)
            for metric in self.customed_log:
                log[metric] = round(
                    self._nested_gather(self.customed_log[metric]).mean().item()
                    / steps_since_last_log
                    / self.args.gradient_accumulation_steps,
                    4,
                )
            self.log(log)
            self.customed_log.clear()
            self.control.should_log = True
        super()._maybe_log_save_evaluate(*args, **kwargs)

    def compute_loss(self, *args, return_outputs=False, **kwargs):
        if hasattr(self.model, "set_training_step"):
            self.model.set_training_step(self.state.global_step)
        loss, output = super().compute_loss(*args, return_outputs=True, **kwargs)
        for log_metric in output:
            if log_metric != "loss":
                self.customed_log[log_metric] += output[log_metric]
        return (loss, output) if return_outputs else loss

    def _create_multi_language_eval_dataset(
        self, languages=None
    ) -> MultiLanguageEvaluationDataset:
        if languages is None:
            languages = EVAL_LANGUAGES
        language_configs = {
            lang: {
                "split": EVAL_SPLIT,
                "top_k_candidates": self.data_args.eval_top_k_candidates,
            }
            for lang in languages
        }
        return MultiLanguageEvaluationDataset(
            language_configs=language_configs,
            dataset_class=MIRACLHardNegativesDataset,
            use_prompt=self.data_args.use_prompt,
        )

    def test(self):
        test_dataset = {EVAL_DATASET_NAME: self._create_multi_language_eval_dataset()}
        test_language = {EVAL_DATASET_NAME: EVAL_LANGUAGES}
        test_metrics = self.evaluate(
            eval_dataset=test_dataset,
            eval_languages=test_language,
            metric_key_prefix="test",
        )
        for name in test_metrics:
            result_path = os.path.join(self.args.output_dir, f"test_results_{name}.json")
            master_print(f"Writing test result to: {result_path}")
            json.dump(test_metrics[name], open(result_path, "w"))
        return test_metrics

    def evaluate(
        self,
        eval_dataset=None,
        eval_languages=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
        **kwargs,
    ) -> Dict[str, float]:
        self.model.eval()
        all_metrics = {}
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        if isinstance(eval_dataset, dict):
            for dataset_name in eval_dataset:
                dataset_metrics = self._evaluate_miracl_lazy(dataset_name, eval_dataset[dataset_name])
                all_metrics[dataset_name] = dataset_metrics
                self.log_metrics(f"{metric_key_prefix}_{dataset_name}", dataset_metrics)
                self.log({f"{metric_key_prefix}_{m}": v for m, v in dataset_metrics.items()})
        self.model.train()
        return all_metrics

    def _evaluate_miracl_lazy(
        self,
        dataset_name: str,
        eval_dataset: MultiLanguageEvaluationDataset,
        eval_languages: List[str] = None,
    ) -> Dict[str, float]:
        master_print(f"\n=== Starting evaluation on {dataset_name} ===")
        evaluator = FullCorpusEvaluator(
            model=self.model,
            batch_size=self.args.per_device_eval_batch_size,
            query_max_length=self.data_args.query_max_length,
            passage_max_length=self.data_args.passage_max_length,
            top_k=self.data_args.eval_top_k_candidates,
        )
        all_metrics = evaluator.evaluate_multi_language(
            multi_lang_dataset=eval_dataset,
            languages=eval_languages,
            unload_after_each=True,
        )
        flattened_metrics = {}
        for language, metrics in all_metrics.items():
            if language == "average":
                for metric_name, value in metrics.items():
                    flattened_metrics[f"{dataset_name}_avg_{metric_name}"] = value
            else:
                for metric_name, value in metrics.items():
                    flattened_metrics[f"{dataset_name}_{language}_{metric_name}"] = value
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, flattened_metrics
        )
        return flattened_metrics

    def predict(self, samples):
        if not dist.is_initialized() or dist.get_rank() == 0:
            e_tokenizer = AutoTokenizer.from_pretrained(
                self.model.config.lsr_encoder_checkpoint, trust_remote_code=True
            )
            results = []
            for sample in samples:
                device = next(self.model.parameters()).device
                text = sample["input"]
                inputs = self.data_processor.preprocess_multilingual([text])
                inputs = {k: v.to(device) for k, v in inputs.items()}
                amp_dtype = (
                    torch.bfloat16 if self.args.bf16 else (torch.float16 if self.args.fp16 else None)
                )
                ctx = (
                    torch.autocast(device_type=device.type, dtype=amp_dtype)
                    if amp_dtype
                    else nullcontext()
                )
                with torch.no_grad(), ctx:
                    reps = self.model.encode(**inputs)
                    if getattr(self.model.config, "echo", False):
                        reps = reps[0]
                reps = (reps * 100).int().view(-1)
                top_indices = reps.argsort(descending=True)[: min(1000, (reps > 0).sum())]
                top_weights = reps[top_indices].tolist()
                keys = e_tokenizer.convert_ids_to_tokens(top_indices)
                sample["output"] = json.dumps(dict(zip(keys, top_weights)), ensure_ascii=False)
                results.append(sample)
            return results
        return []


def train_from_args(args=None) -> MilcoTrainer:
    """Parse command-line arguments and run training."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print(os.environ["HF_HOME"])

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

    trainer = MilcoTrainer(model_args=model_args, data_args=data_args, args=training_args)

    progress_callback = WandbPredictionProgressCallback(trainer=trainer)
    trainer.add_callback(progress_callback)

    print("Training started ...........")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model()
    trainer.test()
    return trainer


def test_from_args(args=None) -> MilcoTrainer:
    """Parse command-line arguments and run evaluation only."""
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)

    trainer = MilcoTrainer(model_args=model_args, data_args=data_args, args=training_args)

    print("Testing started ...........")
    trainer.test()
    return trainer

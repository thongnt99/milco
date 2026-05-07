import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import Dict, Any, Callable
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from ..config import MILCOConfig, DataArguments
from ..utils import master_print
import torch 


class DataProcessor:
    """Unified data processor."""

    def __init__(self, config: MILCOConfig, data_args: DataArguments):
        self.config = config
        self.data_args = data_args
        self.return_offsets_mapping = False
        # Load tokenizers
        self.m_tokenizer = AutoTokenizer.from_pretrained(
            config.multilingual_encoder_checkpoint, trust_remote_code=True
        )
        if self.config.model_variant == "llama":
            self.m_tokenizer.pad_token = self.m_tokenizer.eos_token
            self.m_tokenizer.padding_side = "right"
        self.en_tokenizer = AutoTokenizer.from_pretrained(
            config.lsr_encoder_checkpoint, trust_remote_code=True
        )
        self._compute_and_update_prompt_length()
        
    def _compute_and_update_prompt_length(self):
        """Compute prompt length dynamically and update the config."""
        if not hasattr(self.config, "prompt") or self.config.prompt is None or not self.config.prompt:
            self.config.prompt_length = 0
            return
        try:
            if self.config.model_variant == "qwen":
                # Apply chat template to a sample text to see the actual prompt length
                sample_template = self.m_tokenizer.apply_chat_template(
                    [{"role": "user", "content": self.config.prompt + "sample text"}],
                    tokenize=False,
                    add_generation_prompt=True, 
                    enable_thinking=False, 
                )
                sample_without_prompt = self.m_tokenizer.apply_chat_template(
                    [{"role": "user", "content": "sample text"}],
                    tokenize=False,
                    add_generation_prompt=True, 
                    enable_thinking=False, 
                )
                
                tokens_with_prompt = self.m_tokenizer.tokenize(sample_template)
                tokens_without_prompt = self.m_tokenizer.tokenize(sample_without_prompt)
                computed_length = len(tokens_with_prompt) - len(tokens_without_prompt) + 3
            else:
                # For BERT and other models, just tokenize the prompt directly
                tokenized_prompt = self.m_tokenizer.tokenize(self.config.prompt)
                computed_length = len(tokenized_prompt)
            
            self.config.prompt_length = computed_length 
            master_print(f"Dynamically computed prompt length: {computed_length} (model: {self.config.model_variant}, prompt: '{self.config.prompt}')")
            
        except Exception as e:
            master_print(f"Warning: Could not compute prompt length dynamically: {e}")
            self.config.prompt_length = 0
    
    def preprocess_english(self, texts, max_length=512):
        e = self.en_tokenizer(
            texts, 
            padding= True if self.data_args.dynamic_length else "max_length",   
            truncation = True, 
            max_length = max_length,
            return_tensors="pt"
        )
        return e 

    def preprocess_multilingual(self, texts, max_length=512):
        preprocessors = {
            "qwen": self._preprocess_qwen,
            "gemma": self._preprocess_gemma,
            "llama": self._preprocess_llama,
            "bert": self._preprocess_bert,
        }
        if self.config.model_variant not in preprocessors:
            raise ValueError(
                f"Unknown model_variant '{self.config.model_variant}'. "
                f"Expected one of: {sorted(preprocessors)}"
            )
        return preprocessors[self.config.model_variant](texts, max_length)

    def _preprocess_bert(self, texts, max_length=512):
        """Preprocessing function for BERT-based models."""
        # Apply prompt if provided
        if hasattr(self.config, "prompt") and self.config.prompt:
            texts = [self.config.prompt + text for text in texts]

        m = self.m_tokenizer(
            texts, 
            padding=True if self.data_args.dynamic_length else "max_length", 
            # pad_to_multiple_of=16,
            truncation=True, 
            max_length=max_length,
            return_tensors="pt", 
            # return_offsets_mapping=self.return_offsets_mapping,
        )
        return m

    def _preprocess_llama(self, texts, max_length=512):
        """Preprocessing function for BERT-based models."""
        # Apply prompt if provided
        if hasattr(self.config, "prompt") and self.config.prompt:
            texts = [self.config.prompt + text for text in texts]

        m = self.m_tokenizer(
            texts, 
            padding="longest",    #True if self.data_args.dynamic_length else "max_length", 
            pad_to_multiple_of=16,
            truncation=True, 
            max_length=max_length,
            return_tensors="pt", 
            return_offsets_mapping=self.return_offsets_mapping,
        )
        return m
    

    def _preprocess_gemma(self, texts, max_length=512):
        """Preprocessing function for Gemma-based models."""
        # Apply prompt if provided
        if self.config.prompt:
            texts = [f"{self.config.prompt} {text}\n English: " for text in texts]

        m = self.m_tokenizer(
            texts, 
            padding=True if self.data_args.dynamic_length else "max_length", 
            truncation=True, 
            max_length = max_length,
            return_tensors="pt"
        )
        return m 
    
    def _preprocess_qwen(self, texts, max_length=512):
        """Preprocessing function for Qwen-based models."""
        if self.config.prompt:
            texts = [self.config.prompt + text for text in texts]
        
        texts =[
            self.m_tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True, 
                enable_thinking=False, 
            ) for text in texts
        ]
        
        m = self.m_tokenizer(
            texts, 
            padding=True if self.data_args.dynamic_length else "max_length", 
            truncation=True, 
            max_length=max_length,
            return_tensors="pt"
            )
            
        return m
    
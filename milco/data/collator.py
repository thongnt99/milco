import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any

class PaddingCollator:
    """Data collator that pads and batches pre-tokenized inputs."""
    def __init__(self, data_args, processor):
        self.data_args = data_args
        self.processor = processor
        self.max_length = data_args.max_length

    def __call__(self, examples):
        english_texts = [example["english"] for example in examples]
        non_english_texts = [example["non_english"] for example in examples]
        ee_inputs = self.processor.preprocess_english(english_texts, max_length=self.max_length)
        me_inputs = self.processor.preprocess_multilingual(english_texts, max_length=self.max_length)
        mm_inputs = self.processor.preprocess_multilingual(non_english_texts, max_length=self.max_length)

        return {
            "en_inputs": ee_inputs, 
            "me_inputs": me_inputs,  
            "m_inputs": mm_inputs
        }

class ContrastiveCollator:
    def __init__(self, data_args, processor):
        self.data_args = data_args
        self.processor = processor 

    def __call__(self, examples):
        if isinstance(examples[0], list):
            examples = examples[0]
        queries = [example['queries'] for example in examples]
        passages = []
        for example in examples:
            passages.extend(example['passages'])
        query_inputs = self.processor.preprocess_multilingual(queries, max_length=self.data_args.query_max_length)
        passage_inputs = self.processor.preprocess_multilingual(passages, max_length=self.data_args.passage_max_length)
        
        batch = {
            "queries": query_inputs,
            "passages":  passage_inputs
        }
        if "messages" in examples[0]:
            batch['messages'] =[exp["messages"] for exp in examples]
        if "scores" in examples[0]:
            batch["teacher_scores"] = [exp["scores"] for exp in examples]
        return batch 
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk
import random
import torch.distributed as dist


dataset_configs = {
    "bge-distillation": {"dataset_name": "omai-research/bge-multilingual-distillation-dataset"},
    "qwen3-4b-scores": {"dataset_name": "omai-research/bge-retrieval-distillation-qwen3-4b"}
}

class DistillationDataset(Dataset):
    def __init__(self,
        data_args,
        dataset_names,
    ):
        self.data_args = data_args
        unknown = sorted(set(dataset_names) - set(dataset_configs))
        if unknown:
            raise ValueError(
                f"Unknown distillation dataset(s): {unknown}. "
                f"Available: {sorted(dataset_configs)}"
            )
        if len(dataset_names) != 1:
            raise ValueError(
                f"DistillationDataset currently supports exactly one dataset, "
                f"got {len(dataset_names)}: {dataset_names}"
            )
        dataset_name = dataset_names[0]
        train_dataset_name = dataset_configs[dataset_name]["dataset_name"]
        self.dataset = load_dataset(train_dataset_name, split="train")

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        query = sample["query"]
        train_group_size = self.data_args.train_group_size
        n_neg = train_group_size - 1
        pos_texts = list(sample['pos'])
        pos_scores = list(sample['pos_scores'])
        neg_texts = list(sample['neg'])
        neg_scores = list(sample['neg_scores'])
        pos_idx = random.choice(range(len(pos_texts)))
        pos = pos_texts[pos_idx]
        pos_score = pos_scores[pos_idx]
        all_neg_indices = list(range(len(neg_texts)))
        if len(neg_texts) < n_neg:
            neg_indices = random.choices(all_neg_indices, k=n_neg)
        else:
            neg_indices = random.sample(all_neg_indices, k=n_neg)
        negs = [neg_texts[i] for i in neg_indices]
        neg_scores_sampled = [neg_scores[i] for i in neg_indices]

        return {
            "queries": query,
            "passages": [pos] + negs,
            "scores": [pos_score] + neg_scores_sampled
        }

    def __len__(self):
        return len(self.dataset)

def prepare_distillation_datasets(data_args, dataset_names):
    return DistillationDataset(data_args, dataset_names)
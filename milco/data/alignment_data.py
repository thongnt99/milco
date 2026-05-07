import os
import json
from typing import List, Any
import torch
from datasets import load_dataset, concatenate_datasets

def prepare_alignment_datasets(dataset_names):
    unknown = sorted(set(dataset_names) - set(dataset_configs))
    if unknown:
        raise ValueError(
            f"Unknown alignment dataset(s): {unknown}. "
            f"Available: {sorted(dataset_configs)}"
        )
    raw_datasets = []
    for dataset_name in dataset_names:
        data_config = dataset_configs[dataset_name]
        if isinstance(data_config['subset'], list):
            for i, subset_name in enumerate(data_config['subset']):
                print(f"  Loading {dataset_name}/{subset_name} ({i+1}/{len(data_config['subset'])})")
                raw_datasets.append(load_dataset(data_config["dataset_name"], subset_name, split=data_config["split"]))
        else:
            print(f"Loading {dataset_name}/{data_config['subset']}")
            raw_datasets.append(load_dataset(data_config["dataset_name"], data_config["subset"], split=data_config["split"]))
    print(f"Loaded {len(raw_datasets)} subsets, concatenating...")
    # Drop 'id' column if present (inconsistent types across subsets)
    raw_datasets = [ds.remove_columns("id") if "id" in ds.column_names else ds for ds in raw_datasets]
    print(f"Concatenating {len(raw_datasets)} subsets...")
    all_dataset = concatenate_datasets(raw_datasets)
    return all_dataset

dataset_configs = {
    "finetranslations-edu": {"dataset_name": "omai-research/finetranslation-edu", "subset": "default", "split": "train"},
    "mmarco_passage": {"dataset_name": "omai-research/parallel_mmarco_passage", "subset": [
        "en_ar", "en_zh", "en_nl", "en_fr", "en_de", "en_hi", "en_id",
        "en_it", "en_ja", "en_pt", "en_ru", "en_es", "en_vi"
    ], "split": "train"},
    "mmarco_query": {"dataset_name": "omai-research/parallel_mmarco_query", "subset": [
        "ar", "zh", "nl", "fr", "de", "hi", "id",
        "it", "ja", "pt", "ru", "es", "vi"
    ], "split": "train"},
    "wikititles": {"dataset_name": "sentence-transformers/parallel-sentences-wikititles", "subset": "default", "split": "train"},
    "wikimatrix": {"dataset_name": "sentence-transformers/parallel-sentences-wikimatrix", "subset": "all", "split": "train"},
    "europarl": {"dataset_name": "sentence-transformers/parallel-sentences-europarl", "subset": "all", "split": "train"},
    "ccmatrix": {"dataset_name": "sentence-transformers/parallel-sentences-ccmatrix", "subset": [
        "en-af", "en-ar", "en-ast", "en-az", "en-be", "en-bg", "en-bn", "en-br",
        "en-ca", "en-ceb", "en-cs", "en-da", "en-de", "en-el", "en-eo", "en-es",
        "en-et", "en-eu", "en-fa", "en-fi", "en-fr", "en-fy", "en-ga", "en-gd",
        "en-gl", "en-ha", "en-he", "en-hi", "en-hr", "en-hu", "en-id", "en-ig",
        "en-ilo", "en-is", "en-it", "en-ja", "en-jv", "en-ko", "en-la", "en-lb",
        "en-lt", "en-lv", "en-mg", "en-mk", "en-ml", "en-mr", "en-ms", "en-ne",
        "en-nl", "en-no", "en-oc", "en-or", "en-pl", "en-pt", "en-ro", "en-ru",
        "en-sd", "en-si", "en-sk", "en-sl", "en-so", "en-sq", "en-sr", "en-su",
        "en-sv", "en-sw", "en-ta", "en-tl", "en-tr", "en-uk", "en-ur", "en-vi",
        "en-xh", "en-yi", "en-zh"
    ], "split": "train"},
    "opensubtitles": {"dataset_name": "sentence-transformers/parallel-sentences-opensubtitles", "subset": "all", "split": "train"},
    "talks": {"dataset_name": "sentence-transformers/parallel-sentences-talks", "subset": "all", "split": "train"},
    "tatoeba": {"dataset_name": "sentence-transformers/parallel-sentences-tatoeba", "subset": "all", "split": "train"},
    "jw300": {"dataset_name": "sentence-transformers/parallel-sentences-jw300", "subset": "all", "split": "train"},
    "news-commentary": {"dataset_name": "sentence-transformers/parallel-sentences-news-commentary", "subset": "all", "split": "train"},
}
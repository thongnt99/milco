from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

from ..utils import master_print


class BaseEvaluationDataset(ABC):
    """
    Abstract base class for retrieval evaluation datasets.

    Provides common functionality for evaluation datasets and enforces the required interface.
    All evaluation datasets should inherit from this class.
    """

    @abstractmethod
    def get_candidate_ids(self, query_id: str) -> List[Tuple[str, Optional[int], Optional[float]]]:
        """
        Get candidate document ids for a query.

        Args:
            query_id: Query identifier

        Returns:
            List of ``(doc_id, rank, score)`` tuples for candidates. ``rank`` and
            ``score`` may be ``None`` for datasets without a prior retrieval stage
            (e.g. MLDR, where the full corpus is the candidate set).
        """
        pass

    def __len__(self) -> int:
        """Return number of queries in the dataset."""
        return len(self.queries)

    def __getitem__(self, query_id: str) -> Dict[str, Any]:
        """Get evaluation data for a specific query."""
        if query_id not in self.queries:
            raise KeyError(f"Query ID {query_id} not found")

        return {
            'query_id': query_id,
            'query': self.queries[query_id],
            'candidate_ids': self.get_candidate_ids(query_id),
            'candidate_documents': self.get_candidate_documents(query_id),
            'qrels': self.qrels.get(query_id, {})
        }


class MultiLanguageEvaluationDataset:
    """
    Container for multiple language evaluation datasets with lazy loading.

    This allows you to define evaluation for multiple languages without
    loading all the data upfront.
    """

    def __init__(self, language_configs: Dict[str, Dict[str, Any]], dataset_class: BaseEvaluationDataset, use_prompt: bool = False):
        """
        Initialize multi-language evaluation dataset.

        Args:
            language_configs: Dict mapping language code to config dict.
                Each config should contain parameters for the dataset class.

        Example:
            language_configs = {
                "ja": {"split": "dev", "top_k_candidates": 100},
                "ko": {"split": "dev", "top_k_candidates": 100},
                "zh": {"split": "dev", "top_k_candidates": 50}
            }
        """
        self.language_configs = language_configs
        self.dataset_class = dataset_class
        self.use_prompt = use_prompt

    def get_languages(self):
        """Get list of available languages in the configuration."""
        return list(self.language_configs.keys())

    def get_dataset(self, language_code):
        config = self.language_configs[language_code]
        dataset = self.dataset_class(language_code, **config, use_prompt=self.use_prompt)
        master_print(f"Loading evaluation dataset for {language_code}")
        print(dataset.get_statistics())
        return dataset

# MILCO: Learned Sparse Retrieval Across Languages via a Multilingual Connector

[![Paper](https://img.shields.io/badge/ICLR%202026-Paper-blue)](https://openreview.net/forum?id=Z6dVYEqurT)
[![arXiv](https://img.shields.io/badge/arXiv-2510.00671-b31b1b)](https://arxiv.org/abs/2510.00671)
[![Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/collections/omai-research/milco-multilingual-learned-sparse-retrieval)

**Authors**: Thong Nguyen, Yibin Lei, Jia-Huei Ju, Eugene Yang, Andrew Yates

## Overview

Learned Sparse Retrieval (LSR) combines the efficiency of bi-encoders with the transparency of lexical matching, but existing approaches struggle to scale beyond English. **MILCO** addresses this by mapping queries and documents from different languages into a shared English lexical space via a multilingual connector. MILCO supports both **monolingual** and **cross-lingual** retrieval and has been evaluated on **39+ languages**, with the potential to generalize to additional languages supported by the underlying multilingual encoder.

MILCO is trained with a two-stage regime that combines **Sparse Alignment Pretraining** with **contrastive learning**, designed to provide representation transparency and effectiveness while mitigating semantic collapse.

We also introduce the **LexEcho head**, which addresses the observation that uncommon entities/rare terms (e.g., proper nouns, code-switched terms) are often lost when projected into English. The LexEcho head augments the English lexical representation with a source-language view, preserving the original multilingual token weights alongside the projected English terms.

## Results

MILCO achieves state-of-the-art multilingual and cross-lingual LSR performance, outperforming leading dense, sparse, and multi-vector baselines including **BGE-M3** and **Qwen3-Embed** on standard multilingual benchmarks.

With mass-based pruning to reduce document representations to only 30 active dimensions on average, MILCO 560M outperforms the similarly-sized Qwen3-Embed 0.6B (1024 dimensions) while achieving **3x lower retrieval latency** and **10x smaller index size**.

## Models

Pretrained models are available on the [MILCO HuggingFace Collection](https://huggingface.co/collections/omai-research/milco-multilingual-learned-sparse-retrieval).

## Quick Start

### Installation

```bash
pip install transformers torch
```

### Loading a model

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "omai-research/milco",  # replace with a specific model from the collection
    trust_remote_code=True,
)
```

### Encoding text

```python
# Sparse tensor output
sparse_reps = model.encode_text([
    "Baltimore: The Greatest City in America",
    "巴尔的摩：美国最伟大的城市",
    "Baltimore : La plus grande ville d'Amérique",
])

# Token-weight dictionary output (sorted by weight descending)
results = model.encode_text(
    ["Baltimore : La plus grande ville d'Amérique", "巴尔的摩：美国最伟大的城市"],
    return_dict=True,
)
print(results[0])
# {'e_baltimore': 1.802, 'e_maryland': 1.253, 'e_city': 1.202, 'e_largest': 0.944, ...}
print(results[1])
# {'e_baltimore': 1.652, 'e_city': 1.322, 'e_greatest': 1.119, 'e_usa': 0.929, ...}
```

### LexEcho: Dual-view encoding

When `source_view=True`, the LexEcho head augments the pivot representation with source-language token weights, preserving entities and terms that may not have English equivalents:

```python
results = model.encode_text(["巴尔的摩：美国最伟大的城市"], return_dict=True, source_view=True)
print(results[0])
# {'m_</s>': 2.094, 'e_baltimore': 1.652, 'e_city': 1.322, ..., 'm_伟大的': 0.777, 'm_美国': 0.684, ...}
# "e_" prefix = English (pivot) vocabulary
# "m_" prefix = multilingual (source) vocabulary
```

### Retrieval scoring

```python
import torch

q_reps = model.encode_query(queries)
d_reps = model.encode_document(documents)
scores = torch.sparse.mm(q_reps, d_reps.t())
```

## Architecture

MILCO consists of:

- **Multilingual encoder** (`m_model`): A pretrained multilingual transformer that produces contextual embeddings for input text in any language.
- **Projector**: A linear layer that maps multilingual hidden states into the English encoder's hidden dimension.
- **LexEcho head**:
  - **LM head**: Borrowed from a pretrained English masked language model, produces logits over the English vocabulary. Used to generate the pivot (English) sparse representation.
  - **Echo token**: A special [ECHO] token added to the MLM head to score each input token, producing the source-language view that preserves uncommon entities that may be lost during projection into the English vocabulary.
  - 
The pivot view is computed via max-pooling over token-level LM logits, followed by `log1p(relu(...))` activation. When LexEcho is enabled, both pivot and source views are concatenated into a single sparse vector over the combined English + multilingual vocabulary.

## Citation

```bibtex
@inproceedings{nguyen2026milco,
  title={MILCO: Learned Sparse Retrieval Across Languages via a Multilingual Connector},
  author={Nguyen, Thong and Lei, Yibin and Ju, Jia-Huei and Yang, Eugene and Yates, Andrew},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

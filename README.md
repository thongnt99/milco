# MILCO: Learned Sparse Retrieval Across Languages via a Multilingual Connector

[![Paper](https://img.shields.io/badge/ICLR%202026-Paper-blue)](https://openreview.net/forum?id=Z6dVYEqurT)
[![arXiv](https://img.shields.io/badge/arXiv-2510.00671-b31b1b)](https://arxiv.org/abs/2510.00671)
[![Models](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/collections/omai-research/milco-multilingual-learned-sparse-retrieval)

**Authors**: Thong Nguyen, Yibin Lei, Jia-Huei Ju, Eugene Yang, Andrew Yates

Pretrained models are available on the [MILCO HuggingFace Collection](https://huggingface.co/collections/omai-research/milco-multilingual-learned-sparse-retrieval).

## Training

### Installation

We use [`uv`](https://docs.astral.sh/uv/) to manage the environment and dependencies.

```bash
# Create a virtual environment
uv venv milcoenv
source milcoenv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

To add a new package later:

```bash
uv pip install <package-name>
```

### Launching on SLURM

Both training scripts are designed to run under SLURM and read the following
environment variables set automatically by `srun`:

| Variable               | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| `SLURM_NNODES`         | Number of nodes (`--nnodes`)             |
| `SLURM_GPUS_PER_NODE`  | GPUs per node (`--nproc_per_node`)       |
| `SLURM_PROCID`         | Node rank (`--node_rank`)                |
| `SLURM_JOBID`          | Rendezvous ID (`--rdzv-id`)              |

A typical submission script wraps the training command in `srun`:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

srun bash scripts/alignment.sh    # or scripts/distillation.sh
```

Adjust `--nodes`, `--gpus-per-node`, `--time`, and your account/partition flags
to match your cluster. Set `MASTER_PORT` to a free port if the default (25900)
is in use.

### Stage 1: Sparse Alignment Pretraining

Aligns the multilingual encoder to the English LSR space using parallel and
multilingual corpora.

```bash
srun bash scripts/alignment.sh
```

Key arguments in [scripts/alignment.sh](scripts/alignment.sh):

- `--multilingual_encoder_checkpoint` — multilingual backbone (e.g. `BAAI/bge-m3-unsupervised`).
- `--lsr_encoder_checkpoint` — English LSR teacher providing the target lexical space (e.g. `naver/splade-v3`).
- `--train_datasets` — alignment corpora (mMARCO, WikiMatrix, Europarl, OpenSubtitles, Talks, Tatoeba, JW300, news-commentary).
- `--output_dir` — checkpoint location, consumed by Stage 2.

### Stage 2: Contrastive Distillation

Trains the aligned model with hard negatives and teacher-score distillation.

```bash
srun bash scripts/distillation.sh
```

Key arguments in [scripts/distillation.sh](scripts/distillation.sh):

- `--pretrained_alignment_checkpoint` — checkpoint produced by Stage 1.
- `--echo` — enables the LexEcho head (source-language view).
- `--train_datasets bge-distillation` — distillation data with teacher scores.
- `--train_group_size` — number of passages per query (1 positive + N−1 negatives).
- `--lambda_q` / `--lambda_d` — FLOPS regularization weights for queries and documents.

## Citation

```bibtex
@inproceedings{nguyen2026milco,
  title={MILCO: Learned Sparse Retrieval Across Languages via a Multilingual Connector},
  author={Nguyen, Thong and Lei, Yibin and Ju, Jia-Huei and Yang, Eugene and Yates, Andrew},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```

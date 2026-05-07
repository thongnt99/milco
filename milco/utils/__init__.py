"""
Utility functions for the multilingual connector.
"""

# Distributed utilities
import torch.distributed as dist

def master_print(*args, **kwargs):
    """Print only on master node (rank 0) in distributed training, or always in single-node training."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)

__all__ = [
    # Distributed utilities
    "master_print",
] 



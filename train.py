#!/usr/bin/env python3
"""
Entry point for training milco
"""

import sys
import os
import torch
from milco import train_from_args, master_print


def main():
    """Main training function."""
    master_print("Starting model training...")    
    trainer = train_from_args()
    master_print("Training completed successfully!")


if __name__ == "__main__":
    main()

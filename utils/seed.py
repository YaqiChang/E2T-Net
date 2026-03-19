"""Seed utility scaffold for the E2T prototype line.

This file is reserved for consistent random seed initialization across:
- Python
- NumPy
- PyTorch
- CUDA backends
"""


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducible prototype experiments."""
    raise NotImplementedError("Prototype scaffold only.")

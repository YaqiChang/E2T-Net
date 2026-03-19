"""Metric utility scaffold for the E2T prototype line.

Planned metrics:
- accuracy
- precision
- recall
- F1
- balanced accuracy
"""

from typing import Dict


def binary_classification_report(pred, target) -> Dict[str, float]:
    """Compute binary classification metrics for prototype experiments."""
    raise NotImplementedError("Prototype scaffold only.")

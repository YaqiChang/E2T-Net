"""Prototype loss interfaces for E2T experiments.

This file is reserved for training objectives used by the prototype line.
The concrete implementations are intentionally omitted for now.

Planned losses:
- intent classification loss
- temporal smoothness regularization
- early trigger / anticipation regularization
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor


@dataclass
class LossConfig:
    """Configuration stub for prototype loss weighting."""

    intent_weight: float = 1.0
    smoothness_weight: float = 0.0
    early_trigger_weight: float = 0.0


def compute_intent_classification_loss(intent_logits: Tensor, target: Tensor) -> Tensor:
    """Compute the main prototype classification loss."""
    raise NotImplementedError("Prototype scaffold only.")


def compute_temporal_smoothness_loss(belief_seq: Tensor) -> Tensor:
    """Encourage smooth temporal belief evolution if needed."""
    raise NotImplementedError("Prototype scaffold only.")


def compute_early_trigger_loss(intent_prob_seq: Tensor, target: Tensor) -> Tensor:
    """Encourage earlier confidence rise before the crossing event."""
    raise NotImplementedError("Prototype scaffold only.")


def compute_total_loss(
    outputs: Dict[str, Tensor],
    batch: Dict[str, Tensor],
    config: LossConfig,
) -> Dict[str, Tensor]:
    """Combine prototype losses into one structured result."""
    raise NotImplementedError("Prototype scaffold only.")

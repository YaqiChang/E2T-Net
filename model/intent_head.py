"""Prototype intent prediction head for E2T experiments.

This module maps belief states to binary intent outputs.
The goal is to keep the interface simple while preserving room for
sequence-level or frame-level prediction variants later.
"""

from dataclasses import dataclass
from typing import Dict

from torch import Tensor, nn


@dataclass
class IntentHeadConfig:
    """Configuration stub for the intent head."""

    belief_dim: int
    num_classes: int = 2
    pooling: str = "last"


class IntentHead(nn.Module):
    """Minimal interface for mapping belief states to intent predictions.

    Expected future behavior:
    - Input ``belief_seq``: ``[batch, seq_len, belief_dim]``
    - Output ``intent_logits``: ``[batch, num_classes]``
    - Output ``intent_prob``: probabilities for analysis and visualization
    """

    def __init__(self, config: IntentHeadConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, belief_seq: Tensor) -> Dict[str, Tensor]:
        """Predict crossing intent from a sequence of belief states."""
        raise NotImplementedError("Prototype scaffold only.")

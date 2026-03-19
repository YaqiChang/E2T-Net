"""Prototype belief updater for E2T experiments.

This module defines the interface for sequential belief updates based on
per-frame evidence. The implementation is intentionally left blank.

Planned modes:
- ``leaky``: simple leaky accumulation
- ``gru``: recurrent update using a GRU-style hidden state
"""

from dataclasses import dataclass
from typing import Dict, Literal

import torch
from torch import Tensor, nn


BeliefUpdaterType = Literal["leaky", "gru"]


@dataclass
class BeliefUpdaterConfig:
    """Configuration stub for the belief update module."""

    updater_type: BeliefUpdaterType
    evidence_dim: int
    belief_dim: int
    leak: float = 0.9
    dropout: float = 0.0


class BeliefUpdater(nn.Module):
    """Minimal interface for converting evidence sequences into belief states.

    Expected future behavior:
    - Input ``evidence_seq``: ``[batch, seq_len, evidence_dim]``
    - Output ``belief_seq``: ``[batch, seq_len, belief_dim]``
    - Optional hidden summaries for debugging or visualization
    """

    def __init__(self, config: BeliefUpdaterConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, evidence_seq: Tensor) -> Dict[str, Tensor]:
        """Update latent belief state over time from evidence inputs."""
        raise NotImplementedError("Prototype scaffold only.")

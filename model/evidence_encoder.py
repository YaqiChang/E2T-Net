"""Prototype evidence encoder for E2T experiments.

This module is intentionally isolated from the current baseline.
It defines the minimal interfaces for mapping per-frame input features
into per-frame evidence vectors.

Planned responsibility:
- Accept fused frame-wise features such as trajectory, pose, and scene cues.
- Produce an evidence sequence for downstream belief updating.
- Optionally expose a scalar evidence score per frame for analysis.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn


@dataclass
class EvidenceEncoderConfig:
    """Configuration stub for the evidence encoder prototype."""

    input_dim: int
    hidden_dim: int
    evidence_dim: int
    dropout: float = 0.0
    use_score_head: bool = True


class EvidenceEncoder(nn.Module):
    """Minimal interface for frame-wise evidence extraction.

    Expected future behavior:
    - Input shape: ``[batch, seq_len, input_dim]``
    - Output ``evidence_seq`` shape: ``[batch, seq_len, evidence_dim]``
    - Optional ``evidence_score`` shape: ``[batch, seq_len, 1]``
    """

    def __init__(self, config: EvidenceEncoderConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, features: Tensor) -> Dict[str, Optional[Tensor]]:
        """Encode fused per-frame features into evidence tensors.

        Args:
            features: Concatenated frame-wise features.

        Returns:
            A dictionary with at least:
            - ``evidence_seq``: per-frame evidence embeddings
            - ``evidence_score``: optional scalar evidence intensity
        """
        raise NotImplementedError("Prototype scaffold only.")

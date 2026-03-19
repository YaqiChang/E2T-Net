"""Prototype end-to-end E2T network scaffold.

This file wires together:
- evidence encoder
- belief updater
- intent head

It is intentionally kept independent from the existing baseline model.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from model.belief_updater import BeliefUpdater, BeliefUpdaterConfig
from model.evidence_encoder import EvidenceEncoder, EvidenceEncoderConfig
from model.intent_head import IntentHead, IntentHeadConfig


@dataclass
class E2TNetConfig:
    """Configuration stub for the prototype network."""

    evidence: EvidenceEncoderConfig
    belief: BeliefUpdaterConfig
    intent: IntentHeadConfig


class E2TNet(nn.Module):
    """Minimal E2T prototype network interface.

    Planned outputs:
    - ``evidence_seq``
    - ``belief_seq``
    - ``intent_logits``
    - ``intent_prob``
    - Optional debug tensors for visualization
    """

    def __init__(self, config: E2TNetConfig) -> None:
        super().__init__()
        self.config = config
        self.evidence_encoder = EvidenceEncoder(config.evidence)
        self.belief_updater = BeliefUpdater(config.belief)
        self.intent_head = IntentHead(config.intent)

    def forward(
        self,
        trajectory_features: Tensor,
        pose_features: Optional[Tensor] = None,
        scene_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Run the prototype E2T pipeline on fused sequential inputs."""
        raise NotImplementedError("Prototype scaffold only.")

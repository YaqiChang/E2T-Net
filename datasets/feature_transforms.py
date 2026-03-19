"""Feature transform stubs for the E2T prototype line.

This file is intended to centralize preprocessing that should not be spread
across the dataset loader and training script.

Planned transforms:
- normalization
- feature concatenation
- temporal difference features
- speed / acceleration features
"""

from typing import Dict, Optional

import torch
from torch import Tensor


def normalize_features(features: Tensor, stats: Optional[Dict[str, Tensor]] = None) -> Tensor:
    """Normalize raw sequential features for prototype experiments."""
    raise NotImplementedError("Prototype scaffold only.")


def concatenate_modalities(
    trajectory_features: Tensor,
    pose_features: Optional[Tensor] = None,
    scene_features: Optional[Tensor] = None,
) -> Tensor:
    """Concatenate modality-specific features into one per-frame tensor."""
    raise NotImplementedError("Prototype scaffold only.")


def compute_temporal_differences(features: Tensor) -> Tensor:
    """Create simple delta features across adjacent frames."""
    raise NotImplementedError("Prototype scaffold only.")


def compute_motion_features(positions: Tensor) -> Tensor:
    """Create speed or acceleration style features from positions."""
    raise NotImplementedError("Prototype scaffold only.")

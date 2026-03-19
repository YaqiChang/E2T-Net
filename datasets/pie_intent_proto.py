"""Prototype dataset scaffold for E2T intent experiments.

This dataset is intentionally separated from the current baseline data
pipeline. It is designed for rapid prototyping of:
- trajectory-only inputs
- trajectory + pose inputs
- trajectory + pose + scene inputs
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class PIEIntentProtoConfig:
    """Configuration stub for the prototype dataset."""

    data_dir: str
    split: str
    obs_len: int
    pred_len: int
    use_pose: bool = False
    use_scene: bool = False
    return_sample_name: bool = True


class PIEIntentProtoDataset(Dataset):
    """Minimal dataset interface for prototype intent experiments.

    Planned sample outputs:
    - ``trajectory_features``
    - ``pose_features`` (optional)
    - ``scene_features`` (optional)
    - ``intent_label``
    - ``time_index``
    - ``sample_name`` (optional)
    """

    def __init__(self, config: PIEIntentProtoConfig) -> None:
        self.config = config

    def __len__(self) -> int:
        """Return dataset length once the prototype loader is implemented."""
        raise NotImplementedError("Prototype scaffold only.")

    def __getitem__(self, index: int) -> Dict[str, Optional[Tensor]]:
        """Fetch one prototype sample for E2T training or evaluation."""
        raise NotImplementedError("Prototype scaffold only.")

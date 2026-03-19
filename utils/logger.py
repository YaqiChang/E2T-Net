"""Logging utility scaffold for the E2T prototype line.

This file is reserved for:
- text logging
- scalar metric logging
- optional TensorBoard integration
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class LoggerConfig:
    """Configuration stub for prototype logging."""

    output_dir: str
    use_tensorboard: bool = False


class ExperimentLogger:
    """Minimal logging interface for prototype training and evaluation."""

    def __init__(self, config: LoggerConfig) -> None:
        self.config = config

    def log_scalars(self, step: int, values: Dict[str, float]) -> None:
        """Log a flat dictionary of scalar values."""
        raise NotImplementedError("Prototype scaffold only.")

    def close(self) -> None:
        """Close logger resources if any backend is attached."""
        raise NotImplementedError("Prototype scaffold only.")

"""Standalone training entrypoint for the E2T prototype line.

This script is intentionally separate from the baseline training script.
It is meant to support:
- reading a dedicated prototype config
- building prototype datasets
- building the E2T prototype model
- running training
- saving logs and checkpoints
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for prototype training."""
    parser = argparse.ArgumentParser(description="Train E2T prototype model")
    parser.add_argument("--config", type=str, default="configs/e2t_proto.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="output/e2t_proto")
    return parser.parse_args()


def main() -> None:
    """Prototype training scaffold entrypoint."""
    raise SystemExit("Prototype scaffold only. Training loop not implemented yet.")


if __name__ == "__main__":
    main()

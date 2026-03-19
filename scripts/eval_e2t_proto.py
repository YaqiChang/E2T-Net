"""Standalone evaluation entrypoint for the E2T prototype line.

This script is reserved for:
- loading a prototype checkpoint
- evaluating classification metrics
- exporting intermediate tensors
- saving outputs used by visualization scripts
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for prototype evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate E2T prototype model")
    parser.add_argument("--config", type=str, default="configs/e2t_proto.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="output/e2t_proto_eval")
    return parser.parse_args()


def main() -> None:
    """Prototype evaluation scaffold entrypoint."""
    raise SystemExit("Prototype scaffold only. Evaluation loop not implemented yet.")


if __name__ == "__main__":
    main()

"""Visualization scaffold for prototype evidence curves.

Planned plots:
- per-sample evidence magnitude over time
- class-wise evidence averages
- modality-specific evidence shifts before crossing
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evidence curve plotting."""
    parser = argparse.ArgumentParser(description="Plot prototype evidence curves")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="evidence_curve.png")
    parser.add_argument("--sample_id", type=str, default="")
    return parser.parse_args()


def main() -> None:
    """Evidence visualization scaffold entrypoint."""
    raise SystemExit("Prototype scaffold only. Plotting logic not implemented yet.")


if __name__ == "__main__":
    main()

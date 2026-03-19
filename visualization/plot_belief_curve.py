"""Visualization scaffold for prototype belief curves.

Planned plots:
- single-sample temporal belief curve
- class-wise average belief curves
- confidence rise before crossing events
"""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for belief curve plotting."""
    parser = argparse.ArgumentParser(description="Plot prototype belief curves")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="belief_curve.png")
    parser.add_argument("--sample_id", type=str, default="")
    return parser.parse_args()


def main() -> None:
    """Belief visualization scaffold entrypoint."""
    raise SystemExit("Prototype scaffold only. Plotting logic not implemented yet.")


if __name__ == "__main__":
    main()

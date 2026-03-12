import argparse
from ast import literal_eval
from pathlib import Path

import pandas as pd


def _parse(value):
    if isinstance(value, str):
        return literal_eval(value)
    return value


def main():
    parser = argparse.ArgumentParser(description="Check crossing_true/label consistency.")
    parser.add_argument(
        "--csv",
        default="output/jaad_train_5_5_5.csv",
        help="Path to dataset CSV (default: output/jaad_train_5_5_5.csv)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of positive samples to print (default: 5)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "crossing_true" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain crossing_true and label columns.")

    crossing = df["crossing_true"].apply(_parse)
    label = df["label"].astype(float)

    derived_label = crossing.apply(lambda x: 1.0 if 1.0 in x else 0.0)
    mismatch = int((derived_label != label).sum())
    print(f"label mismatch count: {mismatch}/{len(df)}")

    if args.samples <= 0:
        return

    pos_idx = df.index[derived_label == 1.0][: args.samples]
    for i in pos_idx:
        print("ID:", df.loc[i, "ID"])
        print("crossing_true:", crossing[i])
        print("label:", label[i])
        if "filename" in df.columns:
            files = _parse(df.loc[i, "filename"])
            print("first frames:", files[:3])
        print("---")


if __name__ == "__main__":
    main()

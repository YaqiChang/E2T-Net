import argparse
import re
import subprocess
import sys
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Run train.py with different pos_weight_override values.")
    parser.add_argument(
        "--values",
        nargs="+",
        type=float,
        default=[2, 5, 10, 20, 30, 40, 50],
        help="pos_weight_override values to test",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="n_epochs for each run",
    )
    parser.add_argument(
        "--debug_mode",
        type=bool,
        default=True,
        help="whether to enable debug_mode",
    )
    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="extra args passed to train.py",
    )
    parser.add_argument(
        "--log_path",
        default="output/log.txt",
        help="path to log.txt for parsing results",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%m_%d_%H_%M")
    for val in args.values:
        cmd = [
            sys.executable,
            "train.py",
            "--debug_mode",
            str(args.debug_mode),
            "--debug_only_crossing_steps",
            "0",
            "--n_epochs",
            str(args.epochs),
            "--pos_weight_override",
            str(val),
        ]
        if args.extra_args:
            cmd.extend(args.extra_args)
        print(f"\n=== Running pos_weight_override={val} ===")
        ret = subprocess.call(cmd)
        if ret != 0:
            print(f"Run failed for pos_weight_override={val}, exit={ret}")
            break
        summary = parse_log(args.log_path)
        if summary:
            intent_acc, f1_int, tp, fp, fn, tn = summary
            print(
                f"[pos_weight={val}] intent_acc={intent_acc:.4f} "
                f"f1_int={f1_int:.4f} TP/FP/FN/TN={tp}/{fp}/{fn}/{tn}"
            )
        else:
            print(f"[pos_weight={val}] No metrics found in log.")


def parse_log(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None

    intent_conf = re.findall(r"Debug intent confusion TP/FP/FN/TN: (\\d+) (\\d+) (\\d+) (\\d+)", text)
    intent_acc = re.findall(r"\\| intention_acc: ([0-9.]+)", text)
    f1_int = re.findall(r"\\| f1_int: ([0-9.]+)", text)

    if not intent_conf or not intent_acc or not f1_int:
        return None

    tp, fp, fn, tn = map(int, intent_conf[-1])
    intent_acc_val = float(intent_acc[-1])
    f1_int_val = float(f1_int[-1])
    return intent_acc_val, f1_int_val, tp, fp, fn, tn


if __name__ == "__main__":
    main()

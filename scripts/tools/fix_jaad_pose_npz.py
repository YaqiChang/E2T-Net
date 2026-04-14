#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from path_config import get_path_value, load_path_config, normalize_dataset_path


def fix_keypoints(keypoints: np.ndarray) -> np.ndarray:
    # rot90_cw_hflip => swap x/y without mirroring
    fixed = keypoints.copy()
    y = fixed[:, :, 0].copy()
    x = fixed[:, :, 1].copy()
    fixed[:, :, 0] = x
    fixed[:, :, 1] = y
    return fixed


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="preprocess/config/default.yaml")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_path_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Fix JAAD pose npz keypoints orientation.")
    parser.add_argument(
        "--config",
        type=str,
        default=pre_args.config,
        help="Path to preprocess config YAML.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=get_path_value(
            "JAAD_pose_npz_raw",
            str(Path(get_path_value("JAAD_pn_root", "", config=config)) / "jaad_pose_annotations.npz"),
            config=config,
        ),
        help="Input pose npz.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=get_path_value(
            "JAAD_pose_npz_fixed",
            str(Path(get_path_value("JAAD_pn_root", "", config=config)) / "jaad_pose_annotations_fixed.npz"),
            config=config,
        ),
        help="Output pose npz.",
    )
    args = parser.parse_args()

    in_path = Path(normalize_dataset_path(args.input, config=config))
    out_path = Path(normalize_dataset_path(args.output, config=config))
    payload = np.load(in_path, allow_pickle=True)

    keypoints = payload["keypoints"]
    fixed_keypoints = fix_keypoints(keypoints)

    np.savez_compressed(
        out_path,
        ped_ids=payload["ped_ids"],
        ped_ptr=payload["ped_ptr"],
        video_id=payload["video_id"],
        frame=payload["frame"],
        bbox=payload["bbox"],
        keypoints=fixed_keypoints,
    )
    print(f"Wrote fixed pose npz to: {out_path}")


if __name__ == "__main__":
    main()

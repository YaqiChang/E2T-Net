#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def fix_keypoints(keypoints: np.ndarray) -> np.ndarray:
    # rot90_cw_hflip => swap x/y without mirroring
    fixed = keypoints.copy()
    y = fixed[:, :, 0].copy()
    x = fixed[:, :, 1].copy()
    fixed[:, :, 0] = x
    fixed[:, :, 1] = y
    return fixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix JAAD pose npz keypoints orientation.")
    parser.add_argument(
        "--input",
        type=str,
        default="/media/meta/File/datasets/Intention/JAAD_dataset/PN_ego/jaad_pose_annotations.npz",
        help="Input pose npz.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/media/meta/File/datasets/Intention/JAAD_dataset/PN_ego/jaad_pose_annotations_fixed.npz",
        help="Output pose npz.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
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

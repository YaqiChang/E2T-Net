#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from path_config import get_path_value, load_path_config, normalize_dataset_path


POSE_KEYS = ["ped_ids", "ped_ptr", "video_id", "frame", "bbox", "keypoints"]
SEQ_KEYS = [
    "ID",
    "ped_id",
    "video_id",
    "frame_obs",
    "ped_attribute",
    "bounding_box",
    "future_bounding_box",
    "ped_behavior",
    "scene_attribute",
    "imagefolderpath",
    "filename",
    "crossing_obs",
    "crossing_true",
    "label",
]


def fmt_shape(arr: np.ndarray) -> str:
    return "x".join(str(dim) for dim in arr.shape)


def nan_stats(arr: np.ndarray):
    if not np.issubdtype(arr.dtype, np.floating):
        return None
    total = arr.size
    nan_count = int(np.isnan(arr).sum())
    return nan_count, total


def inspect_pose_npz(path: Path, lines: List[str]) -> None:
    lines.append(f"Pose NPZ: {path}")
    if not path.exists():
        lines.append("  MISSING file")
        return
    payload = np.load(path, allow_pickle=True)
    missing = [k for k in POSE_KEYS if k not in payload.files]
    if missing:
        lines.append(f"  Missing keys: {', '.join(missing)}")
    for key in POSE_KEYS:
        if key not in payload.files:
            continue
        arr = payload[key]
        lines.append(f"  {key}: shape={fmt_shape(arr)} dtype={arr.dtype}")

    if "keypoints" in payload.files:
        keypoints = payload["keypoints"]
        n = keypoints.shape[0]
        if keypoints.ndim == 3:
            missing_mask = np.isnan(keypoints).all(axis=(1, 2))
            missing_count = int(missing_mask.sum())
            missing_rate = (missing_count / n) if n else 0.0
            lines.append(f"  keypoints_missing: {missing_count}/{n} ({missing_rate:.2%})")
        stats = nan_stats(keypoints)
        if stats is not None:
            nan_count, total = stats
            lines.append(f"  keypoints_nan: {nan_count}/{total} ({nan_count / total:.2%})")


def inspect_sequence_npz(npz_dir: Path, lines: List[str]) -> None:
    lines.append(f"\nSequence NPZ dir: {npz_dir}")
    if not npz_dir.is_dir():
        lines.append("  MISSING dir")
        return
    files = sorted(npz_dir.glob("jaad_*.npz"))
    if not files:
        lines.append("  No jaad_*.npz files found")
        return
    for npz_path in files:
        lines.append(f"\n  File: {npz_path.name}")
        payload = np.load(npz_path, allow_pickle=True)
        missing = [k for k in SEQ_KEYS if k not in payload.files]
        if missing:
            lines.append(f"    Missing keys: {', '.join(missing)}")
        else:
            lines.append("    Keys: OK")
        for key in SEQ_KEYS:
            if key not in payload.files:
                continue
            arr = payload[key]
            lines.append(f"    {key}: shape={fmt_shape(arr)} dtype={arr.dtype}")
            stats = nan_stats(arr)
            if stats is not None:
                nan_count, total = stats
                if total > 0:
                    lines.append(f"      nan: {nan_count}/{total} ({nan_count / total:.2%})")


def find_frame_image(images_root: Path, vid_id: str, frame_id: int) -> Path:
    frame_name = f"{frame_id + 1:05d}"
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = images_root / vid_id / f"{frame_name}{ext}"
        if candidate.exists():
            return candidate
    return images_root / vid_id / f"{frame_name}.png"


def resolve_images_root(root: Path, image_root: Optional[Path]) -> Path:
    if image_root is not None:
        return image_root
    dataset_root = root.parent
    for candidate in [
        dataset_root / "JAAD_clips" / "frames",
        dataset_root / "images",
    ]:
        if candidate.is_dir():
            return candidate
    return dataset_root / "JAAD_clips" / "frames"


def load_skeleton(joint_count: int) -> Tuple[list, Optional[object]]:
    repo_root = Path(__file__).resolve().parents[1]
    simple_hrnet_root = repo_root / "simple-HRNet"
    sys.path.insert(0, str(simple_hrnet_root))
    from misc.visualization import joints_dict  # noqa: WPS433

    joints_meta = joints_dict()
    if joint_count == 17:
        return joints_meta["coco"]["skeleton"], joints_meta
    if joint_count == 16:
        return joints_meta["mpii"]["skeleton"], joints_meta
    return [], joints_meta


def find_video_csv(root: Path, vid_id: str) -> Optional[Path]:
    for split in ["train", "val", "test"]:
        candidate = root / split / f"{vid_id}.csv"
        if candidate.exists():
            return candidate
    return None


def load_frame_labels(csv_path: Path, ped_id: str) -> Tuple[dict, bool]:
    df = pd.read_csv(csv_path)
    if "ID" not in df.columns or "frame" not in df.columns:
        return {}, False
    df["ID"] = df["ID"].astype(str)
    df = df[df["ID"] == str(ped_id)]
    if df.empty:
        return {}, "intention_prob" in df.columns
    has_intention = "intention_prob" in df.columns
    labels = {}
    for _, row in df.iterrows():
        frame_id = int(row["frame"])
        crossing = row["crossing_true"] if "crossing_true" in df.columns else np.nan
        intention = row["intention_prob"] if has_intention else np.nan
        labels[frame_id] = (crossing, intention)
    return labels, has_intention


def expand_ped_ids(payload: np.lib.npyio.NpzFile, total: int) -> np.ndarray:
    ped_ids = payload["ped_ids"]
    if ped_ids.shape[0] == total:
        return ped_ids.astype(str)
    if "ped_ptr" not in payload.files:
        return ped_ids.astype(str)
    ped_ptr = payload["ped_ptr"]
    expanded = np.empty((total,), dtype=object)
    for i in range(len(ped_ids)):
        start = int(ped_ptr[i])
        end = int(ped_ptr[i + 1])
        expanded[start:end] = ped_ids[i]
    return expanded.astype(str)


def transform_pose(pose: np.ndarray, image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return pose
    h, w = image.shape[:2]
    out = pose.copy()
    y = out[:, 0].copy()
    x = out[:, 1].copy()
    if mode == "swap":
        out[:, 0] = x
        out[:, 1] = y
    elif mode == "hflip":
        out[:, 0] = y
        out[:, 1] = (w - 1) - x
    elif mode == "vflip":
        out[:, 0] = (h - 1) - y
        out[:, 1] = x
    elif mode == "rot90_cw":
        out[:, 0] = x
        out[:, 1] = (h - 1) - y
    elif mode == "rot90_ccw":
        out[:, 0] = (w - 1) - x
        out[:, 1] = y
    elif mode == "rot90_cw_hflip":
        # rotate clockwise, then horizontal flip in rotated frame
        out[:, 0] = x
        out[:, 1] = y
    elif mode == "rot90_cw_vflip":
        # rotate clockwise, then vertical flip in rotated frame
        out[:, 0] = (w - 1) - x
        out[:, 1] = (h - 1) - y
    return out


def visualize_random_video_pose(
    root: Path,
    image_root: Optional[Path],
    lines: List[str],
    grid_size: int,
    pose_transform: str,
) -> None:
    pose_path = root / "jaad_pose_annotations_fixed.npz"
    lines.append("\nPose visualization:")
    if not pose_path.exists():
        lines.append("  pose npz missing; skip visualization")
        return

    payload = np.load(pose_path, allow_pickle=True)
    keypoints = payload["keypoints"]
    if keypoints.size == 0:
        lines.append("  empty keypoints; skip visualization")
        return

    video_ids = payload["video_id"]
    if video_ids.size == 0:
        lines.append("  no video ids; skip visualization")
        return

    unique_videos = np.unique(video_ids.astype(str))
    vid_id = str(np.random.choice(unique_videos))
    vid_indices = np.where(video_ids.astype(str) == vid_id)[0]
    if vid_indices.size == 0:
        lines.append("  no poses for chosen video; skip visualization")
        return

    ped_ids_full = expand_ped_ids(payload, len(video_ids))
    ped_ids_for_video = ped_ids_full[vid_indices].astype(str)
    unique_peds = np.unique(ped_ids_for_video)
    if unique_peds.size == 0:
        lines.append("  no ped ids for chosen video; skip visualization")
        return
    ped_id = str(np.random.choice(unique_peds))
    indices = vid_indices[ped_ids_for_video == ped_id]
    if indices.size == 0:
        lines.append("  no poses for chosen person; skip visualization")
        return

    images_root = resolve_images_root(root, image_root)
    csv_path = find_video_csv(root, vid_id)
    frame_labels = {}
    has_intention = False
    if csv_path is not None:
        frame_labels, has_intention = load_frame_labels(csv_path, ped_id)
    else:
        lines.append(f"  csv not found for video {vid_id}")
    first_image = None
    for idx in indices:
        frame_id = int(payload["frame"][idx])
        image_path = find_frame_image(images_root, vid_id, frame_id)
        if image_path.exists():
            first_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if first_image is not None:
                break
    if first_image is None:
        lines.append(f"  could not load any frame for video {vid_id}")
        return

    skeleton, _ = load_skeleton(keypoints.shape[1])
    tile_h, tile_w = first_image.shape[:2]
    total_tiles = grid_size * grid_size
    if indices.size > total_tiles:
        chosen = np.random.choice(indices, size=total_tiles, replace=False)
    else:
        chosen = indices

    grid = np.zeros((tile_h * grid_size, tile_w * grid_size, 3), dtype=first_image.dtype)
    for tile_idx, idx in enumerate(chosen):
        frame_id = int(payload["frame"][idx])
        image_path = find_frame_image(images_root, vid_id, frame_id)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            image = np.zeros((tile_h, tile_w, 3), dtype=first_image.dtype)
        else:
            image = cv2.resize(image, (tile_w, tile_h))

        pose = transform_pose(keypoints[idx], image, pose_transform)
        bbox = payload["bbox"][idx]
        for joint in pose:
            y, x, conf = joint
            if np.isnan(conf) or conf <= 0.0:
                continue
            cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)
        for a, b in skeleton:
            if a >= pose.shape[0] or b >= pose.shape[0]:
                continue
            y1, x1, c1 = pose[a]
            y2, x2, c2 = pose[b]
            if np.isnan(c1) or np.isnan(c2) or c1 <= 0.0 or c2 <= 0.0:
                continue
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        crossing = frame_labels.get(frame_id, (np.nan, np.nan))[0]
        intention = frame_labels.get(frame_id, (np.nan, np.nan))[1]
        label = f"cross:{int(crossing)}"
        if has_intention and not np.isnan(intention):
            label += f" int:{float(intention):.2f}"
        else:
            label += " int:NA"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        x0, y0 = image.shape[1] - text_w - 12, 8
        x1, y1 = image.shape[1] - 4, y0 + text_h + 6
        cv2.rectangle(image, (x0, y0), (x1, y1), (255, 255, 255), -1)
        cv2.putText(
            image,
            label,
            (x0 + 4, y1 - 4),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

        row = tile_idx // grid_size
        col = tile_idx % grid_size
        y0 = row * tile_h
        x0 = col * tile_w
        grid[y0 : y0 + tile_h, x0 : x0 + tile_w] = image

    out_path = root / "pose_random_video_grid.jpg"
    if not cv2.imwrite(str(out_path), grid):
        lines.append("  failed to write visualization")
        return
    lines.append(f"  saved: {out_path}")
    lines.append(f"  sample: video={vid_id} ped_id={ped_id} poses={indices.size}")


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="preprocess/config/default.yaml")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_path_config(pre_args.config)

    parser = argparse.ArgumentParser(description="Inspect JAAD pose and sequence NPZ files.")
    parser.add_argument(
        "--config",
        type=str,
        default=pre_args.config,
        help="Path to preprocess config YAML.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=get_path_value("JAAD_pn_root", "", config=config),
        help="JAAD PN_ego root directory.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="pose_npz_report.txt",
        help="Report filename (written under --root if relative).",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="Optional JAAD images root (e.g. JAAD_clips/frames).",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=4,
        help="Grid size N for N x N visualization.",
    )
    parser.add_argument(
        "--pose_transform",
        type=str,
        default="none",
        choices=[
            "none",
            "swap",
            "hflip",
            "vflip",
            "rot90_cw",
            "rot90_ccw",
            "rot90_cw_hflip",
            "rot90_cw_vflip",
        ],
        help="Apply a pose coordinate transform for debugging.",
    )
    args = parser.parse_args()

    root = Path(normalize_dataset_path(args.root, config=config))
    image_root = Path(normalize_dataset_path(args.image_root, config=config)) if args.image_root else None
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = root / report_path

    lines: List[str] = []
    inspect_pose_npz(root / "jaad_pose_annotations_fixed.npz", lines)
    inspect_sequence_npz(root / "npz", lines)
    visualize_random_video_pose(root, image_root, lines, args.grid_size, args.pose_transform)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()

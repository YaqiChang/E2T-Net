#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import pandas as pd
import numpy as np
import sys


def pose_is_valid(pose: np.ndarray) -> bool:
    if pose is None:
        return False
    return not np.isnan(pose).all()


def find_video_with_change(root: Path, pose_index: dict) -> Tuple[Optional[Path], Optional[str], Optional[int]]:
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for csv_path in sorted(split_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            if "ID" not in df.columns or "crossing_true" not in df.columns:
                continue
            for ped_id, group in df.groupby("ID"):
                labels = group["crossing_true"].dropna().unique()
                if len(labels) < 2:
                    continue
                group = group.sort_values("frame")
                prev = None
                for _, row in group.iterrows():
                    val = row["crossing_true"]
                    if prev is not None and val != prev:
                        key = (csv_path.stem, int(row["frame"]), str(ped_id))
                        pose = pose_index.get(key)
                        if not pose_is_valid(pose):
                            print("detect change, but no pose detected go on for next...")
                            prev = val
                            continue
                        return csv_path, str(ped_id), int(row["frame"])
                    prev = val
    return None, None, None


def find_video_changes(root: Path, pose_index: dict, max_count: int):
    results = []
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for csv_path in sorted(split_dir.glob("*.csv")):
            df = pd.read_csv(csv_path)
            if "ID" not in df.columns or "crossing_true" not in df.columns:
                continue
            for ped_id, group in df.groupby("ID"):
                labels = group["crossing_true"].dropna().unique()
                if len(labels) < 2:
                    continue
                group = group.sort_values("frame")
                prev = None
                for _, row in group.iterrows():
                    val = row["crossing_true"]
                    if prev is not None and val != prev:
                        key = (csv_path.stem, int(row["frame"]), str(ped_id))
                        pose = pose_index.get(key)
                        if not pose_is_valid(pose):
                            print("detect change, but no pose detected go on for next...")
                            prev = val
                            continue
                        results.append((csv_path, str(ped_id), int(row["frame"])))
                        if len(results) >= max_count:
                            return results
                        break
                    prev = val
    return results


def load_ped_group(csv_path: Path, ped_id: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["ID"] = df["ID"].astype(str)
    group = df[df["ID"] == ped_id].sort_values("frame")
    return group


def resolve_image_path(row: pd.Series) -> Path:
    folder = Path(row["imagefolderpath"])
    filename = str(row["filename"])
    if folder.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        return folder
    return folder / filename


def draw_tile(image: cv2.Mat, row: pd.Series, source_text: str) -> cv2.Mat:
    x = float(row["x"])
    y = float(row["y"])
    w = float(row["w"])
    h = float(row["h"])
    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + w))
    y2 = int(round(y + h))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"cross:{int(row['crossing_true'])}"
    if "intention_prob" in row:
        label += f" int:{float(row['intention_prob']):.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
    x0, y0 = image.shape[1] - text_w - 12, 8
    x1b, y1b = image.shape[1] - 4, y0 + text_h + 6
    cv2.rectangle(image, (x0, y0), (x1b, y1b), (255, 255, 255), -1)
    cv2.putText(
        image,
        label,
        (x0 + 4, y1b - 4),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(source_text, font, font_scale, thickness)
    x0, y0 = 6, image.shape[0] - text_h - 10
    x1b, y1b = x0 + text_w + 6, y0 + text_h + 6
    cv2.rectangle(image, (x0, y0), (x1b, y1b), (255, 255, 255), -1)
    cv2.putText(
        image,
        source_text,
        (x0 + 3, y1b - 3),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )
    return image


def load_pose_index(pose_path: Path):
    payload = np.load(pose_path, allow_pickle=True)
    keypoints = payload["keypoints"]
    video_ids = payload["video_id"]
    frames = payload["frame"]
    if payload["ped_ids"].shape[0] == len(video_ids):
        ped_ids = payload["ped_ids"].astype(str)
    else:
        ped_ptr = payload["ped_ptr"]
        ped_ids = np.empty((len(video_ids),), dtype=object)
        for i in range(len(payload["ped_ids"])):
            start = int(ped_ptr[i])
            end = int(ped_ptr[i + 1])
            ped_ids[start:end] = payload["ped_ids"][i]
        ped_ids = ped_ids.astype(str)

    pose_index = {}
    for idx in range(len(frames)):
        key = (str(video_ids[idx]), int(frames[idx]), str(ped_ids[idx]))
        pose_index[key] = keypoints[idx]
    return pose_index, keypoints.shape[1]


def load_skeleton(joint_count: int):
    repo_root = Path(__file__).resolve().parents[1]
    simple_hrnet_root = repo_root / "simple-HRNet"
    sys.path.insert(0, str(simple_hrnet_root))
    from misc.visualization import joints_dict  # noqa: WPS433

    joints_meta = joints_dict()
    if joint_count == 17:
        return joints_meta["coco"]["skeleton"]
    if joint_count == 16:
        return joints_meta["mpii"]["skeleton"]
    return []


def transform_pose(pose: np.ndarray, image: np.ndarray, mode: str) -> np.ndarray:
    if pose is None or mode == "none":
        return pose
    h, w = image.shape[:2]
    out = pose.copy()
    y = out[:, 0].copy()
    x = out[:, 1].copy()
    if mode == "rot90_cw_hflip":
        out[:, 0] = x
        out[:, 1] = y
    elif mode == "swap":
        out[:, 0] = x
        out[:, 1] = y
    return out


def draw_pose(image: np.ndarray, pose: np.ndarray, skeleton) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect JAAD crossing label changes.")
    parser.add_argument(
        "--root",
        type=str,
        default="/media/meta/File/datasets/Intention/JAAD_dataset/PN_ego",
        help="PN_ego root with train/val/test CSVs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inspect_change.jpg",
        help="Output image filename (written under --root if relative).",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=4,
        help="Grid size N for N x N visualization.",
    )
    parser.add_argument(
        "--max_groups",
        type=int,
        default=1,
        help="Number of change groups to export.",
    )
    parser.add_argument(
        "--pose_npz",
        type=str,
        default="/media/meta/File/datasets/Intention/JAAD_dataset/PN_ego/jaad_pose_annotations_fixed.npz",
        help="Pose npz path.",
    )
    parser.add_argument(
        "--pose_transform",
        type=str,
        default="none",
        choices=["none", "swap", "rot90_cw_hflip"],
        help="Pose transform mode.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / out_path

    pose_index, joints = load_pose_index(Path(args.pose_npz))
    skeleton = load_skeleton(joints)

    targets = find_video_changes(root, pose_index, args.max_groups)
    if not targets:
        print("No video with crossing label change found.")
        return
    for idx, (csv_path, ped_id, frame_id) in enumerate(targets, start=1):
        group = load_ped_group(csv_path, ped_id)
        if group.empty:
            print("Failed to locate matching row for visualization.")
            continue

        frames = group["frame"].tolist()
        if frame_id not in frames:
            print("Change frame not found in group.")
            continue

        change_idx = frames.index(frame_id)
        total_tiles = args.grid_size * args.grid_size
        half = total_tiles // 2
        start = max(0, change_idx - half)
        end = min(len(frames), start + total_tiles)
        start = max(0, end - total_tiles)
        selected_frames = frames[start:end]

        first_row = group[group["frame"] == selected_frames[0]].iloc[0]
        first_image = cv2.imread(str(resolve_image_path(first_row)), cv2.IMREAD_COLOR)
        if first_image is None:
            print("Failed to read first image for grid.")
            continue

        tile_h, tile_w = first_image.shape[:2]
        grid = np.zeros((tile_h * args.grid_size, tile_w * args.grid_size, 3), dtype=first_image.dtype)

        for tile_idx, frame in enumerate(selected_frames):
            row = group[group["frame"] == frame].iloc[0]
            image_path = resolve_image_path(row)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                image = first_image.copy()
            else:
                image = cv2.resize(image, (tile_w, tile_h))
            key = (str(csv_path.stem), int(frame), str(ped_id))
            pose = pose_index.get(key)
            if pose_is_valid(pose):
                pose = transform_pose(pose, image, args.pose_transform)
                draw_pose(image, pose, skeleton)
            source_text = f"{csv_path.stem}/{row['filename']}"
            image = draw_tile(image, row, source_text)
            r = tile_idx // args.grid_size
            c = tile_idx % args.grid_size
            y0 = r * tile_h
            x0 = c * tile_w
            grid[y0 : y0 + tile_h, x0 : x0 + tile_w] = image

        suffix = f"_{idx}" if args.max_groups > 1 else ""
        out_file = out_path.with_name(f"{out_path.stem}{suffix}{out_path.suffix}")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_file), grid):
            raise RuntimeError(f"Failed to write image: {out_file}")
        print(f"Saved: {out_file}")
        print(f"Source CSV: {csv_path}")
        print(f"Video: {csv_path.stem}, ped_id: {ped_id}, frame: {frame_id}")


if __name__ == "__main__":
    main()

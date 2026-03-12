#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "preprocess"))

import pie_data  # noqa: E402

def iter_annotation_videos(annotations_root: Path):
    for set_dir in sorted(p for p in annotations_root.iterdir() if p.is_dir()):
        set_id = set_dir.name
        for xml_file in sorted(set_dir.glob("*_annt.xml")):
            vid_id = xml_file.name.split("_annt.xml")[0]
            yield set_id, vid_id


def count_ped_frames(pie: pie_data.PIE, set_id: str, vid_id: str) -> int:
    vid_data = pie._get_annotations(set_id, vid_id)
    ped_frames = set()
    for ped in vid_data["ped_annotations"].values():
        ped_frames.update(ped.get("frames", []))
    return len(ped_frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count PIE frames that contain pedestrian bboxes.")
    parser.add_argument("--data_path", required=True, help="PIE root directory (annotations/, images/).")
    parser.add_argument("--set_id", default="", help="Optional PIE set id, e.g. set03.")
    parser.add_argument("--video_id", default="", help="Optional PIE video id, e.g. video_0001.")
    args = parser.parse_args()

    pie = pie_data.PIE(data_path=args.data_path)
    annotations_root = Path(args.data_path) / "annotations"

    totals = {"videos": 0, "frames_with_peds": 0}
    per_set = defaultdict(lambda: {"videos": 0, "frames_with_peds": 0})

    for set_id, vid_id in iter_annotation_videos(annotations_root):
        if args.set_id and set_id != args.set_id:
            continue
        if args.video_id and vid_id != args.video_id:
            continue
        frame_count = count_ped_frames(pie, set_id, vid_id)
        print(f"{set_id}/{vid_id}: {frame_count} frames with peds")
        totals["videos"] += 1
        totals["frames_with_peds"] += frame_count
        per_set[set_id]["videos"] += 1
        per_set[set_id]["frames_with_peds"] += frame_count

    print("\nTotals:")
    for set_id in sorted(per_set):
        info = per_set[set_id]
        print(f"  {set_id}: {info['videos']} videos, {info['frames_with_peds']} frames with peds")
    print(f"  ALL: {totals['videos']} videos, {totals['frames_with_peds']} frames with peds")


if __name__ == "__main__":
    main()

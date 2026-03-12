import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

import pie_data


def parse_resolution(value: str) -> Tuple[int, int]:
    parts = [item.strip() for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Resolution must be 'height,width'.")
    try:
        height = int(parts[0])
        width = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Resolution must be 'height,width'.") from exc
    return height, width


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as file:
        payload = yaml.safe_load(file)
    return payload or {}


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="preprocess/config/default.yaml")
    pre_args, _ = pre_parser.parse_known_args()
    config = load_config(Path(pre_args.config))

    required_keys = [
        "PIE_root",
        "PIE_preproc_root",
        "PIE_pose_root",
        "PIE_pose_npz",
        "hrnet_checkpoint",
        "hrnet_c",
        "hrnet_joints",
        "hrnet_resolution",
        "hrnet_device",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")

    data_path_default = config["PIE_root"]
    image_root_default = config["PIE_preproc_root"]
    output_dir_default = config["PIE_pose_root"]
    pose_npz_default = config["PIE_pose_npz"]
    checkpoint_default = config["hrnet_checkpoint"]
    c_default = int(config["hrnet_c"])
    joints_default = int(config["hrnet_joints"])
    resolution_default = parse_resolution(config["hrnet_resolution"])
    device_default = config["hrnet_device"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=pre_args.config,
        help="Path to preprocess config YAML.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=data_path_default,
        help="Path to cloned PIE repository.",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=image_root_default,
        help="Root directory that contains PIE images (setXX/video_xxxx).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir_default,
        help="Directory to write CSVs and pose npz.",
    )
    parser.add_argument(
        "--pose_npz",
        type=str,
        default=pose_npz_default,
        help="Pose output npz file (absolute or relative to output_dir).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=checkpoint_default,
        help="Path to HRNet checkpoint.",
    )
    parser.add_argument("--c", type=int, default=c_default, help="HRNet channels (32 or 48).")
    parser.add_argument("--joints", type=int, default=joints_default, help="Number of joints.")
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=resolution_default,
        help="Input resolution as 'height,width'.",
    )
    parser.add_argument(
        "--device",
        default=device_default,
        help="Torch device string, e.g. cpu, cuda, cuda:0.",
    )
    parser.add_argument("--simple_test", action="store_true", help="Run single-frame test with visualization.")
    parser.add_argument("--process_dataset", action="store_true", help="Process the full PIE dataset.")
    parser.add_argument(
        "--test_image",
        type=str,
        default="",
        help="Path to a test image. If set, --test_bboxes is required.",
    )
    parser.add_argument(
        "--test_bboxes",
        type=str,
        default="",
        help="Path to a bbox txt file: one line 'x1,y1,x2,y2' per person.",
    )
    parser.add_argument("--test_set", type=str, default="set02", help="PIE set id for simple test.")
    parser.add_argument("--test_video", type=str, default="", help="PIE video id for simple test.")
    parser.add_argument("--test_frame", type=int, default=-1, help="Frame id for simple test.")
    parser.add_argument("--save_vis", action="store_true", help="Save visualization image.")
    parser.add_argument("--save_crop_vis", action="store_true", help="Save per-person crop visualizations.")
    parser.add_argument("--save_skeleton_only", action="store_true", help="Save skeleton-only images per person.")
    parser.add_argument("--vis_output", type=str, default="hrnet_pie_vis.jpg", help="Vis output path.")
    parser.add_argument(
        "--vis_conf",
        type=float,
        default=0.2,
        help="Confidence threshold for drawing keypoints.",
    )
    parser.add_argument(
        "--bbox_expand",
        type=float,
        default=1.6,
        help="Expand bbox by this scale before pose inference (e.g. 1.2).",
    )
    parser.add_argument(
        "--crop_vis_dir",
        type=str,
        default="hrnet_crops",
        help="Directory to write per-person crop visualizations.",
    )
    parser.add_argument(
        "--skeleton_vis_dir",
        type=str,
        default="hrnet_pie_skeletons",
        help="Directory to write skeleton-only visualizations.",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Skip CSVs that already exist.")
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_images_root(image_root: str) -> Path:
    base = Path(image_root)
    images_dir = base / "images"
    if images_dir.is_dir():
        return images_dir
    return base


def find_frame_image(images_root: Path, set_id: str, vid_id: str, frame_id: int) -> Path:
    frame_name = f"{frame_id:05d}"
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = images_root / set_id / vid_id / f"{frame_name}{ext}"
        if candidate.exists():
            return candidate
    return images_root / set_id / vid_id / f"{frame_name}.png"


def load_hrnet(args: argparse.Namespace):
    repo_root = resolve_repo_root()
    simple_hrnet_root = repo_root / "simple-HRNet"
    sys.path.insert(0, str(simple_hrnet_root))

    from SimpleHRNet import SimpleHRNet  # noqa: WPS433

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = repo_root / checkpoint_path

    return SimpleHRNet(
        args.c,
        args.joints,
        str(checkpoint_path),
        resolution=args.resolution,
        multiperson=False,
        device=torch.device(args.device),
    )


def clamp_bbox(bbox, width, height, expand: float = 1.0):
    x1, y1, x2, y2 = bbox
    if expand and expand != 1.0:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = (x2 - x1) * expand
        h = (y2 - y1) * expand
        x1 = cx - w / 2.0
        x2 = cx + w / 2.0
        y1 = cy - h / 2.0
        y2 = cy + h / 2.0
    x1 = max(0, min(int(np.floor(x1)), width - 1))
    y1 = max(0, min(int(np.floor(y1)), height - 1))
    x2 = max(0, min(int(np.ceil(x2)), width))
    y2 = max(0, min(int(np.ceil(y2)), height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def predict_pose_for_bbox(model, image, bbox, expand: float = 1.0):
    height, width = image.shape[:2]
    clamped = clamp_bbox(bbox, width, height, expand=expand)
    if clamped is None:
        return None
    x1, y1, x2, y2 = clamped
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    joints = model.predict(crop)
    if joints is None or len(joints) == 0:
        return None
    if joints.ndim == 2:
        joints = joints[None, ...]
    joints = joints[0]
    joints[:, 0] += y1
    joints[:, 1] += x1
    return joints


def to_xyc(joints):
    if joints is None:
        return None
    return np.stack([joints[:, 1], joints[:, 0], joints[:, 2]], axis=1)


def load_bboxes_from_file(bbox_path):
    bboxes = []
    with open(bbox_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = [p.strip() for p in stripped.split(",")]
            if len(parts) != 4:
                raise ValueError(f"Invalid bbox line: {line}")
            bboxes.append([float(p) for p in parts])
    return bboxes


def get_bboxes_from_pie(vid_data, frame_id):
    bboxes = []
    for _, ped_data in vid_data["ped_annotations"].items():
        frames = ped_data["frames"]
        if frame_id in frames:
            idx = frames.index(frame_id)
            bboxes.append(ped_data["bbox"][idx])
    return bboxes


def simple_test(args, model):
    images_root = resolve_images_root(args.image_root)
    if args.test_image:
        image_path = Path(args.test_image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not args.test_bboxes:
            raise ValueError("--test_bboxes is required when --test_image is used.")
        bboxes = load_bboxes_from_file(args.test_bboxes)
    else:
        if not args.test_video or args.test_frame < 0:
            raise ValueError("For dataset test, set --test_video and --test_frame.")
        pie = pie_data.PIE(data_path=args.data_path)
        vid_data = pie._get_annotations(args.test_set, args.test_video)
        bboxes = get_bboxes_from_pie(vid_data, args.test_frame)
        image_path = find_frame_image(images_root, args.test_set, args.test_video, args.test_frame)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    poses = []
    for bbox in bboxes:
        pose = predict_pose_for_bbox(model, image, bbox, expand=args.bbox_expand)
        poses.append(pose)

    print(f"Found {len(bboxes)} bboxes, {sum(p is not None for p in poses)} poses")
    for idx, pose in enumerate(poses):
        if pose is not None:
            print(f"person[{idx}] joints shape: {pose.shape}")

    if args.save_vis or args.save_crop_vis or args.save_skeleton_only:
        repo_root = resolve_repo_root()
        simple_hrnet_root = repo_root / "simple-HRNet"
        sys.path.insert(0, str(simple_hrnet_root))
        from misc.visualization import draw_points_and_skeleton, joints_dict  # noqa: WPS433

        joints_meta = joints_dict()
        if args.joints == 17:
            skeleton = joints_meta["coco"]["skeleton"]
        elif args.joints == 16:
            skeleton = joints_meta["mpii"]["skeleton"]
        else:
            skeleton = []

        if args.save_vis:
            vis_image = image.copy()
            for person_index, pose in enumerate(poses):
                if pose is None:
                    continue
                vis_image = draw_points_and_skeleton(
                    vis_image,
                    pose,
                    skeleton,
                    person_index=person_index,
                    confidence_threshold=args.vis_conf,
                )
            for bbox in bboxes:
                clamped = clamp_bbox(bbox, image.shape[1], image.shape[0], expand=args.bbox_expand)
                if clamped is None:
                    continue
                x1, y1, x2, y2 = clamped
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            out_path = Path(args.vis_output)
            if not out_path.is_absolute():
                out_path = resolve_repo_root() / out_path
            ensure_dir(str(out_path.parent))
            if not cv2.imwrite(str(out_path), vis_image):
                raise RuntimeError(f"Failed to write visualization to: {out_path}")
            print(f"Saved visualization to: {out_path}")

        if args.save_crop_vis:
            crop_dir = Path(args.crop_vis_dir)
            if not crop_dir.is_absolute():
                crop_dir = repo_root / crop_dir
            ensure_dir(str(crop_dir))

        if args.save_skeleton_only:
            skeleton_dir = Path(args.skeleton_vis_dir)
            if not skeleton_dir.is_absolute():
                skeleton_dir = repo_root / skeleton_dir
            ensure_dir(str(skeleton_dir))

            if args.test_image:
                base_name = Path(args.test_image).stem
            else:
                base_name = f"{args.test_set}_{args.test_video}_{args.test_frame:05d}"

            for person_index, pose in enumerate(poses):
                if pose is None:
                    continue
                clamped = clamp_bbox(bboxes[person_index], image.shape[1], image.shape[0], expand=args.bbox_expand)
                if clamped is None:
                    continue
                x1, y1, x2, y2 = clamped
                crop = image[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue
                crop_pose = pose.copy()
                crop_pose[:, 0] -= y1
                crop_pose[:, 1] -= x1
                crop_vis = draw_points_and_skeleton(
                    crop,
                    crop_pose,
                    skeleton,
                    person_index=person_index,
                    confidence_threshold=args.vis_conf,
                )
                out_path = crop_dir / f"{base_name}_person{person_index:02d}.jpg"
                if not cv2.imwrite(str(out_path), crop_vis):
                    raise RuntimeError(f"Failed to write crop visualization to: {out_path}")

                if args.save_skeleton_only:
                    blank = np.zeros_like(crop)
                    skeleton_vis = draw_points_and_skeleton(
                        blank,
                        crop_pose,
                        skeleton,
                        person_index=person_index,
                        confidence_threshold=args.vis_conf,
                    )
                    skel_path = skeleton_dir / f"{base_name}_person{person_index:02d}.jpg"
                    if not cv2.imwrite(str(skel_path), skeleton_vis):
                        raise RuntimeError(f"Failed to write skeleton visualization to: {skel_path}")


def process_dataset(args, model):
    images_root = resolve_images_root(args.image_root)
    pie = pie_data.PIE(data_path=args.data_path)
    dataset = pie.generate_database()

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = Path(args.data_path) / output_root
    ensure_dir(str(output_root))
    for split in ["train", "val", "test"]:
        ensure_dir(str(output_root / split))

    train_sets = ["set02", "set04", "set05"]
    val_sets = ["set06"]
    test_sets = ["set03"]

    pose_set_ids = []
    pose_video_ids = []
    pose_ped_ids = []
    pose_frames = []
    pose_bboxes = []
    pose_keypoints = []
    ped_ids = []
    ped_ptr = [0]

    ped_id = 1
    for set_id in dataset:
        print("Processing", set_id, "...")
        vids = dataset[set_id]
        for vid_id in vids:
            print("Processing", vid_id, "...")
            vi_seq = vids[vid_id]
            for ped in vi_seq["ped_annotations"]:
                if not vi_seq["ped_annotations"][ped]["behavior"]:
                    continue
                output_path = None
                if set_id in train_sets:
                    output_path = output_root / "train" / f"{ped_id:05d}.csv"
                elif set_id in val_sets:
                    output_path = output_root / "val" / f"{ped_id:05d}.csv"
                elif set_id in test_sets:
                    output_path = output_root / "test" / f"{ped_id:05d}.csv"
                if output_path is None:
                    ped_id += 1
                    continue
                write_csv = True
                if args.skip_existing and output_path.exists():
                    write_csv = False

                ped_ids.append(ped_id)

                frames = np.array(vi_seq["ped_annotations"][ped]["frames"]).reshape(-1, 1)
                ids = np.repeat(ped_id, frames.shape[0]).reshape(-1, 1)
                bbox = np.array(vi_seq["ped_annotations"][ped]["bbox"])
                x = bbox[:, 0].reshape(-1, 1)
                y = bbox[:, 1].reshape(-1, 1)
                w = np.abs(bbox[:, 0] - bbox[:, 2]).reshape(-1, 1)
                h = np.abs(bbox[:, 1] - bbox[:, 3]).reshape(-1, 1)
                imagefolderpath = np.repeat(
                    os.path.join(str(images_root), set_id, vid_id),
                    frames.shape[0],
                ).reshape(-1, 1)

                converted_cross = [0 if v == -1 else v for v in vi_seq["ped_annotations"][ped]["behavior"]["cross"]]
                cross = np.array(converted_cross).reshape(-1, 1)
                look = np.array(vi_seq["ped_annotations"][ped]["behavior"]["look"]).reshape(-1, 1)
                action = np.array(vi_seq["ped_annotations"][ped]["behavior"]["action"]).reshape(-1, 1)
                gesture = np.array(vi_seq["ped_annotations"][ped]["behavior"]["gesture"]).reshape(-1, 1)

                age = np.repeat(vi_seq["ped_annotations"][ped]["attributes"]["age"], frames.shape[0]).reshape(-1, 1)
                gender = np.repeat(vi_seq["ped_annotations"][ped]["attributes"]["gender"], frames.shape[0]).reshape(-1, 1)
                intersection = np.repeat(
                    vi_seq["ped_annotations"][ped]["attributes"]["intersection"],
                    frames.shape[0],
                ).reshape(-1, 1)
                traffic_direction = np.repeat(
                    vi_seq["ped_annotations"][ped]["attributes"]["traffic_direction"],
                    frames.shape[0],
                ).reshape(-1, 1)
                num_lanes = np.repeat(
                    vi_seq["ped_annotations"][ped]["attributes"]["num_lanes"],
                    frames.shape[0],
                ).reshape(-1, 1)
                signalized = np.repeat(
                    vi_seq["ped_annotations"][ped]["attributes"]["signalized"],
                    frames.shape[0],
                ).reshape(-1, 1)
                intention_prob = np.repeat(
                    vi_seq["ped_annotations"][ped]["attributes"]["intention_prob"],
                    frames.shape[0],
                ).reshape(-1, 1)

                accelx, accely, accelz, odb_speed, heading_angle = [], [], [], [], []
                for fr in range(len(frames)):
                    frame_id = frames[fr][0]
                    accelx.append(vi_seq["vehicle_annotations"][frame_id]["accX"])
                    accely.append(vi_seq["vehicle_annotations"][frame_id]["accY"])
                    accelz.append(vi_seq["vehicle_annotations"][frame_id]["accZ"])
                    odb_speed.append(vi_seq["vehicle_annotations"][frame_id]["OBD_speed"])
                    heading_angle.append(vi_seq["vehicle_annotations"][frame_id]["heading_angle"])

                accx = np.array(accelx).reshape(-1, 1)
                accy = np.array(accely).reshape(-1, 1)
                accz = np.array(accelz).reshape(-1, 1)
                o_speed = np.array(odb_speed).reshape(-1, 1)
                h_angle = np.array(heading_angle).reshape(-1, 1)

                for i, frame_id in enumerate(frames.reshape(-1)):
                    image_path = os.path.join(
                        str(images_root),
                        set_id,
                        vid_id,
                        f"{int(frame_id):05d}.png",
                    )
                    image_path = str(find_frame_image(images_root, set_id, vid_id, int(frame_id)))
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    if image is None:
                        pose_set_ids.append(set_id)
                        pose_video_ids.append(vid_id)
                        pose_ped_ids.append(ped_id)
                        pose_frames.append(int(frame_id))
                        pose_bboxes.append(bbox[i])
                        pose_keypoints.append(np.full((args.joints, 3), np.nan, dtype=np.float32))
                        continue
                    pose = predict_pose_for_bbox(model, image, bbox[i], expand=args.bbox_expand)
                    pose_xyc = to_xyc(pose)
                    pose_set_ids.append(set_id)
                    pose_video_ids.append(vid_id)
                    pose_ped_ids.append(ped_id)
                    pose_frames.append(int(frame_id))
                    pose_bboxes.append(bbox[i])
                    if pose_xyc is None:
                        pose_keypoints.append(np.full((args.joints, 3), np.nan, dtype=np.float32))
                    else:
                        pose_keypoints.append(pose_xyc.astype(np.float32))

                ped_ptr.append(len(pose_frames))
                ped_data = np.hstack(
                    (
                        frames,
                        ids,
                        x,
                        y,
                        w,
                        h,
                        imagefolderpath,
                        cross,
                        look,
                        action,
                        gesture,
                        age,
                        gender,
                        intersection,
                        traffic_direction,
                        num_lanes,
                        signalized,
                        intention_prob,
                        accx,
                        accy,
                        accz,
                        o_speed,
                        h_angle,
                    )
                )
                data_to_write = pd.DataFrame(
                    {
                        "frame": ped_data[:, 0].reshape(-1),
                        "ID": ped_data[:, 1].reshape(-1),
                        "x": ped_data[:, 2].reshape(-1),
                        "y": ped_data[:, 3].reshape(-1),
                        "w": ped_data[:, 4].reshape(-1),
                        "h": ped_data[:, 5].reshape(-1),
                        "imagefolderpath": ped_data[:, 6].reshape(-1),
                        "crossing_true": ped_data[:, 7].reshape(-1),
                        "look": ped_data[:, 8].reshape(-1),
                        "action": ped_data[:, 9].reshape(-1),
                        "gesture": ped_data[:, 10].reshape(-1),
                        "age": ped_data[:, 11].reshape(-1),
                        "gender": ped_data[:, 12].reshape(-1),
                        "intersection": ped_data[:, 13].reshape(-1),
                        "traffic_direction": ped_data[:, 14].reshape(-1),
                        "num_lanes": ped_data[:, 15].reshape(-1),
                        "signalized": ped_data[:, 16].reshape(-1),
                        "intention_prob": ped_data[:, 17].reshape(-1),
                        "accx": ped_data[:, 18].reshape(-1),
                        "accy": ped_data[:, 19].reshape(-1),
                        "accz": ped_data[:, 20].reshape(-1),
                        "o_speed": ped_data[:, 21].reshape(-1),
                        "h_angle": ped_data[:, 22].reshape(-1),
                    }
                )
                data_to_write["filename"] = data_to_write.frame
                data_to_write.filename = data_to_write.filename.apply(lambda v: f"{int(v):05d}.png")

                if write_csv:
                    data_to_write.to_csv(output_path, index=False)
                ped_id += 1

    pose_npz_path = Path(args.pose_npz)
    if not pose_npz_path.is_absolute():
        pose_npz_path = output_root / pose_npz_path
    ensure_dir(str(pose_npz_path.parent))
    if pose_keypoints:
        np.savez_compressed(
            pose_npz_path,
            ped_ids=np.array(ped_ids, dtype=np.int32),
            ped_ptr=np.array(ped_ptr, dtype=np.int32),
            set_id=np.array(pose_set_ids, dtype=object),
            video_id=np.array(pose_video_ids, dtype=object),
            frame=np.array(pose_frames, dtype=np.int32),
            bbox=np.array(pose_bboxes, dtype=np.float32),
            keypoints=np.stack(pose_keypoints, axis=0),
        )
        print(f"Saved pose annotations to: {pose_npz_path}")
    else:
        print("No pose annotations collected; pose npz not written.")


def main():
    args = parse_args()
    if not args.simple_test and not args.process_dataset:
        raise ValueError("Set --simple_test or --process_dataset.")
    model = load_hrnet(args)
    if args.simple_test:
        simple_test(args, model)
    if args.process_dataset:
        process_dataset(args, model)


if __name__ == "__main__":
    main()

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

import jaad_data

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils  # noqa: E402


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
        "JAAD_root",
        "JAAD_preproc_root",
        "hrnet_checkpoint",
        "hrnet_c",
        "hrnet_joints",
        "hrnet_resolution",
        "hrnet_device",
    ]
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(f"Missing required config keys: {', '.join(missing)}")

    data_path_default = config["JAAD_root"]
    image_root_default = config["JAAD_preproc_root"]
    output_dir_default = str(Path(config["JAAD_root"]) / "PN_ego")
    pose_npz_default = "jaad_pose_annotations.npz"
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
        help="Path to JAAD dataset root (annotations/, images/).",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=image_root_default,
        help="Root directory that contains JAAD images (video_id/frame).",
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
    parser.add_argument("--process_dataset", action="store_true", help="Process the full JAAD dataset.")
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
    parser.add_argument("--test_video", type=str, default="", help="JAAD video id for simple test.")
    parser.add_argument("--test_frame", type=int, default=-1, help="Frame id for simple test.")
    parser.add_argument("--save_vis", action="store_true", help="Save visualization image.")
    parser.add_argument("--save_crop_only", action="store_true", help="Save per-person cropped images.")
    parser.add_argument("--save_skeleton_only", action="store_true", help="Save skeleton-only images per person.")
    parser.add_argument("--vis_output", type=str, default="hrnet_jaad_vis.jpg", help="Vis output path.")
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
        help="Directory to write per-person crops and pose-only images.",
    )
    parser.add_argument(
        "--split_ids_subset",
        type=str,
        default="default",
        help="JAAD split_ids subset name (e.g. default, high_visibility, all_videos).",
    )
    parser.add_argument(
        "--no_split_ids",
        action="store_false",
        dest="use_split_ids",
        default=True,
        help="Disable JAAD official split_ids and use ratios instead.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train video ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Val video ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test video ratio.")
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


def find_frame_image(images_root: Path, vid_id: str, frame_id: int) -> Path:
    frame_name = f"{frame_id + 1:05d}"
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = images_root / vid_id / f"{frame_name}{ext}"
        if candidate.exists():
            return candidate
    return images_root / vid_id / f"{frame_name}.png"


def load_bboxes_from_file(bbox_path: str):
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
        jaad = jaad_data.JAAD(data_path=args.data_path)
        vid_data = jaad._get_annotations(args.test_video)
        bboxes = []
        for _, ped_data in vid_data["ped_annotations"].items():
            frames = ped_data["frames"]
            if args.test_frame in frames:
                idx = frames.index(args.test_frame)
                bboxes.append(ped_data["bbox"][idx])
        image_path = find_frame_image(images_root, args.test_video, args.test_frame)
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

    if args.save_vis or args.save_crop_only or args.save_skeleton_only:
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

        if args.save_crop_only or args.save_skeleton_only:
            crop_dir = Path(args.crop_vis_dir)
            if not crop_dir.is_absolute():
                crop_dir = repo_root / crop_dir
            ensure_dir(str(crop_dir))

            if args.test_image:
                base_name = Path(args.test_image).stem
            else:
                base_name = f"{args.test_video}_{args.test_frame:05d}"

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
                if args.save_crop_only:
                    out_path = crop_dir / f"{base_name}_person{person_index:02d}_crop.jpg"
                    if not cv2.imwrite(str(out_path), crop):
                        raise RuntimeError(f"Failed to write crop image to: {out_path}")

                if args.save_skeleton_only:
                    blank = np.zeros_like(crop)
                    skeleton_vis = draw_points_and_skeleton(
                        blank,
                        crop_pose,
                        skeleton,
                        person_index=person_index,
                        confidence_threshold=args.vis_conf,
                    )
                    skel_path = crop_dir / f"{base_name}_person{person_index:02d}_pose.jpg"
                    if not cv2.imwrite(str(skel_path), skeleton_vis):
                        raise RuntimeError(f"Failed to write skeleton visualization to: {skel_path}")


def split_videos(videos, train_ratio: float, val_ratio: float, test_ratio: float):
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise ValueError("Sum of split ratios must be > 0.")
    if total_ratio > 1.0:
        raise ValueError("Sum of split ratios must be <= 1.0.")
    total = len(videos)
    n_train = int(train_ratio * total)
    n_val = int(val_ratio * total)
    train_videos = videos[:n_train]
    val_videos = videos[n_train:n_train + n_val]
    test_videos = videos[n_train + n_val:] if test_ratio > 0 else []
    return set(train_videos), set(val_videos), set(test_videos)


def load_split_ids(data_path: str, subset: str):
    split_root = Path(data_path) / "split_ids" / subset
    train_path = split_root / "train.txt"
    val_path = split_root / "val.txt"
    test_path = split_root / "test.txt"
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        return None
    train_ids = train_path.read_text().splitlines()
    val_ids = val_path.read_text().splitlines()
    test_ids = test_path.read_text().splitlines()
    return set(train_ids), set(val_ids), set(test_ids)


def build_sequence_npz(
    split_dir: Path,
    output_root: Path,
    split_name: str,
    input_len: int,
    output_len: int,
    stride: int,
):
    print(f"Building sequence npz for {split_name}...")
    df = pd.DataFrame()
    new_index = 0
    for file_path in sorted(split_dir.glob("*.csv")):
        temp = pd.read_csv(file_path)
        if temp.empty:
            continue
        video_id = file_path.stem
        temp["video_id"] = video_id
        temp["ped_id"] = temp["ID"]
        for index in temp.ID.unique():
            new_index += 1
            temp.ID = temp.ID.replace(index, f"{video_id}_{index}_{new_index}")
        temp = temp.sort_values(["ID", "frame"], axis=0)
        df = pd.concat((df, temp), ignore_index=True)

    if df.empty:
        print(f"No data found for split {split_name}.")
        return

    df.insert(0, "sequence", df.ID)
    df = df.apply(lambda row: utils.compute_center(row), axis=1)
    df = df.reset_index(drop=True)

    df["bounding_box"] = df[["x", "y", "w", "h"]].apply(
        lambda row: [row.x, row.y, row.w, row.h],
        axis=1,
    )
    df["ped_attribute"] = df[["age", "gender", "group_size"]].apply(
        lambda row: [row.age, row.gender, row.group_size],
        axis=1,
    )
    df["ped_behavior"] = df[["reaction", "hand_gesture", "look", "nod"]].apply(
        lambda row: [row.reaction, row.hand_gesture, row.look, row.nod],
        axis=1,
    )
    df["scene_attribute"] = df[
        [
            "designated",
            "motion_direction",
            "num_lanes",
            "signalized",
            "traffic_direction",
            "ped_crossing",
            "ped_sign",
            "stop_sign",
            "traffic_light",
            "road_type",
        ]
    ].apply(
        lambda row: [
            row.designated,
            row.motion_direction,
            row.num_lanes,
            row.signalized,
            row.traffic_direction,
            row.ped_crossing,
            row.ped_sign,
            row.stop_sign,
            row.traffic_light,
            row.road_type,
        ],
        axis=1,
    )

    bb = df.groupby(["ID"])["bounding_box"].apply(list).reset_index(name="bounding_box")
    s = df.groupby(["ID"])["imagefolderpath"].apply(list).reset_index(name="imagefolderpath").drop(columns="ID")
    f = df.groupby(["ID"])["filename"].apply(list).reset_index(name="filename").drop(columns="ID")
    c = df.groupby(["ID"])["crossing_true"].apply(list).reset_index(name="crossing_true").drop(columns="ID")
    t = df.groupby(["ID"])["ped_attribute"].apply(list).reset_index(name="ped_attribute").drop(columns="ID")
    h = df.groupby(["ID"])["ped_behavior"].apply(list).reset_index(name="ped_behavior").drop(columns="ID")
    w = df.groupby(["ID"])["scene_attribute"].apply(list).reset_index(name="scene_attribute").drop(columns="ID")
    fr = df.groupby(["ID"])["frame"].apply(list).reset_index(name="frame").drop(columns="ID")
    pid = df.groupby(["ID"])["ped_id"].apply(list).reset_index(name="ped_id").drop(columns="ID")
    vid = df.groupby(["ID"])["video_id"].apply(list).reset_index(name="video_id").drop(columns="ID")
    d = bb.join(s).join(f).join(c).join(t).join(h).join(w).join(fr).join(pid).join(vid)

    d["label"] = d["crossing_true"]
    d.label = d.label.apply(lambda x: 1 if 1 in x else 0)
    d = d.drop(d[d.bounding_box.apply(lambda x: len(x) < input_len + output_len)].index)
    d = d.reset_index(drop=True)

    bounding_box_o = np.empty((0, input_len, 4))
    bounding_box_t = np.empty((0, output_len, 4))
    scene_o = np.empty((0, input_len))
    file = np.empty((0, input_len))
    cross_o = np.empty((0, input_len))
    cross = np.empty((0, output_len))
    ind = np.empty((0, 1), dtype=object)
    p_attribute = np.empty((0, 3))
    p_behavior = np.empty((0, input_len, 4))
    s_attribute = np.empty((0, input_len, 10))
    ped_ids = np.empty((0, 1), dtype=object)
    video_ids = np.empty((0, 1), dtype=object)
    frame_obs = np.empty((0, input_len))

    for i in range(d.shape[0]):
        ped = d.loc[i]
        k = 0
        while (k + input_len + output_len) <= len(ped.bounding_box):
            ind = np.vstack((ind, ped["ID"]))
            p_attribute = np.vstack((p_attribute, ped["ped_attribute"][0]))
            bounding_box_o = np.vstack(
                (bounding_box_o, np.array(ped.bounding_box[k : k + input_len]).reshape(1, input_len, 4))
            )
            bounding_box_t = np.vstack(
                (bounding_box_t, np.array(ped.bounding_box[k + input_len : k + input_len + output_len]).reshape(1, output_len, 4))
            )
            scene_o = np.vstack((scene_o, np.array(ped.imagefolderpath[k : k + input_len]).reshape(1, input_len)))
            p_behavior = np.vstack((p_behavior, np.array(ped.ped_behavior[k : k + input_len]).reshape(1, input_len, 4)))
            s_attribute = np.vstack((s_attribute, np.array(ped.scene_attribute[k : k + input_len]).reshape(1, input_len, 10)))
            file = np.vstack((file, np.array(ped.filename[k : k + input_len]).reshape(1, input_len)))
            cross_o = np.vstack((cross_o, np.array(ped.crossing_true[k : k + input_len]).reshape(1, input_len)))
            cross = np.vstack((cross, np.array(ped.crossing_true[k + input_len : k + input_len + output_len]).reshape(1, output_len)))
            frame_obs = np.vstack((frame_obs, np.array(ped.frame[k : k + input_len]).reshape(1, input_len)))
            ped_ids = np.vstack((ped_ids, ped.ped_id[0]))
            video_ids = np.vstack((video_ids, ped.video_id[0]))
            k += stride

    data = pd.DataFrame(
        {
            "ID": ind.reshape(-1),
            "ped_id": ped_ids.reshape(-1),
            "video_id": video_ids.reshape(-1),
            "frame_obs": frame_obs.reshape(-1, input_len).tolist(),
            "ped_attribute": p_attribute.reshape(-1, 3).tolist(),
            "bounding_box": bounding_box_o.reshape(-1, 1, input_len, 4).tolist(),
            "future_bounding_box": bounding_box_t.reshape(-1, 1, output_len, 4).tolist(),
            "ped_behavior": p_behavior.reshape(-1, 1, input_len, 4).tolist(),
            "scene_attribute": s_attribute.reshape(-1, 1, input_len, 10).tolist(),
            "imagefolderpath": scene_o.reshape(-1, input_len).tolist(),
            "filename": file.reshape(-1, input_len).tolist(),
            "crossing_obs": cross_o.reshape(-1, input_len).tolist(),
            "crossing_true": cross.reshape(-1, output_len).tolist(),
        }
    )
    data.bounding_box = data.bounding_box.apply(lambda x: x[0])
    data.future_bounding_box = data.future_bounding_box.apply(lambda x: x[0])
    data.ped_behavior = data.ped_behavior.apply(lambda x: x[0])
    data.scene_attribute = data.scene_attribute.apply(lambda x: x[0])

    data = data.drop(data[data.crossing_obs.apply(lambda x: 1.0 in x)].index)
    data["label"] = data.crossing_true.apply(lambda x: 1.0 if 1.0 in x else 0.0)
    data = data.reset_index(drop=True)

    npz_dir = output_root / "npz"
    ensure_dir(str(npz_dir))
    npz_path = npz_dir / f"jaad_{split_name}_{input_len}_{output_len}_{stride}.npz"
    np.savez_compressed(
        npz_path,
        ID=data["ID"].to_numpy(dtype=object),
        ped_id=data["ped_id"].to_numpy(dtype=object),
        video_id=data["video_id"].to_numpy(dtype=object),
        frame_obs=np.array(data["frame_obs"].tolist(), dtype=np.int32),
        ped_attribute=np.array(data["ped_attribute"].tolist(), dtype=np.float32),
        bounding_box=np.array(data["bounding_box"].tolist(), dtype=np.float32),
        future_bounding_box=np.array(data["future_bounding_box"].tolist(), dtype=np.float32),
        ped_behavior=np.array(data["ped_behavior"].tolist(), dtype=np.float32),
        scene_attribute=np.array(data["scene_attribute"].tolist(), dtype=np.float32),
        imagefolderpath=np.array(data["imagefolderpath"].tolist(), dtype=object),
        filename=np.array(data["filename"].tolist(), dtype=object),
        crossing_obs=np.array(data["crossing_obs"].tolist(), dtype=np.float32),
        crossing_true=np.array(data["crossing_true"].tolist(), dtype=np.float32),
        label=np.array(data["label"].tolist(), dtype=np.float32),
    )
    print(f"Saved sequence npz to: {npz_path}")


def process_dataset(args, model):
    images_root = resolve_images_root(args.image_root)
    jaad = jaad_data.JAAD(data_path=args.data_path)
    dataset = jaad.generate_database()

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = Path(args.data_path) / output_root
    ensure_dir(str(output_root))
    for split in ["train", "val", "test"]:
        ensure_dir(str(output_root / split))

    videos = list(dataset.keys())
    split_ids = None
    if args.use_split_ids:
        split_ids = load_split_ids(args.data_path, args.split_ids_subset)
        if split_ids is None:
            raise FileNotFoundError(
                f"split_ids not found for subset '{args.split_ids_subset}' under {args.data_path}"
            )

    if split_ids is None:
        train_videos, val_videos, test_videos = split_videos(
            videos,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )
    else:
        train_videos, val_videos, test_videos = split_ids

    pose_video_ids = []
    pose_ped_ids = []
    pose_frames = []
    pose_bboxes = []
    pose_keypoints = []
    ped_ids = []
    ped_ptr = [0]

    for video in dataset:
        print("Processing", video, "...")
        vid = dataset[video]
        data_rows = []
        output_path = None
        if video in train_videos:
            output_path = output_root / "train" / f"{video}.csv"
        elif video in val_videos:
            output_path = output_root / "val" / f"{video}.csv"
        elif video in test_videos:
            output_path = output_root / "test" / f"{video}.csv"
        if output_path is None:
            continue

        write_csv = True
        if args.skip_existing and output_path.exists():
            write_csv = False

        for ped in vid["ped_annotations"]:
            if not vid["ped_annotations"][ped]["behavior"]:
                continue

            frames = np.array(vid["ped_annotations"][ped]["frames"]).reshape(-1, 1)
            ids = np.repeat(vid["ped_annotations"][ped]["old_id"], frames.shape[0]).reshape(-1, 1)
            bbox = np.array(vid["ped_annotations"][ped]["bbox"])
            x = bbox[:, 0].reshape(-1, 1)
            y = bbox[:, 1].reshape(-1, 1)
            w = np.abs(bbox[:, 0] - bbox[:, 2]).reshape(-1, 1)
            h = np.abs(bbox[:, 1] - bbox[:, 3]).reshape(-1, 1)
            imagefolderpath = np.array(
                [
                    str(find_frame_image(images_root, video, int(frames[fr][0])))
                    for fr in range(0, frames.shape[0])
                ]
            ).reshape(-1, 1)

            cross = np.array(vid["ped_annotations"][ped]["behavior"]["cross"]).reshape(-1, 1)
            reaction = np.array(vid["ped_annotations"][ped]["behavior"]["reaction"]).reshape(-1, 1)
            hand_gesture = np.array(vid["ped_annotations"][ped]["behavior"]["hand_gesture"]).reshape(-1, 1)
            look = np.array(vid["ped_annotations"][ped]["behavior"]["look"]).reshape(-1, 1)
            nod = np.array(vid["ped_annotations"][ped]["behavior"]["nod"]).reshape(-1, 1)

            age = np.repeat(vid["ped_annotations"][ped]["attributes"]["age"], frames.shape[0]).reshape(-1, 1)
            gender = np.repeat(vid["ped_annotations"][ped]["attributes"]["gender"], frames.shape[0]).reshape(-1, 1)
            group_size = np.repeat(vid["ped_annotations"][ped]["attributes"]["group_size"], frames.shape[0]).reshape(-1, 1)

            designated = np.repeat(vid["ped_annotations"][ped]["attributes"]["designated"], frames.shape[0]).reshape(-1, 1)
            motion_direction = np.repeat(
                vid["ped_annotations"][ped]["attributes"]["motion_direction"],
                frames.shape[0],
            ).reshape(-1, 1)
            num_lanes = np.repeat(vid["ped_annotations"][ped]["attributes"]["num_lanes"], frames.shape[0]).reshape(-1, 1)
            signalized = np.repeat(vid["ped_annotations"][ped]["attributes"]["signalized"], frames.shape[0]).reshape(-1, 1)
            traffic_direction = np.repeat(
                vid["ped_annotations"][ped]["attributes"]["traffic_direction"],
                frames.shape[0],
            ).reshape(-1, 1)
            road_type = np.repeat(vid["traffic_annotations"]["road_type"], frames.shape[0]).reshape(-1, 1)

            ped_crossing, ped_sign, stop_sign, traffic_light = [], [], [], []
            for f in frames:
                ped_crossing.append(vid["traffic_annotations"][f[0]]["ped_crossing"])
                ped_sign.append(vid["traffic_annotations"][f[0]]["ped_sign"])
                stop_sign.append(vid["traffic_annotations"][f[0]]["stop_sign"])
                traffic_light.append(vid["traffic_annotations"][f[0]]["traffic_light"])

            ped_crossing = np.array(ped_crossing).reshape(-1, 1)
            ped_sign = np.array(ped_sign).reshape(-1, 1)
            stop_sign = np.array(stop_sign).reshape(-1, 1)
            traffic_light = np.array(traffic_light).reshape(-1, 1)

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
                    reaction,
                    hand_gesture,
                    look,
                    nod,
                    age,
                    gender,
                    group_size,
                    designated,
                    motion_direction,
                    num_lanes,
                    signalized,
                    traffic_direction,
                    ped_crossing,
                    ped_sign,
                    stop_sign,
                    traffic_light,
                    road_type,
                )
            )

            data_rows.append(ped_data)

            ped_ids.append(vid["ped_annotations"][ped]["old_id"])
            for i, frame_id in enumerate(frames.reshape(-1)):
                image_path = str(find_frame_image(images_root, video, int(frame_id)))
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                pose_video_ids.append(video)
                pose_ped_ids.append(vid["ped_annotations"][ped]["old_id"])
                pose_frames.append(int(frame_id))
                pose_bboxes.append(bbox[i])
                if image is None:
                    pose_keypoints.append(np.full((args.joints, 3), np.nan, dtype=np.float32))
                    continue
                pose = predict_pose_for_bbox(model, image, bbox[i], expand=args.bbox_expand)
                pose_xyc = to_xyc(pose)
                if pose_xyc is None:
                    pose_keypoints.append(np.full((args.joints, 3), np.nan, dtype=np.float32))
                else:
                    pose_keypoints.append(pose_xyc.astype(np.float32))
            ped_ptr.append(len(pose_frames))

        if write_csv and data_rows:
            data = np.vstack(data_rows)
            data_to_write = pd.DataFrame(
                {
                    "frame": data[:, 0].reshape(-1),
                    "ID": data[:, 1].reshape(-1),
                    "x": data[:, 2].reshape(-1),
                    "y": data[:, 3].reshape(-1),
                    "w": data[:, 4].reshape(-1),
                    "h": data[:, 5].reshape(-1),
                    "imagefolderpath": data[:, 6].reshape(-1),
                    "crossing_true": data[:, 7].reshape(-1),
                    "reaction": data[:, 8].reshape(-1),
                    "hand_gesture": data[:, 9].reshape(-1),
                    "look": data[:, 10].reshape(-1),
                    "nod": data[:, 11].reshape(-1),
                    "age": data[:, 12].reshape(-1),
                    "gender": data[:, 13].reshape(-1),
                    "group_size": data[:, 14].reshape(-1),
                    "designated": data[:, 15].reshape(-1),
                    "motion_direction": data[:, 16].reshape(-1),
                    "num_lanes": data[:, 17].reshape(-1),
                    "signalized": data[:, 18].reshape(-1),
                    "traffic_direction": data[:, 19].reshape(-1),
                    "ped_crossing": data[:, 20].reshape(-1),
                    "ped_sign": data[:, 21].reshape(-1),
                    "stop_sign": data[:, 22].reshape(-1),
                    "traffic_light": data[:, 23].reshape(-1),
                    "road_type": data[:, 24].reshape(-1),
                }
            )
            data_to_write["filename"] = data_to_write.frame
            data_to_write.filename = data_to_write.filename.apply(lambda v: f"{int(v) + 1:05d}.png")
            data_to_write.to_csv(output_path, index=False)

    pose_npz_path = Path(args.pose_npz)
    if not pose_npz_path.is_absolute():
        pose_npz_path = output_root / pose_npz_path
    ensure_dir(str(pose_npz_path.parent))
    if pose_keypoints:
        np.savez_compressed(
            pose_npz_path,
            ped_ids=np.array(ped_ids, dtype=object),
            ped_ptr=np.array(ped_ptr, dtype=np.int32),
            video_id=np.array(pose_video_ids, dtype=object),
            frame=np.array(pose_frames, dtype=np.int32),
            bbox=np.array(pose_bboxes, dtype=np.float32),
            keypoints=np.stack(pose_keypoints, axis=0),
        )
        print(f"Saved pose annotations to: {pose_npz_path}")
    else:
        print("No pose annotations collected; pose npz not written.")

    for split in ["train", "val", "test"]:
        split_dir = output_root / split
        if split_dir.is_dir():
            build_sequence_npz(
                split_dir,
                output_root,
                split,
                args.input,
                args.output,
                args.stride,
            )


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

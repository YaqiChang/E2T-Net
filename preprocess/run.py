from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
from typing import List, Tuple

import yaml

from .core.io import list_images
from .core.manifest import ManifestStore
from .core.pipeline import Pipeline, PipelineContext
from .core.registry import ModuleRegistry
from .modules.frames import FramesModule
from .modules.flow_raft import FlowRaftModule
from .modules.pose_pedcontrast import PosePedContrastModule
from .modules.pose_hrnet import PoseHrnetModule
from .modules.sam_mask import SamMaskModule
from .modules.sanity import SanityModule


def build_registry() -> ModuleRegistry:
    registry = ModuleRegistry()
    registry.register(FramesModule)
    registry.register(FlowRaftModule)
    registry.register(PosePedContrastModule)
    registry.register(PoseHrnetModule)
    registry.register(SamMaskModule)
    registry.register(SanityModule)
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PTINet preprocess pipeline")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. JAAD")
    parser.add_argument(
        "--mods",
        required=True,
        help="Comma-separated module list, e.g. frames,flow,pose",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract frames from mp4 videos in raw_root and exit",
    )
    parser.add_argument(
        "--build-manifests",
        action="store_true",
        help="Generate manifests from existing frames and exit",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="Path to ffmpeg binary",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Overwrite existing extracted frames",
    )
    parser.add_argument(
        "--config",
        default="preprocess/config/default.yaml",
        help="Path to preprocess config YAML",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root data directory (overrides config data_root)",
    )
    parser.add_argument(
        "--video-ids",
        default="",
        help="Optional comma-separated video ids. Default: use raw subdirs.",
    )
    parser.add_argument(
        "--pose-use-mask",
        action="store_true",
        help="If set, pose depends on mask module.",
    )
    return parser.parse_args()

def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as file:
        payload = yaml.safe_load(file)
    return payload or {}


def resolve_dataset_root(dataset: str, data_root: Path, config: dict) -> Path:
    root_key = f"{dataset}_root"
    dataset_root = config.get(root_key)
    if dataset_root:
        return Path(dataset_root)
    return data_root / dataset

def resolve_raw_root(dataset: str, dataset_root: Path, config: dict) -> Path:
    root_key = f"{dataset}_raw_root"
    raw_root = config.get(root_key)
    if raw_root:
        return Path(raw_root)
    return dataset_root / "raw"


def resolve_preproc_root(dataset: str, dataset_root: Path, config: dict) -> Path:
    root_key = f"{dataset}_preproc_root"
    preproc_root = config.get(root_key)
    if preproc_root:
        return Path(preproc_root)
    return dataset_root / "preproc"


def resolve_video_ids(raw_root: Path, video_ids_arg: str) -> List[str]:
    if video_ids_arg:
        return [item.strip() for item in video_ids_arg.split(",") if item.strip()]
    if raw_root.exists():
        return sorted([path.name for path in raw_root.iterdir() if path.is_dir()])
    return []

def resolve_video_paths(raw_root: Path, video_ids: List[str]) -> List[Path]:
    if video_ids:
        return [raw_root / f"{video_id}.mp4" for video_id in video_ids]
    if raw_root.exists():
        return sorted(raw_root.rglob("*.mp4"))
    return []


def resolve_output_dir(frames_root: Path, raw_root: Path, video_path: Path) -> Path:
    relative_parent = video_path.parent.relative_to(raw_root)
    return frames_root / relative_parent / video_path.stem


def resolve_video_id(raw_root: Path, video_path: Path) -> str:
    relative_parent = video_path.parent.relative_to(raw_root)
    if relative_parent == Path("."):
        return video_path.stem
    return (relative_parent / video_path.stem).as_posix()

def resolve_frame_dirs(frames_root: Path, video_ids: List[str]) -> List[Tuple[str, Path]]:
    frame_dirs: List[Tuple[str, Path]] = []
    if video_ids:
        for video_id in video_ids:
            frame_dirs.append((video_id, frames_root / video_id))
        return frame_dirs
    if frames_root.exists():
        for path in sorted(frames_root.rglob("*")):
            if path.is_dir() and any(child.is_file() for child in path.iterdir()):
                video_id = path.relative_to(frames_root).as_posix()
                frame_dirs.append((video_id, path))
    return frame_dirs


def update_manifest_for_frames(
    manifest_store: ManifestStore,
    video_id: str,
    frames_dir: Path,
    source_path: Path,
) -> None:
    images = list_images(frames_dir)
    frame_ids = [path.stem for path in images]
    frame_ext = images[0].suffix.lower() if images else ".jpg"
    manifest = manifest_store.load(video_id)
    manifest.frame_ids = frame_ids
    manifest.frames_template = str(frames_dir / f"{{frame_id}}{frame_ext}")
    manifest.versions["frames"] = {"source": str(source_path)}
    manifest_store.save(manifest)


def extract_frames(
    ffmpeg_bin: str,
    video_path: Path,
    output_dir: Path,
    force: bool,
) -> None:
    if output_dir.exists() and not force:
        has_frames = any(path.is_file() for path in output_dir.iterdir())
        if has_frames:
            logging.info("Skipping %s (frames already exist)", video_path.name)
            return
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "%05d.jpg")
    cmd = [
        ffmpeg_bin,
        "-y" if force else "-n",
        "-i",
        str(video_path),
        "-vsync",
        "0",
        "-q:v",
        "2",
        output_pattern,
    ]
    logging.info("Extracting frames: %s -> %s", video_path.name, output_dir)
    subprocess.run(cmd, check=True)


def has_frames(output_dir: Path) -> bool:
    return output_dir.exists() and any(path.is_file() for path in output_dir.iterdir())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    config = load_config(Path(args.config))
    data_root_value = args.data_root or config.get("data_root", "data")
    data_root = Path(data_root_value)
    dataset_root = resolve_dataset_root(args.dataset, data_root, config)
    raw_root = resolve_raw_root(args.dataset, dataset_root, config)
    preproc_root = resolve_preproc_root(args.dataset, dataset_root, config)
    manifest_store = ManifestStore(preproc_root / "manifests")

    if args.extract_frames:
        video_ids = []
        if args.video_ids:
            video_ids = [item.strip() for item in args.video_ids.split(",") if item.strip()]
        videos = resolve_video_paths(raw_root, video_ids)
        if not videos:
            raise RuntimeError("No mp4 videos found in raw_root.")
        frames_root = preproc_root / "frames"
        pending_outputs = []
        for video_path in videos:
            output_dir = resolve_output_dir(frames_root, raw_root, video_path)
            if not has_frames(output_dir) or args.force_extract:
                pending_outputs.append(video_path)
        logging.info(
            "Preflight ok: %d videos, %d already have frames, %d pending.",
            len(videos),
            len(videos) - len(pending_outputs),
            len(pending_outputs),
        )
        for video_path in pending_outputs:
            video_id = resolve_video_id(raw_root, video_path)
            output_dir = resolve_output_dir(frames_root, raw_root, video_path)
            extract_frames(
                args.ffmpeg,
                video_path,
                output_dir,
                args.force_extract,
            )
            update_manifest_for_frames(
                manifest_store,
                video_id,
                output_dir,
                video_path,
            )
        return

    if args.build_manifests:
        frames_root = preproc_root / "frames"
        video_ids = []
        if args.video_ids:
            video_ids = [item.strip() for item in args.video_ids.split(",") if item.strip()]
        frame_dirs = resolve_frame_dirs(frames_root, video_ids)
        if not frame_dirs:
            raise RuntimeError("No frames found under preproc_root/frames.")
        for video_id, frames_dir in frame_dirs:
            update_manifest_for_frames(
                manifest_store,
                video_id,
                frames_dir,
                frames_dir,
            )
        return

    registry = build_registry()
    pipeline = Pipeline(registry)
    config = {"pose_use_mask": args.pose_use_mask}

    requested_modules = [item.strip() for item in args.mods.split(",") if item.strip()]
    modules = pipeline.resolve_dependencies(requested_modules, config)
    video_ids = resolve_video_ids(raw_root, args.video_ids)
    if not video_ids:
        raise RuntimeError("No video ids found. Provide --video-ids or raw data.")

    context = PipelineContext(
        dataset=args.dataset,
        data_root=data_root,
        preproc_root=preproc_root,
        raw_root=raw_root,
        manifest_store=manifest_store,
        config=config,
    )

    logging.info("Running modules: %s", ", ".join(modules))
    pipeline.run(context, modules, video_ids)


if __name__ == "__main__":
    main()

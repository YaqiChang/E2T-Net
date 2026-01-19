from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from preprocess.core.manifest import ManifestStore
from preprocess.core.pipeline import Pipeline, PipelineContext
from preprocess.core.registry import ModuleRegistry
from preprocess.modules.frames import FramesModule
from preprocess.modules.flow_raft import FlowRaftModule
from preprocess.modules.pose_pedcontrast import PosePedContrastModule
from preprocess.modules.pose_hrnet import PoseHrnetModule
from preprocess.modules.sam_mask import SamMaskModule
from preprocess.modules.sanity import SanityModule


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
    parser.add_argument("--data-root", default="data", help="Root data directory")
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


def resolve_video_ids(raw_root: Path, video_ids_arg: str) -> List[str]:
    if video_ids_arg:
        return [item.strip() for item in video_ids_arg.split(",") if item.strip()]
    if raw_root.exists():
        return sorted([path.name for path in raw_root.iterdir() if path.is_dir()])
    return []


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    data_root = Path(args.data_root)
    dataset_root = data_root / args.dataset
    raw_root = dataset_root / "raw"
    preproc_root = dataset_root / "preproc"
    manifest_store = ManifestStore(preproc_root / "manifests")

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

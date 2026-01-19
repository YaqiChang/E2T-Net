from __future__ import annotations

from pathlib import Path
from typing import Dict

from preprocess.core.io import copy_images, ensure_paths_exist
from preprocess.core.manifest import Manifest
from preprocess.core.registry import PreprocessModule
from preprocess.core.utils import ensure_dir


class FramesModule(PreprocessModule):
    name = "frames"
    dependencies = []

    def run(self, context: "PipelineContext", video_id: str) -> None:
        raw_frames_dir = self._resolve_raw_frames(context, video_id)
        ensure_paths_exist([raw_frames_dir])
        output_dir = context.preproc_root / "frames" / video_id
        frame_ids = copy_images(raw_frames_dir, output_dir)
        frame_ext = self._resolve_frame_ext(output_dir)

        manifest = context.manifest_store.load(video_id)
        manifest.frame_ids = frame_ids
        manifest.frames_template = str(output_dir / f"{{frame_id}}{frame_ext}")
        manifest.versions["frames"] = {"source": str(raw_frames_dir)}
        context.manifest_store.save(manifest)

    def _resolve_raw_frames(self, context: "PipelineContext", video_id: str) -> Path:
        candidates = [
            context.raw_root / "frames" / video_id,
            context.raw_root / video_id,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _resolve_frame_ext(self, output_dir: Path) -> str:
        for path in output_dir.iterdir():
            if path.is_file():
                return path.suffix.lower()
        return ".jpg"


def module_config() -> Dict[str, object]:
    return {}

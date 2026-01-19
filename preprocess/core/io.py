from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .utils import ensure_dir


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(path: Path) -> List[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def copy_images(src_dir: Path, dst_dir: Path) -> List[str]:
    ensure_dir(dst_dir)
    frame_ids: List[str] = []
    for image_path in list_images(src_dir):
        frame_id = image_path.stem
        target = dst_dir / f"{frame_id}{image_path.suffix.lower()}"
        if not target.exists():
            target.write_bytes(image_path.read_bytes())
        frame_ids.append(frame_id)
    return frame_ids


def ensure_paths_exist(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing paths: {missing}")

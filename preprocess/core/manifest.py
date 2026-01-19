from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import read_json, write_json


@dataclass
class Manifest:
    video_id: str
    frame_ids: List[str] = field(default_factory=list)
    frames_template: Optional[str] = None
    flow_template: Optional[str] = None
    pose_template: Optional[str] = None
    mask_template: Optional[str] = None
    versions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "frame_ids": self.frame_ids,
            "frames_template": self.frames_template,
            "flow_template": self.flow_template,
            "pose_template": self.pose_template,
            "mask_template": self.mask_template,
            "versions": self.versions,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Manifest":
        return cls(
            video_id=payload["video_id"],
            frame_ids=payload.get("frame_ids", []),
            frames_template=payload.get("frames_template"),
            flow_template=payload.get("flow_template"),
            pose_template=payload.get("pose_template"),
            mask_template=payload.get("mask_template"),
            versions=payload.get("versions", {}),
        )


class ManifestStore:
    def __init__(self, manifest_dir: Path):
        self.manifest_dir = manifest_dir

    def path_for(self, video_id: str) -> Path:
        return self.manifest_dir / f"{video_id}.json"

    def load(self, video_id: str) -> Manifest:
        path = self.path_for(video_id)
        if not path.exists():
            return Manifest(video_id=video_id)
        return Manifest.from_dict(read_json(path))

    def save(self, manifest: Manifest) -> None:
        write_json(self.path_for(manifest.video_id), manifest.to_dict())

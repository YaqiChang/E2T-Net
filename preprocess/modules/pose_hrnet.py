from __future__ import annotations

from ..core.registry import PreprocessModule


class PoseHrnetModule(PreprocessModule):
    name = "pose_hrnet"
    dependencies = ["frames"]

    def run(self, context: "PipelineContext", video_id: str) -> None:
        raise RuntimeError(
            "HRNet pose module is not implemented yet. "
            "Wire your HRNet inference and save pose npz."
        )

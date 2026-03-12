from __future__ import annotations

from ..core.registry import PreprocessModule


class PosePedContrastModule(PreprocessModule):
    name = "pose"
    dependencies = ["frames"]
    optional_dependencies = ["mask"]

    def run(self, context: "PipelineContext", video_id: str) -> None:
        raise RuntimeError(
            "Pose module is not implemented yet. "
            "Invoke PedContrast and convert outputs to pose npz."
        )

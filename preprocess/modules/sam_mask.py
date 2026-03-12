from __future__ import annotations

from ..core.registry import PreprocessModule


class SamMaskModule(PreprocessModule):
    name = "mask"
    dependencies = ["frames"]

    def run(self, context: "PipelineContext", video_id: str) -> None:
        raise RuntimeError(
            "Mask module is not implemented yet. "
            "Add SAM or detector-based mask extraction."
        )

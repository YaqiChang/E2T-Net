from __future__ import annotations

from preprocess.core.registry import PreprocessModule


class SanityModule(PreprocessModule):
    name = "sanity"
    dependencies = ["frames"]

    def run(self, context: "PipelineContext", video_id: str) -> None:
        raise RuntimeError(
            "Sanity checks are not implemented yet. "
            "Add pose/flow overlays and statistics."
        )

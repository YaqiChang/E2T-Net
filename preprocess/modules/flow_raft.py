from __future__ import annotations

from ..core.registry import PreprocessModule


class FlowRaftModule(PreprocessModule):
    name = "flow"
    dependencies = ["frames"]

    def run(self, context: "PipelineContext", video_id: str) -> None:
        raise RuntimeError(
            "Flow module is not implemented yet. "
            "Implement RAFT invocation and manifest writing."
        )

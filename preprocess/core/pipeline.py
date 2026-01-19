from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set

from .manifest import ManifestStore
from .registry import ModuleRegistry, PreprocessModule
from .utils import ensure_dir


@dataclass
class PipelineContext:
    dataset: str
    data_root: Path
    preproc_root: Path
    raw_root: Path
    manifest_store: ManifestStore
    config: Dict[str, object]


class Pipeline:
    def __init__(self, registry: ModuleRegistry) -> None:
        self.registry = registry
        self.logger = logging.getLogger("preprocess")

    def resolve_dependencies(self, modules: Iterable[str], config: Dict[str, object]) -> List[str]:
        resolved: List[str] = []
        seen: Set[str] = set()

        def add_module(name: str) -> None:
            if name in seen:
                return
            module_cls = self.registry.get(name)
            for dep in module_cls.dependencies:
                add_module(dep)
            if name == "pose" and config.get("pose_use_mask"):
                for dep in module_cls.optional_dependencies:
                    add_module(dep)
            seen.add(name)
            resolved.append(name)

        for module in modules:
            add_module(module)
        return resolved

    def run(self, context: PipelineContext, modules: Iterable[str], video_ids: Iterable[str]) -> None:
        ensure_dir(context.preproc_root)
        failures: Dict[str, List[str]] = {}
        for module_name in modules:
            module = self.registry.get(module_name)()
            for video_id in video_ids:
                try:
                    self.logger.info("Running %s on %s", module_name, video_id)
                    module.run(context, video_id)
                except Exception as exc:  # noqa: BLE001
                    self.logger.exception("Failed %s on %s", module_name, video_id)
                    failures.setdefault(module_name, []).append(video_id)
        if failures:
            failure_path = context.preproc_root / "failures.json"
            from .utils import write_json

            write_json(failure_path, failures)
            raise RuntimeError(f"Pipeline failed with errors. See {failure_path}.")

from __future__ import annotations

from typing import Dict, Iterable, List, Type


class ModuleRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, Type["PreprocessModule"]] = {}

    def register(self, module_cls: Type["PreprocessModule"]) -> None:
        if not module_cls.name:
            raise ValueError("Module class must define a name.")
        self._registry[module_cls.name] = module_cls

    def get(self, name: str) -> Type["PreprocessModule"]:
        if name not in self._registry:
            raise KeyError(f"Unknown module '{name}'.")
        return self._registry[name]

    def available(self) -> List[str]:
        return sorted(self._registry.keys())

    def create_all(self, names: Iterable[str]) -> List["PreprocessModule"]:
        return [self.get(name)() for name in names]


class PreprocessModule:
    name: str = ""
    dependencies: List[str] = []
    optional_dependencies: List[str] = []

    def run(self, context: "PipelineContext", video_id: str) -> None:
        raise NotImplementedError

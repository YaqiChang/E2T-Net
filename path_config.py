from pathlib import Path
from typing import Any, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PATH_CONFIG = REPO_ROOT / "preprocess" / "config" / "default.yaml"


def load_path_config(config_path: Optional[str] = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_PATH_CONFIG
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload or {}


def get_path_value(key: str, default: Any = None, config: Optional[dict] = None) -> Any:
    payload = config if config is not None else load_path_config()
    return payload.get(key, default)


def normalize_dataset_path(path_value: Any, config: Optional[dict] = None) -> Any:
    if not isinstance(path_value, str) or not path_value:
        return path_value

    payload = config if config is not None else load_path_config()
    target_prefix = str(payload.get("dataset_path_prefix", "")).rstrip("/")
    legacy_prefixes = payload.get("legacy_dataset_path_prefixes", []) or []
    normalized = path_value

    for legacy_prefix in legacy_prefixes:
        legacy_prefix = str(legacy_prefix).rstrip("/")
        if not legacy_prefix or not target_prefix:
            continue
        if normalized == legacy_prefix or normalized.startswith(legacy_prefix + "/"):
            normalized = target_prefix + normalized[len(legacy_prefix):]
            break

    return normalized


def normalize_path_sequence(value: Any, config: Optional[dict] = None) -> Any:
    if isinstance(value, list):
        return [normalize_dataset_path(item, config=config) for item in value]
    return normalize_dataset_path(value, config=config)

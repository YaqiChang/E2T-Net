from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def hash_dict(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

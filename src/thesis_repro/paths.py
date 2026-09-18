from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def product_root() -> Path:
    configured = os.environ.get("THESIS_REPRO_ROOT")
    return Path(configured).resolve() if configured else Path(__file__).resolve().parents[2]


ROOT = product_root()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


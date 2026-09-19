from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StageResult:
    stage: str
    status: str
    execution_class: str
    implemented: bool = True
    executed: bool = False
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    parent_hashes: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status == "PASS" and self.execution_class == "REAL_COMPUTE" and self.executed is not True:
            raise ValueError(f"scientific invariant violated: {self.stage} PASS/REAL_COMPUTE requires executed=True")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_stage_artifact(paths, relative: str, payload: dict[str, Any], logical_id: str, parents=()):
    target = paths.run_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"logical_id": logical_id, "path": str(target.relative_to(paths.root)).replace("\\", "/"), "sha256": sha256_file(target), "size_bytes": target.stat().st_size, "producer": "thesis_repro.stages", "parents": list(parents)}

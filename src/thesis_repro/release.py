from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .paths import ROOT, load_json


RELEASE_ID = "THESIS_V43_SUBMISSION_FINAL_20260917"
EXPECTED_RELEASE_ROOT_SHA256 = "67660326740f091ff848ebd31e7dc725d87ebf5747821419a4bd237bab82ae67"


def release_root(root: Path = ROOT) -> Path:
    return Path(root).resolve() / "frozen/release/submission" / RELEASE_ID


def certified_release_root_hash(release: dict[str, Any]) -> str:
    """Reproduce the certified source-release root hash exactly."""
    inputs = release.get("root_inputs", [])
    basis = "\n".join(
        f"{item['logical_id']}\t{item['sha256']}"
        for item in sorted(inputs, key=lambda item: item["logical_id"])
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def load_certified_release(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    base = release_root(root)
    release = load_json(base / "release.json")
    scope = load_json(base / "release_scope.json")
    pointer_path = root / "frozen/release/CURRENT_RELEASE.json"
    pointer = load_json(pointer_path)
    if release.get("release_id") != RELEASE_ID:
        raise ValueError("CERTIFIED_RELEASE_ID_MISMATCH")
    if release.get("overall") != "THESIS_REPRO_RELEASE_READY" or release.get("ready") is not True:
        raise ValueError("CERTIFIED_RELEASE_NOT_READY")
    if release.get("scientific_authority") != "V1.3_FROZEN_EVIDENCE":
        raise ValueError("CERTIFIED_RELEASE_AUTHORITY_MISMATCH")
    if scope.get("scope") != "FROZEN_EVIDENCE_REPRODUCTION_AND_REPORTING":
        raise ValueError("CERTIFIED_RELEASE_SCOPE_MISMATCH")
    if pointer.get("release_id") != RELEASE_ID:
        raise ValueError("CURRENT_RELEASE_ID_MISMATCH")
    expected_path = str(Path("frozen/release/submission") / RELEASE_ID).replace("\\", "/")
    if str(pointer.get("release_path", "")).replace("\\", "/") != expected_path:
        raise ValueError("CURRENT_RELEASE_PATH_MISMATCH")
    if pointer.get("release_root_sha256") != EXPECTED_RELEASE_ROOT_SHA256:
        raise ValueError("CURRENT_RELEASE_ROOT_HASH_MISMATCH")
    computed = certified_release_root_hash(release)
    if computed != EXPECTED_RELEASE_ROOT_SHA256 or release.get("release_root_sha256") != EXPECTED_RELEASE_ROOT_SHA256:
        raise ValueError("CERTIFIED_RELEASE_ROOT_HASH_MISMATCH")
    return {"release": release, "scope": scope, "pointer": pointer, "root": str(base), "computed_root_sha256": computed}

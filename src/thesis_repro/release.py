from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths import ROOT, load_json


RELEASE_ID = "THESIS_V43_SUBMISSION_FINAL_20260917"


def release_root(root: Path = ROOT) -> Path:
    return Path(root).resolve() / "frozen/release/submission" / RELEASE_ID


def load_certified_release(root: Path = ROOT) -> dict[str, Any]:
    base = release_root(root)
    release = load_json(base / "release.json")
    scope = load_json(base / "release_scope.json")
    if release.get("release_id") != RELEASE_ID:
        raise ValueError("CERTIFIED_RELEASE_ID_MISMATCH")
    if release.get("overall") != "THESIS_REPRO_RELEASE_READY" or release.get("ready") is not True:
        raise ValueError("CERTIFIED_RELEASE_NOT_READY")
    if release.get("scientific_authority") != "V1.3_FROZEN_EVIDENCE":
        raise ValueError("CERTIFIED_RELEASE_AUTHORITY_MISMATCH")
    if scope.get("scope") != "FROZEN_EVIDENCE_REPRODUCTION_AND_REPORTING":
        raise ValueError("CERTIFIED_RELEASE_SCOPE_MISMATCH")
    return {"release": release, "scope": scope, "root": str(base)}

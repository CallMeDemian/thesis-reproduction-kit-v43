from __future__ import annotations

"""Single-source loader for the corrected Oracle/RL research profile."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

CONTRACT_RELATIVE_PATH = Path("configs/current/final_freeze/final_oracle_rl_contract.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_canonical_rl_profile(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(project_root).resolve()
    path = root / CONTRACT_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Canonical Oracle/RL contract is missing: {path}")
    contract = json.loads(path.read_text(encoding="utf-8-sig"))
    profile = dict(contract.get("canonical_rl_profile") or {})
    required = {"profile_id", "profile_version", "stage2", "stage3", "stage4", "stage5"}
    missing = sorted(required.difference(profile))
    if missing:
        raise ValueError(f"Canonical Oracle/RL profile is incomplete: {missing}")
    metadata = {
        "canonical_rl_profile_id": str(profile["profile_id"]),
        "canonical_rl_profile_version": str(profile["profile_version"]),
        "canonical_rl_contract_path": CONTRACT_RELATIVE_PATH.as_posix(),
        "canonical_rl_contract_sha256": _sha256(path),
    }
    return profile, metadata


def require_profile_name(requested: str, profile: Mapping[str, Any]) -> None:
    expected = str(profile["profile_id"])
    if requested not in ("", expected):
        raise ValueError(f"Unknown research profile {requested!r}; canonical profile is {expected!r}")


def same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def profile_mismatches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {key: {"actual": actual.get(key), "expected": value} for key, value in expected.items() if not same_value(actual.get(key), value)}

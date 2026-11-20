"""Strict loader for the production V4.3 nine-action Stage2 contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CANONICAL_ACTION_CONTRACT_PATH = Path(
    "data/final_freeze/stage2_candidate_projection/candidate_action_contract_v4_3.json"
)
ACTION_CONTRACT_PROMOTION_MANIFEST_PATH = Path(
    "data/final_freeze/stage2_candidate_projection/candidate_action_contract_v4_3_manifest.json"
)
ACTION_CONTRACT_VERIFIER_PATH = Path(
    "data/final_freeze/stage2_candidate_projection/verification/action_contract_v4_3/"
    "02_verification/strict_verifier.json"
)
EXPECTED_ACTION_CONTRACT_SHA256 = (
    "f722e189a265e111c018341254a648ddb9cecc05aa8c12c3547a023ea4c458c0"
)
EXPECTED_ACTION_CONTRACT_CONTENT_HASH = (
    "ffa9a065a15774d92e19109a1373683b3cb50690a8e735fb0d938c46c9c8363a"
)
ACTION_IDS = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")
ACTION_SEMANTIC_VERSION = "simulator_action_semantic_contract_v4_3_dl_interest_full_year"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v43_action_contract(project_root: Path) -> tuple[dict[str, Any], str]:
    root = Path(project_root).resolve()
    path = root / CANONICAL_ACTION_CONTRACT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Canonical V4.3 action contract is missing: {path}")
    digest = file_sha256(path)
    if digest != EXPECTED_ACTION_CONTRACT_SHA256:
        raise ValueError(
            "Canonical V4.3 action contract file hash changed: "
            f"expected={EXPECTED_ACTION_CONTRACT_SHA256} actual={digest}"
        )
    contract = json.loads(path.read_text(encoding="utf-8"))
    if tuple(contract.get("candidate_ids") or ()) != ACTION_IDS:
        raise ValueError("Canonical V4.3 action order/count changed")
    if contract.get("simulator_action_semantic_contract_version") != ACTION_SEMANTIC_VERSION:
        raise ValueError("Canonical V4.3 action semantic version changed")
    if contract.get("candidate_action_contract_hash") != EXPECTED_ACTION_CONTRACT_CONTENT_HASH:
        raise ValueError("Canonical V4.3 action contract content hash changed")
    from credit_recourse.rl.common.semantic_action_v4_1_contract import validate_timing_contract
    validate_timing_contract(contract)
    return contract, digest

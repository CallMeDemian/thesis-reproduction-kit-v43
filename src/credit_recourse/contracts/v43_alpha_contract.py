"""Strict loader for the production V4.3 Oracle Alpha Stage1 contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

def _runtime(project_root: Path | None = None):
    from credit_recourse.oracle.fresh_runtime import resolve_fresh_oracle_runtime
    return resolve_fresh_oracle_runtime(project_root or Path.cwd())


def canonical_alpha_contract_path(project_root: Path | None = None) -> Path:
    return _runtime(project_root).backends_root / "alpha/oracle_alpha_params.json"


def alpha_contract_promotion_manifest_path(project_root: Path | None = None) -> Path:
    return _runtime(project_root).ledgers_root / "stage1_alpha_contract_manifest.json"


def alpha_strict_verifier_path(project_root: Path | None = None) -> Path:
    return _runtime(project_root).ledgers_root / "stage1_alpha_contract_verification.json"


def canonical_v43_oracle_registry_path(project_root: Path | None = None) -> Path:
    return _runtime(project_root).registry_path
EXPECTED_ALPHA_CONTRACT_SHA256 = (
    "7ddc6ebea4fe8936332b323d07438c9ec82ffc19b030c9138ff85dc116c4694b"
)
EXPECTED_ALPHA_CONTENT_HASH = (
    "9c8d379e375c7337d445d8db8dcec3a1418061c32320302d21155c9466544142"
)
ALPHA_CONTRACT_VERSION = "oracle_alpha_monotone_empty_bins_v1"
EMPTY_BIN_REPAIR_RULE = "linear_between_populated_bin_centers_constant_endpoint_step_lookup_v1"
SELECTED_VARIABLES = (
    "R006", "R064", "R085", "R116", "R157", "R182",
    "cap_change_count_3y", "log_assets", "operating_loss_freq_3y",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_v43_alpha_contract(project_root: Path) -> tuple[dict[str, Any], str]:
    root = Path(project_root).resolve()
    path = canonical_alpha_contract_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical V4.3 Alpha contract is missing: {path}")
    digest = file_sha256(path)
    if digest != EXPECTED_ALPHA_CONTRACT_SHA256:
        raise ValueError(
            "Canonical V4.3 Alpha contract file hash changed: "
            f"expected={EXPECTED_ALPHA_CONTRACT_SHA256} actual={digest}"
        )
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("oracle_alpha_contract_version") != ALPHA_CONTRACT_VERSION:
        raise ValueError("Canonical V4.3 Alpha contract version changed")
    if contract.get("oracle_alpha_contract_hash") != EXPECTED_ALPHA_CONTENT_HASH:
        raise ValueError("Canonical V4.3 Alpha internal contract hash changed")
    if tuple(contract.get("selected_variables") or ()) != SELECTED_VARIABLES:
        raise ValueError("Canonical V4.3 Alpha selected-variable order changed")
    repair = contract.get("empty_bin_repair") or {}
    if repair.get("rule") != EMPTY_BIN_REPAIR_RULE:
        raise ValueError("Canonical V4.3 Alpha empty-bin repair rule changed")
    if repair.get("evaluation_data_used_for_rule_or_scores") is not False:
        raise ValueError("Canonical V4.3 Alpha contract is evaluation-contaminated")
    if repair.get("weights_or_grade_cutoffs_refit") is not False:
        raise ValueError("Canonical V4.3 Alpha weights or cutoffs were refit")
    return contract, digest

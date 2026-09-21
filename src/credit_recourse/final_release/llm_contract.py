from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from .action_validation import CANDIDATES, DIMENSIONS
from .common import ContractError, find_repo_root, load_json, sha256_file
from .contract import CONDITIONS, MODEL_KEYS, load_design


def validate_llm_contract(root: Path | None = None) -> dict[str, Any]:
    repo = root or find_repo_root()
    bundle = load_design(repo)
    configured_root = os.environ.get("THESIS_REPRO_LLM_CONFIG_ROOT")
    config_root = Path(configured_root) if configured_root and Path(configured_root).is_absolute() else (repo / configured_root if configured_root else repo / "frozen/evidence/llm")
    # The published simulator registry is not part of the fresh checkout.
    # Resolve the active design bundle first, then use its run-local or
    # evidence-local C6-EX files.  A legacy contract_manifest remains an
    # optional compatibility source, never the fresh runtime authority.
    registry_path = config_root / "contract_manifest.json"
    registry = load_json(registry_path) if registry_path.is_file() else {
        "rl_release_hash": bundle.release["c3e_release_hash"],
        "c6ex_manifest": str(config_root / "C6EX_manifest.json"),
        "llm_matrix": str(config_root / "experiment_matrix.csv"),
        "c6ex_materialized": str(config_root / "C6EX_materialized.parquet"),
        "c6ex_permutation": str(config_root / "C6EX_permutation.parquet"),
    }
    ref = bundle.design["active_reference_policy"]
    hashes = {registry["rl_release_hash"], ref["release_hash"], bundle.release["c3e_release_hash"]}
    c6_manifest = Path(str(registry["c6ex_manifest"]))
    if not c6_manifest.is_absolute():
        c6_manifest = config_root / c6_manifest.name if str(c6_manifest).startswith("configs/current") else repo / c6_manifest
    c6 = load_json(c6_manifest)
    hashes.add(c6["c3e_release_hash"])
    if len(hashes) != 1:
        raise ContractError(f"Cross-component C3-E hash mismatch: {hashes}")
    matrix = pd.DataFrame(bundle.matrix)
    counts_model = matrix.groupby("model").size().to_dict()
    requests_model = matrix.groupby("model")["expected_requests"].apply(lambda s: s.astype(int).sum()).to_dict()
    phase_requests = matrix.groupby("phase")["expected_requests"].apply(lambda s: s.astype(int).sum()).to_dict()
    materialized_path = Path(str(registry["c6ex_materialized"]))
    if not materialized_path.is_absolute():
        materialized_path = config_root / materialized_path.name if str(materialized_path).startswith("configs/current") else repo / materialized_path
    permutation_path = Path(str(registry["c6ex_permutation"]))
    if not permutation_path.is_absolute():
        permutation_path = config_root / permutation_path.name if str(permutation_path).startswith("configs/current") else repo / permutation_path
    matrix_path = Path(str(registry["llm_matrix"]))
    if not matrix_path.is_absolute():
        matrix_path = config_root / matrix_path.name if str(matrix_path).startswith("configs/current") else repo / matrix_path
    materialized = pd.read_parquet(materialized_path).sort_values("row_id")
    if len(materialized) != 575 or materialized["self_donor_collision"].any():
        raise ContractError("Current C6-EX materialization is not a 575-firm derangement")
    if sha256_file(materialized_path) != c6["materialized_sha256"]:
        raise ContractError("Current C6-EX materialized hash drift")
    if sha256_file(permutation_path) != c6["permutation_sha256"]:
        raise ContractError("Current C6-EX permutation hash drift")
    return {
        "status": "PASS", "design_release_hash": bundle.design_release_hash,
        "models": list(MODEL_KEYS), "conditions": list(CONDITIONS), "total_cells": len(matrix),
        "cells_per_model": counts_model, "requests_per_model": requests_model,
        "total_api_calls": int(matrix["expected_requests"].astype(int).sum()),
        "main_requests": int(phase_requests["MAIN"]), "stability_requests": int(phase_requests["STABILITY"]),
        "reference_policy": ref, "cross_component_reference_hash": next(iter(hashes)),
        "c6ex_self_collisions": 0, "c6ex_action_label_collisions": int(c6["action_label_collision_count"]),
        "action_dimensions": list(DIMENSIONS), "candidate_order": list(CANDIDATES),
        "matrix_sha256": sha256_file(matrix_path),
    }


def industry_binding_status(root: Path | None = None) -> dict[str, Any]:
    """Validate the frozen OpenDART binding used by IC-b and IC-c."""
    repo = root or find_repo_root()
    configured_root = os.environ.get("THESIS_REPRO_LLM_CONFIG_ROOT")
    config_root = Path(configured_root) if configured_root and Path(configured_root).is_absolute() else (repo / configured_root if configured_root else repo / "frozen/evidence/llm")
    contract = load_json(config_root / "information_contract.json")
    manifest_path = config_root / contract["industry_binding_manifest"] if not Path(str(contract["industry_binding_manifest"])).is_absolute() else Path(str(contract["industry_binding_manifest"]))
    evidence = load_json(manifest_path) if manifest_path.is_file() else {}
    binding_rel = str(evidence.get("binding_artifact", ""))
    binding_path = config_root / binding_rel if binding_rel and not Path(binding_rel).is_absolute() else Path(binding_rel) if binding_rel else None
    declared_hash = str(evidence.get("binding_artifact_sha256", ""))
    artifact_hash_valid = bool(binding_path and binding_path.is_file() and declared_hash and sha256_file(binding_path) == declared_hash)
    rows = unique_firms = nonmissing = 0
    if artifact_hash_valid and binding_path is not None:
        try:
            binding = pd.read_parquet(binding_path)
            rows = len(binding)
            unique_firms = int(binding["firm_key"].nunique()) if "firm_key" in binding else 0
            nonmissing = int(binding["induty_code"].notna().sum()) if "induty_code" in binding else 0
            if "induty_code" in binding:
                nonmissing = int(binding["induty_code"].astype(str).str.strip().ne("").sum())
        except Exception:
            artifact_hash_valid = False
    leak_path = repo / "repro/manifests/runtime_gates/api_key_leak_scan_full.json"
    leak = load_json(leak_path) if leak_path.is_file() else {}
    leak_scan_pass = leak.get("status") == "PASS" and int(leak.get("key_literal_hits", -1)) == 0
    fresh_runtime = bool(configured_root)
    ready = bool(
        evidence.get("frozen")
        and int(evidence.get("unresolved_count", 575)) == 0
        and artifact_hash_valid
        and rows == 575
        and unique_firms == 575
        and nonmissing == 575
        and (fresh_runtime or leak_scan_pass)
    )
    return {
        "ready": ready, "status": evidence.get("status", "MISSING"),
        "nonmissing_count": int(contract["valid_industry_nonmissing"]),
        "missing_count": int(contract["valid_industry_missing"]),
        "coverage_rate": float(contract["valid_industry_coverage_rate"]),
        "unresolved_count": int(evidence.get("unresolved_count", 575)),
        "binding_artifact": binding_rel or None,
        "path": contract["industry_binding_manifest"],
        "sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "artifact_hash_valid": artifact_hash_valid,
        "binding_rows": rows,
        "binding_unique_firms": unique_firms,
        "binding_nonmissing_induty_code": nonmissing,
        "api_key_leak_scan": "NOT_APPLICABLE_FRESH_RUNTIME" if fresh_runtime else ("PASS" if leak_scan_pass else "FAIL_OR_MISSING"),
    }


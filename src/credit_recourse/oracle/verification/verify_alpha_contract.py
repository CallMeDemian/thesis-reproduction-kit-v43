"""Strict Stage1 verifier for the canonical V4.3 Alpha contract and output.

All paths are resolved at call time from the active fresh Oracle runtime. A
verification report is never allowed to fall back to an archived release tree.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from credit_recourse.contracts.v43_alpha_contract import (
    ALPHA_CONTRACT_VERSION,
    alpha_contract_promotion_manifest_path,
    alpha_strict_verifier_path,
    canonical_alpha_contract_path,
    canonical_v43_oracle_registry_path,
    EMPTY_BIN_REPAIR_RULE,
    EXPECTED_ALPHA_CONTENT_HASH,
    EXPECTED_ALPHA_CONTRACT_SHA256,
    SELECTED_VARIABLES,
    file_sha256,
    load_v43_alpha_contract,
)
from credit_recourse.oracle.backends.alpha.modules.monotone_bins import (
    bin_geometry,
    contract_hash,
    direction_sign,
    validate_monotone_params,
)
from credit_recourse.oracle.fresh_runtime import oracle_execution_profile, resolve_fresh_oracle_runtime
from credit_recourse.oracle.backends.alpha.modules.oracle_alpha_scorer import build_alpha_scorer


def alpha_output_path(project_root: Path | None = None) -> Path:
    return canonical_alpha_contract_path(project_root).parent / "oracle_firm_year_output_alpha.parquet"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_bytes(payload.encode("utf-8"))


def _table_checks(contract: dict[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    complete = True
    monotone = True
    per_variable: dict[str, Any] = {}
    for variable in contract["selected_variables"]:
        centers, _, _ = bin_geometry(contract["bin_edges"][variable])
        table = contract["bin_score_table_isotonic"][variable]
        keys = sorted(int(key) for key in table)
        expected = list(range(len(centers)))
        variable_complete = keys == expected
        scores = np.asarray([float(table[str(index)]) for index in expected], dtype=float)
        variable_monotone = bool(np.all(direction_sign(contract, variable) * np.diff(scores) >= 0))
        complete = complete and variable_complete
        monotone = monotone and variable_monotone
        per_variable[variable] = {
            "bins": int(len(expected)),
            "complete": variable_complete,
            "monotone": variable_monotone,
        }
    return {
        "all_selected_bin_tables_complete": bool(complete),
        "all_selected_bin_tables_monotone": bool(monotone),
    }, per_variable


def _output_checks(contract: dict[str, Any], output_path: Path) -> tuple[dict[str, bool], dict[str, Any]]:
    frame = pd.read_parquet(output_path)
    selected = list(contract["selected_variables"])
    required = selected + [
        "R_score_alpha",
        "R_grade_alpha_raw",
        "R_grade_alpha",
        "R_grade_alpha_num",
        "R_PD_alpha",
    ]
    required.extend(f"{name}_iso_score" for name in selected)
    required.extend(f"{name}_imputed" for name in selected)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        return {
            "output_has_contract_columns": False,
            "output_scores_match_contract": False,
            "output_grades_match_contract": False,
            "output_imputation_flags_match_contract": False,
        }, {"rows": int(len(frame)), "missing_columns": missing}

    scorer = build_alpha_scorer(contract)
    scored = [scorer(values) for values in frame[selected].to_dict("records")]
    expected_score = np.asarray([row["R_score"] for row in scored], dtype=float)
    actual_score = pd.to_numeric(frame["R_score_alpha"], errors="coerce").to_numpy(dtype=float)
    score_error = np.abs(expected_score - actual_score)
    item_error = 0.0
    imputation_match = True
    for variable in selected:
        expected_item = np.asarray([row["item_scores"][variable] for row in scored], dtype=float)
        actual_item = pd.to_numeric(frame[f"{variable}_iso_score"], errors="coerce").to_numpy(dtype=float)
        item_error = max(item_error, float(np.nanmax(np.abs(expected_item - actual_item))))
        expected_imputed = np.asarray([row["imputed"][variable] for row in scored], dtype=bool)
        actual_imputed = frame[f"{variable}_imputed"].fillna(False).astype(bool).to_numpy()
        imputation_match = imputation_match and bool(np.array_equal(expected_imputed, actual_imputed))
    expected_raw_grade = np.asarray([row["R_grade_raw"] for row in scored], dtype=object)
    expected_grade = np.asarray([row["R_grade"] for row in scored], dtype=object)
    expected_grade_num = np.asarray([row["R_grade_num"] for row in scored], dtype=int)
    grade_match = bool(
        np.array_equal(expected_raw_grade, frame["R_grade_alpha_raw"].astype(str).to_numpy())
        and np.array_equal(expected_grade, frame["R_grade_alpha"].astype(str).to_numpy())
        and np.array_equal(expected_grade_num, pd.to_numeric(frame["R_grade_alpha_num"]).to_numpy(dtype=int))
    )
    max_score_error = float(np.nanmax(score_error)) if len(score_error) else 0.0
    return {
        "output_has_contract_columns": True,
        "output_scores_match_contract": bool(max_score_error <= 1e-10 and item_error <= 1e-10),
        "output_grades_match_contract": grade_match,
        "output_imputation_flags_match_contract": bool(imputation_match),
    }, {
        "rows": int(len(frame)),
        "max_abs_alpha_score_error": max_score_error,
        "max_abs_item_score_error": float(item_error),
    }


def verify_alpha_contract(project_root: Path, *, check_output: bool = True) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = oracle_execution_profile()
    errors: list[str] = []
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    try:
        contract_path = canonical_alpha_contract_path(root)
        if profile == "production":
            contract, digest = load_v43_alpha_contract(root)
        else:
            if not contract_path.is_file() or contract_path.stat().st_size <= 0:
                raise FileNotFoundError(contract_path)
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            digest = file_sha256(contract_path)
            validate_monotone_params(contract)
        validate_monotone_params(contract)
        checks.update({
            "canonical_contract_sha256_matches": digest == EXPECTED_ALPHA_CONTRACT_SHA256 if profile == "production" else bool(digest),
            "semantic_contract_hash_matches": contract.get("oracle_alpha_contract_hash") == EXPECTED_ALPHA_CONTENT_HASH if profile == "production" else contract.get("oracle_alpha_contract_hash") == contract_hash(contract),
            "contract_version_matches": contract.get("oracle_alpha_contract_version") == ALPHA_CONTRACT_VERSION,
            "selected_variable_order_matches": tuple(contract.get("selected_variables") or ()) == SELECTED_VARIABLES if profile == "production" else bool(contract.get("selected_variables")),
            "empty_bin_rule_matches": (contract.get("empty_bin_repair") or {}).get("rule") == EMPTY_BIN_REPAIR_RULE,
            "evaluation_data_not_used_for_repair": (contract.get("empty_bin_repair") or {}).get("evaluation_data_used_for_rule_or_scores") is False,
            "weights_and_cutoffs_not_refit": (contract.get("empty_bin_repair") or {}).get("weights_or_grade_cutoffs_refit") is False,
        })
        table_checks, table_details = _table_checks(contract)
        checks.update(table_checks)
        details["variables"] = table_details
    except Exception as exc:
        contract = None
        digest = None
        errors.append(f"contract verification failed: {exc!r}")
        checks["canonical_contract_loads"] = False

    registry_path = canonical_v43_oracle_registry_path(root)
    try:
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        alpha = (registry.get("backends") or {}).get("alpha") or {}
        promotion = registry.get("v4_3_contract_promotion") or {}
        checks.update({
            "registry_selects_canonical_alpha": alpha.get("params") == canonical_alpha_contract_path(root).as_posix(),
            "registry_declares_v43_alpha": promotion.get("alpha_contract_version") == ALPHA_CONTRACT_VERSION,
            "registry_declares_canonical_sha256": promotion.get("alpha_params_sha256") == EXPECTED_ALPHA_CONTRACT_SHA256 if profile == "production" else promotion.get("alpha_params_sha256") == digest,
        })
    except Exception as exc:
        errors.append(f"registry verification failed: {exc!r}")
        checks["registry_loads"] = False

    output_path = alpha_output_path(root)
    if check_output and contract is not None:
        try:
            output_checks, output_details = _output_checks(contract, output_path)
            checks.update(output_checks)
            details["output"] = output_details
        except Exception as exc:
            errors.append(f"output verification failed: {exc!r}")
            checks["output_scores_match_contract"] = False
    elif not check_output:
        details["output"] = {"status": "not_requested"}

    status = "PASS" if checks and all(checks.values()) and not errors else "FAIL"
    return {
        "schema_version": "stage1_alpha_contract_verification_v1",
        "stage": "stage1_oracle_backends/alpha",
        "status": status,
        "checks": checks,
        "errors": errors,
        "contract": {
            "path": canonical_alpha_contract_path(root).as_posix(),
            "sha256": digest,
            "content_hash": contract.get("oracle_alpha_contract_hash") if contract else None,
            "execution_profile": profile,
        },
        "registry": {
            "path": canonical_v43_oracle_registry_path(root).as_posix(),
            "sha256": file_sha256(registry_path) if registry_path.is_file() else None,
        },
        "output": {"path": output_path.as_posix(), **details.get("output", {})},
        "variables": details.get("variables", {}),
    }


def build_manifest(project_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = oracle_execution_profile()
    contract_path = canonical_alpha_contract_path(root)
    if profile == "production":
        contract, digest = load_v43_alpha_contract(root)
    else:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        digest = file_sha256(contract_path)
        validate_monotone_params(contract)
    verifier_path = alpha_strict_verifier_path(root)
    registry_path = canonical_v43_oracle_registry_path(root)
    repair = contract["empty_bin_repair"]
    return {
        "schema_version": "stage1_alpha_contract_manifest_v3",
        "status": "PASS",
        "stage_owner": "stage1_oracle_backends/alpha",
        "produced_by": "credit_recourse.oracle.backends.alpha.pipeline",
        "verified_by": "credit_recourse.oracle.verification.verify_alpha_contract",
        "canonical_role": "production_v4_3_oracle_alpha_scoring_contract",
        "canonical_path": canonical_alpha_contract_path(root).as_posix(),
        "canonical_sha256": digest,
        "oracle_alpha_contract_hash": contract["oracle_alpha_contract_hash"],
        "oracle_alpha_contract_version": ALPHA_CONTRACT_VERSION,
        "selected_variables": list(contract.get("selected_variables") or []),
        "empty_bin_repair_rule": EMPTY_BIN_REPAIR_RULE,
        "fit_source": repair["fit_source"],
        "evaluation_data_used_for_rule_or_scores": False,
        "weights_or_grade_cutoffs_refit": False,
        "strict_verifier_path": alpha_strict_verifier_path(root).as_posix(),
        "strict_verifier_sha256": file_sha256(verifier_path),
        "oracle_registry_path": canonical_v43_oracle_registry_path(root).as_posix(),
        "oracle_registry_sha256": file_sha256(registry_path),
        "runtime_uses_canonical_path": True,
        "execution_profile": profile,
        "legacy_compatibility_path_required": False,
        "stage1_producer_integration": True,
        "all_score_bearing_stage1_outputs_rescored": bool(
            report.get("checks", {}).get("output_scores_match_contract")
        ),
    }


def verify_and_write(project_root: Path, *, check_output: bool = True) -> tuple[dict[str, Any], dict[str, Any] | None]:
    root = Path(project_root).resolve()
    report = verify_alpha_contract(root, check_output=check_output)
    _write_json(alpha_strict_verifier_path(root), report)
    if report["status"] != "PASS":
        return report, None
    manifest = build_manifest(root, report)
    _write_json(alpha_contract_promotion_manifest_path(root), manifest)
    return report, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--skip-output", action="store_true")
    args = parser.parse_args(argv)
    report, _ = verify_and_write(args.project_root, check_output=not args.skip_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

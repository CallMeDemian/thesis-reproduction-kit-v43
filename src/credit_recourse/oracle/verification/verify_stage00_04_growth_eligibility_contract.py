"""Verify Stage00_02 growth eligibility reaches Stage00_04 without resurrection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.oracle.stage1.stage00_04_variable_selection.growth_candidate_contract import (
    GROWTH_CANDIDATE_CONTRACT_VERSION,
    as_bool_series,
    load_canonical_financial_inputs,
)


def verify_contract(project_root: Path, *, require_selection_output: bool = False) -> dict[str, Any]:
    root = Path(project_root).resolve()
    final = root / "data" / "final_freeze"
    stage2 = final / "stage1_oracle_inputs" / "stage00_02_financial_ratio_engineering"
    stage4 = final / "stage1_oracle_inputs" / "stage00_04_variable_selection"

    _, candidates, canonical = load_canonical_financial_inputs(stage2)
    denom_mask = as_bool_series(candidates["growth_denom_instability"])
    selected_mask = as_bool_series(candidates["selected_eligible"])
    errors: list[str] = []
    if (denom_mask & selected_mask).any():
        errors.append("canonical pool contains selected denominator-unstable growth candidates")

    checks: dict[str, Any] = {
        "canonical_contract_version": canonical.get("contract_version"),
        "canonical_growth_candidates": canonical.get("growth_candidate_count"),
        "canonical_growth_selected_eligible": canonical.get("growth_selected_eligible_count"),
        "denominator_instability_blocked_ratio_ids": canonical.get("denominator_instability_blocked_ratio_ids", []),
    }

    if require_selection_output:
        metadata_path = stage4 / "growth_candidate_input_contract.json"
        master_path = stage4 / "selected_variable_master.csv"
        score_path = stage4 / "selection_score_table.csv"
        for label, path in [
            ("Stage00_04 growth input contract", metadata_path),
            ("Stage00_04 selected variable master", master_path),
            ("Stage00_04 selection score table", score_path),
        ]:
            if not path.exists() or path.stat().st_size <= 0:
                errors.append(f"missing {label}: {path}")

        if not errors:
            observed = json.loads(metadata_path.read_text(encoding="utf-8"))
            if observed.get("contract_version") != GROWTH_CANDIDATE_CONTRACT_VERSION:
                errors.append("Stage00_04 growth input contract version mismatch")
            expected_hashes = {
                key: value.get("sha256") for key, value in canonical.get("input_artifacts", {}).items()
            }
            observed_hashes = {
                key: value.get("sha256") for key, value in observed.get("input_artifacts", {}).items()
            }
            if observed_hashes != expected_hashes:
                errors.append("Stage00_04 growth input hashes do not match active Stage00_02 canonical artifacts")

            master = pd.read_csv(master_path)
            scores = pd.read_csv(score_path)
            selected_ids = set(master["variable_id"].astype(str))
            blocked_ids = set(canonical.get("denominator_instability_blocked_ratio_ids", []))
            overlap = sorted(selected_ids & blocked_ids)
            if overlap:
                errors.append(f"denominator-unstable growth variables selected: {overlap}")
            selected_growth = master[master["category"].astype(str).eq("성장성")]
            if len(selected_growth) > 1:
                errors.append(f"expected at most one selected growth representative, observed={len(selected_growth)}")
            elif len(selected_growth) == 0:
                status_path = stage00_04_dir / "category_selection_status.json"
                if not status_path.exists():
                    errors.append("growth has no selected representative but category_selection_status.json is missing")
                else:
                    status = json.loads(status_path.read_text(encoding="utf-8"))
                    growth_rows = [x for x in status.get("categories", []) if str(x.get("category")) == "성장성"]
                    if len(growth_rows) != 1 or growth_rows[0].get("status") != "NO_VALID_CANDIDATE" or growth_rows[0].get("allowed_by_contract") is not True:
                        errors.append(f"growth no-candidate state is not explicit/allowed: {growth_rows}")
            elif "candidate_pool_contract_version" not in selected_growth.columns:
                errors.append("selected growth representative missing candidate_pool_contract_version")
            elif not selected_growth["candidate_pool_contract_version"].astype(str).eq(GROWTH_CANDIDATE_CONTRACT_VERSION).all():
                errors.append("selected growth representative has stale candidate-pool contract")

            blocked_scores = scores[scores["variable_id"].astype(str).isin(blocked_ids)]
            if len(blocked_scores) != len(blocked_ids):
                errors.append("selection score table does not retain every denominator-instability diagnostic row")
            elif as_bool_series(blocked_scores["selected_eligible"]).any():
                errors.append("selection score table marks denominator-instability row selected_eligible=True")
            checks.update({
                "selected_variables": sorted(selected_ids),
                "selected_growth_representative": selected_growth["variable_id"].astype(str).tolist(),
                "selection_output_contract_path": str(metadata_path),
            })

    return {
        "status": "PASS" if not errors else "FAIL",
        "contract_version": GROWTH_CANDIDATE_CONTRACT_VERSION,
        "require_selection_output": bool(require_selection_output),
        "errors": errors,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--require-selection-output", action="store_true")
    args = parser.parse_args(argv)
    result = verify_contract(Path(args.project_root), require_selection_output=args.require_selection_output)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

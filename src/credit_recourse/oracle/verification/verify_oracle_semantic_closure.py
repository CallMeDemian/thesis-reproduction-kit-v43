"""Strict closure audit for the corrected fresh Oracle development path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from credit_recourse.oracle.stage1.stage00_04_variable_selection.growth_candidate_contract import (
    load_canonical_financial_inputs,
)
from credit_recourse.simulator.oracle_variables import audit_formula_registry
from credit_recourse.oracle.selected_variable_current_contract import (
    validate_dynamic_selected_variable_records,
)


QUALITY_SCOPE = "full_available_panel_including_2024_retained_by_research_contract"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    final = root / "data" / "final_freeze"
    inputs = final / "stage1_oracle_inputs"
    s1 = inputs / "stage00_01_rating_statement_integration"
    s2 = inputs / "stage00_02_financial_ratio_engineering"
    s3 = inputs / "stage00_03_nonfinancial_metadata"
    s4 = inputs / "stage00_04_variable_selection"
    backends = final / "stage1_oracle_backends"
    checks: list[dict[str, Any]] = []
    errors: list[str] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"check": name, "status": "PASS" if condition else "FAIL", "detail": detail})
        if not condition:
            errors.append(f"{name}: {detail}")

    try:
        s1_meta = _read_json(s1 / "stage00_01_context_contract_repair.json")
        s1_coverage = _read_json(s1 / "stage00_01_statement_coverage_contract.json")
        check("stage00_01_backward_asof", "latest_source_year_le_target" in str(s1_meta.get("temporal_context_contract")), s1_meta)
        check("stage00_01_no_future_context", int(s1_meta.get("future_context_rows", -1)) == 0, s1_meta.get("future_context_rows"))
        check("stage00_01_statement_coverage_from_cleaned_panels", s1_coverage.get("contract_version") == "stage00_01_cleaned_panel_membership_coverage_v1" and int(s1_coverage.get("coverage_unique_count", 0)) >= 2, s1_coverage)

        _, candidates, canonical = load_canonical_financial_inputs(s2)
        check("stage00_02_canonical_contract", canonical.get("status") == "PASS", canonical)
        check("quality_gate_scope_retained", canonical.get("quality_gate_scope") == QUALITY_SCOPE, canonical.get("quality_gate_scope"))
        check("denominator_instability_excluded", not bool(candidates.loc[candidates["growth_denom_instability"].astype(bool), "selected_eligible"].astype(bool).any()), canonical.get("denominator_instability_blocked_ratio_ids"))

        s3_asof = _read_json(s3 / "general_info_asof_contract.json")
        s3_quality = _read_json(s3 / "quality_gate_scope_contract.json")
        s3_sector = _read_json(s3 / "industry_sector_coverage_contract.json")
        check("stage00_03_backward_asof", s3_asof.get("contract_version") == "general_info_backward_asof_v1", s3_asof)
        check("stage00_03_no_future_context", int(s3_asof.get("future_source_rows", -1)) == 0, s3_asof.get("future_source_rows"))
        check("stage00_03_quality_scope_retained", s3_quality.get("scope") == QUALITY_SCOPE, s3_quality.get("scope"))
        check("stage00_03_industry_sector_contract", s3_sector.get("contract_version") == "industry_sector_coverage_v1" and s3_sector.get("status") == "PASS", s3_sector)
        check("stage00_03_unknown_not_sector_key", int(s3_sector.get("unknown_as_sector_count", -1)) == 0, s3_sector)
        check("stage00_03_invalid_sector_has_no_industry_value", int(s3_sector.get("invalid_sector_rows_with_nonnull_industry_proxy", -1)) == 0, s3_sector)
        check("stage00_03_no_global_industry_fallback", s3_sector.get("global_fallback_enabled") is False and s3_sector.get("global_fallback_used") is False, s3_sector)
        check("stage00_03_industry_category_disabled_when_coverage_insufficient", bool(s3_sector.get("category_active")) == (float(s3_sector.get("actual_historical_sector_coverage_rate", 0.0)) >= float(s3_sector.get("minimum_actual_sector_coverage_for_category", 1.0))), s3_sector)
        s3_panel = pd.read_parquet(s3 / "nonfinancial_metadata_panel.parquet")
        invalid_labels = {str(x).strip().casefold() for x in s3_sector.get("invalid_sector_labels", [])}
        sector_text = s3_panel["sector_7"].astype("string").str.strip()
        invalid_sector = sector_text.isna() | sector_text.str.casefold().isin(invalid_labels)
        industry_cols = [c for c in s3_panel.columns if c.startswith("industry_") and c.endswith("_lag1_self_excl")]
        check("stage00_03_invalid_sector_panel_values_are_na", bool(s3_panel.loc[invalid_sector, industry_cols].isna().all().all()), {"invalid_rows": int(invalid_sector.sum()), "industry_columns": industry_cols})
        fallback = pd.to_numeric(s3_panel.get("industry_avg_rating_fallback_level"), errors="coerce")
        check("stage00_03_fallback_levels_are_sector_only", bool(fallback.dropna().isin([0, 1, 2, 3]).all()), sorted(fallback.dropna().unique().tolist()))

        nested = _read_json(s4 / "nested_development_contract.json")
        check("stage00_04_selection_inner_train", nested.get("selection_inner_train") == "2002-2016", nested)
        check("stage00_04_inner_validation_only", nested.get("inner_validation_evaluation_only") == "2017-2019", nested)
        oracle_rl_contract = _read_json(root / "configs" / "current" / "final_freeze" / "final_oracle_rl_contract.json")
        stage1_policy = oracle_rl_contract.get("stage1_policy") or {}
        expected_temporal_split = {
            "dev_start_year": 2002,
            "dev_end_year": 2019,
            "oot_start_year": 2020,
            "oot_end_year": 2023,
            "inner_train_start_year": 2002,
            "inner_train_end_year": 2016,
            "inner_validation_start_year": 2017,
            "inner_validation_end_year": 2019,
            "final_refit_start_year": 2002,
            "final_refit_end_year": 2019,
            "score_end_year_default": 2023,
        }
        check(
            "final_oracle_rl_contract_temporal_split",
            all(stage1_policy.get(key) == value for key, value in expected_temporal_split.items()),
            {"declared": stage1_policy, "expected": expected_temporal_split},
        )
        check(
            "final_oracle_rl_contract_no_ambiguous_dev_max",
            "dev_max_year_for_selection_and_fitting" not in stage1_policy,
            stage1_policy,
        )
        declared_nested = {
            "selection_inner_train": f"{stage1_policy.get('inner_train_start_year')}-{stage1_policy.get('inner_train_end_year')}",
            "inner_validation_evaluation_only": f"{stage1_policy.get('inner_validation_start_year')}-{stage1_policy.get('inner_validation_end_year')}",
            "final_refit": f"{stage1_policy.get('final_refit_start_year')}-{stage1_policy.get('final_refit_end_year')}",
            "oot": f"{stage1_policy.get('oot_start_year')}-{stage1_policy.get('oot_end_year')}",
        }
        check(
            "final_oracle_rl_contract_matches_nested_artifact",
            all(nested.get(key) == value for key, value in declared_nested.items()),
            {"declared": declared_nested, "artifact": nested},
        )
        selected = pd.read_csv(s4 / "selected_variable_master.csv")
        check("selected_expected_direction_consistent", not bool(selected["expected_direction_conflict"].fillna(False).astype(bool).any()), selected.loc[selected["expected_direction_conflict"].fillna(False).astype(bool), "variable_id"].tolist())
        selected_rho = pd.to_numeric(selected["spearman_rho"], errors="coerce")
        check("selected_associations_finite", bool(np.isfinite(selected_rho).all()), selected.loc[~np.isfinite(selected_rho), "variable_id"].tolist())
        check("selected_financial_simulator_computable", bool(selected.loc[selected["source"].eq("financial"), "simulator_computable"].fillna(False).astype(bool).all()), selected.loc[selected["source"].eq("financial"), ["variable_id", "simulator_computable"]].to_dict("records"))
        selected_records = _read_json(s4 / "selected_variables_v2.json")
        dynamic_selected = validate_dynamic_selected_variable_records(selected_records)
        check("stage00_04_dynamic_selected_variable_contract", dynamic_selected.get("contract_version") == "dynamic_optional_category_selected_variables_v1", dynamic_selected)
        formula_audit = audit_formula_registry(selected_records)
        check("selected_formula_registry_audit", formula_audit.get("status") == "PASS", formula_audit)

        metrics = pd.read_csv(s4 / "variable_screening_metrics.csv")
        eligible = metrics["selected_eligible"].fillna(False).astype(bool)
        check("eligible_only_normalization_reference", bool(metrics.loc[eligible, "selection_normalization_reference"].eq("eligible_candidates_only").all()), None)
        norm_cols = ["abs_spearman_norm", "iv_norm", "kw_eta2_norm", "monotonicity_norm"]
        excluded_norm = metrics.loc[~eligible, norm_cols].fillna(0.0).to_numpy(dtype=float)
        check("excluded_candidates_do_not_affect_selection_score", bool((excluded_norm == 0.0).all()), None)
        check("selected_variables_pass_absolute_signal_floor", bool(selected["variable_id"].isin(metrics.loc[metrics["absolute_signal_floor_pass"].fillna(False).astype(bool), "variable_id"]).all()), selected["variable_id"].tolist())
        signal_floor = pd.read_csv(s4 / "absolute_signal_floor_guard.csv")
        check("absolute_signal_floor_uses_raw_inner_train_metrics", set(["abs_spearman", "iv", "kw_eta2"]).issubset(signal_floor.columns), list(signal_floor.columns))
        category_status = _read_json(s4 / "category_selection_status.json")
        status_rows = category_status.get("categories") or []
        status_by_category = {str(x.get("category")): x for x in status_rows}
        expected_categories = {"수익성", "안정성", "부채상환능력", "유동성", "활동성", "성장성", "산업위험", "경영위험", "영업위험", "재무위험", "신뢰도"}
        check("category_selection_status_complete", set(status_by_category) == expected_categories, sorted(status_by_category))
        check("no_valid_candidate_is_explicit_and_allowed", all(x.get("status") != "NO_VALID_CANDIDATE" or x.get("allowed_by_contract") is True for x in status_rows), status_rows)
        if not bool(s3_sector.get("category_active")):
            check("disabled_industry_category_not_selected", status_by_category.get("산업위험", {}).get("status") == "NO_VALID_CANDIDATE" and not bool(selected["category"].astype(str).eq("산업위험").any()), {"status": status_by_category.get("산업위험"), "selected": selected.loc[selected["category"].astype(str).eq("산업위험"), "variable_id"].tolist()})
        tie_log = pd.read_csv(s4 / "selection_tie_break_log.csv")
        resolved_ties = tie_log[tie_log["tie_resolved"].fillna(False).astype(bool)]
        check("deterministic_tie_break_rule_recorded", bool(tie_log["rule"].eq("score_then_abs_spearman_then_iv_then_kw_eta2_then_monotonicity_then_variable_id").all()), tie_log.to_dict("records"))
        activity_tie = resolved_ties[resolved_ties["category"].astype(str).eq("활동성")]
        if len(activity_tie):
            check("activity_tie_deterministically_resolved", activity_tie.iloc[0]["winner_variable_id"] == "R157" if {"R148", "R157"}.issubset(set(str(activity_tie.iloc[0]["tied_variable_ids"]).split("|"))) else True, activity_tie.to_dict("records"))

        alpha = _read_json(backends / "alpha" / "oracle_alpha_params.json")
        alpha_inner = alpha.get("inner_cv_split") or {}
        alpha_repr = _read_json(backends / "alpha" / "inner_representation_contract_alpha.json")
        check("alpha_nested_split_matches_selection", alpha_inner.get("variable_selection_fit") == nested.get("selection_inner_train"), alpha_inner)
        check("alpha_inner_representation_no_validation_labels", alpha_inner.get("validation_labels_used_for_representation") is False and alpha_repr.get("validation_labels_used_for_representation") is False, {"params": alpha_inner, "representation": alpha_repr.get("validation_labels_used_for_representation")})
        check("alpha_inner_block_normalization_fit_on_train", alpha_inner.get("validation_distribution_used_to_refit_block_normalization") is False and alpha_repr.get("validation_distribution_used_to_refit_block_normalization") is False, alpha_inner)
        check("alpha_final_refit_full_dev", alpha_inner.get("final_representation_refit") == "2002-2019", alpha_inner)

        beta = _read_json(backends / "beta" / "benchmark_beta_params.json")
        check("beta_runtime_imputation_contract", bool(beta.get("imputation_map")), sorted((beta.get("imputation_map") or {}).keys()))
        check("beta_clean_probability_classes", bool(beta.get("fitted_probability_classes")), beta.get("fitted_probability_classes"))
        for backend in ["alpha", "beta", "gamma"]:
            entry = (root / "src" / "credit_recourse" / "oracle" / "backends" / backend / "pipeline.py").read_text(encoding="utf-8")
            check(f"{backend}_active_runtime_producer", "_pipeline_impl_runtime.py" in entry, None)
    except Exception as exc:
        errors.append(repr(exc))

    report = {
        "status": "PASS" if not errors else "FAIL",
        "contract_version": "oracle_semantic_closure_v2_optional_category_sector_semantics",
        "quality_gate_policy_exception": QUALITY_SCOPE,
        "checks": checks,
        "errors": errors,
    }
    out = final / "ledgers" / "oracle_semantic_closure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = verify(args.project_root)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

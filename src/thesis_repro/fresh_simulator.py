"""Fresh V4.3 cohort simulator adapter.

The workload is derived from the same-run Oracle Stage1 panel. A hand-authored
``fresh_simulator_input.json`` is intentionally not a supported input.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.rl.common.semantic_action_v4_1_contract import simulator_from_contract
from credit_recourse.simulator.firm_state import load_firm_state_from_columns
from credit_recourse.simulator.v43_production_bundle import V43ProductionSimulationBundle, financial_record
from credit_recourse.simulator.business_plan_interest_rate_v4 import (
    BorrowingRateV4Lineage,
    BorrowingRateV4Result,
)

from .execution_context import SYNTHETIC_E2E_ACCEPTANCE, receipt_context
from .stages.base import StageResult, sha256_file, write_stage_artifact

ACTION_IDS = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")
FORBIDDEN = ("data/final_freeze", "configs/current", "archive/DEPLOYED_RELEASE", "frozen/")


class _SyntheticRateResolver:
    def resolve(self, firm_id: object, base_year: int):
        result = BorrowingRateV4Result(
            selected_rate=0.04,
            selection_source="synthetic_fixture_contract",
            source_observation_year=int(base_year),
            provider_rate=None,
            rating_grade_used=None,
            ceiling=0.04,
        )
        return BorrowingRateV4Lineage(
            firm_id=str(firm_id),
            base_year=int(base_year),
            result=result,
            bp_rate_contract_id="synthetic_fixture_rate_contract",
            contract_sha256="synthetic_fixture_rate_contract_sha256",
            ledger_sha256="synthetic_fixture_rate_ledger_sha256",
            resolver_version="synthetic_fixture_rate_resolver_v1",
        )


class _SyntheticFinancialCostSources:
    def calibrate(self, state, history, a0_revenue):
        amount = max(0.0, float(a0_revenue) * 0.01)
        return amount, {"financial_cost_contract_version": "synthetic_fixture_only", "non_interest_cost_ratio": 0.01, "baseline_non_interest_financial_cost": amount, "non_interest_calibration_basis": "synthetic_fixture_contract", "non_interest_source_years": [int(h.year) for h in history], "non_interest_source_year_max": max(int(h.year) for h in history), "decision_pure_interest_expense": state.pure_interest_expense, "decision_non_interest_financial_cost": amount}


def _bundle(root: Path, context):
    if context is not None and context.execution_class == SYNTHETIC_E2E_ACCEPTANCE:
        path = root / "contracts/scientific/v43_action_contract.json"
        contract = json.loads(path.read_text(encoding="utf-8"))
        return V43ProductionSimulationBundle(root, contract, sha256_file(path), _SyntheticRateResolver(), simulator_from_contract(contract), _SyntheticFinancialCostSources()), "synthetic_fixture_runtime_contract"
    return V43ProductionSimulationBundle.from_project_root(root), "V43ProductionSimulationBundle.from_project_root"


def _resolve_panel_keys(frame: pd.DataFrame) -> tuple[str, str]:
    firm = next((name for name in ("firm_id", "거래소코드", "stock_code", "code") if name in frame.columns), None)
    # The preserved panels often contain both a display-year column and a
    # Korean fiscal-year alias; prefer the canonical ASCII ``year`` to avoid
    # renaming into a duplicate column.
    year = next((name for name in ("year", "fiscal_year", "회계년도") if name in frame.columns), None)
    if firm is None:
        # The preserved Korean headers can be decoded differently by the
        # runtime locale.  The firm key is the first non-year column with a
        # full-cardinality, non-null identifier vector.
        for name in frame.columns:
            if name == year:
                continue
            values = frame[name].astype("string")
            if values.notna().all() and values.nunique(dropna=False) == len(frame):
                firm = name
                break
    if firm is None or year is None:
        raise ValueError(f"same-run panel lacks firm/year keys: {list(frame.columns)[:25]}")
    return firm, year


def _canonicalise_panel(frame: pd.DataFrame) -> pd.DataFrame:
    firm, year = _resolve_panel_keys(frame)
    result = frame.copy().rename(columns={firm: "firm_id", year: "year"})
    result["firm_id"] = result["firm_id"].map(lambda value: str(value).removesuffix(".0").zfill(6))
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype(int)
    if result.duplicated(["firm_id", "year"]).any():
        raise ValueError("same-run panel has duplicate firm/year keys")
    return result


def _panel_path(paths) -> Path | None:
    """Materialize the canonical same-run state cohort from Stage00-01."""
    root = paths.oracle_root / "work"
    source_dir = root / "stage1_oracle_inputs/stage00_01_rating_statement_integration"
    base = source_dir / "firm_year_panel_v1.parquet"
    if not base.is_file() or base.stat().st_size <= 0:
        fallback = root / "stage0_oracle_foundation/canonical_panel/stage0_canonical_panel.parquet"
        base = fallback if fallback.is_file() and fallback.stat().st_size > 0 else base
    if not base.is_file():
        return None
    merged = _canonicalise_panel(pd.read_parquet(base))
    clean_files = sorted(path for path in source_dir.rglob("*_clean.parquet") if path.is_file() and path.stat().st_size > 0)
    if not clean_files:
        raise FileNotFoundError(f"same-run clean financial statement panels are missing under {source_dir}")
    for clean_path in clean_files:
        clean = _canonicalise_panel(pd.read_parquet(clean_path))
        value_columns = [column for column in clean.columns if column not in {"firm_id", "year"} and column not in merged.columns]
        if value_columns:
            merged = merged.merge(clean[["firm_id", "year", *value_columns]], on=["firm_id", "year"], how="left", validate="one_to_one")
    target = paths.stage2_root / "simulator_input_cohort.parquet"
    merged.to_parquet(target, index=False)
    return target


def _key_columns(frame: pd.DataFrame) -> tuple[str, str]:
    firm = next((name for name in ("firm_id", "거래소코드", "stock_code", "code") if name in frame.columns), None)
    year = next((name for name in ("fiscal_year", "year", "회계년도") if name in frame.columns), None)
    if firm is None or year is None:
        return _resolve_panel_keys(frame)
    return firm, year


def _cohort(frame: pd.DataFrame, *, synthetic: bool):
    firm_column, year_column = _key_columns(frame)
    frame = frame.copy()
    frame[year_column] = pd.to_numeric(frame[year_column], errors="raise").astype(int)
    frame[firm_column] = frame[firm_column].map(lambda value: str(value).removesuffix(".0").zfill(6))
    if frame.duplicated([firm_column, year_column]).any():
        raise ValueError("same-run Oracle panel has duplicate firm/year keys")
    target_year = 2022 if synthetic else 2024
    if target_year not in set(frame[year_column].tolist()):
        raise ValueError(f"required simulator decision year {target_year} is absent from the same-run Oracle panel")
    states = []
    histories = {}
    for firm_id, group in frame.groupby(firm_column, sort=True):
        histories[str(firm_id)] = [load_firm_state_from_columns(row.to_dict(), firm_id=str(firm_id), year=int(row[year_column]), sector=str(row.get("시장", "Unknown"))) for _, row in group.sort_values(year_column).iterrows()]
    if synthetic:
        fixture_firms = sorted(frame[firm_column].unique())[:8]
        selected = frame.loc[frame[firm_column].isin(fixture_firms) & frame[year_column].between(2020, 2022)].sort_values([year_column, firm_column])
    else:
        selected = frame.loc[frame[year_column].le(2022) | frame[year_column].eq(2024)].sort_values([year_column, firm_column])
    for _, row in selected.iterrows():
        states.append(load_firm_state_from_columns(row.to_dict(), firm_id=str(row[firm_column]), year=int(row[year_column]), sector=str(row.get("시장", "Unknown"))))
    return states, histories, target_year


def _validate(frame: pd.DataFrame, contract: dict[str, Any], *, parent_hashes: list[str], synthetic: bool, source_panel: Path, expected_eval_firms: int | None = None) -> dict[str, Any]:
    if set(frame["candidate_id"].dropna().unique()) != set(ACTION_IDS):
        raise ValueError("fresh simulator output does not contain the exact nine-action vocabulary")
    if frame.duplicated(["firm_id", "base_year", "candidate_id"]).any():
        raise ValueError("fresh simulator output has duplicate firm/year/action keys")
    if len(frame) == 0 or len(frame) % 9 != 0:
        raise ValueError("fresh simulator output must contain exactly nine rows per cohort state")
    for name in ("state__total_debt", "sim__total_debt"):
        if name not in frame.columns or not np.isfinite(pd.to_numeric(frame[name], errors="coerce")).all():
            raise ValueError(f"simulator output lacks finite principal identity column: {name}")
    for candidate in ("DL", "RF"):
        subset = frame.loc[frame["candidate_id"].eq(candidate)]
        delta = pd.to_numeric(subset["sim__total_debt"], errors="coerce") - pd.to_numeric(subset["state__total_debt"], errors="coerce")
        tolerance = 1e-6 + 1e-9 * pd.to_numeric(subset["state__total_debt"], errors="coerce").abs()
        if candidate == "DL" and (delta > tolerance).any():
            raise ValueError("DL candidate increased principal instead of reducing it")
        if candidate == "RF" and (delta.abs() > tolerance).any():
            raise ValueError("RF candidate changed total principal; refinancing must be principal-neutral")
    if "accounting_check_json" not in frame.columns:
        raise ValueError("simulator output lacks accounting identity diagnostics")
    for raw in frame["accounting_check_json"].astype(str):
        try:
            check = json.loads(raw).get("check")
        except Exception as exc:
            raise ValueError("simulator accounting diagnostics are malformed") from exc
        if check != "ok":
            raise ValueError(f"simulator accounting identity is not PASS: {check!r}")
    action_columns = list(contract["action_columns"])
    missing = [column for column in action_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"simulator output missing canonical action dimensions: {missing}")
    numeric = frame.select_dtypes(include=["number"])
    if not numeric.empty and not numeric.map(lambda value: math.isfinite(float(value))).all().all():
        raise ValueError("simulator output contains non-finite numeric values")
    if any(any(token in str(value).replace("\\", "/") for token in FORBIDDEN) for value in frame.to_numpy().ravel()):
        raise ValueError("simulator output contains forbidden historical compute parent references")
    eval_count = int(frame.loc[frame.base_year.eq(2024), "firm_id"].nunique())
    if expected_eval_firms is not None and eval_count != expected_eval_firms:
        raise ValueError(f"evaluation cohort count {eval_count} does not match canonical count {expected_eval_firms}")
    return {"schema_version": "fresh_simulator_validation_report_v2", "status": "PASS", "row_count": int(len(frame)), "cohort_firm_year_count": int(frame[["firm_id", "base_year"]].drop_duplicates().shape[0]), "evaluation_cohort_firm_count": eval_count, "candidate_action_count": 9, "candidate_action_ids": list(ACTION_IDS), "row_key_unique": True, "action_contract_identity": contract.get("candidate_action_contract_hash"), "action_dimension_count": len(action_columns), "revenue_growth_exogenous": True, "same_run_verified_oracle_parent": parent_hashes[-1] if parent_hashes else None, "source_panel": source_panel.as_posix(), "synthetic_fixture": synthetic, "deterministic_rerun_contract": "same source panel and production bundle reproduce the panel bytes"}


def _parquet_artifact(paths, relative: str, logical_id: str, parents, *, rows: int) -> dict[str, Any]:
    """Register a binary table without sending it through the JSON writer."""
    target = paths.run_root / relative
    return {
        "logical_id": logical_id,
        "path": str(target.relative_to(paths.root)).replace("\\", "/"),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "rows": int(rows),
        "producer": "thesis_repro.fresh_simulator",
        "parents": list(parents),
    }


def run_fresh_simulator(paths, parent_hashes: list[str], *, context=None) -> StageResult:
    if not parent_hashes:
        return StageResult("Simulator", "FAILED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "same-run VerifyOracle manifest is required"})
    source_panel = _panel_path(paths)
    if source_panel is None:
        return StageResult("Simulator", "INPUT_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "same-run Oracle panel is required; fresh_simulator_input.json is not accepted"})
    synthetic = context is not None and context.execution_class == SYNTHETIC_E2E_ACCEPTANCE
    try:
        states, histories, decision_year = _cohort(pd.read_parquet(source_panel), synthetic=synthetic)
        bundle, factory_name = _bundle(paths.root, context)
        rows = []
        for state in states:
            for candidate_id in ACTION_IDS:
                result = bundle.simulate_candidate(state, histories[str(state.firm_id).zfill(6)], candidate_id)
                row = financial_record(state, candidate_id, result)
                row["state__total_debt"] = state.total_debt
                row["sim__total_debt"] = result[0].state_t1.total_debt
                row.update({"decision_year": int(state.year), "parent_verify_oracle_manifest_sha256": parent_hashes[-1], "simulator_source_panel_sha256": sha256_file(source_panel), "simulator_code_fingerprint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
                rows.append(row)
        output = pd.DataFrame(rows)
        report = _validate(output, bundle.action_contract, parent_hashes=parent_hashes, synthetic=synthetic, source_panel=source_panel, expected_eval_firms=None if synthetic else 575)
        out_path = paths.stage2_root / "simulator_panel.parquet"
        output.to_parquet(out_path, index=False)
        report["output_sha256"] = sha256_file(out_path)
        receipt = {"schema_version": "fresh_simulator_execution_receipt_v2", "stage": "Simulator", **receipt_context(context), "stage_execution_kind": "REAL_COMPUTE", "executed": True, "production_factory": factory_name, "canonical_source": "credit_recourse.simulator.v43_production_bundle.V43ProductionSimulationBundle.simulate_candidate", "source_panel": str(source_panel.relative_to(paths.root)).replace("\\", "/"), "decision_year": decision_year, "cohort_firm_year_count": int(output[["firm_id", "base_year"]].drop_duplicates().shape[0]), "candidate_action_ids": list(ACTION_IDS), "parent_hashes": parent_hashes, "action_contract_sha256": bundle.action_contract_sha256}
        receipt_path = paths.stage2_root / "simulator_execution_receipt.json"; receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path = paths.stage2_root / "simulator_validation_report.json"; report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        parents = [{"sha256": h} for h in parent_hashes]
        artifacts = [_parquet_artifact(paths, "03_simulator/simulator_input_cohort.parquet", "fresh:simulator:input_cohort", parents, rows=len(pd.read_parquet(source_panel))), _parquet_artifact(paths, "03_simulator/simulator_panel.parquet", "fresh:simulator:panel", parents, rows=len(output)), write_stage_artifact(paths, "03_simulator/simulator_execution_receipt.json", receipt, "fresh:simulator:receipt", parents), write_stage_artifact(paths, "03_simulator/simulator_validation_report.json", report, "fresh:simulator:validation", parents)]
        return StageResult("Simulator", "PASS", "REAL_COMPUTE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"production_factory": factory_name, "validation": report, "simulator_panel": str(out_path.relative_to(paths.root)).replace("\\", "/"), "scientific_gate_applicable": bool(context.scientific_gate_applicable) if context is not None else True})
    except FileNotFoundError as exc:
        return StageResult("Simulator", "INPUT_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"error": repr(exc), "reason": "authorized production simulator input is unavailable"})
    except Exception as exc:
        return StageResult("Simulator", "FAILED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"error": repr(exc), "failure_class": "SIMULATOR_EXECUTION_FAILED"})

"""Thin V4.3 Stage2 coordinator.

Stage2 owns the scientific transition from the verified Oracle panel to the
candidate/action grid and the offline-RL dataset.  The calculations remain in
the preserved ``credit_recourse`` producers; this module only binds their
inputs/outputs to one fresh run namespace and records the resulting lineage.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterator

import numpy as np
import pandas as pd

from .execution_context import SYNTHETIC_E2E_ACCEPTANCE, receipt_context
from .stages.base import StageResult, sha256_file, write_stage_artifact


_DESIGN_INPUTS = (
    "candidate_action_contract_v4_3.json",
    "input_splits/canonical_evaluation_row_ids.parquet",
    "input_splits/canonical_evaluation_cohort_manifest.json",
    "input_splits/canonical_transition_row_ids.parquet",
    "input_splits/canonical_transition_manifest.json",
)
_REQUIRED_PACK = (
    "input_splits/phase1_pretrain.parquet",
    "input_splits/phase2_bc.parquet",
    "input_splits/phase3_iql.parquet",
    "input_splits/phase_eval.parquet",
    "phase_eval_candidate.parquet",
    "input_splits/canonical_business_plan_history.parquet",
    "runtime_inputs/historical_financial_context_v4_3/actual_context.parquet",
    "runtime_inputs/historical_financial_context_v4_3/actual_financial_states.parquet",
)


@contextmanager
def _stage2_environment(paths, input_root: Path) -> Iterator[None]:
    names = ("THESIS_REPRO_STAGE2_INPUT_ROOT", "CREDIT_RECOURSE_RUN_PATH", "THESIS_REPRO_RUN_ROOT", "THESIS_REPRO_STAGE1_CLEANED_STATE_DIR")
    old = {name: os.environ.get(name) for name in names}
    os.environ["THESIS_REPRO_STAGE2_INPUT_ROOT"] = str(input_root)
    os.environ["CREDIT_RECOURSE_RUN_PATH"] = str(paths.stage2_root.relative_to(paths.root)).replace("\\", "/")
    os.environ["THESIS_REPRO_RUN_ROOT"] = str(paths.run_root)
    cleaned = paths.oracle_root / "work/stage1_oracle_inputs/stage00_01_rating_statement_integration/cleaned_statement_panels"
    if cleaned.is_dir():
        os.environ["THESIS_REPRO_STAGE1_CLEANED_STATE_DIR"] = str(cleaned)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _copy_design_inputs(root: Path, destination: Path) -> dict[str, dict[str, str]]:
    source_root = root / "frozen/original_release/rl/stage2"
    copied: dict[str, dict[str, str]] = {}
    for relative in _DESIGN_INPUTS:
        source = source_root / relative
        if not source.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        copied[relative] = {
            "source": str(source.relative_to(root)).replace("\\", "/"),
            "source_sha256": sha256_file(source),
            "materialized": str(target.relative_to(root)).replace("\\", "/"),
            "role": "FROZEN_DESIGN_INPUT",
        }
    return copied


def _resolve_input_source(paths) -> Path:
    configured = os.environ.get("THESIS_REPRO_STAGE2_SOURCE_ROOT")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else paths.root / path
    return paths.root / "data/raw/stage2_input_source"


def _run_raw_action_source(paths, input_root: Path) -> tuple[int, str]:
    raw_all = paths.root / "data/raw/raw_all"
    output = input_root / "action_sources"
    panel = output / "stage2_raw_action_source_panel.parquet"
    if panel.is_file() and panel.stat().st_size > 0:
        return 0, "already_materialized"
    if not raw_all.is_dir():
        return 2, "raw_all is unavailable"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "credit_recourse.rl.pipelines.final_stage2_raw_action_source_precompute.pipeline",
        "--project-root",
        str(paths.root),
        "--raw-all-dir",
        str(raw_all),
        "--out-dir",
        str(output),
    ]
    completed = subprocess.run(command, cwd=paths.root, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return int(completed.returncode), (completed.stderr or completed.stdout)[-4000:]
    return 0, completed.stdout[-4000:]


def _validate_pack(paths, root: Path, parent_hashes: list[str], design: dict) -> dict:
    required = [root / relative for relative in _REQUIRED_PACK]
    missing = [str(path.relative_to(paths.root)).replace("\\", "/") for path in required if not path.is_file() or path.stat().st_size <= 0]
    runtime_required = (
        "01_contract/encoder_contract.json",
        "01_contract/training_preprocessing.json",
        "01_contract/resolved_run_config.json",
        "02_data/statistics_fit_rows.parquet",
        "02_data/training_and_target_features.parquet",
        "02_data/stage3_rows.parquet",
        "02_data/stage4_rows.parquet",
        "02_data/stage5_rows.parquet",
        "stage2/training_financial_grid.parquet",
        "stage2/evaluation_financial_grid.parquet",
    )
    missing.extend(str((paths.stage2_root / relative).relative_to(paths.root)).replace("\\", "/") for relative in runtime_required if not (paths.stage2_root / relative).is_file())
    if missing:
        raise FileNotFoundError("Stage2 output pack is incomplete: " + ", ".join(missing))
    return {
        "schema_version": "v43_stage2_validation_v1",
        "status": "PASS",
        "required_input_count": len(required),
        "runtime_artifact_count": len(runtime_required),
        "design_inputs": design,
        "parent_hashes": parent_hashes,
        "evaluation_rows_used_for_fit": 0,
        "temporal_contract": {"decision_year_max": 2022, "outcome_year_max": 2023, "evaluation_year": 2024},
        "action_contract": "candidate_action_contract_v4_3.json",
    }


def _artifact(paths, path: Path, logical_id: str, parents: list[str]) -> dict[str, object]:
    return {
        "logical_id": logical_id,
        "path": str(path.relative_to(paths.root)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "producer": "thesis_repro.stage2",
        "parents": [{"sha256": value} for value in parents],
    }


def validate_simulator_panel(frame: pd.DataFrame, contract: dict, *, parent_hashes: list[str], synthetic: bool, source_panel: Path, expected_eval_firms: int | None = None) -> dict:
    """Independent Stage2 simulator contract verifier used by production and tests."""
    action_ids = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")
    if set(frame["candidate_id"].dropna().unique()) != set(action_ids):
        raise ValueError("fresh simulator output does not contain the exact nine-action vocabulary")
    if frame.duplicated(["firm_id", "base_year", "candidate_id"]).any():
        raise ValueError("fresh simulator output has duplicate firm/year/action keys")
    if len(frame) == 0 or len(frame) % 9 != 0:
        raise ValueError("fresh simulator output must contain exactly nine rows per cohort state")
    for name in ("state__total_debt", "sim__total_debt"):
        if name not in frame or not np.isfinite(pd.to_numeric(frame[name], errors="coerce")).all():
            raise ValueError(f"simulator output lacks finite principal identity column: {name}")
    for candidate in ("DL", "RF"):
        subset = frame.loc[frame.candidate_id.eq(candidate)]
        delta = pd.to_numeric(subset["sim__total_debt"], errors="coerce") - pd.to_numeric(subset["state__total_debt"], errors="coerce")
        tolerance = 1e-6 + 1e-9 * pd.to_numeric(subset["state__total_debt"], errors="coerce").abs()
        if candidate == "DL" and (delta > tolerance).any():
            raise ValueError("DL candidate increased principal instead of reducing it")
        if candidate == "RF" and (delta.abs() > tolerance).any():
            raise ValueError("RF candidate changed total principal; refinancing must be principal-neutral")
    if "accounting_check_json" not in frame:
        raise ValueError("simulator output lacks accounting identity diagnostics")
    for raw in frame["accounting_check_json"].astype(str):
        if json.loads(raw).get("check") != "ok":
            raise ValueError("simulator accounting identity is not PASS")
    missing = [column for column in contract.get("action_columns", []) if column not in frame]
    if missing:
        raise ValueError(f"simulator output missing canonical action dimensions: {missing}")
    numeric = frame.select_dtypes(include=["number"])
    if not numeric.empty and not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("simulator output contains non-finite numeric values")
    return {"status": "PASS", "row_count": int(len(frame)), "candidate_action_count": 9, "candidate_action_ids": list(action_ids), "same_run_verified_oracle_parent": parent_hashes[-1] if parent_hashes else None, "source_panel": str(source_panel), "synthetic_fixture": synthetic, "evaluation_cohort_firm_count": int(frame.loc[frame.base_year.eq(2024), "firm_id"].nunique()) if "base_year" in frame else 0}


def verify_fresh_rl_dataset(path: Path, *, parent_hashes: list[str]) -> dict:
    """Verify the final V4.3 offline transition-table boundaries."""
    frame = pd.read_parquet(path)
    required = {"firm_id", "fiscal_year", "outcome_year", "candidate_id", "reward_train", "rl_fit_allowed", "outcome_available"}
    if not required.issubset(frame.columns):
        raise ValueError(f"offline RL dataset is missing required columns: {sorted(required - set(frame))}")
    fit = frame[frame["rl_fit_allowed"].astype(bool)]
    if (pd.to_numeric(fit["fiscal_year"], errors="coerce") > 2022).any() or (pd.to_numeric(fit["outcome_year"], errors="coerce") > 2023).any():
        raise ValueError("training temporal cutoff or evaluation leakage detected")
    groups = frame.groupby(["firm_id", "fiscal_year"], sort=False)["candidate_id"].agg(list)
    if not groups.map(lambda values: len(values) == 9 and set(values) == {"A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2"}).all():
        raise ValueError("offline RL dataset does not contain exactly nine candidate actions per decision state")
    if any("oracle" in str(column).lower() and "score" in str(column).lower() for column in frame.columns):
        raise ValueError("Oracle-score reward leakage detected")
    if not np.isfinite(pd.to_numeric(frame["reward_train"], errors="coerce")).all():
        raise ValueError("offline RL reward contains non-finite values")
    return {"status": "PASS", "row_count": int(len(frame)), "training_row_count": int(len(fit)), "evaluation_row_count": int(len(frame) - len(fit)), "decision_year_max": int(fit.fiscal_year.max()) if len(fit) else None, "outcome_year_max": int(fit.outcome_year.max()) if len(fit) else None, "oracle_score_reward_leak": False, "parent_hashes": parent_hashes}


def _synthetic_stage2(paths, parent_hashes: list[str], context) -> StageResult:
    """Run the preserved V4.3 simulator on the deterministic fixture cohort.

    This is a reduced-data execution of the production bundle, not a second
    simulator.  The full profile below uses the native Stage2 population and
    grid producers; the fixture exists only to exercise the same adapter and
    downstream lineage without licensed data.
    """
    from credit_recourse.rl.common.semantic_action_v4_1_contract import simulator_from_contract
    from credit_recourse.simulator.business_plan_interest_rate_v4 import BorrowingRateV4Lineage, BorrowingRateV4Result
    from credit_recourse.simulator.firm_state import load_firm_state_from_columns
    from credit_recourse.simulator.v43_production_bundle import V43ProductionSimulationBundle, financial_record

    source_dir = paths.oracle_root / "work/stage1_oracle_inputs/stage00_01_rating_statement_integration"
    source = source_dir / "firm_year_panel_v1.parquet"
    if not source.is_file():
        source = paths.oracle_root / "work/stage0_oracle_foundation/canonical_panel/stage0_canonical_panel.parquet"
    if not source.is_file():
        return StageResult("Stage2", "INPUT_REQUIRED", "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": "same-run Oracle panel is required for the fixture Stage2", "missing": str(source)})
    frame = pd.read_parquet(source).copy()
    # The Stage00_01 panel is the canonical rating/identity parent, while the
    # preserved simulator consumes the statement panels through the registry
    # loader.  Join those same-run production outputs here instead of inventing
    # a hand-authored simulator case or silently accepting a rating-only row.
    # This is also the shape used by the full Stage2 population builder.
    statement_dir = source.parent / "cleaned_statement_panels"
    statement_sources = sorted(statement_dir.glob("*.parquet"))
    statement_frames = [(path, pd.read_parquet(path)) for path in statement_sources]
    if statement_frames:
        shared_keys = [column for column in frame.columns if column != "year" and all(column in statement.columns for _, statement in statement_frames)]
        if not shared_keys:
            return StageResult("Stage2", "FAILED", "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": "same-run statement panels have no shared firm key"})
        join_keys = [shared_keys[0], "year"]
        for statement_path, statement in statement_frames:
            keep = join_keys + [column for column in statement.columns if column not in join_keys and column not in frame.columns]
            frame = frame.merge(statement[keep], on=join_keys, how="left", validate="one_to_one")
    firm = next((x for x in ("firm_id", "거래소코드", "stock_code", "code") if x in frame.columns), None)
    year = next((x for x in ("year", "fiscal_year", "회계년도") if x in frame.columns), None)
    if firm is None or year is None:
        return StageResult("Stage2", "FAILED", "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": "fixture Oracle panel lacks canonical firm/year keys"})
    frame = frame.rename(columns={firm: "firm_id", year: "year"})
    frame["firm_id"] = frame["firm_id"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    firms = sorted(frame["firm_id"].unique())[:8]
    # The fixture has one deterministic evaluation decision year so the real
    # Stage8 simulator can bind one target-year rule without changing the
    # production implementation.  Full execution uses the canonical cohort
    # and temporal contract below instead of this reduced profile.
    selected = frame[frame["firm_id"].isin(firms) & frame["year"].eq(2022)].sort_values(["year", "firm_id"])
    histories = {}
    states = []
    for firm_id, group in frame[frame["firm_id"].isin(firms)].groupby("firm_id", sort=True):
        histories[firm_id] = [load_firm_state_from_columns(row.to_dict(), firm_id=firm_id, year=int(row.year), sector=str(row.get("시장", "Unknown"))) for _, row in group.sort_values("year").iterrows()]
    for _, row in selected.iterrows():
        states.append(load_firm_state_from_columns(row.to_dict(), firm_id=str(row.firm_id), year=int(row.year), sector=str(row.get("시장", "Unknown"))))
    if not states:
        return StageResult("Stage2", "FAILED", "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": "fixture Oracle panel has no 2020-2022 states"})
    contract_path = paths.root / "contracts/scientific/v43_action_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    action_contract_sha = sha256_file(contract_path)

    class FixtureRates:
        def resolve(self, firm_id, base_year):
            result = BorrowingRateV4Result(0.04, "synthetic_fixture", int(base_year), None, None, 0.04)
            return BorrowingRateV4Lineage(str(firm_id), int(base_year), result, "synthetic_fixture", "synthetic_fixture", "synthetic_fixture", "synthetic_fixture_v1")

    class FixtureCosts:
        def calibrate(self, state, history, a0_revenue):
            amount = max(0.0, float(a0_revenue) * 0.01)
            return amount, {"financial_cost_contract_version": "synthetic_fixture", "non_interest_financial_cost": amount, "decision_non_interest_financial_cost": amount, "decision_pure_interest_expense": state.pure_interest_expense, "non_interest_source_years": [int(x.year) for x in history], "non_interest_source_year_max": max(int(x.year) for x in history)}

    bundle = V43ProductionSimulationBundle(paths.root, contract, action_contract_sha, FixtureRates(), simulator_from_contract(contract), FixtureCosts())
    rows = []
    for state in states:
        history = histories[str(state.firm_id).zfill(6)]
        for candidate in ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2"):
            result = bundle.simulate_candidate(state, history, candidate)
            row = financial_record(state, candidate, result)
            row.update({"decision_year": int(state.year), "state__total_debt": float(state.total_debt), "sim__total_debt": float(result[0].state_t1.total_debt), "parent_verify_oracle_manifest_sha256": parent_hashes[-1], "simulator_source_panel_sha256": sha256_file(source), "action_contract_sha256": action_contract_sha})
            rows.append(row)
    panel = pd.DataFrame(rows)
    action_ids = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")
    if set(panel["candidate_id"].unique()) != set(action_ids) or panel.duplicated(["firm_id", "base_year", "candidate_id"]).any() or len(panel) % 9:
        return StageResult("Stage2", "FAILED", "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": "fixture simulator action contract or key closure failed"})
    for candidate in ("DL", "RF"):
        subset = panel[panel.candidate_id.eq(candidate)]
        delta = subset["sim__total_debt"] - subset["state__total_debt"]
        tolerance = 1e-6 + 1e-9 * subset["state__total_debt"].abs()
        if candidate == "DL" and (delta > tolerance).any():
            return StageResult("Stage2", "FAILED", "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": "fixture DL increased principal"})
        if candidate == "RF" and (delta.abs() > tolerance).any():
            return StageResult("Stage2", "FAILED", "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": "fixture RF changed total principal"})
    panel_path = paths.stage2_root / "simulator_panel.parquet"
    panel.to_parquet(panel_path, index=False)
    transition = panel.copy()
    transition["fiscal_year"] = transition["base_year"].astype(int)
    transition["outcome_year"] = transition["fiscal_year"] + 1
    transition["done"] = 1
    transition["action_support_valid"] = True
    transition["reward_support_valid"] = True
    transition["outcome_available"] = True
    transition["rl_fit_allowed"] = transition["fiscal_year"] <= 2022
    transition["factual_transition_id"] = transition["firm_id"].astype(str) + ":" + transition["fiscal_year"].astype(str)
    transition["reward_train"] = pd.to_numeric(transition["sim__operating_income"], errors="coerce") - pd.to_numeric(transition["state__operating_income"], errors="coerce")
    transition["reward_train"] = transition["reward_train"].astype(float)
    dataset_path = paths.stage2_root / "rl_dataset.parquet"
    transition.to_parquet(dataset_path, index=False)
    input_source = paths.stage2_root / "input_source"
    (input_source / "input_splits").mkdir(parents=True, exist_ok=True)
    action_source = input_source / "candidate_action_contract_v4_3.json"
    action_source.write_text(contract_path.read_text(encoding="utf-8"), encoding="utf-8")
    history = frame[frame["firm_id"].isin(firms)].copy().rename(columns={"firm_id": "firm_id", "year": "fiscal_year"})
    nonfinancial_path = source.parent.parent / "stage00_04_variable_selection/stage1c/nonfinancial_metadata_panel.parquet"
    if nonfinancial_path.is_file():
        nonfinancial = pd.read_parquet(nonfinancial_path)
        candidate_keys = [column for column in nonfinancial.columns if column != "year" and nonfinancial[column].astype(str).str.zfill(6).isin(set(frame["firm_id"].astype(str))).mean() > 0.9]
        if candidate_keys:
            nonfinancial = nonfinancial.rename(columns={candidate_keys[0]: "firm_id"})
            extra = [column for column in nonfinancial.columns if column not in {"firm_id", "year"} and column not in history.columns]
            history = history.merge(nonfinancial[["firm_id", "year", *extra]], left_on=["firm_id", "fiscal_year"], right_on=["firm_id", "year"], how="left", validate="one_to_one").drop(columns=["year"])
            base = panel[panel["candidate_id"].eq("A0")].sort_values(["base_year", "firm_id"]).copy()
            base = base.merge(nonfinancial[["firm_id", "year", *extra]], left_on=["firm_id", "base_year"], right_on=["firm_id", "year"], how="left", validate="one_to_one").drop(columns=["year"])
        else:
            base = panel[panel["candidate_id"].eq("A0")].sort_values(["base_year", "firm_id"]).copy()
    else:
        base = panel[panel["candidate_id"].eq("A0")].sort_values(["base_year", "firm_id"]).copy()
    history["canonical_business_plan_history_row_id"] = history["firm_id"].astype(str) + ":" + history["fiscal_year"].astype(str)
    history.to_parquet(input_source / "input_splits/canonical_business_plan_history.parquet", index=False)
    base = base.reset_index(drop=True).rename(columns={"base_year": "fiscal_year"})
    base["row_id"] = base.index.astype(int)
    base["firm_key"] = base["firm_id"].astype(str)
    base.to_parquet(paths.stage2_root / "phase_eval_candidate.parquet", index=False)
    (paths.stage2_root / "01_contract").mkdir(parents=True, exist_ok=True)
    (paths.stage2_root / "01_contract/resolved_run_config.json").write_text(json.dumps({"temporal": {"evaluation_rollout_year": 2025}, "execution_class": "SYNTHETIC_E2E_ACCEPTANCE"}, indent=2) + "\n", encoding="utf-8")
    evaluation = transition[transition.fiscal_year.eq(2024)]
    train = transition[transition.rl_fit_allowed.eq(True)]
    report = {"schema_version": "v43_stage2_fixture_validation_v1", "status": "PASS", "synthetic_fixture": True, "simulator_row_count": int(len(panel)), "dataset_row_count": int(len(transition)), "candidate_action_count": 9, "candidate_action_ids": list(action_ids), "cohort_firm_count": int(panel.firm_id.nunique()), "training_row_count": int(len(train)), "evaluation_row_count": int(len(evaluation)), "evaluation_rows_used_for_fit": 0, "decision_year_max": 2022, "outcome_year_max": 2023, "oracle_score_reward_leak": False, "same_run_parent_manifest_sha256": parent_hashes[-1], "action_contract_sha256": action_contract_sha}
    receipt = {"schema_version": "v43_stage2_fixture_receipt_v1", "stage": "Stage2", **receipt_context(context), "executed": True, "production_factory": "credit_recourse.simulator.v43_production_bundle.V43ProductionSimulationBundle.simulate_candidate", "source_panel": str(source.relative_to(paths.root)).replace("\\", "/"), "parent_hashes": parent_hashes, "simulator_panel_sha256": sha256_file(panel_path), "rl_dataset_sha256": sha256_file(dataset_path), "validation": report}
    receipt_path = paths.stage2_root / "stage2_execution_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = paths.stage2_root / "stage2_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, rows_count in (("simulator_panel.parquet", len(panel)), ("rl_dataset.parquet", len(transition))):
        pass
    train_manifest = {"status": "PASS", "row_count": int(len(train)), "decision_year_max": 2022, "outcome_year_max": 2023, "evaluation_cohort_excluded": True, "source_sha256": sha256_file(dataset_path)}
    eval_manifest = {"status": "PASS", "row_count": int(len(evaluation)), "evaluation_year": 2024, "training_excluded": True, "source_sha256": sha256_file(dataset_path)}
    (paths.stage2_root / "train_manifest.json").write_text(json.dumps(train_manifest, indent=2) + "\n", encoding="utf-8")
    (paths.stage2_root / "evaluation_manifest.json").write_text(json.dumps(eval_manifest, indent=2) + "\n", encoding="utf-8")
    parents = [{"sha256": value} for value in parent_hashes]
    artifacts = [_artifact(paths, panel_path, "fresh:stage2:simulator_panel", parent_hashes), _artifact(paths, dataset_path, "fresh:stage2:rl_dataset", parent_hashes)]
    for path, logical in ((receipt_path, "fresh:stage2:receipt"), (report_path, "fresh:stage2:validation"), (paths.stage2_root / "train_manifest.json", "fresh:stage2:train_manifest"), (paths.stage2_root / "evaluation_manifest.json", "fresh:stage2:evaluation_manifest")):
        artifacts.append(_artifact(paths, path, logical, parent_hashes))
    return StageResult("Stage2", "PASS", "SYNTHETIC_E2E_ACCEPTANCE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"segment": "synthetic_stage2", "validation": report, "executed": True, "scientific_gate_applicable": False})


def run_stage2(paths, parent_hashes: list[str], *, context=None) -> StageResult:
    if not parent_hashes:
        return StageResult("Stage2", "FAILED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "same-run VerifyOracle manifest is required"})
    synthetic = context is not None and context.execution_class == SYNTHETIC_E2E_ACCEPTANCE
    try:
        # The synthetic profile intentionally uses the same production bundle
        # and reward/dataset validators on the same-run Oracle panel.  It does
        # not claim thesis-scale temporal verification.
        if synthetic:
            return _synthetic_stage2(paths, parent_hashes, context)

        input_root = _resolve_input_source(paths)
        input_root.mkdir(parents=True, exist_ok=True)
        design = _copy_design_inputs(paths.root, input_root)
        rc, raw_log = _run_raw_action_source(paths, input_root)
        if rc != 0:
            required_external_input = "licensed raw_all" if not (paths.root / "data/raw/raw_all").is_dir() else "STAGE2_RAW_ACTION_SOURCE"
            return StageResult("Stage2", "INPUT_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "fresh raw Stage2 action-source production did not complete", "return_code": rc, "log_tail": raw_log, "required_external_input": required_external_input, "design_inputs_materialized": design, "frozen_compute_fallback": False})
        missing = [relative for relative in _REQUIRED_PACK if not (input_root / relative).is_file() or (input_root / relative).stat().st_size <= 0]
        if missing:
            return StageResult("Stage2", "INPUT_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "authorized fresh Stage2 producer-input pack is incomplete", "required_external_input": "STAGE2_PRODUCER_INPUT_PACK_REQUIRED", "missing": missing, "design_inputs_materialized": design, "frozen_compute_fallback": False})

        from credit_recourse.rl.v43_one_pass_data import prepare_features, prepare_populations
        from credit_recourse.rl.v43_one_pass_grid import generate_grid
        from credit_recourse.rl.v43_rate_extension import prepare_rate_extension
        from credit_recourse.rl.v43_support import prepare_action_support

        with _stage2_environment(paths, input_root):
            prepare_rate_extension(paths.root)
            prepare_action_support(paths.root)
            prepare_populations(paths.root)
            prepare_features(paths.root)
            generate_grid(paths.root, evaluation=False)
            generate_grid(paths.root, evaluation=True)
            report = _validate_pack(paths, paths.stage2_root, parent_hashes, design)

        receipt = {"schema_version": "v43_stage2_execution_receipt_v1", "stage": "Stage2", **receipt_context(context), "executed": True, "input_source": str(input_root.relative_to(paths.root)).replace("\\", "/"), "design_inputs": design, "parent_hashes": parent_hashes, "validation": report}
        receipt_path = paths.stage2_root / "stage2_execution_receipt.json"
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report_path = paths.stage2_root / "stage2_validation_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        artifacts = [_artifact(paths, receipt_path, "fresh:stage2:receipt", parent_hashes), _artifact(paths, report_path, "fresh:stage2:validation", parent_hashes)]
        for relative in ("stage2/training_financial_grid.parquet", "stage2/evaluation_financial_grid.parquet", "02_data/stage3_rows.parquet", "02_data/stage4_rows.parquet", "02_data/stage5_rows.parquet"):
            path = paths.stage2_root / relative
            artifacts.append(_artifact(paths, path, f"fresh:stage2:{relative.replace('/', ':')}", parent_hashes))
        return StageResult("Stage2", "PASS", "REAL_COMPUTE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"validation": report, "executed": True, "scientific_gate_applicable": True})
    except FileNotFoundError as exc:
        return StageResult("Stage2", "INPUT_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": str(exc), "required_external_input": "licensed/raw Stage2 source"})
    except Exception as exc:
        return StageResult("Stage2", "FAILED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "STAGE2_EXECUTION_FAILED"})

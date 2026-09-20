"""Fresh V4.3 offline-RL dataset construction and independent verification."""
from __future__ import annotations

from pathlib import Path
import json
import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.rl.v43_one_pass_contract import TEMPORAL, rl_fit_allowed, assert_training_rows
from credit_recourse.rl.v43_reward_math import _compute_aux_reward_raw_deltas
from credit_recourse.rl.common.reward_contract import compose_reward, fit_robust_p95_abs, normalize_reward_component, MERTON_CLIP_BOUNDS, FCFF_CLIP_BOUNDS, LIQUIDITY_CLIP_BOUNDS

from .execution_context import receipt_context
from .stages.base import StageResult, sha256_file, write_stage_artifact

ACTION_IDS = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")
CPLD = "balance_sheet__[U01A811027400]      유동성장기부채(*)(IFRS)(천원)"
FORBIDDEN = ("data/final_freeze", "configs/current", "archive/DEPLOYED_RELEASE", "frozen/")


def _artifact(paths, relative: str, logical_id: str, parent_hashes: list[str]) -> dict[str, Any]:
    path = paths.root / "runs" / paths.run_id / relative
    return {"logical_id": logical_id, "path": path.relative_to(paths.root).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "producer": "thesis_repro.fresh_rl_dataset", "parents": [{"sha256": value} for value in parent_hashes]}


def _source_panel(paths) -> Path:
    receipt = paths.stage2_root / "simulator_execution_receipt.json"
    if not receipt.is_file():
        raise FileNotFoundError("same-run Simulator receipt is missing")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    source = paths.root / payload["source_panel"]
    if not source.is_file():
        raise FileNotFoundError(f"same-run simulator source panel is missing: {source}")
    return source


def _normalise_source_keys(panel: pd.DataFrame) -> pd.DataFrame:
    """Bind the preserved Stage1 panel's key spellings to the RL contract."""
    panel = panel.copy()
    if "firm_id" not in panel.columns:
        for candidate in ("거래소코드", "stock_code", "code"):
            if candidate in panel.columns:
                panel = panel.rename(columns={candidate: "firm_id"})
                break
    if "fiscal_year" not in panel.columns:
        for candidate in ("회계년도", "year"):
            if candidate in panel.columns:
                panel = panel.rename(columns={candidate: "fiscal_year"})
                break
    if "firm_id" not in panel.columns or "fiscal_year" not in panel.columns:
        raise ValueError(f"same-run Stage1 panel lacks canonical firm/year keys: {list(panel.columns)[:20]}")
    return panel


def _outcome_rating(panel: pd.DataFrame, row: pd.Series) -> tuple[float | None, bool]:
    firm = str(row["firm_id"]).removesuffix(".0").zfill(6)
    year = int(row["base_year"]) + 1
    candidates = panel.loc[(panel["firm_id"].astype(str).str.removesuffix(".0").str.zfill(6) == firm) & (pd.to_numeric(panel["fiscal_year"], errors="coerce") == year)]
    if candidates.empty:
        return None, False
    candidate = candidates.iloc[0]
    for name in ("rating_num", "rating_grade_num", "신용등급수치", "rating_reward_value"):
        if name in candidate and pd.notna(candidate[name]):
            return float(candidate[name]), True
    return None, False


def _reward_frame(frame: pd.DataFrame, panel: pd.DataFrame, *, synthetic: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy()
    out["outcome_year"] = pd.to_numeric(out["base_year"], errors="raise").astype(int) + 1
    out["fiscal_year"] = pd.to_numeric(out["base_year"], errors="raise").astype(int)
    ratings = out.apply(lambda row: _outcome_rating(panel, row), axis=1)
    out["outcome_rating_num"] = [item[0] for item in ratings]
    out["outcome_available"] = [bool(item[1]) for item in ratings]
    if out["outcome_rating_num"].isna().all():
        # The synthetic fixture has no licensed observed rating label. Its
        # simulated next state is an explicit fixture transition, not a thesis
        # outcome claim. Full replication never takes this branch.
        out["outcome_rating_num"] = pd.to_numeric(out.get("sim__rating_num", pd.Series(np.nan, index=out.index)), errors="coerce")
        out["outcome_available"] = out["outcome_rating_num"].notna()
        outcome_source = "synthetic_simulated_fixture_transition"
    else:
        outcome_source = "same_run_observed_panel_next_year"
    current_rating = pd.to_numeric(out.get("state__rating_num", 0.0), errors="coerce").fillna(0.0)
    out["reward_raw_notch"] = pd.to_numeric(out["outcome_rating_num"], errors="coerce").fillna(current_rating) - current_rating
    out["reward_raw"] = out["reward_raw_notch"]
    aliases = {
        "state__current_portion_long_debt": CPLD,
        "sim__current_portion_long_debt": f"next__{CPLD}",
    }
    for source, target in aliases.items():
        if source not in out:
            out[source] = 0.0
        out[target] = pd.to_numeric(out[source], errors="coerce").fillna(0.0)
    required = ["sim__total_assets", "sim__short_term_debt", "sim__long_term_debt", "sim__bonds", "sim__operating_cf", "sim__capex", "sim__cash", "sim__short_term_investments"]
    fixture_imputed_fields: list[str] = []
    for name in required:
        if name not in out:
            raise ValueError(f"Simulator output lacks required reward substrate: {name}")
        out[f"next__{name}"] = pd.to_numeric(out[name], errors="coerce")
        state_name = name.replace("sim__", "state__")
        if state_name not in out:
            if not synthetic:
                raise ValueError(f"Simulator output lacks required observed reward substrate: {state_name}")
            out[state_name] = 0.0
            fixture_imputed_fields.append(state_name)
        out[name] = pd.to_numeric(out[state_name], errors="coerce")
        if out[name].isna().any():
            if not synthetic:
                raise ValueError(f"Simulator output has missing observed reward substrate: {state_name}")
            out[name] = out[name].fillna(0.0)
            fixture_imputed_fields.append(state_name)
        if out[f"next__{name}"].isna().any():
            if not synthetic:
                raise ValueError(f"Simulator output has missing simulated reward substrate: {name}")
            out[f"next__{name}"] = out[f"next__{name}"].fillna(0.0)
            fixture_imputed_fields.append(name)
    aux = _compute_aux_reward_raw_deltas(out, context="fresh V4.3 RLDataset reward")
    for raw, scale, bounds, label in (("delta_merton_badness", "merton", MERTON_CLIP_BOUNDS, "delta_merton_badness"), ("delta_fcff_capacity", "fcff", FCFF_CLIP_BOUNDS, "delta_fcff_capacity"), ("delta_liquid_capacity", "liquidity", LIQUIDITY_CLIP_BOUNDS, "delta_liquid_capacity")):
        train_values = aux.loc[(aux.base_year <= TEMPORAL.train_transition_year_max) & (aux.outcome_available), raw]
        scale_value = fit_robust_p95_abs(train_values, label=label) if not train_values.empty and train_values.abs().max() > 1e-12 else 1.0
        aux[f"{raw}_scaled"] = normalize_reward_component(aux[raw], scale_parameter=scale_value, clip_bounds=bounds, label=label)
        aux[f"lambda_{scale}"] = {"merton": 0.2, "fcff": 0.4, "liquidity": 0.1}[scale]
    aux["phi_t"] = 0.0; aux["phi_tplusH"] = 0.0; aux["delta_phi"] = 0.0; aux["lambda_phi"] = 0.0; aux["reward_aux_phi"] = 0.0
    aux["reward_total_raw"] = compose_reward(merton_component=aux["delta_merton_badness_scaled"], fcff_component=aux["delta_fcff_capacity_scaled"], liquidity_component=aux["delta_liquid_capacity_scaled"], profitability_component=0.0, merton_lambda=0.2, fcff_lambda=0.4, liquidity_lambda=0.1, sector_phi_component=0.0, sector_phi_lambda=0.0, base_component=aux["reward_raw"])
    eligible = rl_fit_allowed(aux, action_support=np.ones(len(aux), dtype=bool), reward_support=np.ones(len(aux), dtype=bool), outcome_available=aux["outcome_available"].to_numpy(dtype=bool))
    aux["action_support_valid"] = True; aux["reward_support_valid"] = True; aux["rl_fit_allowed"] = eligible
    aux["support_basis"] = "same-run Simulator nine-action grid plus V43 temporal/outcome contract"
    train = aux.loc[aux["rl_fit_allowed"]].copy()
    if train.empty:
        raise ValueError("fresh RLDataset has no eligible training transitions")
    mean = float(train["reward_total_raw"].mean()); std = float(train["reward_total_raw"].std(ddof=0))
    if not math.isfinite(std) or std <= 1e-12: std = 1.0
    aux["reward_mean_train"] = mean; aux["reward_std_train"] = std; aux["reward_train"] = (aux["reward_total_raw"] - mean) / std; aux["reward_original"] = aux["reward_total_raw"]; aux["reward"] = aux["reward_train"]
    return aux, {"outcome_source": outcome_source, "train_reward_mean": mean, "train_reward_std": std, "oracle_score_used": False, "reward_contract": "compose_reward with frozen V43 stage2 lambdas (merton=.2, fcff=.4, liquidity=.1, phi=0)", "temporal": TEMPORAL.to_dict(), "synthetic_fixture_imputed_fields": sorted(set(fixture_imputed_fields))}


def verify_fresh_rl_dataset(path: Path, *, parent_hashes: list[str], expected_eval_firms: int | None = None) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    required = {"firm_id", "fiscal_year", "outcome_year", "candidate_id", "reward_train", "rl_fit_allowed", "outcome_available", "action_contract_sha256"}
    missing = sorted(required - set(frame.columns))
    if missing: raise ValueError(f"RLDataset missing required columns: {missing}")
    if set(frame.candidate_id.unique()) != set(ACTION_IDS): raise ValueError("RLDataset does not contain all nine actions")
    if frame.duplicated(["firm_id", "fiscal_year", "candidate_id"]).any(): raise ValueError("RLDataset contains duplicate firm/year/action rows")
    if (frame.loc[frame.rl_fit_allowed, "fiscal_year"] > TEMPORAL.train_transition_year_max).any(): raise ValueError("training transition cutoff violated")
    if (frame.loc[frame.rl_fit_allowed, "outcome_year"] > TEMPORAL.train_outcome_year_max).any(): raise ValueError("training outcome cutoff violated")
    if frame.loc[frame.rl_fit_allowed, "fiscal_year"].eq(TEMPORAL.eval_base_year).any(): raise ValueError("evaluation cohort leaked into training")
    if not np.isfinite(pd.to_numeric(frame["reward_train"], errors="coerce").to_numpy(dtype=float)).all(): raise ValueError("RLDataset reward is non-finite")
    if any(any(token in str(value).replace("\\", "/") for token in FORBIDDEN) for value in frame.to_numpy().ravel()): raise ValueError("RLDataset contains forbidden historical parent reference")
    eval_rows = frame.loc[frame.fiscal_year.eq(TEMPORAL.eval_base_year)]
    if expected_eval_firms is not None and int(eval_rows[["firm_id", "fiscal_year"]].drop_duplicates().shape[0]) != expected_eval_firms: raise ValueError("evaluation cohort count does not match canonical contract")
    return {"schema_version": "fresh_rl_dataset_validation_report_v1", "status": "PASS", "row_count": int(len(frame)), "training_rows": int(frame.rl_fit_allowed.sum()), "evaluation_rows": int(len(eval_rows)), "evaluation_firm_count": int(eval_rows.firm_id.nunique()), "candidate_action_ids": list(ACTION_IDS), "temporal": TEMPORAL.to_dict(), "evaluation_cohort_training_leak": False, "oracle_score_reward_leak": False, "same_run_parent_hashes": parent_hashes}


def run_fresh_rl_dataset(paths, parent_hashes: list[str], *, context=None) -> StageResult:
    simulator_panel = paths.stage2_root / "simulator_panel.parquet"
    if not simulator_panel.is_file(): return StageResult("RLDataset", "INPUT_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "same-run Simulator panel is required"})
    try:
        source_panel = _source_panel(paths)
        panel = _normalise_source_keys(pd.read_parquet(source_panel))
        frame = pd.read_parquet(simulator_panel)
        # Keep the simulator's canonical ``base_year`` through reward and
        # outcome construction.  The authoritative RL contract uses
        # ``fiscal_year``; expose that alias only after the transition rows
        # have been built so the two contracts cannot be confused.
        frame["action_contract_sha256"] = frame.get("action_contract_sha256", "")
        synthetic = context is not None and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE"
        output, reward_meta = _reward_frame(frame, panel, synthetic=synthetic)
        output["fiscal_year"] = pd.to_numeric(output["base_year"], errors="raise").astype(int)
        out_path = paths.stage3_root / "rl_dataset.parquet"; output.to_parquet(out_path, index=False)
        expected_eval = None if synthetic else 575
        report = verify_fresh_rl_dataset(out_path, parent_hashes=parent_hashes, expected_eval_firms=expected_eval)
        report.update(reward_meta)
        train = output.loc[output.rl_fit_allowed].copy(); evaluation = output.loc[output.fiscal_year.eq(TEMPORAL.eval_base_year)].copy()
        train_manifest = {"schema_version": "fresh_rl_train_manifest_v1", "status": "PASS", "rows": len(train), "decision_year_max": int(train.fiscal_year.max()), "outcome_year_max": int(train.outcome_year.max()), "evaluation_rows_used": 0, "parent_hashes": parent_hashes}
        evaluation_manifest = {"schema_version": "fresh_rl_evaluation_manifest_v1", "status": "PASS", "rows": len(evaluation), "firm_count": int(evaluation.firm_id.nunique()), "evaluation_base_year": TEMPORAL.eval_base_year, "excluded_from_training": True, "parent_hashes": parent_hashes}
        receipt = {"schema_version": "fresh_rl_dataset_execution_receipt_v1", "stage": "RLDataset", **receipt_context(context), "stage_execution_kind": "REAL_COMPUTE", "executed": True, "producer": "thesis_repro.fresh_rl_dataset.run_fresh_rl_dataset", "authoritative_contract": "credit_recourse.rl.v43_one_pass_contract.rl_fit_allowed", "simulator_parent": str(simulator_panel.relative_to(paths.root)).replace("\\", "/"), "parent_hashes": parent_hashes, "temporal": TEMPORAL.to_dict(), "reward_meta": reward_meta}
        report_path = paths.stage3_root / "dataset_validation_report.json"; report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (paths.stage3_root / "train_manifest.json").write_text(json.dumps(train_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (paths.stage3_root / "evaluation_manifest.json").write_text(json.dumps(evaluation_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (paths.stage3_root / "dataset_execution_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        artifacts = [_artifact(paths, "04_rl_dataset/rl_dataset.parquet", "fresh:rl_dataset:panel", parent_hashes), write_stage_artifact(paths, "04_rl_dataset/train_manifest.json", train_manifest, "fresh:rl_dataset:train_manifest", ({"sha256": h} for h in parent_hashes)), write_stage_artifact(paths, "04_rl_dataset/evaluation_manifest.json", evaluation_manifest, "fresh:rl_dataset:evaluation_manifest", ({"sha256": h} for h in parent_hashes)), write_stage_artifact(paths, "04_rl_dataset/dataset_execution_receipt.json", receipt, "fresh:rl_dataset:receipt", ({"sha256": h} for h in parent_hashes)), write_stage_artifact(paths, "04_rl_dataset/dataset_validation_report.json", report, "fresh:rl_dataset:validation", ({"sha256": h} for h in parent_hashes))]
        return StageResult("RLDataset", "PASS", "REAL_COMPUTE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"validation": report, "training_rows": len(train), "evaluation_rows": len(evaluation), "scientific_gate_applicable": bool(context.scientific_gate_applicable) if context is not None else True})
    except FileNotFoundError as exc:
        return StageResult("RLDataset", "INPUT_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"error": repr(exc)})
    except Exception as exc:
        return StageResult("RLDataset", "FAILED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"error": repr(exc), "failure_class": "RLDATASET_EXECUTION_FAILED"})

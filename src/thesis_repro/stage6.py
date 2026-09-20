"""Run and attest the preserved V4.3 Stage6 final OOT evaluation."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path

from .execution_context import receipt_context
from .stages.base import StageResult, sha256_file, write_stage_artifact


def _materialize_release(paths, parent_hashes: list[str]) -> list[dict]:
    """Normalize the native evaluator output into the run-local Stage6 release."""
    import hashlib
    import json
    import pandas as pd

    source = paths.stage6_root / "all_fixed_action_scores.parquet"
    decisions = paths.stage6_root / "C3_actor_decisions.parquet"
    c2 = paths.stage6_root / "C2_fixed_rule_decisions.parquet"
    if not source.is_file() or not decisions.is_file() or not c2.is_file():
        raise FileNotFoundError("native Stage6 evaluator did not write fixed-action, C3, and C2 tables")
    scores = pd.read_parquet(source).rename(columns={"alpha": "Alpha", "beta": "Beta", "gamma": "Gamma", "base_year": "fiscal_year", "candidate_id": "action"})
    if len(scores) != 575 * 9 or scores.duplicated(["firm_id", "fiscal_year", "action"]).any():
        raise ValueError("fresh Stage6 payoff surface is not the complete 575x9 table")
    scores["row_id"] = scores.groupby("firm_id", sort=True).ngroup() * 9 + scores.groupby("firm_id", sort=True).cumcount()
    surface_cols = ["row_id", "firm_id", "action", "Alpha", "Beta", "Gamma"]
    surface = scores[surface_cols]
    surface_path = paths.stage6_root / "firm_action_oracle_payoffs.parquet"
    surface.to_parquet(surface_path, index=False)
    c3 = pd.read_parquet(decisions).rename(columns={"candidate_id": "action_id"})
    c3 = c3[["firm_id", "fiscal_year", "action_id"]]
    c3_scores = surface.rename(columns={"action": "action_id"})
    c3 = c3.merge(c3_scores, on=["firm_id", "fiscal_year", "action_id"], how="left", validate="one_to_one")
    c2_frame = pd.read_parquet(c2)
    c2_frame = c2_frame.rename(columns={"candidate_id": "C2_action"})[["firm_id", "fiscal_year", "C2_action"]]
    c2_scores = surface.rename(columns={"action": "C2_action", "Alpha": "C2_Alpha", "Beta": "C2_Beta", "Gamma": "C2_Gamma"})[["firm_id", "fiscal_year", "C2_action", "C2_Alpha", "C2_Beta", "C2_Gamma"]]
    c2_frame = c2_frame.merge(c2_scores, on=["firm_id", "fiscal_year", "C2_action"], how="left", validate="one_to_one")
    c3 = c3.merge(c2_frame, on=["firm_id", "fiscal_year"], how="left", validate="one_to_one")
    c3["row_id"] = c3_scores.groupby("firm_id", sort=True).ngroup() if False else range(len(c3))
    c3_path = paths.stage6_root / "C3E_firm_actions_payoffs.parquet"
    c3[["row_id", "firm_id", "fiscal_year", "action_id", "Alpha", "Beta", "Gamma", "C2_action", "C2_Alpha", "C2_Beta", "C2_Gamma"]].to_parquet(c3_path, index=False)
    summary = paths.stage6_root / "evaluation_summary.json"
    payload = json.loads(summary.read_text(encoding="utf-8")) if summary.is_file() else {}
    metrics = paths.stage6_root / "C3E_metrics.json"
    metrics.write_text(json.dumps({"status": "PASS", "evaluation_firm_count": 575, "fixed_action_rows": len(surface), "parent_hashes": parent_hashes, "native_summary_sha256": sha256_file(summary)}, indent=2) + "\n", encoding="utf-8")
    manifest = paths.stage6_root / "EVALUATION_MANIFEST.json"
    manifest.write_text(json.dumps({"status": "PASS", "release_role": "fresh_stage6", "surface_rows": len(surface), "c3_rows": len(c3), "summary_status": payload.get("status"), "parent_hashes": parent_hashes}, indent=2) + "\n", encoding="utf-8")
    release_hash = hashlib.sha256(surface_path.read_bytes() + c3_path.read_bytes()).hexdigest()
    current = paths.stage6_root / "CURRENT_RELEASE.json"
    current.write_text(json.dumps({"release_id": paths.run_id, "release_hash": release_hash, "payoff_surface_path": str(surface_path.relative_to(paths.root)).replace("\\", "/"), "payoff_surface_sha256": sha256_file(surface_path), "artifact_path": str(c3_path.relative_to(paths.root)).replace("\\", "/")}, indent=2) + "\n", encoding="utf-8")
    return [{"logical_id": f"fresh:stage6:{path.name}", "path": str(path.relative_to(paths.root)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "producer": "thesis_repro.stage6", "parents": [{"sha256": value} for value in parent_hashes]} for path in (surface_path, c3_path, metrics, manifest, current)]


def _evaluate_native_c3e(paths, parent_hashes: list[str]) -> None:
    """Run the preserved fixed-action scorer with the fresh C3-E decisions."""
    import json
    import pandas as pd
    from credit_recourse.eval.v43_c2 import c2_row_conditional
    from credit_recourse.eval.v43_stage6_reporting import summarize_policies, validate_grid
    from credit_recourse.eval.v43_stage6_scoring import score_grid
    from credit_recourse.rl.v43_one_pass_data import input_root

    grid_path = paths.stage2_root / "stage2/evaluation_financial_grid.parquet"
    grid = pd.read_parquet(grid_path)
    validate_grid(grid, expected_firms=575)
    source_root = paths.stage2_root / "input_source"
    receipt = paths.stage2_root / "stage2_execution_receipt.json"
    if receipt.is_file():
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        source_root = paths.root / payload.get("input_source", "03_stage2/input_source")
    os.environ["THESIS_REPRO_STAGE2_INPUT_ROOT"] = str(source_root)
    scores, oracle_inputs = score_grid(paths.root, grid, expected_firms=575)
    unique_firms = grid[["firm_id", "base_year"]].drop_duplicates().sort_values(["base_year", "firm_id"]).reset_index(drop=True)
    action_path = paths.c3e_root / "C3E_firm_actions.parquet"
    action_frame = pd.read_parquet(action_path)
    if len(action_frame) != len(unique_firms):
        raise ValueError("fresh C3-E action count does not match Stage6 cohort")
    c3 = unique_firms.copy()
    c3["candidate_id"] = action_frame.sort_values("evaluation_ordinal")["action_id"].to_numpy()
    c3 = c3.rename(columns={"base_year": "fiscal_year"})
    c3["policy"] = "C3"
    reference = pd.read_parquet(paths.stage2_root / "02_data/stage5_factual_reward_inputs.parquet")
    if int(reference.fiscal_year.max()) > 2022:
        raise ValueError("future data in C2 reference")
    base = pd.read_parquet(input_root(paths.root) / "phase_eval_candidate.parquet")
    c2 = c2_row_conditional(base, None, reference, candidate_ids_only=True)
    c2 = pd.concat([unique_firms.reset_index(drop=True), c2.reset_index(drop=True)], axis=1).rename(columns={"base_year": "fiscal_year"})
    c2["policy"] = "C2"
    scores.to_parquet(paths.stage6_root / "all_fixed_action_scores.parquet", index=False)
    oracle_inputs.to_parquet(paths.stage6_root / "oracle_inputs_2025.parquet", index=False)
    c3.to_parquet(paths.stage6_root / "C3_actor_decisions.parquet", index=False)
    c2.to_parquet(paths.stage6_root / "C2_fixed_rule_decisions.parquet", index=False)
    reports = summarize_policies(scores, c3.rename(columns={"fiscal_year": "base_year"}), c2.rename(columns={"fiscal_year": "base_year"}))
    for name, frame in reports.items():
        if hasattr(frame, "to_parquet"):
            frame.to_parquet(paths.stage6_root / f"{name}.parquet", index=False)
            if name in {"comparison", "contrasts", "action_distribution"}:
                frame.to_csv(paths.stage6_root / f"{name}.csv", index=False)
    summary = {"status": "PASS", "role": "final_OOT_2024", "firms": 575, "rows": len(grid), "c3e_parent_hashes": parent_hashes, "comparison": reports["comparison"].to_dict("records"), "contrasts": reports["contrasts"].to_dict("records"), "Oracle_used_for_action_choice": False}
    (paths.stage6_root / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")


@contextmanager
def _environment(paths):
    values = {
        "THESIS_REPRO_STAGE2_ROOT": str(paths.stage2_root),
        "THESIS_REPRO_STAGE5_ROOT": str(paths.rl_iql_root),
        "THESIS_REPRO_STAGE6_ROOT": str(paths.stage6_root),
        "THESIS_REPRO_RUN_ROOT": str(paths.run_root),
        "CREDIT_REPRO_RUN_PATH": str(paths.run_root),
        "THESIS_REPRO_ORACLE_CONFIG_ROOT": str(paths.oracle_root / "work/contracts/oracle_components"),
        "THESIS_REPRO_STAGE2_INPUT_ROOT": str(paths.stage2_root / "input_source"),
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_stage6(paths, parent_hashes: list[str], *, context=None) -> StageResult:
    if context is not None and context.execution_class == "SYNTHETIC_E2E_ACCEPTANCE":
        return _run_synthetic_stage6(paths, parent_hashes, context)
    required = [paths.rl_iql_root / "actor_graph.json", paths.c3e_root / "release.json", paths.stage2_root / "stage2/evaluation_financial_grid.parquet"]
    missing = [str(path.relative_to(paths.root)).replace("\\", "/") for path in required if not path.is_file()]
    if missing:
        return StageResult("Stage6", "INPUT_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": "same-run Stage2, C3-E and Stage5 outputs are required", "missing": missing, "frozen_compute_fallback": False})
    try:
        with _environment(paths):
            _evaluate_native_c3e(paths, parent_hashes)
        summary = paths.stage6_root / "evaluation_summary.json"
        if not summary.is_file():
            raise FileNotFoundError(summary)
        artifact = {
            "logical_id": "fresh:stage6:evaluation_summary",
            "path": str(summary.relative_to(paths.root)).replace("\\", "/"),
            "sha256": sha256_file(summary),
            "size_bytes": summary.stat().st_size,
            "producer": "credit_recourse.eval.v43_one_pass_evaluation.evaluate",
            "parents": [{"sha256": value} for value in parent_hashes],
        }
        release_artifacts = _materialize_release(paths, parent_hashes)
        report = write_stage_artifact(paths, "08_stage6/stage6_execution_receipt.json", {"status": "PASS", "stage": "Stage6", **receipt_context(context), "summary_sha256": artifact["sha256"], "parent_hashes": parent_hashes}, "fresh:stage6:receipt", ({"sha256": value} for value in parent_hashes))
        return StageResult("Stage6", "PASS", "REAL_COMPUTE", executed=True, artifacts=[artifact, report, *release_artifacts], parent_hashes=parent_hashes, details={"evaluation_summary": str(summary.relative_to(paths.root)).replace("\\", "/"), "scientific_gate_applicable": bool(context.scientific_gate_applicable) if context else True})
    except FileNotFoundError as exc:
        return StageResult("Stage6", "INPUT_REQUIRED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": str(exc), "frozen_compute_fallback": False})
    except Exception as exc:
        return StageResult("Stage6", "FAILED", "REAL_COMPUTE", executed=False, parent_hashes=parent_hashes, details={"reason": repr(exc), "failure_class": "STAGE6_EXECUTION_FAILED"})


def _run_synthetic_stage6(paths, parent_hashes: list[str], context=None) -> StageResult:
    """Reduced structural Stage6 using the real action/scoring vocabulary."""
    import hashlib
    import json
    import pandas as pd
    canonical_actions = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")

    panel_path = paths.stage2_root / "simulator_panel.parquet"
    action_path = paths.c3e_root / "C3E_firm_actions.parquet"
    if not panel_path.is_file() or not action_path.is_file():
        return StageResult("Stage6", "INPUT_REQUIRED", "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": "same-run reduced Stage2 and C3-E outputs are required"})
    panel = pd.read_parquet(panel_path)
    base = panel.drop_duplicates(["firm_id", "base_year"])[["firm_id", "base_year"]].sort_values(["base_year", "firm_id"]).reset_index(drop=True)
    actions = pd.read_parquet(action_path).sort_values("evaluation_ordinal")
    if len(actions) != len(base):
        return StageResult("Stage6", "FAILED", "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": "reduced C3-E cohort mismatch"})
    rows = []
    simulator_action_column = "action_id" if "action_id" in panel.columns else "candidate_id"
    if simulator_action_column not in panel.columns:
        return StageResult("Stage6", "FAILED", "SYNTHETIC_E2E_ACCEPTANCE", executed=False, parent_hashes=parent_hashes, details={"reason": "same-run simulator panel lacks canonical action identity"})
    for ordinal, item in base.iterrows():
        firm = item.firm_id
        year = int(item.base_year)
        firm_rows = panel[(panel.firm_id == firm) & (panel.base_year == year)]
        for action in canonical_actions:
            selected = firm_rows[firm_rows[simulator_action_column].astype(str) == action]
            value = float(selected["sim__operating_income"].iloc[0]) if not selected.empty else 0.0
            rows.append({"row_id": ordinal * 9 + len([r for r in rows if r["firm_id"] == firm]), "firm_id": firm, "fiscal_year": year, "action": action, "Alpha": value, "Beta": value, "Gamma": value})
    surface = pd.DataFrame(rows)
    output = paths.stage6_root / "firm_action_oracle_payoffs.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    surface.to_parquet(output, index=False)
    selected_actions = actions["action_id"].astype(str).tolist()
    c3rows = []
    for ordinal, item in base.iterrows():
        action = selected_actions[ordinal]
        chosen = surface[(surface.firm_id == item.firm_id) & (surface.fiscal_year == int(item.base_year)) & (surface.action == action)].iloc[0]
        a0 = surface[(surface.firm_id == item.firm_id) & (surface.fiscal_year == int(item.base_year)) & (surface.action == "A0")].iloc[0]
        c3rows.append({"row_id": ordinal, "firm_id": item.firm_id, "fiscal_year": int(item.base_year), "action_id": action, "Alpha": chosen.Alpha, "Beta": chosen.Beta, "Gamma": chosen.Gamma, "C2_action": "A0", "C2_Alpha": a0.Alpha, "C2_Beta": a0.Beta, "C2_Gamma": a0.Gamma})
    deployed = pd.DataFrame(c3rows)
    deployed_path = paths.stage6_root / "C3E_firm_actions_payoffs.parquet"
    deployed.to_parquet(deployed_path, index=False)
    payload = {"status": "PASS", "release_id": paths.run_id, "release_hash": hashlib.sha256(output.read_bytes() + deployed_path.read_bytes()).hexdigest(), "payoff_surface_path": str(output.relative_to(paths.root)).replace("\\", "/"), "payoff_surface_sha256": sha256_file(output), "artifact_path": str(deployed_path.relative_to(paths.root)).replace("\\", "/")}
    (paths.stage6_root / "CURRENT_RELEASE.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (paths.stage6_root / "C3E_metrics.json").write_text(json.dumps({"status": "PASS", "evaluation_firm_count": int(base.firm_id.nunique()), "fixed_action_rows": len(surface), "parent_hashes": parent_hashes}, indent=2) + "\n", encoding="utf-8")
    (paths.stage6_root / "EVALUATION_MANIFEST.json").write_text(json.dumps({"status": "PASS", "surface_rows": len(surface), "c3_rows": len(deployed), "fixture_only": True, "parent_hashes": parent_hashes}, indent=2) + "\n", encoding="utf-8")
    receipt = write_stage_artifact(paths, "08_stage6/stage6_execution_receipt.json", {"status": "PASS", "stage": "Stage6", **receipt_context(context), "fixture_only": True, "parent_hashes": parent_hashes}, "fresh:stage6:receipt", ({"sha256": value} for value in parent_hashes))
    artifacts = [receipt]
    for path in (output, deployed_path, paths.stage6_root / "CURRENT_RELEASE.json", paths.stage6_root / "C3E_metrics.json", paths.stage6_root / "EVALUATION_MANIFEST.json"):
        artifacts.append({"logical_id": f"fresh:stage6:{path.name}", "path": str(path.relative_to(paths.root)).replace("\\", "/"), "sha256": sha256_file(path), "size_bytes": path.stat().st_size, "producer": "thesis_repro.stage6._run_synthetic_stage6", "parents": [{"sha256": value} for value in parent_hashes]})
    return StageResult("Stage6", "PASS", "SYNTHETIC_E2E_ACCEPTANCE", executed=True, artifacts=artifacts, parent_hashes=parent_hashes, details={"fixture_only": True, "evaluation_firm_count": int(base.firm_id.nunique()), "fixed_action_rows": len(surface)})

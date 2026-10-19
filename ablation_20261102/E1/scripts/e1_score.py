"""E1 scoring path: frozen simulator + Oracles (alpha, beta, gamma), the Stage 8 route in minimal form.

Mirrors ``final_stage8_llm_multi_oracle_eval.pipeline.run_stage8``: ``simulate_llm_policy_states`` on the
ITT action vectors, then ``score_alpha`` / ``score_beta_ordered_logit_params`` / ``score_gamma_model``.
``delta_R_score_<o>`` = R_score_<o> minus the A0 (no-action) score of the same firm; the A0 score is
produced by the same path and, in the gate, compared with the frozen ``noop_R_score_<o>``.

Run root: ``..\\runroot`` laid out by ``e1_setup_runroot.ps1`` (project root and THESIS_REPRO_RUN_ROOT).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DIMENSIONS = (
    "growth_capex_reduction_pct", "deleveraging_total_debt_pct", "refinancing_short_debt_pct",
    "inv_turnover_chg", "ar_turnover_chg", "ap_turnover_chg", "cogs_ratio_chg", "sga_ratio_chg",
)
ACTION_COLS = tuple(f"action__{d}" for d in DIMENSIONS)
ORACLES = ("alpha", "beta", "gamma")


def setup(kit_root: Path, run_root: Path) -> None:
    os.environ["THESIS_REPRO_RUN_ROOT"] = str(Path(run_root).resolve())
    src = str(Path(kit_root).resolve() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _simulate_unique(actions: pd.DataFrame, run_root: Path):
    """Simulate each distinct (row_id, action vector) once; return the state frame aligned to ``actions``."""
    from credit_recourse.eval.final_stage8_llm_multi_oracle_eval import pipeline as p
    from credit_recourse.simulator.business_plan_factory import CALIBRATED_BP_RATE_V4_MODE

    space = p.load_current_action_space(run_root)
    ident = p._load_stage6_simulator_identity(run_root)
    assert ident["sim_business_plan_mode"] == CALIBRATED_BP_RATE_V4_MODE and ident["preserve_current_non_current_residual"] is True
    base = pd.read_parquet(Path(run_root) / "03_stage2" / "phase_eval_candidate.parquet")
    key = actions[["row_id", *ACTION_COLS]].astype(float).round(15).apply(tuple, axis=1)
    uniq_idx = ~key.duplicated()
    uniq = actions.loc[uniq_idx].copy().reset_index(drop=True)
    uniq["policy"] = "E1"
    uniq["candidate_id"] = np.where(uniq[list(ACTION_COLS)].abs().sum(axis=1) == 0.0, "A0", "FREE8")
    state, _ = p.simulate_llm_policy_states(
        base, uniq, space, Path(run_root), predicted_fiscal_year=int(ident["predicted_fiscal_year"]),
        preserve_current_non_current_residual=True, sim_business_plan_mode=str(ident["sim_business_plan_mode"]),
        write_outputs=False, collect_audit=False, project_root=Path(run_root),
    )
    return p, state.reset_index(drop=True), key, uniq_idx


def score_actions(actions: pd.DataFrame, run_root: Path) -> pd.DataFrame:
    """actions: columns row_id + ACTION_COLS (ITT action vectors). Returns R_score_alpha/beta/gamma per input row."""
    from credit_recourse.oracle.artifact_io import load_registry, resolve_backend_artifact
    from credit_recourse.eval.v43_oracle_backends import score_alpha, score_beta_ordered_logit_params, score_gamma_model
    from credit_recourse.contracts.runtime_assets import active_oracle_registry
    from credit_recourse.contracts.stage_paths import final_root

    run_root = Path(run_root)
    actions = actions.reset_index(drop=True)
    p, state, key, uniq_idx = _simulate_unique(actions, run_root)
    reg = load_registry(active_oracle_registry(run_root))
    backends = reg["backends"]
    assert reg.get("final_result_allowed") is True
    final = final_root(run_root)
    scores = pd.DataFrame(index=state.index)
    for backend in ORACLES:
        b = backends[backend]
        params = resolve_backend_artifact(run_root, final, b.get("params", ""))
        if backend == "alpha":
            scores["R_score_alpha"] = score_alpha(state, params).to_numpy()
        elif backend == "beta":
            scores["R_score_beta"] = score_beta_ordered_logit_params(state, params).to_numpy()
        else:
            model = resolve_backend_artifact(run_root, final, b.get("model", ""))
            scores["R_score_gamma"] = score_gamma_model(state, params, model).to_numpy()
    uniq_keys = key.loc[uniq_idx].reset_index(drop=True)
    lookup = {k: i for i, k in enumerate(uniq_keys)}
    pos = key.map(lookup).to_numpy()
    out = scores.iloc[pos].reset_index(drop=True)
    out.insert(0, "row_id", actions["row_id"].to_numpy())
    return out


def score_with_noop(actions: pd.DataFrame, run_root: Path) -> pd.DataFrame:
    """Scores plus own A0 scores and delta_R_score_<o> (both produced by this path)."""
    actions = actions.reset_index(drop=True)
    rows = sorted(actions["row_id"].unique())
    zero = pd.DataFrame({"row_id": rows})
    for c in ACTION_COLS:
        zero[c] = 0.0
    # one simulate call for the A0 rows and the actions together (setup cost is paid once)
    both = score_actions(pd.concat([zero, actions[["row_id", *ACTION_COLS]]], ignore_index=True), run_root)
    noop = both.iloc[: len(zero)].set_index("row_id")
    scored = both.iloc[len(zero):].reset_index(drop=True)
    for o in ORACLES:
        scored[f"noop_R_score_{o}"] = scored["row_id"].map(noop[f"R_score_{o}"])
        scored[f"delta_R_score_{o}"] = scored[f"R_score_{o}"] - scored[f"noop_R_score_{o}"]
    return scored

"""Fast post-processing and integrity checks for b1_run.py outputs."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
ANALYSIS = TASK / "A_reproduction" / "R2_R8_replay_workspace" / "analysis_20261117"
KIT = TASK / "A_reproduction" / "E1_score_gate_replay" / "runroot_zip" / "THESIS_REPRO_KIT_v2.1.1_FINAL"
ABL = TASK / "source_inputs" / "github_release_ablation_20261102_readonly" / "contents" / "ablation_20261102" / "E1" / "observations" / "e1_observations.parquet"
os.environ["KIT_ROOT"] = str(KIT).replace("\\", "/")
os.environ["HZ_SRC"] = str(KIT / "src").replace("\\", "/")
os.environ["ABL_E1"] = str(ABL).replace("\\", "/")
sys.path.insert(0, str(ANALYSIS))
import prep2 as P  # noqa: E402
from hz import _row_to_firm_state  # noqa: E402

N = 575
AX = list(P.AX)
EVALS = ["alpha", "beta", "gamma", "delta1", "delta2", "epsilon1", "epsilon2"]
FIN_FIELDS = ["revenue", "operating_income", "pretax_income", "pure_interest_expense", "total_debt",
              "short_term_debt", "cash", "current_assets", "current_liabilities", "total_equity", "operating_cf"]
UNIT_SIGNS = {AX[0]: 1.0, AX[1]: 1.0, AX[2]: 1.0, AX[3]: 1.0, AX[4]: 1.0, AX[5]: -1.0, AX[6]: -1.0, AX[7]: -1.0}


def norm(action):
    return np.array([float(action.get(k, 0.0)) / float(P.SC[k]) for k in AX], dtype=float)


def raw(units):
    return {k: float(np.asarray(units, dtype=float)[j] * P.SC[k]) for j, k in enumerate(AX)}


def sig(action):
    return tuple(np.round(norm(action), 10).tolist())


def sig_json(action):
    return json.dumps(list(sig(action)), separators=(",", ":"))


def unit_action(axis):
    u = np.zeros(len(AX), dtype=float)
    u[AX.index(axis)] = UNIT_SIGNS[axis]
    return raw(u)


def action_labels_and_actions():
    ids = P.a3.sort_values("row_id").action_id.astype(str).to_numpy()
    p0 = [P.CAND[c] for c in ids]
    p1 = [P.CAND["OE"] for _ in range(N)]
    p2a = unit_action("cogs_ratio_chg")
    p2 = [p2a for _ in range(N)]
    p3 = [p2a if c == "OE" else P.CAND[c] for c in ids]
    p4 = []
    for c in ids:
        if c == "OE":
            p4.append(p2a)
        else:
            u = 0.5 * norm(P.CAND[c])
            u[AX.index("cogs_ratio_chg")] -= 0.5
            p4.append(raw(u))
    return ids, {"P0_RL_C3E": p0, "P1_Heuristic_OE": p1, "P2_fixed_COGS_minus1": p2,
                 "P3_C3E_OE_to_P2": p3, "P4_half_C3E_plus_COGS_minus0.5": p4}


OUT = HERE
meta_path = OUT / "b1_run_metadata.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
surface = pd.read_parquet(OUT / "b1_simulator_action_surface.parquet")
surface_key = {(int(r.row_id), r.action_signature): r for r in surface.itertuples(index=False)}
ids, actions = action_labels_and_actions()
baseline_sig = sig_json(P.CAND["A0"])
base_abs = np.vstack([[getattr(surface_key[(i, baseline_sig)], "absolute_score_" + e) for e in EVALS] for i in range(N)])
a9 = np.load(ANALYSIS / "A9.npy")
direct_p0 = np.vstack([[getattr(surface_key[(i, sig_json(actions["P0_RL_C3E"][i]))], "absolute_score_" + e) for e in EVALS] for i in range(N)])
direct_p1 = np.vstack([[getattr(surface_key[(i, sig_json(actions["P1_Heuristic_OE"][i]))], "absolute_score_" + e) for e in EVALS] for i in range(N)])
max_p0 = float(np.max(np.abs((direct_p0 - base_abs) - a9[np.arange(N), P.RLID, :])))
max_p1 = float(np.max(np.abs((direct_p1 - base_abs) - a9[:, P.IDS.index("OE"), :])))
ag = np.load(ANALYSIS / "Ag.npy")
u_grid = np.load(ANALYSIS / "Ugrid.npy")
zero_ids = np.flatnonzero(np.abs(u_grid).sum(1) < 1e-12)
grid_zero = float(np.max(np.abs(ag[:, int(zero_ids[0]), :])))
meta.pop("grid_A0_vs_direct_A0_max_abs_score_difference", None)
meta["table"].pop("grid_A0_vs_direct_A0_max_abs_score_difference", None)
meta.setdefault("surface_crosschecks", {})
meta["surface_crosschecks"].update({
    "direct_P0_C3E_vs_stored_A9_max_abs_score_difference": max_p0,
    "direct_P1_Heuristic_vs_stored_A9_max_abs_score_difference": max_p1,
    "606_grid_noop_action_effect_max_abs": grid_zero,
    "meaning": "Ag/A9 arrays are score changes relative to A0; direct Simulator scores were differenced by the same-firm A0 before comparison.",
})

# Make zero simulator effects, missing values, and failures explicit in the prompt-safe tables.
tables = pd.read_parquet(OUT / "e2_simulator_10row_tables.parquet")
delta_cols = ["delta_" + k for k in FIN_FIELDS]
def effect_status(row):
    if not bool(row["simulator_ok"]):
        return "simulator_failure"
    if not bool(row["action_valid"]):
        return "invalid_action"
    values = pd.to_numeric(row[delta_cols], errors="coerce").to_numpy(dtype=float)
    if np.isnan(values).any():
        return "financial_item_missing"
    if row["row_label"] == "A0_no_action":
        return "A0_baseline_zero_change"
    if np.max(np.abs(values)) <= 1e-9:
        return "zero_financial_effect"
    return "nonzero_financial_effect"
tables["financial_effect_status"] = tables.apply(effect_status, axis=1)
tables["financial_items_missing_count"] = tables[delta_cols].isna().sum(axis=1)
tables.to_parquet(OUT / "e2_simulator_10row_tables.parquet", index=False)
ex_id = tables.table_id.iloc[0]
tables[tables.table_id == ex_id].to_csv(OUT / "e2_simulator_10row_example.csv", index=False, encoding="utf-8-sig")
prompt_cols = ["table_id", "firm_id", "row_id", "baseline_year", "outcome_year", "model", "replicate",
               "row_order", "row_label", "currency_unit", "horizon"] + delta_cols
prompt_view = tables[prompt_cols].copy()
prompt_view.to_parquet(OUT / "e2_simulator_10row_prompt_view.parquet", index=False)
prompt_view[prompt_view.table_id == ex_id].to_csv(OUT / "e2_simulator_10row_prompt_example.csv", index=False, encoding="utf-8-sig")
table_summary = tables.groupby("row_label", dropna=False).agg(
    rows=("row_label", "size"), sim_failures=("simulator_ok", lambda x: int((~x).sum())),
    invalid_actions=("action_valid", lambda x: int((~x).sum())),
    zero_effect=("financial_effect_status", lambda x: int((x == "zero_financial_effect").sum())),
    missing_financial_rows=("financial_effect_status", lambda x: int((x == "financial_item_missing").sum())),
).reset_index()
for col in delta_cols:
    summary = tables.groupby("row_label")[col].agg(["mean", "median", "min", "max"]).reset_index()
    summary = summary.rename(columns={x: f"{col}_{x}" for x in ["mean", "median", "min", "max"]})
    table_summary = table_summary.merge(summary, on="row_label", how="left")
table_summary.to_csv(OUT / "e2_simulator_10row_action_summary.csv", index=False, encoding="utf-8-sig")

# Reconstruct and save action selection/distribution labels, including the exact P5 selected option.
base_path = KIT / "archive" / "DEPLOYED_RELEASE" / "stage2_candidate_projection" / "phase_eval_candidate.parquet"
base = pd.read_parquet(base_path).reset_index(drop=True)
assert len(base) == N
base_states = [_row_to_firm_state(base.iloc[i]) for i in range(N)]
assert all(float(s.total_assets) > 0 for s in base_states)
p5_labels, p5_rows = [], []
unit_actions = {axis: unit_action(axis) for axis in AX}
for i in range(N):
    c3e_id = ids[i]
    cand = [("P0_C3E", actions["P0_RL_C3E"][i])]
    cand += [("unit_" + axis, unit_actions[axis]) for axis in AX]
    cand += [("P4", actions["P4_half_C3E_plus_COGS_minus0.5"][i])]
    b = surface_key[(i, baseline_sig)]
    bsum = float(b.state_t1_pretax_income) + float(b.state_t1_cash)
    vals = []
    for label, action in cand:
        r = surface_key[(i, sig_json(action))]
        numer = float(r.state_t1_pretax_income) + float(r.state_t1_cash) - bsum
        objective = numer / float(base_states[i].total_assets)
        vals.append((label, action, numer, objective))
    k = int(np.argmax([v[3] for v in vals]))
    label, action, numer, objective = vals[k]
    p5_labels.append(label)
    p5_rows.append({"row_id": i, "firm_id": str(base.iloc[i].firm_id), "C3E_action_id": c3e_id,
                    "P5_choice": label, "pretax_plus_cash_change_KRW_thousand": numer,
                    "initial_total_assets_KRW_thousand": float(base_states[i].total_assets),
                    "P5_objective_per_asset": objective, "action_T": float(np.abs(norm(action)).sum()),
                    "action_units_json": json.dumps({k: float(norm(action)[j]) for j, k in enumerate(AX)}, sort_keys=True, separators=(",", ":"))})
pd.DataFrame(p5_rows).to_csv(OUT / "b1_p5_selected_actions.csv", index=False, encoding="utf-8-sig")
print("P5 choices reconstructed", pd.Series(p5_labels).value_counts().to_dict())

labels_by_policy = {
    "P0_RL_C3E": ids,
    "P1_Heuristic_OE": ["OE"] * N,
    "P2_fixed_COGS_minus1": ["COGS_minus1"] * N,
    "P3_C3E_OE_to_P2": ["COGS_minus1" if c == "OE" else c for c in ids],
    "P4_half_C3E_plus_COGS_minus0.5": ["COGS_minus1" if c == "OE" else "P4_from_" + c for c in ids],
    "P5_financial_outcome_selector": p5_labels,
}
dist_rows, axis_rows = [], []
all_policy_actions = dict(actions)
all_policy_actions["P5_financial_outcome_selector"] = [
    # reconstruct from the persisted choice label and the candidate list used above
    (actions["P0_RL_C3E"][i] if p5_labels[i] == "P0_C3E" else
     actions["P4_half_C3E_plus_COGS_minus0.5"][i] if p5_labels[i] == "P4" else
     unit_actions[p5_labels[i].replace("unit_", "")]) for i in range(N)
]
for name, labs in labels_by_policy.items():
    counts = pd.Series(labs).value_counts()
    for lab, count in counts.items():
        dist_rows.append({"policy": name, "action_type": str(lab), "firm_count": int(count), "share": float(count / N)})
    u = np.vstack([norm(a) for a in all_policy_actions[name]])
    for j, axis in enumerate(AX):
        axis_rows.append({"policy": name, "axis": axis, "mean_signed_normalized_units": float(u[:, j].mean()),
                          "mean_absolute_normalized_units": float(np.abs(u[:, j]).mean())})
pd.DataFrame(dist_rows).to_csv(OUT / "b1_policy_action_distribution.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(axis_rows).to_csv(OUT / "b1_policy_mean_action_axes.csv", index=False, encoding="utf-8-sig")

# Add paired firm-cluster summaries for every fixed cost allocation against each row's original split.
cost = pd.read_parquet(OUT / "cost_allocation_counterfactual.parquet")
firm_scenario = cost.groupby(["firm_id", "scenario"], as_index=False).mean(numeric_only=True)
cf_pair_rows = []
fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
for f in fractions:
    scenario = f"COGS_SHARE_{f:.2f}"
    left = firm_scenario[firm_scenario.scenario == scenario].set_index("firm_id")
    right = firm_scenario[firm_scenario.scenario == "ORIGINAL"].set_index("firm_id")
    common = left.index.intersection(right.index)
    for metric in ["score_change_vs_original_" + e for e in EVALS] + ["financial_change_vs_original_" + x for x in FIN_FIELDS]:
        x = (left.loc[common, metric] - right.loc[common, metric]).dropna().to_numpy(dtype=float)
        if len(x) == 0:
            continue
        rng = np.random.default_rng(20261118)
        idx = rng.integers(0, len(x), size=(10000, len(x)))
        bs = x[idx].mean(axis=1)
        cf_pair_rows.append({"scenario": scenario, "cogs_share": f, "metric": metric,
                             "mean_paired_change_vs_original": float(x.mean()),
                             "ci95_low": float(np.percentile(bs, 2.5)), "ci95_high": float(np.percentile(bs, 97.5)),
                             "n_firms": int(len(x)), "bootstrap_unit": "firm",
                             "interpretation": "fixed action-intensity Simulator counterfactual; not a realized causal effect"})
pd.DataFrame(cf_pair_rows).to_csv(OUT / "cost_allocation_counterfactual_paired_summary.csv", index=False, encoding="utf-8-sig")

# Correct final metadata and record explicit zero-effect/missing distinctions.
meta["table"].update({
    "effect_status_counts": tables.financial_effect_status.value_counts().to_dict(),
    "missing_financial_cells": {k: int(tables["delta_" + k].isna().sum()) for k in FIN_FIELDS},
    "table_action_summary_file": "e2_simulator_10row_action_summary.csv",
    "prompt_safe_table_file": "e2_simulator_10row_tables.parquet",
    "prompt_view_file": "e2_simulator_10row_prompt_view.parquet",
    "E2_T_and_E2_RT_table_identity": "the same stored table payload is assigned to both conditions; no separate copies can diverge",
})
meta["cost_counterfactual"].update({
    "paired_summary_file": "cost_allocation_counterfactual_paired_summary.csv",
    "cogs_share_scenarios": [0.0, 0.25, 0.5, 0.75, 1.0],
    "original_split_included": True,
})
meta["p5_selection_counts"] = pd.Series(p5_labels).value_counts().to_dict()
meta["p5_criterion"] = "argmax[(delta pretax income + delta cash) / initial total assets]; P0_C3E first for exact ties; no Oracle score used"
meta["C2_status"] = "defer E2-B: Stage4 source code is behavior cloning, but C2_action-to-Stage4-checkpoint/policy-decision mapping is not verified in released artifacts"
meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

artifact_names = [
    "b1_policy_value_decomposition.csv", "b1_policy_contrasts_vs_rl_heuristic.csv", "b1_policy_action_distribution.csv",
    "b1_policy_mean_action_axes.csv", "b1_oracle_grid_upper_bound_separate.csv", "b1_p5_selected_actions.csv",
    "e2_simulator_10row_tables.parquet", "e2_simulator_10row_manifest.csv", "e2_simulator_10row_example.csv",
    "e2_simulator_10row_prompt_view.parquet", "e2_simulator_10row_prompt_example.csv",
    "e2_simulator_10row_action_summary.csv", "cost_allocation_counterfactual.parquet",
    "cost_allocation_counterfactual_summary.csv", "cost_allocation_counterfactual_paired_summary.csv",
    "b1_simulator_action_surface.parquet", "b1_run_metadata.json", "b1_run.py", "b1_finalize.py",
]
artifact_hashes = {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in artifact_names if (OUT / name).is_file()}
(OUT / "b1_artifact_sha256.json").write_text(json.dumps(artifact_hashes, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"crosschecks": meta["surface_crosschecks"], "table": meta["table"], "cost": meta["cost_counterfactual"],
                  "p5": meta["p5_selection_counts"], "artifacts": len(artifact_hashes)}, indent=2, ensure_ascii=False))

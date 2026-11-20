"""Exploratory E2 B1 diagnostics using the frozen Simulator and evaluator code.

No LLM calls, model training, or RL fitting are performed here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
TASK = HERE.parent
WORKSPACE = TASK.parent
ANALYSIS = TASK / "A_reproduction" / "R2_R8_replay_workspace" / "analysis_20261117"
KIT = TASK / "A_reproduction" / "E1_score_gate_replay" / "runroot_zip" / "THESIS_REPRO_KIT_v2.1.1_FINAL"
E31 = TASK / "A_reproduction" / "e31_policy_actions.parquet"
ABL = TASK / "source_inputs" / "github_release_ablation_20261102_readonly" / "contents" / "ablation_20261102" / "E1" / "observations" / "e1_observations.parquet"
OUT = HERE
OUT.mkdir(parents=True, exist_ok=True)

os.environ["KIT_ROOT"] = str(KIT).replace("\\", "/")
os.environ["HZ_SRC"] = str(KIT / "src").replace("\\", "/")
os.environ["ABL_E1"] = str(ABL).replace("\\", "/")
sys.path.insert(0, str(ANALYSIS))

import prep2 as P  # noqa: E402
from delta_train import feats_from_state  # noqa: E402
from fam2 import feats2, make_scorer  # noqa: E402
from hz import Harness  # noqa: E402


AX = list(P.AX)
EVALS = ["alpha", "beta", "gamma", "delta1", "delta2", "epsilon1", "epsilon2"]
N = 575
NAMES = {"GPT": "openai_gpt54mini", "Gemini": "google_gemini31flashlite"}
FIN_FIELDS = [
    "revenue", "operating_income", "pretax_income", "pure_interest_expense",
    "total_debt", "short_term_debt", "cash", "current_assets",
    "current_liabilities", "total_equity", "operating_cf",
]
STATE_FIELDS = list(dict.fromkeys(FIN_FIELDS + ["financial_cost", "total_assets", "total_liabilities", "non_current_assets", "non_current_liabilities"]))
UNIT_SIGNS = {
    "growth_capex_reduction_pct": 1.0,
    "deleveraging_total_debt_pct": 1.0,
    "refinancing_short_debt_pct": 1.0,
    "inv_turnover_chg": 1.0,
    "ar_turnover_chg": 1.0,
    "ap_turnover_chg": -1.0,
    "cogs_ratio_chg": -1.0,
    "sga_ratio_chg": -1.0,
}


def scalar(x):
    try:
        y = float(x)
        return y if np.isfinite(y) else np.nan
    except (TypeError, ValueError):
        return np.nan


def norm_vector(action):
    return np.array([scalar(action.get(k, 0.0)) / float(H.scale[k]) for k in AX], dtype=float)


def raw_action(units):
    u = np.asarray(units, dtype=float)
    return {k: float(u[j] * H.scale[k]) for j, k in enumerate(AX)}


def action_sig(action):
    return tuple(np.round(norm_vector(action), 10).tolist())


def one_unit(axis):
    u = np.zeros(len(AX), dtype=float)
    j = AX.index(axis)
    u[j] = UNIT_SIGNS[axis]
    return raw_action(u)


def valid_action(action):
    v = norm_vector(action)
    t = float(np.abs(v).sum())
    problems = []
    bounds = H.contract["action_bounds"]
    for k in AX:
        x = scalar(action.get(k, 0.0))
        lo, hi = bounds["action__" + k]
        if not np.isfinite(x) or x < lo - 1e-9 or x > hi + 1e-9:
            problems.append("bound:" + k)
    if t > 1.0 + 1e-8:
        problems.append("T>1")
    if v[AX.index("deleveraging_total_debt_pct")] > 1e-9 and v[AX.index("refinancing_short_debt_pct")] > 1e-9:
        problems.append("DL_and_RF")
    return not problems, t, ";".join(problems)


def build_scorers():
    m1 = joblib.load(ANALYSIS / "delta_models.joblib")
    m2 = joblib.load(ANALYSIS / "delta2_models.joblib")
    eps = make_scorer(m2)

    def dscore(features, model):
        x = np.asarray(features, dtype=float).reshape(1, -1)
        z = np.clip(np.where(np.isnan(x), model["med"][None, :], x), model["lo"][None, :], model["hi"][None, :])
        scale_score = lambda pred: np.clip(100.0 * (10.0 - pred) / 9.0, 0.0, 100.0)
        return [float(scale_score(model["d1"].predict((z - model["mu"]) / model["sd"])[0])),
                float(scale_score(model["d2"].predict(z)[0]))]

    return m1, eps, dscore


print("B1: initializing frozen Simulator and evaluator harness", flush=True)
t0 = time.time()
H = Harness()
print(f"B1: Harness ready in {time.time()-t0:.1f}s; firms={len(H.base)}", flush=True)
assert len(H.base) == N
assert len(P.a3) == N and P.a3.row_id.to_numpy().tolist() == list(range(N))
M1, EPS_SCORER, DELTA_SCORE = build_scorers()

record_cache: dict[tuple[int, tuple[float, ...]], dict] = {}
raw_for_sig: dict[tuple[float, ...], dict] = {}
label_for_sig: dict[tuple[float, ...], str] = {}


def remember(action, label=""):
    sig = action_sig(action)
    raw_for_sig.setdefault(sig, {k: float(action.get(k, 0.0)) for k in AX})
    if label and sig not in label_for_sig:
        label_for_sig[sig] = label
    return sig


def simulate(i, action, label=""):
    sig = remember(action, label)
    key = (int(i), sig)
    if key in record_cache:
        return record_cache[key]
    ok_action, t, issue = valid_action(action)
    if not ok_action:
        rec = {"ok": False, "error": "invalid_action:" + issue, "T": t, "action_issue": issue,
               "scores": np.full(7, np.nan), "state": {}, "accounting_error": np.nan}
        record_cache[key] = rec
        return rec
    try:
        out = H.sim_row(int(i), action, label=label or "B1", extra=True)
        state = out["_state"]
        s3 = H.score([out]).iloc[0]
        f1 = feats_from_state(state)
        f2 = feats2(state)
        d1, d2 = DELTA_SCORE(f1, M1)
        e1, e2 = EPS_SCORER([f2])[0].tolist()
        scores = np.array([float(s3["alpha"]), float(s3["beta"]), float(s3["gamma"]), d1, d2, float(e1), float(e2)])
        st = {k: scalar(getattr(state, k, np.nan)) for k in STATE_FIELDS}
        try:
            accounting = state.accounting_identity_check(tol=1.0)
            identity_error = scalar(accounting.get("diff"))
            identity_check = accounting.get("check", "skipped")
        except Exception:
            identity_error, identity_check = np.nan, "error"
        rec = {"ok": True, "error": "", "T": t, "action_issue": "", "scores": scores,
               "state": st, "year": int(state.year), "firm_id": str(state.firm_id),
               "accounting_error": identity_error, "accounting_check": identity_check}
    except Exception as exc:  # preserve failures instead of silently converting them to zero
        rec = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:240]}", "T": t,
               "action_issue": "", "scores": np.full(7, np.nan), "state": {},
               "accounting_error": np.nan, "accounting_check": "error"}
    record_cache[key] = rec
    if len(record_cache) % 1000 == 0:
        print(f"B1: simulated/cached {len(record_cache)} firm-action rows ({time.time()-t0:.0f}s)", flush=True)
    return rec


def base_record(i):
    return simulate(i, H.cands["A0"], "A0")


def financial_delta(i, rec):
    b = base_record(i)
    if not rec.get("ok") or not b.get("ok"):
        return {"delta_" + k: np.nan for k in FIN_FIELDS}
    result = {}
    for k in FIN_FIELDS:
        a, z = scalar(rec["state"].get(k)), scalar(b["state"].get(k))
        result["delta_" + k] = a - z if np.isfinite(a) and np.isfinite(z) else np.nan
    return result


def score_delta(i, rec):
    b = base_record(i)
    return rec["scores"] - b["scores"] if rec.get("ok") and b.get("ok") else np.full(7, np.nan)


def record_fin_row(i, action, label=""):
    rec = simulate(i, action, label)
    base = base_record(i)
    ok_action, t, issue = valid_action(action)
    d = financial_delta(i, rec)
    a0state = base.get("state", {})
    actionstate = rec.get("state", {})
    err = (scalar(actionstate.get("total_assets")) - scalar(actionstate.get("total_liabilities")) - scalar(actionstate.get("total_equity"))) if rec.get("ok") else np.nan
    err0 = (scalar(a0state.get("total_assets")) - scalar(a0state.get("total_liabilities")) - scalar(a0state.get("total_equity"))) if base.get("ok") else np.nan
    bal_delta_err = (d.get("delta_total_assets", np.nan) - (scalar(actionstate.get("total_liabilities")) - scalar(a0state.get("total_liabilities"))) - d.get("delta_total_equity", np.nan)) if rec.get("ok") and base.get("ok") else np.nan
    row = {"simulator_ok": bool(rec.get("ok")), "simulator_error": rec.get("error", ""),
           "action_valid": bool(ok_action), "action_issue": issue, "action_T": t,
           "outcome_year": rec.get("year", np.nan), "firm_id": rec.get("firm_id", str(H.base.iloc[i].firm_id)),
           "identity_error_t1": err, "identity_error_A0": err0,
           "identity_check_t1_1k_tolerance": rec.get("accounting_check", "skipped"),
           "balance_change_identity_error": bal_delta_err}
    row.update(d)
    return row


print("B1: forming P0-P5 policies and fixed one-axis candidates", flush=True)
unit_actions = {axis: one_unit(axis) for axis in AX}
p0_ids = P.a3.action_id.astype(str).to_numpy()
p0_actions = [H.cands[c] for c in p0_ids]
p1_actions = [H.cands["OE"] for _ in range(N)]
p2_action = one_unit("cogs_ratio_chg")
p2_actions = [p2_action for _ in range(N)]
p3_actions = [p2_action if c == "OE" else H.cands[c] for c in p0_ids]
p4_actions = []
for c in p0_ids:
    if c == "OE":
        p4_actions.append(p2_action)
    else:
        u = 0.5 * norm_vector(H.cands[c])
        u[AX.index("cogs_ratio_chg")] -= 0.5
        p4_actions.append(raw_action(u))

policy_actions = {
    "P0_RL_C3E": p0_actions,
    "P1_Heuristic_OE": p1_actions,
    "P2_fixed_COGS_minus1": p2_actions,
    "P3_C3E_OE_to_P2": p3_actions,
    "P4_half_C3E_plus_COGS_minus0.5": p4_actions,
}

for name, actions in policy_actions.items():
    bad = [(i, valid_action(a)[2], valid_action(a)[1]) for i, a in enumerate(actions) if not valid_action(a)[0]]
    print(f"B1: {name} invalid={len(bad)}; T range={min(valid_action(a)[1] for a in actions):.6f}..{max(valid_action(a)[1] for a in actions):.6f}", flush=True)
    if bad:
        print("INVALID EXAMPLES", name, bad[:5], flush=True)

print("B1: simulating baseline, P0-P4 and P5 financial-choice candidates", flush=True)
for i in range(N):
    base_record(i)
    for name in policy_actions:
        simulate(i, policy_actions[name][i], name)
    for axis, action in unit_actions.items():
        simulate(i, action, "unit_" + axis)

p5_actions, p5_choice, p5_objective = [], [], []
for i in range(N):
    candidate_actions = [("P0_C3E", p0_actions[i])]
    candidate_actions += [("unit_" + axis, unit_actions[axis]) for axis in AX]
    candidate_actions += [("P4", p4_actions[i])]
    b = base_record(i)
    base_assets = scalar(H.firm(i)[1].total_assets)
    vals = []
    for label, action in candidate_actions:
        rec = simulate(i, action, "P5_candidate_" + label)
        fd = financial_delta(i, rec)
        if rec.get("ok") and np.isfinite(base_assets) and abs(base_assets) > 1e-9:
            vals.append((label, action, (fd["delta_pretax_income"] + fd["delta_cash"]) / base_assets))
    if len(vals) != len(candidate_actions):
        p5_actions.append(H.cands["A0"])
        p5_choice.append("UNAVAILABLE")
        p5_objective.append(np.nan)
    else:
        j = int(np.argmax([v[2] for v in vals]))  # P0_C3E is first, so ties resolve to P0
        p5_choice.append(vals[j][0])
        p5_actions.append(vals[j][1])
        p5_objective.append(float(vals[j][2]))
policy_actions["P5_financial_outcome_selector"] = p5_actions
print("B1: P5 choices", pd.Series(p5_choice).value_counts().to_dict(), flush=True)

# Evaluate each distinct B1 policy action on every firm to support exact off-diagonal reassignment.
policy_sigs = {name: [remember(a, name) for a in acts] for name, acts in policy_actions.items()}
policy_sig_union = sorted(set(sig for arr in policy_sigs.values() for sig in arr))
print(f"B1: exact policy cross-firm score surface: {len(policy_sig_union)} distinct actions × {N} firms", flush=True)
for j, sig in enumerate(policy_sig_union, 1):
    action = raw_for_sig[sig]
    for i in range(N):
        simulate(i, action, label_for_sig.get(sig, "policy_action"))
    if j % 4 == 0 or j == len(policy_sig_union):
        print(f"B1: policy cross-surface actions {j}/{len(policy_sig_union)}", flush=True)

base_scores = np.vstack([base_record(i)["scores"] for i in range(N)])
surface = {sig: np.vstack([simulate(i, raw_for_sig[sig])["scores"] for i in range(N)]) for sig in policy_sig_union}
policy_parts = {}
for name, sigs in policy_sigs.items():
    sigs = list(sigs)
    own_abs = np.vstack([surface[sigs[i]][i] for i in range(N)])
    own = own_abs - base_scores
    unique = sorted(set(sigs))
    counts = [sigs.count(sig) for sig in unique]
    count_by_sig = {str(sig): int(c) for sig, c in zip(unique, counts)}
    sum_all = np.zeros((N, 7), dtype=float)
    for sig, count in zip(unique, counts):
        sum_all += surface[sig] * int(count)
    comp = (sum_all - own_abs) / (N - 1) - base_scores
    assignment = own - comp
    policy_parts[name] = {"V_i": own, "C_i": comp, "A_i": assignment, "sigs": sigs}

policy_value_rows = []
for name, parts in policy_parts.items():
    for j, evaluator in enumerate(EVALS):
        policy_value_rows.append({
            "policy": name, "evaluator": evaluator,
            "total_value_V": float(np.nanmean(parts["V_i"][:, j])),
            "composition_value_C": float(np.nanmean(parts["C_i"][:, j])),
            "assignment_value_A": float(np.nanmean(parts["A_i"][:, j])),
            "decomposition_error_V_minus_C_minus_A": float(np.nanmean(parts["V_i"][:, j] - parts["C_i"][:, j] - parts["A_i"][:, j])),
        })
pd.DataFrame(policy_value_rows).to_csv(OUT / "b1_policy_value_decomposition.csv", index=False, encoding="utf-8-sig")

rng = np.random.default_rng(20261118)
boot_idx = rng.integers(0, N, size=(10000, N), endpoint=False)
contrast_rows = []
for name, parts in policy_parts.items():
    if name in ("P0_RL_C3E", "P1_Heuristic_OE"):
        continue
    for reference in ("P0_RL_C3E", "P1_Heuristic_OE"):
        for component in ("V_i", "C_i", "A_i"):
            delta = parts[component] - policy_parts[reference][component]
            for j, evaluator in enumerate(EVALS):
                x = delta[:, j]
                bs = x[boot_idx].mean(axis=1)
                contrast_rows.append({
                    "policy": name, "reference": reference, "component": component[0], "evaluator": evaluator,
                    "difference": float(np.nanmean(x)), "ci95_low": float(np.nanpercentile(bs, 2.5)),
                    "ci95_high": float(np.nanpercentile(bs, 97.5)), "n_firms": int(np.isfinite(x).sum()),
                    "ci_note": "paired firm bootstrap; action-distribution pool held fixed",
                })
pd.DataFrame(contrast_rows).to_csv(OUT / "b1_policy_contrasts_vs_rl_heuristic.csv", index=False, encoding="utf-8-sig")

policy_dist_rows = []
for name, sigs in policy_sigs.items():
    vc = pd.Series([label_for_sig.get(sig, str(sig)) for sig in sigs]).value_counts()
    u = np.vstack([norm_vector(a) for a in policy_actions[name]])
    for action_label, count in vc.items():
        policy_dist_rows.append({"policy": name, "action_label": action_label, "firm_count": int(count), "share": float(count / N)})
    for j, axis in enumerate(AX):
        policy_dist_rows.append({"policy": name, "action_label": "MEAN_NORMALIZED_" + axis, "firm_count": np.nan, "share": float(u[:, j].mean())})
pd.DataFrame(policy_dist_rows).to_csv(OUT / "b1_policy_action_distribution.csv", index=False, encoding="utf-8-sig")

# Separate, explicitly in-sample diagnostic: pick the best of 606 grid actions per evaluator.
grid = np.load(ANALYSIS / "Ag.npy")
U = np.load(ANALYSIS / "Ugrid.npy")
assert grid.shape == (N, len(U), 7)
zero_ids = np.flatnonzero(np.abs(U).sum(1) < 1e-12)
assert len(zero_ids) == 1
zero_id = int(zero_ids[0])
grid_base = grid[:, zero_id, :]
a9_surface = np.load(ANALYSIS / "A9.npy")
direct_p0_a9_max_abs_diff = float(np.nanmax(np.abs(policy_parts["P0_RL_C3E"]["V_i"] - a9_surface[np.arange(N), P.RLID, :])))
direct_p1_a9_max_abs_diff = float(np.nanmax(np.abs(policy_parts["P1_Heuristic_OE"]["V_i"] - a9_surface[:, P.IDS.index("OE"), :])))
grid_zero_effect_max_abs = float(np.nanmax(np.abs(grid_base)))
oracle_rows = []
for chosen_eval, evaluator in enumerate(EVALS):
    selected = np.argmax(grid[:, :, chosen_eval], axis=1)
    own_abs = grid[np.arange(N), selected, :]
    own = own_abs - grid_base
    counts = np.bincount(selected, minlength=grid.shape[1])
    sum_all = np.einsum("iae,a->ie", grid, counts)
    comp = (sum_all - own_abs) / (N - 1) - grid_base
    assign = own - comp
    for j, score_eval in enumerate(EVALS):
        oracle_rows.append({
            "selection_evaluator": evaluator, "scoring_evaluator": score_eval,
            "total_value_V": float(own[:, j].mean()), "composition_value_C": float(comp[:, j].mean()),
            "assignment_value_A": float(assign[:, j].mean()),
            "selection": "per-firm argmax on same 606-action grid; in-sample; exploratory upper bound only",
        })
pd.DataFrame(oracle_rows).to_csv(OUT / "b1_oracle_grid_upper_bound_separate.csv", index=False, encoding="utf-8-sig")
print(f"B1: direct P0/P1 vs stored A9 max score differences={direct_p0_a9_max_abs_diff:.3e}/{direct_p1_a9_max_abs_diff:.3e}; grid no-op delta max={grid_zero_effect_max_abs:.3e}", flush=True)

# Full 10-row tables. The archived high-strict C4 output is used only as a frozen draft proxy;
# it does not stand in for a future E2-generated first draft.
table_rows = []
unit_labels = {
    "growth_capex_reduction_pct": "capex_reduction_1_unit",
    "deleveraging_total_debt_pct": "deleveraging_1_unit",
    "refinancing_short_debt_pct": "refinancing_1_unit",
    "inv_turnover_chg": "inventory_turnover_plus_1_unit",
    "ar_turnover_chg": "receivables_turnover_plus_1_unit",
    "ap_turnover_chg": "payables_turnover_minus_1_unit",
    "cogs_ratio_chg": "cogs_ratio_minus_1_unit",
    "sga_ratio_chg": "sga_ratio_minus_1_unit",
}
table_manifest_rows = []
for model in ("GPT", "Gemini"):
    for rep in (1, 2):
        frame = P.get("free8", "B1", "high", "C4", model, rep=rep)
        drafts = P.acts(frame)
        usable = frame.strict_usable.to_numpy(dtype=bool)
        for i in range(N):
            draft_action = drafts[i] if usable[i] else H.cands["A0"]
            candidates = [("A0_no_action", H.cands["A0"], "baseline")]
            candidates.append(("draft_C4_frozen_proxy" if usable[i] else "draft_unusable_A0_fallback", draft_action, "draft"))
            candidates += [(unit_labels[axis], unit_actions[axis], axis) for axis in AX]
            table_id = f"{H.base.iloc[i].firm_id}__{model}__rep{rep}"
            per_table = []
            for row_order, (row_label, action, axis) in enumerate(candidates, start=1):
                simrow = record_fin_row(i, action, "E2_table_" + row_label)
                units = norm_vector(action)
                payload = {
                    "table_id": table_id, "firm_id": str(H.base.iloc[i].firm_id), "row_id": i,
                    "baseline_year": int(H.firm(i)[1].year), "outcome_year": simrow["outcome_year"],
                    "model": model, "replicate": rep, "table_pair": "E2-T_AND_E2-RT_SHARED",
                    "row_order": row_order, "row_label": row_label, "action_axis": axis,
                    "draft_proxy_source": "archived HIGH/free8/B1/C4; not an E2-generated draft" if axis == "draft" else "",
                    "draft_source_strict_usable": bool(usable[i]) if axis == "draft" else np.nan,
                    "action_units_json": json.dumps({k: float(units[j]) for j, k in enumerate(AX)}, sort_keys=True, separators=(",", ":")),
                    "currency_unit": "KRW_thousand", "horizon": "decision_year_plus_one",
                    **simrow,
                }
                # The prompt-ready table contains no evaluator score or model ranking fields.
                table_rows.append(payload)
                per_table.append({k: payload[k] for k in ["row_order", "row_label", "action_units_json"] + ["delta_" + f for f in FIN_FIELDS] + ["simulator_ok", "action_valid"]})
            canonical = json.dumps(per_table, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            table_manifest_rows.append({"table_id": table_id, "firm_id": str(H.base.iloc[i].firm_id), "row_id": i,
                                        "model": model, "replicate": rep,
                                        "condition_pair": "E2-T/E2-RT", "rows": len(per_table),
                                        "table_payload_sha256": hashlib.sha256(canonical).hexdigest(),
                                        "draft_proxy_usable": bool(usable[i])})
tables = pd.DataFrame(table_rows)
tables.to_parquet(OUT / "e2_simulator_10row_tables.parquet", index=False)
pd.DataFrame(table_manifest_rows).to_csv(OUT / "e2_simulator_10row_manifest.csv", index=False, encoding="utf-8-sig")
example_id = tables.table_id.iloc[0]
tables[tables.table_id == example_id].to_csv(OUT / "e2_simulator_10row_example.csv", index=False, encoding="utf-8-sig")

# Counterfactual reallocation, tied to the exact e33 eligible observations and signed actions.
print("B1: preparing matched e33 cost-allocation Simulator counterfactuals", flush=True)
e31 = pd.read_parquet(E31)
raw = P.S[(P.S["analysis_arm"] == "HIGH_STRICT_ITT") & (P.S["mode"] == "free8") &
          (P.S["budget"] == "B1") & (P.S["reasoning_arm"] == "HIGH")].copy()
join_keys = ["row_id", "model_key", "policy_condition", "info", "replicate"]
merged = e31.merge(raw[join_keys + ["strict_itt_action_json"]], on=join_keys, how="left", validate="one_to_one")
assert len(merged) == len(e31) == 11500 and merged.strict_itt_action_json.notna().all()
merged["cost_units"] = merged.u_cogs_ratio_chg + merged.u_sga_ratio_chg
merged["other_units"] = merged["T"] - merged["cost_units"]
eligible = merged[merged.strict_usable & (merged.cost_units >= 0.9) & (merged.other_units < 0.02)].copy()
eligible["raw_action"] = eligible.strict_itt_action_json.map(json.loads)
eligible["raw_units"] = eligible.raw_action.map(norm_vector)
eligible["original_f"] = eligible.u_cogs_ratio_chg / eligible.cost_units
eligible["source_T_check"] = eligible.raw_units.map(lambda v: float(np.abs(v).sum()))
source_T_error = float((eligible.source_T_check - eligible["T"]).abs().max())
sign_cogs = eligible.raw_units.map(lambda v: float(v[AX.index("cogs_ratio_chg")]))
sign_sga = eligible.raw_units.map(lambda v: float(v[AX.index("sga_ratio_chg")]))
assert (sign_cogs <= 1e-9).all() and (sign_sga <= 1e-9).all()
assert source_T_error <= 1e-7
cost_rows = []
fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
for e in eligible.itertuples(index=False):
    orig = dict(e.raw_action)
    orig_units = np.asarray(e.raw_units, dtype=float)
    cogs_idx, sga_idx = AX.index("cogs_ratio_chg"), AX.index("sga_ratio_chg")
    cost_total = float(e.cost_units)
    variants = [("ORIGINAL", float(e.original_f), orig)]
    for f in fractions:
        units = orig_units.copy()
        units[cogs_idx] = -float(f) * cost_total
        units[sga_idx] = -(1.0 - float(f)) * cost_total
        variants.append((f"COGS_SHARE_{f:.2f}", float(f), raw_action(units)))
    original_rec = simulate(int(e.row_id), orig, "costmix_original")
    original_fin = financial_delta(int(e.row_id), original_rec)
    original_sc = score_delta(int(e.row_id), original_rec)
    for scenario, f, action in variants:
        rec = simulate(int(e.row_id), action, "costmix_" + scenario)
        ok_action, action_t, issue = valid_action(action)
        fd = financial_delta(int(e.row_id), rec)
        sd = score_delta(int(e.row_id), rec)
        row = {
            "source_row_id": int(e.row_id), "firm_id": str(H.base.iloc[int(e.row_id)].firm_id),
            "model_key": e.model_key, "policy_condition": e.policy_condition, "info": e.info, "replicate": int(e.replicate),
            "scenario": scenario, "cogs_share": f, "original_cogs_share": float(e.original_f),
            "original_cost_units": cost_total, "source_T": float(e.T), "counterfactual_T": action_t,
            "T_preservation_error": action_t - float(e.T), "other_axes_fixed_exactly": bool(np.allclose(norm_vector(action)[[j for j in range(8) if j not in (cogs_idx, sga_idx)]], orig_units[[j for j in range(8) if j not in (cogs_idx, sga_idx)]], atol=1e-10, rtol=0)),
            "action_valid": bool(ok_action), "action_issue": issue, "simulator_ok": bool(rec.get("ok")), "simulator_error": rec.get("error", ""),
            "currency_unit": "KRW_thousand", "outcome_year": rec.get("year", np.nan),
            **fd,
        }
        for j, ev in enumerate(EVALS):
            row["score_gain_vs_A0_" + ev] = float(sd[j])
            row["score_change_vs_original_" + ev] = float(sd[j] - original_sc[j])
        for k in FIN_FIELDS:
            row["financial_change_vs_original_" + k] = float(fd["delta_" + k] - original_fin["delta_" + k])
        cost_rows.append(row)
cost = pd.DataFrame(cost_rows)
cost.to_parquet(OUT / "cost_allocation_counterfactual.parquet", index=False)

cost_summary_rows = []
firm_mean = cost.groupby(["firm_id", "scenario"], dropna=False).mean(numeric_only=True).reset_index()
for scenario in ["ORIGINAL"] + [f"COGS_SHARE_{f:.2f}" for f in fractions]:
    part = firm_mean[firm_mean.scenario == scenario]
    for ev in EVALS:
        cost_summary_rows.append({"scenario": scenario, "cogs_share": (np.nan if scenario == "ORIGINAL" else float(scenario.split("_")[-1])),
                                  "metric": "score_gain_vs_A0_" + ev, "mean_across_firm_means": float(part["score_gain_vs_A0_" + ev].mean()),
                                  "n_firms": int(part.firm_id.nunique()), "interpretation": "Simulator output under fixed action budget; exploratory counterfactual"})
    for k in ["pretax_income", "pure_interest_expense", "total_debt", "cash", "operating_cf", "revenue"]:
        cost_summary_rows.append({"scenario": scenario, "cogs_share": (np.nan if scenario == "ORIGINAL" else float(scenario.split("_")[-1])),
                                  "metric": "delta_" + k, "mean_across_firm_means": float(part["delta_" + k].mean()),
                                  "n_firms": int(part.firm_id.nunique()), "interpretation": "KRW thousand vs same-firm A0 Simulator trajectory"})

firm_pivot = firm_mean.pivot(index="firm_id", columns="scenario")
for s1, s0 in [("COGS_SHARE_1.00", "COGS_SHARE_0.00"), ("COGS_SHARE_0.75", "COGS_SHARE_0.25")]:
    for ev in EVALS:
        col = "score_gain_vs_A0_" + ev
        if (col, s1) in firm_pivot.columns and (col, s0) in firm_pivot.columns:
            x = (firm_pivot[(col, s1)] - firm_pivot[(col, s0)]).dropna().to_numpy()
            # Bootstrap at the unique firm level for this paired endpoint contrast.
            ids = np.random.default_rng(20261118).integers(0, len(x), size=(10000, len(x)))
            boot = x[ids].mean(axis=1)
            cost_summary_rows.append({"scenario": f"{s1} minus {s0}", "cogs_share": np.nan,
                                      "metric": "paired_difference_" + ev, "mean_across_firm_means": float(x.mean()),
                                      "ci95_low": float(np.percentile(boot, 2.5)), "ci95_high": float(np.percentile(boot, 97.5)),
                                      "n_firms": int(len(x)), "interpretation": "within-firm Simulator contrast; does not estimate realized causal effects"})
pd.DataFrame(cost_summary_rows).to_csv(OUT / "cost_allocation_counterfactual_summary.csv", index=False, encoding="utf-8-sig")

# Persist the exact Simulator surface used by all policy and financial-table results.
surface_rows = []
for (i, sig), rec in record_cache.items():
    if rec.get("ok"):
        row = {"row_id": int(i), "firm_id": str(H.base.iloc[int(i)].firm_id),
               "action_signature": json.dumps(list(sig), separators=(",", ":")),
               "action_label_first_seen": label_for_sig.get(sig, ""), "action_T": float(rec["T"]),
               "simulator_ok": True, "accounting_error_t1": rec.get("accounting_error", np.nan),
               "accounting_check_t1_1k_tolerance": rec.get("accounting_check", "skipped")}
        for j, ev in enumerate(EVALS):
            row["absolute_score_" + ev] = float(rec["scores"][j])
        for k, v in rec["state"].items():
            row["state_t1_" + k] = v
        surface_rows.append(row)
pd.DataFrame(surface_rows).to_parquet(OUT / "b1_simulator_action_surface.parquet", index=False)

# QA and compact metadata.
table_sha_count = int(pd.DataFrame(table_manifest_rows).table_payload_sha256.nunique())
table_sizes = tables.groupby("table_id").size()
table_zero = np.nanmax(np.abs(tables.loc[tables.row_label == "A0_no_action", ["delta_" + f for f in FIN_FIELDS]].to_numpy(dtype=float)))
policy_invalid = {name: int(sum(not valid_action(a)[0] for a in acts)) for name, acts in policy_actions.items()}
cf_t_error = float(cost.T_preservation_error.abs().max())
cf_other_fixed = bool(cost.other_axes_fixed_exactly.all())
table_sim_fail = int((~tables.simulator_ok).sum())
table_action_fail = int((~tables.action_valid).sum())
identity_fail = int((tables.identity_check_t1_1k_tolerance == "fail").sum())
delta_identity_fail = int((tables.balance_change_identity_error.abs() > 1.0).sum())
meta = {
    "status": "B1 exploratory calculation complete",
    "n_firms": N, "evaluators": EVALS, "policies": list(policy_actions),
    "source_hashes": {
        str(E31.name): hashlib.sha256(E31.read_bytes()).hexdigest(),
        "Ag.npy": hashlib.sha256((ANALYSIS / "Ag.npy").read_bytes()).hexdigest(),
        "A9.npy": hashlib.sha256((ANALYSIS / "A9.npy").read_bytes()).hexdigest(),
        "Ugrid.npy": hashlib.sha256((ANALYSIS / "Ugrid.npy").read_bytes()).hexdigest(),
        "e33_costmix.py": hashlib.sha256((TASK / "source_inputs" / "tech_review_20261118" / "e33_costmix.py").read_bytes()).hexdigest(),
        "e34_costmix_noref.py": hashlib.sha256((TASK / "source_inputs" / "tech_review_20261118" / "e34_costmix_noref.py").read_bytes()).hexdigest(),
        "candidate_action_contract_v4_3.json": hashlib.sha256((KIT / "archive" / "DEPLOYED_RELEASE" / "stage2_candidate_projection" / "candidate_action_contract_v4_3.json").read_bytes()).hexdigest(),
    },
    "policy_invalid_count": policy_invalid,
    "policy_action_intensity": {name: {"min": float(min(valid_action(a)[1] for a in acts)), "max": float(max(valid_action(a)[1] for a in acts))} for name, acts in policy_actions.items()},
    "table": {"rows": int(len(tables)), "unique_tables": int(tables.table_id.nunique()), "rows_per_table_min": int(table_sizes.min()),
              "rows_per_table_max": int(table_sizes.max()), "unique_table_payload_hashes": table_sha_count,
              "simulator_failures": table_sim_fail, "invalid_actions": table_action_fail,
              "A0_max_abs_delta_KRW_thousand": float(table_zero), "accounting_identity_fail_at_1k": identity_fail,
              "balance_change_identity_residual_over_1k": delta_identity_fail,
              "draft_unusable_fallback_rows": int((tables.row_label == "draft_unusable_A0_fallback").sum()),
              "direct_P0_vs_A9_max_abs_score_difference": direct_p0_a9_max_abs_diff,
              "direct_P1_vs_A9_max_abs_score_difference": direct_p1_a9_max_abs_diff,
              "grid_noop_delta_max_abs": grid_zero_effect_max_abs},
    "cost_counterfactual": {"eligible_rows": int(len(eligible)), "unique_firms": int(eligible.row_id.nunique()),
                            "both_cost_axes_reductions": int(((sign_cogs < -1e-12) & (sign_sga < -1e-12)).sum()),
                            "one_cost_axis_zero": int(((sign_cogs.abs() <= 1e-12) | (sign_sga.abs() <= 1e-12)).sum()),
                            "source_action_T_max_abs_check_error": source_T_error,
                            "T_preservation_max_abs_error": cf_t_error, "other_axes_fixed": cf_other_fixed,
                            "action_invalid_rows": int((~cost.action_valid).sum()), "simulator_failures": int((~cost.simulator_ok).sum()),
                            "scenarios_per_observation": 6, "rows": int(len(cost))},
    "simulator_action_surface_rows": int(len(surface_rows)),
    "simulator_cached_firm_action_count": int(len(record_cache)),
    "elapsed_seconds": round(time.time() - t0, 2),
    "note": "All findings are exploratory and Simulator-based. The C4 draft row is a frozen proxy. Oracle grid maxima are a separate in-sample diagnostic and not deployment policies or prompt references.",
}
(OUT / "b1_run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
print(f"B1 complete in {time.time()-t0:.1f}s; outputs at {OUT}", flush=True)

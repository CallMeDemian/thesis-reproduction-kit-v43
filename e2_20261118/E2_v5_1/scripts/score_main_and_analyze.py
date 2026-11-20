from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


E2 = Path(__file__).resolve().parents[1]
TASK = E2.parent
WORKSPACE = TASK.parent
ANALYSIS = TASK / "A_reproduction/R2_R8_replay_workspace/analysis_20261117"
KIT = TASK / "A_reproduction/E1_score_gate_replay/runroot_zip/THESIS_REPRO_KIT_v2.1.1_FINAL"
B1 = TASK / "B1_diagnostics"
OUT = E2 / "results"
SEED = 20261118
B = 10000
N = 575
AX = ["growth_capex_reduction_pct", "deleveraging_total_debt_pct", "refinancing_short_debt_pct",
      "inv_turnover_chg", "ar_turnover_chg", "ap_turnover_chg", "cogs_ratio_chg", "sga_ratio_chg"]
SCORE_METRICS = ["alpha", "beta", "gamma"]
FIN_FIELDS = ["revenue", "operating_income", "pretax_income", "pure_interest_expense", "total_debt",
              "short_term_debt", "cash", "current_assets", "current_liabilities", "total_equity", "operating_cf"]
NAMES = {"openai_gpt54mini": "GPT", "google_gemini31flashlite": "Gemini"}
ACTION_JSON_ORDER = AX

os.environ["KIT_ROOT"] = str(KIT).replace("\\", "/")
os.environ["HZ_SRC"] = str(KIT / "src").replace("\\", "/")
os.environ["ABL_E1"] = str(TASK / "source_inputs/github_release_ablation_20261102_readonly/contents/ablation_20261102/E1/observations/e1_observations.parquet").replace("\\", "/")
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))

import prep2 as P  # noqa: E402
from hz import Harness  # noqa: E402


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def action_signature(action: dict[str, float], h: Any) -> tuple[float, ...]:
    return tuple(np.round([float(action.get(k, 0.0)) / float(h.scale[k]) for k in AX], 10).tolist())


def sig_text(sig: tuple[float, ...]) -> str:
    return json.dumps(list(sig), separators=(",", ":"))


def norm_action(action: dict[str, float], h: Any) -> np.ndarray:
    return np.asarray([float(action.get(k, 0.0)) / float(h.scale[k]) for k in AX], dtype=float)


def raw_action(units: np.ndarray, h: Any) -> dict[str, float]:
    return {k: float(units[j] * float(h.scale[k])) for j, k in enumerate(AX)}


def validate_action(action: dict[str, float], h: Any) -> tuple[bool, float, str]:
    u = norm_action(action, h)
    t = float(np.abs(u).sum())
    problems = []
    bounds = h.contract["action_bounds"]
    for k in AX:
        x = float(action.get(k, 0.0))
        lo, hi = bounds["action__" + k]
        if not np.isfinite(x) or x < float(lo) - 1e-9 or x > float(hi) + 1e-9:
            problems.append("bound:" + k)
    if t > 1.0 + 1e-8:
        problems.append("T>1")
    if u[1] > 1e-9 and u[2] > 1e-9:
        problems.append("DL_and_RF")
    return not problems, t, ";".join(problems)


def fixed_policy_actions(h: Any) -> dict[str, list[dict[str, float]]]:
    ids = P.a3.sort_values("row_id").action_id.astype(str).tolist()
    if len(ids) != N:
        raise RuntimeError("Frozen RL action map does not contain 575 rows")
    p0 = [dict(h.cands[c]) for c in ids]
    p1 = [dict(h.cands["OE"]) for _ in range(N)]
    unit = np.zeros(len(AX), dtype=float)
    unit[AX.index("cogs_ratio_chg")] = -1.0
    p2_action = raw_action(unit, h)
    p2 = [dict(p2_action) for _ in range(N)]
    p3 = [dict(p2_action) if c == "OE" else dict(h.cands[c]) for c in ids]
    p4 = []
    for c in ids:
        if c == "OE":
            p4.append(dict(p2_action))
        else:
            u = 0.5 * norm_action(h.cands[c], h)
            u[AX.index("cogs_ratio_chg")] -= 0.5
            p4.append(raw_action(u, h))
    p5_df = pd.read_csv(B1 / "b1_p5_selected_actions.csv")
    if len(p5_df) != N or p5_df.row_id.astype(int).tolist() != list(range(N)):
        raise RuntimeError("Frozen P5 selected action rows are missing or misordered")
    p5 = []
    for r in p5_df.itertuples(index=False):
        u = json.loads(r.action_units_json)
        p5.append({k: float(u[k]) * float(h.scale[k]) for k in AX})
    return {
        "P0_RL_C3E": p0,
        "P1_Heuristic_OE": p1,
        "P2_fixed_COGS_minus1": p2,
        "P3_C3E_OE_to_P2": p3,
        "P4_half_C3E_plus_COGS_minus0.5": p4,
        "P5_financial_outcome_selector": p5,
    }


def action_items_from_main() -> list[dict[str, Any]]:
    frame = pd.read_parquet(OUT / "main_policy_responses.parquet")
    if len(frame) != 9200:
        raise RuntimeError("Main response parser must yield 9,200 policy outcomes")
    items = []
    for r in frame.itertuples(index=False):
        if not r.strict_itt_action_json:
            raise RuntimeError(f"Delivered response has no STRICT ITT action: {r.request_id}")
        action = json.loads(r.strict_itt_action_json)
        items.append({"source": "main", "policy": str(r.condition), "model_key": str(r.model_key),
                      "replicate": int(r.replicate), "row_id": int(r.row_id), "firm_key": str(r.firm_key),
                      "request_id": str(r.request_id), "strict_usable": bool(r.strict_usable), "action": action})
    expected = {c: 2300 for c in ["N", "R", "T", "RT"]}
    if frame.condition.value_counts().to_dict() != expected:
        raise RuntimeError("Main policy conditions do not balance to 2,300 per condition")
    return items


def action_items_from_pilot() -> list[dict[str, Any]]:
    path = OUT / "pilot_actions.parquet"
    frame = pd.read_parquet(path)
    provider_keys = {"openai": "openai_gpt54mini", "google": "google_gemini31flashlite"}
    items = []
    for r in frame.itertuples(index=False):
        if not r.strict_itt_action_json:
            raise RuntimeError(f"Pilot action missing for auxiliary analysis: {r.request_id}")
        items.append({"source": "pilot_aux", "policy": str(r.condition), "model_key": provider_keys[str(r.provider)],
                      "replicate": 1, "row_id": int(r.row_id), "firm_key": str(r.firm_key),
                      "request_id": str(r.request_id), "strict_usable": bool(r.strict_usable),
                      "action": json.loads(r.strict_itt_action_json)})
    if len(items) != 800:
        raise RuntimeError(f"Expected 800 pilot auxiliary policy actions, found {len(items)}")
    return items


def bootstrap_indices(n: int = N, b: int = B, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(np.random.PCG64(seed)).integers(0, n, size=(b, n), endpoint=False)


def mean_ci(x: np.ndarray, indices: np.ndarray) -> tuple[float, float, float, int]:
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x)
    if not valid.any():
        return float("nan"), float("nan"), float("nan"), 0
    if valid.all():
        bs = x[indices].mean(axis=1)
    else:
        bs = np.nanmean(x[indices], axis=1)
    bs = bs[np.isfinite(bs)]
    return float(np.nanmean(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(valid.sum())


def centered_bootstrap_p(x: np.ndarray, indices: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    if indices.shape[1] != len(x):
        idx = np.random.default_rng(np.random.PCG64(SEED + len(x))).integers(0, len(x), size=(B, len(x)))
    else:
        idx = indices
    observed = abs(float(x.mean()))
    centered = x - x.mean()
    bs = centered[idx].mean(axis=1)
    return float((np.count_nonzero(np.abs(bs) >= observed) + 1) / (len(bs) + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def read_derangements() -> list[np.ndarray]:
    path = E2 / "manifests_main/r8_derangements_seed20261118.csv"
    meta = json.loads((E2 / "logs_main/r8_derangement_freeze.json").read_text(encoding="utf-8"))
    if meta.get("status") != "FROZEN_BEFORE_E2_SCORING" or hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("sha256"):
        raise RuntimeError("R8 derangement freeze hash mismatch")
    df = pd.read_csv(path)
    if df.derangement.nunique() != 100 or len(df) != 100 * N:
        raise RuntimeError("R8 derangement manifest must have 100×575 pairs")
    out = []
    for d, g in df.groupby("derangement", sort=True):
        g = g.sort_values("source_row_id")
        if g.source_row_id.astype(int).tolist() != list(range(N)):
            raise RuntimeError(f"Derangement {d} source indices malformed")
        p = g.target_row_id.astype(int).to_numpy()
        if sorted(p.tolist()) != list(range(N)) or np.any(p == np.arange(N)):
            raise RuntimeError(f"Derangement {d} is not a derangement permutation")
        out.append(p)
    return out


def build_harness():
    h = Harness()
    if len(h.base) != N or P.a3.row_id.astype(int).tolist() != list(range(N)):
        raise RuntimeError("Harness/source firm order mismatch")
    return h


def run_analysis() -> None:
    integrity_path = E2 / "logs_main/main_response_integrity.json"
    if not integrity_path.exists():
        raise RuntimeError("Response integrity report is missing")
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if integrity.get("status") != "RESPONSE_INTEGRITY_COMPLETE" or integrity.get("scores_computed") is not False:
        raise RuntimeError("Scoring blocked until response integrity is complete and score-free")
    if integrity.get("new_api_requests") != 5150 or integrity.get("new_api_response_ids") != 5150:
        raise RuntimeError("Expected all 5,150 new API responses before scoring")

    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    h = build_harness()
    main_items = action_items_from_main()
    pilot_items = action_items_from_pilot()
    fixed_actions = fixed_policy_actions(h)
    fixed_items = [{"source": "fixed_policy", "policy": name, "model_key": "ALL", "replicate": 1,
                    "row_id": i, "firm_key": str(h.base.iloc[i].firm_id), "request_id": None,
                    "strict_usable": True, "action": actions[i]}
                   for name, actions in fixed_actions.items() for i in range(N)]
    for item in main_items + pilot_items + fixed_items:
        ok, t, issue = validate_action(item["action"], h)
        if not ok:
            raise RuntimeError(f"Action outside frozen contract before scoring: {item['source']}/{item['policy']}/{item['row_id']} {issue} T={t}")
        item["T"] = t
        item["signature"] = action_signature(item["action"], h)
        item["action_sha256"] = sha256_bytes(sig_text(item["signature"]).encode("utf-8"))

    all_direct = main_items + pilot_items + fixed_items
    action_manifest = pd.DataFrame([{k: v for k, v in x.items() if k not in {"action"}} for x in all_direct])
    action_manifest.to_csv(E2 / "manifests_main/score_action_manifest.csv", index=False, encoding="utf-8-sig")
    action_manifest_hash = hashlib.sha256((E2 / "manifests_main/score_action_manifest.csv").read_bytes()).hexdigest()
    derangements = read_derangements()

    metric_idx = {metric: i for i, metric in enumerate(SCORE_METRICS)}
    baseline_action = dict(h.cands["A0"])
    scenario_cache: dict[str, dict[tuple[int, tuple[float, ...]], dict[str, Any]]] = {s: {} for s in ["S0", "S1", "S2"]}
    s1_audit = {"calls": 0, "principal_sum": 0.0, "surcharge_sum": 0.0, "max_formula_residual": 0.0}
    s2_audit: list[dict[str, Any]] = []
    progress_path = E2 / "logs_main/scoring_progress.log"

    def log(message: str) -> None:
        line = f"{pd.Timestamp.now(tz='UTC').isoformat()} {message}"
        print(line, flush=True)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def ensure_scores(scenario: str, pairs: dict[tuple[int, tuple[float, ...]], dict[str, float]], batch_size: int = 500) -> None:
        cache = scenario_cache[scenario]
        missing = [(key, action) for key, action in pairs.items() if key not in cache]
        if not missing:
            return
        for start in range(0, len(missing), batch_size):
            chunk = missing[start:start + batch_size]
            recs: list[dict[str, Any]] = []
            rec_keys: list[tuple[int, tuple[float, ...]]] = []
            errs: dict[tuple[int, tuple[float, ...]], str] = {}
            for key, action in chunk:
                row_id, sig = key
                try:
                    rec = h.sim_row(int(row_id), action, label=f"E2_{scenario}", extra=True)
                    recs.append(rec)
                    rec_keys.append(key)
                except Exception as exc:
                    errs[key] = f"{type(exc).__name__}: {str(exc)[:240]}"
            if recs:
                try:
                    scores = h.score(recs)
                    if len(scores) != len(recs):
                        raise RuntimeError("Scorer returned a different row count")
                except Exception as exc:
                    log(f"SCORER_BATCH_ERROR {scenario} {type(exc).__name__}: {str(exc)[:200]}; falling back rowwise")
                    values = []
                    for rec in recs:
                        try:
                            values.append(h.score([rec]).iloc[0])
                        except Exception as row_exc:
                            values.append(None)
                    scores = values
                for index, (key, rec) in enumerate(zip(rec_keys, recs)):
                    score_row = scores[index] if isinstance(scores, list) else scores.iloc[index]
                    if score_row is None:
                        errs[key] = "score_failed"
                        continue
                    score_values = [float(score_row[m]) for m in SCORE_METRICS]
                    state = rec["_state"]
                    state_values = {field: (float(getattr(state, field)) if getattr(state, field, None) is not None else np.nan)
                                    for field in FIN_FIELDS}
                    try:
                        acct = state.accounting_identity_check(tol=1.0)
                        acct_error = float(acct.get("diff")) if acct.get("diff") is not None else np.nan
                        acct_check = str(acct.get("check", "skipped"))
                    except Exception:
                        acct_error, acct_check = np.nan, "error"
                    cache[key] = {"ok": True, "scores": score_values, "state": state_values,
                                  "accounting_error": acct_error, "accounting_check": acct_check}
            for key, error in errs.items():
                cache[key] = {"ok": False, "error": error, "scores": [np.nan] * 3,
                              "state": {k: np.nan for k in FIN_FIELDS}, "accounting_error": np.nan,
                              "accounting_check": "error"}
            done = min(start + len(chunk), len(missing))
            if done % 5000 < batch_size or done == len(missing):
                log(f"SCORE {scenario}: newly attempted {done:,}/{len(missing):,}; cache={len(cache):,}")

    def score_scenario(scenario: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        simulator_cls = type(h.bundle._simulator)
        original_interest = simulator_cls.__dict__["_interest"].__func__
        original_plan = simulator_cls.__dict__["_candidate_plan"]
        interest_patch = None
        plan_patch = None
        if scenario == "S1":
            def patched_interest(bp, **kw):
                total, parts = original_interest(bp, **kw)
                principal = float(kw["new_long_tranche"]) + float(kw["new_bond_tranche"])
                extra = 0.01 * principal
                residual = abs(extra - 0.01 * principal)
                s1_audit["calls"] += 1
                s1_audit["principal_sum"] += principal
                s1_audit["surcharge_sum"] += extra
                s1_audit["max_formula_residual"] = max(s1_audit["max_formula_residual"], residual)
                new_parts = dict(parts)
                new_parts["e2_long_tranche_surcharge_1y"] = extra
                return total + extra, new_parts
            interest_patch = staticmethod(patched_interest)
            simulator_cls._interest = interest_patch
        elif scenario == "S2":
            def patched_candidate_plan(self, state, bp, action):
                bp_eff, audit = original_plan(self, state, bp, action)
                for field in ("cogs_ratio", "sga_ratio"):
                    baseline = float(audit[field]["baseline"])
                    original_target = float(audit[field]["target_effective"])
                    effective_target = baseline + 0.5 * (original_target - baseline) if original_target < baseline else original_target
                    setattr(bp_eff, field, effective_target)
                    audit[field]["target_effective_before_e2_s2"] = original_target
                    audit[field]["target_effective"] = effective_target
                    audit[field]["realized_delta"] = effective_target - baseline
                    audit[field]["e2_s2_cost_savings_realization_factor"] = 0.5 if original_target < baseline else 1.0
                s2_audit.append({"firm_id": str(state.firm_id),
                                 "requested_cogs_ratio_chg": float(getattr(action, "cogs_ratio_chg", 0.0)),
                                 "requested_sga_ratio_chg": float(getattr(action, "sga_ratio_chg", 0.0)),
                                 "cogs_baseline": float(audit["cogs_ratio"]["baseline"]),
                                 "cogs_original_target": float(audit["cogs_ratio"]["target_effective_before_e2_s2"]),
                                 "cogs_effective_target": float(audit["cogs_ratio"]["target_effective"]),
                                 "sga_baseline": float(audit["sga_ratio"]["baseline"]),
                                 "sga_original_target": float(audit["sga_ratio"]["target_effective_before_e2_s2"]),
                                 "sga_effective_target": float(audit["sga_ratio"]["target_effective"])})
                return bp_eff, audit
            plan_patch = patched_candidate_plan
            simulator_cls._candidate_plan = plan_patch
        log(f"BEGIN {scenario}: items={len(items):,}")
        try:
            pairs: dict[tuple[int, tuple[float, ...]], dict[str, float]] = {}
            for item in items:
                pair_key = (int(item["row_id"]), item["signature"])
                pairs.setdefault(pair_key, item["action"])
            base_sig = action_signature(baseline_action, h)
            for i in range(N):
                pairs.setdefault((i, base_sig), baseline_action)
            ensure_scores(scenario, pairs)
        finally:
            simulator_cls._interest = staticmethod(original_interest)
            simulator_cls._candidate_plan = original_plan
        cache = scenario_cache[scenario]
        base_sig = action_signature(baseline_action, h)
        base = {i: cache[(i, base_sig)] for i in range(N)}
        out_rows = []
        for item in items:
            i = int(item["row_id"])
            rec = cache[(i, item["signature"])]
            b0 = base[i]
            row = {k: v for k, v in item.items() if k not in {"action", "signature"}}
            row.update({"scenario": scenario, "action_signature": sig_text(item["signature"]),
                        "action_sha256": item["action_sha256"], "simulator_ok": bool(rec["ok"]),
                        "simulator_error": rec.get("error", ""), "accounting_error": rec.get("accounting_error", np.nan),
                        "accounting_check": rec.get("accounting_check", "failed")})
            for j, metric in enumerate(SCORE_METRICS):
                row["absolute_score_" + metric] = rec["scores"][j]
                row["score_delta_" + metric] = rec["scores"][j] - b0["scores"][j] if rec["ok"] and b0["ok"] else np.nan
            for field in FIN_FIELDS:
                row["state_" + field] = rec["state"].get(field, np.nan)
                row["delta_" + field] = rec["state"].get(field, np.nan) - b0["state"].get(field, np.nan) if rec["ok"] and b0["ok"] else np.nan
            out_rows.append(row)
        log(f"END {scenario}: cache={len(cache):,}; direct rows={len(out_rows):,}")
        return {"rows": out_rows, "base": base}

    # S0, then fixed-action sensitivity re-scoring under S1 and S2.
    s0_items = all_direct
    s1_s2_items = main_items + fixed_items
    s0_result = score_scenario("S0", s0_items)
    s1_result = score_scenario("S1", s1_s2_items)
    s2_result = score_scenario("S2", s1_s2_items)
    direct_rows = s0_result["rows"] + s1_result["rows"] + s2_result["rows"]
    direct = pd.DataFrame(direct_rows)
    direct.to_parquet(OUT / "main_policy_scores.parquet", index=False)
    direct.to_csv(OUT / "main_policy_scores.csv", index=False, encoding="utf-8-sig")

    identity_keys = ["source", "policy", "model_key", "replicate", "row_id"]
    hash_pivot = direct[direct.source.isin(["main", "fixed_policy"])].pivot(
        index=identity_keys, columns="scenario", values="action_sha256")
    hash_mismatches = int(((hash_pivot["S0"] != hash_pivot["S1"]) |
                           (hash_pivot["S0"] != hash_pivot["S2"])).sum())
    main_lookup = {(x["source"], x["policy"], x["model_key"], int(x["replicate"]), int(x["row_id"])): x
                   for x in all_direct if x["source"] in ["main", "fixed_policy"]}
    score_lookup = {(str(r.source), str(r.policy), str(r.model_key), int(r.replicate), int(r.row_id), str(r.scenario)): r
                    for r in direct.itertuples(index=False) if str(r.source) in ["main", "fixed_policy"]}
    scenario_qa_rows = []
    for scenario, axes in [("S1", ["refinancing_short_debt_pct"]), ("S2", ["cogs_ratio_chg", "sga_ratio_chg"])]:
        unaffected = []
        for key, item in main_lookup.items():
            u = norm_action(item["action"], h)
            if all(abs(u[AX.index(axis)]) <= 1e-12 for axis in axes):
                s0r = score_lookup[(*key, "S0")]
                sr = score_lookup[(*key, scenario)]
                diffs = [abs(float(getattr(s0r, "absolute_score_" + m)) - float(getattr(sr, "absolute_score_" + m))) for m in SCORE_METRICS]
                diffs += [abs(float(getattr(s0r, "delta_" + f)) - float(getattr(sr, "delta_" + f))) for f in FIN_FIELDS]
                unaffected.append(max(diffs))
        scenario_qa_rows.append({"scenario": scenario, "unaffected_action_rows": len(unaffected),
                                 "max_abs_score_or_financial_change_difference_vs_s0": max(unaffected) if unaffected else np.nan,
                                 "action_hash_mismatches_vs_s0": hash_mismatches,
                                 "expected": "same result when the perturbed action axis is unused"})
    if hash_mismatches:
        raise RuntimeError(f"Action vector changed across stress scenarios: {hash_mismatches} rows")
    pd.DataFrame(scenario_qa_rows).to_csv(OUT / "main_scenario_action_identity_qa.csv", index=False, encoding="utf-8-sig")

    # Cross-check the frozen S0 action/evaluator seam against the already-run B1 action surface.
    surface = pd.read_parquet(B1 / "b1_simulator_action_surface.parquet")
    surface_lookup = {(int(r.row_id), str(r.action_signature)): r for r in surface.itertuples(index=False)}
    b1_checks = []
    for row in s0_result["rows"]:
        if row["source"] != "fixed_policy":
            continue
        key = (int(row["row_id"]), row["action_signature"])
        ref = surface_lookup.get(key)
        if ref is None:
            b1_checks.append({"policy": row["policy"], "row_id": row["row_id"], "match": False, "max_abs_difference": np.nan})
            continue
        diffs = [abs(float(row["absolute_score_" + m]) - float(getattr(ref, "absolute_score_" + m))) for m in SCORE_METRICS]
        b1_checks.append({"policy": row["policy"], "row_id": row["row_id"], "match": True, "max_abs_difference": max(diffs)})
    b1_checks_df = pd.DataFrame(b1_checks)
    b1_checks_df.to_csv(OUT / "main_s0_b1_scorer_crosscheck.csv", index=False, encoding="utf-8-sig")

    boot = bootstrap_indices()
    policy_rows = direct[(direct.scenario == "S0") & direct.source.eq("main")].copy()
    condition_values: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for condition in ["N", "R", "T", "RT"]:
        for model_key in ["ALL", *NAMES.keys()]:
            g = policy_rows[policy_rows.policy.eq(condition)]
            if model_key != "ALL":
                g = g[g.model_key.eq(model_key)]
            firm = g.groupby("row_id")[["score_delta_alpha", "score_delta_beta", "score_delta_gamma"]].mean().reindex(range(N))
            if len(firm) != N or firm.isna().any().any():
                raise RuntimeError(f"Incomplete S0 policy values for {condition}/{model_key}")
            condition_values[(condition, model_key)] = {m: firm["score_delta_" + m].to_numpy(dtype=float) for m in SCORE_METRICS}
    rows = []
    for (condition, model_key), metrics in condition_values.items():
        for metric, x in metrics.items():
            mean, lo, hi, n = mean_ci(x, boot)
            rows.append({"condition": condition, "model": model_key, "evaluator": metric,
                         "policy_value_vs_A0": mean, "ci95_low": lo, "ci95_high": hi,
                         "n_firms": n, "bootstrap_unit": "firm", "bootstrap_reps": B, "seed": SEED})
    pd.DataFrame(rows).to_csv(OUT / "main_condition_values.csv", index=False, encoding="utf-8-sig")

    contrast_specs = {"T_minus_N": ("T", "N"), "RT_minus_R": ("RT", "R"),
                      "RT_minus_T": ("RT", "T"), "R_minus_N": ("R", "N")}
    contrast_rows = []
    for model_key in ["ALL", *NAMES.keys()]:
        for contrast, (left, right) in contrast_specs.items():
            for metric in SCORE_METRICS:
                x = condition_values[(left, model_key)][metric] - condition_values[(right, model_key)][metric]
                mean, lo, hi, n = mean_ci(x, boot)
                contrast_rows.append({"model": model_key, "contrast": contrast, "evaluator": metric,
                                      "difference": mean, "ci95_low": lo, "ci95_high": hi, "n_firms": n,
                                      "interpretation": "package comparison; full-sample N/R are historical-vintage; T/RT use common TASK"})
        for metric in SCORE_METRICS:
            x = (condition_values[("RT", model_key)][metric] - condition_values[("R", model_key)][metric]
                 - condition_values[("T", model_key)][metric] + condition_values[("N", model_key)][metric])
            mean, lo, hi, n = mean_ci(x, boot)
            contrast_rows.append({"model": model_key, "contrast": "interaction_RT_minus_R_minus_T_plus_N", "evaluator": metric,
                                  "difference": mean, "ci95_low": lo, "ci95_high": hi, "n_firms": n,
                                  "interpretation": "factorial interaction of information package under mixed generation vintage"})
    factorial = pd.DataFrame(contrast_rows)
    factorial.to_csv(OUT / "main_factorial_contrasts.csv", index=False, encoding="utf-8-sig")
    h1_models = []
    for model_key in NAMES:
        x = condition_values[("RT", model_key)]["alpha"] - condition_values[("R", model_key)]["alpha"]
        mean, lo, hi, n = mean_ci(x, boot)
        h1_models.append({"model": model_key, "difference": mean, "ci95_low": lo, "ci95_high": hi,
                          "p_centered_bootstrap": centered_bootstrap_p(x, boot), "n_firms": n})
    h1_adjusted = holm_adjust([x["p_centered_bootstrap"] for x in h1_models])
    for row, adj in zip(h1_models, h1_adjusted):
        row["p_holm_two_models"] = adj
    pooled_h1 = factorial.query("model == 'ALL' and contrast == 'RT_minus_R' and evaluator == 'alpha'").iloc[0]
    pd.DataFrame([{"primary_hypothesis": "H1 alpha RT−R", **pooled_h1.to_dict(),
                   "interpretation_rule": "CI lower bound >0 supports improvement; non-significance does not establish equivalence"}]).to_csv(
        OUT / "main_h1_primary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(h1_models).to_csv(OUT / "main_h1_by_model.csv", index=False, encoding="utf-8-sig")

    # P0-P5 contrasts, as fixed policy values against the pooled two-model/two-repeat RT result.
    fixed_s0 = direct[(direct.scenario == "S0") & direct.source.eq("fixed_policy")]
    rt_values = condition_values[("RT", "ALL")]
    baseline_policy_values: dict[str, dict[str, np.ndarray]] = {}
    for name in fixed_actions:
        g = fixed_s0[fixed_s0.policy.eq(name)].sort_values("row_id")
        if len(g) != N:
            raise RuntimeError(f"Missing fixed policy score rows: {name}")
        baseline_policy_values[name] = {m: g["score_delta_" + m].to_numpy(dtype=float) for m in SCORE_METRICS}
    policy_contrasts = []
    for policy, by_metric in baseline_policy_values.items():
        for metric in SCORE_METRICS:
            for model_key in ["ALL", *NAMES.keys()]:
                rt = condition_values[("RT", model_key)][metric]
                x = rt - by_metric[metric]
                mean, lo, hi, n = mean_ci(x, boot)
                policy_contrasts.append({"condition": "RT", "model": model_key, "reference_policy": policy,
                                         "evaluator": metric, "difference": mean, "ci95_low": lo,
                                         "ci95_high": hi, "n_firms": n})
    pd.DataFrame(policy_contrasts).to_csv(OUT / "main_rt_vs_p0_p5.csv", index=False, encoding="utf-8-sig")

    # Fixed-action stress contrasts, all with the same original action vectors.
    stress_rows = []
    stress_condition_values = {}
    for scenario in ["S0", "S1", "S2"]:
        df = direct[direct.scenario.eq(scenario)]
        for condition in ["R", "RT"]:
            g = df[(df.source == "main") & df.policy.eq(condition)]
            for model_key in ["ALL", *NAMES.keys()]:
                gm = g if model_key == "ALL" else g[g.model_key.eq(model_key)]
                firm = gm.groupby("row_id")[["score_delta_alpha", "score_delta_beta", "score_delta_gamma"]].mean().reindex(range(N))
                if firm.isna().any().any():
                    raise RuntimeError(f"Stress outcome incomplete: {scenario}/{condition}/{model_key}")
                stress_condition_values[(scenario, condition, model_key)] = {m: firm["score_delta_" + m].to_numpy(float) for m in SCORE_METRICS}
        fixed_df = df[df.source.eq("fixed_policy")]
        fixed_vals = {}
        for policy in ["P0_RL_C3E", "P2_fixed_COGS_minus1", "P3_C3E_OE_to_P2"]:
            g = fixed_df[fixed_df.policy.eq(policy)].sort_values("row_id")
            fixed_vals[policy] = {m: g["score_delta_" + m].to_numpy(float) for m in SCORE_METRICS}
        for model_key in ["ALL", *NAMES.keys()]:
            rt = stress_condition_values[(scenario, "RT", model_key)]
            r = stress_condition_values[(scenario, "R", model_key)]
            for metric in SCORE_METRICS:
                contrasts = {"RT_minus_R": rt[metric] - r[metric]}
                for policy, values in fixed_vals.items():
                    contrasts["RT_minus_" + policy] = rt[metric] - values[metric]
                for contrast, x in contrasts.items():
                    mean, lo, hi, n = mean_ci(x, boot)
                    stress_rows.append({"scenario": scenario, "model": model_key, "contrast": contrast,
                                        "evaluator": metric, "difference": mean, "ci95_low": lo,
                                        "ci95_high": hi, "n_firms": n})
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(OUT / "main_sensitivity_contrasts.csv", index=False, encoding="utf-8-sig")
    stress_delta_rows = []
    for scenario in ["S1", "S2"]:
        for model_key in ["ALL", *NAMES.keys()]:
            for metric in SCORE_METRICS:
                for contrast in ["RT_minus_R", "RT_minus_P0_RL_C3E", "RT_minus_P2_fixed_COGS_minus1", "RT_minus_P3_C3E_OE_to_P2"]:
                    s = stress[(stress.scenario == scenario) & (stress.model == model_key) &
                               (stress.evaluator == metric) & (stress.contrast == contrast)].iloc[0]
                    z = stress[(stress.scenario == "S0") & (stress.model == model_key) &
                               (stress.evaluator == metric) & (stress.contrast == contrast)].iloc[0]
                    if contrast == "RT_minus_R":
                        delta = ((stress_condition_values[(scenario, "RT", model_key)][metric] - stress_condition_values[(scenario, "R", model_key)][metric])
                                 - (stress_condition_values[("S0", "RT", model_key)][metric] - stress_condition_values[("S0", "R", model_key)][metric]))
                    else:
                        policy = contrast.removeprefix("RT_minus_")
                        scenario_fixed = direct[(direct.scenario == scenario) & (direct.source == "fixed_policy") & direct.policy.eq(policy)].sort_values("row_id")
                        s0_fixed = direct[(direct.scenario == "S0") & (direct.source == "fixed_policy") & direct.policy.eq(policy)].sort_values("row_id")
                        delta = ((stress_condition_values[(scenario, "RT", model_key)][metric] - scenario_fixed["score_delta_" + metric].to_numpy(float))
                                 - (stress_condition_values[("S0", "RT", model_key)][metric] - s0_fixed["score_delta_" + metric].to_numpy(float)))
                    mean, lo, hi, n = mean_ci(delta, boot)
                    stress_delta_rows.append({"scenario_minus_S0": scenario + "_minus_S0", "model": model_key,
                                              "contrast": contrast, "evaluator": metric, "difference_change": mean,
                                              "ci95_low": lo, "ci95_high": hi, "n_firms": n})
    pd.DataFrame(stress_delta_rows).to_csv(OUT / "main_sensitivity_change_vs_s0.csv", index=False, encoding="utf-8-sig")

    # P3 gap recovery, as defined before scoring.
    p3 = baseline_policy_values["P3_C3E_OE_to_P2"]
    p3_rows = []
    for metric in SCORE_METRICS:
        r = condition_values[("R", "ALL")][metric]
        rt = condition_values[("RT", "ALL")][metric]
        d = p3[metric] - r
        g = rt - r
        dmean = float(d.mean()); gmean = float(g.mean())
        point = gmean / dmean if dmean > 1e-12 else np.nan
        bs_d = d[boot].mean(axis=1)
        bs_g = g[boot].mean(axis=1)
        bs_ratio = np.divide(bs_g, bs_d, out=np.full_like(bs_g, np.nan), where=bs_d > 1e-12)
        good = bs_ratio[np.isfinite(bs_ratio)]
        p3_rows.append({"evaluator": metric, "P3_minus_R_gap_D": dmean, "RT_minus_R_gain_G": gmean,
                        "gap_recovery_G_over_D": point,
                        "ci95_low": float(np.percentile(good, 2.5)) if len(good) else np.nan,
                        "ci95_high": float(np.percentile(good, 97.5)) if len(good) else np.nan,
                        "bootstrap_reps_positive_denominator": len(good), "n_firms": N,
                        "interpretation": "score-gap recovery only; not evidence of shared action mechanism"})
    pd.DataFrame(p3_rows).to_csv(OUT / "main_p3_gap_recovery.csv", index=False, encoding="utf-8-sig")

    # Action composition, intensity, axis allocation and repeated-run agreement.
    action_rows = []
    for item in main_items + pilot_items + fixed_items:
        u = norm_action(item["action"], h)
        active = [AX[j] for j in range(len(AX)) if abs(u[j]) > 1e-8]
        cost_sum = abs(u[6]) + abs(u[7])
        action_rows.append({"source": item["source"], "policy": item["policy"], "model_key": item["model_key"],
                            "replicate": item["replicate"], "row_id": item["row_id"], "firm_key": item["firm_key"],
                            "strict_usable": item["strict_usable"], "T_normalized": float(np.abs(u).sum()),
                            "active_axis_count": len(active), "active_axis_combo": "+".join(active) if active else "A0",
                            "cogs_norm_signed": u[6], "sga_norm_signed": u[7],
                            "cogs_sga_total_abs_norm": cost_sum,
                            "cogs_share_within_cost_axes": abs(u[6]) / cost_sum if cost_sum > 1e-12 else np.nan,
                            **{"norm_" + AX[j]: u[j] for j in range(len(AX))}})
    action_df = pd.DataFrame(action_rows)
    action_df.to_csv(OUT / "main_action_vectors_and_axes.csv", index=False, encoding="utf-8-sig")
    action_summary = action_df.groupby(["source", "policy", "model_key"], as_index=False).agg(
        n=("row_id", "size"), strict_usable_rate=("strict_usable", "mean"), mean_T=("T_normalized", "mean"),
        median_T=("T_normalized", "median"), mean_active_axes=("active_axis_count", "mean"),
        mean_cost_axis_intensity=("cogs_sga_total_abs_norm", "mean"), cogs_share_nonzero_cost_actions=("cogs_share_within_cost_axes", "mean"),
        A0_rate=("active_axis_count", lambda x: float((x == 0).mean())),
    )
    for axis in AX:
        m = action_df.groupby(["source", "policy", "model_key"])["norm_" + axis].agg(["mean", lambda x: float(np.abs(x).mean())]).reset_index()
        m.columns = ["source", "policy", "model_key", "mean_signed_norm_" + axis, "mean_abs_norm_" + axis]
        action_summary = action_summary.merge(m, on=["source", "policy", "model_key"], how="left")
    action_summary.to_csv(OUT / "main_action_distribution_summary.csv", index=False, encoding="utf-8-sig")
    combo = action_df[action_df.source.isin(["main", "pilot_aux"])].groupby(["source", "policy", "model_key", "active_axis_combo"], as_index=False).size()
    combo["share"] = combo["size"] / combo.groupby(["source", "policy", "model_key"])["size"].transform("sum")
    combo.to_csv(OUT / "main_action_combo_distribution.csv", index=False, encoding="utf-8-sig")
    repeat_rows = []
    for (source, policy, model_key), g in action_df[action_df.source.eq("main")].groupby(["source", "policy", "model_key"]):
        a = g[g.replicate == 1].set_index("row_id")
        b2 = g[g.replicate == 2].set_index("row_id")
        common = a.index.intersection(b2.index)
        sig_a = a.loc[common, ["norm_" + k for k in AX]].to_numpy(float)
        sig_b = b2.loc[common, ["norm_" + k for k in AX]].to_numpy(float)
        exact = np.all(np.round(sig_a, 10) == np.round(sig_b, 10), axis=1)
        active_a = np.abs(sig_a) > 1e-8; active_b = np.abs(sig_b) > 1e-8
        l1 = np.abs(sig_a - sig_b).sum(axis=1)
        repeat_rows.append({"condition": policy, "model_key": model_key, "n_firms": len(common),
                            "exact_action_repeat_match_rate": float(exact.mean()),
                            "same_active_axis_combo_rate": float(np.all(active_a == active_b, axis=1).mean()),
                            "mean_normalized_L1_action_distance": float(l1.mean()),
                            "mean_normalized_L2_action_distance": float(np.sqrt(((sig_a - sig_b) ** 2).sum(axis=1)).mean())})
    pd.DataFrame(repeat_rows).to_csv(OUT / "main_repeat_agreement.csv", index=False, encoding="utf-8-sig")

    # R8: use the 100 pre-frozen common derangements; C/A intervals are approximate by design.
    r8_items = main_items + fixed_items
    groups: dict[tuple[str, str, str, int], dict[int, dict[str, float]]] = {}
    group_meta: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for item in r8_items:
        key = (item["source"], item["policy"], item["model_key"], int(item["replicate"]))
        if item["row_id"] in groups.setdefault(key, {}):
            raise RuntimeError(f"Duplicate firm action within R8 policy cell: {key} {item['row_id']}")
        groups[key][int(item["row_id"])] = item["action"]
        group_meta[key] = {"source": item["source"], "policy": item["policy"], "model_key": item["model_key"], "replicate": item["replicate"]}
    group_sigs: dict[tuple[str, str, str, int], dict[int, tuple[float, ...]]] = {}
    for key, amap in groups.items():
        if set(amap) != set(range(N)):
            raise RuntimeError(f"R8 policy cell does not have all firms: {key}")
        group_sigs[key] = {i: action_signature(amap[i], h) for i in range(N)}
    r8_firm_rows = []
    r8_cell_rows = []
    inverse_derangements = [np.argsort(p) for p in derangements]
    direct_s0_keys = {(int(x["row_id"]), x["signature"]) for x in all_direct}
    base_signature = action_signature(baseline_action, h)
    direct_s0_keys.update((i, base_signature) for i in range(N))
    total_unique_cross_pairs = 0
    for key, amap in groups.items():
        meta = group_meta[key]
        cross_pairs: dict[tuple[int, tuple[float, ...]], dict[str, float]] = {}
        for inv in inverse_derangements:
            for target in range(N):
                source = int(inv[target])
                sig = group_sigs[key][source]
                cross_pairs.setdefault((target, sig), amap[source])
        total_unique_cross_pairs += len(cross_pairs)
        log(f"R8 cross-surface {meta['source']}/{meta['policy']}/{meta['model_key']}/r{meta['replicate']}: {len(cross_pairs):,} unique pairs")
        ensure_scores("S0", cross_pairs)
        own = np.full((N, 3), np.nan)
        comp = np.full((N, 3), np.nan)
        valid_pair_count = 0
        total_pair_count = N * len(derangements)
        for firm in range(N):
            rec = scenario_cache["S0"][(firm, group_sigs[key][firm])]
            base = s0_result["base"][firm]
            if rec["ok"] and base["ok"]:
                own[firm, :] = np.asarray(rec["scores"]) - np.asarray(base["scores"])
            contributions = []
            for inv in inverse_derangements:
                source = int(inv[firm])
                cross = scenario_cache["S0"][(firm, group_sigs[key][source])]
                if cross["ok"] and base["ok"]:
                    contributions.append(np.asarray(cross["scores"]) - np.asarray(base["scores"]))
                    valid_pair_count += 1
            if contributions:
                comp[firm, :] = np.vstack(contributions).mean(axis=0)
        assign = own - comp
        for i in range(N):
            for j, metric in enumerate(SCORE_METRICS):
                r8_firm_rows.append({**meta, "row_id": i, "evaluator": metric,
                                     "V_i": own[i, j], "C_i": comp[i, j], "A_i": assign[i, j]})
        valid_share = valid_pair_count / total_pair_count if total_pair_count else np.nan
        row = {**meta, "valid_cross_pair_share": valid_share,
               "cross_pair_valid": valid_pair_count, "cross_pair_total": total_pair_count,
               "composition_method": "100 shared derangements; approximate assignment/composition CI"}
        for j, metric in enumerate(SCORE_METRICS):
            for component, arr in [("V", own[:, j]), ("C", comp[:, j]), ("A", assign[:, j])]:
                mean, lo, hi, nf = mean_ci(arr, boot)
                row.update({f"{component}_{metric}": mean, f"{component}_{metric}_ci95_low": lo,
                            f"{component}_{metric}_ci95_high": hi, f"{component}_{metric}_n_firms": nf})
        r8_cell_rows.append(row)
        for pair_key in cross_pairs:
            if pair_key not in direct_s0_keys:
                scenario_cache["S0"].pop(pair_key, None)
        del cross_pairs
    log(f"R8 cross-surface total unique pairs across policy cells: {total_unique_cross_pairs:,}")
    r8_firm = pd.DataFrame(r8_firm_rows)
    r8_firm.to_parquet(OUT / "main_r8_decomposition_firm_cells.parquet", index=False)
    r8_cells = pd.DataFrame(r8_cell_rows)
    r8_cells.to_csv(OUT / "main_r8_decomposition_cells.csv", index=False, encoding="utf-8-sig")
    pooled_parts = []
    for source in ["main", "fixed_policy"]:
        policies = ["N", "R", "T", "RT"] if source == "main" else list(fixed_actions)
        for policy in policies:
            g = r8_firm[(r8_firm.source == source) & (r8_firm.policy == policy)]
            means = g.groupby(["row_id", "evaluator"], as_index=False)[["V_i", "C_i", "A_i"]].mean()
            for metric in SCORE_METRICS:
                gm = means[means.evaluator.eq(metric)].sort_values("row_id")
                row = {"source": source, "policy": policy, "model_key": "ALL", "evaluator": metric,
                       "composition_method": "100 common derangements; cluster bootstrap on target firms"}
                for component in ["V_i", "C_i", "A_i"]:
                    mean, lo, hi, nf = mean_ci(gm[component].to_numpy(float), boot)
                    label = {"V_i": "V", "C_i": "C", "A_i": "A"}[component]
                    row.update({label: mean, label + "_ci95_low": lo, label + "_ci95_high": hi, label + "_n_firms": nf})
                pooled_parts.append(row)
    r8_pooled = pd.DataFrame(pooled_parts)
    r8_pooled.to_csv(OUT / "main_r8_policy_decomposition.csv", index=False, encoding="utf-8-sig")
    r8_contrast_rows = []
    for metric in SCORE_METRICS:
        for component in ["V_i", "C_i", "A_i"]:
            by_cond = {}
            for cond in ["N", "R", "T", "RT"]:
                gg = r8_firm[(r8_firm.source == "main") & (r8_firm.policy == cond) & (r8_firm.evaluator == metric)]
                by_cond[cond] = gg.groupby("row_id")[component].mean().reindex(range(N)).to_numpy(float)
            for contrast, x in {"RT_minus_R": by_cond["RT"]-by_cond["R"],
                                "T_minus_N": by_cond["T"]-by_cond["N"],
                                "interaction_RT_minus_R_minus_T_plus_N": by_cond["RT"]-by_cond["R"]-by_cond["T"]+by_cond["N"]}.items():
                mean, lo, hi, nf = mean_ci(x, boot)
                r8_contrast_rows.append({"contrast": contrast, "component": component[0], "evaluator": metric,
                                         "difference": mean, "ci95_low": lo, "ci95_high": hi, "n_firms": nf,
                                         "interpretation": "descriptive value decomposition; C/A interval approximates 100 derangements"})
    pd.DataFrame(r8_contrast_rows).to_csv(OUT / "main_r8_decomposition_contrasts.csv", index=False, encoding="utf-8-sig")

    # The contemporaneous 100-firm pilot is a separate paired auxiliary 2×2 comparison.
    pilot_s0 = pd.DataFrame(s0_result["rows"])
    pvals = pilot_s0[pilot_s0.source.eq("pilot_aux")]
    pilot_cell_values = {}
    pilot_boot = bootstrap_indices(n=100, b=B, seed=SEED)
    for cond in ["N", "R", "T", "RT"]:
        g = pvals[pvals.policy.eq(cond)]
        pilot_cell_values[cond] = {}
        for model_key in ["ALL", *NAMES.keys()]:
            gm = g if model_key == "ALL" else g[g.model_key.eq(model_key)]
            firm = gm.groupby("row_id")[["score_delta_alpha", "score_delta_beta", "score_delta_gamma"]].mean().reindex(sorted(gm.row_id.unique()))
            if model_key == "ALL" and len(firm) != 100:
                raise RuntimeError(f"Pilot auxiliary firm count is not 100 for {cond}")
            pilot_cell_values[cond][model_key] = {m: firm["score_delta_" + m].to_numpy(float) for m in SCORE_METRICS}
    pilot_contrasts = []
    for model_key in ["ALL", *NAMES.keys()]:
        for contrast, (left, right) in contrast_specs.items():
            for metric in SCORE_METRICS:
                x = pilot_cell_values[left][model_key][metric] - pilot_cell_values[right][model_key][metric]
                mean, lo, hi, nf = mean_ci(x, pilot_boot)
                pilot_contrasts.append({"model": model_key, "contrast": contrast, "evaluator": metric,
                                        "difference": mean, "ci95_low": lo, "ci95_high": hi, "n_firms": nf,
                                        "sample": "same-time 100-firm pilot; auxiliary, stratified and not independent"})
        for metric in SCORE_METRICS:
            x = (pilot_cell_values["RT"][model_key][metric]-pilot_cell_values["R"][model_key][metric]
                 -pilot_cell_values["T"][model_key][metric]+pilot_cell_values["N"][model_key][metric])
            mean, lo, hi, nf = mean_ci(x, pilot_boot)
            pilot_contrasts.append({"model": model_key, "contrast": "interaction_RT_minus_R_minus_T_plus_N", "evaluator": metric,
                                    "difference": mean, "ci95_low": lo, "ci95_high": hi, "n_firms": nf,
                                    "sample": "same-time 100-firm pilot; auxiliary, stratified and not independent"})
    pd.DataFrame(pilot_contrasts).to_csv(OUT / "main_pilot_contemporaneous_2x2.csv", index=False, encoding="utf-8-sig")

    # Score Q against the already-frozen B1 financial table; no model-generated amount is sign-corrected.
    q_df = pd.read_parquet(OUT / "main_q_responses.parquet")
    b1_tables = pd.read_parquet(B1 / "e2_simulator_10row_tables.parquet")
    model_alias = {"openai_gpt54mini": "GPT", "google_gemini31flashlite": "Gemini"}
    table_groups = {(str(m), int(rid), int(rep)): g.sort_values("row_order")
                    for (m, rid, rep), g in b1_tables.groupby(["model", "row_id", "replicate"])}
    q_detail = []
    for qr in q_df.itertuples(index=False):
        try:
            assets = float(h.firm(int(qr.row_id))[1].total_assets)
        except Exception:
            assets = np.nan
        forecast = json.loads(qr.forecast_json) if qr.forecast_json and bool(qr.json_schema_valid) else None
        tab = table_groups.get((model_alias[str(qr.model_key)], int(qr.row_id), 1))
        if tab is None or len(tab) != 10:
            raise RuntimeError(f"B1 simulator table missing for Q row {qr.request_id}")
        for candidate_id in range(1, 10):
            row_order = candidate_id + 1
            actual_row = tab[tab.row_order.eq(row_order)].iloc[0]
            for metric, col in [("pretax_income", "delta_pretax_income"), ("interest_expense", "delta_pure_interest_expense"),
                                ("total_debt", "delta_total_debt"), ("cash", "delta_cash"),
                                ("current_liabilities", "delta_current_liabilities")]:
                item = None
                if forecast is not None:
                    item = next((x for x in forecast["forecasts"] if x.get("candidate_id") == candidate_id), None)
                pred = float(item[metric]["change_thousand_krw"]) if item is not None else np.nan
                pred_direction = item[metric]["direction"] if item is not None else None
                actual = float(actual_row[col]) if pd.notna(actual_row[col]) else np.nan
                actual_valid = bool(actual_row.simulator_ok) and np.isfinite(actual) and np.isfinite(assets) and assets > 0
                normalized_error = abs(pred - actual) / assets if actual_valid and np.isfinite(pred) else np.nan
                truth_by_bp = {}
                for bp in [0, 1, 10]:
                    threshold = assets * (bp / 10000.0) if np.isfinite(assets) and assets > 0 else np.nan
                    if not actual_valid:
                        truth_by_bp[bp] = None
                    elif abs(actual) <= threshold:
                        truth_by_bp[bp] = "no_material_change"
                    elif actual > 0:
                        truth_by_bp[bp] = "increase"
                    else:
                        truth_by_bp[bp] = "decrease"
                row = {"request_id": qr.request_id, "model_key": qr.model_key, "row_id": int(qr.row_id),
                       "firm_key": qr.firm_key, "candidate_id": candidate_id, "candidate_row_order": row_order,
                       "metric": metric, "q_schema_valid": bool(qr.json_schema_valid), "simulator_actual_valid": actual_valid,
                       "base_total_assets_KRW_thousand": assets, "actual_change_KRW_thousand": actual,
                       "predicted_change_KRW_thousand": pred, "predicted_direction": pred_direction,
                       "normalized_absolute_error_by_assets": normalized_error,
                       "internal_direction_amount_mismatch_count_for_response": int(qr.direction_sign_mismatch_count)}
                for bp in [0, 1, 10]:
                    row[f"actual_direction_{bp}bp"] = truth_by_bp[bp]
                    row[f"direction_correct_{bp}bp"] = (pred_direction == truth_by_bp[bp]) if truth_by_bp[bp] is not None and pred_direction is not None else None
                    row[f"no_change_false_positive_{bp}bp"] = (truth_by_bp[bp] == "no_material_change" and pred_direction != "no_material_change") if truth_by_bp[bp] is not None and pred_direction is not None else None
                if actual_valid and np.isfinite(pred) and abs(actual) > assets * 1e-4 and pred_direction == truth_by_bp[1]:
                    ratio = abs(pred) / abs(actual)
                    row["magnitude_ratio_predicted_to_actual"] = ratio
                    row["magnitude_outside_half_to_double"] = not (0.5 <= ratio <= 2.0)
                else:
                    row["magnitude_ratio_predicted_to_actual"] = np.nan
                    row["magnitude_outside_half_to_double"] = np.nan
                q_detail.append(row)
    q_rows = pd.DataFrame(q_detail)
    q_rows.to_parquet(OUT / "main_q_forecast_detail.parquet", index=False)
    q_rows.to_csv(OUT / "main_q_forecast_detail.csv", index=False, encoding="utf-8-sig")
    q_summary = []
    for (model_key, metric), g in q_rows.groupby(["model_key", "metric"]):
        valid = g[g.q_schema_valid & g.simulator_actual_valid]
        for bp in [0, 1, 10]:
            q_summary.append({"model_key": model_key, "metric": metric, "threshold_bp": bp,
                              "forecast_metric_rows": len(g), "schema_valid_rows": int(g.q_schema_valid.sum()),
                              "scorable_rows": len(valid),
                              "direction_accuracy": float(valid[f"direction_correct_{bp}bp"].mean()) if len(valid) else np.nan,
                              "no_change_false_positive_rate": float(valid[f"no_change_false_positive_{bp}bp"].mean()) if len(valid) else np.nan,
                              "median_normalized_absolute_error_by_assets": float(valid.normalized_absolute_error_by_assets.median()) if len(valid) else np.nan,
                              "magnitude_outside_half_to_double_share": float(valid.magnitude_outside_half_to_double.dropna().mean()) if len(valid.magnitude_outside_half_to_double.dropna()) else np.nan,
                              "direction_amount_mismatch_response_count": int(q_df[q_df.model_key.eq(model_key)].direction_sign_mismatch_count.gt(0).sum())})
    pd.DataFrame(q_summary).to_csv(OUT / "main_q_forecast_accuracy.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"model_key": model_key, "requests": len(g), "schema_valid": int(g.json_schema_valid.sum()),
                   "schema_valid_rate": float(g.json_schema_valid.mean()),
                   "direction_amount_mismatch_metrics": int(g.direction_sign_mismatch_count.sum()),
                   "unscorable_rows": int((~g.json_schema_valid).sum())}
                  for model_key, g in q_df.groupby("model_key")]).to_csv(OUT / "main_q_structure_summary.csv", index=False, encoding="utf-8-sig")

    # Financial changes already attached to every direct policy outcome.
    financial = direct[(direct.source == "main") | (direct.source == "fixed_policy")]
    fin_rows = []
    for (scenario, source, policy, model_key), g in financial.groupby(["scenario", "source", "policy", "model_key"]):
        for field in FIN_FIELDS:
            x = pd.to_numeric(g["delta_" + field], errors="coerce").to_numpy(float)
            fin_rows.append({"scenario": scenario, "source": source, "policy": policy, "model_key": model_key,
                             "financial_item": field, "n": int(np.isfinite(x).sum()),
                             "mean_change_KRW_thousand": float(np.nanmean(x)),
                             "median_change_KRW_thousand": float(np.nanmedian(x))})
    pd.DataFrame(fin_rows).to_csv(OUT / "main_simulator_financial_changes.csv", index=False, encoding="utf-8-sig")

    # Stress accounting/action integrity logs and exact run provenance.
    pd.DataFrame(s2_audit).to_csv(OUT / "main_s2_effective_cost_targets.csv", index=False, encoding="utf-8-sig")
    s2_df = pd.DataFrame(s2_audit)
    if len(s2_df):
        cogs_expected = np.where(s2_df.cogs_original_target < s2_df.cogs_baseline,
                                 s2_df.cogs_baseline + 0.5 * (s2_df.cogs_original_target - s2_df.cogs_baseline),
                                 s2_df.cogs_original_target)
        sga_expected = np.where(s2_df.sga_original_target < s2_df.sga_baseline,
                                s2_df.sga_baseline + 0.5 * (s2_df.sga_original_target - s2_df.sga_baseline),
                                s2_df.sga_original_target)
        s2_max_residual = float(max(np.max(np.abs(s2_df.cogs_effective_target - cogs_expected)),
                                    np.max(np.abs(s2_df.sga_effective_target - sga_expected))))
    else:
        s2_max_residual = np.nan
    accounting = direct[direct.scenario.isin(["S0", "S1", "S2"])].groupby("scenario").agg(
        rows=("simulator_ok", "size"), simulator_failures=("simulator_ok", lambda x: int((~x).sum())),
        accounting_checks_pass=("accounting_check", lambda x: int(x.isin(["pass", "PASS", "ok", "OK"]).sum())),
        accounting_error_max_abs=("accounting_error", lambda x: float(pd.to_numeric(x, errors="coerce").abs().max()))).reset_index()
    accounting.to_csv(OUT / "main_simulator_accounting_qa.csv", index=False, encoding="utf-8-sig")
    if s1_audit["max_formula_residual"] > 1e-12 or s2_max_residual > 1e-12:
        raise RuntimeError(f"Stress implementation QA failed: S1={s1_audit['max_formula_residual']}, S2={s2_max_residual}")

    # Create compact static figures for the report.
    try:
        import matplotlib.pyplot as plt
        cv = pd.read_csv(OUT / "main_condition_values.csv")
        x = cv[(cv.model == "ALL") & (cv.evaluator == "alpha")].set_index("condition").loc[["N", "R", "T", "RT"]]
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.bar(x.index, x.policy_value_vs_A0, color=["#7d91a6", "#567a9d", "#c8a456", "#628b72"])
        ax.errorbar(range(4), x.policy_value_vs_A0, yerr=[x.policy_value_vs_A0 - x.ci95_low, x.ci95_high - x.policy_value_vs_A0], fmt="none", ecolor="#222", capsize=4)
        ax.set_ylabel("α policy value vs A0")
        ax.set_title("E2 policy value (575 firms; two models and repeats averaged)")
        ax.axhline(0, color="#555", linewidth=0.8)
        fig.tight_layout(); fig.savefig(OUT / "main_2x2_alpha.png", dpi=180); plt.close(fig)
        comp = r8_pooled[(r8_pooled.source == "main") & (r8_pooled.evaluator == "alpha")].set_index("policy").loc[["N", "R", "T", "RT"]]
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.bar(comp.index, comp.C, label="Composition", color="#d0a657")
        ax.bar(comp.index, comp.A, bottom=comp.C, label="Assignment", color="#618971")
        ax.set_ylabel("α value change vs A0")
        ax.set_title("E2 composition and assignment decomposition")
        ax.legend(frameon=False); ax.axhline(0, color="#555", linewidth=0.8)
        fig.tight_layout(); fig.savefig(OUT / "main_composition_assignment.png", dpi=180); plt.close(fig)
        stress_plot = stress[(stress.model == "ALL") & (stress.evaluator == "alpha") & (stress.contrast == "RT_minus_R")].set_index("scenario").loc[["S0", "S1", "S2"]]
        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.bar(stress_plot.index, stress_plot.difference, color=["#628b72", "#b8845a", "#8d7ca8"])
        ax.errorbar(range(3), stress_plot.difference, yerr=[stress_plot.difference-stress_plot.ci95_low, stress_plot.ci95_high-stress_plot.difference], fmt="none", ecolor="#222", capsize=4)
        ax.set_ylabel("α RT−R")
        ax.set_title("Fixed-action simulator sensitivity")
        ax.axhline(0, color="#555", linewidth=0.8)
        fig.tight_layout(); fig.savefig(OUT / "main_stress_sensitivity.png", dpi=180); plt.close(fig)
    except Exception as exc:
        (E2 / "logs_main/figure_generation_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

    metadata = {
        "status": "ANALYSIS_COMPLETE",
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "harness_firms": N,
        "evaluation_models": ["alpha", "beta", "gamma"],
        "scenarios": ["S0", "S1 long-tranche surcharge +1%p for 1 year", "S2 cost-savings realization 50% only for cost-ratio decreases"],
        "policy_score_rows": len(direct),
        "fixed_action_input_manifest_sha256": action_manifest_hash,
        "r8_derangements_sha256": json.loads((E2 / "logs_main/r8_derangement_freeze.json").read_text(encoding="utf-8"))["sha256"],
        "r8_derangement_count": len(derangements),
        "r8_ci_note": "V is direct; C is estimated from 100 common derangements; C/A confidence intervals are approximate firm-cluster bootstrap intervals.",
        "b1_s0_crosscheck": {"rows": len(b1_checks_df), "missing_surface_rows": int((~b1_checks_df.match).sum()),
                             "max_abs_difference": float(b1_checks_df.max_abs_difference.max()) if len(b1_checks_df) else None},
        "S1_audit": s1_audit,
        "S2_effective_target_audit_rows": len(s2_df), "S2_max_target_formula_residual": s2_max_residual,
        "scoring_runtime_seconds": round(time.time() - started, 3),
        "scoring_scripts": [str(Path(__file__).relative_to(WORKSPACE)), "scripts/parse_main_responses.py"],
        "important_interpretive_limit": "Full-sample N/R are historical requests; T/RT use the commonized v5.1 prompt, so differences are not pure table-only causal effects. Pilot 100-firm simultaneous 2×2 is a separate auxiliary comparison.",
    }
    (E2 / "logs_main/main_analysis_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log("ANALYSIS_COMPLETE " + json.dumps({"policy_rows": len(direct), "r8_groups": len(r8_cells),
                                           "b1_crosscheck": metadata["b1_s0_crosscheck"],
                                           "runtime_seconds": metadata["scoring_runtime_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    run_analysis()

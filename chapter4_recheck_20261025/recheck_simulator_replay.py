"""Recheck of thesis Table 4-2: the evaluation Simulator replaying what firms actually did, against what actually happened.

Population: rated firm-years of the deployed Oracle-alpha panel (scored year y = 2008..2023) and the 575 evaluation
firms (2023 -> 2024). For each firm, year t = y - 1:
1. the opening state t, the firm's three-year history and the business plan come from the canonical financial history
   and the production bundle (V43ProductionSimulationBundle.build_plan, calibrated_bp_rate_v4);
2. what the firm actually did from t to t+1 (revenue growth, cost ratios, turnovers, capital expenditure, debt
   principal) is rebuilt with credit_recourse.simulator.historical_primitives.reconstruct and executed with
   bundle.simulate_historical (HISTORICAL_REPLAY_MODE);
3. the simulated state t+1, the actual state t+1 and the actual state t are scored by the same Oracle-alpha variable
   formulas (compute_oracle_variables, as in eval/v43_alpha_evaluation). The capital-event count, the only
   nonfinancial alpha input, is the Stage1 value of the scored year and is the same for the simulated and the actual
   state;
4. reports, by scored year (DEV <= 2019, OOT 2020..2023, POST 2024): the score gap (simulated - actual), the rank
   correlation, the direction of the score change from the actual year-t score, and the direction of the next rating
   change (Loop-B1 lead convention: score change t -> t+1 against rating change t+1 -> t+2). Reference points: the
   rank correlation of last year's actual score, and the business plan alone (candidate A0, no actual actions).
The replay sets revenue, cost ratios, turnovers, capital expenditure and debt principal to what happened; the
Simulator supplies depreciation, interest, other income, tax, dividends, cash and equity. It checks the accounting
given what happened, not a forecast.

Inputs: the distribution zip (regular file, no Git LFS) and the E1 scoring run root
(ablation_20261102/E1/scripts/e1_setup_runroot.ps1).
Usage (kit root, Python 3.11 kit environment):
    python chapter4_recheck_20261025/recheck_simulator_replay.py --run-root <run root>
Writes chapter4_recheck_20261025/table4_2_simulator_replay.json and exits 1 if a result differs from the thesis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
ZIP = KIT / "frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip"
P = "THESIS_REPRO_KIT_v2.1.1_FINAL/archive/DEPLOYED_RELEASE/"
HIST = P + "stage2_candidate_projection/input_splits/canonical_business_plan_history.parquet"
ORACLE_PANEL = P + "stage1_oracle_backends/alpha/oracle_modeling_panel_alpha.parquet"
ORACLE_OUTPUT = P + "stage1_oracle_backends/alpha/oracle_firm_year_output_alpha.parquet"
SPLITS = ("DEV", "OOT", "POST")

# Thesis Table 4-2 and section 4.2 (2026-10-25 revision)
EXPECTED = {
    "replayed": {"DEV": 1540, "OOT": 1404, "POST": 465},
    "median_gap": {"DEV": 0.72, "OOT": 0.78, "POST": 0.0},
    "median_abs_gap": {"DEV": 3.22, "OOT": 3.29, "POST": 3.62},
    "share_sim_higher_pct": {"DEV": 55.5, "OOT": 56.4, "POST": 49.0},
    "rank_correlation": {"DEV": 0.941, "OOT": 0.946, "POST": 0.936},
    "rank_correlation_last_year_score": {"DEV": 0.866, "OOT": 0.895, "POST": 0.894},
    "direction_n": {"DEV": 1153, "OOT": 1088, "POST": 409},
    "direction_pct": {"DEV": 74.3, "OOT": 75.5, "POST": 73.8},
    "direction_ci95_pct": {"DEV": [71.7, 76.8], "OOT": [72.8, 77.9], "POST": [69.4, 77.9]},
    "direction_plan_only_pct": {"DEV": 60.7, "OOT": 59.7, "POST": 63.1},
    "shortfall_borrowing_pct": {"DEV": 34.8, "OOT": 27.6, "POST": 29.7},
    "rating_link_oot": {"movers": 140, "simulated_pct": 66.4, "simulated_ci95_pct": [58.3, 73.7], "actual_pct": 71.4, "upgrade_pct": 54.3},
    "excluded": {"no_business_plan": 623, "primitives_not_rebuilt": 0, "simulator_refused": 247, "no_consecutive_years": 47},
    "path_check": {"matched": 2944, "same_score_pct": 85.8, "rank_correlation": 0.991},
}
IDENTITY_MAX = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-root", required=True, type=Path)
    run = ap.parse_args().run_root.resolve()
    os.environ["THESIS_REPRO_RUN_ROOT"] = str(run)
    sys.path.insert(0, str(KIT / "src"))
    from dataclasses import fields
    from scipy import stats
    from credit_recourse.simulator.firm_state import FirmState
    from credit_recourse.simulator.historical_source import read_financial_panel, state_from_record, firm_key
    from credit_recourse.simulator.historical_primitives import reconstruct
    from credit_recourse.simulator.executed_primitives import capex_baseline
    from credit_recourse.simulator.v43_production_bundle import V43ProductionSimulationBundle
    from credit_recourse.simulator.oracle_variables import compute_oracle_variables
    from credit_recourse.eval.v43_alpha_evaluation import _target_operating_loss_freq_3y, _state_to_frame_dict, alpha_from_variables
    from credit_recourse.oracle.artifact_io import load_registry, resolve_backend_artifact
    from credit_recourse.contracts.runtime_assets import active_oracle_registry
    from credit_recourse.contracts.stage_paths import final_root
    from credit_recourse.oracle.backends.alpha.modules.oracle_alpha_scorer import build_alpha_scorer
    from credit_recourse.oracle.verification.verify_stage1_substrate_validation import compute_direction_agreement

    if ZIP.read_bytes()[:64].startswith(b"version https://git-lfs"):
        raise SystemExit(f"{ZIP.name} is a Git LFS pointer; run: git lfs pull --include \"{ZIP.relative_to(KIT).as_posix()}\"")
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(ZIP) as z:
        tmp = Path(tmp)
        for member in (HIST, ORACLE_PANEL, ORACLE_OUTPUT):
            (tmp / Path(member).name).write_bytes(z.read(member))
        members = {m: sha256(tmp / Path(m).name) for m in (HIST, ORACLE_PANEL, ORACLE_OUTPUT)}
        accounts, _ = read_financial_panel(tmp / Path(HIST).name)
        hist_raw = pd.read_parquet(tmp / Path(HIST).name, columns=["firm_id", "fiscal_year", "cap_change_count_3y"])
        panel = pd.read_parquet(tmp / Path(ORACLE_PANEL).name, columns=["거래소코드", "year", "cap_change_count_3y", "rating_num_10"])
        output = pd.read_parquet(tmp / Path(ORACLE_OUTPUT).name, columns=["거래소코드", "year", "R_score_alpha"])
    hist_raw["firm_id"] = hist_raw.firm_id.map(firm_key)
    for frame in (panel, output):
        frame["firm_id"] = frame["거래소코드"].map(firm_key)
        frame["year"] = pd.to_numeric(frame.year).astype(int)

    states, histories = {}, {}
    for rec in accounts.to_dict("records"):
        s = state_from_record(rec)
        states[(s.firm_id, s.year)] = s
        histories.setdefault(s.firm_id, []).append(s)
    for k in histories:
        histories[k].sort(key=lambda s: s.year)
    cap = {(f, int(y)): v for f, y, v in panel[["firm_id", "year", "cap_change_count_3y"]].itertuples(index=False)}
    cap.update({(f, 2024): v for f, y, v in hist_raw[["firm_id", "fiscal_year", "cap_change_count_3y"]].itertuples(index=False) if int(y) == 2024})
    rating = {(f, int(y)): v for f, y, v in panel[["firm_id", "year", "rating_num_10"]].itertuples(index=False)}
    targets = [(f, int(y)) for f, y in panel[["firm_id", "year"]].itertuples(index=False) if 2008 <= int(y) <= 2023]
    targets += [(f, 2024) for f in sorted({f for (f, y) in states if y == 2024})]

    bundle = V43ProductionSimulationBundle.from_project_root(run)
    reg = load_registry(active_oracle_registry(run))
    params_path = resolve_backend_artifact(run, final_root(run), reg["backends"]["alpha"]["params"])
    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    selected = params.get("selected_variables") or [v.get("variable_id") for v in params.get("variables", []) if isinstance(v, dict)]
    scorer = build_alpha_scorer(params)

    def alpha_of(prev, cur, cap_value):
        if cap_value is None or not np.isfinite(float(cap_value)):
            raise ValueError("no Stage1 capital-event count for the scored year")
        hist = [s for s in histories[prev.firm_id] if s.year <= prev.year][-3:]
        exog = {"cap_change_count_3y_target": float(cap_value), "operating_loss_freq_3y_target": _target_operating_loss_freq_3y(hist, cur)}
        variables = _state_to_frame_dict(compute_oracle_variables(cur, prev_state=prev, exogenous=exog), pd.Series({}, dtype=object))
        return alpha_from_variables(variables, scorer, selected)["alpha"]

    def observed(x):
        return x is not None and np.isfinite(float(x))

    def masks(o, c):
        # observation masks of the reconstruction contract: an account change is observed when both years are observed
        # (a move from a zero balance counts, as in the Stage2 action panel)
        m = {name: observed(getattr(o, f)) and observed(getattr(c, f))
             for name, f in (("short_debt_pct", "short_term_debt"), ("long_debt_pct", "long_term_debt"), ("bond_pct", "bonds"))}
        m.update({name + "_chg": True for name in ("inv_turnover", "ar_turnover", "ap_turnover", "cogs_ratio", "sga_ratio")})
        m["revenue_growth"] = observed(o.revenue) and observed(c.revenue) and float(o.revenue) > 0
        return m

    rows, identity = [], []
    for firm, y in targets:
        t = y - 1
        row = {"firm_id": firm, "year": y, "split": "DEV" if y <= 2019 else "OOT" if y <= 2023 else "POST"}
        o, c = states.get((firm, t)), states.get((firm, y))
        if o is None or c is None:
            rows.append({**row, "status": "no_consecutive_years"}); continue
        hist = [s for s in histories[firm] if s.year <= t]
        try:
            bp, rate = bundle.build_plan(o, hist)
        except (KeyError, ValueError):
            rows.append({**row, "status": "no_business_plan"}); continue
        rec, prim = reconstruct(o, c, bp, masks(o, c), bp_years=[s.year for s in hist[-3:]], bp_source_hash="table4_2")
        if prim is None:
            rows.append({**row, "status": "primitives_not_rebuilt"}); continue
        context = {"firm_id": firm, "base_year": t, "primitive_hash": prim.content_hash, **capex_baseline(o, bp), "actual_capex": rec["actual_capex"]}
        try:
            result, _, _, _ = bundle.simulate_historical(o, hist, prim, context)
        except ValueError as error:
            if "NONPHYSICAL_STATE" not in str(error):
                raise
            rows.append({**row, "status": "simulator_refused"}); continue
        sim = result.state_t1
        a0_state = bundle.simulate_candidate(o, hist, "A0")[0].state_t1      # the business plan alone, no actual actions
        ta = float(sim.total_assets)
        identity.append(max(abs(ta - sim.total_liabilities - sim.total_equity), abs(ta - sim.current_assets - sim.non_current_assets),
                            abs(sim.total_liabilities - sim.current_liabilities - sim.non_current_liabilities)) / abs(ta))
        row.update(status="ok", alpha_sim=alpha_of(o, sim, cap.get((firm, y))), alpha_act=alpha_of(o, c, cap.get((firm, y))),
                   alpha_plan_only=alpha_of(o, a0_state, cap.get((firm, y))),
                   shortfall_borrowing=float(result.plug_amount or 0.0) > 1e-6,
                   negative_cash=float(sim.cash) < -1e-6, negative_current_assets=float(sim.current_assets) < -1e-6)
        p = states.get((firm, t - 1))
        if p is not None and observed(cap.get((firm, t))):
            row["alpha_act_prev"] = alpha_of(p, o, cap.get((firm, t)))
        rows.append(row)
    d = pd.DataFrame(rows)
    ok_rows = d[d.status == "ok"].copy()
    ok_rows["gap"] = ok_rows.alpha_sim - ok_rows.alpha_act

    out = {"source_members_sha256": members, "run_root_simulator": bundle.action_contract["simulator_action_semantic_contract_version"],
           "targets": int(len(d)), "status_counts": {k: int(v) for k, v in d.status.value_counts().items()},
           "status_by_split": d.groupby(["split", "status"]).size().unstack(fill_value=0).astype(int).to_dict(orient="index"),
           "replay_identity_max_relative_error": float(max(identity)), "replay_negative_cash_states": int(ok_rows.negative_cash.sum()),
           "replay_negative_current_assets_states": int(ok_rows.negative_current_assets.sum())}
    by = {}
    for split in SPLITS:
        g = ok_rows[ok_rows.split == split]
        h = g.dropna(subset=["alpha_act_prev"])
        a = compute_direction_agreement(h.alpha_sim - h.alpha_act_prev, h.alpha_act - h.alpha_act_prev)
        a0 = compute_direction_agreement(h.alpha_plan_only - h.alpha_act_prev, h.alpha_act - h.alpha_act_prev)
        by[split] = {"replayed": int(len(g)), "median_gap": float(g.gap.median()), "median_abs_gap": float(g.gap.abs().median()),
                     "share_sim_higher": float((g.gap > 0).mean()),
                     "rank_correlation": float(stats.spearmanr(g.alpha_sim, g.alpha_act).statistic),
                     "direction_n": a["n_movers"], "direction": a["direction_agreement"], "direction_ci95": a["direction_agreement_ci95"],
                     # reference points: last year's actual score, and the business plan without the actual actions
                     "rank_correlation_last_year_score": float(stats.spearmanr(h.alpha_act_prev, h.alpha_act).statistic),
                     "rank_correlation_plan_only": float(stats.spearmanr(g.alpha_plan_only, g.alpha_act).statistic),
                     "direction_plan_only": a0["direction_agreement"], "direction_plan_only_ci95": a0["direction_agreement_ci95"],
                     "shortfall_borrowing_share": float(g.shortfall_borrowing.mean())}
    out["by_split"] = by
    h = ok_rows.dropna(subset=["alpha_act_prev"]).copy()
    h["r0"] = [rating.get((f, y)) for f, y in zip(h.firm_id, h.year)]
    h["r1"] = [rating.get((f, y + 1)) for f, y in zip(h.firm_id, h.year)]
    h = h.dropna(subset=["r0", "r1"])
    h["improvement"] = -(h.r1 - h.r0)                     # rating_num_10: lower is better
    oot = h[(h.year + 1 >= 2020) & (h.year + 1 <= 2023)]
    sim_link = compute_direction_agreement(oot.alpha_sim - oot.alpha_act_prev, oot.improvement)
    act_link = compute_direction_agreement(oot.alpha_act - oot.alpha_act_prev, oot.improvement)
    movers = oot[oot.improvement != 0]
    out["rating_link_oot"] = {"movers": sim_link["n_movers"], "simulated": sim_link["direction_agreement"], "simulated_ci95": sim_link["direction_agreement_ci95"],
                              "actual": act_link["direction_agreement"], "actual_ci95": act_link["direction_agreement_ci95"],
                              "upgrade_share": float((movers.improvement > 0).mean())}
    m = ok_rows.merge(output, on=["firm_id", "year"], how="inner")
    out["path_check"] = {"matched": int(len(m)), "same_score_share": float(((m.alpha_act - m.R_score_alpha).abs() < 1e-6).mean()),
                         "rank_correlation": float(stats.spearmanr(m.alpha_act, m.R_score_alpha).statistic)}

    got = {
        "replayed": {s: by[s]["replayed"] for s in SPLITS},
        "median_gap": {s: round(by[s]["median_gap"], 2) for s in SPLITS},
        "median_abs_gap": {s: round(by[s]["median_abs_gap"], 2) for s in SPLITS},
        "share_sim_higher_pct": {s: round(100 * by[s]["share_sim_higher"], 1) for s in SPLITS},
        "rank_correlation": {s: round(by[s]["rank_correlation"], 3) for s in SPLITS},
        "rank_correlation_last_year_score": {s: round(by[s]["rank_correlation_last_year_score"], 3) for s in SPLITS},
        "direction_n": {s: by[s]["direction_n"] for s in SPLITS},
        "direction_pct": {s: round(100 * by[s]["direction"], 1) for s in SPLITS},
        "direction_ci95_pct": {s: [round(100 * v, 1) for v in by[s]["direction_ci95"]] for s in SPLITS},
        "direction_plan_only_pct": {s: round(100 * by[s]["direction_plan_only"], 1) for s in SPLITS},
        "shortfall_borrowing_pct": {s: round(100 * by[s]["shortfall_borrowing_share"], 1) for s in SPLITS},
        "rating_link_oot": {"movers": out["rating_link_oot"]["movers"], "simulated_pct": round(100 * out["rating_link_oot"]["simulated"], 1),
                            "simulated_ci95_pct": [round(100 * v, 1) for v in out["rating_link_oot"]["simulated_ci95"]],
                            "actual_pct": round(100 * out["rating_link_oot"]["actual"], 1), "upgrade_pct": round(100 * out["rating_link_oot"]["upgrade_share"], 1)},
        "excluded": {k: out["status_counts"].get(k, 0) for k in ("no_business_plan", "primitives_not_rebuilt", "simulator_refused", "no_consecutive_years")},
        "path_check": {"matched": out["path_check"]["matched"], "same_score_pct": round(100 * out["path_check"]["same_score_share"], 1),
                       "rank_correlation": round(out["path_check"]["rank_correlation"], 3)},
    }
    out["thesis_values"] = got
    out["checks"] = {k: got[k] == EXPECTED[k] for k in EXPECTED}
    ok = (all(out["checks"].values()) and out["replay_identity_max_relative_error"] <= IDENTITY_MAX
          and out["replay_negative_cash_states"] == 0 and out["replay_negative_current_assets_states"] == 0)
    out["status"] = "PASS" if ok else "FAIL"
    (HERE / "table4_2_simulator_replay.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    for s in SPLITS:
        b = by[s]
        print(f"{s:4s} replayed {b['replayed']:5d}  gap median {b['median_gap']:+.2f}  |gap| median {b['median_abs_gap']:.2f}  "
              f"sim higher {100 * b['share_sim_higher']:.1f}%  rank {b['rank_correlation']:.3f}  "
              f"direction {100 * b['direction']:.1f}% [{100 * b['direction_ci95'][0]:.1f}, {100 * b['direction_ci95'][1]:.1f}] (n={b['direction_n']})  "
              f"| plan only {100 * b['direction_plan_only']:.1f}%, last-year rank {b['rank_correlation_last_year_score']:.3f}, shortfall borrowing {100 * b['shortfall_borrowing_share']:.1f}%")
    r = out["rating_link_oot"]
    print(f"rating link OOT movers {r['movers']}: simulated {100 * r['simulated']:.1f}%, actual {100 * r['actual']:.1f}%, upgrades {100 * r['upgrade_share']:.1f}%")
    print("excluded", got["excluded"], "| path check", got["path_check"], "| identity max", f"{out['replay_identity_max_relative_error']:.1e}",
          "| negative cash / current assets", out["replay_negative_cash_states"], out["replay_negative_current_assets_states"])
    print({k: v for k, v in out["checks"].items() if not v} or "all checks match", "->", out["status"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

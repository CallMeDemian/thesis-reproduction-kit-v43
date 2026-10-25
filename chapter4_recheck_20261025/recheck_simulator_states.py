"""Recheck of thesis Table 4-3: how the evaluation Simulator applies the 8 fixed candidates to the 575 evaluation firms.

Simulates A0 and the 8 fixed candidates for every evaluation firm with the production path used for the main
experiment (final_stage8_llm_multi_oracle_eval.pipeline.simulate_llm_policy_states, stage6 simulator identity), then
1. checks the balance-sheet identities and non-negative cash / current assets on all 5,175 states;
2. compares each candidate with the same firm's A0 state: the accounts the candidate must move have to move in the
   contracted direction (or stay put when there is no room to act), revenue must not change, the short-to-long
   refinancing must keep total debt, interest, cash and total assets unchanged, and the operating candidates
   (CX, WC1, WC2, OE) may raise cash or lower the closing-date shortfall borrowing but never the reverse;
3. explains every firm where DL, RF or CX changes nothing, from the simulator's own repayment diagnostics.

Run root: the E1 scoring run root laid out by ablation_20261102/E1/scripts/e1_setup_runroot.ps1.
Usage (kit root, Python 3.11 kit environment):
    python chapter4_recheck_20261025/recheck_simulator_states.py --run-root <run root>
Writes chapter4_recheck_20261025/table4_3_simulator_actions.json and exits 1 if a result differs from the thesis.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KIT = HERE.parent
OBS = KIT / "frozen/evidence/v1.3/THESIS_REPRO_KIT_v1.3_FINAL_RENDER_VERIFIED_CLEAN/frozen_inputs/stage8/canonical_itt_observations.parquet"
DIMENSIONS = ("growth_capex_reduction_pct", "deleveraging_total_debt_pct", "refinancing_short_debt_pct", "inv_turnover_chg",
              "ar_turnover_chg", "ap_turnover_chg", "cogs_ratio_chg", "sga_ratio_chg")
ACTION_COLS = tuple(f"action__{d}" for d in DIMENSIONS)

# (account, expected change versus A0). '-'/'+': may not move the other way, counted when it moves;
# '0': must not change; '-?'/'+?': may not move the other way, not counted as reflection.
# 'debt' = short-term debt + current portion of long-term debt + long-term debt + bonds.
RULES = {
    "DL": [("debt", "-"), ("sim__pure_interest_expense", "-"), ("sim__cash", "-")],
    "RF": [("sim__short_term_debt", "-"), ("long_gross", "+"), ("sim__current_liabilities", "-"), ("debt", "0"),
           ("sim__pure_interest_expense", "0"), ("sim__cash", "0"), ("sim__total_assets", "0")],
    "CX": [("sim__capex", "-"), ("sim__ppe", "-"), ("sim__cash", "+?"), ("debt", "-?")],
    "WC1": [("sim__inventory", "-"), ("sim__receivables", "-"), ("sim__cash", "+?"), ("debt", "-?")],
    "WC2": [("sim__inventory", "-"), ("sim__receivables", "-"), ("sim__payables", "+"), ("sim__cash", "+?"), ("debt", "-?")],
    "OE": [("sim__cogs", "-"), ("sim__sga", "-"), ("sim__operating_income", "+"), ("sim__net_income", "+"), ("sim__total_equity", "+"),
           ("sim__cash", "+?"), ("debt", "-?")],
    "MX1": [("debt", "-"), ("sim__pure_interest_expense", "-"), ("sim__operating_income", "+")],
    "MX2": [("debt", "-"), ("sim__inventory", "-"), ("sim__receivables", "-"), ("sim__payables", "+")],
}
# Thesis Table 4-3 (2026-10-25 revision) and section 4.2.
EXPECTED_REFLECTED = {"DL": 267, "RF": 383, "CX": 448, "WC1": 575, "WC2": 575, "OE": 575, "MX1": 575, "MX2": 575}
EXPECTED_REPAYMENT = {"DL": 267, "MX1": 279, "MX2": 298}
EXPECTED_NOT_REFLECTED = {"DL_no_opening_debt": 116, "DL_debt_but_no_cash_above_liquidity_reserve": 192,
                          "RF_no_opening_short_term_debt": 192, "CX_plan_capex_not_above_depreciation": 127}
IDENTITY_MAX = 1e-12


def tolerance(base: pd.Series) -> np.ndarray:
    # accounting_closing_contract: incremental residual 10 (thousand KRW) absolute or 1e-8 relative
    return np.maximum(10.0, 1e-8 * base.abs().to_numpy())


def fcf(state: pd.DataFrame, key: str) -> pd.Series:
    return state["diagnostics_json"].map(lambda s: float(json.loads(s)["fcf_components"][key]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-root", required=True, type=Path)
    run = ap.parse_args().run_root.resolve()
    os.environ["THESIS_REPRO_RUN_ROOT"] = str(run)
    sys.path.insert(0, str(KIT / "src"))
    from credit_recourse.eval.final_stage8_llm_multi_oracle_eval import pipeline as p
    from credit_recourse.simulator.business_plan_factory import CALIBRATED_BP_RATE_V4_MODE

    contract = json.loads((run / "contracts/scientific/v43_action_contract.json").read_text(encoding="utf-8"))
    ident = p._load_stage6_simulator_identity(run)
    if ident["sim_business_plan_mode"] != CALIBRATED_BP_RATE_V4_MODE or ident["preserve_current_non_current_residual"] is not True:
        raise SystemExit(f"run root is not the production simulator identity: {ident}")
    base = pd.read_parquet(run / "03_stage2" / "phase_eval_candidate.parquet")
    rows = sorted(pd.read_parquet(OBS, columns=["row_id"]).row_id.astype(int).unique())
    if len(rows) != 575:
        raise SystemExit(f"expected 575 evaluation firms, got {len(rows)}")
    acts = pd.DataFrame([{"row_id": r, "candidate_id": cid, "policy": "TABLE4_3",
                          **{c: float(contract["fixed_candidates"][cid][c]) for c in ACTION_COLS}}
                         for cid in contract["candidate_ids"] for r in rows])
    state, audit = p.simulate_llm_policy_states(
        base, acts, p.load_current_action_space(run), run, predicted_fiscal_year=int(ident["predicted_fiscal_year"]),
        preserve_current_non_current_residual=True, sim_business_plan_mode=str(ident["sim_business_plan_mode"]),
        write_outputs=False, collect_audit=True, project_root=run)

    def col(name: str) -> pd.Series:
        return state[f"sim__{name}"].astype(float)

    state["debt"] = col("short_term_debt") + col("current_portion_long_debt") + col("long_term_debt") + col("bonds")
    state["long_gross"] = col("current_portion_long_debt") + col("long_term_debt") + col("bonds")
    ta = col("total_assets")
    out: dict = {"run_root_identity": {k: ident[k] for k in ("sim_business_plan_mode", "preserve_current_non_current_residual", "predicted_fiscal_year")},
                 "action_contract": contract.get("candidate_action_contract_version"),
                 "simulator_semantic_contract": contract.get("simulator_action_semantic_contract_version"),
                 "states": int(len(state)), "firms": len(rows)}
    out["identity_max_relative_error"] = {
        "assets = liabilities + equity": float(((ta - col("total_liabilities") - col("total_equity")).abs() / ta.abs()).max()),
        "assets = current + non-current": float(((ta - col("current_assets") - col("non_current_assets")).abs() / ta.abs()).max()),
        "liabilities = current + non-current": float(((col("total_liabilities") - col("current_liabilities") - col("non_current_liabilities")).abs() / ta.abs()).max()),
    }
    out["negative_cash_states"] = int((col("cash") < -1e-6).sum())
    out["negative_current_assets_states"] = int((col("current_assets") < -1e-6).sum())
    a0 = state[state.candidate_id == "A0"].set_index("row_id")
    checks, reflected, ok = {}, {}, True
    for cid, rules in RULES.items():
        x = state[state.candidate_id == cid].set_index("row_id")
        ref = a0.loc[x.index]
        res = {}
        any_moved = np.zeros(len(x), dtype=bool)
        for account, sign in [*rules, ("sim__revenue", "0")]:
            d = (x[account] - ref[account]).to_numpy()
            t = tolerance(ref[account])
            if sign == "0":
                res[account] = {"expected": "unchanged", "violations": int((np.abs(d) > t).sum())}
                continue
            down = sign.startswith("-")
            wrong = d > t if down else d < -t
            moved = d < -t if down else d > t
            if not sign.endswith("?"):
                any_moved |= moved
            res[account] = {"expected": ("decrease" if down else "increase") + (" or unchanged" if sign.endswith("?") else ""),
                            "moved": int(moved.sum()), "wrong_direction": int(wrong.sum())}
        checks[cid] = res
        reflected[cid] = pd.Series(any_moved, index=x.index)
        ok &= int(any_moved.sum()) == EXPECTED_REFLECTED[cid]
        ok &= all(v.get("wrong_direction", 0) == 0 and v.get("violations", 0) == 0 for v in res.values())
    out["candidate_vs_A0"] = checks
    out["reflected_firms"] = {k: int(v.sum()) for k, v in reflected.items()}

    # repayment and the reasons for no change, from the simulator's own diagnostics (fcf_components)
    def cand(cid: str) -> pd.DataFrame:
        return state[state.candidate_id == cid].set_index("row_id")

    repaid = {cid: fcf(cand(cid), "realized_repayment") > 1e-6 for cid in ("DL", "MX1", "MX2")}
    out["repayment_firms"] = {k: int(v.sum()) for k, v in repaid.items()}
    ok &= out["repayment_firms"] == EXPECTED_REPAYMENT
    dl = cand("DL")
    opening_debt = fcf(dl, "total_interest_bearing_debt_old")
    available = fcf(dl, "available_repayment_cash")
    dl_not = ~reflected["DL"].reindex(dl.index)
    opening_sd = state.loc[state.candidate_id == "RF"].set_index("row_id")["state__short_term_debt"].astype(float)
    rf_not = ~reflected["RF"]
    cx_not = ~reflected["CX"]
    out["not_reflected"] = {
        "DL_no_opening_debt": int((dl_not & (opening_debt <= 10.0)).sum()),
        "DL_debt_but_no_cash_above_liquidity_reserve": int((dl_not & (opening_debt > 10.0) & (available <= 1e-6)).sum()),
        "RF_no_opening_short_term_debt": int((rf_not & (opening_sd.reindex(rf_not.index).fillna(0.0).abs() <= 10.0)).sum()),
        "CX_plan_capex_not_above_depreciation": int((cx_not & (a0.loc[cx_not.index, "sim__capex"] <= a0.loc[cx_not.index, "sim__depreciation"] + 10.0)).sum()),
    }
    out["not_reflected_unexplained"] = {
        "DL": int(dl_not.sum()) - out["not_reflected"]["DL_no_opening_debt"] - out["not_reflected"]["DL_debt_but_no_cash_above_liquidity_reserve"],
        "RF": int(rf_not.sum()) - out["not_reflected"]["RF_no_opening_short_term_debt"],
        "CX": int(cx_not.sum()) - out["not_reflected"]["CX_plan_capex_not_above_depreciation"],
    }
    ok &= out["not_reflected"] == EXPECTED_NOT_REFLECTED and not any(out["not_reflected_unexplained"].values())
    out["dl_partial_repayment_firms"] = int((repaid["DL"] & (fcf(dl, "cash_feasibility_scale") < 1 - 1e-12)).sum())
    a0_plug = fcf(cand("A0"), "liquidity_shortfall_financing") > 1e-6
    out["shortfall_borrowing_firms"] = {"A0": int(a0_plug.sum()),
                                        **{cid: int((fcf(cand(cid), "liquidity_shortfall_financing") > 1e-6).sum()) for cid in RULES}}
    out["audit"] = {"rows": int(len(audit)),
                    "accounting_check_not_ok": int((~audit.accounting_check.astype(str).str.contains('"ok"')).sum()),
                    "preflight_not_ok": int((audit.simulator_preflight_status.astype(str) != "ok").sum()),
                    "clipped": int(audit.clipped.astype(str).eq("True").sum())}
    ok &= max(out["identity_max_relative_error"].values()) <= IDENTITY_MAX
    ok &= out["negative_cash_states"] == 0 and out["negative_current_assets_states"] == 0
    ok &= out["audit"]["accounting_check_not_ok"] == 0 and out["audit"]["preflight_not_ok"] == 0
    out["status"] = "PASS" if ok else "FAIL"
    (HERE / "table4_3_simulator_actions.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"{out['states']} states; identity max rel error {max(out['identity_max_relative_error'].values()):.1e}; "
          f"negative cash {out['negative_cash_states']}, negative current assets {out['negative_current_assets_states']}")
    for cid, res in checks.items():
        print(cid, "reflected", out["reflected_firms"][cid], {k.replace("sim__", ""): (v.get("moved"), v.get("wrong_direction", v.get("violations"))) for k, v in res.items()})
    print("repayment", out["repayment_firms"], "| partial DL", out["dl_partial_repayment_firms"], "| not reflected", out["not_reflected"],
          "| unexplained", out["not_reflected_unexplained"], "| shortfall borrowing", out["shortfall_borrowing_firms"])
    print("->", out["status"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""T2 (S1): High minus Baseline for C4 and C6-E; FREE8, IC-b, Run1, Strict ITT; B1 and BINF; 24 rows, one Holm family.

EXPLORATORY: the generation setting package (reasoning strength, output limit, run date) changes together, and
High was run after the Baseline results were known.
"""
from __future__ import annotations

import pandas as pd

import common as c

NOTE = "generation-setting package (reasoning strength, output limit, run time) changes together; High was run after Baseline results"


def main() -> pd.DataFrame:
    obs = c.load_obs()
    base = obs[(obs.execution_policy == "STRICT") & (obs.analysis_population.astype(str).str.lower() == "itt")
               & (obs.action_space == "FREE8") & (obs.information_condition == "IC-b") & (obs.replicate == 1)]
    rows = []
    for policy in ("C4", "C6-E"):
        for budget in ("B1", "BINF"):
            for model in c.MODELS:
                for oracle in c.ORACLES:
                    sub = base[(base.policy_condition == policy) & (base.budget == budget) & (base.model_key == model)]
                    pair = c.paired_by(sub, "reasoning_regime", "HIGH", "BASELINE", f"delta_R_score_{oracle}", ["firm_key"])
                    if len(pair) != 575:
                        raise SystemExit(f"S1: {policy}/{budget}/{model}/{oracle} has {len(pair)} firms")
                    key = c.seed_key(analysis_id="ABL_S1", lhs=policy, rhs=policy, model=c.MODEL_CODE[model], oracle=oracle,
                                     reasoning_regime="HIGH_MINUS_BASELINE", budget=budget, action_space="FREE8")
                    res = c.boot(pair["difference"].to_numpy(), key)
                    rows.append({
                        "result_id": f"T2_S1_{policy}_{c.MODEL_CODE[model]}_{budget}_{oracle}", "status": "EXPLORATORY",
                        "contrast_id": "S1", "lhs": policy, "rhs": policy, "model": c.MODEL_CODE[model],
                        "reasoning_regime": "HIGH_MINUS_BASELINE", "oracle": oracle, "budget": budget, "action_space": "FREE8",
                        "execution_policy": "STRICT", "replicate": "RUN1", "information_condition": "IC-b",
                        "n_firms": res["N"], "estimate": res["estimate"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                        "p_raw": res["raw_p"], "holm_p": float("nan"), "holm_family": "S1_ALL24", "holm_p_60": float("nan"),
                        "source_file": c.OBS_PATH.name, "source_key": res["key_json"], "source_sha256": c.sha256_file(c.OBS_PATH),
                        "analysis_commit": c.git_commit(), "notes": NOTE + f"; draws_sha256={res['bootstrap_draw_sha256']}",
                    })
    t2 = pd.DataFrame(rows, columns=c.RESULT_COLS)
    assert len(t2) == 24
    t2["holm_p"] = c.holm(t2["p_raw"].to_numpy())
    c.write_csv(t2, c.RESULTS / "T2_S1.csv")
    return t2


if __name__ == "__main__":
    t = main()
    print(len(t), "rows; positive", int((t.estimate > 0).sum()), "Holm<.05", int((t.holm_p < .05).sum()))

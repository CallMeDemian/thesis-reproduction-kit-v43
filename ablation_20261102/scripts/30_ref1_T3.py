"""T3 (REF-1): C6-E minus C3-E, Strict ITT, IC-b, Run1, B1.

CANDIDATE9: 12 tested rows (2 regimes x 2 models x 3 Oracles), one Holm family, EXPLORATORY.
FREE8: 12 descriptive rows (mean and interval, no test). C3-E is a fixed per-firm value (no generation setting,
no budget): matched by firm only. firm_key is empty in the stage9 file, so row_id joins it to the observations.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as c


def c3e_values(obs: pd.DataFrame) -> pd.DataFrame:
    s9 = pd.read_csv(c.require_file(c.C3E_PATH), low_memory=False)
    x = s9[(s9.policy == "C3-E") & (s9.analysis_population == "itt")].copy()
    m = obs[["row_id", "firm_key"]].drop_duplicates()
    x = x.drop(columns=["firm_key"]).merge(m, on="row_id", how="left", validate="one_to_one")
    if len(x) != 575 or x.firm_key.isna().any() or x.firm_key.duplicated().any():
        raise SystemExit("T3: C3-E row_id <-> firm_key join is not one-to-one over 575 firms")
    return x.set_index("firm_key").sort_index()


def main() -> pd.DataFrame:
    obs = c.load_obs()
    c3e = c3e_values(obs)
    checks = {"c3e_alpha_mean": float(c3e["delta_R_score_alpha"].mean()), "c3e_rows": len(c3e)}
    llm = obs[(obs.execution_policy == "STRICT") & (obs.analysis_population.astype(str).str.lower() == "itt")
              & (obs.information_condition == "IC-b") & (obs.replicate == 1) & (obs.budget == "B1") & (obs.policy_condition == "C6-E")]
    rows = []
    for space, aid, status in (("CANDIDATE9", "ABL_REF1", "EXPLORATORY"), ("FREE8", "ABL_REF1_FREE8", "DESCRIPTIVE")):
        for regime in ("BASELINE", "HIGH"):
            for model in c.MODELS:
                for oracle in c.ORACLES:
                    sub = llm[(llm.action_space == space) & (llm.reasoning_regime == regime) & (llm.model_key == model)]
                    if sub.firm_key.duplicated().any() or len(sub) != 575:
                        raise SystemExit(f"REF-1: {space}/{regime}/{model} has {len(sub)} C6-E rows")
                    sub = sub.set_index("firm_key").sort_index()
                    if list(sub.index) != list(c3e.index):
                        raise SystemExit("REF-1: firm sets differ")
                    diff = (sub[f"delta_R_score_{oracle}"] - c3e[f"delta_R_score_{oracle}"]).to_numpy()
                    key = c.seed_key(analysis_id=aid, lhs="C6-E", rhs="C3-E", model=c.MODEL_CODE[model], oracle=oracle,
                                     reasoning_regime=regime, budget="B1", action_space=space)
                    res = c.boot(diff, key)
                    tested = status == "EXPLORATORY"
                    if regime == "HIGH" and space == "FREE8":
                        pass
                    rows.append({
                        "result_id": f"T3_REF1_{space}_{c.MODEL_CODE[model]}_{regime}_{oracle}", "status": status,
                        "contrast_id": "REF-1", "lhs": "C6-E", "rhs": "C3-E", "model": c.MODEL_CODE[model],
                        "reasoning_regime": regime, "oracle": oracle, "budget": "B1", "action_space": space,
                        "execution_policy": "STRICT", "replicate": "RUN1", "information_condition": "IC-b",
                        "n_firms": res["N"], "estimate": res["estimate"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
                        "p_raw": res["raw_p"] if tested else float("nan"), "holm_p": float("nan"),
                        "holm_family": "REF1_CANDIDATE9_12" if tested else "N/A", "holm_p_60": float("nan"),
                        "source_file": f"{c.OBS_PATH.name}; {c.C3E_PATH.name}", "source_key": res["key_json"],
                        "source_sha256": c.sha256_file(c.OBS_PATH) + ";" + c.sha256_file(c.C3E_PATH),
                        "analysis_commit": c.git_commit(),
                        "notes": ("same 9 candidates; baseline comparison, not a component removal; no equivalence claim; C3-E was selected on 2024 data"
                                  if tested else "different action spaces (free8 vs C3-E 9 candidates); mean and interval only, no test")
                                 + f"; draws_sha256={res['bootstrap_draw_sha256']}",
                    })
    t3 = pd.DataFrame(rows, columns=c.RESULT_COLS)
    assert len(t3) == 24 and (t3.status == "EXPLORATORY").sum() == 12
    mask = t3.holm_family == "REF1_CANDIDATE9_12"
    t3.loc[mask, "holm_p"] = c.holm(t3.loc[mask, "p_raw"].to_numpy())
    c.write_csv(t3, c.RESULTS / "T3_REF1.csv")
    # check against the stage9 paired file: High mean difference vs C3-E must equal the stage9 gap (ITT)
    gaps = pd.read_parquet(c.require_file(c.KIT / c.S9 / "llm_stage9_paired_vs_C3-E.parquet"))
    checks["stage9_gap_columns"] = [x for x in gaps.columns if x.startswith("gap_")]
    checks["stage9_gap_rows"] = int(len(gaps))
    out = {}
    pol = [x for x in ("policy", "policy_condition") if x in gaps.columns]
    for model in c.MODELS:
        for oracle in c.ORACLES:
            for space, sp in (("CANDIDATE9", "candidate9"), ("FREE8", "free8")):
                g = gaps
                for col, val in (("analysis_population", "itt"), ("model_key", model), ("mode", sp), ("replicate", 1), ("budget", "B1"),
                                 ("information_condition", "IC-b"), ("policy", "C6-E")):
                    if col in g.columns:
                        g = g[g[col] == val]
                if "wave" in g.columns:
                    pass
                mean_gap = [float(g[f"gap_{oracle}"].mean()), int(len(g))]
                mine = t3[(t3.model == c.MODEL_CODE[model]) & (t3.oracle == oracle) & (t3.action_space == space) & (t3.reasoning_regime == "HIGH")]
                out[f"{c.MODEL_CODE[model]}_{oracle}_{space}"] = {"stage9_gap_mean_n": mean_gap, "t3_high_estimate": float(mine.estimate.iloc[0])}
    checks["high_mean_vs_stage9_gap"] = out
    (c.RESULTS / "_T3_checks.json").write_text(json.dumps(checks, indent=1) + "\n", encoding="utf-8", newline="\n")
    return t3


if __name__ == "__main__":
    t = main()
    tt = t[t.status == "EXPLORATORY"]
    print(len(t), "rows; candidate9 positive", int((tt.estimate > 0).sum()), "/12; Holm<.05", int((tt.holm_p < .05).sum()))
    print(json.dumps(json.load(open(c.RESULTS / "_T3_checks.json")), indent=1)[:2500])

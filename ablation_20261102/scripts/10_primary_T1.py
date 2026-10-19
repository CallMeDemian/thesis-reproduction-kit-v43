"""T1: the 96 main-analysis rows (A1-A4) through the kit's MANUSCRIPT_FROZEN runner path.

Means are recomputed from the firm-level observations and must equal the frozen means under
np.isclose(atol=1e-12, rtol=1e-12) (the check inside the kit's ``_primary_results``). CI, p and Holm are the
frozen values. High rows get holm_p_60 from HIGH_REASONING_SUPPLEMENTAL.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as c
from credit_recourse.analysis.final_v43.runner import _primary_results
from credit_recourse.analysis.final_v43.schema import validate

CONTRAST = {"C5-C4": ("A1", "C5", "C4"), "C4R-C4": ("A2", "C4R", "C4"),
            "C6-E-C4R": ("A3", "C6-E", "C4R"), "C6-E-C6-EX": ("A4", "C6-E", "C6-EX")}


def main() -> pd.DataFrame:
    obs = c.load_obs()
    validate(obs)
    prim = _primary_results(c.KIT, obs, "MANUSCRIPT_FROZEN", "ABLATION_20261102")
    assert len(prim) == 96
    supp = pd.read_csv(c.SUPP_PATH)
    s60 = supp[supp.holm_family_id == "HIGH_REASONING_SUPPLEMENTAL"].copy()
    s60["oracle_l"] = s60.oracle.str.lower()
    rows = []
    for _, r in prim.iterrows():
        cid, lhs, rhs = CONTRAST[r["contrast"]]
        p60 = np.nan
        note = "Strict ITT; means recomputed from observations and equal frozen means (atol=rtol=1e-12); CI, p, Holm frozen"
        if r["reasoning_regime"] == "HIGH":
            m = s60[(s60.model == c.MODEL_CODE[r["model_key"]]) & (s60.oracle_l == r["oracle"]) & (s60.budget == r["budget"])
                    & (s60.action_space == "FREE8") & (s60.lhs == lhs) & (s60.rhs == rhs)]
            if len(m) != 1:
                raise SystemExit(f"T1: no unique 60-family row for {r['model_key']}/{r['oracle']}/{r['budget']}/{r['contrast']}")
            if not np.isclose(m["estimate"].iloc[0], r["estimate"], atol=1e-6):
                raise SystemExit("T1: 60-family estimate differs from primary estimate")
            p60 = float(m["holm_p"].iloc[0])
            note += "; High rows were added after the Baseline results (SUPPLEMENTAL_REASONING); holm_p_60 from HIGH_REASONING_SUPPLEMENTAL"
        src = c.PRIM_BASE if r["reasoning_regime"] == "BASELINE" else c.PRIM_HIGH
        rows.append({
            "result_id": f"T1_{cid}_{c.MODEL_CODE[r['model_key']]}_{r['reasoning_regime']}_{r['budget']}_{r['oracle']}",
            "status": "FROZEN", "contrast_id": cid, "lhs": lhs, "rhs": rhs, "model": c.MODEL_CODE[r["model_key"]],
            "reasoning_regime": r["reasoning_regime"], "oracle": r["oracle"], "budget": r["budget"], "action_space": "FREE8",
            "execution_policy": "STRICT", "replicate": "RUN1", "information_condition": "IC-b",
            "n_firms": int(r["N_frozen"]), "estimate": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "p_raw": r["raw_p"], "holm_p": r["holm_p"],
            "holm_family": f"PRIMARY_{c.MODEL_CODE[r['model_key']]}_{r['oracle']}_{r['reasoning_regime']}_8",
            "holm_p_60": p60, "source_file": src.name,
            "source_key": f"{r['model_key']}|{r['reasoning_regime']}|{r['budget']}|{r['contrast']}|{r['oracle']}",
            "source_sha256": c.sha256_file(src), "analysis_commit": c.git_commit(), "notes": note,
        })
    t1 = pd.DataFrame(rows, columns=c.RESULT_COLS)
    assert len(t1) == 96 and not t1.result_id.duplicated().any()
    assert (t1.n_firms == 575).all()
    c.write_csv(t1, c.RESULTS / "T1_primary_A1_A4.csv")
    meta = {"rows": 96, "mean_check": "np.isclose(atol=1e-12, rtol=1e-12) PASS (kit runner)",
            "a3_sign_counts": {"positive_of_24_free8": int(((t1.contrast_id == "A3") & (t1.estimate > 0)).sum()),
                               "holm_lt_05": int(((t1.contrast_id == "A3") & (t1.holm_p < 0.05)).sum()),
                               "holm60_lt_05": int(((t1.contrast_id == "A3") & (t1.holm_p_60 < 0.05)).sum())}}
    (c.RESULTS / "_T1_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return t1


if __name__ == "__main__":
    t = main()
    print(len(t), "rows written")
    for cid in ("A1", "A2", "A3", "A4"):
        s = t[t.contrast_id == cid]
        h = s[s.reasoning_regime == "HIGH"]
        print(cid, "positive", int((s.estimate > 0).sum()), "/", len(s), "| Holm<.05", int((s.holm_p < .05).sum()),
              "| High rows:", len(h), "holm60<.05", int((h.holm_p_60 < .05).sum()))

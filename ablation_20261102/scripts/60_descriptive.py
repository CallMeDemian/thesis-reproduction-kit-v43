"""Descriptive tables computed from the result files (no new statistics, nothing pre-registered).

D1_E1c_vs_A3.csv   E1c (new batch) next to the frozen A3 (High, free8, B1); run dates differ, no test between them.
D2_S1_direction.csv  direction and significance of the 24 S1 rows with the significant negative rows flagged.
D3_E1a_by_oracle.csv E1a (343 firms) per model and Oracle with direction and Holm p (pre-registered family).
"""
from __future__ import annotations

import pandas as pd

import common as c


def main():
    t1 = pd.read_csv(c.RESULTS / "T1_primary_A1_A4.csv")
    t2 = pd.read_csv(c.RESULTS / "T2_S1.csv")
    t5 = pd.read_csv(c.RESULTS / "T5_E1.csv")
    rows = []
    for m in ("GPT", "GEMINI"):
        for o in c.ORACLES:
            e = t5[(t5.contrast_id == "E1c") & (t5.model == m) & (t5.oracle == o)].iloc[0]
            a = t1[(t1.contrast_id == "A3") & (t1.model == m) & (t1.oracle == o) & (t1.reasoning_regime == "HIGH") & (t1.budget == "B1")].iloc[0]
            rows.append({"status": "DESCRIPTIVE", "model": m, "oracle": o, "n_firms": 575,
                         "e1c_estimate": e.estimate, "e1c_ci_low": e.ci_low, "e1c_ci_high": e.ci_high, "e1c_holm_p": e.holm_p,
                         "a3_estimate": a.estimate, "a3_ci_low": a.ci_low, "a3_ci_high": a.ci_high, "a3_holm_p": a.holm_p, "a3_holm_p_60": a.holm_p_60,
                         "difference_e1c_minus_a3": e.estimate - a.estimate,
                         "notes": "not pre-registered; different run dates; no test between the two; A3 is frozen (free8, High, B1, Strict ITT)"})
    c.write_csv(pd.DataFrame(rows), c.RESULTS / "D1_E1c_vs_A3.csv")

    s = t2.copy()
    s["direction"] = (s.estimate > 0).map({True: "positive", False: "negative"})
    s["holm_lt_05"] = s.holm_p < 0.05
    s["significant_negative"] = s.holm_lt_05 & (s.estimate < 0)
    d2 = s[["result_id", "lhs", "budget", "model", "oracle", "estimate", "ci_low", "ci_high", "holm_p", "direction", "holm_lt_05", "significant_negative"]].copy()
    d2.insert(0, "status", "DESCRIPTIVE")
    c.write_csv(d2, c.RESULTS / "D2_S1_direction.csv")

    a = t5[t5.contrast_id == "E1a"].copy()
    a["direction"] = (a.estimate > 0).map({True: "positive", False: "negative"})
    a["holm_lt_05"] = a.holm_p < 0.05
    d3 = a[["contrast_id", "model", "oracle", "n_firms", "estimate", "ci_low", "ci_high", "p_raw", "holm_p", "direction", "holm_lt_05"]].copy()
    d3.insert(0, "status", "EXPLORATORY")
    c.write_csv(d3, c.RESULTS / "D3_E1a_by_oracle.csv")
    return d2


def d4():
    """D4: execution-rule and repetition values for the report and slides (free8, B1, Oracle-alpha), taken from frozen rows."""
    supp = pd.read_csv(c.SUPP_PATH)
    t1 = pd.read_csv(c.RESULTS / "T1_primary_A1_A4.csv")
    rep = supp[(supp.holm_family_id == "REPAIRED_SUPPLEMENTAL") & (supp.action_space == "FREE8") & (supp.budget == "B1") & (supp.oracle == "ALPHA")]
    exe = supp[(supp.analysis_id == "EXECUTION_SENSITIVITY") & (supp.action_space == "FREE8") & (supp.budget == "B1") & (supp.oracle == "ALPHA")]
    stab = supp[(supp.holm_family_id == "STABILITY_SUPPLEMENTAL") & (supp.execution_policy == "STRICT") & (supp.budget == "B1") & (supp.oracle == "ALPHA") & (supp.replicate == 2)]
    rows = []
    for cid, lhs, rhs in (("A2", "C4R", "C4"), ("A3", "C6-E", "C4R")):
        for m in ("GPT", "GEMINI"):
            for r in ("BASELINE", "HIGH"):
                s = t1[(t1.contrast_id == cid) & (t1.model == m) & (t1.reasoning_regime == r) & (t1.budget == "B1") & (t1.oracle == "alpha")].iloc[0]
                p = rep[(rep.model == m) & (rep.reasoning_regime == r) & (rep.lhs == lhs) & (rep.rhs == rhs)].iloc[0]
                x = exe[(exe.model == m) & (exe.reasoning_regime == r) & (exe.lhs == lhs) & (exe.rhs == rhs)].iloc[0]
                q = stab[(stab.model == m) & (stab.reasoning_regime == r) & (stab.lhs == lhs) & (stab.rhs == rhs)].iloc[0]
                rows.append({"status": "FROZEN", "contrast_id": cid, "model": m, "reasoning_regime": r, "oracle": "alpha", "budget": "B1",
                             "strict_estimate": s.estimate, "strict_ci_low": s.ci_low, "strict_ci_high": s.ci_high,
                             "repaired_estimate": p.estimate, "repaired_ci_low": p.ci_low, "repaired_ci_high": p.ci_high,
                             "repaired_minus_strict": x.estimate, "rms_ci_low": x.ci_low, "rms_ci_high": x.ci_high,
                             "run1_strict_estimate": s.estimate, "run2_strict_estimate": q.estimate, "run2_ci_low": q.ci_low, "run2_ci_high": q.ci_high,
                             "source": "T1; V13_SUPPLEMENTAL_RESULTS (REPAIRED_SUPPLEMENTAL, EXECUTION_SENSITIVITY, STABILITY_SUPPLEMENTAL)"})
    c.write_csv(pd.DataFrame(rows), c.RESULTS / "D4_execution_and_repetition.csv")


B, E = "<!-- QA_INTERP_BEGIN -->", "<!-- QA_INTERP_END -->"


def f3(x):
    return f"{x:+.3f}"


def qa_block():
    """Interpretation records for QA_REPORT.md, generated from D1-D3 and T5 (every number traced to a results file)."""
    d1 = pd.read_csv(c.RESULTS / "D1_E1c_vs_A3.csv")
    d2 = pd.read_csv(c.RESULTS / "D2_S1_direction.csv")
    d3 = pd.read_csv(c.RESULTS / "D3_E1a_by_oracle.csv")
    L = [B, "", "## Interpretation records (generated by scripts/60_descriptive.py)", "",
         "### E1a (EXPLORATORY, 343 firms whose C3-E action is not OE; results/D3_E1a_by_oracle.csv)", "",
         "| Model | Oracle | Difference [95% CI] | Holm p | Direction |", "| --- | --- | --- | --- | --- |"]
    for r in d3.itertuples():
        L.append(f"| {r.model} | {r.oracle} | {f3(r.estimate)} [{f3(r.ci_low)}, {f3(r.ci_high)}] | {r.holm_p:.3f} | {r.direction}{'' if r.holm_lt_05 else ' (not significant)'} |")
    L += ["", "Reading: the learned action content raised policy value on alpha for both models (significant), on gamma for GPT (significant) and Gemini (positive, not significant), "
          "and lowered it slightly on beta for both models (significant). The pre-registered sentence is kept: \"평가모형에 따라 엇갈렸다\", always with the per-Oracle values above. "
          "Alpha is the evaluation model used to select C3-E on 2024 data, so the alpha gain is not read as independent evidence. Label: EXPLORATORY. "
          "The placebo gate (232 firms) passed (all Holm p = 1.000).", "",
          "### E1c next to the frozen A3 (DESCRIPTIVE, not pre-registered; results/D1_E1c_vs_A3.csv)", "",
          "Free-text, High, B1, Strict ITT, 575 firms. The two runs have different dates; no test between them.", "",
          "| Model | Oracle | E1c [95% CI] (Holm p) | A3 [95% CI] (Holm p, 60-family p) | E1c - A3 |", "| --- | --- | --- | --- | --- |"]
    for r in d1.itertuples():
        L.append(f"| {r.model} | {r.oracle} | {f3(r.e1c_estimate)} [{f3(r.e1c_ci_low)}, {f3(r.e1c_ci_high)}] ({r.e1c_holm_p:.3f}) | "
                 f"{f3(r.a3_estimate)} [{f3(r.a3_ci_low)}, {f3(r.a3_ci_high)}] ({r.a3_holm_p:.3f}, {r.a3_holm_p_60:.3f}) | {f3(r.difference_e1c_minus_a3)} |")
    neg = d2[d2.significant_negative]
    L += ["", "### S1 direction (EXPLORATORY; results/D2_S1_direction.csv)", "",
          f"Of {len(d2)} rows, {int((d2.direction == 'positive').sum())} are positive and {int((d2.direction == 'negative').sum())} negative; {int(d2.holm_lt_05.sum())} have Holm p < .05 "
          f"({int((d2.holm_lt_05 & (d2.direction == 'positive')).sum())} positive, {len(neg)} negative). The significant negative rows are:", ""]
    for r in neg.itertuples():
        L.append(f"- {r.model}, {r.lhs}, {r.budget}, Oracle-{r.oracle}: {f3(r.estimate)} [{f3(r.ci_low)}, {f3(r.ci_high)}], Holm p {r.holm_p:.3f}")
    L += ["", "Every S1 sentence names these rows and does not say that High raised policy value in every combination.", "", E]
    p = c.ABL / "QA_REPORT.md"
    text = p.read_text(encoding="utf-8")
    block = "\n".join(L)
    if B in text:
        pre, rest = text.split(B, 1)
        post = rest.split(E, 1)[1] if E in rest else ""
        text = pre + block + post
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    p.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    d2 = main()
    d4()
    qa_block()
    print("S1 significant negative:", d2[d2.significant_negative][["lhs", "budget", "model", "oracle"]].values.tolist())

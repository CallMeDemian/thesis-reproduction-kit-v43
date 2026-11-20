"""Detectable effect size for paired contrasts with one run vs the average of two runs.
Usage: python3 -I power.py obs.parquet
"""
import sys
import numpy as np, pandas as pd
df = pd.read_parquet(sys.argv[1])
SC = ["delta_R_score_alpha", "delta_R_score_beta", "delta_R_score_gamma"]
sub = df[(df.analysis_arm == "HIGH_STRICT_ITT") & (df["mode"] == "free8") & (df["info"] == "IC-b") & (df.budget == "B1")]
def get(m, pc, r):
    g = sub[(sub.model_key == m) & (sub.policy_condition == pc) & (sub.replicate == r)].sort_values("row_id")
    return g[SC].values
for m in sorted(sub.model_key.unique()):
    print("=====", m)
    A1, A2 = get(m, "C6-E", 1), get(m, "C6-E", 2)
    B1, B2 = get(m, "C4R", 1), get(m, "C4R", 2)
    D1, D2 = get(m, "C4", 1), get(m, "C4", 2)
    n = len(A1)
    # pooled sd for standardization (per evaluator, across all LLM outputs in this slice)
    allS = sub[sub.model_key == m][SC].values
    sd = allS.std(axis=0)
    for name, X1, X2 in [("C6-E - C4R", A1 - B1, A2 - B2), ("C6-E - C4", A1 - D1, A2 - D2), ("C4R - C4", B1 - D1, B2 - D2)]:
        avg = (X1 + X2) / 2
        for j, lab in enumerate(["alpha", "beta", "gamma"]):
            se1 = X1[:, j].std(ddof=1) / np.sqrt(n); se2 = X2[:, j].std(ddof=1) / np.sqrt(n); sea = avg[:, j].std(ddof=1) / np.sqrt(n)
            print(f"{name:11s} {lab:5s} r1 {X1[:, j].mean():+.3f} (se {se1:.3f}, MDE80 {2.8*se1:.3f}) | r2 {X2[:, j].mean():+.3f} (se {se2:.3f}) | avg {avg[:, j].mean():+.3f} (se {sea:.3f}, MDE80 {2.8*sea:.3f}) | corr(r1,r2) {np.corrcoef(X1[:, j], X2[:, j])[0,1]:+.2f}")
        z1 = (X1 / sd).mean(axis=1); z2 = (X2 / sd).mean(axis=1); za = (z1 + z2) / 2
        print(f"{name:11s} zmean r1 {z1.mean():+.3f} (se {z1.std(ddof=1)/np.sqrt(n):.3f}, t {z1.mean()/(z1.std(ddof=1)/np.sqrt(n)):.1f}) | avg {za.mean():+.3f} (se {za.std(ddof=1)/np.sqrt(n):.3f}, t {za.mean()/(za.std(ddof=1)/np.sqrt(n)):.1f})")

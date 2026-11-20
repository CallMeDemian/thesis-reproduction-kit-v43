"""Share of firms where a recommendation is worse than no action on some evaluator (harm) / better on all.
Usage: python3 -I harm.py obs.parquet
"""
import sys
import numpy as np, pandas as pd
df = pd.read_parquet(sys.argv[1])
rng = np.random.default_rng(11)
def ci(x, B=4000):
    x = np.asarray(x, float); n = len(x)
    m = x[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])
SC = ["delta_R_score_alpha", "delta_R_score_beta", "delta_R_score_gamma"]
eps = 1e-9
for arm in ["HIGH_STRICT_ITT"]:
    for mode in ["free8", "candidate9"]:
        sub = df[(df.analysis_arm == arm) & (df["mode"] == mode) & (df["info"] == "IC-b") & (df.budget == "B1") & (df.replicate == 1)]
        print(f"\n#### {arm} / {mode} / IC-b / B1 / r1")
        print(f"{'model':26s} {'cond':6s} {'a<0':>6s} {'b<0':>6s} {'g<0':>6s} {'any<0':>6s} {'all>0':>6s} {'all>=0,some>0':>14s} {'all==0':>7s}")
        keep = {}
        for (m, pc), g in sub.groupby(["model_key", "policy_condition"]):
            g = g.sort_values("row_id"); S = g[SC].values
            neg = S < -eps; pos = S > eps
            anyneg = neg.any(1); allpos = pos.all(1); dom = (~neg).all(1) & pos.any(1); zero = (~neg & ~pos).all(1)
            keep[(m, pc)] = dict(anyneg=anyneg, allpos=allpos)
            print(f"{m:26s} {pc:6s} {neg[:,0].mean():6.3f} {neg[:,1].mean():6.3f} {neg[:,2].mean():6.3f} {anyneg.mean():6.3f} {allpos.mean():6.3f} {dom.mean():14.3f} {zero.mean():7.3f}")
        for m in sorted({k[0] for k in keep}):
            for a, b in [("C6-E", "C4R"), ("C6-E", "C4"), ("C4R", "C4"), ("C6-E", "C6-EX")]:
                if (m, a) in keep and (m, b) in keep:
                    for k in ["anyneg", "allpos"]:
                        d = keep[(m, a)][k].astype(float) - keep[(m, b)][k].astype(float)
                        lo, hi = ci(d)
                        print(f"   {m[:12]:12s} {a} - {b:5s} [{k}] {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

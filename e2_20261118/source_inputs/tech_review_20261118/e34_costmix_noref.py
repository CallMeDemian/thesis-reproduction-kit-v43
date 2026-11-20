import sys
import numpy as np, pandas as pd
d = pd.read_parquet(sys.argv[1]); d = d[d.strict_usable]
cost = d.u_cogs_ratio_chg + d.u_sga_ratio_chg
x = d[(cost >= 0.9) & (d["T"] - cost < 0.02)].copy()
x["f"] = x.u_cogs_ratio_chg / (x.u_cogs_ratio_chg + x.u_sga_ratio_chg)
x["bin"] = np.where(x.f > 0.55, "cogs_heavy", np.where(x.f < 0.45, "sga_heavy", "balanced"))
x["grp"] = np.where(x.policy_condition.isin(["C6-E", "C6-EX"]), "with_reference", "no_reference(C4,C5,C4R)")
print(pd.crosstab([x.model_key, x.grp], x.bin, margins=True).to_string())
t = pd.crosstab(x.grp, x.bin); print((t.div(t.sum(1), axis=0)).round(3).to_string())
# among no-reference outputs only: firms with variation and slope
SC = ["delta_R_score_alpha", "delta_R_score_beta", "delta_R_score_gamma"]
y = x[x.grp.str.startswith("no_ref")].copy()
var = y.groupby("row_id").f.transform(lambda s: s.max() - s.min()); y = y[var > 0.15]
print("no-reference only: firms with variation", y.row_id.nunique(), "outputs", len(y))
rng = np.random.default_rng(5)
for c in SC:
    y["fd"] = y.f - y.groupby("row_id").f.transform("mean"); y["sd"] = y[c] - y.groupby("row_id")[c].transform("mean")
    slope = (y.fd * y.sd).sum() / (y.fd ** 2).sum()
    ids = y.row_id.unique(); byf = {i: t for i, t in y.groupby("row_id")}; bs = []
    for _ in range(1000):
        s = pd.concat([byf[i] for i in rng.choice(ids, len(ids))]); bs.append((s.fd * s.sd).sum() / (s.fd ** 2).sum())
    print(f"{c[14:]:6s} slope {slope:+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
# all usable no-reference outputs: mean units by axis
z = d[~d.policy_condition.isin(["C6-E", "C6-EX"])]
print("no-reference outputs:", len(z)); print(z[[c for c in z.columns if c.startswith("u_")]].mean().round(3).to_string())

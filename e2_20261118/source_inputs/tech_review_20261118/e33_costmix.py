"""Within-firm test: among cost-only actions, does shifting budget from SG&A to COGS raise scores? Usage: e33_costmix.py OUT.parquet"""
import sys
import numpy as np, pandas as pd
d = pd.read_parquet(sys.argv[1])
d = d[d.strict_usable]
cost = d.u_cogs_ratio_chg + d.u_sga_ratio_chg
x = d[(cost >= 0.9) & (d["T"] - cost < 0.02)].copy()
x["f"] = x.u_cogs_ratio_chg / (x.u_cogs_ratio_chg + x.u_sga_ratio_chg)
print("cost-only outputs", len(x), "firms", x.row_id.nunique())
print(pd.cut(x.f, [-0.01, 0.2, 0.45, 0.55, 0.8, 1.0]).value_counts().sort_index().to_string())
SC = ["delta_R_score_alpha", "delta_R_score_beta", "delta_R_score_gamma"]
# firms with variation in f
g = x.groupby("row_id")
var = g.f.transform(lambda s: s.max() - s.min())
y = x[var > 0.15].copy()
print("firms with variation in COGS fraction:", y.row_id.nunique(), "outputs", len(y), "f range", y.f.min().round(2), y.f.max().round(2))
rng = np.random.default_rng(5)
for c in SC:
    yy = y.copy()
    yy["fd"] = yy.f - yy.groupby("row_id").f.transform("mean")
    yy["sd"] = yy[c] - yy.groupby("row_id")[c].transform("mean")
    slope = (yy.fd * yy.sd).sum() / (yy.fd ** 2).sum()
    ids = yy.row_id.unique(); bs = []
    byf = {i: t for i, t in yy.groupby("row_id")}
    for _ in range(1000):
        s = pd.concat([byf[i] for i in rng.choice(ids, len(ids))])
        bs.append((s.fd * s.sd).sum() / (s.fd ** 2).sum())
    print(f"{c[14:]:6s} within-firm slope per +1.0 COGS fraction: {slope:+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
# paired: per firm, mean score of COGS-heavy (f>=0.55) vs balanced (0.45-0.55) vs SGA-heavy (f<=0.45)
x["bin"] = np.where(x.f >= 0.6, "cogs_heavy", np.where(x.f <= 0.4, "sga_heavy", "balanced"))
piv = x.pivot_table(index="row_id", columns="bin", values=SC, aggfunc="mean")
for c in SC:
    for a, b in [("cogs_heavy", "balanced"), ("balanced", "sga_heavy"), ("cogs_heavy", "sga_heavy")]:
        if a in piv[c] and b in piv[c]:
            dd = (piv[c][a] - piv[c][b]).dropna()
            if len(dd) > 5:
                bs = [dd.sample(len(dd), replace=True, random_state=int(r)).mean() for r in rng.integers(0, 1e9, 1000)]
                print(f"{c[14:]:6s} {a} - {b}: n={len(dd)} mean {dd.mean():+.3f} [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")

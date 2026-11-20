"""Evaluator-free self-consistency (medoid) over stored reference-free outputs, vs single draft and RL-reference revision.
Usage: python3 -I selfconsistency.py obs.parquet action_contract.json
"""
import sys, json
import numpy as np
import pandas as pd

df = pd.read_parquet(sys.argv[1])
con = json.load(open(sys.argv[2]))
dims = con["managerial_intensity_dimensions"]
scale = np.array([con["normalization_scale_by_dimension"][d] for d in dims])
def parse(js):
    d = json.loads(js); return np.array([float(d.get(k, 0.0)) for k in dims]) / scale
rng = np.random.default_rng(7)
def ci(x, B=4000):
    x = np.asarray(x, float); n = len(x)
    m = x[rng.integers(0, n, size=(B, n))].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])
def fmt(x):
    lo, hi = ci(x); star = "*" if (lo > 0 or hi < 0) else " "
    return f"{np.mean(x):+.3f}{star}[{lo:+.3f},{hi:+.3f}]"

arm = "HIGH_STRICT_ITT"
sub = df[(df.analysis_arm == arm) & (df["mode"] == "free8") & (df.budget == "B1")].copy()
sub["key"] = sub["policy_condition"] + "|" + sub["info"] + "|r" + sub["replicate"].astype(str)
print(sub.groupby(["model_key", "key"]).size().unstack(0).to_string())
SC = ["delta_R_score_alpha", "delta_R_score_beta", "delta_R_score_gamma"]
sets = {
  "drafts5 (C4 r1,r2 + IC-a + IC-c + C5)": ["C4|IC-b|r1", "C4|IC-b|r2", "C4|IC-a|r1", "C4|IC-c|r1", "C5|IC-b|r1"],
  "same-prompt2+rev (C4 r1,r2,C4R r1,r2)": ["C4|IC-b|r1", "C4|IC-b|r2", "C4R|IC-b|r1", "C4R|IC-b|r2"],
  "all7 reference-free": ["C4|IC-b|r1", "C4|IC-b|r2", "C4|IC-a|r1", "C4|IC-c|r1", "C5|IC-b|r1", "C4R|IC-b|r1", "C4R|IC-b|r2"],
}
for m, gm in sub.groupby("model_key"):
    print(f"\n===== {m}")
    piv = {}
    for k, g in gm.groupby("key"):
        g = g.sort_values("row_id")
        piv[k] = dict(U=np.vstack([parse(s) for s in g["strict_itt_action_json"]]), S=g[SC].values, rows=g["row_id"].values)
    base = piv["C4|IC-b|r1"]["S"]; c4r = piv["C4R|IC-b|r1"]["S"]; c6 = piv["C6-E|IC-b|r1"]["S"]
    print("mean C4 r1      :", np.round(base.mean(0), 3))
    print("mean C4R r1     :", np.round(c4r.mean(0), 3))
    print("mean C6-E r1    :", np.round(c6.mean(0), 3))
    for name, keys in sets.items():
        U = np.stack([piv[k]["U"] for k in keys], axis=1)      # firm x K x 8
        S = np.stack([piv[k]["S"] for k in keys], axis=1)      # firm x K x 3
        K = U.shape[1]
        D = np.abs(U[:, :, None, :] - U[:, None, :, :]).sum(-1)  # firm x K x K
        tot = D.sum(-1)
        # medoid; ties -> first in list (the main draft first)
        med = tot.argmin(axis=1)
        Smed = S[np.arange(len(S)), med]
        Savg = S.mean(axis=1)
        # modal exact-match cluster size
        nun = np.array([len({tuple(np.round(u, 9)) for u in U[i]}) for i in range(len(U))])
        print(f"-- {name}: K={K}, mean distinct actions per firm {nun.mean():.2f}, medoid==first {np.mean(med==0):.3f}")
        print("   medoid mean          :", np.round(Smed.mean(0), 3))
        print("   set average mean     :", np.round(Savg.mean(0), 3))
        for j, lab in enumerate(["alpha", "beta", "gamma"]):
            print(f"   [{lab}] medoid - C4 r1 {fmt(Smed[:, j]-base[:, j])} | medoid - set avg {fmt(Smed[:, j]-Savg[:, j])} | C6-E r1 - medoid {fmt(c6[:, j]-Smed[:, j])}")

"""Build a per-output table: unit actions, overlap with the RL reference action, scores. Usage: e31_prep.py OBS CON OUT"""
import sys, json
import numpy as np, pandas as pd
df = pd.read_parquet(sys.argv[1]); con = json.load(open(sys.argv[2]))
dims = con["managerial_intensity_dimensions"]
scale = np.array([con["normalization_scale_by_dimension"][d] for d in dims])
cand = {k: np.abs(np.array([v["action__" + d] for d in dims]) / scale) for k, v in con["fixed_candidates"].items()}
sub = df[(df.analysis_arm == "HIGH_STRICT_ITT") & (df["mode"] == "free8") & (df.budget == "B1")].copy()
ref = sub[sub.policy_condition == "C6-E"].groupby("row_id")["reference_candidate_id"].first()
U = np.abs(np.vstack([[float(json.loads(s).get(k, 0.0)) for k in dims] for s in sub["strict_itt_action_json"]]) / scale)
sub = sub.reset_index(drop=True)
for j, d in enumerate(dims): sub["u_" + d] = U[:, j]
sub["T"] = U.sum(1)
sub["rl"] = sub["row_id"].map(ref)
R = np.vstack([cand[r] for r in sub["rl"]])
sub["overlap"] = np.minimum(U, R).sum(1)
sub["exact_rl"] = np.abs(U - R).max(1) < 1e-6
keep = ["row_id", "model_key", "policy_condition", "info", "replicate", "strict_usable", "rl", "T", "overlap", "exact_rl",
        "delta_R_score_alpha", "delta_R_score_beta", "delta_R_score_gamma"] + ["u_" + d for d in dims]
sub[keep].to_parquet(sys.argv[3], index=False)
print(sub.shape, sub.strict_usable.mean())
print(ref.value_counts().to_string())
us = sub[sub.strict_usable]
print("mean units by axis:"); print(us[["u_" + d for d in dims] + ["T", "overlap"]].mean().round(3).to_string())
# pure patterns
tol = 0.02
others = lambda d: us["T"] - us["u_" + d]
for d in dims:
    pure = (us["u_" + d] > 0.9) & (others(d) < tol)
    print(f"pure {d:32s} outputs {int(pure.sum()):6d} firms {us[pure].row_id.nunique():4d}")
print("exact RL outputs", int(us.exact_rl.sum()), "firms", us[us.exact_rl].row_id.nunique())
print(us.groupby("rl").exact_rl.agg(["sum", lambda s: us.loc[s.index][s].row_id.nunique()]).to_string())

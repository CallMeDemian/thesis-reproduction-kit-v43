"""Run-to-run and cross-model agreement of final actions, by condition.
Usage: python3 -I consistency.py obs.parquet action_contract.json
"""
import sys, json
import numpy as np
import pandas as pd

obs_path, contract_path = sys.argv[1], sys.argv[2]
df = pd.read_parquet(obs_path)
con = json.load(open(contract_path))
dims = con["managerial_intensity_dimensions"]
scale = np.array([con["normalization_scale_by_dimension"][d] for d in dims])
cands = {k: np.array([v["action__" + d] for d in dims]) / scale for k, v in con["fixed_candidates"].items()}

def parse(js):
    d = json.loads(js)
    return np.array([float(d.get(k, 0.0)) for k in dims]) / scale

rng = np.random.default_rng(20261118)

def boot_ci(x, B=2000):
    x = np.asarray(x, float)
    n = len(x)
    idx = rng.integers(0, n, size=(B, n))
    m = x[idx].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])

def metrics(U1, U2):
    l1 = np.abs(U1 - U2).sum(axis=1)
    exact = (np.abs(U1 - U2).max(axis=1) < 1e-9)
    near = l1 <= 0.25
    top1 = np.abs(U1).argmax(axis=1); top2 = np.abs(U2).argmax(axis=1)
    z1 = np.abs(U1).sum(axis=1) < 1e-12; z2 = np.abs(U2).sum(axis=1) < 1e-12
    top = (top1 == top2) & ~(z1 ^ z2)
    supp = ((np.abs(U1) > 1e-12) == (np.abs(U2) > 1e-12)).all(axis=1)
    return dict(exact=exact, near=near, top=top, supp=supp, l1=l1)

out = {}
for arm in ["HIGH_STRICT_ITT", "BASELINE_STRICT_ITT"]:
    for budget in ["B1", "BINF"]:
        sub = df[(df.analysis_arm == arm) & (df["mode"] == "free8") & (df["info"] == "IC-b") & (df.budget == budget)]
        print(f"\n######## {arm} / free8 / IC-b / {budget}")
        acts = {}
        for (m, pc, rep), g in sub.groupby(["model_key", "policy_condition", "replicate"]):
            g = g.sort_values("row_id")
            U = np.vstack([parse(s) for s in g["strict_itt_action_json"]])
            acts[(m, pc, rep)] = dict(U=U, rows=g["row_id"].values, usable=g["strict_usable"].values,
                                      a=g["delta_R_score_alpha"].values, b=g["delta_R_score_beta"].values, c=g["delta_R_score_gamma"].values,
                                      ref=g["reference_candidate_id"].values)
        # run-to-run
        print("-- run-to-run agreement (replicate 1 vs 2), same model")
        print(f"{'model':28s} {'cond':6s} {'usable1':>7s} {'usable2':>7s} {'exact':>7s} {'near':>7s} {'top':>7s} {'supp':>7s} {'L1mean':>7s} {'L1med':>7s} {'|dA|':>7s} {'sameA':>7s} {'rA':>6s} {'rB':>6s} {'rG':>6s}")
        store = {}
        for m in sorted({k[0] for k in acts}):
            for pc in ["C4", "C4R", "C6-E"]:
                if (m, pc, 1) in acts and (m, pc, 2) in acts:
                    A, B = acts[(m, pc, 1)], acts[(m, pc, 2)]
                    assert (A["rows"] == B["rows"]).all()
                    mt = metrics(A["U"], B["U"])
                    store[(m, pc)] = mt
                    dA = np.abs(A["a"] - B["a"])
                    def sp(x, y):
                        return pd.Series(x).corr(pd.Series(y), method="spearman")
                    print(f"{m:28s} {pc:6s} {A['usable'].mean():7.3f} {B['usable'].mean():7.3f} {mt['exact'].mean():7.3f} {mt['near'].mean():7.3f} {mt['top'].mean():7.3f} {mt['supp'].mean():7.3f} {mt['l1'].mean():7.3f} {np.median(mt['l1']):7.3f} {dA.mean():7.3f} {(dA<1e-9).mean():7.3f} {sp(A['a'],B['a']):6.2f} {sp(A['b'],B['b']):6.2f} {sp(A['c'],B['c']):6.2f}")
            # differences C6-E vs C4R, C6-E vs C4
            for base in ["C4R", "C4"]:
                if (m, "C6-E") in store and (m, base) in store:
                    for k in ["exact", "near", "top", "supp", "l1"]:
                        d = store[(m, "C6-E")][k].astype(float) - store[(m, base)][k].astype(float)
                        lo, hi = boot_ci(d)
                        print(f"   diff C6-E - {base:3s} [{k:5s}] {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        # adoption of RL reference in C6-E and agreement conditional
        print("-- C6-E: exact adoption of RL reference action, by replicate; and run agreement by adoption pattern")
        for m in sorted({k[0] for k in acts}):
            if (m, "C6-E", 1) in acts and (m, "C6-E", 2) in acts:
                A, B = acts[(m, "C6-E", 1)], acts[(m, "C6-E", 2)]
                refU1 = np.vstack([cands[r] for r in A["ref"]]); refU2 = np.vstack([cands[r] for r in B["ref"]])
                assert (A["ref"] == B["ref"]).all()
                ad1 = np.abs(A["U"] - refU1).max(axis=1) < 1e-6; ad2 = np.abs(B["U"] - refU2).max(axis=1) < 1e-6
                ex = store[(m, "C6-E")]["exact"]
                print(f"{m:28s} adopt r1 {ad1.mean():.3f} r2 {ad2.mean():.3f} both {(ad1&ad2).mean():.3f} one {(ad1^ad2).mean():.3f} none {(~ad1&~ad2).mean():.3f} | exact-agree among none-adopt {ex[~ad1&~ad2].mean():.3f} (n={int((~ad1&~ad2).sum())})")
                # distance to RL ref
                d1 = np.abs(A["U"] - refU1).sum(axis=1); d2 = np.abs(B["U"] - refU2).sum(axis=1)
                for pc in ["C4", "C4R"]:
                    if (m, pc, 1) in acts:
                        dd = np.abs(acts[(m, pc, 1)]["U"] - refU1).sum(axis=1)
                        print(f"     mean L1 distance to RL action: {pc} r1 {dd.mean():.3f}  | C6-E r1 {d1.mean():.3f} r2 {d2.mean():.3f}")
        # cross-model agreement
        print("-- cross-model agreement (GPT vs Gemini), replicate 1")
        ms = sorted({k[0] for k in acts})
        cm = {}
        for pc in ["C4", "C4R", "C6-E", "C6-EX", "C5"]:
            if (ms[0], pc, 1) in acts and (ms[1], pc, 1) in acts:
                A, B = acts[(ms[0], pc, 1)], acts[(ms[1], pc, 1)]
                mt = metrics(A["U"], B["U"]); cm[pc] = mt
                print(f"{pc:6s} exact {mt['exact'].mean():.3f} near {mt['near'].mean():.3f} top {mt['top'].mean():.3f} supp {mt['supp'].mean():.3f} L1mean {mt['l1'].mean():.3f} L1med {np.median(mt['l1']):.3f}")
        for base in ["C4R", "C4", "C6-EX"]:
            if "C6-E" in cm and base in cm:
                for k in ["exact", "near", "top", "l1"]:
                    d = cm["C6-E"][k].astype(float) - cm[base][k].astype(float)
                    lo, hi = boot_ci(d)
                    print(f"   diff C6-E - {base:5s} [{k:5s}] {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

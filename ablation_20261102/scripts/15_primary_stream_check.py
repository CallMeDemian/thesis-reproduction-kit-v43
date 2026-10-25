"""Primary 96 rows: CI, p and Holm recomputed from the firm-level observations and compared with the frozen streams.

T1 takes CI, p and Holm from the frozen production streams (V13_*_PRIMARY_FULLSTREAM.csv); the kit itself recomputes
only N and the means. This check rebuilds the rest with the kit's Stage 9 code
(credit_recourse.eval.final_stage9_llm_rl_comparison.revision_metrics): one PCG64 stream per analysis arm
(seed 20260912, 10,000 draws), groups visited in the production order. The production stream also ran over the PP
population and over the other modes, information conditions and replicates of the arm, so those groups are rebuilt
too (PP = strict-usable responses) and consumed in the same order; only the 96 primary rows are compared.

Output: results/_primary_stream_check.json. Exit status 1 on any mismatch.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

import common as c
from credit_recourse.eval.final_stage9_llm_rl_comparison import revision_metrics as rm

B, SEED = 10_000, 20_260_912
ARMS = ("BASELINE_STRICT_ITT", "HIGH_STRICT_ITT")
GROUP_KEYS = ("contrast", "treatment_condition", "control_condition", "model_key", "mode", "info",
              "information_condition", "budget", "replicate", "analysis_population")
COMPARED = ("N", "estimate", "ci_low", "ci_high", "raw_p", "neg", "holm_p")


def stream(firm: pd.DataFrame) -> pd.DataFrame:
    """Same traversal and draws as rm._summarize_effects, keeping the draws for the two-sided bootstrap p."""
    keys = [k for k in GROUP_KEYS if k in firm]
    rng = np.random.Generator(np.random.PCG64(SEED))
    rows = []
    for values, group in firm.groupby(keys, sort=True, dropna=False):
        base = dict(zip(keys, values))
        for oracle in c.ORACLES:
            v = pd.to_numeric(group[f"effect_{oracle}"], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(v):
                continue
            draws = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            neg, zero, pos = int((draws < 0).sum()), int((draws == 0).sum()), int((draws > 0).sum())
            rows.append({**base, "oracle": oracle, "N": len(v), "estimate": float(v.mean()), "ci_low": float(low),
                         "ci_high": float(high), "neg": neg,
                         "raw_p": min(1.0, 2.0 * (min(neg + zero, pos + zero) + 1) / (B + 1))})
    return pd.DataFrame(rows)


def main() -> dict:
    obs = c.load_obs()
    rebuilt, production_ci_identical = [], True
    for arm in ARMS:
        a = obs[obs.analysis_arm == arm]
        pp = a[a.strict_usable.astype(bool)].assign(analysis_population="pp")
        firm = rm._contrast_pairs(pd.concat([a, pp], ignore_index=True).rename(columns={"policy_condition": "policy"}))
        mine = stream(firm)
        prod = rm._summarize_effects(firm, bootstrap_replicates=B, bootstrap_seed=SEED).rename(columns={"backend": "oracle"})
        keys = [k for k in GROUP_KEYS if k in mine] + ["oracle"]
        both = mine.merge(prod[keys + ["bootstrap_ci95_low", "bootstrap_ci95_high"]], on=keys, validate="one_to_one")
        production_ci_identical &= bool(len(both) == len(mine) == len(prod)
                                        and (both.ci_low == both.bootstrap_ci95_low).all()
                                        and (both.ci_high == both.bootstrap_ci95_high).all())
        prim = mine[(mine["mode"] == "free8") & (mine.information_condition == "IC-b") & (mine.replicate == 1)
                    & (mine.analysis_population == "itt")].copy()
        prim["arm"] = arm
        rebuilt.append(prim)
    r = pd.concat(rebuilt, ignore_index=True).rename(columns={"model_key": "model"})
    r["holm_p"] = np.nan
    for _, idx in r.groupby(["arm", "model", "oracle"]).groups.items():
        r.loc[idx, "holm_p"] = c.holm(r.loc[idx, "raw_p"].to_numpy())
    frozen_b, frozen_h = pd.read_csv(c.PRIM_BASE), pd.read_csv(c.PRIM_HIGH)
    if "arm" not in frozen_h:
        frozen_h.insert(0, "arm", "HIGH_STRICT_ITT")
    frozen = pd.concat([frozen_b, frozen_h], ignore_index=True)
    m = r.merge(frozen, on=["arm", "model", "budget", "contrast", "oracle"], suffixes=("", "_frozen"), validate="one_to_one")
    columns = {}
    for col in COMPARED:
        stored = m[f"{col}_frozen"].astype(float)
        have = stored.notna()  # the High stream file carries no `pos`, Baseline and High both carry `neg`
        diff = (m.loc[have, col].astype(float) - stored[have]).abs()
        match = np.isclose(m.loc[have, col].astype(float), stored[have], atol=1e-12, rtol=1e-12)
        columns[col] = {"compared": int(have.sum()), "match": int(match.sum()), "max_abs_diff": float(diff.max()) if len(diff) else None}
    ok = len(m) == 96 and production_ci_identical and all(v["match"] == v["compared"] for v in columns.values())
    out = {"status": "PASS" if ok else "FAIL", "rows": int(len(m)), "columns": columns,
           "production_summarize_effects_ci_identical": production_ci_identical,
           "method": "kit Stage 9 pairing and group order; ITT + PP rows of each arm; one PCG64 stream per arm",
           "seed": SEED, "draws": B, "holm_family": "arm x model x oracle (8 rows)",
           "sources": {p.name: c.sha256_file(p) for p in (c.OBS_PATH, c.PRIM_BASE, c.PRIM_HIGH)}}
    c.RESULTS.mkdir(parents=True, exist_ok=True)
    (c.RESULTS / "_primary_stream_check.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\r\n")
    return out


if __name__ == "__main__":
    res = main()
    print(res["status"], {k: f"{v['match']}/{v['compared']}" for k, v in res["columns"].items()})
    sys.exit(0 if res["status"] == "PASS" else 1)

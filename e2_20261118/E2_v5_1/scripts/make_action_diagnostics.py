from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


E2 = Path(__file__).resolve().parents[1]
TASK = E2.parent
KIT = TASK / "A_reproduction/E1_score_gate_replay/runroot_zip/THESIS_REPRO_KIT_v2.1.1_FINAL"
ANALYSIS = TASK / "A_reproduction/R2_R8_replay_workspace/analysis_20261117"
OUT = E2 / "results"
AX = ["growth_capex_reduction_pct", "deleveraging_total_debt_pct", "refinancing_short_debt_pct",
      "inv_turnover_chg", "ar_turnover_chg", "ap_turnover_chg", "cogs_ratio_chg", "sga_ratio_chg"]
SEED = 20261118
B = 10000

os.environ["KIT_ROOT"] = str(KIT).replace("\\", "/")
os.environ["HZ_SRC"] = str(KIT / "src").replace("\\", "/")
os.environ["ABL_E1"] = str(TASK / "source_inputs/github_release_ablation_20261102_readonly/contents/ablation_20261102/E1/observations/e1_observations.parquet").replace("\\", "/")
if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))
import prep2 as P  # noqa: E402


def bootstrap_mean_ci(values: np.ndarray, seed_offset: int = 0) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(SEED + seed_offset)
    bs = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def norm_vec(action: dict[str, float]) -> np.ndarray:
    return np.asarray([float(action.get(k, 0.0)) / float(P.SC[k]) for k in AX], dtype=float)


def main() -> None:
    parsed = pd.read_csv(OUT / "main_policy_responses.csv", dtype={"firm_key": str})
    parsed = parsed[parsed.condition.isin(["R", "T", "RT"])].copy()
    parsed["action"] = parsed.strict_itt_action_json.map(lambda s: json.loads(s) if isinstance(s, str) and s else {k: 0.0 for k in AX})
    action_lookup = {}
    for row in parsed.itertuples(index=False):
        action_lookup[(row.condition, row.model_key, int(row.replicate), int(row.row_id))] = row.action

    rl = P.a3.sort_values("row_id").reset_index(drop=True)
    if len(rl) != 575 or rl.row_id.astype(int).tolist() != list(range(575)):
        raise RuntimeError("Frozen C3-E action map must contain ordered 575 firm rows")
    c3e_ids = rl.action_id.astype(str).tolist()
    p2 = {k: (-float(P.SC[k]) if k == "cogs_ratio_chg" else 0.0) for k in AX}
    p3_actions = [p2 if cid == "OE" else dict(P.CAND[cid]) for cid in c3e_ids]
    rl_actions = [dict(P.CAND[cid]) for cid in c3e_ids]

    adoption_rows = []
    for condition in ["R", "RT"]:
        for model in sorted(parsed.model_key.unique()):
            vectors = []
            ref_vectors = []
            for replicate in [1, 2]:
                for row_id in range(575):
                    action = action_lookup[(condition, model, replicate, row_id)]
                    vectors.append(norm_vec(action))
                    ref_vectors.append(norm_vec(rl_actions[row_id]))
            v = np.vstack(vectors)
            r = np.vstack(ref_vectors)
            distance = np.abs(v - r)
            exact = (distance.max(axis=1) <= 1e-8)
            adoption_rows.append({
                "condition": condition, "model_key": model, "n": len(v),
                "exact_match_to_C3E_reference_rate": float(exact.mean()),
                "any_axis_modified_rate": float((~exact).mean()),
                "mean_changed_axes": float((distance > 1e-8).sum(axis=1).mean()),
                "mean_normalized_L1_distance_from_C3E": float(distance.sum(axis=1).mean()),
                "median_normalized_L1_distance_from_C3E": float(np.median(distance.sum(axis=1))),
            })
    pd.DataFrame(adoption_rows).to_csv(OUT / "main_rl_reference_adoption.csv", index=False, encoding="utf-8-sig")

    p3_rows = []
    for condition in ["T", "RT"]:
        for model in sorted(parsed.model_key.unique()):
            vectors = []
            ref_vectors = []
            for replicate in [1, 2]:
                for row_id in range(575):
                    action = action_lookup[(condition, model, replicate, row_id)]
                    vectors.append(norm_vec(action))
                    ref_vectors.append(norm_vec(p3_actions[row_id]))
            v = np.vstack(vectors)
            r = np.vstack(ref_vectors)
            distance = np.abs(v - r)
            exact = (distance.max(axis=1) <= 1e-8)
            p3_rows.append({
                "condition": condition, "model_key": model, "n": len(v),
                "exact_match_to_P3_rate": float(exact.mean()),
                "mean_normalized_L1_distance_from_P3": float(distance.sum(axis=1).mean()),
                "median_normalized_L1_distance_from_P3": float(np.median(distance.sum(axis=1))),
            })
    pd.DataFrame(p3_rows).to_csv(OUT / "main_p3_action_similarity.csv", index=False, encoding="utf-8-sig")

    all_scores = pd.read_csv(OUT / "main_policy_scores.csv", dtype={"firm_key": str})
    all_scores = all_scores[(all_scores.source == "main") & (all_scores.scenario == "S0")].copy()
    scores = all_scores[all_scores.policy.isin(["R", "RT"])].copy()
    pivot = scores.pivot_table(index=["row_id", "firm_key", "model_key", "replicate"], columns="policy",
                               values=["score_delta_alpha", "score_delta_beta", "score_delta_gamma"], aggfunc="first")
    subgroup_rows = []
    for metric in ["alpha", "beta", "gamma"]:
        pair = (pivot[("score_delta_" + metric, "RT")] - pivot[("score_delta_" + metric, "R")]).rename("difference").reset_index()
        pair["C3E_action_group"] = pair.row_id.map(lambda i: "OE" if c3e_ids[int(i)] == "OE" else "non_OE")
        firm_means = pair.groupby(["firm_key", "C3E_action_group"], as_index=False).difference.mean()
        for group_name, g in firm_means.groupby("C3E_action_group"):
            mean, lo, hi = bootstrap_mean_ci(g.difference.to_numpy(float), seed_offset=len(subgroup_rows))
            subgroup_rows.append({"contrast": "RT_minus_R", "evaluator": metric, "C3E_action_group": group_name,
                                  "n_firms": len(g), "mean_difference": mean, "ci95_low": lo, "ci95_high": hi,
                                  "interpretation": "exploratory subgroup; firm is bootstrap unit"})
    pd.DataFrame(subgroup_rows).to_csv(OUT / "main_oe_reference_subgroup.csv", index=False, encoding="utf-8-sig")

    adverse_rows = []
    direct = all_scores
    for (policy, model, metric), g in direct.melt(
            id_vars=["policy", "model_key"], value_vars=["score_delta_alpha", "score_delta_beta", "score_delta_gamma"],
            var_name="score_metric", value_name="delta").groupby(["policy", "model_key", "score_metric"]):
        adverse_rows.append({"condition": policy, "model_key": model, "evaluator": metric.removeprefix("score_delta_"),
                             "rows": len(g), "below_A0_rate": float((g.delta < 0).mean())})
    pd.DataFrame(adverse_rows).to_csv(OUT / "main_policy_below_A0_share.csv", index=False, encoding="utf-8-sig")
    print("Created RL-reference adoption, P3 similarity, OE subgroup, and below-A0 diagnostics")


if __name__ == "__main__":
    main()

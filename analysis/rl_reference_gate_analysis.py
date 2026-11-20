#!/usr/bin/env python3
"""Reproducible RL-internal-signal gate analysis for the V4.3 frozen release.

This program never calls an LLM or retrains RL.  It consumes the 28 frozen
actor/critic diagnostic files and a materialized Stage-8 table, then evaluates
reference gating with firm-level nested cross-validation.  Q values are only
used after within-(member, firm) standardisation, because reward scales differ
across the four ensemble configurations.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CONFIGS = ("B27", "M2_S0935", "DT06", "T15_REWARD_S088")
SEEDS = ("2", "11", "12", "13", "14", "15", "16")
POLICIES = ("C4R", "C6-E")
ORACLES = ("alpha", "beta", "gamma")
PRIMARY_SEED = 20261117


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-14, 1.0)
    return -(p * np.log(p)).sum(axis=-1) / np.log(p.shape[-1])


def firm_key(firm_id: object, year: object) -> str:
    raw = str(firm_id).strip()
    if raw.endswith(".0") and raw[:-2].isdigit():
        raw = raw[:-2]
    if raw.isdigit():
        raw = raw.zfill(6)
    return f"{raw}::{int(float(year))}"


def require_parquet(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        magic = f.read(28)
    if magic.startswith(b"version https://git-lfs"):
        raise RuntimeError(f"Git LFS pointer, not data: {path}")
    if magic[:4] != b"PAR1":
        raise RuntimeError(f"Not a parquet file: {path}")


def load_rl_features(stage5: Path) -> tuple[pd.DataFrame, dict]:
    member_root = stage5 / "C3E_E2_7SEED_BALANCED_DFEBAFA6" / "members"
    paths = [member_root / c / f"seed_{s}" / "actor_critic_outputs.parquet"
             for c, s in itertools.product(CONFIGS, SEEDS)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise RuntimeError(f"Expected all 28 frozen actor outputs; missing: {missing}")

    member_probabilities, member_qz, member_actions = [], [], []
    keys: list[str] | None = None
    action_order: list[str] | None = None
    member_audit: list[dict] = []
    any_value_columns = False
    for path in paths:
        require_parquet(path)
        frame = pd.read_parquet(path)
        required = {"firm_id", "fiscal_year", "selected_candidate_id"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"Missing {sorted(required - set(frame.columns))} in {path}")
        pcols = [c for c in frame.columns if c.startswith("probability__")]
        qcols = [c for c in frame.columns if c.startswith("Q__")]
        actions = [c.removeprefix("probability__") for c in pcols]
        if len(actions) != 9 or set(actions) != {c.removeprefix("Q__") for c in qcols}:
            raise RuntimeError(f"Expected aligned 9-action probability/Q columns in {path}")
        if action_order is None:
            action_order = actions
        elif action_order != actions:
            raise RuntimeError(f"Action-column order drift in {path}")
        frame = frame.copy()
        frame["firm_key"] = [firm_key(a, b) for a, b in zip(frame.firm_id, frame.fiscal_year)]
        if frame.firm_key.duplicated().any():
            raise RuntimeError(f"Duplicate firm key in {path}")
        frame = frame.set_index("firm_key").sort_index()
        if keys is None:
            keys = frame.index.tolist()
        elif frame.index.tolist() != keys:
            raise RuntimeError(f"Firm cohort/order drift in {path}")
        probs = frame[pcols].to_numpy(float)
        q = frame[[f"Q__{a}" for a in action_order]].to_numpy(float)
        if not np.isfinite(probs).all() or not np.isfinite(q).all():
            raise RuntimeError(f"Non-finite RL values in {path}")
        if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError(f"Actor probabilities do not sum to one in {path}")
        qsd = q.std(axis=1, keepdims=True)
        qz = (q - q.mean(axis=1, keepdims=True)) / np.maximum(qsd, 1e-6)
        member_probabilities.append(probs)
        member_qz.append(qz)
        member_actions.append(frame.selected_candidate_id.astype(str).to_numpy())
        any_value_columns |= any(c.lower() in {"v", "value", "v_estimate", "v_pred"} for c in frame.columns)
        member_audit.append({"configuration": path.parent.parent.name, "seed": path.parent.name,
                             "path": str(path), "sha256": sha256(path), "rows": len(frame),
                             "has_value_column": any(c.lower() in {"v", "value", "v_estimate", "v_pred"} for c in frame.columns)})

    if keys is None or action_order is None or len(keys) != 575:
        raise RuntimeError(f"Expected 575 RL firms, found {0 if keys is None else len(keys)}")
    p = np.stack(member_probabilities)  # member, firm, action
    qz = np.stack(member_qz)
    mean_p = p.mean(axis=0)  # equal because every configuration has seven seeds
    chosen = mean_p.argmax(axis=1)
    rows = np.arange(len(keys))
    votes = p.argmax(axis=2)
    vote_p = np.stack([(votes == a).mean(axis=0) for a in range(p.shape[2])], axis=1)
    member_h = entropy(p).mean(axis=0)
    predictive_h = entropy(mean_p)
    sorted_p = np.sort(mean_p, axis=1)
    q_selected = qz[:, rows, chosen]
    q_sorted = np.sort(qz, axis=2)
    action_ids = np.asarray(action_order)
    selected_labels = action_ids[chosen]
    actor_selected = np.stack(member_actions)
    data = pd.DataFrame({
        "firm_key": keys,
        "actor_mean_entropy": member_h,
        "ensemble_predictive_entropy": predictive_h,
        "ensemble_disagreement_js": np.maximum(predictive_h - member_h, 0.0),
        "vote_entropy": entropy(vote_p),
        "chosen_vote_fraction": vote_p[rows, chosen],
        "top1_probability": sorted_p[:, -1],
        "top1_top2_probability_margin": sorted_p[:, -1] - sorted_p[:, -2],
        "q_selected_relative": q_selected.mean(axis=0),
        "q_selected_vs_best_relative": (q_selected - qz.max(axis=2)).mean(axis=0),
        "q_top1_top2_relative_margin": (q_sorted[:, :, -1] - q_sorted[:, :, -2]).mean(axis=0),
        "q_argmax_actor_agreement": (qz.argmax(axis=2) == chosen[None, :]).mean(axis=0),
        "actor_selected_equals_ensemble_choice": (actor_selected == selected_labels[None, :]).mean(axis=0),
        "rl_chosen_action": selected_labels,
    })
    audit = {
        "member_count": len(paths), "firm_count": len(keys), "action_order": action_order,
        "value_available": False, "value_column_seen": any_value_columns,
        "value_note": "No retained V(s) is used. A checkpoint-compatible extraction is required before Q-V features can be analyzed.",
        "members": member_audit,
    }
    return data, audit


def load_stage8(stage8: Path) -> pd.DataFrame:
    path = stage8 / "llm_stage8_multi_oracle_scores_all_populations.parquet"
    require_parquet(path)
    d = pd.read_parquet(path)
    required = {"row_id", "firm_key", "model_key", "mode", "information_condition", "budget", "policy", "replicate", "analysis_population"}
    required |= {f"delta_R_score_{x}" for x in ORACLES}
    missing = required - set(d.columns)
    if missing:
        raise RuntimeError(f"Stage8 schema missing: {sorted(missing)}")
    d = d[(d["analysis_population"].astype(str) == "itt") &
          (d["mode"].astype(str) == "free8") &
          (d["information_condition"].astype(str) == "IC-b") &
          (d["budget"].astype(str) == "B1") &
          (d["replicate"].astype(str) == "1") &
          (d["policy"].astype(str).isin(POLICIES))].copy()
    if d.empty:
        raise RuntimeError("No HIGH/ITT/FREE8/IC-b/B1/Run1 C4R+C6-E rows found in Stage8")
    if d.duplicated(["model_key", "firm_key", "policy"]).any():
        raise RuntimeError("Duplicate model/firm/policy Stage8 rows after primary filtering")
    return d


def mean_ci(x: np.ndarray, seed: int = PRIMARY_SEED, draws: int = 5000) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for start in range(0, draws, 250):
        n = min(250, draws - start)
        boot[start:start+n] = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


def gate_value(y: np.ndarray, decisions: np.ndarray) -> float:
    return float(np.where(decisions, y, 0.0).mean())


def inner_threshold(x: np.ndarray, y: np.ndarray, *, seed: int) -> tuple[float, bool]:
    quantiles = np.unique(np.quantile(x, np.linspace(.05, .95, 19)))
    cv = KFold(4, shuffle=True, random_state=seed)
    candidates: list[tuple[float, bool, float]] = []
    for threshold in quantiles:
        for high_is_reference in (True, False):
            scores = []
            for _, va in cv.split(x):
                decision = x[va] >= threshold if high_is_reference else x[va] <= threshold
                scores.append(gate_value(y[va], decision) - float(y[va].mean()))
            candidates.append((float(threshold), high_is_reference, float(np.mean(scores))))
    threshold, direction, _ = max(candidates, key=lambda item: item[2])
    return threshold, direction


def nested_predictions(x: np.ndarray, y: np.ndarray, method: str, seed: int) -> np.ndarray:
    """OOF prediction/decision score; every tuning decision remains within train folds."""
    outer = KFold(5, shuffle=True, random_state=seed)
    output = np.zeros(len(y), dtype=float)
    for fold, (train, test) in enumerate(outer.split(x)):
        xt, yt, xe = x[train], y[train], x[test]
        if method.startswith("threshold:"):
            threshold, high = inner_threshold(xt[:, 0], yt, seed=seed + fold + 1)
            output[test] = (xe[:, 0] - threshold) * (1.0 if high else -1.0)
            continue
        if method == "ridge":
            candidates = (0.1, 1.0, 10.0, 100.0)
            inner = KFold(4, shuffle=True, random_state=seed + fold + 100)
            scored = []
            for alpha in candidates:
                val = []
                for it, iv in inner.split(xt):
                    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                    model.fit(xt[it], yt[it])
                    val.append(gate_value(yt[iv], model.predict(xt[iv]) > 0) - float(yt[iv].mean()))
                scored.append((float(np.mean(val)), alpha))
            alpha = max(scored)[1]
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(xt, yt)
            output[test] = model.predict(xe)
            continue
        if method == "logistic":
            label = yt > 0
            if label.min() == label.max():
                output[test] = float(label[0]) - .5
                continue
            candidates = (0.01, 0.1, 1.0, 10.0)
            inner = KFold(4, shuffle=True, random_state=seed + fold + 200)
            scored = []
            for c in candidates:
                val = []
                for it, iv in inner.split(xt):
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=seed))
                    m.fit(xt[it], label[it])
                    score = m.predict_proba(xt[iv])[:, 1] - .5
                    val.append(gate_value(yt[iv], score > 0) - float(yt[iv].mean()))
                scored.append((float(np.mean(val)), c))
            c = max(scored)[1]
            model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=seed)).fit(xt, label)
            output[test] = model.predict_proba(xe)[:, 1] - .5
            continue
        raise ValueError(method)
    return output


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, values[idx] * (len(values) - rank))
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def plot_outputs(out: Path, comparison: pd.DataFrame, correlations: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    primary = comparison[comparison.oracle.eq("alpha")].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    show = correlations[(correlations.oracle == "alpha") & (correlations.model_key == "openai_gpt54mini")]
    ax.barh(show.feature, show.spearman_r)
    ax.axvline(0, color="black", linewidth=.8); ax.set_xlabel("Spearman r with C6-E − C4R")
    fig.tight_layout(); fig.savefig(out / "figure_1_signal_gain_relation.png", dpi=180); plt.close(fig)
    ridge = primary[primary.method.eq("ridge")]
    fig, ax = plt.subplots(figsize=(10, 5))
    for model, group in ridge.groupby("model_key"):
        ax.errorbar(group.feature_group, group.gate_gain_vs_always, yerr=[group.gate_gain_vs_always-group.ci_low, group.ci_high-group.gate_gain_vs_always], fmt="o", capsize=4, label=model)
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("OOF Adaptive − Always RL")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "figure_2_gate_vs_always.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ablation = primary[(primary.method == "ridge") & primary.feature_group.isin(["actor", "q", "combined"])]
    for model, group in ablation.groupby("model_key"):
        ax.plot(group.feature_group, group.gate_gain_vs_always, marker="o", label=model)
    ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("OOF gate gain")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "figure_3_ablation.png", dpi=180); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-root", type=Path, required=True, help="Scratch runtime containing materialized Stage8 outputs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root, out = args.runtime_root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stage5 = root / "data/final_freeze/stage5_candidate_iql"
    stage8 = root / "data/final_freeze/stage8_llm_multi_oracle_eval"
    rl, audit = load_rl_features(stage5)
    scores = load_stage8(stage8)
    rl.to_csv(out / "RL_REFERENCE_RL_FEATURES.csv", index=False)

    actor = [c for c in rl if c not in {"firm_key", "rl_chosen_action"} and not c.startswith("q_")]
    q = [c for c in rl if c.startswith("q_")]
    groups = {"actor": actor, "q": q, "combined": actor + q}
    methods = {"threshold:entropy": ("actor", "ensemble_predictive_entropy"),
               "threshold:agreement": ("actor", "chosen_vote_fraction"),
               "threshold:q_margin": ("q", "q_top1_top2_relative_margin"),
               "ridge": None, "logistic": None}
    comparisons, oof_rows, correlations, quintiles = [], [], [], []
    for model, oracle in itertools.product(sorted(scores.model_key.astype(str).unique()), ORACLES):
        frame = scores[scores.model_key.astype(str).eq(model)].copy()
        wide = frame.pivot(index="firm_key", columns="policy", values=f"delta_R_score_{oracle}")
        if set(POLICIES) - set(wide):
            raise RuntimeError(f"Missing arms for {model}/{oracle}")
        merged = rl.set_index("firm_key").join(wide[list(POLICIES)], how="inner", validate="one_to_one")
        if len(merged) != 575:
            raise RuntimeError(f"RL/LLM firm join is {len(merged)}, expected 575: {model}/{oracle}")
        y = (merged["C6-E"] - merged["C4R"]).to_numpy(float)
        for feature in groups["combined"]:
            rho, p = spearmanr(merged[feature].to_numpy(float), y)
            correlations.append({"model_key": model, "oracle": oracle, "feature": feature,
                                 "spearman_r": rho, "spearman_p_unadjusted": p})
            try:
                bins = pd.qcut(merged[feature], 5, duplicates="drop")
                for bucket, values in pd.Series(y, index=merged.index).groupby(bins, observed=True):
                    quintiles.append({"model_key": model, "oracle": oracle, "feature": feature,
                                      "quintile": str(bucket), "n": len(values), "mean_increment": values.mean(),
                                      "positive_increment_share": (values > 0).mean()})
            except ValueError:
                pass
        for method, spec in methods.items():
            choices = [(spec[0], [spec[1]])] if spec else list(groups.items())
            for group, features in choices:
                pred = nested_predictions(merged[features].to_numpy(float), y, method, PRIMARY_SEED)
                use = pred > 0
                diff = np.where(use, y, 0.0) - y
                lo, hi, pvalue = mean_ci(diff)
                rho = spearmanr(pred, y).statistic
                comparisons.append({"model_key": model, "oracle": oracle, "method": method,
                                    "feature_group": group, "n": len(y), "always_rl_increment": y.mean(),
                                    "never_rl_increment": 0.0, "adaptive_increment": np.where(use, y, 0.0).mean(),
                                    "gate_gain_vs_always": diff.mean(), "ci_low": lo, "ci_high": hi,
                                    "bootstrap_p_raw": pvalue, "reference_share": use.mean(),
                                    "oof_prediction_gain_spearman": rho,
                                    "hindsight_best_of_two_increment": np.maximum(y, 0.0).mean(),
                                    "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "ties": int((diff == 0).sum())})
                oof_rows.append(pd.DataFrame({"firm_key": merged.index, "model_key": model, "oracle": oracle,
                                              "method": method, "feature_group": group, "C4R": merged["C4R"],
                                              "C6E": merged["C6-E"], "actual_increment": y, "oof_prediction": pred,
                                              "gate_use_C6E": use, "gate_minus_always": diff}))
    comparison = pd.DataFrame(comparisons)
    primary = comparison.oracle.eq("alpha")
    comparison.loc[primary, "holm_p_alpha_primary_family"] = holm(comparison.loc[primary, "bootstrap_p_raw"].tolist())
    corr = pd.DataFrame(correlations); quint = pd.DataFrame(quintiles); oof = pd.concat(oof_rows, ignore_index=True)
    comparison.to_csv(out / "RL_REFERENCE_GATE_COMPARISON.csv", index=False)
    oof.to_csv(out / "RL_REFERENCE_GATE_OOF.csv", index=False)
    corr.to_csv(out / "RL_REFERENCE_SIGNAL_ANALYSIS.csv", index=False)
    quint.to_csv(out / "RL_REFERENCE_SIGNAL_QUINTILES.csv", index=False)
    audit.update({"runtime_root": str(root), "stage8": str(stage8 / "llm_stage8_multi_oracle_scores_all_populations.parquet"),
                  "stage8_sha256": sha256(stage8 / "llm_stage8_multi_oracle_scores_all_populations.parquet"),
                  "join_check": "Every model/oracle primary analysis joined exactly 575 firms.",
                  "outcome": "g_i = delta_R_score(C6-E) - delta_R_score(C4R)"})
    (out / "RL_INTERNAL_SIGNAL_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# RL internal-signal adaptive-reference gate", "",
             "This is an exploratory, firm-level 5-fold nested-CV analysis. The gate uses only frozen RL features; C4R/C6-E scores are outcomes, never inputs.",
             "Q values are z-standardized within each member and firm before aggregation. No V(s) was retained, so no Q−V/Advantage claim is made.", "",
             "| Model | Oracle | Gate | Features | Always RL increment | Gate − always | 95% CI | Holm p (alpha) | Reference share |", "|---|---|---|---|---:|---:|---|---:|---:|"]
    for row in comparison.itertuples(index=False):
        holm_p = getattr(row, "holm_p_alpha_primary_family")
        holm_text = "—" if pd.isna(holm_p) else f"{holm_p:.4f}"
        lines.append(f"| {row.model_key} | {row.oracle} | {row.method} | {row.feature_group} | {row.always_rl_increment:+.4f} | {row.gate_gain_vs_always:+.4f} | [{row.ci_low:+.4f}, {row.ci_high:+.4f}] | {holm_text} | {row.reference_share:.1%} |")
    lines += ["", "Primary inference is the adaptive-minus-always paired, firm-resampled confidence interval. Beta and gamma are robustness analyses; alpha p-values are Holm-adjusted across all alpha gate comparisons.", "A positive raw association alone is not evidence of a deployable gate; it must improve OOF policy value over always providing C3-E."]
    (out / "RL_REFERENCE_GATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_outputs(out, comparison, corr)


if __name__ == "__main__":
    main()

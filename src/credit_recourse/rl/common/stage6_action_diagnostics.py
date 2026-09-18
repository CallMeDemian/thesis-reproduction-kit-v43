"""Action-neutral Stage6 diagnostics for Candidate-IQL.

The helpers in this module never privilege a named candidate.  Oracle scores
enter only after the actor has selected one action per firm and are used to
describe the selected action relative to the complete per-firm candidate
surface.  Tied optima are represented as sets and receive fractional mass in
aggregate tables so candidate ordering cannot manufacture a winner.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DIAGNOSTIC_SCHEMA_VERSION = "candidate_iql_stage6_action_diagnostics_v2"
DEFAULT_TOLERANCE = 1e-12


def _json_set(values: Sequence[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _entropy(shares: pd.Series) -> float:
    positive = pd.to_numeric(shares, errors="raise").astype(float)
    positive = positive.loc[positive > 0]
    return float(-(positive * np.log(positive)).sum()) if len(positive) else 0.0


def _distribution(values: Sequence[str]) -> dict[str, Any]:
    series = pd.Series(np.asarray(values, dtype=object), dtype=str)
    counts = series.value_counts(sort=True)
    shares = counts / float(len(series))
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "shares": {str(k): float(v) for k, v in shares.items()},
        "entropy": _entropy(shares),
        "max_value": str(counts.index[0]),
        "max_share": float(shares.iloc[0]),
        "used_count": int(len(counts)),
    }


def _validate_actions(values: Sequence[str], actions: Sequence[str], name: str, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=object).astype(str)
    if arr.shape != (n,):
        raise ValueError(f"{name} must contain exactly {n} actions; got shape={arr.shape}")
    unknown = sorted(set(arr) - set(actions))
    if unknown:
        raise ValueError(f"{name} contains unknown actions: {unknown}")
    return arr


def build_action_neutral_stage6_diagnostics(
    *,
    row_ids: Sequence[int],
    firm_ids: Sequence[str],
    actions: Sequence[str],
    oracle_scores: Mapping[str, np.ndarray],
    actor_actions: Sequence[str],
    critic_actions: Sequence[str],
    c2_actions: Sequence[str],
    actor_probabilities: np.ndarray | None = None,
    critic_q_values: np.ndarray | None = None,
    family_of: Callable[[str], str] | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    """Compare C3 with every per-firm optimum without a privileged baseline.

    Returns a JSON-serializable summary plus DataFrames under ``tables``.  A
    caller can persist those tables with :func:`write_action_diagnostics`.
    """

    action_order = [str(action) for action in actions]
    if not action_order or len(set(action_order)) != len(action_order):
        raise ValueError("actions must be a non-empty unique ordered vocabulary")
    rows = np.asarray(row_ids, dtype=int)
    firms = np.asarray(firm_ids, dtype=object).astype(str)
    n, k = len(rows), len(action_order)
    if len(set(rows.tolist())) != n or firms.shape != (n,):
        raise ValueError("row_ids must be unique and firm_ids must align one-to-one")
    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")

    matrices: dict[str, np.ndarray] = {}
    for axis in ("alpha", "beta", "gamma"):
        matrix = np.asarray(oracle_scores[axis], dtype=np.float64)
        if matrix.shape != (n, k) or not np.isfinite(matrix).all():
            raise ValueError(f"oracle {axis} matrix must be finite with shape {(n, k)}")
        matrices[axis] = matrix

    actor = _validate_actions(actor_actions, action_order, "actor_actions", n)
    critic = _validate_actions(critic_actions, action_order, "critic_actions", n)
    c2 = _validate_actions(c2_actions, action_order, "c2_actions", n)
    action_index = {action: idx for idx, action in enumerate(action_order)}
    actor_idx = np.asarray([action_index[action] for action in actor], dtype=int)
    critic_idx = np.asarray([action_index[action] for action in critic], dtype=int)
    c2_idx = np.asarray([action_index[action] for action in c2], dtype=int)
    row_index = np.arange(n)

    alpha = matrices["alpha"]
    alpha_max = alpha.max(axis=1)
    alpha_min = alpha.min(axis=1)
    optimal_mask = np.isclose(alpha, alpha_max[:, None], atol=tolerance, rtol=0.0)
    optimal_count = optimal_mask.sum(axis=1).astype(int)
    if (optimal_count < 1).any():
        raise ValueError("every firm must have at least one Alpha-optimal action")

    # Rank 1 is best. Ties receive the minimum rank occupied by the tied value.
    strict_better_count = (alpha > alpha[row_index, actor_idx][:, None] + tolerance).sum(axis=1)
    actor_rank = strict_better_count.astype(int) + 1
    critic_strict_better = (alpha > alpha[row_index, critic_idx][:, None] + tolerance).sum(axis=1)
    critic_rank = critic_strict_better.astype(int) + 1
    actor_hit = optimal_mask[row_index, actor_idx]
    critic_hit = optimal_mask[row_index, critic_idx]
    unique_optimal = optimal_count == 1
    unique_optimal_idx = optimal_mask.argmax(axis=1)
    unique_optimal_action = np.where(unique_optimal, np.asarray(action_order, dtype=object)[unique_optimal_idx], None)

    actor_alpha = alpha[row_index, actor_idx]
    critic_alpha = alpha[row_index, critic_idx]
    alpha_range = alpha_max - alpha_min
    actor_regret = alpha_max - actor_alpha
    critic_regret = alpha_max - critic_alpha
    normalized_actor_regret = np.divide(
        actor_regret,
        alpha_range,
        out=np.zeros_like(actor_regret),
        where=alpha_range > tolerance,
    )

    rowwise = pd.DataFrame({
        "row_id": rows,
        "firm_id": firms,
        "alpha_optimal_action_count": optimal_count,
        "alpha_optimal_action_set": [
            _json_set([action_order[j] for j in np.flatnonzero(optimal_mask[i])]) for i in range(n)
        ],
        "alpha_unique_optimal_action": unique_optimal_action,
        "alpha_unique_optimal": unique_optimal,
        "action__C3_actor": actor,
        "action__critic_Q_argmax_diagnosis_only": critic,
        "action__C2": c2,
        "C3_alpha": actor_alpha,
        "critic_alpha": critic_alpha,
        "C2_alpha": alpha[row_index, c2_idx],
        "oracle_alpha_ceiling": alpha_max,
        "oracle_alpha_floor": alpha_min,
        "oracle_alpha_range": alpha_range,
        "C3_alpha_regret": actor_regret,
        "C3_alpha_regret_normalized_by_firm_range": normalized_actor_regret,
        "critic_alpha_regret": critic_regret,
        "C3_alpha_rank": actor_rank,
        "critic_alpha_rank": critic_rank,
        "C3_in_alpha_optimal_set": actor_hit,
        "critic_in_alpha_optimal_set": critic_hit,
        "C3_exact_unique_optimal_hit": unique_optimal & (actor_idx == unique_optimal_idx),
        "critic_exact_unique_optimal_hit": unique_optimal & (critic_idx == unique_optimal_idx),
        "actor_critic_agreement": actor == critic,
        "C3_minus_C2_alpha": actor_alpha - alpha[row_index, c2_idx],
    })
    for axis, matrix in matrices.items():
        rowwise[f"C3_{axis}"] = matrix[row_index, actor_idx]
        rowwise[f"critic_{axis}"] = matrix[row_index, critic_idx]
        rowwise[f"C2_{axis}"] = matrix[row_index, c2_idx]
        rowwise[f"oracle_{axis}_ceiling"] = matrix.max(axis=1)

    # Each firm contributes total weight one across all tied optimal actions.
    optimal_long_rows: list[dict[str, Any]] = []
    for i in range(n):
        weight = 1.0 / float(optimal_count[i])
        for j in np.flatnonzero(optimal_mask[i]):
            optimal_long_rows.append({
                "row_id": int(rows[i]),
                "firm_id": str(firms[i]),
                "alpha_optimal_action": action_order[j],
                "optimal_set_fractional_weight": weight,
                "C3_selected_action": str(actor[i]),
                "C3_in_alpha_optimal_set": bool(actor_hit[i]),
                "C3_alpha_rank": int(actor_rank[i]),
                "C3_alpha_regret": float(actor_regret[i]),
            })
    optimal_to_c3_long = pd.DataFrame(optimal_long_rows)
    optimal_to_c3 = (
        optimal_to_c3_long.groupby(["alpha_optimal_action", "C3_selected_action"], as_index=False)
        .agg(
            fractional_firm_count=("optimal_set_fractional_weight", "sum"),
            contributing_firm_count=("row_id", "nunique"),
            mean_C3_alpha_regret=("C3_alpha_regret", "mean"),
        )
    )
    denom = optimal_to_c3.groupby("alpha_optimal_action")["fractional_firm_count"].transform("sum")
    optimal_to_c3["C3_selection_share_within_optimal_action"] = optimal_to_c3["fractional_firm_count"] / denom

    fractional_prevalence = (
        optimal_to_c3_long.groupby("alpha_optimal_action", as_index=False)["optimal_set_fractional_weight"].sum()
        .rename(columns={"alpha_optimal_action": "action", "optimal_set_fractional_weight": "fractional_optimal_firm_count"})
    )
    fractional_prevalence["fractional_optimal_share"] = fractional_prevalence["fractional_optimal_firm_count"] / float(n)
    inclusion_counts = optimal_to_c3_long.groupby("alpha_optimal_action")["row_id"].nunique()
    fractional_prevalence["optimal_set_inclusion_count"] = fractional_prevalence["action"].map(inclusion_counts).astype(int)
    fractional_prevalence["optimal_set_inclusion_share"] = fractional_prevalence["optimal_set_inclusion_count"] / float(n)
    fractional_prevalence = fractional_prevalence.sort_values(
        ["fractional_optimal_share", "action"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)

    unique_frame = rowwise.loc[rowwise["alpha_unique_optimal"]].copy()
    unique_prevalence = (
        unique_frame["alpha_unique_optimal_action"].value_counts().rename_axis("action").reset_index(name="unique_optimal_count")
    )
    unique_prevalence["unique_optimal_share_all_firms"] = unique_prevalence["unique_optimal_count"] / float(n)
    unique_prevalence["share_within_unique_optimum_firms"] = unique_prevalence["unique_optimal_count"] / max(len(unique_frame), 1)

    unique_confusion = (
        unique_frame.groupby(["alpha_unique_optimal_action", "action__C3_actor"], as_index=False)
        .agg(firm_count=("row_id", "size"), mean_C3_alpha_regret=("C3_alpha_regret", "mean"))
        .rename(columns={"action__C3_actor": "C3_selected_action"})
    )
    if len(unique_confusion):
        unique_denom = unique_confusion.groupby("alpha_unique_optimal_action")["firm_count"].transform("sum")
        unique_confusion["C3_selection_share_within_unique_optimum"] = unique_confusion["firm_count"] / unique_denom

    by_c3 = rowwise.groupby("action__C3_actor", as_index=False).agg(
        firm_count=("row_id", "size"),
        optimal_set_hit_rate=("C3_in_alpha_optimal_set", "mean"),
        exact_unique_optimal_hit_rate=("C3_exact_unique_optimal_hit", "mean"),
        mean_alpha_rank=("C3_alpha_rank", "mean"),
        median_alpha_rank=("C3_alpha_rank", "median"),
        mean_alpha_regret=("C3_alpha_regret", "mean"),
        median_alpha_regret=("C3_alpha_regret", "median"),
        mean_normalized_alpha_regret=("C3_alpha_regret_normalized_by_firm_range", "mean"),
        actor_critic_agreement=("actor_critic_agreement", "mean"),
    ).rename(columns={"action__C3_actor": "C3_selected_action"})
    by_c3["selection_share"] = by_c3["firm_count"] / float(n)
    by_c3 = by_c3.sort_values(["firm_count", "C3_selected_action"], ascending=[False, True], kind="mergesort")

    rank_distribution = rowwise["C3_alpha_rank"].value_counts().sort_index().rename_axis("alpha_rank").reset_index(name="firm_count")
    rank_distribution["firm_share"] = rank_distribution["firm_count"] / float(n)

    q_rank: list[float] = []
    probability_rank: list[float] = []
    if critic_q_values is not None:
        q = np.asarray(critic_q_values, dtype=np.float64)
        if q.shape != (n, k) or not np.isfinite(q).all():
            raise ValueError("critic_q_values must be finite and aligned with the Oracle matrix")
        for i in range(n):
            value = spearmanr(q[i], alpha[i]).statistic
            if np.isfinite(value):
                q_rank.append(float(value))
    if actor_probabilities is not None:
        probabilities = np.asarray(actor_probabilities, dtype=np.float64)
        if probabilities.shape != (n, k) or not np.isfinite(probabilities).all():
            raise ValueError("actor_probabilities must be finite and aligned with the Oracle matrix")
        for i in range(n):
            value = spearmanr(probabilities[i], alpha[i]).statistic
            if np.isfinite(value):
                probability_rank.append(float(value))

    actor_distribution = _distribution(actor)
    critic_distribution = _distribution(critic)
    family_summary: dict[str, Any] = {}
    if family_of is not None:
        actor_families = [str(family_of(action)) for action in actor]
        critic_families = [str(family_of(action)) for action in critic]
        family_summary = {
            "actor": _distribution(actor_families),
            "critic": _distribution(critic_families),
        }

    summary = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "status": "PASS",
        "diagnostic_semantics": "action_neutral_per_firm_optimal_set_and_C3_selection_audit",
        "privileged_reference_action": None,
        "candidate_specific_success_target_used": False,
        "candidate_specific_calibration_used": False,
        "oracle_usage": {
            "used_after_actor_selection_only": True,
            "used_for_stage2_5_training": False,
            "used_for_firm_action_selection": False,
            "used_for_posthoc_action_replacement": False,
        },
        "n_firms": int(n),
        "n_actions": int(k),
        "alpha_tolerance": float(tolerance),
        "optimal_surface": {
            "unique_optimum_firm_count": int(unique_optimal.sum()),
            "unique_optimum_firm_share": float(unique_optimal.mean()),
            "tied_optimum_firm_count": int((~unique_optimal).sum()),
            "tied_optimum_firm_share": float((~unique_optimal).mean()),
            "mean_optimal_set_size": float(optimal_count.mean()),
            "max_optimal_set_size": int(optimal_count.max()),
        },
        "C3_quality": {
            "alpha_optimal_set_hit_rate": float(actor_hit.mean()),
            "exact_unique_optimal_hit_rate_all_firms": float((unique_optimal & (actor_idx == unique_optimal_idx)).mean()),
            "exact_unique_optimal_hit_rate_within_unique_optimum_firms": float((actor_idx[unique_optimal] == unique_optimal_idx[unique_optimal]).mean()) if unique_optimal.any() else None,
            "top2_alpha_action_rate": float((actor_rank <= 2).mean()),
            "top3_alpha_action_rate": float((actor_rank <= 3).mean()),
            "mean_alpha_rank": float(actor_rank.mean()),
            "median_alpha_rank": float(np.median(actor_rank)),
            "mean_alpha_regret": float(actor_regret.mean()),
            "median_alpha_regret": float(np.median(actor_regret)),
            "p90_alpha_regret": float(np.quantile(actor_regret, 0.90)),
            "mean_normalized_alpha_regret": float(normalized_actor_regret.mean()),
            "C3_minus_C2_alpha": float((actor_alpha - alpha[row_index, c2_idx]).mean()),
        },
        "critic_quality": {
            "alpha_optimal_set_hit_rate": float(critic_hit.mean()),
            "exact_unique_optimal_hit_rate_all_firms": float((unique_optimal & (critic_idx == unique_optimal_idx)).mean()),
            "top2_alpha_action_rate": float((critic_rank <= 2).mean()),
            "mean_alpha_rank": float(critic_rank.mean()),
            "mean_alpha_regret": float(critic_regret.mean()),
            "within_firm_alpha_spearman_mean": float(np.mean(q_rank)) if q_rank else None,
            "within_firm_alpha_spearman_median": float(np.median(q_rank)) if q_rank else None,
        },
        "actor_probability_quality": {
            "within_firm_alpha_spearman_mean": float(np.mean(probability_rank)) if probability_rank else None,
            "within_firm_alpha_spearman_median": float(np.median(probability_rank)) if probability_rank else None,
        },
        "actor_critic_agreement": float((actor == critic).mean()),
        "actor_distribution": actor_distribution,
        "critic_distribution": critic_distribution,
        "family_distribution": family_summary,
        "q_argmax_deployment_used": False,
        "posthoc_action_replacement_used": False,
    }

    critic_good = summary["critic_quality"]["alpha_optimal_set_hit_rate"] >= 0.50 or (
        summary["critic_quality"]["within_firm_alpha_spearman_mean"] is not None
        and summary["critic_quality"]["within_firm_alpha_spearman_mean"] >= 0.25
    )
    actor_gap = (
        summary["critic_quality"]["alpha_optimal_set_hit_rate"]
        - summary["C3_quality"]["alpha_optimal_set_hit_rate"] >= 0.10
        and summary["actor_critic_agreement"] < 0.50
    )
    summary["bottleneck_classification"] = (
        "ACTOR_EXTRACTION_BOTTLENECK" if critic_good and actor_gap else
        "MIXED_BOTTLENECK" if (not critic_good) and actor_gap else
        "REWARD_ALIGNMENT_BOTTLENECK" if not critic_good else
        "NO_CLEAR_BOTTLENECK"
    )

    return {
        "summary": summary,
        "tables": {
            "rowwise": rowwise,
            "optimal_action_membership_long": optimal_to_c3_long,
            "optimal_action_prevalence": fractional_prevalence,
            "unique_optimal_action_prevalence": unique_prevalence,
            "optimal_action_to_C3_selection": optimal_to_c3,
            "unique_optimal_action_to_C3_confusion": unique_confusion,
            "C3_selection_quality_by_action": by_c3,
            "C3_alpha_rank_distribution": rank_distribution,
        },
    }


def write_action_diagnostics(output_dir: Path, result: Mapping[str, Any]) -> dict[str, str]:
    """Persist the complete action-neutral diagnostic bundle."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    summary_path = output / "stage6_action_diagnostics.json"
    summary_path.write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    artifacts["summary"] = summary_path.name
    for name, table in result["tables"].items():
        if name in {"rowwise", "optimal_action_membership_long"}:
            path = output / f"{name}.parquet"
            table.to_parquet(path, index=False)
        else:
            path = output / f"{name}.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig")
        artifacts[name] = path.name
    return artifacts

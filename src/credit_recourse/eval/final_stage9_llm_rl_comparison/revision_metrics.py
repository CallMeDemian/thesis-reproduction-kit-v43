from __future__ import annotations

"""Plan-3 revision geometry and pre-specified primary contrasts.

The module is pure analysis code: it consumes frozen Stage7 actions and Stage8
Oracle scores. It never chooses actions, reranks a policy, or feeds Oracle
information back into inference.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PRIMARY_CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("C5-C4", "C5", "C4"),
    ("C4R-C4", "C4R", "C4"),
    ("C6-E-C4R", "C6-E", "C4R"),
    ("C6-E-C6-EX", "C6-E", "C6-EX"),
)
REVISION_PAIRS: tuple[tuple[str, str], ...] = (
    ("C4", "C4R"),
    ("C4", "C6-E"),
    ("C4", "C6-EX"),
)


@dataclass(frozen=True)
class ActionGeometry:
    columns: tuple[str, ...]
    widths: np.ndarray
    fixed_candidates: dict[str, dict[str, float]]

    def candidate_vector(self, candidate_id: str) -> np.ndarray:
        row = self.fixed_candidates[candidate_id]
        return np.asarray([float(row[col]) for col in self.columns], dtype=float)


def load_action_geometry(project_root: Path) -> ActionGeometry:
    path = (
        Path(project_root).resolve()
        / "archive/DEPLOYED_RELEASE/stage2_candidate_projection/candidate_action_contract_v4_3.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    columns = tuple(contract["action_columns"])
    widths = np.asarray(
        [
            float(contract["action_bounds"][col][1])
            - float(contract["action_bounds"][col][0])
            for col in columns
        ],
        dtype=float,
    )
    return ActionGeometry(
        columns=columns,
        widths=widths,
        fixed_candidates={
            key: {col: float(value[col]) for col in columns}
            for key, value in contract["fixed_candidates"].items()
        },
    )


def _action_vec(row: pd.Series, columns: tuple[str, ...]) -> np.ndarray:
    values = pd.to_numeric(row.loc[list(columns)], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Revision analysis refuses a non-finite action")
    return values


def compute_pairwise_revision(
    *,
    a_initial: pd.Series,
    a_revised: pd.Series,
    reference_candidate: str | None,
    geometry: ActionGeometry,
    initial_scores: dict[str, float] | None = None,
    revised_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    a0 = _action_vec(a_initial, geometry.columns)
    a1 = _action_vec(a_revised, geometry.columns)
    a0n = a0 / np.maximum(geometry.widths, 1e-12)
    a1n = a1 / np.maximum(geometry.widths, 1e-12)
    change = a1n - a0n
    n0 = float(np.linalg.norm(a0n))
    n1 = float(np.linalg.norm(a1n))
    cosine = (
        float("nan")
        if n0 < 1e-12 or n1 < 1e-12
        else 1.0 - float(np.dot(a0n, a1n) / (n0 * n1))
    )
    rec: dict[str, Any] = {
        "initial_candidate_id": str(a_initial.get("candidate_id", "")),
        "revised_candidate_id": str(a_revised.get("candidate_id", "")),
        "reference_candidate_id": reference_candidate,
        "reference_source": str(a_revised.get("reference_source", "")),
        "metrics_defined": False,
        "undefined_reason": None,
        "revision_changed_candidate_flag": str(a_initial.get("candidate_id", ""))
        != str(a_revised.get("candidate_id", "")),
        "revision_changed_active_dimensions": int(
            np.sum((np.abs(a0n) > 1e-3) != (np.abs(a1n) > 1e-3))
        ),
        "revision_l1_distance": float(np.sum(np.abs(change))),
        "revision_l2_distance": float(np.linalg.norm(change)),
        "revision_cosine_distance": cosine,
    }
    if reference_candidate is None:
        rec.update(
            undefined_reason="no_action_reference",
            rl_adoption_ratio=float("nan"),
            self_retention_ratio=float("nan"),
            orthogonal_drift=float("nan"),
            u_norm_squared=float("nan"),
        )
    elif reference_candidate not in geometry.fixed_candidates:
        rec.update(
            undefined_reason="reference_not_in_current_candidate9",
            rl_adoption_ratio=float("nan"),
            self_retention_ratio=float("nan"),
            orthogonal_drift=float("nan"),
            u_norm_squared=float("nan"),
        )
    else:
        ref = geometry.candidate_vector(reference_candidate)
        refn = ref / np.maximum(geometry.widths, 1e-12)
        u = refn - a0n
        u_sq = float(np.dot(u, u))
        rec["u_norm_squared"] = u_sq
        if u_sq < 1e-12:
            rec.update(
                undefined_reason="reference_equals_initial_action",
                rl_adoption_ratio=float("nan"),
                self_retention_ratio=float("nan"),
                orthogonal_drift=float("nan"),
            )
        else:
            adoption = float(np.dot(change, u) / u_sq)
            orthogonal = change - adoption * u
            rec.update(
                metrics_defined=True,
                rl_adoption_ratio=adoption,
                self_retention_ratio=1.0 - adoption,
                orthogonal_drift=float(np.linalg.norm(orthogonal) / math.sqrt(u_sq)),
            )

    initial_scores = initial_scores or {}
    revised_scores = revised_scores or {}
    for backend in ("alpha", "beta", "gamma"):
        before = initial_scores.get(backend)
        after = revised_scores.get(backend)
        rec[f"initial_delta_R_score_{backend}"] = (
            float(before) if before is not None and np.isfinite(before) else float("nan")
        )
        rec[f"revised_delta_R_score_{backend}"] = (
            float(after) if after is not None and np.isfinite(after) else float("nan")
        )
        rec[f"revision_delta_R_score_{backend}"] = (
            float(after - before)
            if before is not None
            and after is not None
            and np.isfinite(before)
            and np.isfinite(after)
            else float("nan")
        )
    return rec


def build_revision_table(
    *,
    action_table: pd.DataFrame,
    stage8_scores: pd.DataFrame,
    geometry: ActionGeometry,
) -> pd.DataFrame:
    required = {
        "request_id",
        "parent_request_id",
        "row_id",
        "policy",
        "model_key",
        "mode",
        "information_condition",
        "budget",
        "replicate",
        *geometry.columns,
    }
    missing = required - set(action_table)
    if missing:
        raise ValueError(f"Revision action table missing columns: {sorted(missing)}")
    actions = action_table.set_index("request_id", drop=False)
    scores = stage8_scores.set_index("request_id", drop=False)
    records: list[dict[str, Any]] = []
    for base_condition, revision_condition in REVISION_PAIRS:
        revised = action_table[action_table["policy"].astype(str).eq(revision_condition)]
        for _, row in revised.iterrows():
            parent_id = row.get("parent_request_id")
            if parent_id is None or pd.isna(parent_id) or str(parent_id) not in actions.index:
                continue
            initial = actions.loc[str(parent_id)]
            if isinstance(initial, pd.DataFrame):
                raise ValueError("Stage7 request identity is not unique")
            if str(initial["policy"]) != base_condition:
                raise ValueError("Revision parent is not the exact C4 request")
            if str(row["request_id"]) not in scores.index or str(parent_id) not in scores.index:
                continue
            before_score = scores.loc[str(parent_id)]
            after_score = scores.loc[str(row["request_id"])]
            if isinstance(before_score, pd.DataFrame) or isinstance(after_score, pd.DataFrame):
                raise ValueError("Stage8 score request identity is not unique")
            reference = (
                str(row["reference_candidate_id"])
                if revision_condition in {"C6-E", "C6-EX"}
                and pd.notna(row.get("reference_candidate_id"))
                else None
            )
            rec = compute_pairwise_revision(
                a_initial=initial,
                a_revised=row,
                reference_candidate=reference,
                geometry=geometry,
                initial_scores={
                    key: float(before_score[f"delta_R_score_{key}"])
                    for key in ("alpha", "beta", "gamma")
                },
                revised_scores={
                    key: float(after_score[f"delta_R_score_{key}"])
                    for key in ("alpha", "beta", "gamma")
                },
            )
            for key in (
                "row_id",
                "firm_key",
                "model_key",
                "mode",
                "information_condition",
                "budget",
                "replicate",
                "analysis_population",
            ):
                rec[key] = row.get(key)
            rec.update(
                initial_request_id=str(parent_id),
                revision_request_id=str(row["request_id"]),
                base_condition=base_condition,
                revision_condition=revision_condition,
            )
            records.append(rec)
    return pd.DataFrame(records)


def build_identity_contrast_table(revision_df: pd.DataFrame) -> pd.DataFrame:
    """Firm-matched C6-E minus C6-EX reference-identity behavior contrast."""
    if revision_df.empty:
        return pd.DataFrame()
    keys = [
        key
        for key in (
            "row_id",
            "firm_key",
            "model_key",
            "mode",
            "information_condition",
            "budget",
            "replicate",
            "analysis_population",
        )
        if key in revision_df
    ]
    measures = (
        "rl_adoption_ratio",
        "self_retention_ratio",
        "orthogonal_drift",
        "revision_delta_R_score_alpha",
        "revision_delta_R_score_beta",
        "revision_delta_R_score_gamma",
    )
    left = revision_df[
        revision_df["revision_condition"].astype(str).eq("C6-E")
    ][keys + list(measures)].copy()
    right = revision_df[
        revision_df["revision_condition"].astype(str).eq("C6-EX")
    ][keys + list(measures)].copy()
    out = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_C6E", "_C6EX"),
    )
    for name in measures:
        out[f"C6E_minus_C6EX__{name}"] = (
            pd.to_numeric(out[f"{name}_C6E"], errors="coerce")
            - pd.to_numeric(out[f"{name}_C6EX"], errors="coerce")
        )
    return out


def _contrast_pairs(scores: pd.DataFrame) -> pd.DataFrame:
    keys = [
        key
        for key in (
            "row_id",
            "firm_key",
            "model_key",
            "mode",
            "info",
            "information_condition",
            "budget",
            "replicate",
            "analysis_population",
        )
        if key in scores
    ]
    records: list[pd.DataFrame] = []
    score_cols = [f"R_score_{key}" for key in ("alpha", "beta", "gamma")]
    for label, treatment, control in PRIMARY_CONTRASTS:
        lhs = scores[scores["policy"].astype(str).eq(treatment)][
            keys + ["request_id"] + score_cols
        ].copy()
        rhs = scores[scores["policy"].astype(str).eq(control)][
            keys + ["request_id"] + score_cols
        ].copy()
        paired = lhs.merge(
            rhs,
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_treatment", "_control"),
        )
        if paired.empty:
            continue
        paired.insert(0, "contrast", label)
        paired.insert(1, "treatment_condition", treatment)
        paired.insert(2, "control_condition", control)
        for backend in ("alpha", "beta", "gamma"):
            paired[f"effect_{backend}"] = (
                pd.to_numeric(
                    paired[f"R_score_{backend}_treatment"], errors="coerce"
                )
                - pd.to_numeric(
                    paired[f"R_score_{backend}_control"], errors="coerce"
                )
            )
        records.append(paired)
    return pd.concat(records, ignore_index=True, sort=False) if records else pd.DataFrame()


def _summarize_effects(
    firm_effects: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if firm_effects.empty:
        return pd.DataFrame()
    group_keys = [
        key
        for key in (
            "contrast",
            "treatment_condition",
            "control_condition",
            "model_key",
            "mode",
            "info",
            "information_condition",
            "budget",
            "replicate",
            "analysis_population",
        )
        if key in firm_effects
    ]
    rng = np.random.Generator(np.random.PCG64(int(bootstrap_seed)))
    rows: list[dict[str, Any]] = []
    for keys, group in firm_effects.groupby(group_keys, sort=True, dropna=False):
        values_by_key = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_keys, values_by_key))
        for backend in ("alpha", "beta", "gamma"):
            values = pd.to_numeric(
                group[f"effect_{backend}"], errors="coerce"
            ).dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            mean = float(values.mean())
            sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            se = sd / math.sqrt(len(values)) if len(values) else float("nan")
            if bootstrap_replicates:
                indices = rng.integers(
                    0, len(values), size=(int(bootstrap_replicates), len(values))
                )
                boot = values[indices].mean(axis=1)
                low, high = np.quantile(boot, [0.025, 0.975])
            else:
                low = high = mean
            rows.append(
                {
                    **base,
                    "backend": backend,
                    "n_firms": int(len(values)),
                    "mean_effect": mean,
                    "sd_effect": sd,
                    "se_effect": se,
                    "ci95_normal_low": mean - 1.96 * se,
                    "ci95_normal_high": mean + 1.96 * se,
                    "bootstrap_ci95_low": float(low),
                    "bootstrap_ci95_high": float(high),
                    "bootstrap_replicates": int(bootstrap_replicates),
                    "bootstrap_rng": "numpy.random.Generator(PCG64)",
                    "bootstrap_seed": int(bootstrap_seed),
                }
            )
    return pd.DataFrame(rows)


def build_primary_contrasts(
    stage8_scores: pd.DataFrame,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20_260_912,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {
        "request_id",
        "row_id",
        "model_key",
        "policy",
        "mode",
        "budget",
        "replicate",
        "analysis_population",
        "R_score_alpha",
        "R_score_beta",
        "R_score_gamma",
    }
    missing = required - set(stage8_scores)
    if missing:
        raise ValueError(f"Stage8 contrast input missing columns: {sorted(missing)}")
    conditions = set(stage8_scores["policy"].astype(str))
    allowed = {"C4", "C5", "C4R", "C6-E", "C6-EX"}
    if not conditions.issubset(allowed):
        raise ValueError(f"Stage9 refuses non-Plan-3 conditions: {sorted(conditions-allowed)}")

    firm = _contrast_pairs(stage8_scores)
    summary = _summarize_effects(
        firm,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )

    if firm.empty:
        return firm, summary, pd.DataFrame(), pd.DataFrame()
    interaction_keys = [
        key
        for key in (
            "contrast",
            "treatment_condition",
            "control_condition",
            "row_id",
            "firm_key",
            "model_key",
            "mode",
            "info",
            "information_condition",
            "replicate",
            "analysis_population",
        )
        if key in firm
    ]
    keep = interaction_keys + [f"effect_{key}" for key in ("alpha", "beta", "gamma")]
    b1 = firm[firm["budget"].astype(str).eq("B1")][keep]
    binf = firm[firm["budget"].astype(str).eq("BINF")][keep]
    interaction = b1.merge(
        binf,
        on=interaction_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_B1", "_BINF"),
    )
    for backend in ("alpha", "beta", "gamma"):
        interaction[f"effect_{backend}"] = (
            interaction[f"effect_{backend}_B1"]
            - interaction[f"effect_{backend}_BINF"]
        )
    interaction["budget"] = "B1_MINUS_BINF"
    interaction_summary = _summarize_effects(
        interaction,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    return firm, summary, interaction, interaction_summary

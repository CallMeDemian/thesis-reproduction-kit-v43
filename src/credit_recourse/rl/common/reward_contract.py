from __future__ import annotations

"""Canonical, Oracle-independent credit-reward component contract.

This module is deliberately kept below ``credit_recourse.rl.common`` and imports
neither Oracle nor rating-model code.  The profitability producer accepts only
the three canonical Financial Simulator statement fields listed in
``PROFITABILITY_FIELD_LINEAGE``.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Mapping

import numpy as np
import pandas as pd


REWARD_CONTRACT_VERSION = "credit_reward_v3_profitability"
REWARD_COMPOSER_VERSION = "canonical_credit_reward_composer_v1"
COMPONENT_NORMALIZATION_VERSION = "robust_p95_abs_then_fixed_clip_v1"
PROFITABILITY_METRIC = "delta_operating_income_over_factual_assets"
PROFITABILITY_FORMULA_VERSION = "delta_operating_income_over_factual_assets_v1"
PROFITABILITY_PRODUCER_VERSION = "profitability_reward_producer_v1"
PROFITABILITY_CLIP_BOUNDS = (-0.5, 0.5)
MERTON_CLIP_BOUNDS = (-1.0, 1.0)
FCFF_CLIP_BOUNDS = (-0.5, 0.5)
LIQUIDITY_CLIP_BOUNDS = (-0.5, 0.5)

PROFITABILITY_FIELD_LINEAGE = {
    "factual_operating_income": "sim__operating_income",
    "counterfactual_operating_income": "next__sim__operating_income",
    "factual_total_assets": "sim__total_assets",
}

EXPLICIT_COMPONENT_COLUMNS = (
    "reward_merton_raw",
    "reward_merton_norm",
    "reward_fcff_raw",
    "reward_fcff_norm",
    "reward_profitability_raw",
    "reward_profitability_norm",
    "reward_liquidity_raw",
    "reward_liquidity_norm",
)


class ProfitabilityInputError(ValueError):
    """Raised when the existing Stage2 fail-closed financial QC contract fails."""

    def __init__(self, message: str, *, qc: Mapping[str, Any]):
        super().__init__(message)
        self.qc = dict(qc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _numeric_required(frame: pd.DataFrame, column: str, *, context: str) -> pd.Series:
    if column not in frame.columns:
        qc = {
            "status": "FAIL",
            "context": context,
            "handling": "hard_fail_existing_stage2_financial_qc",
            "missing_required_field_count": 1,
            "missing_required_fields": [column],
            "nonfinite_count": 0,
            "nonpositive_factual_total_assets_count": 0,
            "silent_epsilon_used": False,
        }
        raise ProfitabilityInputError(f"{context}: missing required canonical financial-statement field {column}", qc=qc)
    value = frame[column]
    if isinstance(value, pd.DataFrame):
        qc = {
            "status": "FAIL",
            "context": context,
            "handling": "hard_fail_existing_stage2_financial_qc",
            "duplicate_field": column,
            "nonfinite_count": int(len(frame)),
            "nonpositive_factual_total_assets_count": 0,
            "silent_epsilon_used": False,
        }
        raise ProfitabilityInputError(f"{context}: duplicate field label {column!r} is ambiguous", qc=qc)
    numeric = pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan)
    bad = int(numeric.isna().sum())
    if bad:
        qc = {
            "status": "FAIL",
            "context": context,
            "handling": "hard_fail_existing_stage2_financial_qc",
            "field": column,
            "nonfinite_count": bad,
            "nonpositive_factual_total_assets_count": 0,
            "silent_epsilon_used": False,
        }
        raise ProfitabilityInputError(f"{context}: non-finite values in required field {column}: n={bad}", qc=qc)
    return numeric.astype(float)


def profitability_input_view(
    frame: pd.DataFrame,
    *,
    factual_operating_income_column: str = PROFITABILITY_FIELD_LINEAGE["factual_operating_income"],
    counterfactual_operating_income_column: str = PROFITABILITY_FIELD_LINEAGE["counterfactual_operating_income"],
    factual_total_assets_column: str = PROFITABILITY_FIELD_LINEAGE["factual_total_assets"],
) -> pd.DataFrame:
    """Return the producer's complete accessible input surface.

    Selecting the three fields here is an explicit dependency boundary: callers
    may hold wider Stage2/Stage6 frames, but the profitability definition cannot
    inspect Oracle, rating, candidate-vector, or policy-score columns.
    """

    columns = [
        factual_operating_income_column,
        counterfactual_operating_income_column,
        factual_total_assets_column,
    ]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        qc = {
            "status": "FAIL",
            "handling": "hard_fail_existing_stage2_financial_qc",
            "missing_required_field_count": int(len(missing)),
            "missing_required_fields": missing,
            "nonfinite_count": 0,
            "nonpositive_factual_total_assets_count": 0,
            "silent_epsilon_used": False,
        }
        raise ProfitabilityInputError(f"Missing profitability source-of-truth fields: {missing}", qc=qc)
    return frame.loc[:, columns].copy()


def compute_profitability_raw(
    frame: pd.DataFrame,
    *,
    context: str = "profitability_reward",
    factual_operating_income_column: str = PROFITABILITY_FIELD_LINEAGE["factual_operating_income"],
    counterfactual_operating_income_column: str = PROFITABILITY_FIELD_LINEAGE["counterfactual_operating_income"],
    factual_total_assets_column: str = PROFITABILITY_FIELD_LINEAGE["factual_total_assets"],
) -> tuple[pd.Series, dict[str, Any]]:
    """Compute (OI_cf - OI_factual) / factual assets without any fallback."""

    safe = profitability_input_view(
        frame,
        factual_operating_income_column=factual_operating_income_column,
        counterfactual_operating_income_column=counterfactual_operating_income_column,
        factual_total_assets_column=factual_total_assets_column,
    )
    factual_oi = _numeric_required(safe, factual_operating_income_column, context=context)
    counterfactual_oi = _numeric_required(safe, counterfactual_operating_income_column, context=context)
    factual_assets = _numeric_required(safe, factual_total_assets_column, context=context)
    invalid_assets = factual_assets <= 0.0
    invalid_count = int(invalid_assets.sum())
    if invalid_count:
        qc = {
            "status": "FAIL",
            "context": context,
            "handling": "hard_fail_existing_stage2_financial_qc",
            "missing_required_field_count": 0,
            "nonfinite_count": 0,
            "nonpositive_factual_total_assets_count": invalid_count,
            "silent_epsilon_used": False,
        }
        raise ProfitabilityInputError(
            f"{context}: factual total assets must be positive; n={invalid_count}", qc=qc
        )
    raw = (counterfactual_oi - factual_oi) / factual_assets
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        qc = {
            "status": "FAIL",
            "context": context,
            "handling": "hard_fail_existing_stage2_financial_qc",
            "missing_required_field_count": 0,
            "nonfinite_count": int((~np.isfinite(raw.to_numpy(dtype=float))).sum()),
            "nonpositive_factual_total_assets_count": 0,
            "silent_epsilon_used": False,
        }
        raise ProfitabilityInputError(f"{context}: profitability_raw contains non-finite values", qc=qc)
    raw.name = "reward_profitability_raw"
    return raw, {
        "status": "PASS",
        "context": context,
        "rows": int(len(raw)),
        "handling": "hard_fail_existing_stage2_financial_qc",
        "missing_required_field_count": 0,
        "nonfinite_count": 0,
        "nonpositive_factual_total_assets_count": 0,
        "invalid_case_count": 0,
        "silent_epsilon_used": False,
        "field_lineage": dict(PROFITABILITY_FIELD_LINEAGE),
        "formula": "(counterfactual_operating_income - factual_operating_income) / factual_total_assets",
        "metric_name": PROFITABILITY_METRIC,
    }


def numeric_series_hash(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype="<f8", copy=True)
    return hashlib.sha256(numeric.tobytes(order="C")).hexdigest()


def fit_robust_p95_abs(values: pd.Series, *, label: str) -> float:
    """Canonical scale utility shared by every reward component."""

    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().abs()
    if numeric.empty:
        raise ValueError(f"Cannot compute canonical reward scale for {label}: no finite values")
    scale = float(numeric.quantile(0.95))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"Cannot compute canonical reward scale for {label}: p95_abs={scale}")
    return scale


def normalize_reward_component(
    values: pd.Series,
    *,
    scale_parameter: float,
    clip_bounds: tuple[float, float],
    label: str,
) -> pd.Series:
    """Apply the shared robust-scale and fixed-clipping framework."""

    scale = float(scale_parameter)
    lower, upper = (float(clip_bounds[0]), float(clip_bounds[1]))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"Invalid canonical reward scale for {label}: {scale}")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"Invalid canonical reward clip bounds for {label}: {(lower, upper)}")
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.isna().any():
        raise ValueError(f"Non-finite values in canonical reward component {label}: n={int(numeric.isna().sum())}")
    return (numeric.astype(float) / scale).clip(lower, upper)


def build_profitability_scale_artifact(
    training_raw: pd.Series,
    *,
    fit_sample_definition: str,
    created_from_stage: str,
    input_hash: str | None = None,
) -> dict[str, Any]:
    """Fit the Profitability scale from training-side counterfactual rows only."""

    numeric = pd.to_numeric(training_raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.isna().any():
        raise ValueError(
            "Profitability scale fit sample contains non-finite values; existing Stage2 QC must handle them first"
        )
    if numeric.empty:
        raise ValueError("Profitability scale fit sample is empty")
    quantile_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    quantile_values = numeric.quantile(quantile_levels)
    scale = fit_robust_p95_abs(numeric, label=PROFITABILITY_METRIC)
    artifact = {
        "metric_name": PROFITABILITY_METRIC,
        "formula_version": PROFITABILITY_FORMULA_VERSION,
        "normalization_version": COMPONENT_NORMALIZATION_VERSION,
        "fit_sample_definition": str(fit_sample_definition),
        "evaluation_cohort_used_for_fit": False,
        "stage6_oracle_result_used_for_fit": False,
        "n": int(len(numeric)),
        "median": float(numeric.median()),
        "quantiles": {f"p{int(q * 100):02d}": float(quantile_values.loc[q]) for q in quantile_levels},
        "scale_parameter": scale,
        "scale_statistic": "p95_abs",
        "clip_bounds": list(PROFITABILITY_CLIP_BOUNDS),
        "input_hash": str(input_hash or numeric_series_hash(numeric)),
        "producer_version": PROFITABILITY_PRODUCER_VERSION,
        "created_from_stage": str(created_from_stage),
        "created_utc": _utc_now(),
        "field_lineage": dict(PROFITABILITY_FIELD_LINEAGE),
        "invalid_denominator_handling": "hard_fail_existing_stage2_financial_qc_no_epsilon",
        "oracle_used_in_profitability_definition": False,
        "oracle_used_in_stage5_training": False,
        "oracle_evaluation_informed_reward_design": True,
    }
    artifact["profitability_scale_hash"] = canonical_json_hash(
        {key: value for key, value in artifact.items() if key not in {"created_utc", "profitability_scale_hash"}}
    )
    return artifact


def compose_reward(
    *,
    merton_component: Any,
    fcff_component: Any,
    profitability_component: Any,
    liquidity_component: Any,
    merton_lambda: float,
    fcff_lambda: float,
    profitability_lambda: float = 0.0,
    liquidity_lambda: float,
    sector_phi_component: Any = 0.0,
    sector_phi_lambda: float = 0.0,
    base_component: Any = 0.0,
) -> Any:
    """Single source of truth for Stage2/5/search/verifier reward composition.

    The order intentionally preserves the historical three-component arithmetic:
    base, sector-phi, Merton, FCFF, Profitability, Liquidity.  Adding an exact
    zero Profitability term therefore leaves the old finite result bitwise equal.
    """

    total = base_component + float(sector_phi_lambda) * sector_phi_component
    total = total + float(merton_lambda) * merton_component
    total = total + float(fcff_lambda) * fcff_component
    total = total + float(profitability_lambda) * profitability_component
    total = total + float(liquidity_lambda) * liquidity_component
    return total


def reward_contract_metadata(
    *,
    merton_lambda: float,
    fcff_lambda: float,
    profitability_lambda: float = 0.0,
    liquidity_lambda: float,
    sector_phi_rho: float,
    profitability_scale_hash: str,
) -> dict[str, Any]:
    payload = {
        "reward_contract_version": REWARD_CONTRACT_VERSION,
        "reward_composer_version": REWARD_COMPOSER_VERSION,
        "component_normalization_version": COMPONENT_NORMALIZATION_VERSION,
        "merton_lambda": float(merton_lambda),
        "fcff_lambda": float(fcff_lambda),
        "profitability_lambda": float(profitability_lambda),
        "liquidity_lambda": float(liquidity_lambda),
        "sector_phi_rho": float(sector_phi_rho),
        "profitability_metric": PROFITABILITY_METRIC,
        "profitability_formula_version": PROFITABILITY_FORMULA_VERSION,
        "profitability_scale_hash": str(profitability_scale_hash),
        "profitability_field_lineage": dict(PROFITABILITY_FIELD_LINEAGE),
        "oracle_used_in_profitability_definition": False,
        "oracle_used_in_stage5_training": False,
        "oracle_evaluation_informed_reward_design": True,
    }
    payload["reward_contract_hash"] = canonical_json_hash(payload)
    return payload

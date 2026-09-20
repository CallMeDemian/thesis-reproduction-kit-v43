"""Deterministic P50-normalized total-intensity candidate contract.

This module defines the action-space semantic version requested for the
Candidate-IQL intensity-normalization experiment.  It deliberately does not
touch the Simulator, Oracle, reward, or policy extraction rules.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .action_intensity import (
    MANAGERIAL_ACTION_DIMENSIONS,
    standardized_action_intensity,
    validate_action_intensity_scale_artifact,
)


CANDIDATE_ACTION_CONTRACT_VERSION = "p50_normalized_total_intensity_v2"
INTENSITY_TOLERANCE = 1e-8
VECTOR_TOLERANCE = 1e-12

TARGET_INTENSITY_BY_CANDIDATE: dict[str, float] = {
    "A0": 0.0,
    "DL1_deleverage_mild": 1.0,
    "DL2_deleverage_moderate": 1.5,
    "RF1_short_debt_refinance": 1.0,
    "CX1_capex_discipline": 1.0,
    "WC1_working_capital_tightening": 1.0,
    "WC2_supplier_financing": 1.0,
    "OE1_cost_efficiency_mild": 1.0,
    "OE2_cost_efficiency_moderate": 1.5,
    "MX1_cost_and_deleverage": 1.5,
    "MX2_liquidity_rescue": 1.5,
}

FAMILY_BY_CANDIDATE: dict[str, str] = {
    "A0": "noop",
    "DL1_deleverage_mild": "deleveraging",
    "DL2_deleverage_moderate": "deleveraging",
    "RF1_short_debt_refinance": "refinancing",
    "CX1_capex_discipline": "capex",
    "WC1_working_capital_tightening": "working_capital",
    "WC2_supplier_financing": "working_capital",
    "OE1_cost_efficiency_mild": "cost_efficiency",
    "OE2_cost_efficiency_moderate": "cost_efficiency",
    "MX1_cost_and_deleverage": "mixed",
    "MX2_liquidity_rescue": "mixed",
}


class ActionNormalizationConflict(ValueError):
    """Raised before training when the requested contract cannot be satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(vector: Mapping[str, Any], dimension: str) -> float:
    raw = vector.get(f"action__{dimension}", vector.get(dimension, 0.0))
    value = float(raw or 0.0)
    if not math.isfinite(value):
        raise ActionNormalizationConflict(
            f"Non-finite candidate value for {dimension}: {raw!r}"
        )
    return value


def _active_dimensions(vector: Mapping[str, Any]) -> set[str]:
    return {
        dim
        for dim in MANAGERIAL_ACTION_DIMENSIONS
        if abs(_number(vector, dim)) > VECTOR_TOLERANCE
    }


def _sign(value: float) -> int:
    if value > VECTOR_TOLERANCE:
        return 1
    if value < -VECTOR_TOLERANCE:
        return -1
    return 0


def normalize_candidate_library(
    library: Mapping[str, Any],
    scale_artifact: Mapping[str, Any],
    action_contract: Mapping[str, Any],
    *,
    base_template_hash: str,
    old_runtime_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return a normalized copy plus candidate/dimension audits.

    Every non-noop vector receives exactly one scalar.  Bounds are checked on
    the unmodified scaled vector; no clipping is performed anywhere here.
    """

    scales_payload = validate_action_intensity_scale_artifact(scale_artifact)
    scales = dict(scales_payload["scale_by_dimension"])
    fixed = dict(library.get("fixed_candidates") or {})
    if set(fixed) != set(TARGET_INTENSITY_BY_CANDIDATE):
        raise ActionNormalizationConflict(
            "Fixed candidate set does not match the canonical 11-action target map: "
            f"fixed={sorted(fixed)} expected={sorted(TARGET_INTENSITY_BY_CANDIDATE)}"
        )

    raw_bounds = dict(action_contract.get("action_bounds") or {})
    missing_bounds = [d for d in MANAGERIAL_ACTION_DIMENSIONS if d not in raw_bounds]
    if missing_bounds:
        raise ActionNormalizationConflict(f"Missing canonical action bounds: {missing_bounds}")

    normalized = deepcopy(dict(library))
    normalized_fixed = normalized["fixed_candidates"]
    candidate_rows: list[dict[str, Any]] = []
    dimension_rows: list[dict[str, Any]] = []
    bound_conflicts: list[dict[str, Any]] = []

    for candidate in fixed:
        old_vector = dict(fixed[candidate])
        old_intensity = standardized_action_intensity(old_vector, scales)
        if old_intensity is None:
            raise ActionNormalizationConflict(f"Cannot measure intensity for {candidate}")
        target = float(TARGET_INTENSITY_BY_CANDIDATE[candidate])
        if candidate == "A0":
            if abs(old_intensity) > INTENSITY_TOLERANCE:
                raise ActionNormalizationConflict("A0 is not a zero-intensity vector")
            scale_factor = 1.0
        else:
            if old_intensity <= 0.0:
                raise ActionNormalizationConflict(
                    f"Non-noop candidate has zero intensity: {candidate}"
                )
            scale_factor = target / old_intensity

        new_vector = deepcopy(old_vector)
        old_active = _active_dimensions(old_vector)
        for dimension in MANAGERIAL_ACTION_DIMENSIONS:
            old_value = _number(old_vector, dimension)
            new_value = old_value * scale_factor
            lo, hi = (float(x) for x in raw_bounds[dimension])
            if new_value < lo - VECTOR_TOLERANCE or new_value > hi + VECTOR_TOLERANCE:
                bound_conflicts.append(
                    {
                        "candidate": candidate,
                        "dimension": dimension,
                        "old_value": old_value,
                        "scaled_value": new_value,
                        "bound_lo": lo,
                        "bound_hi": hi,
                    }
                )
            new_vector[f"action__{dimension}"] = float(new_value)
            dimension_rows.append(
                {
                    "candidate": candidate,
                    "family": FAMILY_BY_CANDIDATE[candidate],
                    "dimension": dimension,
                    "p50_scale": float(scales[dimension]),
                    "old_value": old_value,
                    "new_value": float(new_value),
                    "relative_change": 0.0
                    if abs(old_value) <= VECTOR_TOLERANCE
                    else float(new_value / old_value - 1.0),
                    "old_standardized_contribution": abs(old_value) / float(scales[dimension]),
                    "new_standardized_contribution": abs(new_value) / float(scales[dimension]),
                    "sign_preserved": _sign(old_value) == _sign(new_value),
                    "active_axis_preserved": (dimension in old_active)
                    == (abs(new_value) > VECTOR_TOLERANCE),
                    "bound_lo": lo,
                    "bound_hi": hi,
                    "bound_ok": lo - VECTOR_TOLERANCE <= new_value <= hi + VECTOR_TOLERANCE,
                }
            )

        # revenue_growth is a scenario variable, excluded from intensity and
        # fixed at exact zero for every main candidate.
        if abs(_number(old_vector, "revenue_growth")) > VECTOR_TOLERANCE:
            raise ActionNormalizationConflict(
                f"{candidate} has non-zero revenue_growth in the old runtime library"
            )
        new_vector["action__revenue_growth"] = 0.0
        new_active = _active_dimensions(new_vector)
        if old_active != new_active:
            raise ActionNormalizationConflict(
                f"Active-dimension set changed for {candidate}: old={old_active} new={new_active}"
            )
        for dimension in MANAGERIAL_ACTION_DIMENSIONS:
            if _sign(_number(old_vector, dimension)) != _sign(_number(new_vector, dimension)):
                raise ActionNormalizationConflict(
                    f"Sign changed for {candidate}:{dimension}"
                )

        new_intensity = standardized_action_intensity(new_vector, scales)
        if new_intensity is None or abs(new_intensity - target) > INTENSITY_TOLERANCE:
            raise ActionNormalizationConflict(
                f"Target intensity mismatch for {candidate}: measured={new_intensity} target={target}"
            )
        normalized_fixed[candidate] = new_vector
        candidate_rows.append(
            {
                "candidate": candidate,
                "family": FAMILY_BY_CANDIDATE[candidate],
                "old_intensity": float(old_intensity),
                "target_intensity": target,
                "scale_factor": float(scale_factor),
                "new_intensity": float(new_intensity),
                "active_dimension_count": len(old_active),
                "revenue_growth": 0.0,
                "bound_conflict_count": 0,
            }
        )

    if bound_conflicts:
        raise ActionNormalizationConflict(
            "ACTION_NORMALIZATION_BOUND_CONFLICT: "
            + json.dumps(bound_conflicts, ensure_ascii=False, sort_keys=True)
        )

    # Pairwise tolerance-based duplicate guard across all ten action columns.
    action_columns = [
        *(f"action__{d}" for d in MANAGERIAL_ACTION_DIMENSIONS),
        "action__revenue_growth",
    ]
    names = list(normalized_fixed)
    duplicates: list[dict[str, str]] = []
    for i, left in enumerate(names):
        left_vec = normalized_fixed[left]
        for right in names[i + 1 :]:
            right_vec = normalized_fixed[right]
            if all(
                abs(float(left_vec.get(col, 0.0) or 0.0) - float(right_vec.get(col, 0.0) or 0.0))
                <= VECTOR_TOLERANCE
                for col in action_columns
            ):
                duplicates.append({"left": left, "right": right})
    if duplicates:
        raise ActionNormalizationConflict(f"Duplicate normalized candidate vectors: {duplicates}")

    zero_non_noop = [
        name
        for name in names
        if name != "A0"
        and standardized_action_intensity(normalized_fixed[name], scales) <= INTENSITY_TOLERANCE
    ]
    if zero_non_noop:
        raise ActionNormalizationConflict(f"Zero-vector non-noop candidates: {zero_non_noop}")

    normalized["candidate_action_contract_version"] = CANDIDATE_ACTION_CONTRACT_VERSION
    normalized["candidate_intensity_contract"] = {
        "metric": "sum_abs_action_over_stage2_p50_scale_9d",
        "scale_hash": scales_payload["scale_hash"],
        "base_template_hash": str(base_template_hash),
        "old_p50_runtime_hash": str(old_runtime_hash),
        "target_intensity_by_candidate": dict(TARGET_INTENSITY_BY_CANDIDATE),
        "scaling_rule": "single_scalar_target_over_current_intensity_per_non_noop_candidate",
        "revenue_growth_policy": "exact_zero_excluded_from_intensity",
        "silent_clipping_allowed": False,
        "oracle_used_to_choose_targets": False,
    }
    validation = {
        "status": "PASS",
        "candidate_action_contract_version": CANDIDATE_ACTION_CONTRACT_VERSION,
        "candidate_count": len(names),
        "unique_action_vector_count": len(names),
        "duplicate_vector_count": 0,
        "zero_non_noop_count": 0,
        "bound_conflict_count": 0,
        "sign_preservation": True,
        "active_axis_preservation": True,
        "revenue_growth_exact_zero": True,
        "target_intensity_tolerance": INTENSITY_TOLERANCE,
        "target_intensity_all_pass": True,
        "p50_scale_hash": scales_payload["scale_hash"],
        "base_template_hash": str(base_template_hash),
        "old_p50_runtime_hash": str(old_runtime_hash),
    }
    return normalized, candidate_rows, dimension_rows, validation


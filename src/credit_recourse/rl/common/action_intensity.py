"""Canonical P50-normalized managerial action-intensity contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ACTION_INTENSITY_SCALE_SCHEMA_VERSION = "p50_normalized_action_intensity_scales_v1"
ACTION_INTENSITY_METRIC_NAME = "p50_normalized_action_intensity"
MANAGERIAL_ACTION_DIMENSIONS: tuple[str, ...] = (
    "ppe_pct",
    "inv_turnover_chg",
    "ar_turnover_chg",
    "ap_turnover_chg",
    "short_debt_pct",
    "long_debt_pct",
    "bond_pct",
    "cogs_ratio_chg",
    "sga_ratio_chg",
)
EXCLUDED_SCENARIO_DIMENSIONS: tuple[str, ...] = ("revenue_growth",)
FRESH_ACTION_INTENSITY_BUDGETS: dict[str, float | None] = {
    "LOW": 1.0,
    "MEDIUM": 2.0,
    "HIGH": 3.0,
    "UNCONSTRAINED": None,
}
REVENUE_GROWTH_BUDGETED_POLICY = "generation_time_exact_zero_scenario_variable"


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    data = dict(payload)
    data.pop("scale_hash", None)
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_action_intensity_scale_artifact(
    *,
    fit_year_max: int,
    population: str,
    population_rows: int,
    directional_p50_by_dimension: Mapping[str, Mapping[str, Any]],
    source_statistics_hash: str,
) -> dict[str, Any]:
    """Derive one symmetric scale from Stage2's already-computed directional P50s.

    A dimension with both positive and negative candidate directions uses the
    median of those two pre-tier P50 magnitudes. This keeps cost symmetric while
    ensuring the budget statistic is derived from exactly the same directional
    source statistics used by candidate calibration.
    """
    dimensions: dict[str, Any] = {}
    for dim in MANAGERIAL_ACTION_DIMENSIONS:
        prefixed = f"action__{dim}"
        records = dict(directional_p50_by_dimension.get(prefixed) or {})
        values: list[float] = []
        for direction, record in sorted(records.items()):
            value = record.get("p50") if isinstance(record, Mapping) else record
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Missing/non-numeric Stage2 directional P50 for {dim}:{direction}")
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"Invalid Stage2 directional P50 for {dim}:{direction}: {value}")
            values.append(value)
        if not values:
            raise ValueError(f"No Stage2 candidate-direction P50 statistic for managerial dimension {dim}")
        values.sort()
        mid = len(values) // 2
        scale = values[mid] if len(values) % 2 else 0.5 * (values[mid - 1] + values[mid])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid derived action-intensity scale for {dim}: {scale}")
        dimensions[dim] = {
            "scale": float(scale),
            "directional_p50_source": records,
            "symmetric_cost": True,
        }
    payload: dict[str, Any] = {
        "schema_version": ACTION_INTENSITY_SCALE_SCHEMA_VERSION,
        "metric_name": ACTION_INTENSITY_METRIC_NAME,
        "fit_year_max": int(fit_year_max),
        "population": str(population),
        "population_rows": int(population_rows),
        "scale_definition": "median of Stage2 candidate-direction historical P50 magnitudes before candidate tier geometry; symmetric absolute cost at application",
        "source_statistics_hash": str(source_statistics_hash),
        "included_dimensions": list(MANAGERIAL_ACTION_DIMENSIONS),
        "excluded_dimensions": list(EXCLUDED_SCENARIO_DIMENSIONS),
        "dimensions": dimensions,
        "scale_by_dimension": {dim: rec["scale"] for dim, rec in dimensions.items()},
        "revenue_growth_policy": REVENUE_GROWTH_BUDGETED_POLICY,
        "predefined_fresh_budgets": FRESH_ACTION_INTENSITY_BUDGETS,
    }
    payload["scale_hash"] = _canonical_hash(payload)
    return payload


def validate_action_intensity_scale_artifact(data: Mapping[str, Any]) -> dict[str, Any]:
    artifact = dict(data)
    if artifact.get("schema_version") != ACTION_INTENSITY_SCALE_SCHEMA_VERSION:
        raise ValueError(f"Unexpected action-intensity scale schema: {artifact.get('schema_version')!r}")
    if artifact.get("metric_name") != ACTION_INTENSITY_METRIC_NAME:
        raise ValueError(f"Unexpected action-intensity metric: {artifact.get('metric_name')!r}")
    expected_hash = _canonical_hash(artifact)
    if artifact.get("scale_hash") != expected_hash:
        raise ValueError("action_intensity_scales.json scale_hash does not match canonical contents")
    scales = dict(artifact.get("scale_by_dimension") or {})
    if set(scales) != set(MANAGERIAL_ACTION_DIMENSIONS):
        raise ValueError(
            f"Action-intensity scale dimensions mismatch: expected={list(MANAGERIAL_ACTION_DIMENSIONS)} got={sorted(scales)}"
        )
    for dim, value in scales.items():
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Action-intensity scale must be positive finite: {dim}={value}")
        scales[dim] = value
    artifact["scale_by_dimension"] = scales
    return artifact


def load_action_intensity_scale_artifact(project_root: Path) -> tuple[dict[str, Any], Path]:
    path = (
        Path(project_root).resolve()
        / "data"
        / "final_freeze"
        / "stage2_candidate_projection"
        / "action_intensity_scales.json"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing canonical P50 action-intensity scale artifact: {path}. Run corrected Stage2 first."
        )
    return validate_action_intensity_scale_artifact(json.loads(path.read_text(encoding="utf-8"))), path


def standardized_action_intensity(
    vector: Mapping[str, Any] | None,
    scale_by_dimension: Mapping[str, Any],
) -> float | None:
    if vector is None:
        return None
    total = 0.0
    for dim in MANAGERIAL_ACTION_DIMENSIONS:
        raw = vector.get(dim, vector.get(f"action__{dim}", 0.0))
        try:
            value = float(raw) if raw is not None else 0.0
            scale = float(scale_by_dimension[dim])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0.0:
            return None
        total += abs(value) / scale
    return float(total)


def standardized_contributions(
    vector: Mapping[str, Any], scale_by_dimension: Mapping[str, Any]
) -> dict[str, float]:
    return {
        dim: abs(float(vector.get(dim, vector.get(f"action__{dim}", 0.0)) or 0.0))
        / float(scale_by_dimension[dim])
        for dim in MANAGERIAL_ACTION_DIMENSIONS
    }


def raw_l1_10d(vector: Mapping[str, Any] | None) -> float | None:
    if vector is None:
        return None
    total = 0.0
    for dim in (*MANAGERIAL_ACTION_DIMENSIONS, *EXCLUDED_SCENARIO_DIMENSIONS):
        try:
            value = float(vector.get(dim, vector.get(f"action__{dim}", 0.0)) or 0.0)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        total += abs(value)
    return float(total)

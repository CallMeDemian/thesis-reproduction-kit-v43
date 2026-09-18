"""Canonical representation and verifier helpers for semantic action v4."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

from credit_recourse.simulator.semantic_action_v4 import (
    ACTION_SEMANTIC_CONTRACT_VERSION_V4,
    SemanticActionV4,
    V4_ACTION_DIMENSIONS,
)


CANDIDATE_IDS_V4: tuple[str, ...] = (
    "A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2"
)
INTENSITY_TOLERANCE_V4 = 1.0e-8


def canonical_hash(payload: Mapping[str, Any]) -> str:
    data = dict(payload)
    data.pop("candidate_action_contract_hash", None)
    # Creation time is provenance, not scientific semantics.  Excluding it
    # makes re-materialization of identical frozen semantics hash-stable.
    data.pop("created_utc", None)
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def vector_from_candidate(contract: Mapping[str, Any], candidate_id: str) -> dict[str, float]:
    fixed = dict(contract.get("fixed_candidates") or {})
    if candidate_id not in fixed:
        raise KeyError(f"Unknown semantic-v4 candidate: {candidate_id}")
    row = dict(fixed[candidate_id])
    return {f"action__{dim}": float(row.get(f"action__{dim}", 0.0) or 0.0) for dim in V4_ACTION_DIMENSIONS}


def normalized_intensity(vector: Mapping[str, Any], scales: Mapping[str, Any]) -> float:
    total = 0.0
    for dim in V4_ACTION_DIMENSIONS:
        value = float(vector.get(f"action__{dim}", vector.get(dim, 0.0)) or 0.0)
        scale = float(scales[dim])
        if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid intensity input {dim}: value={value}, scale={scale}")
        total += abs(value) / scale
    return float(total)


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("candidate_action_contract_version") != ACTION_SEMANTIC_CONTRACT_VERSION_V4:
        raise ValueError("semantic-v4 contract version mismatch")
    if tuple(contract.get("candidate_ids") or ()) != CANDIDATE_IDS_V4:
        raise ValueError("semantic-v4 candidate order/count mismatch")
    if canonical_hash(contract) != contract.get("candidate_action_contract_hash"):
        raise ValueError("semantic-v4 contract hash mismatch")
    scales = dict(contract.get("normalization_scale_by_dimension") or {})
    if set(scales) != set(V4_ACTION_DIMENSIONS):
        raise ValueError("semantic-v4 intensity scale dimensions mismatch")
    vectors = [vector_from_candidate(contract, candidate) for candidate in CANDIDATE_IDS_V4]
    signatures = [tuple(round(v[f"action__{d}"], 14) for d in V4_ACTION_DIMENSIONS) for v in vectors]
    if len(set(signatures)) != len(signatures):
        raise ValueError("duplicate semantic-v4 candidate vector")
    rows = []
    for candidate, vector in zip(CANDIDATE_IDS_V4, vectors):
        intensity = normalized_intensity(vector, scales)
        target = 0.0 if candidate == "A0" else 1.0
        if abs(intensity - target) > INTENSITY_TOLERANCE_V4:
            raise ValueError(f"semantic-v4 intensity mismatch {candidate}: {intensity} vs {target}")
        action = SemanticActionV4.from_mapping(vector)
        rows.append({"candidate_id": candidate, "nominal_normalized_intensity": intensity, "action": action.to_dict()})
    if contract.get("oracle_used_to_define_or_adjust_vectors") is not False:
        raise ValueError("semantic-v4 contract must explicitly prohibit Oracle-defined vectors")
    return {
        "status": "PASS",
        "candidate_count": len(CANDIDATE_IDS_V4),
        "unique_vector_count": len(set(signatures)),
        "all_nonnoop_intensity_one": True,
        "oracle_used": False,
        "rows": rows,
    }


def action_space_from_contract(contract: Mapping[str, Any]):
    """Build the exact Stage2/4/5 ActionSpace from one v4 contract payload."""
    from credit_recourse.rl.common.actions import ActionSpace

    validate_contract(contract)
    columns = [f"action__{d}" for d in V4_ACTION_DIMENSIONS]
    raw_bounds = {
        "growth_capex_reduction_pct": (0.0, 1.0),
        "deleveraging_total_debt_pct": (0.0, 1.0),
        "refinancing_short_debt_pct": (0.0, 1.0),
        "inv_turnover_chg": (-3.0, 3.0),
        "ar_turnover_chg": (-3.0, 3.0),
        "ap_turnover_chg": (-3.0, 3.0),
        "cogs_ratio_chg": (-0.03, 0.03),
        "sga_ratio_chg": (-0.02, 0.02),
    }
    scales = dict(contract["normalization_scale_by_dimension"])
    families = dict(contract.get("family_by_candidate") or {})
    if set(families) != set(CANDIDATE_IDS_V4):
        raise ValueError("semantic-v4 family mapping must cover exactly the active candidates")
    return ActionSpace(
        columns=columns,
        bounds={f"action__{d}": raw_bounds[d] for d in V4_ACTION_DIMENSIONS},
        fixed_candidates={c: vector_from_candidate(contract, c) for c in CANDIDATE_IDS_V4},
        train_labels=list(CANDIDATE_IDS_V4),
        row_conditional_baselines=[],
        final_rl_label="C3_candidate_iql",
        scenario_candidates={},
        diagnostic_candidates={},
        candidate_library_hash=str(contract["candidate_action_contract_hash"]),
        final_action_contract_hash=str(contract["candidate_action_contract_hash"]),
        candidate_action_contract_version=ACTION_SEMANTIC_CONTRACT_VERSION_V4,
        intensity_scale_by_column={f"action__{d}": float(scales[d]) for d in V4_ACTION_DIMENSIONS},
        family_by_candidate=families,
    )

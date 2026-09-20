from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .common import ContractError, find_repo_root, load_json
from credit_recourse.contracts.runtime_assets import active_action_contract


DIMENSIONS = (
    "growth_capex_reduction_pct",
    "deleveraging_total_debt_pct",
    "refinancing_short_debt_pct",
    "inv_turnover_chg",
    "ar_turnover_chg",
    "ap_turnover_chg",
    "cogs_ratio_chg",
    "sga_ratio_chg",
)
CANDIDATES = ("A0", "DL", "RF", "CX", "WC1", "WC2", "OE", "MX1", "MX2")


@dataclass(frozen=True)
class AppliedAction:
    status: str
    proposed: dict[str, float] | None
    accepted: dict[str, float] | None
    itt: dict[str, float] | None
    reason: str


def load_action_contract(root: Path | None = None) -> dict[str, Any]:
    repo = root or find_repo_root()
    return load_json(active_action_contract(repo))


def _coerce_exact_vector(value: Mapping[str, Any]) -> dict[str, float]:
    if set(value) != set(DIMENSIONS):
        missing = sorted(set(DIMENSIONS) - set(value))
        extra = sorted(set(value) - set(DIMENSIONS))
        raise ContractError(f"Action must have exactly eight canonical keys; missing={missing}, extra={extra}")
    result: dict[str, float] = {}
    for key in DIMENSIONS:
        try:
            number = float(value[key])
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Action value is not numeric: {key}") from exc
        if not math.isfinite(number):
            raise ContractError(f"Action value is not finite: {key}")
        result[key] = number
    return result


def validate_free8(
    value: Mapping[str, Any], budget: str, root: Path | None = None, tolerance: float = 1e-9
) -> dict[str, float]:
    """Validate a proposed action without clipping, rescaling, or projection."""
    action = _coerce_exact_vector(value)
    contract = load_action_contract(root)
    for key, number in action.items():
        low, high = contract["action_bounds"][f"action__{key}"]
        if number < float(low) - tolerance or number > float(high) + tolerance:
            raise ContractError(f"Action bound violation: {key}={number}, expected [{low}, {high}]")
    if action["deleveraging_total_debt_pct"] > tolerance and action["refinancing_short_debt_pct"] > tolerance:
        raise ContractError("DL/RF mutual-exclusion violation")
    if budget == "B1":
        scale = contract["normalization_scale_by_dimension"]
        intensity = sum(abs(action[key]) / float(scale[key]) for key in DIMENSIONS)
        if intensity > 1.0 + tolerance:
            raise ContractError(f"B1 normalized intensity exceeds 1: {intensity}")
    elif budget != "BINF":
        raise ContractError(f"Unknown action budget: {budget}")
    return action


def candidate_action(candidate_id: str, root: Path | None = None) -> dict[str, float]:
    if candidate_id not in CANDIDATES:
        raise ContractError(f"Unknown candidate ID: {candidate_id}")
    contract = load_action_contract(root)
    raw = contract["fixed_candidates"][candidate_id]
    return {key: float(raw[f"action__{key}"]) for key in DIMENSIONS}


def validate_candidate9(candidate_id: str, root: Path | None = None) -> dict[str, float]:
    action = candidate_action(candidate_id, root)
    # Candidate vectors are frozen upstream.  Validate bounds/mutex directly;
    # B1 nominal intensity is a contract property and not reprojected here.
    contract = load_action_contract(root)
    for key, number in action.items():
        low, high = contract["action_bounds"][f"action__{key}"]
        if number < low - 1e-9 or number > high + 1e-9:
            raise ContractError(f"Frozen candidate bound violation: {candidate_id}/{key}")
    if action["deleveraging_total_debt_pct"] > 1e-9 and action["refinancing_short_debt_pct"] > 1e-9:
        raise ContractError(f"Frozen candidate DL/RF mutex violation: {candidate_id}")
    return action


def apply_itt_rule(
    proposed: Mapping[str, Any] | None,
    budget: str,
    *,
    terminal_completion: bool,
    infrastructure_failure: bool,
    mode: str = "free8",
    candidate_id: str | None = None,
    root: Path | None = None,
) -> AppliedAction:
    """Apply the preregistered ITT rule while preserving failure provenance."""
    if infrastructure_failure:
        return AppliedAction("UNRESOLVED_INFRA", None, None, None, "retry/resume; no ITT action assigned")
    if proposed is None:
        if terminal_completion:
            a0 = candidate_action("A0", root)
            return AppliedAction("CONFIRMED_UNUSABLE_COMPLETION", None, None, a0, "terminal unusable output maps to A0")
        return AppliedAction("UNRESOLVED_INFRA", None, None, None, "non-terminal missing output")
    try:
        if mode == "free8":
            accepted = validate_free8(proposed, budget, root)
        elif mode == "candidate9":
            if candidate_id is None:
                raise ContractError("Candidate9 response lacks a candidate_id")
            accepted = validate_candidate9(candidate_id, root)
            if any(
                abs(float(accepted[key]) - float(proposed[key])) > 1e-12
                for key in DIMENSIONS
            ):
                raise ContractError(
                    "Candidate9 proposed vector differs from its frozen candidate"
                )
        else:
            raise ContractError(f"Unknown action mode: {mode}")
    except ContractError as exc:
        if not terminal_completion:
            return AppliedAction("UNRESOLVED_INFRA", dict(proposed), None, None, f"retry/resume: {exc}")
        a0 = candidate_action("A0", root)
        return AppliedAction("CONFIRMED_UNUSABLE_COMPLETION", dict(proposed), None, a0, str(exc))
    return AppliedAction("ACCEPTED", dict(accepted), dict(accepted), dict(accepted), "strict validation passed")

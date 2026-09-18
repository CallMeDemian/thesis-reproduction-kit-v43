"""Versioned simulator factory shared by standalone and future Stage2/C2 users."""

from __future__ import annotations

from typing import Any, Mapping

from credit_recourse.rl.common.semantic_action_v4_contract import validate_contract
from credit_recourse.simulator.dl_interest_timing import (
    DLInterestTiming, DL_INTEREST_SEMANTIC_VERSION, DL_INTEREST_UNCAPPED_SEMANTIC_VERSION,
    DLFullYearInterestTiming, DL_INTEREST_FULL_YEAR_SEMANTIC_VERSION,
)
from credit_recourse.simulator.semantic_action_v4 import SemanticFinancialSimulatorV4


def interest_timing_contract(*, uncapped: bool = False, full_year: bool = False) -> dict[str, Any]:
    if full_year and not uncapped:
        raise ValueError("Full-year DL contract must be uncapped")
    return {
        "repayment_timing_fraction": 1.0 if full_year else 0.5,
        "effective_interest_saving_rate_cap": None if uncapped else 0.02,
        "interest_rate_source": {
            "short": "bp_eff.rate_short", "gross_ltd": "bp_eff.rate_long",
            "bond": "bp_eff.rate_bond",
        },
        "formula": ("sum(realized_repayment_j * annual_rate_j)" if full_year else
                    "sum(realized_repayment_j * 0.5 * annual_rate_j)" if uncapped else
                    "sum(realized_repayment_j * min(0.5 * annual_rate_j, 0.02))"),
        "interest_after_formula": "max(0, interest_without_dl_saving - total_dl_interest_saving)",
        "realized_repayment_only": True,
        "repayment_capacity_excludes_same_action_interest_saving": True,
        "applies_to_candidates": ["DL", "MX1", "MX2"],
        "application_rule": "same primitive for every realized_repayment > 0; no candidate-ID branch",
        "reward_bonus": False,
        "oracle_inputs_used": False,
    }


def validate_timing_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    result = validate_contract(contract)
    version = contract.get("simulator_action_semantic_contract_version")
    if version not in {DL_INTEREST_SEMANTIC_VERSION, DL_INTEREST_UNCAPPED_SEMANTIC_VERSION, DL_INTEREST_FULL_YEAR_SEMANTIC_VERSION}:
        raise ValueError("DL simulator semantic version mismatch")
    full_year = version == DL_INTEREST_FULL_YEAR_SEMANTIC_VERSION
    uncapped = version in {DL_INTEREST_UNCAPPED_SEMANTIC_VERSION, DL_INTEREST_FULL_YEAR_SEMANTIC_VERSION}
    if contract["operators"]["DL"].get("interest_saving") != interest_timing_contract(uncapped=uncapped, full_year=full_year):
        raise ValueError("DL interest timing contract mismatch")
    if uncapped and "capped" in contract["operators"]["DL"].get("repayment_timing", "").replace("uncapped", ""):
        raise ValueError("Stale capped timing description in v4.2 contract")
    if "no within-year" in contract["operators"]["DL"].get("repayment_timing", ""):
        raise ValueError("Stale closing-date-only DL timing in v4.1 contract")
    if full_year and "half-year" in contract["operators"]["DL"].get("repayment_timing", ""):
        raise ValueError("Stale half-year timing in v4.3 contract")
    expected = set(interest_timing_contract()["applies_to_candidates"])
    actual = {name for name, vector in contract["fixed_candidates"].items()
              if float(vector.get("action__deleveraging_total_debt_pct", 0.0)) > 0.0}
    if expected != actual:
        raise ValueError("Declared DL candidates do not match the primitive vectors")
    return result


def simulator_from_contract(contract: Mapping[str, Any]) -> SemanticFinancialSimulatorV4:
    """No policy label is accepted: C2-selected vectors use the same simulator."""
    version = contract.get("simulator_action_semantic_contract_version")
    if version == DL_INTEREST_FULL_YEAR_SEMANTIC_VERSION:
        validate_timing_contract(contract)
        timing = DLFullYearInterestTiming()
    elif version in {DL_INTEREST_SEMANTIC_VERSION, DL_INTEREST_UNCAPPED_SEMANTIC_VERSION}:
        validate_timing_contract(contract)
        timing = DLInterestTiming(effective_interest_saving_rate_cap=(
            None if version == DL_INTEREST_UNCAPPED_SEMANTIC_VERSION else 0.02))
    elif version == "simulator_action_semantic_contract_v4":
        validate_contract(contract)
        if "interest_saving" in contract["operators"]["DL"]:
            raise ValueError("v4 contract cannot silently contain v4.1 timing")
        timing = None
    else:
        raise ValueError(f"Unsupported simulator semantic version: {version}")
    return SemanticFinancialSimulatorV4(
        dl_allocation=contract["operators"]["DL"]["allocation"],
        rf_allocation=contract["operators"]["RF"]["allocation"],
        contract_hash=contract["candidate_action_contract_hash"],
        intensity_scale_by_dimension=contract["normalization_scale_by_dimension"],
        liquidity_reserve_cash_to_cogs=contract["operators"]["DL"]["liquidity_reserve_cash_to_cogs"],
        accounting_incremental_residual_abs_tolerance=contract["accounting_closing_contract"]["incremental_residual_abs_tolerance"],
        accounting_incremental_residual_rel_tolerance=contract["accounting_closing_contract"]["incremental_residual_rel_tolerance"],
        liquidity_cash_abs_tolerance=contract["cash_closing_contract"]["negative_cash_abs_tolerance"],
        dl_interest_timing=timing,
    )

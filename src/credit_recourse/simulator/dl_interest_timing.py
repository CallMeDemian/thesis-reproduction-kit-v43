"""Frozen, candidate-independent DL interest timing; no evaluation inputs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from credit_recourse.simulator.business_plan import BusinessPlan


DL_INTEREST_SEMANTIC_VERSION = "simulator_action_semantic_contract_v4_1_dl_interest_timing"
DL_INTEREST_PRODUCER_VERSION = "semantic_financial_simulator_v4_1_dl_interest_timing_v1"
DL_INTEREST_UNCAPPED_SEMANTIC_VERSION = "simulator_action_semantic_contract_v4_2_dl_interest_uncapped"
DL_INTEREST_UNCAPPED_PRODUCER_VERSION = "semantic_financial_simulator_v4_2_dl_interest_uncapped_v1"
DL_INTEREST_FULL_YEAR_SEMANTIC_VERSION = "simulator_action_semantic_contract_v4_3_dl_interest_full_year"
DL_INTEREST_FULL_YEAR_PRODUCER_VERSION = "semantic_financial_simulator_v4_3_dl_interest_full_year_v1"


@dataclass(frozen=True)
class DLInterestTiming:
    repayment_timing_fraction: float = 0.5
    # The default preserves explicit legacy v4.1 reproduction. The v4.2
    # factory selects None: half-year exposure with no saving-rate cap.
    effective_interest_saving_rate_cap: float | None = 0.02

    def __post_init__(self) -> None:
        # This is a frozen semantic contract, not an evaluation-tunable knob.
        if self.repayment_timing_fraction != 0.5:
            raise ValueError("v4.1 repayment_timing_fraction is frozen at 0.5")
        if self.effective_interest_saving_rate_cap not in (None, 0.02):
            raise ValueError("interest cap is frozen: legacy v4.1=0.02, v4.2=None")

    @property
    def semantic_version(self) -> str:
        return (DL_INTEREST_UNCAPPED_SEMANTIC_VERSION if self.effective_interest_saving_rate_cap is None
                else DL_INTEREST_SEMANTIC_VERSION)

    @property
    def producer_version(self) -> str:
        return (DL_INTEREST_UNCAPPED_PRODUCER_VERSION if self.effective_interest_saving_rate_cap is None
                else DL_INTEREST_PRODUCER_VERSION)


@dataclass(frozen=True)
class DLFullYearInterestTiming(DLInterestTiming):
    """v4.3 only: full annual exposure, no interest-saving rate cap."""
    repayment_timing_fraction: float = 1.0
    effective_interest_saving_rate_cap: float | None = None

    def __post_init__(self) -> None:
        if self.repayment_timing_fraction != 1.0 or self.effective_interest_saving_rate_cap is not None:
            raise ValueError("v4.3 timing is frozen at full-year exposure with no rate cap")

    @property
    def semantic_version(self) -> str:
        return DL_INTEREST_FULL_YEAR_SEMANTIC_VERSION

    @property
    def producer_version(self) -> str:
        return DL_INTEREST_FULL_YEAR_PRODUCER_VERSION


def dl_interest_savings(
    repayment_by_stack: Mapping[str, float],
    bp: BusinessPlan,
    interest_without_dl_saving: float,
    timing: DLInterestTiming,
) -> dict[str, float]:
    """Apply versioned interest exposure to already constrained repayments.

    Capacity/allocation cannot be computed here: the function accepts neither
    cash nor a requested repayment nor a candidate identifier.
    """
    if set(repayment_by_stack) != {"short", "gross_ltd", "bond"}:
        raise ValueError("DL savings require exactly short/gross_ltd/bond repayment stacks")
    before = float(interest_without_dl_saving)
    if not math.isfinite(before) or before < 0.0:
        raise ValueError("pre-saving interest must be finite and nonnegative")
    rates = {"short": bp.rate_short, "gross_ltd": bp.rate_long, "bond": bp.rate_bond}
    result = {"interest_expense_before_dl_saving": before}
    total_repayment = 0.0
    total_saving = 0.0
    for stack, raw_rate in rates.items():
        repayment, annual_rate = float(repayment_by_stack[stack]), float(raw_rate)
        if not math.isfinite(repayment) or repayment < 0.0:
            raise ValueError(f"Invalid realized repayment for {stack}: {repayment}")
        if not math.isfinite(annual_rate) or annual_rate < 0.0:
            raise ValueError(f"Invalid BusinessPlan rate for {stack}: {annual_rate}")
        rate_name = "long" if stack == "gross_ltd" else stack
        effective = timing.repayment_timing_fraction * annual_rate
        if timing.effective_interest_saving_rate_cap is not None:
            effective = min(effective, timing.effective_interest_saving_rate_cap)
        saving = repayment * effective
        result[f"repayment_{stack}"] = repayment
        result[f"annual_rate_{rate_name}"] = annual_rate
        result[f"effective_saving_rate_{rate_name}"] = effective
        result[f"interest_saving_{stack}"] = saving
        total_repayment += repayment
        total_saving += saving
    after = max(0.0, before - total_saving)
    result.update({
        "total_dl_interest_saving": total_saving,
        "interest_expense_after_dl_saving": after,
        "applied_dl_interest_saving": before - after,
        "interest_saving_floor_adjustment": total_saving - (before - after),
        "interest_saving_to_realized_repayment_ratio": (
            total_saving / total_repayment if total_repayment > 0.0 else 0.0
        ),
    })
    return result

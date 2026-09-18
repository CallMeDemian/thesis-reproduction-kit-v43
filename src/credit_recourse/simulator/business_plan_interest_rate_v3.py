"""Production BP borrowing-rate v3.

This module deliberately has no dependency on the preliminary v2 producer.
It is the only rate producer allowed by the v4.3/BP-rate-v3 lineage: pure
interest (U01B550010000) divided by average IBD, with an as-of-only
exposure-weighted fallback.  It neither reads ratings nor financial_cost.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Mapping

from credit_recourse.simulator.business_plan import BusinessPlan

RATE_VERSION = "business_plan_interest_rate_v3_exposure_weighted_bok_2007_0636"
RATE_CONTRACT_VERSION = RATE_VERSION
BOK_2007_RATE = 0.0636
PRIMARY_NUMERATOR_U_CODE = "U01B550010000"
DEBT_FIELDS = ("short_term_debt", "current_portion_long_debt", "long_term_debt", "bonds")


@dataclass(frozen=True)
class BorrowingRateResult:
    blended_rate: float
    source: str
    valid_observation_count: int
    interest_sum: float | None
    average_ibd_sum: float | None


@dataclass(frozen=True)
class BorrowingRateLineage:
    firm_id: str
    base_year: int
    result: BorrowingRateResult
    source_fiscal_years: tuple[int, ...]
    maximum_source_year: int | None
    numerator_u_code: str
    external_contemporaneous_initialization_fallback: bool

    def as_dict(self) -> dict[str, object]:
        r = self.result
        return {
            "business_plan_rate": r.blended_rate,
            "business_plan_rate_short": r.blended_rate,
            "business_plan_rate_long": r.blended_rate,
            "business_plan_rate_bond": r.blended_rate,
            "business_plan_rate_source": r.source,
            "business_plan_rate_fallback_flag": r.source != "firm_exposure_weighted",
            "business_plan_rate_source_fiscal_years": list(self.source_fiscal_years),
            "business_plan_rate_valid_observation_count": r.valid_observation_count,
            "business_plan_rate_interest_sum": r.interest_sum,
            "business_plan_rate_average_ibd_sum": r.average_ibd_sum,
            "business_plan_rate_numerator_u_code": self.numerator_u_code,
            "business_plan_rate_maximum_source_year": self.maximum_source_year,
            "business_plan_rate_contract_version": RATE_CONTRACT_VERSION,
            "external_contemporaneous_initialization_fallback": self.external_contemporaneous_initialization_fallback,
        }


def _key(value: object) -> str:
    return str(value).strip().zfill(6)


def _finite_nonnegative(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def _ibd(row: Mapping[str, object]) -> float | None:
    """Require every IBD stack to be observed; missing is never coerced to zero."""
    values = [_finite_nonnegative(row.get(field)) for field in DEBT_FIELDS]
    return None if any(v is None for v in values) else float(sum(values))


def observed_rate_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Build pure-interest/consecutive-average-IBD observations, conflict-invalid."""
    ordered = sorted(rows, key=lambda r: (_key(r["firm_id"]), int(r["fiscal_year"])))
    prior: dict[str, tuple[int, float]] = {}
    result: list[dict[str, object]] = []
    for row in ordered:
        firm_id = _key(row["firm_id"])
        year = int(row["fiscal_year"])
        closing = _ibd(row)
        opening = prior.get(firm_id)
        primary = _finite_nonnegative(row.get(PRIMARY_NUMERATOR_U_CODE))
        conflict = bool(row.get("interest_source_conflict", False))
        reason: str | None = None
        if conflict:
            reason = "primary_numerator_source_conflict"
        elif primary is None:
            reason = "missing_primary_pure_interest_expense"
        elif opening is None or opening[0] != year - 1:
            reason = "missing_consecutive_opening_IBD"
        elif closing is None:
            reason = "invalid_closing_IBD"
        else:
            average = (opening[1] + closing) / 2.0
            if not math.isfinite(average) or average <= 0.0:
                reason = "nonpositive_average_IBD"
        entry: dict[str, object] = {
            "firm_id": firm_id, "fiscal_year": year, "pure_interest_expense": primary,
            "opening_IBD": None if opening is None else opening[1], "closing_IBD": closing,
            "average_IBD": None, "raw_rate": None, "valid": reason is None,
            "invalid_reason": reason, "numerator_u_code": PRIMARY_NUMERATOR_U_CODE,
        }
        if reason is None:
            entry["average_IBD"] = average
            entry["raw_rate"] = primary / average  # type: ignore[operator]
        result.append(entry)
        if closing is not None:
            prior[firm_id] = (year, closing)
    return result


def select_rate(*, base_year: int, lookback_rows: Iterable[Mapping[str, object]],
                asof_valid_rows: Iterable[Mapping[str, object]]) -> BorrowingRateResult:
    """Select from latest firm observations, BOK-2007, then as-of global history."""
    firm = [r for r in lookback_rows if bool(r.get("valid")) and int(r["fiscal_year"]) <= base_year]
    if firm:
        interest = sum(float(r["pure_interest_expense"]) for r in firm)
        exposure = sum(float(r["average_IBD"]) for r in firm)
        if exposure > 0.0:
            return BorrowingRateResult(interest / exposure, "firm_exposure_weighted", len(firm), interest, exposure)
    if int(base_year) == 2007:
        return BorrowingRateResult(BOK_2007_RATE, "external_contemporaneous_initialization_BOK_2007_all_industry", 0, None, None)
    historic = [r for r in asof_valid_rows if bool(r.get("valid")) and int(r["fiscal_year"]) <= base_year]
    if not historic:
        raise ValueError("No historically available borrowing-rate fallback")
    interest = sum(float(r["pure_interest_expense"]) for r in historic)
    exposure = sum(float(r["average_IBD"]) for r in historic)
    if not math.isfinite(exposure) or exposure <= 0.0:
        raise ValueError("Historical fallback has no positive exposure")
    return BorrowingRateResult(interest / exposure, "historical_global_exposure_weighted", len(historic), interest, exposure)


class BusinessPlanRateResolverV3:
    """Shared, as-of-only resolver for BP substrate, Stage2, standalone and Stage6."""
    def __init__(self, source_rows: Iterable[Mapping[str, object]]):
        self.observations = observed_rate_rows(source_rows)
        self._by_firm: dict[str, list[dict[str, object]]] = {}
        self._asof: dict[int, list[dict[str, object]]] = {}
        for row in self.observations:
            self._by_firm.setdefault(str(row["firm_id"]), []).append(row)
        valid = [r for r in self.observations if bool(r["valid"])]
        for year in sorted({int(r["fiscal_year"]) for r in self.observations}):
            self._asof[year] = [r for r in valid if int(r["fiscal_year"]) <= year]

    def resolve(self, firm_id: object, base_year: int) -> BorrowingRateLineage:
        key = _key(firm_id)
        own = [r for r in self._by_firm.get(key, []) if int(r["fiscal_year"]) <= int(base_year)]
        latest_three = [r for r in own if bool(r["valid"])][-3:]
        asof = self._asof.get(int(base_year), [r for r in self.observations if bool(r["valid"]) and int(r["fiscal_year"]) <= int(base_year)])
        selected = select_rate(base_year=int(base_year), lookback_rows=latest_three, asof_valid_rows=asof)
        if selected.source == "firm_exposure_weighted":
            years = tuple(int(r["fiscal_year"]) for r in latest_three)
        elif selected.source == "historical_global_exposure_weighted":
            years = tuple(sorted({int(r["fiscal_year"]) for r in asof}))
        else:
            years = ()
        return BorrowingRateLineage(key, int(base_year), selected, years, max(years) if years else None,
                                    PRIMARY_NUMERATOR_U_CODE, selected.source.startswith("external_"))

    def apply(self, plan: BusinessPlan, firm_id: object, base_year: int) -> tuple[BusinessPlan, BorrowingRateLineage]:
        lineage = self.resolve(firm_id, base_year)
        rate = lineage.result.blended_rate
        return replace(plan, rate_short=rate, rate_long=rate, rate_bond=rate), lineage

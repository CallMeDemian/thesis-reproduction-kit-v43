"""Legacy BusinessPlan entrypoints. V4.3 construction belongs to its bundle."""
from __future__ import annotations

from typing import Iterable

from credit_recourse.simulator.business_plan import BusinessPlan, calibrate_business_plan
from credit_recourse.simulator.business_plan_interest_rate_v3 import BusinessPlanRateResolverV3
from credit_recourse.simulator.business_plan_interest_rate_v4 import BusinessPlanRateResolverV4


CALIBRATED_BP_RATE_V4_MODE = "calibrated_bp_rate_v4"


def build_business_plan(
    mode: str,
    history: Iterable,
    *,
    rating_grade: str | None = None,
    firm_id: object | None = None,
    base_year: int | None = None,
    rate_resolver: BusinessPlanRateResolverV3 | BusinessPlanRateResolverV4 | None = None,
) -> tuple[BusinessPlan, object | None]:
    """Construct a plan once; V4 never falls back to V3/default rates."""
    history = list(history)
    if mode == "default":
        return BusinessPlan(), None
    if mode == "calibrated":
        return (calibrate_business_plan(history, grade=rating_grade) if history else BusinessPlan()), None
    if mode == "calibrated_bp_rate_v3":
        if not isinstance(rate_resolver, BusinessPlanRateResolverV3) or firm_id is None or base_year is None:
            raise ValueError("calibrated_bp_rate_v3 requires a V3 resolver and firm/base-year key")
        base = calibrate_business_plan(history, grade=rating_grade) if history else BusinessPlan()
        return rate_resolver.apply(base, firm_id, int(base_year))
    if mode == CALIBRATED_BP_RATE_V4_MODE:
        raise ValueError("V4.3 plan assembly requires V43ProductionSimulationBundle.build_plan")
    raise ValueError(f"Unsupported sim_business_plan_mode: {mode}")


"""
BusinessPlan — Simulator의 default 1년 사업계획.

Decision 1 (넓은 status quo) + firm 과거 평균 하이브리드:
- 매출 성장률 0% (action으로 변경 가능)
- 비용 비율 = firm 과거 평균
- CAPEX = firm 과거 평균 capex/revenue ratio
- 회전율 = firm 과거 평균
- 배당 = firm 과거 평균 payout ratio
"""
from dataclasses import dataclass, asdict
from typing import Optional, List
import statistics

from credit_recourse.simulator.firm_state import FirmState


# ---------------------------------------------------------------------
# Constants — Decision 3, 5, 11, 12 ; 등급 spread (Decision 4 확장)
# ---------------------------------------------------------------------
DEFAULT_TAX_RATE = 0.22
DEFAULT_DEPRECIATION_RATE = 0.08         # PP&E 대비
DEFAULT_AMORTIZATION_RATE = 0.10         # 무형자산 대비
DEFAULT_RECLASS_RATE = 0.10              # 장기차입금 → 유동성장기부채 매년 비율

# 등급 spread (annual %p added to base rate)
RATING_SPREAD = {
    "AAA": 0.000, "AA": 0.003, "A": 0.007,
    "BBB": 0.015, "BB": 0.030, "B": 0.050, "C": 0.080, "D": 0.100,
}

DEFAULT_RATES = {
    "short": 0.030,  # 단기차입금
    "long": 0.040,   # 장기차입금
    "bond": 0.045,   # 사채
}


def grade_to_spread(grade: Optional[str]) -> float:
    """등급 → spread 변환. 결측은 BBB로."""
    if grade is None:
        return RATING_SPREAD["BBB"]
    base = grade.rstrip("+-")
    return RATING_SPREAD.get(base, RATING_SPREAD["BBB"])


def effective_rate(debt_type: str, grade: Optional[str]) -> float:
    """차입 종류 + 등급 → 실효이자율."""
    return DEFAULT_RATES[debt_type] + grade_to_spread(grade)


# ---------------------------------------------------------------------
# BusinessPlan
# ---------------------------------------------------------------------
@dataclass
class BusinessPlan:
    """
    Default 1년 사업계획.
    값들은 firm 과거 평균에서 calibrate 후 action으로 perturb.
    """
    # 손익 비율 (대부분 매출 대비)
    revenue_growth: float = 0.0          # default 0% (status quo)
    cogs_ratio: float = 0.70             # COGS / Revenue
    sga_ratio: float = 0.15              # SG&A / Revenue
    non_op_income_ratio: float = 0.005   # (이자수익 + 배당수익) / Revenue
    # Production always supplies the A0 source-calibrated amount.
    # This zero only defines an explicit synthetic plan, never a data fallback.
    non_interest_financial_cost: float = 0.0
    tax_rate: float = DEFAULT_TAX_RATE

    # 회전율 (회 단위; days = 365/회전율)
    ar_turnover: float = 6.0             # 매출 / 평균매출채권
    inv_turnover: float = 8.0            # 매출원가 / 평균재고
    ap_turnover: float = 6.0             # 매출원가 / 평균매입채무

    # 자본 활동
    capex_to_revenue: float = 0.05       # 정기 CAPEX 비율
    depreciation_rate: float = DEFAULT_DEPRECIATION_RATE
    amortization_rate: float = DEFAULT_AMORTIZATION_RATE
    dividend_payout: float = 0.20        # net_income 대비 (양수일 때만)

    # 차입 활동 default
    debt_repayment_ratio: float = 0.0    # 행동 없을 때 차입 변동 0
    reclass_rate: float = DEFAULT_RECLASS_RATE

    # 이자율 (등급별 spread는 grade로 결정)
    rate_short: float = DEFAULT_RATES["short"]
    rate_long: float = DEFAULT_RATES["long"]
    rate_bond: float = DEFAULT_RATES["bond"]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------
# Calibration — firm 과거 데이터에서 BusinessPlan default 추출
# ---------------------------------------------------------------------
def _safe_div(a, b, default=None):
    if a is None or b is None or b == 0:
        return default
    return a / b


def _avg(xs: list, default=None):
    valid = [x for x in xs if x is not None]
    if not valid:
        return default
    return statistics.mean(valid)


NON_RATE_CALIBRATION_VERSION = "business_plan_non_rate_v2_observed_zero_preserved"


def _mean_or_default(values, default):
    """Use a fallback only when no ratio was observed; zero is a valid mean."""
    value = _avg(values)
    return default if value is None else value


def calibrate_business_plan_non_rate(firm_history: List[FirmState]) -> BusinessPlan:
    """Calibrate every BusinessPlan field except borrowing rates.

    This is the only calibration routine used by the v4 production seam.  It
    deliberately contains no financial_cost, grade spread, or debt-stack rate
    logic.
    """
    if not firm_history:
        return BusinessPlan(rate_short=0.0, rate_long=0.0, rate_bond=0.0)
    cogs_ratios = [_safe_div(s.cogs, s.revenue) for s in firm_history]
    sga_ratios = [_safe_div(s.sga, s.revenue) for s in firm_history]
    non_op_ratios = [_safe_div((s.interest_income or 0) + (s.dividend_income or 0), s.revenue) for s in firm_history]
    ar_turnovers = [_safe_div(s.revenue, s.receivables) for s in firm_history]
    inv_turnovers = [_safe_div(s.cogs, s.inventory) for s in firm_history]
    ap_turnovers = [_safe_div(s.cogs, s.payables) for s in firm_history]
    capex_ratios = [_safe_div(max(0.0, float(s.capex)) if s.capex is not None else None, s.revenue) for s in firm_history]
    dep_rates = [_safe_div(s.depreciation, s.ppe) for s in firm_history]
    amort_rates = [_safe_div(s.amortization, s.intangibles) for s in firm_history]
    payout_ratios = [s.cash_dividends / s.net_income for s in firm_history if s.net_income is not None and s.net_income > 0 and s.cash_dividends is not None]
    return BusinessPlan(
        revenue_growth=0.0,
        cogs_ratio=_mean_or_default(cogs_ratios, 0.70),
        sga_ratio=_mean_or_default(sga_ratios, 0.15),
        non_op_income_ratio=_mean_or_default(non_op_ratios, 0.005),
        tax_rate=DEFAULT_TAX_RATE,
        ar_turnover=_mean_or_default(ar_turnovers, 6.0),
        inv_turnover=_mean_or_default(inv_turnovers, 8.0),
        ap_turnover=_mean_or_default(ap_turnovers, 6.0),
        capex_to_revenue=_mean_or_default(capex_ratios, 0.05),
        depreciation_rate=_mean_or_default(dep_rates, DEFAULT_DEPRECIATION_RATE),
        amortization_rate=_mean_or_default(amort_rates, DEFAULT_AMORTIZATION_RATE),
        dividend_payout=_mean_or_default(payout_ratios, 0.0),
        rate_short=0.0, rate_long=0.0, rate_bond=0.0,
    )


def calibrate_business_plan(
    firm_history: List[FirmState],
    grade: Optional[str] = None,
    interest_rate_override: Optional[float] = None,
) -> BusinessPlan:
    """Legacy-compatible calibration wrapper.

    V4 callers must use :func:`calibrate_business_plan_non_rate` and receive
    their three rates only from BusinessPlanRateResolverV4.
    """
    bp = calibrate_business_plan_non_rate(firm_history)
    if interest_rate_override is not None:
        rate = float(interest_rate_override)
        return BusinessPlan(**{**bp.to_dict(), 'rate_short': rate, 'rate_long': rate, 'rate_bond': rate})
    eff_rates = []
    for s in firm_history:
        td = s.total_debt
        if td and td > 0 and s.financial_cost is not None:
            eff_rates.append(s.financial_cost / td)
    avg_eff_rate = _avg(eff_rates) if eff_rates else None
    if avg_eff_rate is not None:
        rates = (max(0.005, avg_eff_rate - 0.005), max(0.005, avg_eff_rate), max(0.005, avg_eff_rate + 0.005))
    else:
        rates = (effective_rate('short', grade), effective_rate('long', grade), effective_rate('bond', grade))
    return BusinessPlan(**{**bp.to_dict(), 'rate_short': rates[0], 'rate_long': rates[1], 'rate_bond': rates[2]})

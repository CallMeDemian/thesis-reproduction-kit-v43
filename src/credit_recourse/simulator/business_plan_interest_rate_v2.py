"""Frozen BP borrowing-rate v2: pure interest / average IBD plus BOK initialization."""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Iterable, Mapping

RATE_VERSION="business_plan_interest_rate_v2_exposure_weighted"
BOK_2007_RATE=.064
DEBT_FIELDS=("short_term_debt","current_portion_long_debt","long_term_debt","bonds")

@dataclass(frozen=True)
class BorrowingRateResult:
    blended_rate: float
    source: str
    valid_observation_count: int
    interest_sum: float | None
    average_ibd_sum: float | None

def _ibd(row: Mapping[str, object]) -> float | None:
    vals=[]
    for field in DEBT_FIELDS:
        value=row.get(field)
        if value is None or not math.isfinite(float(value)) or float(value)<0:return None
        vals.append(float(value))
    return sum(vals)

def observed_rate_rows(rows: Iterable[Mapping[str, object]]) -> list[dict]:
    """Create only valid pure-interest / consecutive-average-IBD observations."""
    panel=sorted(rows,key=lambda r:(str(r["firm_id"]),int(r["fiscal_year"])))
    prior={};out=[]
    for r in panel:
        key=str(r['firm_id']);year=int(r['fiscal_year']);closing=_ibd(r);opening=prior.get(key)
        interest=r.get('pure_interest_expense')
        reason=None
        if opening is None or opening['year'] != year-1:reason='missing_consecutive_opening_IBD'
        elif closing is None:reason='invalid_closing_IBD'
        elif interest is None or not math.isfinite(float(interest)) or float(interest)<0:reason='invalid_pure_interest_expense'
        else:
            avg=.5*(opening['ibd']+closing)
            if not math.isfinite(avg) or avg<=0:reason='nonpositive_average_IBD'
        if reason is None:
            out.append({'firm_id':key,'fiscal_year':year,'pure_interest_expense':float(interest),
                        'opening_IBD':opening['ibd'],'closing_IBD':closing,'average_IBD':avg,
                        'raw_rate':float(interest)/avg,'valid':True,'invalid_reason':None})
        else:out.append({'firm_id':key,'fiscal_year':year,'pure_interest_expense':interest,
                         'opening_IBD':None if opening is None else opening['ibd'],'closing_IBD':closing,
                         'average_IBD':None,'raw_rate':None,'valid':False,'invalid_reason':reason})
        if closing is not None:prior[key]={'year':year,'ibd':closing}
    return out

def select_rate(*, base_year:int, lookback_rows:Iterable[Mapping[str,object]],
                asof_valid_rows:Iterable[Mapping[str,object]]) -> BorrowingRateResult:
    valid=[r for r in lookback_rows if r.get('valid') and int(r['fiscal_year'])<=base_year]
    if valid:
        interest=sum(float(r['pure_interest_expense']) for r in valid); exposure=sum(float(r['average_IBD']) for r in valid)
        if exposure>0:return BorrowingRateResult(interest/exposure,'firm_exposure_weighted',len(valid),interest,exposure)
    if base_year==2007:
        return BorrowingRateResult(BOK_2007_RATE,'external_contemporaneous_initialization_BOK_2007_all_industry',0,None,None)
    historic=[r for r in asof_valid_rows if r.get('valid') and int(r['fiscal_year'])<=base_year]
    if not historic:raise ValueError('No historically available borrowing-rate fallback')
    interest=sum(float(r['pure_interest_expense']) for r in historic); exposure=sum(float(r['average_IBD']) for r in historic)
    if exposure<=0:raise ValueError('Historical fallback has no positive exposure')
    return BorrowingRateResult(interest/exposure,'historical_global_exposure_weighted',0,interest,exposure)

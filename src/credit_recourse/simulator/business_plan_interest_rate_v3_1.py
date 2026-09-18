"""BP-rate v3.1 support-validity contract; no rate-threshold predicates."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Mapping
from credit_recourse.simulator.business_plan_interest_rate_v3 import (
 BOK_2007_RATE, BorrowingRateResult, observed_rate_rows, _key,
)

RATE_VERSION='bp_rate_v3_1_support_validity'
SUPPORT_PREDICATE_VERSION='bp_rate_observation_support_v1_no_rate_or_flow_threshold'

@dataclass(frozen=True)
class SupportObservation:
 row: dict
 status: str
 reason: str
 evidence: str

def support_ledger(rows: Iterable[Mapping[str,object]]) -> list[dict]:
 """Classify before aggregation. No outcome/rate/flow threshold is inspected."""
 out=[]
 for obs in observed_rate_rows(rows):
  raw=dict(obs); interest=raw.get('pure_interest_expense'); opening=raw.get('opening_IBD'); closing=raw.get('closing_IBD')
  if raw.get('invalid_reason')=='primary_numerator_source_conflict': status,reason='source_invalid','conflict_invalid_primary_numerator'
  elif raw.get('invalid_reason') is not None: status,reason='source_invalid',str(raw['invalid_reason'])
  # Strong endpoint evidence only: both endpoint mapped IBD are confirmed zero while interest is positive.
  elif float(interest)>0 and float(opening)==0 and float(closing)==0: status,reason='unsupported_endpoint_exposure','positive_interest_with_confirmed_zero_opening_and_closing_IBD'
  else: status,reason='support_valid','validated_primary_pure_interest_and_positive_consecutive_AvgIBD'
  raw.update({'support_status':status,'support_reason_code':reason,'support_evidence':reason,
              'support_predicate_version':SUPPORT_PREDICATE_VERSION,'producer_repair_applied':False})
  out.append(raw)
 return out

class SupportValidRateResolver:
 def __init__(self, rows: Iterable[Mapping[str,object]]):
  self.ledger=support_ledger(rows); self.byfirm={}
  for r in self.ledger:self.byfirm.setdefault(_key(r['firm_id']),[]).append(r)
  self.asof={}
  for y in sorted({int(r['fiscal_year']) for r in self.ledger}):
   self.asof[y]=[r for r in self.ledger if r['support_status']=='support_valid' and int(r['fiscal_year'])<=y]
 def resolve(self,firm_id,base_year):
  firm=[r for r in self.byfirm.get(_key(firm_id),[]) if r['support_status']=='support_valid' and int(r['fiscal_year'])<=int(base_year)][-3:]
  if firm:
   i=sum(float(r['pure_interest_expense']) for r in firm); e=sum(float(r['average_IBD']) for r in firm)
   return BorrowingRateResult(i/e,'firm_exposure_weighted_support_valid',len(firm),i,e),firm
  if int(base_year)==2007:return BorrowingRateResult(BOK_2007_RATE,'external_contemporaneous_initialization_BOK_2007_all_industry',0,None,None),[]
  pool=self.asof.get(int(base_year),[])
  if not pool: raise ValueError('no_support_valid_asof_global_observation')
  i=sum(float(r['pure_interest_expense']) for r in pool); e=sum(float(r['average_IBD']) for r in pool)
  if e<=0:raise ValueError('nonpositive_support_valid_global_exposure')
  return BorrowingRateResult(i/e,'historical_global_exposure_weighted_support_valid',len(pool),i,e),pool

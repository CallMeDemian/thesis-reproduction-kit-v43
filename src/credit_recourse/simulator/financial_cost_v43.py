"""As-of broad financial cost decomposition; no Oracle inputs."""
from dataclasses import dataclass
from pathlib import Path
import math, statistics, json, hashlib
import pandas as pd
FINANCIAL_COST_VERSION='V43FinancialCost/1_broad_plus_primary_interest'
from credit_recourse.contracts.stage_paths import V43_STAGE2_RUNTIME_PATH

SOURCE_PATH=(V43_STAGE2_RUNTIME_PATH/'01_contract/financial_cost_sources_r2/history_financial_cost_sources.parquet').as_posix()
@dataclass(frozen=True)
class FinancialCostSources:
    rows: dict
    @classmethod
    def from_project_root(cls,root):
        path=Path(root)/SOURCE_PATH
        contract=json.loads((path.parent.parent/"r085_financial_cost_contract.json").read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=contract["source_hashes"][SOURCE_PATH]:
            raise ValueError("Financial-cost source digest changed")
        frame=pd.read_parquet(path)
        # The frozen audit join retains both identical canonical revenue copies.
        if 'revenue' not in frame and {'revenue_x','revenue_y'}.issubset(frame):
            if not (frame.revenue_x.eq(frame.revenue_y)|(frame.revenue_x.isna()&frame.revenue_y.isna())).all():
                raise ValueError('Canonical and audit revenue disagree')
            frame['revenue']=frame.revenue_x

        if frame.duplicated(['firm_id','fiscal_year']).any() or frame.fiscal_year.gt(2024).any():
            raise ValueError('Invalid financial cost source keys or future observations')
        return cls({(str(r['firm_id']),int(r['fiscal_year'])):r for r in frame.to_dict('records')})
    def calibrate(self,state,history,a0_revenue):
        history=list(history)
        if any(str(h.firm_id)!=str(state.firm_id) or h.year>state.year for h in history):
            raise ValueError('Financial cost calibration received foreign/future history')
        history=sorted(history,key=lambda h:h.year)[-3:]
        observations=[];amount_observations=[]
        for h in history:
            row=self.rows.get((str(h.firm_id),int(h.year)))
            if row and bool(row['residual_observation_valid']):
                amount_observations.append((h.year,float(row['non_interest_residual'])))
                if math.isfinite(float(row['revenue'])) and float(row['revenue'])>0:
                    observations.append((h.year,float(row['non_interest_residual'])/float(row['revenue'])))
        if not amount_observations:
            raise ValueError(f'R085_MISSING_ASOF_COMPONENT: no valid paired residual in recent three history rows for {state.firm_id}/{state.year}')
        ratio=statistics.mean(v for _,v in observations) if observations else None
        amount=float(a0_revenue)*ratio if observations else statistics.mean(v for _,v in amount_observations)
        used=observations if observations else amount_observations
        if not math.isfinite(amount) or amount<0:raise ValueError('Invalid calibrated non-interest financial cost')
        current=self.rows.get((str(state.firm_id),int(state.year)),{})
        def observed(name):
            v=current.get(name)
            return float(v) if v is not None and math.isfinite(float(v)) else None
        return amount,{'financial_cost_contract_version':FINANCIAL_COST_VERSION,
            'non_interest_cost_ratio':ratio,'baseline_non_interest_financial_cost':amount,
            'non_interest_calibration_basis':'revenue_ratio_mean' if observations else 'observed_amount_mean_no_valid_revenue',
            'non_interest_source_years':[int(y) for y,_ in used],
            'non_interest_source_year_max':max(y for y,_ in used),
            'decision_pure_interest_expense':observed('U01B550010000'),
            'decision_non_interest_financial_cost':observed('non_interest_residual'),
            'non_interest_candidate_invariance':'A0 amount fixed for all candidates',
            'income_statement_treatment':'pretax = operating income + modeled non-op income - pure interest - A0 non-interest cost'}

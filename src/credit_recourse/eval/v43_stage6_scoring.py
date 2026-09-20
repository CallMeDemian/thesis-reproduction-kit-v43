"""Read-only Alpha/Beta/Gamma scoring of the frozen V43 financial grid."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from credit_recourse.eval.v43_stage6_reporting import GRID_KEYS, evaluation_contract
from credit_recourse.eval.v43_alpha_evaluation import candidate_variables, alpha_from_variables, _prepare_stage6_context_lookup
from credit_recourse.eval.v43_oracle_backends import score_beta_ordered_logit_params, score_gamma_model
from credit_recourse.simulator.historical_source import read_financial_panel, state_from_record, firm_key

def score_grid(root, grid, *, expected_firms=575):
    from credit_recourse.eval.v43_stage6_reporting import validate_grid
    from credit_recourse.oracle.backends.alpha.modules.oracle_alpha_scorer import build_alpha_scorer
    validate_grid(grid,expected_firms=expected_firms)
    root=Path(root);contract=evaluation_contract(root)
    artifacts={name:root/value for name,value in contract['artifacts'].items()}
    params={oracle:json.loads(artifacts[oracle+'_params'].read_text(encoding='utf-8')) for oracle in ('alpha','beta','gamma')}
    scorer=build_alpha_scorer(params['alpha'])
    from credit_recourse.rl.v43_one_pass_data import input_root
    base=pd.read_parquet(input_root(root)/'phase_eval_candidate.parquet')
    base=base.loc[base.firm_id.isin(grid.firm_id.unique())]
    if len(base)!=expected_firms or not base.fiscal_year.eq(2024).all():raise ValueError('Oracle context cohort changed')
    base_lookup={(firm_key(r['firm_id']),int(r['fiscal_year'])):r for r in base.to_dict('records')}
    history_path=input_root(root)/'input_splits/canonical_business_plan_history.parquet'
    accounts,_=read_financial_panel(history_path)
    histories={}
    for raw in accounts.loc[accounts.firm_id.isin(base.firm_id)&accounts.fiscal_year.le(2024)].to_dict('records'):
        state=state_from_record(raw)
        histories.setdefault(firm_key(state.firm_id),[]).append(state)
    raw_history=pd.read_parquet(history_path)
    raw_history=raw_history.loc[raw_history.fiscal_year.le(2024)]
    context=_prepare_stage6_context_lookup(raw_history)
    selected=list(dict.fromkeys(v for p in params.values() for v in p['selected_variables']))
    inputs=[];scores=[]
    for i,raw in enumerate(grid.to_dict('records')):
        variables=candidate_variables(raw,base_lookup[(raw['firm_id'],int(raw['base_year']))],histories,context)
        missing=set(selected)-variables.keys()
        if missing:raise ValueError('Missing Oracle variable definitions: '+str(sorted(missing)))
        keys={k:raw[k] for k in GRID_KEYS}
        inputs.append({**keys,**{v:variables[v] for v in selected},'fiscal_year':2025})
        scores.append({**keys,**alpha_from_variables(variables,scorer,params['alpha']['selected_variables'])})
        if (i+1)%1000==0:print('ONE_PASS_STAGE6_ORACLES',i+1,len(grid),flush=True)
    inputs=pd.DataFrame(inputs);scores=pd.DataFrame(scores)
    scores['beta']=score_beta_ordered_logit_params(inputs,artifacts['beta_params']).to_numpy()
    scores['gamma']=score_gamma_model(inputs,artifacts['gamma_params'],artifacts['gamma_model']).to_numpy()
    if not np.isfinite(scores[['alpha','beta','gamma']].to_numpy()).all():raise ValueError('Nonfinite Oracle score')
    expected=grid.sim__financial_cost.div(grid.sim__revenue.where(grid.sim__revenue.ne(0)))
    np.testing.assert_allclose(inputs.R085,expected,rtol=1e-10,atol=1e-12,equal_nan=True)
    return scores,inputs


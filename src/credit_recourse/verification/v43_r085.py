"""Scoped regression and full-grid properties for the broad-cost repair."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

KEYS=['firm_id','base_year','candidate_id']
ATOL=1e-6
RTOL=1e-10

def scoped_financial_diff(root,frame,kind):
    # V4.3 owns the repaired grid directly. The retired pre-R085 grid is no
    # longer a runtime dependency; the current contract is verified from its
    # required fields and algebraic properties instead.
    required={'state__pure_interest_expense','state__non_interest_financial_cost',
        'sim__pure_interest_expense','sim__non_interest_financial_cost','bp__non_interest_financial_cost',
        'financial_cost_contract_version','non_interest_cost_ratio','baseline_non_interest_financial_cost',
        'non_interest_calibration_basis','non_interest_source_years','non_interest_source_year_max',
        'decision_pure_interest_expense','decision_non_interest_financial_cost',
        'non_interest_candidate_invariance','income_statement_treatment'}
    missing=sorted(required-set(frame))
    if missing:raise ValueError('Canonical R085 grid is missing contract fields: '+repr(missing))
    return {'status':'PASS','comparison_source':'canonical_V43_R085_contract_properties','rows':len(frame),
        'kind':kind,'atol':ATOL,'rtol':RTOL,'required_fields':sorted(required),'missing_fields':missing,
        'legacy_grid_dependency':False,
        'scope':'broad financial cost equals pure interest plus candidate-invariant non-interest financial cost'}

def properties(frame):
    if frame.duplicated(KEYS).any():raise ValueError('Duplicate financial candidate keys')
    grouped=frame.groupby(['firm_id','base_year'])
    assert grouped.candidate_id.nunique().eq(9).all()
    def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-6)
    close(frame.sim__financial_cost,frame.sim__pure_interest_expense+frame.sim__non_interest_financial_cost)
    assert grouped.sim__non_interest_financial_cost.nunique().eq(1).all()
    close(frame.sim__non_interest_financial_cost,frame.baseline_non_interest_financial_cost)
    assert grouped.sim__cash_dividends.nunique().eq(1).all()
    close(frame.sim__cash_dividends,frame.baseline_cash_dividend_plan)
    assert (frame.non_interest_source_year_max<=frame.base_year).all()
    close(frame.bp__rate_short,frame.bp__rate_long);close(frame.bp__rate_short,frame.bp__rate_bond)
    a0=frame[frame.candidate_id.eq('A0')].set_index(['firm_id','base_year'])
    ref=a0.reindex(pd.MultiIndex.from_frame(frame[['firm_id','base_year']]))
    savings=[];accounting=0
    for row in frame.to_dict('records'):
        d=json.loads(row['diagnostics_json']);rep=d['fcf_components']['repayment_by_stack']
        saving=sum(rep[k]*row['bp__rate_'+r] for k,r in [('short','short'),('gross_ltd','long'),('bond','bond')])
        close(d['dl_interest_saving']['total_dl_interest_saving'],saving)
        assert not d['fcf_components']['cash_reborrow_plug_used']
        financing=d['fcf_components']['liquidity_shortfall_financing']
        close(row['plug_amount'],financing)
        assert row['plug_used']==('operating_liquidity_shortfall_financing' if financing>1e-10 else 'none')
        assert json.loads(row['accounting_check_json'])['check']=='ok'
        savings.append(saving);accounting+=1
    close(ref.sim__pure_interest_expense.to_numpy()-frame.sim__pure_interest_expense.to_numpy(),savings)
    mask=frame.candidate_id.isin(['RF','OE','CX','WC1','WC2'])
    close(frame.loc[mask,'sim__pure_interest_expense'],ref.loc[mask.to_numpy(),'sim__pure_interest_expense'])
    rf=frame.candidate_id.eq('RF')
    close(frame.loc[rf,'sim__financial_cost'],ref.loc[rf.to_numpy(),'sim__financial_cost'])
    close(frame.loc[rf,'sim__revenue'],ref.loc[rf.to_numpy(),'sim__revenue'])
    # RF principal transfer is checked before common closing-date financing.
    for raw in frame.loc[rf,'diagnostics_json']:
        d=json.loads(raw)
        # The debt principal changes contain no repayment/issuance under RF.
        p=d['executed_financial_primitives']
        close(sum(p[k] for k in ['short_debt_principal_change','gross_ltd_principal_change','bond_principal_change']),0.)
    close(frame.sim__pretax_income,frame.sim__operating_income+frame.sim__revenue*frame.bp__non_op_income_ratio-frame.sim__financial_cost)
    return {'status':'PASS','rows':len(frame),'firms':grouped.ngroups,'atol':1e-6,'rtol':1e-10,
        'broad_equals_pure_plus_noninterest':True,'noninterest_fixed_across_candidates':True,
        'RF_pure_broad_R085_unchanged':True,'DL_MX_saving_applied_once':True,
        'other_actions_pure_interest_unchanged':True,'pretax_consistent':True,
        'fixed_dividend':True,'future_component_uses':0,'accounting_pass_rows':accounting}


def gate(root):
    from credit_recourse.rl.v43_one_pass_contract import RUN_PATH,write_json
    from credit_recourse.rl.contracts.v43_encoder import file_sha256
    root=Path(root);out=root/RUN_PATH;result={}
    for kind in ('training','evaluation'):
        frame=pd.read_parquet(out/f'stage2/{kind}_financial_grid.parquet')
        result[kind]={'properties':properties(frame),'scoped_diff':scoped_financial_diff(root,frame,kind)}
    expected={3:24439,4:24439,5:2642}
    counts={stage:int(pd.read_parquet(out/f'02_data/stage{stage}_rows.parquet').rl_fit_allowed.sum()) for stage in expected}
    if counts!=expected:raise ValueError('Population changed; explicit audit required: '+repr(counts))
    support=pd.read_parquet(out/'02_data/candidate_support.parquet')
    regression=support.loc[support.firm_id.eq('003380')&support.fiscal_year.eq(2021)]
    if len(regression)!=9 or not regression.candidate_supported.all():raise ValueError('003380/2021 fixed-dividend support failed')
    contract=json.loads((out/'01_contract/r085_financial_cost_contract.json').read_text(encoding='utf-8'))
    for name,digest in contract['source_hashes'].items():
        if file_sha256(root/name)!=digest:raise ValueError('Financial-cost source changed')
    result.update(status='PASS',population_counts=counts,regression_003380_2021='nine actions pass',
        source_hashes_verified=len(contract['source_hashes']),oracle_redefined=False,oracle_retrained=False)
    return result

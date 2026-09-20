"""V43 bundle training and evaluation grids with disjoint statistical roles."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet
from credit_recourse.rl.v43_one_pass_contract import load_run_config, assert_training_rows, write_json, run_path

def _run_path(root):
    return run_path(root)
from credit_recourse.rl.v43_one_pass_data import input_root, source_manifest
from credit_recourse.rl.contracts.v43_encoder import ACTION_IDS, KEYS, V43EncoderContract, file_sha256
from credit_recourse.rl.v43_features import canonical_keys, key_hash
from credit_recourse.rl.v43_reward_math import (
    PHI_COMPONENTS, CPLD_STATE_COLUMN, build_frozen_phi_cdf, compute_phi_with_frozen_cdf,
    _compute_aux_reward_raw_deltas, _derived_from_state_dict, apply_merton_fcff_aux_reward,
)
from credit_recourse.rl.common.reward_contract import fit_robust_p95_abs, compute_profitability_raw, compose_reward
from credit_recourse.simulator.historical_source import read_financial_panel, state_from_record
from credit_recourse.simulator.v43_production_bundle import V43ProductionSimulationBundle, financial_record

def fit_reward_statistics(root,rows):
    root=Path(root);out=root/_run_path(root)/'stage2'
    assert_training_rows(rows)
    factual=pd.read_parquet(root/_run_path(root)/'02_data/stage5_factual_reward_inputs.parquet')
    factual=factual.set_index(list(KEYS)).loc[pd.MultiIndex.from_frame(rows[list(KEYS)])].reset_index()
    stats_path=out/'auxiliary_reward_statistics.json'
    cdf_path=out/'sector_phi_cdf.json'
    if stats_path.exists() and cdf_path.exists():
        stats=json.loads(stats_path.read_text(encoding='utf-8'))
        if stats['fit_key_hash']!=key_hash(rows):raise ValueError('Frozen reward scale population changed')
        return stats,json.loads(cdf_path.read_text(encoding='utf-8'))
    deltas=_compute_aux_reward_raw_deltas(factual,context='full eligible factual reward scale')
    config=load_run_config(root)['stages']['stage2']
    stats={'lambda_merton':config['merton_lambda'],'lambda_fcff':config['fcff_lambda'],
        'lambda_liquidity':config['liquidity_lambda'],'lambda_profitability':0.,'profitability_aux_scale_p95_abs':1.,
        'fit_key_hash':key_hash(rows),'fit_rows':len(rows),'decision_year_max':int(rows.fiscal_year.max()),
        'outcome_year_max':int(rows.outcome_year.max()),'evaluation_rows_used':0}
    for name,column in [('merton','delta_merton_badness'),('fcff','delta_fcff_capacity'),('liquidity','delta_liquid_capacity')]:
        stats[name+'_aux_scale_p95_abs']=fit_robust_p95_abs(deltas[column],label=column)
    cdf=build_frozen_phi_cdf(factual,out)
    write_json(out/'sector_phi_cdf.json',cdf)
    write_json(out/'auxiliary_reward_statistics.json',stats)
    return stats,cdf

def reward_financial_view(record):
    row={}
    for prefix,destination in [('state__','sim__'),('sim__','next__sim__')]:
        state={key[len(prefix):]:value for key,value in record.items() if key.startswith(prefix)}
        for key,value in state.items():
            if isinstance(value,(int,float,np.integer,np.floating)) or value is None:
                row[destination+key]=0. if value is None else float(value)
        derived=_derived_from_state_dict(state)
        row.update({('next__' if prefix=='sim__' else '')+k:v for k,v in derived.items()})
    row[CPLD_STATE_COLUMN]=row['sim__current_portion_long_debt']
    row['next__'+CPLD_STATE_COLUMN]=row['next__sim__current_portion_long_debt']
    return row

def annotate_rating(row,source,candidate):
    event=bool(source['rating_reward_observed'])
    value=source['rating_reward_value']
    if event != bool(pd.notna(value)):
        raise ValueError('Factual rating-event reward support is inconsistent')
    coverage=bool(source['projection_observed_coverage_sufficient'])
    near=bool(source['near_tie_flag'])
    nearest='' if pd.isna(source['candidate_id']) else str(source['candidate_id'])
    hard=event and coverage and not near
    available=hard and candidate==nearest
    reason='no_new_rating_event' if not event else 'insufficient_observed_action_coverage' if not coverage else 'near_tie_projection' if near else ''
    source_name=('projected_observed_rating_event_to_nearest_P50_candidate' if available else
        'no_new_rating_event_tplus1_no_projected_supervision' if not event else
        'excluded_'+reason if reason else 'not_assigned_non_nearest_P50_candidate')
    for key,value_ in source.items():
        if key.startswith('projection_') or key in ('near_tie_flag','out_of_library_flag'):
            row[key]=value_
    row.update(nearest_P50_candidate_id=nearest,rating_reward_value=np.nan,rating_reward_observed=False,
        candidate_rating_outcome_observed=False,factual_rating_event_observed=event,
        rating_supervision_hard_eligible=hard,rating_supervision_exclusion_reason=reason,
        rating_supervision_projected_candidate_id=nearest,rating_supervision_target_candidate_id=nearest if available else '',
        rating_supervision_value=float(value) if available else np.nan,rating_supervision_available=available,
        rating_supervision_source=source_name,rating_supervision_numeric_contribution=float(value) if available else 0.)

def compose_training_rewards(frame,stats,cdf,config):
    assert_training_rows(frame)
    out=frame.copy()
    out['phi_t']=compute_phi_with_frozen_cdf(out,cdf)
    following=out.copy()
    for column in PHI_COMPONENTS: following[column]=out['next__'+column]
    out['phi_tplusH']=compute_phi_with_frozen_cdf(following,cdf)
    out['delta_phi']=out.phi_tplusH-out.phi_t
    out['delta_phi_clipped']=out.delta_phi.clip(-1.,1.)
    out['lambda_phi']=config['rho']
    out['profitability_raw'],qc=compute_profitability_raw(out,context='V43 candidate financial reward')
    out,_=apply_merton_fcff_aux_reward(out,stats,phase_name='V43 one-pass training grid')
    out['reward_total_raw']=compose_reward(base_component=out.rating_supervision_numeric_contribution,
        sector_phi_component=out.delta_phi_clipped,sector_phi_lambda=config['rho'],
        merton_component=out.reward_merton_norm,fcff_component=out.reward_fcff_norm,
        profitability_component=out.reward_profitability_norm,liquidity_component=out.reward_liquidity_norm,
        merton_lambda=stats['lambda_merton'],fcff_lambda=stats['lambda_fcff'],
        profitability_lambda=0.,liquidity_lambda=stats['lambda_liquidity'])
    if not np.isfinite(out.reward_total_raw).all(): raise ValueError('Invalid training grid reward')
    mean=float(out.reward_total_raw.mean());std=float(out.reward_total_raw.std())
    if not np.isfinite(std) or abs(std)<=1e-12:std=1.
    out['reward_mean_train']=mean;out['reward_std_train']=std
    out['reward_train']=(out.reward_total_raw-mean)/std
    out['reward_total']=out.reward_total_raw
    out['reward_raw']=np.nan;out['reward_raw_notch']=np.nan
    return out,{'fit_rows':len(out),'mean':mean,'std':std,'decision_year_max':int(out.fiscal_year.max()),
        'outcome_year_max':int(out.outcome_year.max()),'evaluation_rows_used':0,'profitability_qc':qc}

def generate_grid(root,*,evaluation=False):
    root=Path(root);out=root/_run_path(root)
    config=load_run_config(root);contract=V43EncoderContract.from_project_root(root)
    folder=out/'stage2';folder.mkdir(parents=True,exist_ok=True)
    prefix='evaluation' if evaluation else 'training'
    destination=folder/(prefix+'_financial_grid.parquet')
    if destination.exists():raise FileExistsError('Grid already materialized: '+str(destination))
    if evaluation:
        source = input_root(root)
        rows=pd.read_parquet(source/'input_splits/phase_eval.parquet',columns=list(KEYS))
        rows=canonical_keys(rows)
        if len(rows)!=575 or not rows.fiscal_year.eq(2024).all():raise ValueError('Evaluation cohort changed')
    else:
        rows=pd.read_parquet(out/'02_data/stage5_rows.parquet')
        rows=rows.loc[rows.rl_fit_allowed].reset_index(drop=True)
        assert_training_rows(rows)
        stats,cdf=fit_reward_statistics(root,rows)
    source = input_root(root)
    accounts,_=read_financial_panel(source/'input_splits/canonical_business_plan_history.parquet')
    accounts=accounts.loc[accounts.fiscal_year<=(2024 if evaluation else 2022)].reset_index(drop=True)
    histories={};states={}
    for record in accounts.to_dict('records'):
        state=state_from_record(record)
        states[(str(state.firm_id),int(state.year))]=state
        histories.setdefault(str(state.firm_id),[]).append(state)
    bundle=V43ProductionSimulationBundle.from_project_root(root)
    if not evaluation:
        from dataclasses import replace
        from credit_recourse.rl.v43_rate_extension import training_rate_resolver
        bundle=replace(bundle,_rate_resolver=training_rate_resolver(root))
    if evaluation:
        from credit_recourse.eval.v43_financial_inputs import _row_to_firm_state
        base=pd.read_parquet(source/'phase_eval_candidate.parquet')
        for raw in base.to_dict('records'):
            state=_row_to_firm_state(pd.Series(raw))
            states[(str(state.firm_id),int(state.year))]=state
    cached=None
    if not evaluation:
        from credit_recourse.rl.v43_support import read_action_support
        _,support_metadata=read_action_support(root)
        parts=out/'02_data'/support_metadata['parts_relative_path']
        cached=pd.concat([pd.read_parquet(parts/name) for name in support_metadata['part_hashes'] if name.startswith('financial_')],ignore_index=True)
        requested=set(map(tuple,rows[list(KEYS)].to_numpy()))
        cached=cached.loc[[ (str(f),int(y)) in requested for f,y in zip(cached.firm_id,cached.base_year) ]]
        if len(cached)!=len(rows)*9 or cached.duplicated(['firm_id','base_year','candidate_id']).any():raise ValueError('Support financial cache is incomplete')
        cached={(str(r['firm_id']),int(r['base_year']),str(r['candidate_id'])):r for r in cached.to_dict('records')}
    financial=[];training=[]
    for i,source in enumerate(rows.to_dict('records')):
        key=(str(source['firm_id']),int(source['fiscal_year']))
        state=states[key]
        for candidate in ACTION_IDS:
            try:
                record=cached[(key[0],key[1],candidate)] if cached is not None else financial_record(state,candidate,bundle.simulate_candidate(state,histories[key[0]],candidate))
            except Exception as error:
                raise ValueError(f'V43 {prefix} grid failed for firm={key[0]} year={key[1]} candidate={candidate}: {error}') from error
            financial.append(record)
            if not evaluation:
                row=reward_financial_view(record)
                row.update(firm_id=key[0],fiscal_year=key[1],outcome_year=key[1]+1,candidate_id=candidate,
                    factual_transition_id=f'{key[0]}::{key[1]}',done=1.,rl_fit_allowed=True,
                    action_support_valid=True,reward_support_valid=True,outcome_available=True,
                    action_contract_sha256=contract.action_contract_sha256,sector_7=source.get('sector_7'))
                annotate_rating(row,source,candidate)
                training.append(row)
        if (i+1)%250==0:print('ONE_PASS_GRID',prefix,i+1,len(rows),flush=True)
    financial=pd.DataFrame(financial)
    from credit_recourse.verification.v43_r085 import scoped_financial_diff, properties
    write_json(folder/(prefix+'_semantic_diff.json'),scoped_financial_diff(root,financial,prefix))
    write_json(folder/(prefix+'_r085_properties.json'),properties(financial))
    financial.to_parquet(destination,index=False)
    if evaluation:
        write_json(folder/'evaluation_grid_metadata.json',{'status':'PASS','rows':len(financial),'firms':575,
            'decision_year':2024,'rollout_year':2025,'dataset_sha256':file_sha256(destination),
            'role':'Stage6 evaluation only','optimizer_or_statistics_rows':0,
            'scoped_semantic_regression':'PASS','candidate_dividend_invariance':'PASS',
            'candidate_dividend_policy':config['candidate_dividend_policy'],'config_hash':config['config_hash'],
            'prior_erroneous_grid_score_parity_required':False,
            'reason':'Broad cost includes frozen A0 non-interest expense in pretax and downstream accounting'})
    else:
        train,reward_stats=compose_training_rewards(pd.DataFrame(training),stats,cdf,config['stages']['stage2'])
        assert_training_rows(train)
        from credit_recourse.rl.v43_training_math import _validate_projected_rating_supervision
        supervision_audit=_validate_projected_rating_supervision(train,list(ACTION_IDS))
        schema_folder=out/'04_validation/reward_schema_run';schema_folder.mkdir(parents=True,exist_ok=True)
        path=schema_folder/'training_grid.parquet';train.to_parquet(path,index=False)
        write_json(schema_folder/'reward_standardization.json',reward_stats)
        metadata={'status':'PASS','rows':len(train),'factual_rows':len(rows),'encoder_schema_hash':contract.schema_hash,
            'action_contract_sha256':contract.action_contract_sha256,'simulation_bundle':'V43ProductionSimulationBundle',
            'transition_source':'counterfactual','gamma':0.,'oracle_used_in_stage5_training':False,
            'non_rate_calibration_sha256':contract.non_rate_calibration_sha256,'dataset_sha256':file_sha256(path),
            'schema_fixture':False,'source_hashes':source_manifest(root),'config_hash':config['config_hash'],
            'financial_grid_sha256':file_sha256(destination),'reward_statistics':reward_stats,
            'rating_supervision_audit':supervision_audit,'training_boundary':assert_training_rows(train),
            'financial_origin':'hash-verified frozen V43ProductionSimulationBundle support audit outcomes; no second rollout'}
        metadata['role']='canonical reward recomposition schema run; no Stage5 optimizer'
        write_json(schema_folder/'training_grid_metadata.json',metadata)
        from credit_recourse.rl.v43_reward_components import freeze_components
        freeze_components(train,stats,metadata,folder)
    print('ONE_PASS_GRID_PASS',prefix,len(financial),flush=True)

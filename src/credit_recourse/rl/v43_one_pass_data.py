"""Single temporal population, factual labels and shared feature artifacts."""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from credit_recourse.rl.contracts.v43_encoder import KEYS, V43EncoderContract, file_sha256
from credit_recourse.rl.v43_features import V43FeatureProducer, canonical_keys, key_hash
from credit_recourse.rl.v43_actions import V43ActionCodec
from credit_recourse.rl.v43_one_pass_contract import (
    TEMPORAL, rl_fit_allowed, assert_training_rows, write_json, freeze_run_config, load_run_config, run_path,
)
from credit_recourse.simulator.historical_source import read_financial_panel

PHASES = {3:'phase1_pretrain.parquet',4:'phase2_bc.parquet',5:'phase3_iql.parquet'}

def input_root(root: Path) -> Path:
    configured = os.environ.get('THESIS_REPRO_STAGE2_INPUT_ROOT')
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else Path(root) / path
    raise FileNotFoundError('fresh Stage2 input root is not bound; set THESIS_REPRO_STAGE2_INPUT_ROOT')

def _run_path(root):
    return run_path(root)

def source_manifest(root):
    source = input_root(root)
    paths = [str(source / 'input_splits' / name) for name in PHASES.values()]
    paths += [str(source / 'input_splits/canonical_business_plan_history.parquet'),
        str(source / 'action_sources/stage2_raw_action_source_panel.parquet'),
        str(source / 'runtime_inputs/historical_financial_context_v4_3/actual_context.parquet'),
        str(source / 'runtime_inputs/historical_financial_context_v4_3/actual_financial_states.parquet'),
        str(source / 'v4_3_runtime/01_contract/financial_cost_sources_r2/history_financial_cost_sources.parquet')]
    from credit_recourse.rl.contracts.v43_encoder import RATE_PATH, RATE_CONTRACT_PATH, ACTION_PATH, CALIBRATION_PATH
    paths += [RATE_PATH,RATE_CONTRACT_PATH,ACTION_PATH,CALIBRATION_PATH,
        f"{_run_path(root)}/01_contract/r085_financial_cost_contract.json",
        "src/credit_recourse/simulator/financial_cost_v43.py",
        "src/credit_recourse/contracts/account_registry.py"]
    for name in ('extended_bp_rate_ledger.parquet','extended_bp_rate_preflight.json'):
        path=f'{_run_path(root)}/01_contract/{name}'
        if (Path(root)/path).exists():paths.append(path)
    for name in ('action_support.parquet','action_support_metadata.json','candidate_support.parquet'):
        path=f'{_run_path(root)}/02_data/{name}'
        if (Path(root)/path).exists():paths.append(path)
    return {str(Path(path).resolve().relative_to(Path(root).resolve())).replace('\\', '/'):file_sha256(Path(path)) for path in paths}

def prepare_populations(root):
    root=Path(root); out=root/_run_path(root)
    if (out/'02_data/population_summary.json').exists():
        raise FileExistsError('One-pass populations already materialized')
    contract=V43EncoderContract.from_project_root(root)
    config=freeze_run_config(root,contract)
    write_json(out/'01_contract/temporal_contract.json',TEMPORAL.to_dict())
    write_json(out/'01_contract/encoder_contract.json',{**contract.to_dict(),'schema_hash':contract.schema_hash})
    write_json(out/'01_contract/source_hashes.json',source_manifest(root))
    from credit_recourse.rl.v43_support import read_action_support
    support,support_metadata=read_action_support(root)
    codec=V43ActionCodec(contract)
    source_root = input_root(root)
    raw_path=source_root/'action_sources/stage2_raw_action_source_panel.parquet'
    accounts, account_lineage=read_financial_panel(raw_path)
    accounts=accounts.loc[accounts.fiscal_year<=TEMPORAL.train_outcome_year_max].reset_index(drop=True)
    operating_columns=[*KEYS,*codec.columns[3:],*[d.replace('action__','action_observed__',1) for d in codec.columns[3:]]]
    operating=pd.read_parquet(raw_path,columns=operating_columns,filters=[('fiscal_year','<=',2022)])
    projection, projection_meta=codec.project_observed(accounts,operating,decision_year_max=2022)
    assert projection.fiscal_year.max()<=2022
    folder=out/'02_data'; folder.mkdir(parents=True,exist_ok=True)
    projection.to_parquet(folder/'factual_nine_action_projection.parquet',index=False)
    codec.ledger().to_csv(folder/'nine_action_semantic_ledger.csv',index=False)
    history=pd.read_parquet(source_root/'input_splits/canonical_business_plan_history.parquet',columns=list(KEYS),filters=[('fiscal_year','<=',2023)])
    history=canonical_keys(history)
    current_keys=pd.read_parquet(source_root/'input_splits/phase1_pretrain.parquet',columns=list(KEYS))
    target_keys=canonical_keys(current_keys); target_keys['fiscal_year']+=1
    missing=~pd.MultiIndex.from_frame(target_keys).isin(pd.MultiIndex.from_frame(history))
    actual, _=read_financial_panel(source_root/'runtime_inputs/historical_financial_context_v4_3/actual_financial_states.parquet')
    targets=actual.loc[pd.MultiIndex.from_frame(actual[list(KEYS)]).isin(pd.MultiIndex.from_frame(target_keys.loc[missing]))].reset_index(drop=True)
    if targets.fiscal_year.max()>2023 or len(targets)!=int(missing.sum()):
        raise ValueError('Authoritative next-state supplement is incomplete or outside the time boundary')
    targets.to_parquet(folder/'target_only_financial_accounts.parquet',index=False)
    history_index=pd.MultiIndex.from_frame(pd.concat([history,targets[list(KEYS)]],ignore_index=True))
    summaries=[]; eligible=[]
    for stage,name in PHASES.items():
        path=source_root/'input_splits'/name
        columns=pq.read_schema(path).names
        wanted=[c for c in (*KEYS,'fiscal_year_next','rating_reward_value','rating_reward_observed','rating_event_id','rating_event_id__next','sector_7') if c in columns]
        source_frame=pd.read_parquet(path,columns=wanted)
        source_frame[list(KEYS)]=canonical_keys(source_frame)
        source_frame['outcome_year']=pd.to_numeric(source_frame.pop('fiscal_year_next'),errors='raise').astype('int64') if 'fiscal_year_next' in source_frame else source_frame.fiscal_year+1
        rows=source_frame.merge(projection,on=list(KEYS),how='left',validate='one_to_one')
        next_keys=rows[list(KEYS)].copy(); next_keys['fiscal_year']=rows.outcome_year
        rows['outcome_available']=pd.MultiIndex.from_frame(next_keys).isin(history_index)
        rows=rows.merge(support.rename(columns={'action_support_valid':'candidate_action_support_valid'}),on=list(KEYS),how='left',validate='one_to_one')
        if rows.candidate_action_support_valid.isna().any():raise ValueError('Missing common nine-action support evidence')
        rows['factual_action_support_valid']=rows.action_class_id.notna()
        rows['action_support_valid']=rows.candidate_action_support_valid & rows.factual_action_support_valid
        # The same executable-grid and factual-label support rule applies to every stage.
        rows['reward_support_valid']=True
        if stage == 5:
            from credit_recourse.rl.v43_reward_math import (MERTON_AUX_COMPONENTS,FCFF_AUX_COMPONENTS,
                LIQUIDITY_AUX_COMPONENTS,PHI_COMPONENTS,_compute_aux_reward_raw_deltas)
            primitive=list(dict.fromkeys([*MERTON_AUX_COMPONENTS,*FCFF_AUX_COMPONENTS,*LIQUIDITY_AUX_COMPONENTS]))
            wanted_reward=[*KEYS,*primitive,*['next__'+c for c in primitive],*PHI_COMPONENTS,'sector_7']
            wanted_reward += [c for c in columns if c.startswith(('reward_only__','next__reward_only__'))]
            absent=set(wanted_reward)-set(columns)
            if absent: raise ValueError('Missing authoritative factual reward fields: '+repr(absent))
            factual=pd.read_parquet(path,columns=list(dict.fromkeys(wanted_reward)))
            factual[list(KEYS)]=canonical_keys(factual)
            deltas=_compute_aux_reward_raw_deltas(factual,context='V43 full allowed factual reward support')
            valid=np.isfinite(deltas[['delta_merton_badness','delta_fcff_capacity','delta_liquid_capacity']]).all(axis=1)
            if not valid.all(): raise ValueError('Invalid factual reward support')
            factual.to_parquet(folder/'stage5_factual_reward_inputs_all.parquet',index=False)
        rows['rl_fit_allowed']=rl_fit_allowed(rows,action_support=rows.action_support_valid.to_numpy(dtype=bool),
            reward_support=rows.reward_support_valid.to_numpy(dtype=bool),outcome_available=rows.outcome_available.to_numpy(dtype=bool))
        rows['support_basis']='common nine-action executability AND existing factual-label support AND temporal/outcome/reward support'
        rows.to_parquet(folder/f'stage{stage}_rows.parquet',index=False)
        permitted=rows.loc[rows.rl_fit_allowed].copy()
        if stage==5:
            mask=pd.MultiIndex.from_frame(factual[list(KEYS)]).isin(pd.MultiIndex.from_frame(permitted[list(KEYS)]))
            factual.loc[mask].to_parquet(folder/'stage5_factual_reward_inputs.parquet',index=False)
        audit=assert_training_rows(permitted)
        summaries.append({'stage':stage,'source_rows':len(rows),'eligible_rows':len(permitted),
            'excluded_missing_action':int(rows.action_class_id.isna().sum()),
            'excluded_candidate_support':int((~rows.candidate_action_support_valid).sum()),
            'new_exclusions_on_previously_supported_rows':int((~rows.candidate_action_support_valid & rows.factual_action_support_valid & rows.outcome_available).sum()),
            'excluded_missing_outcome':int((~rows.outcome_available).sum()),'key_hash':key_hash(permitted),**audit})
        eligible.append(permitted[[*KEYS,'outcome_year','action_support_valid','reward_support_valid','outcome_available','rl_fit_allowed']])
    fit=pd.concat(eligible,ignore_index=True).drop_duplicates(list(KEYS)).sort_values(list(KEYS)).reset_index(drop=True)
    assert_training_rows(fit)
    fit.to_parquet(folder/'statistics_fit_rows.parquet',index=False)
    write_json(folder/'population_summary.json',{'status':'PASS','config_hash':config['config_hash'],'stages':summaries,
        'statistics_rows':len(fit),'statistics_key_hash':key_hash(fit),
        'action_support_rule':support_metadata['support_rule'],'action_support_sha256':support_metadata['support_sha256'],
        'action_projection_max_decision_year':2022,'action_projection_max_outcome_year':2023,
        'action_projection_fit_performed':False,'projection_metadata':projection_meta,'evaluation_rows_used':0})
    print(json.dumps(summaries),flush=True)

def prepare_features(root):
    root=Path(root);out=root/_run_path(root)
    if (out/'02_data/feature_manifest.json').exists():
        raise FileExistsError('One-pass feature artifacts already materialized')
    load_run_config(root)
    targets=pd.read_parquet(out/'02_data/target_only_financial_accounts.parquet')
    from credit_recourse.rl.v43_rate_extension import read_extended_rates
    rates,_=read_extended_rates(root)
    producer=V43FeatureProducer.from_project_root(root,source_year_max=2023,target_accounts=targets,rate_rows_override=rates)
    fit=pd.read_parquet(out/'02_data/statistics_fit_rows.parquet')
    assert_training_rows(fit)
    # Only optimizer/statistics decision keys and their <=2023 next-state targets.
    next_keys=fit[list(KEYS)].copy();next_keys['fiscal_year']+=1
    requested=pd.concat([fit[list(KEYS)],next_keys],ignore_index=True).drop_duplicates(list(KEYS)).sort_values(list(KEYS)).reset_index(drop=True)
    frames=[]; lineages=[]
    for start in range(0,len(requested),1000):
        frame,lineage=producer.build(requested.iloc[start:start+1000])
        frames.append(frame);lineages.append(lineage)
        if start%5000==0:print('ONE_PASS_FEATURES',min(start+1000,len(requested)),len(requested),flush=True)
    frame=pd.concat(frames,ignore_index=True)
    lineage=pd.concat(lineages,ignore_index=True)
    fit_index=pd.MultiIndex.from_frame(fit[list(KEYS)])
    subset=frame.loc[pd.MultiIndex.from_frame(frame[list(KEYS)]).isin(fit_index)].reset_index(drop=True)
    statistics=producer.fit_rl(subset,fit)
    stats_path=out/'01_contract/training_preprocessing.json'
    write_json(stats_path,statistics)
    raw_path=out/'02_data/training_and_target_features.parquet'
    frame.to_parquet(raw_path,index=False)
    lineage.to_parquet(out/'02_data/feature_history_lineage.parquet',index=False)
    batch=producer.transform_frame(frame,statistics)
    np.savez(out/'02_data/shared_feature_tensors.npz',continuous=batch.continuous,missing=batch.missing,categorical=batch.categorical)
    write_json(out/'02_data/feature_manifest.json',{'status':'PASS','rows':len(frame),
        'schema_hash':producer.contract.schema_hash,'statistics_hash':statistics['statistics_hash'],
        'raw_sha256':file_sha256(raw_path),'statistics_sha256':file_sha256(stats_path),
        'tensor_sha256':file_sha256(out/'02_data/shared_feature_tensors.npz'),
        'feature_hash':batch.feature_hash,'source_hashes':source_manifest(root),
        'statistics_fit_rows':len(fit),'statistics_fit_key_hash':key_hash(fit),
        'source_year_max':int(producer.accounts.fiscal_year.max()),'evaluation_rows_used':0})
    print('ONE_PASS_FEATURES_PASS',len(frame),len(fit),flush=True)

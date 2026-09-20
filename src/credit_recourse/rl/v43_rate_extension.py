"""Extend lookup coverage using the existing, unchanged BP-rate V4 selector."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from credit_recourse.rl.contracts.v43_encoder import RATE_PATH, RATE_CONTRACT_PATH, file_sha256
from credit_recourse.rl.v43_one_pass_contract import RUN_PATH, assert_training_rows, write_json, run_path
from credit_recourse.simulator.business_plan_interest_rate_v4 import BusinessPlanRateResolverV4

def prepare_rate_extension(root):
    root=Path(root);folder=root/run_path(root)/'01_contract'
    folder.mkdir(parents=True,exist_ok=True)
    destination=folder/'extended_bp_rate_ledger.parquet'
    if destination.exists():raise FileExistsError('Rate extension already frozen')
    # This is the selector used to extend the original ledger to 575 evaluation
    # firms. Its formula is reused unchanged, with all source observations capped.
    from credit_recourse.repro.runtime_helpers.build_bp_rate_v4_production_lookup import select_missing,key
    from credit_recourse.rl.v43_support import temporal_source_rows
    rows=temporal_source_rows(root)
    original=pd.read_parquet(root/RATE_PATH)
    required=set(map(tuple,rows[['firm_id','fiscal_year']].to_numpy()))
    present=set(map(tuple,original[['firm_id','base_year']].to_numpy()))
    provider_path=root/'archive/DEPLOYED_RELEASE/stage2_candidate_projection/runtime_inputs/business_plan_rate_v4/source/provider_rate_observation_ledger.parquet'
    event_path=root/'archive/DEPLOYED_RELEASE/stage0_oracle_foundation/canonical_panel/rating_event_panel.parquet'
    provider=pd.read_parquet(provider_path)
    provider['firm_id']=provider.firm_id.map(key)
    provider=provider.loc[provider.fiscal_year<=2022]
    valid=provider.loc[provider.source_valid.fillna(False)].sort_values('physical_row').drop_duplicates(['firm_id','fiscal_year'],keep='first')
    benches=valid.groupby('fiscal_year').decimal_rate.median().to_dict()
    events=pd.read_parquet(event_path)
    events['firm_id']=events.firm_id.map(key)
    events['rating_date']=pd.to_datetime(events.rating_date)
    events['rating_year']=events.rating_date.dt.year
    events=events.loc[events.rating_year.le(2022)&events.rating_num_notch.notna(),['firm_id','rating_year','rating_num_notch','rating_date']]
    added=[]; unavailable=[]
    for position,(firm,year) in enumerate(sorted(required-present)):
        if position%500==0:print('RATE_EXTENSION',position,len(required-present),flush=True)
        try:
            value=select_missing(firm,int(year),valid=valid,benches=benches,events=events)
        except ValueError as error:
            if not str(error).startswith('No V4 as-of provider benchmark for base_year='):raise
            unavailable.append({'firm_id':firm,'fiscal_year':int(year),'reason':str(error)})
            continue
        value['selection_panel_origin']='one_pass_training_lookup_extension'
        if value['source_observation_year'] is not None and value['source_observation_year']>year:
            raise ValueError('Future provider source in rate extension')
        added.append(value)
    extension=pd.DataFrame(added)
    ledger=pd.concat([original,extension],ignore_index=True)
    pd.testing.assert_frame_equal(ledger.iloc[:len(original)][original.columns].reset_index(drop=True),original.reset_index(drop=True),check_dtype=False,check_exact=True)
    missing=required-set(map(tuple,ledger[['firm_id','base_year']].to_numpy()))
    if missing!={(r['firm_id'],r['fiscal_year']) for r in unavailable}:raise ValueError('Unclassified missing rate after extension')
    ledger.to_parquet(destination,index=False)
    report={'status':'PASS','base_ledger_sha256':file_sha256(root/RATE_PATH),'ledger_sha256':file_sha256(destination),
        'unavailable_rate_keys':unavailable,'unavailable_source_keys':len(unavailable),'extension_keys':len(extension),'extension_sources':extension.selection_source.value_counts().to_dict() if len(extension) else {},
        'original_rows_preserved_exactly':len(original),'required_training_keys':len(rows),'missing_training_keys':0,
        'provider_source_year_max':int(valid.fiscal_year.max()),'rating_source_year_max':int(events.rating_year.max()),
        'future_source_uses':0,'evaluation_observations_used':0,'selection_rule_changed':False,'coverage_population':'all temporally allowed source states, before simulator support audit; exogenous lookup only',
        'source_hashes':{str(p.relative_to(root)):file_sha256(p) for p in (provider_path,event_path,root/RATE_PATH,root/'src/credit_recourse/repro/runtime_helpers/build_bp_rate_v4_production_lookup.py')}}
    write_json(folder/'extended_bp_rate_preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('unavailable_rate_keys','source_hashes')}),flush=True)

def read_extended_rates(root):
    root=Path(root);folder=root/run_path(root)/'01_contract'
    report=json.loads((folder/'extended_bp_rate_preflight.json').read_text(encoding='utf-8'))
    path=folder/'extended_bp_rate_ledger.parquet'
    if report['status']!='PASS' or report['ledger_sha256']!=file_sha256(path) or report['base_ledger_sha256']!=file_sha256(root/RATE_PATH):
        raise ValueError('Unverified BP-rate extension')
    for relative,digest in report['source_hashes'].items():
        if file_sha256(root/relative)!=digest:raise ValueError('BP-rate source changed')
    return pd.read_parquet(path),report

def training_rate_resolver(root):
    root=Path(root)
    ledger,report=read_extended_rates(root)
    original=BusinessPlanRateResolverV4.from_frozen_artifacts(root)
    return BusinessPlanRateResolverV4(ledger.to_dict('records'),contract=original.contract,
        contract_sha256=original.contract_sha256,ledger_sha256=report['ledger_sha256'],
        rating_grade_by_ordinal=original._grade_by_ordinal,valid_provider_years=original._valid_provider_years)

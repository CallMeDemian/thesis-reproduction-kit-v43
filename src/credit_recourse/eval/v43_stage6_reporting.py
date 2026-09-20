"""Keyed reporting on one immutable nine-action financial substrate."""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import yaml
from credit_recourse.rl.contracts.v43_encoder import ACTION_IDS, file_sha256, content_hash
from credit_recourse.rl.v43_one_pass_contract import RUN_PATH, TEMPORAL, assert_training_rows, write_json, load_run_config

GRID_KEYS = ['firm_id', 'base_year', 'candidate_id']
FIRM_KEYS = ['firm_id', 'base_year']
ORACLES = ('alpha', 'beta', 'gamma')

def validate_grid(grid, *, expected_firms=575):
    if len(grid) != expected_firms * len(ACTION_IDS) or grid.duplicated(GRID_KEYS).any():
        raise ValueError('Stage6 requires one unique row per firm and fixed action')
    if grid.firm_id.nunique() != expected_firms or not grid.base_year.eq(TEMPORAL.eval_base_year).all():
        raise ValueError('Stage6 requires the fixed 2024 cohort')
    if not grid['state__year'].eq(2024).all() or not grid['sim__year'].eq(2025).all():
        raise ValueError('Stage6 must use 2024 states and simulated 2025 targets')
    if any(set(g.candidate_id) != set(ACTION_IDS) for _, g in grid.groupby(FIRM_KEYS)):
        raise ValueError('Incomplete or legacy action grid')
    return {'firms':expected_firms,'rows':len(grid),'eval_base_year':2024,'simulated_target_year':2025}

def validate_actor(decisions, *, expected_firms=575):
    logits = decisions[['logit__'+a for a in ACTION_IDS]].to_numpy(dtype=float)
    if len(decisions) != expected_firms or decisions.duplicated(['firm_id','fiscal_year']).any():
        raise ValueError('Actor decisions must cover each evaluation firm exactly once')
    if not decisions.fiscal_year.eq(2024).all() or not np.isfinite(logits).all():
        raise ValueError('Invalid actor year or logits')
    if not np.array_equal(decisions.candidate_id.to_numpy(), np.asarray(ACTION_IDS)[logits.argmax(axis=1)]):
        raise ValueError('C3 must be final actor logits argmax only')

def join_choices(grid, decisions):
    chosen = decisions.rename(columns={'fiscal_year':'base_year'})[GRID_KEYS]
    if chosen.duplicated(FIRM_KEYS).any() or not set(chosen.candidate_id).issubset(ACTION_IDS):
        raise ValueError('Invalid or duplicate policy choices')
    if set(map(tuple,chosen[FIRM_KEYS].to_numpy())) != set(map(tuple,grid[FIRM_KEYS].drop_duplicates().to_numpy())):
        raise ValueError('Policy choices differ from the evaluation cohort')
    result=chosen.merge(grid,on=GRID_KEYS,how='left',validate='one_to_one',indicator=True)
    if not result['_merge'].eq('both').all():
        raise ValueError('Policy choice has no common-grid outcome')
    return result.drop(columns='_merge')

def summarize_policies(scores, c3, c2):
    """Ceilings are computed only here, after both policy ID vectors are fixed."""
    if not np.isfinite(scores[list(ORACLES)].to_numpy()).all():
        raise ValueError('All fixed action scores must be finite')
    policies={'C3':join_choices(scores,c3), 'C2':join_choices(scores,c2)}
    distributions=[]
    for policy,chosen in [('C3',c3),('C2',c2)]:
        for action in ACTION_IDS:
            n=int(chosen.candidate_id.eq(action).sum())
            distributions.append({'policy':policy,'candidate_id':action,'count':n,'share':n/len(chosen)})
    comparisons=[];contrasts=[];firm_contrasts=[];ceiling_frames=[]
    for oracle in ORACLES:
        wide=scores.pivot(index=FIRM_KEYS,columns='candidate_id',values=oracle).loc[:,list(ACTION_IDS)]
        means=wide.mean();best_fixed=str(means.idxmax())
        ceiling=wide.max(axis=1)
        c3_value=policies['C3'].set_index(FIRM_KEYS)[oracle].reindex(wide.index)
        c2_value=policies['C2'].set_index(FIRM_KEYS)[oracle].reindex(wide.index)
        values={'C3':c3_value,'C2':c2_value,**{a:wide[a] for a in ACTION_IDS},'ceiling':ceiling}
        for policy,value in values.items():
            comparisons.append({'oracle':oracle,'policy':policy,'n':len(value),'mean_score':float(value.mean()),
                'policy_value':float((value-wide['A0']).mean()),'mean_delta_vs_A0':float((value-wide['A0']).mean()),
                'reporting_only':policy not in ('C3','C2')})
        delta={'C3_minus_C2':c3_value-c2_value,'C3_minus_RF':c3_value-wide['RF'],
               'C3_minus_best_fixed':c3_value-wide[best_fixed],'ceiling_minus_C3':ceiling-c3_value}
        contrasts.append({'oracle':oracle,'best_fixed_action':best_fixed,'best_fixed_mean_score':float(means[best_fixed]),
                          **{k:float(v.mean()) for k,v in delta.items()},'best_fixed_and_ceiling_reporting_only':True})
        firm_contrasts.append(pd.DataFrame(delta).assign(oracle=oracle,best_fixed_action=best_fixed).reset_index())
        ceiling_frames.append(ceiling.rename('score').reset_index().assign(oracle=oracle,reporting_only=True))
    return {'comparison':pd.DataFrame(comparisons),'contrasts':pd.DataFrame(contrasts),
            'firm_contrasts':pd.concat(firm_contrasts,ignore_index=True),
            'ceiling':pd.concat(ceiling_frames,ignore_index=True),
            'action_distribution':pd.DataFrame(distributions), **policies}

def evaluation_contract(root, *, freeze=False):
    root=Path(root);stage2=Path(os.environ.get('THESIS_REPRO_STAGE2_ROOT', root/RUN_PATH));stage6=Path(os.environ.get('THESIS_REPRO_STAGE6_ROOT', stage2/'stage6'));path=stage6/'evaluation_contract.json'
    registry_file=Path(os.environ.get('THESIS_REPRO_ORACLE_CONFIG_ROOT', root/'contracts/oracle_components'))/'oracle_backend_registry.yaml'
    registry_path=str(registry_file)
    registry=yaml.safe_load(registry_file.read_text(encoding='utf-8'))
    artifacts={'registry':registry_path,
        'alpha_params':str(stage2/'work/stage1_oracle_backends/alpha/oracle_alpha_params.json'),
        'beta_params':str(stage2/'work/stage1_oracle_backends/beta/benchmark_beta_params.json'),
        'gamma_params':str(stage2/'work/stage1_oracle_backends/gamma/benchmark_gamma_params.json'),
        'gamma_model':str(stage2/'work/stage1_oracle_backends/gamma/benchmark_gamma_model.joblib')}
    sources={p:file_sha256(Path(p) if Path(p).is_absolute() else root/p) for p in artifacts.values()}
    value={'version':'V43Stage6/1','eval_base_year':2024,'simulated_target_year':2025,'oracles':list(ORACLES),
        'action_ids':list(ACTION_IDS),'artifacts':artifacts,'source_hashes':sources,
        'C3':'Stage5 final_epoch.pt actor logits argmax; candidate ID join only',
        'C2':'v43_c2.c2_row_conditional candidate_ids_only=True; candidate ID join only',
        'best_fixed_and_firm_ceiling':'post-hoc reporting only; never supplied to C3',
        'score_direction':'higher is better for every backend','config_hash':load_run_config(root)['config_hash']}
    value['contract_hash']=content_hash(value)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=value:raise ValueError('Frozen Stage6 contract changed')
    elif freeze:write_json(path,value)
    else:raise FileNotFoundError('Stage6 contract must be frozen before training')
    return value

def assert_no_evaluation_fit(root):
    root=Path(root);out=Path(os.environ.get('THESIS_REPRO_STAGE2_ROOT', root/RUN_PATH))
    audit={}
    for label in ('stage3','stage4','stage5','statistics_fit'):
        frame=pd.read_parquet(out/'02_data'/f'{label}_rows.parquet')
        if label!='statistics_fit':frame=frame.loc[frame.rl_fit_allowed]
        audit[label]=assert_training_rows(frame)
    for name in ('auxiliary_reward_statistics','reward_standardization'):
        stats=json.loads((out/('04_validation/reward_schema_run' if name=='reward_standardization' else 'stage2')/f'{name}.json').read_text(encoding='utf-8'))
        if stats['decision_year_max']>2022 or stats['outcome_year_max']>2023 or stats['evaluation_rows_used']!=0:
            raise ValueError('Evaluation rows entered reward statistics')
        audit[name]=stats
    preprocessing=json.loads((out/'01_contract/training_preprocessing.json').read_text(encoding='utf-8'))
    from credit_recourse.rl.v43_features import validate_statistics, key_hash
    from credit_recourse.rl.contracts.v43_encoder import V43EncoderContract
    validate_statistics(V43EncoderContract.from_project_root(root),preprocessing)
    fit=pd.read_parquet(out/'02_data/statistics_fit_rows.parquet')
    if preprocessing['fit_key_hash']!=key_hash(fit):raise ValueError('Training preprocessing population changed')
    for stage in (3,4,5):
        execution=json.loads((out/f'stage{stage}/execution.json').read_text(encoding='utf-8'))
        if execution['status']!='PASS' or execution['evaluation_rows_used']!=0:
            raise ValueError('Missing completed training lineage or evaluation optimizer contamination')
    audit['evaluation_optimizer_or_statistics_rows']=0
    return audit


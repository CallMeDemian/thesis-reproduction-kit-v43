"""Final native V43 evaluation; policy IDs join one frozen nine-action grid."""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import pyarrow.parquet
from credit_recourse.rl.v43_one_pass_contract import RUN_PATH, load_run_config, write_json
from credit_recourse.rl.contracts.v43_encoder import KEYS, file_sha256
from credit_recourse.eval.v43_stage6_reporting import (
    validate_grid, validate_actor, join_choices, summarize_policies,
    evaluation_contract, assert_no_evaluation_fit,
)


def evaluate(root):
    root=Path(root)
    out=Path(os.environ.get('THESIS_REPRO_STAGE2_ROOT', root/RUN_PATH))
    folder=Path(os.environ.get('THESIS_REPRO_STAGE6_ROOT', out/'stage6'))
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'evaluation_summary.json').exists():raise FileExistsError('Final OOT evaluation already completed')
    config=load_run_config(root)
    lineage=assert_no_evaluation_fit(root)
    gate=json.loads((out/'04_validation/pretraining_gate.json').read_text(encoding='utf-8'))
    if gate['status']!='PASS' or gate['config_hash']!=config['config_hash']:raise ValueError('Pretraining gate changed')
    for path,digest in gate['source_hashes'].items():
        if file_sha256(root/path)!=digest:raise ValueError('Source changed after pretraining gate: '+path)
    checkpoint=Path(os.environ.get('THESIS_REPRO_STAGE5_ROOT', out/'stage5'))/'final_epoch.pt'
    run=json.loads((checkpoint.parent/'execution.json').read_text(encoding='utf-8'))
    if run['status']!='PASS' or run['final_epoch']!=config['stages']['stage5']['max_epochs'] or run['checkpoint_sha256']!=file_sha256(checkpoint):
        raise ValueError('Stage6 requires the Stage5 fixed final epoch')
    decision_path=folder/'C3_actor_decisions.parquet'
    from credit_recourse.rl.v43_runtime import V43StageConsumer
    if decision_path.exists():
        decisions=pd.read_parquet(decision_path)
        provenance=json.loads((folder/'actor_provenance.json').read_text(encoding='utf-8'))
        if provenance['checkpoint_sha256']!=file_sha256(checkpoint) or provenance['decision_sha256']!=file_sha256(decision_path):
            raise ValueError('Previously fixed actor decisions changed')
    else:
        decisions=V43StageConsumer(root).select_policy(checkpoint)
        validate_actor(decisions)
        decisions['policy']='C3'
        decisions.to_parquet(decision_path,index=False)
        write_json(folder/'actor_provenance.json',{'checkpoint_sha256':file_sha256(checkpoint),
            'decision_sha256':file_sha256(decision_path),'config_hash':config['config_hash'],
            'actor_fixed_before_Oracle_scoring':True,'actor_logit_argmax_only':True,
            'rerank':False,'forced_action_share':False,'action_proportion_constraints_applied':False})
    validate_actor(decisions)
    if set(decisions.config_hash)!={config['config_hash']}:raise ValueError('Actor run configuration changed')
    contract=evaluation_contract(root)
    from credit_recourse.eval.v43_c2 import c2_row_conditional
    from credit_recourse.rl.v43_one_pass_data import input_root
    base=pd.read_parquet(input_root(root)/'phase_eval_candidate.parquet')
    reference=pd.read_parquet(out/'02_data/stage5_factual_reward_inputs.parquet')
    if reference.fiscal_year.max()>2022:raise ValueError('Future data in C2 reference')
    c2=c2_row_conditional(base,None,reference,candidate_ids_only=True)
    c2=pd.concat([base[list(KEYS)].reset_index(drop=True),c2.reset_index(drop=True)],axis=1)
    c2['policy']='C2'
    c2.to_parquet(folder/'C2_fixed_rule_decisions.parquet',index=False)
    meta=json.loads((out/'stage2/evaluation_grid_metadata.json').read_text(encoding='utf-8'))
    grid_path=out/'stage2/evaluation_financial_grid.parquet'
    if meta['status']!='PASS' or meta['dataset_sha256']!=file_sha256(grid_path):raise ValueError('Financial grid changed')
    grid=pd.read_parquet(grid_path);clock=validate_grid(grid)
    for label,chosen in [('C3',decisions),('C2',c2)]:
        join_choices(grid,chosen).to_parquet(folder/(label+'_financial.parquet'),index=False)
    # Scorers see financial outcomes only after the actor and rule IDs are fixed.
    from credit_recourse.eval.v43_stage6_scoring import score_grid
    scores,inputs=score_grid(root,grid)
    scores.to_parquet(folder/'all_fixed_action_scores.parquet',index=False)
    inputs.to_parquet(folder/'oracle_inputs_2025.parquet',index=False)
    reports=summarize_policies(scores,decisions,c2)
    for name,frame in reports.items():
        frame.to_parquet(folder/(name+'.parquet'),index=False)
        if name in ('comparison','contrasts','action_distribution'):frame.to_csv(folder/(name+'.csv'),index=False)
    write_json(folder/'evaluation_summary.json',{'status':'PASS','role':'final_OOT_2024',**clock,
        'config_hash':config['config_hash'],'evaluation_contract_hash':contract['contract_hash'],
        'checkpoint_sha256':file_sha256(checkpoint),'financial_grid_sha256':file_sha256(grid_path),
        'comparison':reports['comparison'].to_dict('records'),'contrasts':reports['contrasts'].to_dict('records'),
        'action_distribution':reports['action_distribution'].to_dict('records'),
        'training_lineage':lineage,'R085_broad_cost_semantics':'PASS','candidate_dividend_policy':config['candidate_dividend_policy'],'C3_C2_common_grid_join':'PASS',
        'rerank':False,'Oracle_used_for_action_choice':False,'forced_action_share':False,'action_proportion_constraints_applied':False,
        'later_result_informed_runs':'exploratory/development evaluation; not another final OOT test'})
    print(reports['comparison'].to_string(index=False),flush=True)
    print(reports['contrasts'].to_string(index=False),flush=True)


def main(argv=None):
    import argparse
    from credit_recourse.rl.v43_one_pass_contract import production_guard
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',required=True)
    parser.add_argument('--sim-business-plan-mode',choices=['calibrated_bp_rate_v4'],default='calibrated_bp_rate_v4')
    args=parser.parse_args(argv)
    with production_guard() as calls:evaluate(args.project_root)
    write_json(Path(args.project_root)/RUN_PATH/'logs/stage6_native_entry_guard.json',{'status':'PASS','calls':calls})
    return 0


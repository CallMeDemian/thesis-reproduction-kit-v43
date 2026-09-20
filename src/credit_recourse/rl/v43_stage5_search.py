"""Independent Stage5 runs consuming one immutable Stage3/4 baseline."""
from pathlib import Path
from dataclasses import dataclass,asdict
import copy,json,math,re,os
import numpy as np
import pandas as pd
from credit_recourse.rl.contracts.v43_encoder import ACTION_IDS, file_sha256, content_hash
from credit_recourse.rl.v43_one_pass_contract import run_path,write_json
from credit_recourse.rl.v43_reward_components import recompose
def search_root(root: Path) -> Path:
    configured = os.environ.get("THESIS_REPRO_SEARCH_ROOT")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else Path(root) / path
    return Path(root) / "runs/search-reproduction/search/rl"

@dataclass(frozen=True)
class SearchConfig:
    run_id:str
    merton_lambda:float=.2
    fcff_lambda:float=.4
    liquidity_lambda:float=.1
    tau:float=.7
    beta:float=2.
    awr_cap:float=10.
    kl_lambda:float=.25
    kl_temperature:float=2.
    q_margin:float=.05
    lr:float=1e-4
    wd:float=.002
    epochs:int=80
    seed:int=2
    def validate(self):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',self.run_id):raise ValueError('Invalid unique run ID')
        for k,v in asdict(self).items():
            if k!='run_id' and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)):raise ValueError('Nonfinite/invalid run parameter: '+k)
        if any(v<0 for v in self.weights().values()) or sum(self.weights().values())<=0:raise ValueError('Invalid M/F/L weights')
        if not 0<self.tau<1 or self.beta<0 or self.awr_cap<1 or self.kl_lambda<0 or self.kl_temperature<=0 or self.q_margin<0 or self.lr<=0 or self.wd<0:raise ValueError('Invalid IQL configuration')
        if type(self.epochs)!=int or self.epochs<1 or type(self.seed)!=int or self.seed<0:raise ValueError('Epoch/seed must be integers')
        return self
    def weights(self):return dict(merton=self.merton_lambda,fcff=self.fcff_lambda,liquidity=self.liquidity_lambda)
    def stage5(self,baseline):
        value=copy.deepcopy(baseline['stages']['stage5'])
        value.update(expectile_tau=self.tau,beta=self.beta,awr_weight_cap=self.awr_cap,
            actor_distill_lambda=self.kl_lambda,actor_distill_temperature=self.kl_temperature,
            actor_distill_margin_min=self.q_margin,learning_rate=self.lr,weight_decay=self.wd,
            max_epochs=self.epochs,seed=self.seed)
        return value

def load_baseline(root, *, training=False):
    root=Path(root).resolve()
    configured = os.environ.get("THESIS_REPRO_SEARCH_BASELINE")
    path = Path(configured) if configured else Path(run_path(root)) / "V43_STAGE5_SEARCH_BASELINE.json"
    if not path.is_absolute():
        path = root / path
    baseline=json.loads(path.read_text(encoding='utf-8'))
    permitted=['READY_FOR_V43_STAGE5_HYPERPARAMETER_SEARCH'] if training else ['PREPARED_FOR_SCHEMA_DRY_RUN','READY_FOR_V43_STAGE5_HYPERPARAMETER_SEARCH']
    if baseline['status'] not in permitted:raise ValueError('Search baseline is not ready')
    if baseline['upstream_baseline_hash']!=content_hash(baseline['upstream_identity']):raise ValueError('Upstream identity hash mismatch')
    if baseline['evaluation_contract_hash']!=content_hash(baseline['evaluation_identity']):raise ValueError('Evaluation identity hash mismatch')
    for name,digest in baseline['immutable_files'].items():
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha256(path)!=digest:raise ValueError('Immutable baseline changed: '+name)
    return baseline

def experiment_identity(baseline,config,standardization):
    run={'parameters':asdict(config),'reward_standardization':standardization,
        'terminal_target':'immediate reward_train','batch_size':128,'encoder_frozen':True,'gamma':0.,'rho':0.}
    # Run ID is a namespace, not a scientific hyperparameter.
    run['parameters'].pop('run_id')
    h=content_hash(run)
    return {'upstream_baseline_hash':baseline['upstream_baseline_hash'],'stage5_run_hash':h,
        'evaluation_contract_hash':baseline['evaluation_contract_hash'],
        'full_experiment_hash':content_hash([baseline['upstream_baseline_hash'],h,baseline['evaluation_contract_hash']]),
        'stage5_run_identity':run}

def leaderboard_schema():
    columns=['run_id','merton_lambda','fcff_lambda','liquidity_lambda','tau','beta','awr_cap','kl_lambda','kl_temperature','q_margin','lr','wd','epochs','seed',
        'upstream_baseline_hash','stage5_run_hash','full_experiment_hash','checkpoint_sha256']
    for oracle in ('alpha','beta','gamma'):
        columns += [oracle+'__'+v for v in ('C3_mean_score','C3_policy_value','C2','best_fixed','best_fixed_action','C3_minus_C2','C3_minus_best_fixed','ceiling_minus_C3')]
        columns += [oracle+'__'+v for v in ('switched_firms','beneficial_switches','harmful_switches','tie_switches','beneficial_gain','harmful_loss','net_switch_gain')]
    for a in ACTION_IDS:
        columns += ['actor_count__'+a,'actor_share__'+a,'critic_count__'+a,'critic_share__'+a]
    columns+=['max_action_share','actor_entropy','actor_critic_agreement','RF_actor_probability','RF_actor_rank','OE_actor_probability','OE_actor_rank','RF_minus_OE_Q_gap','KL_gate_application_rate','AWR_cap_hit_rate','training_diagnostics_population','evaluation_diagnostics_population']
    return {'version':'V43Stage5SearchLeaderboard/1','columns':columns,'unique_key':'run_id',
        'policy':'final actor logits argmax only','diagnostic_only_columns':'critic/action distributions never alter actor choices',
        'score_direction':'higher is better','policy_value':'mean score(policy) minus score(A0)',
        'no_runs_executed_by_schema_definition':True}

def prepare_run(root,config, *, dry_run=True):
    config.validate();root=Path(root).resolve();baseline=load_baseline(root,training=not dry_run)
    destination=search_root(root)/config.run_id
    if destination.exists():raise FileExistsError('Run namespace already exists: '+config.run_id)
    component_path=root/baseline['artifacts']['reward_components']
    frame=pd.read_parquet(component_path)
    reward,stats,supervision=recompose(frame,config.weights())
    identity=experiment_identity(baseline,config,stats)
    destination.mkdir(parents=True,exist_ok=False)
    reward_path=destination/'stage2/training_grid.parquet';reward_path.parent.mkdir()
    reward.to_parquet(reward_path,index=False)
    source_metadata=json.loads((component_path.parent/'reward_components_metadata.json').read_text(encoding='utf-8'))
    metadata={**source_metadata,'dataset_sha256':file_sha256(reward_path),'reward_statistics':stats,
        'rating_supervision_audit':supervision,'schema_fixture':False,'run_id':config.run_id,**identity}
    write_json(destination/'stage2/training_grid_metadata.json',metadata)
    write_json(destination/'stage2/reward_standardization.json',stats)
    write_json(destination/'run_config.json',{**asdict(config),**identity,'mode':'schema_dry_run' if dry_run else 'training'})
    # Validate real complete inputs through the same production loader. No model or optimizer.
    from credit_recourse.rl.v43_training import load_counterfactual
    consumer=SearchConsumer(root,destination,config,baseline,identity)
    *_,lineage=load_counterfactual(consumer,reward_path,destination/'stage2/training_grid_metadata.json')
    proof={'status':'PASS','mode':'schema_dry_run' if dry_run else 'training_preflight','optimizer_steps':0,
        'simulator_calls':0,'trained_C3_evaluations':0,'leaderboard_rows_written':0,
        'upstream_checkpoints_reused':[baseline['artifacts']['stage3_checkpoint'],baseline['artifacts']['stage4_checkpoint']],
        'lineage':lineage,**identity}
    write_json(destination/'preflight.json',proof)
    if dry_run:
        write_json(destination/'dry_run.json',proof)
        return proof
    from credit_recourse.rl.v43_training import train_policy
    train_policy(consumer,5,root/baseline['artifacts']['stage4_checkpoint'],destination/'stage5/final_epoch.pt',
        transition_input=reward_path,transition_metadata=destination/'stage2/training_grid_metadata.json')
    result=evaluate_run(consumer,config,baseline,identity)
    append_leaderboard(root,result)
    return result

# The baseline consumer keeps its original config when validating Stage3/4.
# Stage5 has an independent namespace and run config, while all input methods delegate.
from credit_recourse.rl.v43_runtime import V43StageConsumer
class SearchConsumer(V43StageConsumer):
    def __init__(self,root,destination,config,baseline,identity):
        self.baseline_consumer=V43StageConsumer(root)
        self.__dict__.update({k:v for k,v in self.baseline_consumer.__dict__.items()})
        self.baseline_folder=self.folder;self.folder=Path(destination)
        self.config=copy.deepcopy(self.baseline_consumer.config)
        self.config['seed']=config.seed;self.config['stages']['stage5']=config.stage5(self.config)
        self.config['stages']['stage2'].update(merton_lambda=config.merton_lambda,fcff_lambda=config.fcff_lambda,liquidity_lambda=config.liquidity_lambda)
        self.config['config_hash']=identity['full_experiment_hash'];self.search_baseline=baseline;self.identity=identity
    def rows(self,stage):return self.baseline_consumer.rows(stage)
    def training_features(self,keys):return self.baseline_consumer.training_features(keys)
    def prepare(self,stage,*,limit=None):return self.baseline_consumer.prepare(stage,limit=limit)
    def _verify_final_checkpoint(self,path,payload,stage):
        if stage in (3,4):return self.baseline_consumer._verify_final_checkpoint(path,payload,stage)
        if stage!=5:raise ValueError('Unsupported search checkpoint stage')
        return super()._verify_final_checkpoint(path,payload,stage)
    def begin_run_training(self,stage,destination,rows):
        import os
        from datetime import datetime,timezone
        if stage!=5 or Path(destination).resolve()!=(self.folder/'stage5/final_epoch.pt').resolve():raise ValueError('Search runs may train only Stage5 final epoch')
        fresh=load_baseline(self.root,training=True)
        if fresh['upstream_baseline_hash']!=self.identity['upstream_baseline_hash']:raise ValueError('Baseline identity changed')
        proof=json.loads((self.folder/'preflight.json').read_text(encoding='utf-8'))
        if proof['status']!='PASS' or proof['mode']!='training_preflight' or proof['full_experiment_hash']!=self.config['config_hash']:raise ValueError('Missing run-specific preflight')
        parent=Path(destination).parent;parent.mkdir(parents=True,exist_ok=True)
        with (parent/'execution.json').open('x',encoding='utf-8') as f:
            json.dump(dict(status='RUNNING',stage=5,pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),
                config_hash=self.config['config_hash'],input_rows=rows,fixed_epochs=self.config['stages']['stage5']['max_epochs'],seed=self.config['seed'],**self.identity),f,indent=2)

def evaluate_run(consumer,config,baseline,identity):
    import torch
    from credit_recourse.eval.v43_stage6_reporting import summarize_policies,validate_actor
    checkpoint=consumer.folder/'stage5/final_epoch.pt'
    decisions=consumer.select_policy(checkpoint);validate_actor(decisions)
    folder=consumer.folder/'stage6';folder.mkdir()
    decisions.to_parquet(folder/'actor_decisions.parquet',index=False)
    write_json(folder/'actor_provenance.json',{'actor_logit_argmax_only':True,'checkpoint_sha256':file_sha256(checkpoint),'decisions_sha256':file_sha256(folder/'actor_decisions.parquet'),'oracle_scores_read_after_decisions_saved':True})
    scores=pd.read_parquet(consumer.root/baseline['artifacts']['oracle_surface'])
    c2=pd.read_parquet(consumer.root/baseline['artifacts']['c2_decisions'])
    reports=summarize_policies(scores,decisions,c2)
    out={**asdict(config),**{k:v for k,v in identity.items() if k!='stage5_run_identity'},'checkpoint_sha256':file_sha256(checkpoint)}
    for oracle in ('alpha','beta','gamma'):
        comparison=reports['comparison'].query('oracle == @oracle').set_index('policy')
        contrast=reports['contrasts'].query('oracle == @oracle').iloc[0];fixed=contrast.best_fixed_action
        values={'C3_mean_score':comparison.loc['C3','mean_score'],'C3_policy_value':comparison.loc['C3','policy_value'],
            'C2':comparison.loc['C2','mean_score'],'best_fixed':comparison.loc[fixed,'mean_score'],'best_fixed_action':fixed,
            **{k:contrast[k] for k in ('C3_minus_C2','C3_minus_best_fixed','ceiling_minus_C3')}}
        wide=scores.pivot(index='firm_id',columns='candidate_id',values=oracle)
        chosen=reports['C3'].set_index('firm_id').reindex(wide.index)
        delta=chosen[oracle]-wide[fixed];switch=chosen.candidate_id.ne(fixed);tol=1e-10
        values.update(switched_firms=int(switch.sum()),beneficial_switches=int((switch&(delta>tol)).sum()),harmful_switches=int((switch&(delta< -tol)).sum()),
            tie_switches=int((switch&(delta.abs()<=tol)).sum()),beneficial_gain=float(delta.clip(lower=0).sum()),harmful_loss=float(-delta.clip(upper=0).sum()),net_switch_gain=float(delta.sum()))
        out.update({oracle+'__'+k:v for k,v in values.items()})
    model=consumer.load_policy(checkpoint,expected_stage=5).eval();batch,rows=consumer.prepare(6)
    result=[]
    with torch.no_grad():
        for start in range(0,len(rows),128):
            result.append(model(*(torch.from_numpy(v[start:start+128]) for v in (batch.continuous,batch.missing,batch.categorical))))
    logits=torch.cat([r['actor_logits'] for r in result]);prob=logits.softmax(1);q=torch.minimum(torch.cat([r['q1'] for r in result]),torch.cat([r['q2'] for r in result]))
    actor=logits.argmax(1);critic=q.argmax(1);ranks=(-prob).argsort(1).argsort(1)+1
    for i,a in enumerate(ACTION_IDS):
        out.update({'actor_count__'+a:int((actor==i).sum()),'actor_share__'+a:float((actor==i).float().mean()),
            'critic_count__'+a:int((critic==i).sum()),'critic_share__'+a:float((critic==i).float().mean())})
    out.update(max_action_share=max(out['actor_share__'+a] for a in ACTION_IDS),actor_entropy=float(-(prob*logits.log_softmax(1)).sum(1).mean()),actor_critic_agreement=float((actor==critic).float().mean()))
    for a in ('RF','OE'):
        j=ACTION_IDS.index(a);out[a+'_actor_probability']=float(prob[:,j].mean());out[a+'_actor_rank']=float(ranks[:,j].float().mean())
    out['RF_minus_OE_Q_gap']=float((q[:,ACTION_IDS.index('RF')]-q[:,ACTION_IDS.index('OE')]).mean())
    # Training reward-row diagnostics have their own explicit population.
    from credit_recourse.rl.v43_training import load_counterfactual
    batch,index,actions,rewards,fit,_=load_counterfactual(consumer,consumer.folder/'stage2/training_grid.parquet',consumer.folder/'stage2/training_grid_metadata.json')
    cached=[]
    with torch.no_grad():
        for start in range(0,len(batch.keys),128):
            cached.append(model(*(torch.from_numpy(v[start:start+128]) for v in (batch.continuous,batch.missing,batch.categorical))))
    tq=torch.minimum(torch.cat([r['q1'] for r in cached]),torch.cat([r['q2'] for r in cached]))
    value=torch.cat([r['value'] for r in cached]);top=tq.topk(2,dim=1).values
    out['KL_gate_application_rate']=float(((top[index,0]-top[index,1])>=config.q_margin).float().mean())
    advantage=tq[index,torch.from_numpy(actions)]-value[index]
    out['AWR_cap_hit_rate']=float((config.beta*advantage>=math.log(config.awr_cap)).float().mean())
    out.update(training_diagnostics_population='all eligible nine-candidate training reward rows',evaluation_diagnostics_population='575 decision states in 2024')
    for name in ('comparison','contrasts','action_distribution','firm_contrasts'):reports[name].to_csv(folder/(name+'.csv'),index=False)
    write_json(folder/'leaderboard_row.json',out)
    return out

def append_leaderboard(root,row):
    folder=search_root(root);schema=leaderboard_schema();required=set(schema['columns'])
    if not required.issubset(row):raise ValueError('Incomplete leaderboard diagnostics: '+repr(sorted(required-set(row))))
    path=folder/'leaderboard.jsonl';lock=folder/'leaderboard.append.lock'
    # Exclusive lock protects independent run append and duplicate IDs.
    with lock.open('x',encoding='utf-8') as handle:
        handle.write(row['run_id'])
    try:
        previous=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
        if any(r['run_id']==row['run_id'] for r in previous):raise ValueError('Duplicate leaderboard run ID')
        with path.open('a',encoding='utf-8') as handle:handle.write(json.dumps({k:row[k] for k in schema['columns']},allow_nan=False,default=lambda v:v.item())+'\n')
    finally:lock.unlink()

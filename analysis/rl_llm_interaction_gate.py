"""Frozen-data RL–LLM interaction gate; no LLM call or RL retraining."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from rl_reference_gate_analysis import nested_predictions, mean_ci, holm

ORACLES=("alpha","beta","gamma")

def main():
 p=argparse.ArgumentParser(); p.add_argument('--runtime-root',type=Path,required=True); p.add_argument('--rl-results',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
 root=a.runtime_root.resolve(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
 s5=root/'data/final_freeze/stage5_candidate_iql/C3E_E2_7SEED_BALANCED_DFEBAFA6'
 contract=json.loads((root/'data/final_freeze/stage2_candidate_projection/candidate_action_contract_v4_3.json').read_text())
 cols=contract['action_columns']; scale=np.array([contract['normalization_scale_by_dimension'][c.removeprefix('action__')] for c in cols],float)
 candidates=pd.DataFrame(contract['fixed_candidates']).T[cols].to_numpy(float)/scale
 candidate_ids=np.array(list(contract['fixed_candidates']))
 rl=pd.read_csv(a.rl_results/'RL_REFERENCE_RL_FEATURES.csv').set_index('firm_key')
 actions=pd.read_parquet(root/'stage7_full_materialized/llm_stage7_action_table_itt.parquet')
 c4=actions[(actions['mode'].eq('free8'))&(actions['information_condition'].eq('IC-b'))&(actions['budget'].eq('B1'))&(actions['replicate'].astype(str).eq('1'))&(actions['policy'].eq('C4'))].copy()
 scores=pd.read_parquet(root/'data/final_freeze/stage8_llm_multi_oracle_eval/llm_stage8_multi_oracle_scores_all_populations.parquet')
 scores=scores[(scores['analysis_population'].eq('itt'))&(scores['mode'].eq('free8'))&(scores['information_condition'].eq('IC-b'))&(scores['budget'].eq('B1'))&(scores['replicate'].astype(str).eq('1'))&scores['policy'].isin(['C4R','C6-E'])].copy()
 rl_action=pd.read_parquet(s5/'C3E_firm_probabilities.parquet').set_index(['firm_id','fiscal_year'])
 # Join by frozen firm_key, retaining C4 alone as an admissible gate input.
 c4['firm_key']=c4.firm_key.astype(str); c4v=c4[cols].to_numpy(float)/scale
 c4['_row']=np.arange(len(c4)); c4['c4_total_intensity']=np.abs(c4v).sum(1); c4['c4_active_axes']=(np.abs(c4v)>1e-9).sum(1)
 c4['c4_max_axis_share']=np.divide(np.abs(c4v).max(1),np.abs(c4v).sum(1),out=np.zeros(len(c4)),where=np.abs(c4v).sum(1)>0)
 c4['c4_concentration']=(np.divide(np.abs(c4v),np.abs(c4v).sum(1,keepdims=True),out=np.zeros_like(c4v),where=np.abs(c4v).sum(1,keepdims=True)>0)**2).sum(1)
 c4['c4_action_valid']=c4.action_valid.astype(int); c4['c4_projection_needed']=0
 # candidate choice is the C3-E ensemble action materialized in frozen Stage5.
 chosen=dict(zip((rl_action.index.get_level_values(0).astype(str).str.zfill(6)+'::'+rl_action.index.get_level_values(1).astype(str)),rl_action.action_id.astype(str)))
 c4['rl_chosen_action']=c4.firm_key.map(chosen)
 if c4.rl_chosen_action.isna().any(): raise RuntimeError('C4/RL action join failure')
 ridx=np.array([np.where(candidate_ids==x)[0][0] for x in c4.rl_chosen_action]); rv=candidates[ridx]
 delta=c4v-rv; active_c=np.abs(c4v)>1e-9; active_r=np.abs(rv)>1e-9
 c4['c4_rl_l1_distance']=np.abs(delta).sum(1); c4['c4_rl_l2_distance']=np.sqrt((delta**2).sum(1)); c4['dominant_axis_match']=(np.abs(c4v).argmax(1)==np.abs(rv).argmax(1)).astype(int)
 c4['direction_disagreement']=((np.sign(c4v)!=np.sign(rv)) & active_c & active_r).sum(1); c4['rl_axis_c4_intensity']=(np.abs(c4v)*active_r).sum(1)
 c4['active_axis_jaccard']=(active_c&active_r).sum(1)/np.maximum((active_c|active_r).sum(1),1)
 dist=((c4v[:,None,:]-candidates[None,:,:])**2).sum(2)**.5; near=dist.argmin(1); c4['c4_nearest_rl_candidate']=candidate_ids[near]; c4['c4_candidate_mapping_loss']=dist.min(1); c4['c4_rl_candidate_match']=(c4.c4_nearest_rl_candidate==c4.rl_chosen_action).astype(int)
 # interaction terms are observed at C4 time only.
 interaction=[]
 rl_base=['ensemble_predictive_entropy','ensemble_disagreement_js','vote_entropy','chosen_vote_fraction','top1_probability','top1_top2_probability_margin','q_selected_relative','q_selected_vs_best_relative','q_top1_top2_relative_margin','q_argmax_actor_agreement']
 for f in ['chosen_vote_fraction','q_top1_top2_relative_margin','ensemble_predictive_entropy']:
  name=f'{f}_x_c4_rl_l2_distance'; c4[name]=c4[f] if f in c4 else 0; interaction.append(name)
 records=[]; oofs=[]; mechanism=[]
 for model in sorted(c4.model_key.unique()):
  m=c4[c4.model_key.eq(model)].set_index('firm_key').join(rl[rl_base],how='inner',validate='one_to_one')
  wide=scores[scores.model_key.eq(model)].pivot(index='firm_key',columns='policy',values=[f'delta_R_score_{x}' for x in ORACLES])
  wide.columns=['_'.join(x) for x in wide.columns]; m=m.join(wide,how='inner',validate='one_to_one')
  if len(m)!=575: raise RuntimeError(f'{model}: join {len(m)} != 575')
  # Construct terms after joining RL values.
  for f in ['chosen_vote_fraction','q_top1_top2_relative_margin','ensemble_predictive_entropy']:
   m[f'{f}_x_c4_rl_l2_distance']=m[f]*m.c4_rl_l2_distance
  llm=['c4_total_intensity','c4_active_axes','c4_max_axis_share','c4_concentration','c4_action_valid','c4_projection_needed','c4_rl_l1_distance','c4_rl_l2_distance','dominant_axis_match','direction_disagreement','rl_axis_c4_intensity','active_axis_jaccard','c4_candidate_mapping_loss','c4_rl_candidate_match']
  groups={'rl_only':rl_base,'llm_only':llm,'rl_llm_interaction':rl_base+llm+[f'{f}_x_c4_rl_l2_distance' for f in ['chosen_vote_fraction','q_top1_top2_relative_margin','ensemble_predictive_entropy']]}
  yalpha=(m['delta_R_score_alpha_C6-E']-m['delta_R_score_alpha_C4R']).to_numpy(float)
  for group,fs in groups.items():
   pred=nested_predictions(m[fs].to_numpy(float),yalpha,'ridge',20261117); use=pred>0
   for oracle in ORACLES:
    y=(m[f'delta_R_score_{oracle}_C6-E']-m[f'delta_R_score_{oracle}_C4R']).to_numpy(float); diff=np.where(use,y,0)-y; lo,hi,pv=mean_ci(diff)
    records.append(dict(model_key=model,oracle=oracle,gate=group,n=575,always_rl_increment=y.mean(),gate_increment=np.where(use,y,0).mean(),gate_gain_vs_always=diff.mean(),ci_low=lo,ci_high=hi,bootstrap_p_raw=pv,reference_share=use.mean(),oof_pred_alpha_spearman=spearmanr(pred,yalpha).statistic,hindsight_increment=np.maximum(y,0).mean()))
    oofs.append(pd.DataFrame({'firm_key':m.index,'model_key':model,'oracle':oracle,'gate':group,'actual_increment':y,'oof_pred_alpha':pred,'gate_use_C6E':use,'gate_minus_always':diff}))
   # mechanism is descriptive and separately reported for each Oracle.
  for oracle in ORACLES:
   y=m[f'delta_R_score_{oracle}_C6-E']-m[f'delta_R_score_{oracle}_C4R']
   for feature in ['c4_rl_candidate_match','dominant_axis_match']:
    for val,g in m.assign(gain=y).groupby(feature): mechanism.append(dict(model_key=model,oracle=oracle,analysis='group',feature=feature,level=str(val),n=len(g),mean_increment=g.gain.mean(),positive=(g.gain>0).sum(),negative=(g.gain<0).sum(),tie=(g.gain==0).sum()))
   for bucket,g in m.assign(gain=y).groupby(pd.qcut(m.c4_rl_l2_distance,5,duplicates='drop'),observed=True): mechanism.append(dict(model_key=model,oracle=oracle,analysis='distance_quintile',feature='c4_rl_l2_distance',level=str(bucket),n=len(g),mean_increment=g.gain.mean(),positive=(g.gain>0).sum(),negative=(g.gain<0).sum(),tie=(g.gain==0).sum()))
 comp=pd.DataFrame(records); alpha=comp.oracle.eq('alpha'); comp.loc[alpha,'holm_p_alpha_family']=holm(comp.loc[alpha,'bootstrap_p_raw'].tolist())
 comp.to_csv(out/'GATE_MODEL_COMPARISON.csv',index=False); pd.concat(oofs).to_csv(out/'GATE_OOF_PREDICTIONS.csv',index=False); pd.DataFrame(mechanism).to_csv(out/'MECHANISM_ANALYSIS.csv',index=False)
 c4.join(rl,how='left',rsuffix='_rl').to_csv(out/'INTERACTION_FEATURES.csv',index=False)
 (out/'DATA_AUDIT.md').write_text(f'# Data audit\n\n- C4 primary actions: {len(c4)} rows (575 per model).\n- C4R/C6-E primary scores: {len(scores)} rows.\n- Every model-level gate joins 575 firms.\n- C4R/C6-E are outcomes only; all gate features are available immediately after C4.\n- Value/Advantage remains unavailable.\n',encoding='utf8')
 lines=['# RL–LLM interaction-aware adaptive guidance','', 'Primary gates are 5-fold OOF Ridge models trained on Alpha only. Beta/Gamma reuse the same Alpha-trained gate decisions.','', '| Model | Oracle | Gate | Gate − always | 95% CI | Holm p (Alpha) | Reference rate |','|---|---|---|---:|---|---:|---:|']
 for x in comp.itertuples(): lines.append(f'| {x.model_key} | {x.oracle} | {x.gate} | {x.gate_gain_vs_always:+.4f} | [{x.ci_low:+.4f}, {x.ci_high:+.4f}] | {"—" if pd.isna(x.holm_p_alpha_family) else f"{x.holm_p_alpha_family:.4f}"} | {x.reference_share:.1%} |')
 lines += ['', 'A positive point estimate alone is insufficient: the primary criterion is a positive paired OOF CI and a multiplicity-adjusted result, with matching directions in Beta/Gamma.']
 (out/'FINAL_TECHNICAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 # figures
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots();
 for model,g in m.groupby('model_key') if False else []: pass
 # aggregate mechanism after output to prevent any chart from implying a causal fit.
 z=pd.DataFrame(mechanism); q=z[(z.analysis=='distance_quintile')&(z.oracle=='alpha')]; fig,ax=plt.subplots(figsize=(9,4));
 for model,g in q.groupby('model_key'): ax.plot(range(len(g)),g.mean_increment,marker='o',label=model)
 ax.axhline(0,color='black',lw=.8); ax.set_xlabel('C4–RL L2-distance quintile'); ax.set_ylabel('C6-E − C4R (Alpha)'); ax.legend(); fig.tight_layout(); fig.savefig(out/'figure_1_distance_increment.png',dpi=180); plt.close(fig)
 q=comp[comp.oracle.eq('alpha')]; fig,ax=plt.subplots(figsize=(9,4));
 for model,g in q.groupby('model_key'): ax.plot(g.gate,g.gate_gain_vs_always,marker='o',label=model)
 ax.axhline(0,color='black',lw=.8); ax.set_ylabel('OOF Gate − Always RL'); ax.legend(); fig.tight_layout(); fig.savefig(out/'figure_2_gate_comparison.png',dpi=180); plt.close(fig)
 q=comp[comp.gate.eq('rl_llm_interaction')]; fig,ax=plt.subplots(figsize=(9,4));
 for model,g in q.groupby('model_key'): ax.errorbar(g.oracle,g.gate_gain_vs_always,yerr=[g.gate_gain_vs_always-g.ci_low,g.ci_high-g.gate_gain_vs_always],fmt='o-',capsize=4,label=model)
 ax.axhline(0,color='black',lw=.8); ax.set_ylabel('Interaction gate − Always RL'); ax.legend(); fig.tight_layout(); fig.savefig(out/'figure_3_robustness.png',dpi=180); plt.close(fig)
if __name__=='__main__': main()

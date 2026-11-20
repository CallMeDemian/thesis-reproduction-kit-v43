"""Shared loaders for stored policies (thesis ITT observations, E1 ablation, C3-E) on the row_id axis (0..574)."""
import numpy as np, pandas as pd, json, os
K=os.environ.get('KIT_ROOT','/home/claude/kitzip/THESIS_REPRO_KIT_v2.1.1_FINAL'); DR=K+'/archive/DEPLOYED_RELEASE'
E=['alpha','beta','gamma']; IDS=['A0','DL','RF','CX','WC1','WC2','OE','MX1','MX2']
AX=['growth_capex_reduction_pct','deleveraging_total_debt_pct','refinancing_short_debt_pct','inv_turnover_chg','ar_turnover_chg','ap_turnover_chg','cogs_ratio_chg','sga_ratio_chg']
C=json.load(open(DR+'/stage2_candidate_projection/candidate_action_contract_v4_3.json')); SC=C['normalization_scale_by_dimension']
CAND={k:{a:float(v['action__'+a]) for a in AX} for k,v in C['fixed_candidates'].items()}
def norm(d): return np.array([float(d.get(a,0.0))/SC[a] for a in AX])
itt=pd.read_parquet(K+'/frozen_replay/v1.3/frozen_inputs/stage8/canonical_itt_observations.parquet')
S=itt[itt.execution_policy=='STRICT'].copy()
S['m']=S.model_key.map({'openai_gpt54mini':'GPT','google_gemini31flashlite':'Gemini'}); S['r']=S.reasoning_arm.map({'BASELINE':'low','HIGH':'high'})
def get(mode,budget,r,cond,m,info='IC-b',rep=1):
    x=S[(S['mode']==mode)&(S.budget==budget)&(S.r==r)&(S.policy_condition==cond)&(S.m==m)&(S['info']==info)&(S.replicate==rep)].sort_values('row_id')
    assert len(x)==575 and (x.row_id.values==np.arange(575)).all(); return x.reset_index(drop=True)
def vals(x): return np.stack([x['delta_R_score_'+e].values for e in E],1)
def acts(x): return [json.loads(s) for s in x.strict_itt_action_json]
a3=pd.read_parquet(DR+'/stage5_candidate_iql/C3E_E2_7SEED_BALANCED_DFEBAFA6/C3E_firm_actions.parquet').sort_values('row_id').reset_index(drop=True)
p3=pd.read_parquet(DR+'/stage5_candidate_iql/C3E_E2_7SEED_BALANCED_DFEBAFA6/C3E_firm_probabilities.parquet').sort_values('row_id').reset_index(drop=True)
P3=p3[['prob_'+k for k in IDS]].values; RLID=np.array([IDS.index(x) for x in a3.action_id])
e1=pd.read_parquet(os.environ.get('ABL_E1','/home/claude/abl/E1/observations/e1_observations.parquet'))  # ablation_20261102/E1/observations/e1_observations.parquet
e1['m']=e1.model_key.map({'openai_gpt54mini':'GPT','google_gemini31flashlite':'Gemini'})
def gete1(cond,m):
    x=e1[(e1.e1_condition==cond)&(e1.m==m)].sort_values('row_id'); assert len(x)==575 and (x.row_id.values==np.arange(575)).all(); return x.reset_index(drop=True)
def boot(x,B=4000,seed=0):
    x=np.asarray(x,float); r=np.random.default_rng(seed); bs=x[r.integers(0,len(x),(B,len(x)))].mean(1); return x.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
def ci(x): m,l,h=boot(x); return f'{m:+.3f} [{l:+.3f},{h:+.3f}]'
def rnd(x): return np.round(x,9)

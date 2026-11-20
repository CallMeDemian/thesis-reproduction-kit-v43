import sys; sys.path.insert(0,'.')
from hz import *
H=Harness()
K=str(ROOT)
itt=pd.read_parquet(f'{K}/frozen_replay/v1.3/frozen_inputs/stage8/canonical_itt_observations.parquet')
S=itt[itt.execution_policy=='STRICT'].reset_index(drop=True)
print(len(S), 'row_id range', S.row_id.min(), S.row_id.max())
fid=H.base['firm_id'].astype(str).values
assert (S.firm_key.str.split('::').str[0].values==fid[S.row_id.values]).all()
acts=[json.loads(x) for x in S.strict_itt_action_json]
t=time.time(); out=H.evaluate(list(zip(S.row_id.values,acts))); print('sims %.1fs'%(time.time()-t),'fail',(~out.ok).sum())
for e in ['alpha','beta','gamma']:
    d=(out[e].values-S['R_score_'+e].values); print(e,'max|d| %.3e'%np.nanmax(np.abs(d)),' n>1e-6:',int((np.abs(d)>1e-6).sum()))
# strict_usable anomaly
un=S[~S.strict_usable]
nz=(un[['delta_R_score_alpha','delta_R_score_beta','delta_R_score_gamma']].abs().sum(axis=1)>0)
print('unusable rows',len(un),'with nonzero delta',int(nz.sum()))
print(un[nz].groupby(['reasoning_arm','mode','budget','policy_condition']).size().head(20))
x=un[nz].iloc[0]; print(x[['cell_id','action_application_status','candidate_id','strict_usable','repaired','repair_method','strict_itt_action_json','delta_R_score_alpha']].to_string())
print(pd.crosstab(S.strict_usable,S.action_application_status))
print(pd.crosstab(S.strict_usable,S.candidate_id.eq('A0')))

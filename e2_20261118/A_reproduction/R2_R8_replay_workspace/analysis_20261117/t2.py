import sys; sys.path.insert(0,'.')
from hz import *
t=time.time(); H=Harness(); print('init %.1fs'%(time.time()-t))
K=str(ROOT)
# --- full 575 x 9 grid
t=time.time()
ids=list(H.cands); items=[(i,H.cands[k]) for i in range(len(H.base)) for k in ids]
out=H.evaluate(items); out['row_id']=[i for i in range(len(H.base)) for k in ids]; out['cand']=[k for i in range(len(H.base)) for k in ids]
print('grid sims %.1fs'%(time.time()-t), 'fail', (~out.ok).sum())
out['firm_id']=H.base['firm_id'].astype(str).values[out.row_id.values]
out.to_parquet('grid9.parquet',index=False)
a=pd.read_parquet(DR/'stage5_candidate_iql/C3E_E2_7SEED_BALANCED_DFEBAFA6/C3E_firm_actions.parquet')
g=out.set_index(['row_id','cand'])
for e,E_ in [('alpha','Alpha'),('beta','Beta'),('gamma','Gamma')]:
    mine=np.array([g.loc[(r,c),e] for r,c in zip(a.row_id,a.action_id)]); d=np.abs(mine-a[E_].values)
    mo=np.array([g.loc[(r,'OE'),e] for r in a.row_id]); do=np.abs(mo-a['OE_'+E_].values)
    mc=np.array([g.loc[(r,c),e] for r,c in zip(a.row_id,a.C2_action)]); dc=np.abs(mc-a['C2_'+E_].values)
    print(e,'C3E max|d| %.2e  OE max|d| %.2e  C2 max|d| %.2e'%(d.max(),do.max(),dc.max()))
itt=pd.read_parquet(f'{K}/frozen_replay/v1.3/frozen_inputs/stage8/canonical_itt_observations.parquet')
itt['firm']=itt.firm_key.str.split('::').str[0]
nz=itt.drop_duplicates('firm').set_index('firm')
for e in ['alpha','beta','gamma']:
    mine=out[out.cand=='A0'].set_index('firm_id')[e]; d=(mine-nz.loc[mine.index,'noop_R_score_'+e]).abs()
    print(e,'A0 vs stored noop max|d| %.2e'%d.max())
piv=out.pivot(index='row_id',columns='cand',values='alpha'); a0=piv['A0']
print('alpha policy values vs A0:', {k: round((piv[k]-a0).mean(),3) for k in ids}, 'upper', round((piv.max(axis=1)-a0).mean(),3))
for e in ['beta','gamma']:
    p=out.pivot(index='row_id',columns='cand',values=e); print(e,{k: round((p[k]-p['A0']).mean(),3) for k in ids}, 'upper', round((p.max(axis=1)-p['A0']).mean(),3))

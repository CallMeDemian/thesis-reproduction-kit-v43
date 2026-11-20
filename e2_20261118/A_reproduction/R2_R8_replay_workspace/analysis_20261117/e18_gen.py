"""e18: score every stored LLM output of the <=1 budget condition on all 7 evaluators (alpha/beta/gamma are re-computed and checked against the frozen values)."""
import sys; sys.path.insert(0,'.')
from hz import *
from hz import _target_operating_loss_freq_3y, _state_to_frame_dict
from delta_train import feats_from_state
from fam2 import feats2, make_scorer
import prep2 as P, joblib
S1=make_scorer(joblib.load('delta_models.joblib')); S2=make_scorer(joblib.load('delta2_models.joblib'))
H=Harness(); n=575
def full(items):
    rows=[];F1=[];F2=[]
    for i,a in items:
        r,fs,hist,exo=H.firm(i); res=H.bundle.simulate_action(fs,hist,dict(a),action_label='X')[0]
        ex=dict(exo); ex['operating_loss_freq_3y_target']=_target_operating_loss_freq_3y(hist,res.state_t1)
        sv=compute_oracle_variables(res.state_t1,prev_state=fs,exogenous=ex); row=_state_to_frame_dict(sv,r)
        o={v:(row[v] if v in row else r.get(v)) for v in H.need}; o['row_id']=i; rows.append(o); F1.append(feats_from_state(res.state_t1)); F2.append(feats2(res.state_t1))
    s=H.score(rows); return np.concatenate([s[['alpha','beta','gamma']].values,S1(F1),S2(F2)],1)
A0=full([(i,H.cands['A0']) for i in range(n)])
B=P.S[P.S.budget=='B1']; out={}; t=time.time(); worst=0.0
for (m,mode,r,c,info,rep),g in B.groupby(['m','mode','r','policy_condition','info','replicate']):
    g=g.sort_values('row_id'); assert len(g)==n and (g.row_id.values==np.arange(n)).all()
    v=full(list(enumerate([json.loads(s) for s in g.strict_itt_action_json])))-A0
    worst=max(worst,float(np.abs(v[:,:3]-P.vals(g)).max()))
    out[f'{m}|{mode}|{r}|{c}|{info}|{rep}']=v; out[f'{m}|{mode}|{r}|{c}|{info}|{rep}|usable']=g.strict_usable.values.astype(float)
    print(m,mode,r,c,info,rep,'%.0fs'%(time.time()-t),flush=True)
print('max |recomputed - frozen| on alpha/beta/gamma: %.2e'%worst)
np.savez('e18_b1_all7.npz',**out); print('done',flush=True)

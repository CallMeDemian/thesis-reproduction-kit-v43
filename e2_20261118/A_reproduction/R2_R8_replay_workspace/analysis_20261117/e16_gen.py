"""e16: assignment value of LLM arms after controlling for applicability.
Actions are permuted only among firms that share the same applicability pattern (which of DL / RF / CX do anything for the firm),
so what remains is selection among actions that are equally available to the firm."""
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
A9=np.load('A9.npy'); app=(np.abs(A9)>1e-9).any(2)
key=np.array([''.join('1' if x else '0' for x in r) for r in app[:,1:4]])   # DL, RF, CX
strata=[np.where(key==k)[0] for k in sorted(set(key))]; print('strata',{k:int((key==k).sum()) for k in sorted(set(key))},flush=True)
A0=full([(i,H.cands['A0']) for i in range(n)])
pol={}
for m in ['GPT','Gemini']:
    for c in ['C4','C4R','C6-E','C6-EX']: pol[f'{m} {c}']=P.acts(P.get('free8','B1','high',c,m))
    for c in ['E1-RL','E1-OE','E1-C4R']: pol[f'{m} {c}']=[json.loads(s) for s in P.gete1(c,m).strict_itt_action_json]
rng=np.random.default_rng(11); NP=20; t=time.time(); out={}
perms=[]
for _ in range(NP):
    pm=np.arange(n)
    for idx in strata: pm[idx]=rng.permutation(idx)
    perms.append(pm)
for nm,al in pol.items():
    mix=np.zeros((NP,n,7))
    for k,pm in enumerate(perms): mix[k]=full([(i,al[pm[i]]) for i in range(n)])-A0
    out[nm+'|smix']=mix.mean(0); print(nm,'%.0fs'%(time.time()-t),flush=True)
    np.savez('e16_strat.npz',key=key,**out)
print('done',flush=True)

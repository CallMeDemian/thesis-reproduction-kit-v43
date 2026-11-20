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
U=np.load('Ugrid.npy'); V9=np.load('V9.npy'); Vg=np.load('Vg.npy'); OEi=P.IDS.index('OE'); VOE=V9[:,OEi,:]
gact=lambda u:{k:float(u[j]*H.scale[k]) for j,k in enumerate(AX)}
def sel(idx):
    k=np.argmax(P.rnd(np.concatenate([VOE[:,None,:],Vg],1)[:,:,idx]-VOE[:,None,idx]).min(2),1); return [H.cands['OE'] if kk==0 else gact(U[kk-1]) for kk in k]
pol={'verified grid (sel αβγ)':sel([0,1,2])}
for m in ['GPT','Gemini']:
    for c in ['C4','C4R','C6-E','C6-EX']: pol[f'{m} {c}']=P.acts(P.get('free8','B1','high',c,m))
    for c in ['E1-RL','E1-OE','E1-C4R']: pol[f'{m} {c}']=[json.loads(s) for s in P.gete1(c,m).strict_itt_action_json]
rng=np.random.default_rng(7); NP=20; t=time.time(); out={}
perms=[rng.permutation(n) for _ in range(NP)]
for nm,al in pol.items():
    own=full(list(enumerate(al)))-A0
    mix=np.zeros((NP,n,7))
    for k,pm in enumerate(perms): mix[k]=full([(i,al[pm[i]]) for i in range(n)])-A0
    out[nm+'|own']=own; out[nm+'|mix']=mix.mean(0); print(nm,'%.0fs'%(time.time()-t),flush=True)
np.savez('e12_assign.npz',**out)

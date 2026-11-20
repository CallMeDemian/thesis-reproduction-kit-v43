"""e23: firm-specific part of LLM outputs chosen by a verifier (best-of-10), within applicability strata, same permutations as e16."""
import sys; sys.path.insert(0,'.')
from hz import *
from hz import _target_operating_loss_freq_3y, _state_to_frame_dict
from delta_train import feats_from_state
from fam2 import feats2, make_scorer
import prep2 as P, joblib
S1=make_scorer(joblib.load('delta_models.joblib')); S2=make_scorer(joblib.load('delta2_models.joblib'))
H=Harness(); n=575; ar=np.arange(n)
def full(items):
    rows=[];F1=[];F2=[]
    for i,a in items:
        r,fs,hist,exo=H.firm(i); res=H.bundle.simulate_action(fs,hist,dict(a),action_label='X')[0]
        ex=dict(exo); ex['operating_loss_freq_3y_target']=_target_operating_loss_freq_3y(hist,res.state_t1)
        sv=compute_oracle_variables(res.state_t1,prev_state=fs,exogenous=ex); row=_state_to_frame_dict(sv,r)
        o={v:(row[v] if v in row else r.get(v)) for v in H.need}; o['row_id']=i; rows.append(o); F1.append(feats_from_state(res.state_t1)); F2.append(feats2(res.state_t1))
    s=H.score(rows); return np.concatenate([s[['alpha','beta','gamma']].values,S1(F1),S2(F2)],1)
Z=dict(np.load('e18_b1_all7.npz')); A=dict(np.load('e12_assign.npz')); A9=np.load('A9.npy'); Ag=np.load('Ag.npy')
OE=A9[:,P.IDS.index('OE'),:]; sdg=(Ag-OE[:,None,:]).reshape(-1,7).std(0); T,D,E=[0,1,2],[3,4],[5,6]
key=np.load('e16_strat.npz',allow_pickle=True)['key']; strata=[np.where(key==k)[0] for k in sorted(set(key))]
rng=np.random.default_rng(11); NP=20; perms=[]
for _ in range(NP):
    pm=np.arange(n)
    for idx in strata: pm[idx]=rng.permutation(idx)
    perms.append(pm)
A0=full([(i,H.cands['A0']) for i in range(n)]); out={}
specs=[('free8','high','C4','IC-b',1),('free8','high','C4R','IC-b',1),('free8','high','C4','IC-b',2),('free8','high','C4R','IC-b',2),('free8','high','C4','IC-a',1),('free8','high','C4','IC-c',1),('free8','high','C5','IC-b',1),('candidate9','high','C4','IC-b',1),('candidate9','high','C4R','IC-b',1)]
for m in ['GPT','Gemini']:
    vals=[Z[f'{m}|{a}|{b}|{c}|{d}|{e}'] for a,b,c,d,e in specs]+[A[f'{m} E1-C4R|own']]
    acts=[P.acts(P.get(a,'B1',b,c,m,d,e)) for a,b,c,d,e in specs]+[[json.loads(s) for s in P.gete1('E1-C4R',m).strict_itt_action_json]]
    HI=np.stack(vals,1)
    for sn,sel in [('T',T),('DE',D+E)]:
        k=np.argmax(P.rnd((HI[:,:,sel]/sdg[sel]).mean(2)),1); al=[acts[k[i]][i] for i in range(n)]
        own=full(list(enumerate(al)))-A0; assert np.abs(own-HI[ar,k,:]).max()<1e-9
        mix=np.zeros((NP,n,7))
        for j,pm in enumerate(perms): mix[j]=full([(i,al[pm[i]]) for i in range(n)])-A0
        out[f'{m}|bo10_{sn}|own']=own; out[f'{m}|bo10_{sn}|smix']=mix.mean(0); print(m,sn,'done',flush=True)
np.savez('e23_bo10.npz',**out); print('saved')

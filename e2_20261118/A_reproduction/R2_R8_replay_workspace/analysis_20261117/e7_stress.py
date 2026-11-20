import sys; sys.path.insert(0,'.')
from hz import *
from hz import _target_operating_loss_freq_3y, _state_to_frame_dict
from delta_train import feats_from_state
import prep2 as P, joblib
M=joblib.load('delta_models.joblib')
def dscore(X):
    X=np.asarray(X,float); Z=np.clip(np.where(np.isnan(X),M['med'][None,:],X),M['lo'][None,:],M['hi'][None,:])
    sc=lambda p: np.clip(100*(10-p)/9,0,100)
    return np.stack([sc(M['d1'].predict((Z-M['mu'])/M['sd'])), sc(M['d2'].predict(Z))],1)
H=Harness(); n=575; ar=np.arange(n)
cls=type(H.bundle._simulator); orig=cls.__dict__['_interest'].__func__; PREM={'p':0.0}
def patched(bp,**kw):
    total,parts=orig(bp,**kw); extra=(kw['new_long_tranche']+kw['new_bond_tranche'])*PREM['p']; parts=dict(parts); parts['x']=extra; return total+extra,parts
cls._interest=staticmethod(patched)
def full(items):
    rows=[];F=[]
    for i,a in items:
        r,fs,hist,exo=H.firm(i); res=H.bundle.simulate_action(fs,hist,dict(a),action_label='X')[0]
        ex=dict(exo); ex['operating_loss_freq_3y_target']=_target_operating_loss_freq_3y(hist,res.state_t1)
        sv=compute_oracle_variables(res.state_t1,prev_state=fs,exogenous=ex); row=_state_to_frame_dict(sv,r)
        o={v:(row[v] if v in row else r.get(v)) for v in H.need}; o['row_id']=i; rows.append(o); F.append(feats_from_state(res.state_t1))
    s=H.score(rows); return np.concatenate([s[['alpha','beta','gamma']].values,dscore(F)],1)      # [m,5]
def scaled(a,f): b=dict(a); b['cogs_ratio_chg']=a['cogs_ratio_chg']*f; b['sga_ratio_chg']=a['sga_ratio_chg']*f; return b
# baseline-world selections (verified search): grid chosen by (alpha,beta,gamma) and LOEO pairs
U=np.load('Ugrid.npy'); V9=np.load('V9.npy'); Vg=np.load('Vg.npy'); OEi=P.IDS.index('OE'); VOE=V9[:,OEi,:]
gact=lambda u:{k:float(u[j]*H.scale[k]) for j,k in enumerate(AX)}
def sel(idx):
    k=np.argmax(P.rnd(np.concatenate([VOE[:,None,:],Vg],1)[:,:,idx]-VOE[:,None,idx]).min(2),1)
    return [H.cands['OE'] if kk==0 else gact(U[kk-1]) for kk in k]
pol={'OE':[H.cands['OE']]*n,'RL (C3-E)':[H.cands[P.IDS[k]] for k in P.RLID],'RF':[H.cands['RF']]*n,
     'verified grid (sel αβγ)':sel([0,1,2]),'verified grid (sel βγ)':sel([1,2]),'verified grid (sel αγ)':sel([0,2]),'verified grid (sel αβ)':sel([0,1])}
for m in ['GPT','Gemini']:
    for c in ['C4','C4R','C6-E']: pol[f'{m} {c}']=P.acts(P.get('free8','B1','high',c,m))
EN=['alpha','beta','gamma','δ1','δ2']; out=[]; store={}
def run(tag,f=1.0):
    A0=full([(i,H.cands['A0']) for i in range(n)])
    for nm,al in pol.items():
        v=full([(i,scaled(a,f)) for i,a in enumerate(al)])-A0; store[(tag,nm)]=v
        out.append(dict(scenario=tag,policy=nm,**{e:v[:,j].mean() for j,e in enumerate(EN)}))
t=time.time()
for p in [0.0,0.01,0.02]:
    PREM['p']=p; run(f'RF premium {p*100:.0f}%p'); print('done',p,'%.0fs'%(time.time()-t),flush=True)
PREM['p']=0.0; run('cost-cut realised 50%',0.5)
r=pd.DataFrame(out); pd.set_option('display.width',260); pd.set_option('display.max_rows',300)
for sc_,g in r.groupby('scenario',sort=False): print('\n##',sc_); print(g.drop(columns='scenario').round(3).to_string(index=False))
print('\n== key contrasts with 95% CI ==')
for tag in ['RF premium 0%p','RF premium 1%p','RF premium 2%p','cost-cut realised 50%']:
    print('\n#',tag)
    hold={'alpha':'verified grid (sel βγ)','beta':'verified grid (sel αγ)','gamma':'verified grid (sel αβ)','δ1':'verified grid (sel αβγ)','δ2':'verified grid (sel αβγ)'}
    for j,e in enumerate(EN):
        vs=store[(tag,hold[e])][:,j]
        print(f"  {e:5s} RL−OE {P.ci(store[(tag,'RL (C3-E)')][:,j]-store[(tag,'OE')][:,j])} | verified(held-out)−OE {P.ci(vs-store[(tag,'OE')][:,j])} | verified−RL {P.ci(vs-store[(tag,'RL (C3-E)')][:,j])} | GPT C6-E−C4R {P.ci(store[(tag,'GPT C6-E')][:,j]-store[(tag,'GPT C4R')][:,j])} | Gem C6-E−C4R {P.ci(store[(tag,'Gemini C6-E')][:,j]-store[(tag,'Gemini C4R')][:,j])} | GPT C4−OE {P.ci(store[(tag,'GPT C4')][:,j]-store[(tag,'OE')][:,j])}")
import pickle; pickle.dump(store,open('e7_store.pkl','wb'))

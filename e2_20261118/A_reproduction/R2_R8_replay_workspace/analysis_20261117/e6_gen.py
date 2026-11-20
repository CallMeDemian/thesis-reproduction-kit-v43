import sys; sys.path.insert(0,'.')
from hz import *
from delta_train import feats_from_state, FEATS
import prep2 as P, joblib, itertools
M=joblib.load('delta_models.joblib')
def dscore(X):
    X=np.asarray(X,float); Z=np.clip(np.where(np.isnan(X),M['med'][None,:],X),M['lo'][None,:],M['hi'][None,:])
    sc=lambda p: np.clip(100*(10-p)/9,0,100)
    return np.stack([sc(M['d1'].predict((Z-M['mu'])/M['sd'])), sc(M['d2'].predict(Z))],1)
H=Harness(); n=575
def feats(i,a):
    r,fs,hist,exo=H.firm(i)
    res=H.bundle.simulate_action(fs,hist,dict(a),action_label='X')[0]
    return feats_from_state(res.state_t1)
U=np.load('Ugrid.npy'); AXs=AX
gacts=[{k:float(u[j]*H.scale[k]) for j,k in enumerate(AXs)} for u in U]
t=time.time(); D=np.zeros((n,len(gacts),2))
for i in range(n):
    D[i]=dscore([feats(i,a) for a in gacts])
    if i%100==0: print(i,'%.0fs'%(time.time()-t),flush=True)
np.save('Dgrid_abs.npy',D)
out={}
for m in ['GPT','Gemini']:
    for c in ['C4','C4R','C6-E','C6-EX']:
        for rep in ([1,2] if c!='C6-EX' else [1]):
            al=P.acts(P.get('free8','B1','high',c,m,rep=rep)); out[f'{m}|{c}|high|{rep}']=dscore([feats(i,a) for i,a in enumerate(al)])
    out[f'{m}|C5|high|1']=dscore([feats(i,a) for i,a in enumerate(P.acts(P.get('free8','B1','high','C5',m)))])
    for c in ['E1-C4R','E1-OE','E1-RL']:
        al=[json.loads(s) for s in P.gete1(c,m).strict_itt_action_json]; out[f'{m}|{c}']=dscore([feats(i,a) for i,a in enumerate(al)])
    for c in ['C4','C4R','C6-E']:
        al=P.acts(P.get('candidate9','B1','high',c,m)); out[f'{m}|cand9|{c}']=dscore([feats(i,a) for i,a in enumerate(al)])
np.savez('Darms_abs.npz',**out); print('done %.0fs'%(time.time()-t))

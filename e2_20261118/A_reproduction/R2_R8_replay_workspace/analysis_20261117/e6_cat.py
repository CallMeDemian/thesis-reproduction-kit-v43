import sys; sys.path.insert(0,'.')
from hz import *
from delta_train import feats_from_state
import prep2 as P, joblib
M=joblib.load('delta_models.joblib')
def dscore(X):
    X=np.asarray(X,float); Z=np.clip(np.where(np.isnan(X),M['med'][None,:],X),M['lo'][None,:],M['hi'][None,:])
    sc=lambda p: np.clip(100*(10-p)/9,0,100)
    return np.stack([sc(M['d1'].predict((Z-M['mu'])/M['sd'])), sc(M['d2'].predict(Z))],1)
H=Harness(); n=575
def feats(i,a):
    r,fs,hist,exo=H.firm(i); return feats_from_state(H.bundle.simulate_action(fs,hist,dict(a),action_label='X')[0].state_t1)
D9=np.stack([dscore([feats(i,H.cands[k]) for i in range(n)]) for k in P.IDS],1)   # [n,9,2] absolute
np.save('D9_abs.npy',D9); print('ok',D9.shape)
# decision-year consistency check: features of base FirmState vs A0-simulated (sanity only)

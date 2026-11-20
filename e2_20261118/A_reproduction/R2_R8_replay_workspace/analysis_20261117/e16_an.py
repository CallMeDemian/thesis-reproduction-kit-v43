"""e16 analysis: assignment value of LLM arms within applicability strata (actions permuted only among firms with the same DL/RF/CX applicability)."""
import sys; sys.path.insert(0,'.')
from prep2 import *
pd.set_option('display.width',280)
Z=dict(np.load('e12_assign.npz')); S=dict(np.load('e16_strat.npz',allow_pickle=True)); key=S.pop('key'); A9=np.load('A9.npy'); n=575; ar=np.arange(n)
EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']
app=(np.abs(A9)>1e-9).any(2); k3=np.array([''.join('1' if x else '0' for x in r) for r in app[:,1:4]]); assert (k3==key).all()
def strat_exact(ids):
    out=np.zeros((n,7))
    for k in sorted(set(k3)):
        idx=np.where(k3==k)[0]; ws=np.bincount(ids[idx],minlength=9)/len(idx); out[idx]=(A9[idx]*ws[None,:,None]).sum(1)
    return out
RL=A9[ar,RLID,:]; c2=np.array([IDS.index(c) for c in a3.C2_action]); BC=A9[ar,c2,:]
AVu={}; AVs={}
AVu['RL policy (C3-E)']=RL-(A9*(np.bincount(RLID,minlength=9)/n)[None,:,None]).sum(1); AVs['RL policy (C3-E)']=RL-strat_exact(RLID)
AVu['BC policy (C2)']=BC-(A9*(np.bincount(c2,minlength=9)/n)[None,:,None]).sum(1); AVs['BC policy (C2)']=BC-strat_exact(c2)
for k in sorted({x.split('|')[0] for x in S}): AVu[k]=Z[k+'|own']-Z[k+'|mix']; AVs[k]=Z[k+'|own']-S[k+'|smix']
order=['BC policy (C2)','RL policy (C3-E)','GPT C4','GPT C4R','GPT C6-EX','GPT C6-E','GPT E1-C4R','GPT E1-OE','GPT E1-RL','Gemini C4','Gemini C4R','Gemini C6-EX','Gemini C6-E','Gemini E1-C4R','Gemini E1-OE','Gemini E1-RL']
print('assignment value, unrestricted permutation'); print(pd.DataFrame({k:AVu[k].mean(0) for k in order},index=EN).T.round(3).to_string())
print('\nassignment value within applicability strata'); print(pd.DataFrame({k:AVs[k].mean(0) for k in order},index=EN).T.round(3).to_string())
print('\nwith 95% CI'); print(pd.DataFrame({k:[ci(AVs[k][:,j]) for j in range(7)] for k in order},index=EN).T.to_string())
print('\nshare of the assignment value that remains within strata'); print(pd.DataFrame({k:AVs[k].mean(0)/AVu[k].mean(0) for k in order if not k.startswith('BC')},index=EN).T.round(2).to_string())
print('\nincrements of within-strata assignment value')
for m in ['GPT','Gemini']:
    for a,b in [('C6-E','C4R'),('C6-E','C6-EX'),('C4R','C4'),('E1-RL','E1-C4R'),('E1-OE','E1-C4R'),('E1-RL','E1-OE')]:
        d=AVs[f'{m} {a}']-AVs[f'{m} {b}']; print(f'  {m} {a} − {b}:',' | '.join(f'{e} {ci(d[:,j])}' for j,e in enumerate(EN)))
    for a in ['C4','C4R','C6-E']:
        d=AVs['RL policy (C3-E)']-AVs[f'{m} {a}']; print(f'  RL − {m} {a}:',' | '.join(f'{e} {ci(d[:,j])}' for j,e in enumerate(EN)))
print('\nratio to RL (within strata):'); print(pd.DataFrame({k:AVs[k].mean(0)/AVs['RL policy (C3-E)'].mean(0) for k in order if k.startswith(('GPT C','Gemini C'))},index=EN).T.round(2).to_string())
from scipy.stats import spearmanr
keys=[k for k in order if k.startswith(('GPT C','Gemini C','RL','BC'))]
M=np.array([AVs[k].mean(0) for k in keys]); c=[spearmanr(M[:,a],M[:,b])[0] for a in range(7) for b in range(a+1,7)]
print('policy-ranking agreement across evaluators on within-strata assignment value (%d policies): mean %.2f min %.2f'%(len(keys),np.mean(c),np.min(c)))

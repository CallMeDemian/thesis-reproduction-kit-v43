"""e26d: evaluator-free agreement gate — use the LLM's edit only where independent runs agree on it; otherwise RL."""
import sys; sys.path.insert(0,'/home/claude/harness')
import numpy as np, prep2 as P
A9=np.load('A9.npy'); Ag=np.load('Ag.npy'); n=575; ar=np.arange(n)
OE=A9[:,6,:]; RL=A9[ar,P.RLID,:]; sdg=(Ag-OE[:,None,:]).reshape(-1,7).std(0)
e18=np.load('e18_b1_all7.npz'); N=['α','β','γ','δ1','δ2','ε1','ε2']
def V(m,rep,c='C6-E'): return np.array([P.norm(d) for d in P.acts(P.get('free8','B1','high',c,m,rep=rep))])
def v7(m,rep,c='C6-E'): return e18[f'{m}|free8|high|{c}|IC-b|{rep}']
rlv=np.array([P.norm(P.CAND[P.IDS[k]]) for k in P.RLID])
same=lambda a,b: np.abs(a-b).sum(1)<1e-6
G1,G2,M1,M2=V('GPT',1),V('GPT',2),V('Gemini',1),V('Gemini',2)
rules={'GPT r1=r2, ≠RL':(same(G1,G2)&~same(G1,rlv),v7('GPT',1)),
       'Gemini r1=r2, ≠RL':(same(M1,M2)&~same(M1,rlv),v7('Gemini',1)),
       'GPT r1 = Gemini r1, ≠RL':(same(G1,M1)&~same(G1,rlv),v7('GPT',1)),
       '≥3 of 4 runs identical, ≠RL':None}
# >=3 of 4 identical
allv=[G1,G2,M1,M2]; allval=[v7('GPT',1),v7('GPT',2),v7('Gemini',1),v7('Gemini',2)]
mask=np.zeros(n,bool); pick=np.zeros((n,7))
for i in range(n):
    for a in range(4):
        cnt=sum(np.abs(allv[a][i]-allv[b][i]).sum()<1e-6 for b in range(4))
        if cnt>=3 and np.abs(allv[a][i]-rlv[i]).sum()>1e-6: mask[i]=True; pick[i]=allval[a][i]; break
rules['≥3 of 4 runs identical, ≠RL']=(mask,pick)
print('policy = LLM edit where the rule fires, RL elsewhere; standardized difference vs RL over all 575 firms [95% CI]')
for nm,(msk,val) in rules.items():
    pol=np.where(msk[:,None],val,RL); d=(pol-RL)/sdg
    print(f'\n{nm}: fires on {msk.sum()} firms')
    print('  '+'  '.join(f'{N[j]} {P.ci(d[:,j])}' for j in range(7)))
    print('  family αβγ',P.ci(d[:,:3].mean(1)),' δε',P.ci(d[:,3:].mean(1)))
    if msk.sum()>0:
        dd=d[msk]; print('  within fired firms, mean per firm: αβγ %+.3f  δε %+.3f'%(dd[:,:3].mean(),dd[:,3:].mean()))

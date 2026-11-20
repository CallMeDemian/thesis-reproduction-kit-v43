import sys; sys.path.insert(0,'.')
from prep2 import *
pd.set_option('display.width',260)
V9=np.load('V9.npy'); Vg=np.load('Vg.npy'); U=np.load('Ugrid.npy'); n=575; ar=np.arange(n)
Dabs=np.load('Dgrid_abs.npy'); a0=np.where(np.abs(U).sum(1)<1e-12)[0][0]; A0=Dabs[:,a0,:]; Dg=Dabs-A0[:,None,:]
D9=np.load('D9_abs.npy')-A0[:,None,:]; assert np.abs(D9[:,0,:]).max()<1e-9
arms={k:v-A0 for k,v in np.load('Darms_abs.npz').items()}
DN=['δ1 ridge','δ2 gbm']; OEi=IDS.index('OE')
print('fixed policies under challengers'); print(pd.DataFrame(D9.mean(0),index=IDS,columns=DN).round(3).T.to_string())
DOE=D9[:,OEi,:]; DRL=D9[ar,RLID,:]; VOE=V9[:,OEi,:]; VRL=V9[ar,RLID,:]
print('\nupper bound (own optimum on grid):',Dg.max(1).mean(0).round(3),' catalog-9:',D9.max(1).mean(0).round(3))
rows=[]
def add(name,d): rows.append(dict(policy=name,**{DN[k]:d[:,k].mean() for k in range(2)},**{DN[k]+' vs OE':ci(d[:,k]-DOE[:,k]) for k in range(2)},**{DN[k]+' vs RL':ci(d[:,k]-DRL[:,k]) for k in range(2)}))
add('Heuristic (OE)',DOE); add('RL policy (C3-E)',DRL); add('random (catalog mean)',D9.mean(1))
for m in ['GPT','Gemini']:
    for c in ['C4','C4R','C6-E','C6-EX']: add(f'{m} {c} (free, high)',arms[f'{m}|{c}|high|1'])
# verified selection using the three thesis evaluators -> challenger (variables never used in selection)
def sel(Vs,idx,ref): return np.argmax(rnd(Vs[:,:,idx]-ref[:,None,idx]).min(2),1)
k9=sel(np.concatenate([VOE[:,None,:],V9],1),[0,1,2],VOE); D9x=np.concatenate([DOE[:,None,:],D9],1); add('verified: OE+catalog-9 (select by α,β,γ)',D9x[ar,k9,:])
kg=sel(np.concatenate([VOE[:,None,:],Vg],1),[0,1,2],VOE); Dgx=np.concatenate([DOE[:,None,:],Dg],1); add('verified: OE+grid-606 (select by α,β,γ)',Dgx[ar,kg,:])
kr=sel(np.stack([VOE,VRL],1),[0,1,2],VOE); add('verified: {OE, RL}',np.stack([DOE,DRL],1)[ar,kr,:])
for m in ['GPT','Gemini']:
    names=[f'{m}|{c}|high|{r}' for c in ['C4','C4R','C6-E'] for r in [1,2]]+[f'{m}|C5|high|1',f'{m}|C6-EX|high|1']
    Vl=[]
    for nm in names:
        mm,c,_,r=nm.split('|'); Vl.append(vals(get('free8','B1','high',c,mm,rep=int(r))))
    Vs=np.concatenate([VOE[:,None,:],VRL[:,None,:],np.stack(Vl,1)],1); Ds=np.concatenate([DOE[:,None,:],DRL[:,None,:],np.stack([arms[x] for x in names],1)],1)
    k=sel(Vs,[0,1,2],VOE); add(f'verified: OE+RL+{m} LLM(8, high)',Ds[ar,k,:])
t=pd.DataFrame(rows); print(t.round(3).to_string(index=False))
print('\nreference increments under challengers:')
for m in ['GPT','Gemini']:
    for k in range(2):
        print(m,DN[k],'C6-E−C4R',ci(arms[f'{m}|C6-E|high|1'][:,k]-arms[f'{m}|C4R|high|1'][:,k]),'| C6-E−C6-EX',ci(arms[f'{m}|C6-E|high|1'][:,k]-arms[f'{m}|C6-EX|high|1'][:,k]),
              '| E1: RL−C4R',ci(arms[f'{m}|E1-RL'][:,k]-arms[f'{m}|E1-C4R'][:,k]),'| OE−C4R',ci(arms[f'{m}|E1-OE'][:,k]-arms[f'{m}|E1-C4R'][:,k]),'| RLref−OEref(343)',ci((arms[f'{m}|E1-RL'][:,k]-arms[f'{m}|E1-OE'][:,k])[RLID!=OEi]))
# transfer: optimise on each thesis evaluator / challenger, score on the others (grid, ties->OE)
allV=np.concatenate([np.concatenate([VOE[:,None,:],Vg],1),np.concatenate([DOE[:,None,:],Dg],1)],2); ref=np.concatenate([VOE,DOE],1); names5=E+DN
T=np.zeros((5,5))
for i in range(5):
    k=np.argmax(rnd(allV[:,:,i]-ref[:,None,i]),1)
    for j in range(5): T[i,j]=allV[ar,k,j].mean()
print('\ngrid search on one evaluator (row), scored on each (col); OE row/col for reference:'); print(pd.DataFrame(T,index=names5,columns=names5).round(3).to_string()); print('OE:',ref.mean(0).round(3),' RL:',np.concatenate([VRL,DRL],1).mean(0).round(3))
# correlation of action effects between thesis evaluators and challengers (grid deltas vs OE)
d=(allV[:,1:,:]-ref[:,None,:]).reshape(-1,5); print('\ncorr of action effects (vs OE) across evaluators:'); print(pd.DataFrame(np.corrcoef(d.T),index=names5,columns=names5).round(2).to_string())

import numpy as np, pandas as pd, json, itertools
pd.set_option('display.width',250)
import os; K=os.environ.get('KIT_ROOT','/home/claude/kitzip/THESIS_REPRO_KIT_v2.1.1_FINAL'); DR=K+'/archive/DEPLOYED_RELEASE'
g=pd.read_parquet('grid9.parquet')
IDS=['A0','DL','RF','CX','WC1','WC2','OE','MX1','MX2']; E=['alpha','beta','gamma']
V=np.stack([g.pivot(index='row_id',columns='cand',values=e)[IDS].values for e in E],2)   # [575,9,3] absolute
V=V-V[:,[0],:]                                                                            # policy value vs A0
n=V.shape[0]; OE=IDS.index('OE')
def rnd(x): return np.round(x,9)
print('fixed-policy values'); print(pd.DataFrame(V.mean(0),index=IDS,columns=E).round(3).T.to_string())
# --- agreement
best=[set(np.flatnonzero(rnd(V[i,:,j])==rnd(V[i,:,j]).max()) ) for i in range(n) for j in range(3)]
best=np.array(best,dtype=object).reshape(n,3)
common=np.mean([len(best[i,0]&best[i,1]&best[i,2])>0 for i in range(n)])
ident=np.mean([best[i,0]==best[i,1]==best[i,2] for i in range(n)])
alltie=[np.mean([len(best[i,j])==9 for i in range(n)]) for j in range(3)]
print('common best exists %.3f | identical best sets %.3f | all-tie share'%(common,ident), np.round(alltie,3))
# --- transfer matrix: optimise on i (ties -> average over tied set), evaluate on j
T=np.zeros((3,3)); 
for i in range(3):
    for j in range(3):
        T[i,j]=np.mean([V[f,list(best[f,i]),j].mean() for f in range(n)])
print('transfer: rows=optimised on, cols=scored on'); print(pd.DataFrame(T,index=E,columns=E).round(3).to_string())
print('as share of own optimum:'); print(pd.DataFrame(T/np.diag(T)[None,:],index=E,columns=E).round(3).to_string())
# sign agreement and correlation of action effects (vs OE), non-OE candidates
D=V-V[:,[OE],:]; mask=np.ones(9,bool); mask[OE]=False
d=D[:,mask,:].reshape(-1,3)
for a,b in [(0,1),(0,2),(1,2)]:
    nz=(np.abs(d[:,a])>1e-9)&(np.abs(d[:,b])>1e-9)
    print(E[a],E[b],'pearson %.3f spearman %.3f sign-agree(nonzero both, n=%d) %.3f'%(np.corrcoef(d[:,a],d[:,b])[0,1],pd.Series(d[:,a]).corr(pd.Series(d[:,b]),method='spearman'),nz.sum(),(np.sign(d[nz,a])==np.sign(d[nz,b])).mean()))
# --- selection rules
def select(Vsub, idx, ref):
    """Vsub [n,K,3]; idx selection evaluators; maximin of improvement over ref [n,3]; ties -> first (order = priority)."""
    S=rnd(Vsub[:,:,idx]-ref[:,None,idx]).min(2); return np.argmax(S,1)
def loeo(Vsub, ref, order=None):
    out={}
    for j,e in enumerate(E):
        idx=[i for i in range(3) if i!=j]; k=select(Vsub,idx,ref); out[e]=Vsub[np.arange(n),k,j]
    return out
def insample(Vsub, ref):
    k=select(Vsub,[0,1,2],ref); return {e:Vsub[np.arange(n),k,j] for j,e in enumerate(E)}
def boot(x,B=4000,seed=0):
    r=np.random.default_rng(seed); bs=x[r.integers(0,len(x),(B,len(x)))].mean(1); return x.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5)
a=pd.read_parquet(DR+'/stage5_candidate_iql/C3E_E2_7SEED_BALANCED_DFEBAFA6/C3E_firm_actions.parquet').sort_values('row_id')
p=pd.read_parquet(DR+'/stage5_candidate_iql/C3E_E2_7SEED_BALANCED_DFEBAFA6/C3E_firm_probabilities.parquet').sort_values('row_id')
P=p[['prob_'+k for k in IDS]].values; rl=np.array([IDS.index(x) for x in a.action_id]); assert (P.argmax(1)==rl).all()
RL=V[np.arange(n),rl,:]; VOE=V[:,OE,:]
print('\nRL',RL.mean(0).round(3),'OE',VOE.mean(0).round(3),'upper',V.max(1).mean(0).round(3))
res=[]
def rep(name,Vsub,ref=VOE):
    lo=loeo(Vsub,ref); ins=insample(Vsub,ref)
    row=dict(pool=name)
    for j,e in enumerate(E):
        row[e+'_held']=lo[e].mean(); row[e+'_in']=ins[e].mean()
        m,l,h=boot(lo[e]-RL[:,j]); row[e+'_vsRL']=f'{m:+.3f}[{l:+.3f},{h:+.3f}]'
        m,l,h=boot(lo[e]-VOE[:,j]); row[e+'_vsOE']=f'{m:+.3f}[{l:+.3f},{h:+.3f}]'
    res.append(row)
order_all=[OE]+[i for i in range(9) if i!=OE]         # OE first (default), then catalog order
rep('catalog-9 (verifier only)',V[:,order_all,:])
# RL top-K (+OE default first)
rank=np.argsort(-P,1)
for Kk in [1,2,3,4,6,9]:
    cols=np.concatenate([np.full((n,1),OE),rank[:,:Kk]],1)
    rep(f'OE + RL top-{Kk}',V[np.arange(n)[:,None],cols,:])
# random-K from catalog (+OE), averaged over draws
rng=np.random.default_rng(0)
for Kk in [1,2,3,4,6]:
    acc={e:[] for e in E}
    for d_ in range(200):
        cols=np.stack([rng.choice([i for i in range(9) if i!=OE],Kk,replace=False) for _ in range(n)])
        cols=np.concatenate([np.full((n,1),OE),cols],1); lo=loeo(V[np.arange(n)[:,None],cols,:],VOE)
        for e in E: acc[e].append(lo[e].mean())
    res.append(dict(pool=f'OE + random-{Kk} (mean of 200)',**{e+'_held':np.mean(acc[e]) for e in E}))
t=pd.DataFrame(res); print(t[['pool']+[e+'_held' for e in E]+[e+'_in' for e in E]].round(3).to_string(index=False))
print(t[['pool']+[e+'_vsRL' for e in E]+[e+'_vsOE' for e in E]].dropna().to_string(index=False))
# A0-referenced sensitivity for catalog-9
lo=loeo(V[:,[0]+list(range(1,9)),:],np.zeros((n,3)))
print('catalog-9, maximin vs A0 (ties->A0):',{e:round(lo[e].mean(),3) for e in E})
np.save('V9.npy',V)

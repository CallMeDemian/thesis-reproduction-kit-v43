import sys; sys.path.insert(0,'.')
from prep2 import *
from scipy.stats import rankdata
pd.set_option('display.width',260)
V9=np.load('V9.npy'); Vg=np.load('Vg.npy'); n=575
OEi=IDS.index('OE'); VOE=V9[:,OEi,:]; RL=V9[np.arange(n),RLID,:]
# ---------- (A) sensitivity of the selection rule ----------
def sel_maximin(Vs,idx,ref): return np.argmax(rnd(Vs[:,:,idx]-ref[:,None,idx]).min(2),1)
def sel_sumz(Vs,idx,ref):
    D=Vs[:,:,idx]-ref[:,None,idx]; sd=np.array([D[:,:,k].std() for k in range(len(idx))]); return np.argmax(rnd((D/sd).sum(2)),1)
def sel_borda(Vs,idx,ref):
    R=np.zeros(Vs.shape[:2])
    for k in idx: R+=np.apply_along_axis(lambda v: rankdata(rnd(v),method='average'),1,Vs[:,:,k])
    return np.argmax(rnd(R),1)
def sel_nonharm_sumz(Vs,idx,ref):
    D=rnd(Vs[:,:,idx]-ref[:,None,idx]); sd=np.array([D[:,:,k].std() for k in range(len(idx))]); s=(D/sd).sum(2); s[(D<0).any(2)]=-np.inf; s[:,0]=np.maximum(s[:,0],0); return np.argmax(rnd(s),1)
RULES={'maximin vs OE (primary)':(sel_maximin,VOE),'maximin vs A0':(sel_maximin,np.zeros((n,3))),'sum of z-scores':(sel_sumz,VOE),'Borda rank sum':(sel_borda,VOE),'no-harm then sum-z':(sel_nonharm_sumz,VOE)}
def loeo(Vs,rule):
    f,ref=RULES[rule]; out={}
    for j,e in enumerate(E):
        idx=[i for i in range(3) if i!=j]; k=f(Vs,idx,ref); out[e]=Vs[np.arange(n),k,j]
    return out
pools={'OE+catalog-9':np.concatenate([VOE[:,None,:],V9],1),'OE+grid-606':np.concatenate([VOE[:,None,:],Vg],1)}
rows=[]
for pn,Vs in pools.items():
    for rn in RULES:
        lo=loeo(Vs,rn); rows.append(dict(pool=pn,rule=rn,**{e:lo[e].mean() for e in E},**{e+'_vsRL':ci(lo[e]-RL[:,j]) for j,e in enumerate(E)},**{e+'_vsOE':ci(lo[e]-VOE[:,j]) for j,e in enumerate(E)}))
t=pd.DataFrame(rows); print(t[['pool','rule']+E].round(3).to_string(index=False)); print(t[['pool','rule']+[e+'_vsOE' for e in E]].to_string(index=False))
# single-evaluator selection -> each other evaluator (grid)
print('\nsingle-evaluator verified search on grid (select on row, score on col; ties->OE):')
T=np.zeros((3,3)); Vs=pools['OE+grid-606']
for i in range(3):
    k=sel_maximin(Vs,[i],VOE)
    for j in range(3): T[i,j]=Vs[np.arange(n),k,j].mean()
print(pd.DataFrame(T,index=E,columns=E).round(3).to_string())
# ---------- (D) proposal efficiency at equal K ----------
print('\n=== proposal efficiency: OE + K proposals, verified by two evaluators, scored on the third ===')
def pool_llm(m,which):
    out=[]
    for r in (['high'] if which=='high' else ['low','high']):
        for c in ['C4','C4R','C6-E']:
            for rep in [1,2]: out.append(vals(get('free8','B1',r,c,m,rep=rep)))
        for c in ['C5','C6-EX']: out.append(vals(get('free8','B1',r,c,m)))
    return np.stack(out,1)
rng=np.random.default_rng(1); res=[]
def curve(name,L,Ks,draws=100):
    for Kk in Ks:
        acc={e:[] for e in E}
        for d_ in range(draws if Kk<L.shape[1] else 1):
            cols=np.stack([rng.choice(L.shape[1],Kk,replace=False) for _ in range(n)])
            Vs=np.concatenate([VOE[:,None,:],L[np.arange(n)[:,None],cols,:]],1); lo=loeo(Vs,'maximin vs OE (primary)')
            for e in E: acc[e].append(lo[e].mean())
        res.append(dict(source=name,K=Kk,**{e:np.mean(acc[e]) for e in E}))
for m in ['GPT','Gemini']:
    curve(f'LLM {m} (high reasoning, 8 outputs)',pool_llm(m,'high'),[1,2,4,8])
    curve(f'LLM {m} (all 16 outputs)',pool_llm(m,'all'),[16])
both=np.concatenate([pool_llm('GPT','high'),pool_llm('Gemini','high')],1); curve('LLM both models (high, 16 outputs)',both,[16])
curve('random grid points',Vg,[1,2,4,8,16,32,64],draws=40)
rank=np.argsort(-P3,1)
for Kk in [1,2,3,4,8]:
    Vs=np.concatenate([VOE[:,None,:],V9[np.arange(n)[:,None],rank[:,:Kk],:]],1); lo=loeo(Vs,'maximin vs OE (primary)'); res.append(dict(source='RL top-K (catalog)',K=Kk,**{e:lo[e].mean() for e in E}))
r=pd.DataFrame(res); print(r.round(3).to_string(index=False))

import sys; sys.path.insert(0,'.')
from prep2 import *
pd.set_option('display.width',260)
V9=np.load('V9.npy'); n=575; ar=np.arange(n)
def refval(ids): return V9[ar,[IDS.index(x) for x in ids],:]
def fit(y,x):
    X=np.column_stack([np.ones(len(x)),np.maximum(x,0),np.minimum(x,0)]); return np.linalg.lstsq(X,y,rcond=None)[0]
def bootfit(y,x,B=2000,seed=0):
    r=np.random.default_rng(seed); b=fit(y,x); bs=np.array([fit(y[i],x[i]) for i in r.integers(0,len(x),(B,len(x)))])
    lo,hi=np.percentile(bs,[2.5,97.5],axis=0); d=bs[:,1]-bs[:,2]; return b,lo,hi,np.percentile(d,[2.5,97.5])
inert9=(np.abs(V9).sum(2)<1e-9)                          # candidate has no effect at all for the firm
print('inert share by candidate:',dict(zip(IDS,inert9.mean(0).round(3))))
rows=[]
for m in ['GPT','Gemini']:
    D=get('free8','B1','high','C4',m); R=get('free8','B1','high','C4R',m); A=get('free8','B1','high','C6-E',m); X=get('free8','B1','high','C6-EX',m)
    vD,vR,vA,vX=vals(D),vals(R),vals(A),vals(X); ok=(D.strict_usable&R.strict_usable&A.strict_usable&X.strict_usable).values
    own=refval(A.reference_candidate_id); oth=refval(X.reference_candidate_id)
    in_own=inert9[ar,[IDS.index(x) for x in A.reference_candidate_id]]; in_oth=inert9[ar,[IDS.index(x) for x in X.reference_candidate_id]]
    eR=vals(gete1('E1-C4R',m)); eL=vals(gete1('E1-RL',m)); eO=vals(gete1('E1-OE',m)); oeref=refval(['OE']*n); in_oe=inert9[:,IDS.index('OE')]
    uA=np.stack([norm(a) for a in acts(A)]); uX=np.stack([norm(a) for a in acts(X)]); uR=np.stack([norm(a) for a in acts(R)])
    refA=np.stack([norm(CAND[c]) for c in A.reference_candidate_id]); refX=np.stack([norm(CAND[c]) for c in X.reference_candidate_id])
    copyA=np.abs(uA-refA).sum(1)<1e-6; copyX=np.abs(uX-refX).sum(1)<1e-6
    print(f'\n{m}: inert refs: own {in_own[ok].mean():.3f} other-firm {in_oth[ok].mean():.3f} | copy rate own: inert {copyA[ok&in_own].mean():.3f} vs effective {copyA[ok&~in_own].mean():.3f} | other-firm: inert {copyX[ok&in_oth].mean():.3f} vs effective {copyX[ok&~in_oth].mean():.3f}')
    for j,e in enumerate(E):
        for nm,y,x,mk in [('own RL ref, effective only',vA[:,j]-vR[:,j],own[:,j]-vD[:,j],ok&~in_own),
                          ('other-firm ref, effective only',vX[:,j]-vR[:,j],oth[:,j]-vD[:,j],ok&~in_oth),
                          ('E1-RL own ref, effective only',eL[:,j]-eR[:,j],own[:,j]-vD[:,j],ok&~in_own),
                          ('E1-OE Heuristic ref, effective only',eO[:,j]-eR[:,j],oeref[:,j]-vD[:,j],ok&~in_oe)]:
            b,lo,hi,dci=bootfit(y[mk],x[mk]); w=x[mk]<-1e-9; bt=x[mk]>1e-9
            cp=(copyA if nm.startswith('own') else copyX if nm.startswith('other') else None)
            rows.append(dict(m=m,e=e,arm=nm,n=int(mk.sum()),n_better=int(bt.sum()),n_worse=int(w.sum()),slope_better=f'{b[1]:.2f} [{lo[1]:.2f},{hi[1]:.2f}]',slope_worse=f'{b[2]:.2f} [{lo[2]:.2f},{hi[2]:.2f}]',diff_CI=f'[{dci[0]:+.2f},{dci[1]:+.2f}]',
               copy_better=(cp[mk][bt].mean() if cp is not None else np.nan),copy_worse=(cp[mk][w].mean() if cp is not None else np.nan)))
t=pd.DataFrame(rows)
for e in E: print('\n#####',e); print(t[t.e==e].drop(columns=['e']).round(3).to_string(index=False))

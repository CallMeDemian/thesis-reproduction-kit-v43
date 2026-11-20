import sys; sys.path.insert(0,'.')
from prep2 import *
pd.set_option('display.width',260)
V9=np.load('V9.npy'); n=575; ar=np.arange(n)
def refval(ids): return V9[ar,[IDS.index(x) for x in ids],:]
def fit(y,x):
    X=np.column_stack([np.ones(len(x)),np.maximum(x,0),np.minimum(x,0)]); b=np.linalg.lstsq(X,y,rcond=None)[0]; return b
def bootfit(y,x,B=2000,seed=0):
    r=np.random.default_rng(seed); b=fit(y,x); bs=np.array([fit(y[i],x[i]) for i in r.integers(0,len(x),(B,len(x)))])
    lo,hi=np.percentile(bs,[2.5,97.5],axis=0); d=bs[:,1]-bs[:,2]; return b,lo,hi,np.percentile(d,[2.5,97.5])
rows=[]
for m in ['GPT','Gemini']:
    D=get('free8','B1','high','C4',m); R=get('free8','B1','high','C4R',m); A=get('free8','B1','high','C6-E',m); X=get('free8','B1','high','C6-EX',m)
    vD,vR,vA,vX=vals(D),vals(R),vals(A),vals(X); ok=(D.strict_usable&R.strict_usable&A.strict_usable&X.strict_usable).values
    own=refval(A.reference_candidate_id); oth=refval(X.reference_candidate_id)
    eR=vals(gete1('E1-C4R',m)); eO=vals(gete1('E1-OE',m)); eL=vals(gete1('E1-RL',m))
    oeref=refval(['OE']*n)
    for j,e in enumerate(E):
        specs=[('C6-E: own RL ref',vA[:,j]-vR[:,j],own[:,j]-vD[:,j],ok),
               ('C6-EX: received (other-firm) ref',vX[:,j]-vR[:,j],oth[:,j]-vD[:,j],ok),
               ('C6-EX placebo: own ref NOT shown',vX[:,j]-vR[:,j],own[:,j]-vD[:,j],ok&(X.reference_candidate_id.values!=A.reference_candidate_id.values)),
               ('E1-OE: Heuristic ref',eO[:,j]-eR[:,j],oeref[:,j]-vD[:,j],ok),
               ('E1-RL: own RL ref (new batch)',eL[:,j]-eR[:,j],own[:,j]-vD[:,j],ok)]
        for nm,y,x,mk in specs:
            b,lo,hi,dci=bootfit(y[mk],x[mk])
            rows.append(dict(m=m,e=e,arm=nm,n=int(mk.sum()),share_ref_better=(x[mk]>1e-9).mean(),share_ref_worse=(x[mk]<-1e-9).mean(),
                slope_if_ref_better=f'{b[1]:.2f} [{lo[1]:.2f},{hi[1]:.2f}]',slope_if_ref_worse=f'{b[2]:.2f} [{lo[2]:.2f},{hi[2]:.2f}]',diff_CI=f'[{dci[0]:+.2f},{dci[1]:+.2f}]',
                mean_gain_better=y[mk][x[mk]>1e-9].mean(),mean_gain_worse=y[mk][x[mk]<-1e-9].mean(),mean_gap_better=x[mk][x[mk]>1e-9].mean(),mean_gap_worse=x[mk][x[mk]<-1e-9].mean()))
t=pd.DataFrame(rows)
for e in E:
    print('\n#####',e); print(t[t.e==e].drop(columns=['e']).round(3).to_string(index=False))
t.to_csv('e4_passthrough.csv',index=False)

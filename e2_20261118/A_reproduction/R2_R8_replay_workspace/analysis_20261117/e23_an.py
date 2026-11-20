"""e23 analysis: at the LLM level, is the evaluator-specific excess of a verifier-selected output located in the firm-specific part?
All quantities are increments over the model's own draft (C4), standardized per evaluator, averaged over the consulted family / the other families."""
import sys; sys.path.insert(0,'.')
from prep2 import *
Z=dict(np.load('e18_b1_all7.npz')); Q=dict(np.load('e23_bo10.npz')); A=dict(np.load('e12_assign.npz')); S=dict(np.load('e16_strat.npz',allow_pickle=True))
A9=np.load('A9.npy'); Ag=np.load('Ag.npy'); n=575; ar=np.arange(n); EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; T,D,E=[0,1,2],[3,4],[5,6]
OE=A9[:,IDS.index('OE'),:]; sdg=(Ag-OE[:,None,:]).reshape(-1,7).std(0)
E22=dict(np.load('e22_split.npz'))
rng=np.random.default_rng(0); IDX=rng.integers(0,n,(4000,n))
def fam(x,idx): return (x[:,idx]/sdg[idx]).mean(1)
def cmp(label,inc,home,out):
    h=fam(inc,home); o=fam(inc,out); d=h-o; hb=h[IDX].mean(1); ob=o[IDX].mean(1); db=d[IDX].mean(1)
    return f'{label:58s} consulted {h.mean():+.3f} [{np.percentile(hb,2.5):+.3f},{np.percentile(hb,97.5):+.3f}] | other {o.mean():+.3f} [{np.percentile(ob,2.5):+.3f},{np.percentile(ob,97.5):+.3f}] | difference {d.mean():+.3f} [{np.percentile(db,2.5):+.3f},{np.percentile(db,97.5):+.3f}]'
for m in ['GPT','Gemini']:
    d_own=Z[f'{m}|free8|high|C4|IC-b|1']; d_fs=d_own-S[f'{m} C4|smix']
    a_own=Z[f'{m}|free8|high|C6-E|IC-b|1']; a_fs=a_own-S[f'{m} C6-E|smix']
    print(f'\n===== {m} (increments over the draft; "consulted" = family used by the verifier; for the reference, the thesis family) =====')
    for sn,home,out in [('T',T,D+E),('DE',D+E,T)]:
        o=Q[f'{m}|bo10_{sn}|own']; fs=o-Q[f'{m}|bo10_{sn}|smix']
        print(cmp(f'verifier best-of-10 by {"α,β,γ" if sn=="T" else "δ,ε"}: total',o-d_own,home,out))
        print(cmp(f'   firm-specific part',fs-d_fs,home,out))
        print(cmp(f'   firm-blind part',(o-fs)-(d_own-d_fs),home,out))
    print(cmp('RL reference (C6-E): total',a_own-d_own,T,D+E))
    print(cmp('   firm-specific part',a_fs-d_fs,T,D+E))
    print(cmp('   firm-blind part',(a_own-a_fs)-(d_own-d_fs),T,D+E))
print('\npolicy level (increments over the Heuristic; firm-specific part within strata; Heuristic has none):')
RLo=E22['RL policy|own']; RLs=E22['RL policy|sav']
print(cmp('RL policy: firm-specific part',RLs,T,D+E))
for nm,home,out in [('catalog argmax by α,β,γ',T,D+E),('catalog argmax by δ,ε',D+E,T),('grid argmax by α,β,γ',T,D+E),('grid argmax by δ,ε',D+E,T),('catalog argmax by α',[0],D+E),('grid argmax by α',[0],D+E)]:
    print(cmp(f'{nm}: firm-specific part',E22[f'{nm}|sav'],home,out))
    print(cmp(f'{nm}: firm-blind part',E22[f'{nm}|own']-E22[f'{nm}|sav']-OE,home,out))

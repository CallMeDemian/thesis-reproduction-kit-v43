"""e26c: is LLM run-to-run dispersion an uncertainty signal? (dispersion vs evaluator disagreement, RL fragility, harmful LLM edits)"""
import sys; sys.path.insert(0,'/home/claude/harness')
import numpy as np, prep2 as P
from scipy.stats import spearmanr
A9=np.load('A9.npy'); Ag=np.load('Ag.npy'); n=575; ar=np.arange(n)
OE=A9[:,6,:]; RL=A9[ar,P.RLID,:]; sdg=(Ag-OE[:,None,:]).reshape(-1,7).std(0)
Z9=A9/sdg; Zg=Ag/sdg
fa=lambda Z: Z[...,:3].mean(-1); fb=lambda Z: Z[...,3:].mean(-1)
def V(cond,m,rep): return np.array([P.norm(d) for d in P.acts(P.get('free8','B1','high',cond,m,rep=rep))])
X={(c,m,r):V(c,m,r) for c in ['C6-E','C4'] for m in ['GPT','Gemini'] for r in [1,2]}
def disp(keys):
    vs=[X[k] for k in keys]; d=[np.abs(vs[i]-vs[j]).sum(1) for i in range(len(vs)) for j in range(i+1,len(vs))]; return np.mean(d,0)
D={'C6-E 4 runs':disp([('C6-E',m,r) for m in ['GPT','Gemini'] for r in [1,2]]),
   'C6-E GPT r1-r2':disp([('C6-E','GPT',1),('C6-E','GPT',2)]),
   'C6-E Gemini r1-r2':disp([('C6-E','Gemini',1),('C6-E','Gemini',2)]),
   'C4 4 runs':disp([('C4',m,r) for m in ['GPT','Gemini'] for r in [1,2]])}
rlv=np.array([P.norm(P.CAND[P.IDS[k]]) for k in P.RLID])
D['C6-E mean L1 to RL']=np.mean([np.abs(X[('C6-E',m,r)]-rlv).sum(1) for m in ['GPT','Gemini'] for r in [1,2]],0)
# targets
cat_dis=np.array([1-spearmanr(fa(Z9[i]),fb(Z9[i]))[0] if np.std(fa(Z9[i]))>0 and np.std(fb(Z9[i]))>0 else np.nan for i in range(n)])
grid_dis=np.array([1-spearmanr(fa(Zg[i]),fb(Zg[i]))[0] for i in range(n)])
best_diff=(fa(Z9).argmax(1)!=fb(Z9).argmax(1)).astype(float)
rl_neg=(RL<-1e-9).any(1).astype(float)                       # RL worse than no action under >=1 evaluator
rlz=RL/sdg; spread=rlz.max(1)-rlz.min(1)                      # disagreement of evaluators about RL's own action
# harmful LLM edit: C6-E (GPT r1, Gemini r1) worse than RL on the family mean of both families
def val(m,rep): return np.stack([P.get('free8','B1','high','C6-E',m,rep=rep)['delta_R_score_'+e].values for e in P.E],1)
import numpy as np
e18=np.load('e18_b1_all7.npz')
def v7(m,c,rep): return e18[f'{m}|free8|high|{c}|IC-b|{rep}']
harm=np.mean([((v7(m,'C6-E',r)/sdg).mean(1)<(rlz.mean(1)-0.1)).astype(float) for m in ['GPT','Gemini'] for r in [1,2]],0)
T={'evaluator disagreement, 9 candidates (1-ρ A vs B)':cat_dis,'evaluator disagreement, 606 grid (1-ρ)':grid_dis,
   'best candidate differs A vs B':best_diff,'RL below no-action on >=1 evaluator':rl_neg,'spread of RL value across 7 evaluators':spread,
   'share of 4 C6-E runs worse than RL by >0.1 sd':harm}
print('Spearman ρ between dispersion (rows) and target (cols); [top-quintile − bottom-quintile mean, 95% CI]')
for dn,d in D.items():
    print('\n'+dn, ' mean %.3f'%np.nanmean(d))
    q=np.nanpercentile(d,[20,80]); lo=d<=q[0]; hi=d>=q[1]
    for tn,t in T.items():
        ok=~np.isnan(t); r=spearmanr(d[ok],t[ok])[0]
        rng=np.random.default_rng(0); bs=[]
        ih=np.where(hi&ok)[0]; il=np.where(lo&ok)[0]
        for _ in range(2000): bs.append(t[rng.choice(ih,len(ih))].mean()-t[rng.choice(il,len(il))].mean())
        print(f'  {tn:<52} ρ={r:+.3f}  diff={t[ih].mean()-t[il].mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]')

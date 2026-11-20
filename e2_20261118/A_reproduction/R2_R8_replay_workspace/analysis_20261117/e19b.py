"""e19b: robustness of the pooled system (RL action as default + LLM proposals + verifier from other evaluator families) and what kind of LLM proposal gets selected."""
import sys; sys.path.insert(0,'.')
from prep2 import *
Z=dict(np.load('e18_b1_all7.npz')); A=dict(np.load('e12_assign.npz')); A9=np.load('A9.npy'); n=575; ar=np.arange(n)
EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; RL=A9[ar,RLID,:]; T,D,E=[0,1,2],[3,4],[5,6]
CATV=np.stack([norm(CAND[k]) for k in IDS])
def sel_meanz(P,sel,sd): return np.argmax(rnd((P[:,:,sel]/sd[sel]).mean(2)),1)
def sel_maximin(P,sel,sd): return np.argmax(rnd(((P[:,:,sel]-P[:,[0],:][:,:,sel])/sd[sel]).min(2)),1)      # improvement over the default (RL), worst evaluator
def sel_pareto(P,sel,sd):                                                                                      # only move if no selection evaluator gets worse
    d=rnd(P[:,:,sel]-P[:,[0],:][:,:,sel]); ok=(d>=0).all(2); s=np.where(ok,(d/sd[sel]).mean(2),-1e9); s[:,0]=0; return np.argmax(s,1)
def line(d,idx): return ' | '.join(f'{EN[j]} {ci(d[:,j])}' for j in idx)
for m in ['GPT','Gemini']:
    g=lambda mode,r,c,info='IC-b',rep=1: get(mode,'B1',r,c,m,info,rep)
    k=lambda mode,r,c,info='IC-b',rep=1: Z[f'{m}|{mode}|{r}|{c}|{info}|{rep}']
    specs=[('free8','high','C4','IC-b',1),('free8','high','C4R','IC-b',1),('free8','high','C4','IC-b',2),('free8','high','C4R','IC-b',2),('free8','high','C4','IC-a',1),('free8','high','C4','IC-c',1),('free8','high','C5','IC-b',1),('candidate9','high','C4','IC-b',1),('candidate9','high','C4R','IC-b',1)]
    advs=[('free8','high','C6-E','IC-b',1),('free8','high','C6-E','IC-b',2),('candidate9','high','C6-E','IC-b',1)]
    HI=[k(*s) for s in specs]+[A[f'{m} E1-C4R|own']]; ADV=[k(*s) for s in advs]+[A[f'{m} E1-RL|own']]
    U=[np.stack([norm(a) for a in acts(g(*s))]) for s in specs]+[np.stack([norm(json.loads(x)) for x in gete1('E1-C4R',m).strict_itt_action_json])]+[np.stack([norm(a) for a in acts(g(*s))]) for s in advs]+[np.stack([norm(json.loads(x)) for x in gete1('E1-RL',m).strict_itt_action_json])]
    P=np.stack([RL]+HI+ADV,1); sd=np.stack(HI,1).reshape(-1,7).std(0)
    print(f'\n===== {m}: pool = RL action + 14 LLM outputs =====')
    for rn,rule in [('mean-z',sel_meanz),('maximin vs RL',sel_maximin),('no selection evaluator worse',sel_pareto)]:
        for sn,sel,hold in [('δ+ε',D+E,T),('δ',D,T),('ε',E,T),('α,β,γ',T,D+E),('β,γ',[1,2],[0])]:
            kk=rule(P,sel,sd); b=P[ar,kk,:]; print(f'{rn:28s} select by {sn:6s} keeps RL {np.mean(kk==0):.3f} | system − RL: {line(b-RL,hold)}')
    kk=sel_meanz(P,D+E,sd); b=P[ar,kk,:]; mv=kk>0
    u=np.stack([U[j-1][i] if j>0 else CATV[RLID[i]] for i,j in enumerate(kk)])
    incat=np.array([np.min(np.abs(CATV-u[i]).sum(1))<1e-6 for i in range(n)])
    print('firms where an LLM proposal replaces the RL action: %.3f | of those, proposal is not one of the 9 catalog actions: %.3f | mean total size of selected proposals %.2f'%(mv.mean(),(~incat[mv]).mean(),np.abs(u[mv]).sum(1).mean()))
    names=['CX','DL','RF','INV','AR','AP','COGS','SGA']; print('   budget share of selected LLM proposals:',dict(zip(names,(np.abs(u[mv]).mean(0)).round(3))),'| of the RL actions they replace:',dict(zip(names,np.abs(CATV[RLID[mv]]).mean(0).round(3))))
    print('   on α: gain over RL where replaced %.3f (n=%d), share of replaced firms with higher/equal/lower α: %.3f / %.3f / %.3f'%((b-RL)[mv,0].mean(),mv.sum(),((b-RL)[mv,0]>1e-9).mean(),(np.abs((b-RL)[mv,0])<=1e-9).mean(),((b-RL)[mv,0]<-1e-9).mean()))
    # single-evaluator firm-level check on the thesis's main evaluator: bootstrap p-like summary
    d=(b-RL)[:,0]; r=np.random.default_rng(0); bs=d[r.integers(0,n,(20000,n))].mean(1); print('   α difference %.3f, share of bootstrap draws <= 0: %.4f'%(d.mean(),(bs<=0).mean()))

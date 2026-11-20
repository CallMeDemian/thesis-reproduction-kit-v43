"""e19: does a generate-and-verify system that pools the RL policy's action with the LLM's own proposals beat the RL policy alone
on evaluators that were not used for the selection?  (ties keep the first candidate = the RL action)"""
import sys; sys.path.insert(0,'.')
from prep2 import *
Z=dict(np.load('e18_b1_all7.npz')); A=dict(np.load('e12_assign.npz')); A9=np.load('A9.npy'); n=575; ar=np.arange(n)
EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; RL=A9[ar,RLID,:]; OE=A9[:,IDS.index('OE'),:]
T,D,E=[0,1,2],[3,4],[5,6]
def bon(P,sel,sd): s=(P[:,:,sel]/sd[sel]).mean(2); k=np.argmax(rnd(s),1); return P[ar,k,:],k
def line(d,idx): return ' | '.join(f'{EN[j]} {ci(d[:,j])}' for j in idx)
for m in ['GPT','Gemini']:
    k=lambda mode,r,c,info='IC-b',rep=1: Z[f'{m}|{mode}|{r}|{c}|{info}|{rep}']
    draft=k('free8','high','C4'); adv=k('free8','high','C6-E')
    HI=[draft,k('free8','high','C4R'),k('free8','high','C4',rep=2),k('free8','high','C4R',rep=2),k('free8','high','C4','IC-a'),k('free8','high','C4','IC-c'),k('free8','high','C5'),k('candidate9','high','C4'),k('candidate9','high','C4R'),A[f'{m} E1-C4R|own']]
    ADV=[adv,k('free8','high','C6-E',rep=2),k('candidate9','high','C6-E'),A[f'{m} E1-RL|own']]
    sd=np.stack(HI,1).reshape(-1,7).std(0)
    print(f'\n===== {m} =====')
    for nm,pool in [('RL + Heuristic',[RL,OE]),('RL + 10 no-advice LLM outputs',[RL]+HI),('RL + 4 advised LLM outputs',[RL]+ADV),('RL + all 14 LLM outputs',[RL]+HI+ADV),('RL + Heuristic + all 14 LLM outputs',[RL,OE]+HI+ADV)]:
        P=np.stack(pool,1)
        for sn,sel,hold in [('δ+ε',D+E,T),('α,β,γ',T,D+E),('α,β,γ+δ',T+D,E),('α,β,γ+ε',T+E,D)]:
            b,kk=bon(P,sel,sd); print(f'{nm:38s} select by {sn:9s} | keeps RL action {np.mean(kk==0):.3f} | system − RL on held-out: {line(b-RL,hold)}')
    # same systems against the advised revision alone and the draft, on alpha with the δ+ε selector
    P=np.stack([RL]+HI+ADV,1); b,kk=bon(P,D+E,sd)
    print('RL + all 14 LLM outputs, δ+ε selector: value on α %.3f (RL %.3f, advised revision %.3f, draft %.3f) | minus advised revision: %s'%(b[:,0].mean(),RL[:,0].mean(),adv[:,0].mean(),draft[:,0].mean(),line(b-adv,T)))
    # which source wins the selection
    src=np.array(['RL']+['LLM no advice']*10+['LLM advised']*4); print('   selected source share:',pd.Series(src[kk]).value_counts(normalize=True).round(3).to_dict())
    # LLM-only systems against RL (no RL action in the pool)
    for nm,pool in [('10 no-advice LLM outputs',HI),('all 14 LLM outputs',HI+ADV)]:
        P=np.stack(pool,1)
        for sn,sel,hold in [('δ+ε',D+E,T),('α,β,γ',T,D+E)]:
            b,_=bon(P,sel,sd); print(f'{nm:38s} select by {sn:9s} | system − RL on held-out: {line(b-RL,hold)}')

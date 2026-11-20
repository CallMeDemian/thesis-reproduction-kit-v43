"""e20: equal-cost comparison on evaluators not used for selection.
 advice route   = draft + one revision with the RL reference (2 LLM calls)
 verifier route = draft + one further sample, choose by evaluators from the other families (2 LLM calls) ; also best-of-10"""
import sys; sys.path.insert(0,'.')
from prep2 import *
Z=dict(np.load('e18_b1_all7.npz')); A=dict(np.load('e12_assign.npz')); n=575; ar=np.arange(n)
EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; T,D,E=[0,1,2],[3,4],[5,6]; OTHER={0:D+E,1:D+E,2:D+E,3:T+E,4:T+E,5:T+D,6:T+D}
def bon(P,sel,sd): return P[ar,np.argmax(rnd((P[:,:,sel]/sd[sel]).mean(2)),1),:]
for m in ['GPT','Gemini']:
    k=lambda mode,r,c,info='IC-b',rep=1: Z[f'{m}|{mode}|{r}|{c}|{info}|{rep}']
    draft=k('free8','high','C4'); adv=k('free8','high','C6-E'); selfrev=k('free8','high','C4R')
    HI=np.stack([draft,selfrev,k('free8','high','C4',rep=2),k('free8','high','C4R',rep=2),k('free8','high','C4','IC-a'),k('free8','high','C4','IC-c'),k('free8','high','C5'),k('candidate9','high','C4'),k('candidate9','high','C4R'),A[f'{m} E1-C4R|own']],1)
    sd=HI.reshape(-1,7).std(0); P2=np.stack([draft,k('free8','high','C4',rep=2)],1)
    print(f'\n===== {m} (each column: scored on that evaluator, selection by the two other families) =====')
    b2=np.stack([bon(P2,OTHER[j],sd)[:,j] for j in range(7)],1); b10=np.stack([bon(HI,OTHER[j],sd)[:,j] for j in range(7)],1)
    i2=np.stack([bon(P2,[j],sd)[:,j] for j in range(7)],1); i10=np.stack([bon(HI,[j],sd)[:,j] for j in range(7)],1)
    f=lambda d: ' | '.join(f'{EN[j]} {ci(d[:,j])}' for j in range(7))
    print('self-review − draft            :',f(selfrev-draft))
    print('advice − draft                 :',f(adv-draft))
    print('best-of-2 (held-out) − draft   :',f(b2-draft))
    print('best-of-10 (held-out) − draft  :',f(b10-draft))
    print('best-of-10 (in-sample) − draft :',f(i10-draft))
    print('advice − best-of-2 (held-out)  :',f(adv-b2))
    print('advice − best-of-10 (held-out) :',f(adv-b10))
    g=lambda x:(x-draft).mean(0)
    print('ratio in-sample / held-out gain, best-of-10:',dict(zip(EN,(g(i10)/g(b10)).round(1))),'| best-of-2:',dict(zip(EN,(g(i2)/g(b2)).round(1))))
    print('held-out gain as share of in-sample, best-of-10:',dict(zip(EN,(g(b10)/g(i10)).round(2))))
    same=np.mean([np.abs(P2[:,0,:]-P2[:,1,:]).sum(1)<1e-9]); print('second sample identical to the draft (same 7 scores): %.3f'%same)

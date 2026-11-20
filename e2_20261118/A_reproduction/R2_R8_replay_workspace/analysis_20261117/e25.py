"""e25: headroom for a simulator-grounded revision method, estimated without new LLM calls.
A judge that sees only next-year financial outcomes is stood in for by evaluator families whose variables differ from the
evaluator being scored (leave-one-family-out). RL action is always the first candidate (ties keep RL)."""
import sys; sys.path.insert(0,'.')
from prep2 import *
Z=dict(np.load('e18_b1_all7.npz')); A=dict(np.load('e12_assign.npz')); A9=np.load('A9.npy'); Ag=np.load('Ag.npy'); n=575; ar=np.arange(n)
EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; T,D,E=[0,1,2],[3,4],[5,6]; OTHER={0:D+E,1:D+E,2:D+E,3:T+E,4:T+E,5:T+D,6:T+D}
OE=A9[:,IDS.index('OE'),:]; RL=A9[ar,RLID,:]; sdg=(Ag-OE[:,None,:]).reshape(-1,7).std(0)
def lofo(P):   # value on each evaluator j when the choice is made by the families not containing j
    out=np.zeros((n,7))
    for j in range(7):
        sel=OTHER[j]; k=np.argmax(rnd((P[:,:,sel]/sdg[sel]).mean(2)),1); out[:,j]=P[ar,k,j]
    return out
def line(d): return ' | '.join(f'{EN[j]} {ci(d[:,j])}' for j in range(7))
def hs(P): return ' | '.join(f'{EN[j]} {(P[:,:,j].max(1)-RL[:,j]).mean():+.3f}' for j in range(7))
G=np.concatenate([RL[:,None,:],Ag],1)
print('RL + 606 grid actions (model-free)')
print('  hindsight ceiling − RL:',hs(G))
print('  judged by other families − RL:',line(lofo(G)-RL))
for m in ['GPT','Gemini']:
    k=lambda mode,r,c,info='IC-b',rep=1: Z[f'{m}|{mode}|{r}|{c}|{info}|{rep}']
    C4=k('free8','high','C4'); C6=k('free8','high','C6-E')
    pool=[k('free8','high','C4'),k('free8','high','C4R'),k('free8','high','C4',rep=2),k('free8','high','C4R',rep=2),k('free8','high','C4','IC-a'),k('free8','high','C4','IC-c'),k('free8','high','C5'),k('candidate9','high','C4'),k('candidate9','high','C4R'),A[f'{m} E1-C4R|own'],C6,k('free8','high','C6-E',rep=2),k('candidate9','high','C6-E'),A[f'{m} E1-RL|own']]
    print(f'\n===== {m} =====  (C6-E − RL: {line(C6-RL)})')
    for nm,P in [('{RL, own draft C4}',np.stack([RL,C4],1)),('{RL, C4, C6-E}',np.stack([RL,C4,C6],1)),('{RL + 14 LLM outputs}',np.stack([RL]+pool,1)),('{RL + 14 LLM outputs + 606 grid}',np.concatenate([np.stack([RL]+pool,1),Ag],1))]:
        L=lofo(P)
        print(f'{nm:34s} hindsight ceiling − RL: {hs(P)}')
        print(f'{"":34s} judged by other families − RL : {line(L-RL)}')
        print(f'{"":34s} judged by other families − C6-E: {line(L-C6)}')

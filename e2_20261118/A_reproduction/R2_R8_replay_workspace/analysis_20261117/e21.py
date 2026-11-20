"""e21: does a gain obtained by consulting an evaluator stay with that evaluator family?
For each method: standardized gain on the 'home' family (the family used to select/tune; for evaluator-free methods the thesis family,
since the RL configuration was chosen on alpha) vs on the other families. retention = out-family gain / home-family gain."""
import sys; sys.path.insert(0,'.')
from prep2 import *
A9=np.load('A9.npy'); Ag=np.load('Ag.npy'); Z=dict(np.load('e18_b1_all7.npz')); A=dict(np.load('e12_assign.npz'))
n=575; ar=np.arange(n); EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; T,D,E=[0,1,2],[3,4],[5,6]
OEi=IDS.index('OE'); OE=A9[:,OEi,:]; RL=A9[ar,RLID,:]
c2=np.array([IDS.index(c) for c in a3.C2_action]); BC=A9[ar,c2,:]
sdg=(Ag-OE[:,None,:]).reshape(-1,7).std(0)
def zg(X,ref,idx): return ((X-ref)[:,idx]/sdg[idx]).mean(1)
def pick(P,sel): return P[ar,np.argmax(rnd((P[:,:,sel]/sdg[sel]).mean(2)),1),:]
C=np.concatenate([OE[:,None,:],np.delete(A9,OEi,1)],1)          # catalog, OE first -> ties keep the Heuristic
G=np.concatenate([OE[:,None,:],Ag],1)
rng=np.random.default_rng(0); B=4000; IDX=rng.integers(0,n,(B,n))
def row(name,level,X,ref,home,out):
    h=zg(X,ref,home); o=zg(X,ref,out); hb=h[IDX].mean(1); ob=o[IDX].mean(1); rb=ob/hb
    return dict(level=level,method=name,home='+'.join(EN[i] for i in home),home_gain=f'{h.mean():+.3f} [{np.percentile(hb,2.5):+.3f},{np.percentile(hb,97.5):+.3f}]',
                out_gain=f'{o.mean():+.3f} [{np.percentile(ob,2.5):+.3f},{np.percentile(ob,97.5):+.3f}]',retention=f'{o.mean()/h.mean():.2f} [{np.percentile(rb,2.5):.2f},{np.percentile(rb,97.5):.2f}]',
                h=h.mean(),o=o.mean())
rows=[]
# policy level, reference = Heuristic
rows.append(row('RL policy (no evaluator in reward)','policy',RL,OE,T,D+E))
rows.append(row('behaviour cloning','policy',BC,OE,T,D+E))
rows.append(row('catalog argmax by α,β,γ','policy',pick(C,T),OE,T,D+E))
rows.append(row('catalog argmax by α','policy',pick(C,[0]),OE,[0],D+E))
rows.append(row('catalog argmax by δ,ε','policy',pick(C,D+E),OE,D+E,T))
rows.append(row('grid argmax by α,β,γ','policy',pick(G,T),OE,T,D+E))
rows.append(row('grid argmax by α','policy',pick(G,[0]),OE,[0],D+E))
rows.append(row('grid argmax by δ,ε','policy',pick(G,D+E),OE,D+E,T))
# LLM level, reference = the model's own draft
for m in ['GPT','Gemini']:
    k=lambda mode,r,c,info='IC-b',rep=1: Z[f'{m}|{mode}|{r}|{c}|{info}|{rep}']
    draft=k('free8','high','C4')
    HI=np.stack([draft,k('free8','high','C4R'),k('free8','high','C4',rep=2),k('free8','high','C4R',rep=2),k('free8','high','C4','IC-a'),k('free8','high','C4','IC-c'),k('free8','high','C5'),k('candidate9','high','C4'),k('candidate9','high','C4R'),A[f'{m} E1-C4R|own']],1)
    P2=np.stack([draft,k('free8','high','C4',rep=2)],1)
    rows.append(row(f'{m}: RL reference (C6-E)','LLM',k('free8','high','C6-E'),draft,T,D+E))
    rows.append(row(f'{m}: RL reference, E1 (vs E1 self-review)','LLM',A[f'{m} E1-RL|own'],A[f'{m} E1-C4R|own'],T,D+E))
    rows.append(row(f'{m}: verifier best-of-10 by α,β,γ','LLM',pick(HI,T),draft,T,D+E))
    rows.append(row(f'{m}: verifier best-of-10 by α','LLM',pick(HI,[0]),draft,[0],D+E))
    rows.append(row(f'{m}: verifier best-of-10 by δ,ε','LLM',pick(HI,D+E),draft,D+E,T))
    rows.append(row(f'{m}: verifier best-of-2 by α,β,γ','LLM',pick(P2,T),draft,T,D+E))
    rows.append(row(f'{m}: self-review (C4R)','LLM',k('free8','high','C4R'),draft,T,D+E))
t=pd.DataFrame(rows); pd.set_option('display.width',250); pd.set_option('display.max_colwidth',60)
print(t.drop(columns=['h','o']).to_string(index=False))
t[['level','method','home','h','o']].to_csv('e21_transfer.csv',index=False)
# catalog, leave-one-family-out, raw values vs the RL policy (same 9 actions)
print('\ncatalog policies (same 9 actions as the RL policy), value on held-out evaluators minus RL policy:')
for sn,sel,hold in [('α,β,γ',T,D+E),('δ,ε',D+E,T),('α,β,γ,δ',T+D,E),('α,β,γ,ε',T+E,D),('α',[0],D+E)]:
    X=pick(C,sel); print(f'  select by {sn:9s}:',' | '.join(f'{EN[j]} {ci((X-RL)[:,j])}' for j in hold))

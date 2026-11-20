import sys; sys.path.insert(0,'.')
from prep2 import *
pd.set_option('display.width',280)
A9=np.load('A9.npy'); n=575; ar=np.arange(n); EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; OEi=IDS.index('OE')
RL=A9[ar,RLID,:]; OE=A9[:,OEi,:]
x=get('free8','B1','high','C6-EX','GPT'); perm=np.array([IDS.index(c) for c in x.reference_candidate_id]); SH=A9[ar,perm,:]
c2=np.array([IDS.index(c) for c in a3.C2_action]); BC=A9[ar,c2,:]
# expected value under RL's marginal action distribution (analytic shuffle)
w=np.bincount(RLID,minlength=9)/n; MARG=(A9*w[None,:,None]).sum(1)
rows=[]
for nm,v in [('Heuristic (OE)',OE),('BC policy (C2)',BC),('RL policy (C3-E)',RL),('RL actions, permuted across firms',SH),('RL action mix, firm-blind (expected)',MARG),('random (catalog mean)',A9.mean(1)),('upper (ex-post best of 9)',A9.max(1))]:
    rows.append(dict(policy=nm,**{e:v[:,j].mean() for j,e in enumerate(EN)}))
print(pd.DataFrame(rows).round(3).to_string(index=False))
for nm,d in [('RL − OE',RL-OE),('RL − BC',RL-BC),('RL − permuted RL',RL-SH),('RL − firm-blind mix',RL-MARG),('firm-blind mix − OE',MARG-OE),('BC − OE',BC-OE)]:
    print(f'{nm:22s}',' | '.join(f'{e} {ci(d[:,j])}' for j,e in enumerate(EN)))
print('C2 actions:',a3.C2_action.value_counts().to_dict())
# share of RL's advantage over OE explained by firm-specific assignment vs action mix
print('decomposition RL−OE = (mix−OE) + (RL−mix):')
for j,e in enumerate(EN): print(f'  {e}: total {(RL-OE)[:,j].mean():+.3f} = mix {(MARG-OE)[:,j].mean():+.3f} + assignment {(RL-MARG)[:,j].mean():+.3f}')

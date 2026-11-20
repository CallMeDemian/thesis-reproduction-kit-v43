"""e22: split each policy's value into the firm-blind part (its action distribution given to a random firm) and the
firm-specific part (assignment value), for policies that consulted an evaluator vs the RL policy that never did.
Exact: V_mix(i) = mean_j V(firm i, action chosen for firm j); also within applicability strata (DL/RF/CX usable or not)."""
import sys; sys.path.insert(0,'.')
from prep2 import *
A9=np.load('A9.npy'); Ag=np.load('Ag.npy'); n=575; ar=np.arange(n)
EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; T,D,E=[0,1,2],[3,4],[5,6]; FAM={0:'T',1:'T',2:'T',3:'D',4:'D',5:'E',6:'E'}
OEi=IDS.index('OE'); OE=A9[:,OEi,:]
sdg=(Ag-OE[:,None,:]).reshape(-1,7).std(0)
C=np.concatenate([OE[:,None,:],np.delete(A9,OEi,1)],1); cmap=np.array([OEi]+[i for i in range(9) if i!=OEi])
G=np.concatenate([OE[:,None,:],Ag],1)
app=(np.abs(A9)>1e-9).any(2); key=np.array([''.join('1' if x else '0' for x in r) for r in app[:,1:4]])
def argpick(P,sel): return np.argmax(rnd((P[:,:,sel]/sdg[sel]).mean(2)),1)
def split(P,k):
    own=P[ar,k,:]; mix=P[:,k,:].mean(1)                         # firm i scored with every firm's chosen action
    smix=np.zeros_like(own)
    for s in set(key):
        idx=np.where(key==s)[0]; smix[idx]=P[idx][:,k[idx],:].mean(1)
    return own,own-mix,own-smix
pol={}
pol['RL policy']=split(A9,RLID)
c2=np.array([IDS.index(c) for c in a3.C2_action]); pol['behaviour cloning']=split(A9,c2)
for nm,sel in [('α,β,γ',T),('α',[0]),('β',[1]),('γ',[2]),('δ,ε',D+E),('δ',D),('ε',E)]:
    pol[f'catalog argmax by {nm}']=split(C,argpick(C,sel))
    pol[f'grid argmax by {nm}']=split(G,argpick(G,sel))
rows=[]
for nm,(own,av,sav) in pol.items():
    rows.append(dict(policy=nm,**{f'{e}':f'{av[:,j].mean():.3f}/{sav[:,j].mean():.3f}' for j,e in enumerate(EN)}))
pd.set_option('display.width',250)
print('assignment value (firm-specific part): all-firm shuffle / within applicability strata')
print(pd.DataFrame(rows).to_string(index=False))
RLo,RLa,RLs=pol['RL policy']
print('\nfirm-specific part of evaluator-consulting policies minus the RL policy (within strata), on evaluators NOT consulted:')
for nm,sel in [('α,β,γ',T),('α',[0]),('β',[1]),('γ',[2]),('δ,ε',D+E),('δ',D),('ε',E)]:
    fams={FAM[j] for j in sel}; hold=[j for j in range(7) if FAM[j] not in fams]
    for lev in ['catalog','grid']:
        o,a,s=pol[f'{lev} argmax by {nm}']
        print(f'  {lev:7s} by {nm:6s} | firm-specific:',' | '.join(f'{EN[j]} {ci((s-RLs)[:,j])}' for j in hold))
        print(f'  {"":7s}    {"":6s} | total value  :',' | '.join(f'{EN[j]} {ci((o-RLo)[:,j])}' for j in hold))
print('\nsame comparison on the consulted evaluators (in-sample), catalog by α,β,γ:')
o,a,s=pol['catalog argmax by α,β,γ']; print('  firm-specific:',' | '.join(f'{EN[j]} {ci((s-RLs)[:,j])}' for j in T)); print('  total value  :',' | '.join(f'{EN[j]} {ci((o-RLo)[:,j])}' for j in T))
o,a,s=pol['catalog argmax by δ,ε']; print('catalog by δ,ε in-sample firm-specific:',' | '.join(f'{EN[j]} {ci((s-RLs)[:,j])}' for j in D+E))
# standardized summary: firm-specific part on consulted vs not-consulted families
print('\nstandardized firm-specific part (within strata), mean over evaluators: consulted family | other families')
for nm,sel in [('α,β,γ',T),('α',[0]),('δ,ε',D+E)]:
    fams={FAM[j] for j in sel}; ins=[j for j in range(7) if FAM[j] in fams]; hold=[j for j in range(7) if FAM[j] not in fams]
    for lev in ['catalog','grid']:
        s=pol[f'{lev} argmax by {nm}'][2]; print(f'  {lev:7s} by {nm:6s}: {(s[:,ins]/sdg[ins]).mean():.3f} | {(s[:,hold]/sdg[hold]).mean():.3f}   (RL policy on the same sets: {(RLs[:,ins]/sdg[ins]).mean():.3f} | {(RLs[:,hold]/sdg[hold]).mean():.3f})')
np.savez('e22_split.npz',**{k+'|own':v[0] for k,v in pol.items()},**{k+'|av':v[1] for k,v in pol.items()},**{k+'|sav':v[2] for k,v in pol.items()})

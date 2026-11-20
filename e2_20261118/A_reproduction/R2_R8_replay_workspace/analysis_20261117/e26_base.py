"""e26a: how strong are trivial constant policies relative to RL C3-E on the 575 test firms (7 evaluators)?"""
import sys; sys.path.insert(0,'/home/claude/harness')
import numpy as np, prep2 as P
A9=np.load('A9.npy'); Ag=np.load('Ag.npy'); U=np.load('Ugrid.npy')
n=575; ar=np.arange(n); OE=A9[:,6,:]; RL=A9[ar,P.RLID,:]
sdg=(Ag-OE[:,None,:]).reshape(-1,7).std(0)
N=['α','β','γ','δ1','δ2','ε1','ε2']
def row(nm,V): print(f'{nm:<28}'+' '.join(f'{x:7.3f}' for x in V.mean(0)))
print(f'{"policy (raw value vs no action)":<28}'+' '.join(f'{x:>7}' for x in N))
row('RL C3-E',RL); row('OE heuristic',OE)
for k,nm in enumerate(P.IDS): row('const '+nm,A9[:,k,:])
gi=lambda v: int(np.where((np.abs(U-np.array(v))<1e-9).all(1))[0][0])
ic=gi([0,0,0,0,0,0,-1,0]); isg=gi([0,0,0,0,0,0,0,-1])
row('const max COGS cut (grid)',Ag[:,ic,:]); row('const max SG&A cut (grid)',Ag[:,isg,:])
best=Ag.mean(0).argmax(0); print('best constant grid action per evaluator:')
for j in range(7): print(' ',N[j],U[best[j]].round(2),'value %.3f (RL %.3f)'%(Ag[:,best[j],j].mean(),RL[:,j].mean()))
print('\nstandardized RL − constant max COGS cut, per evaluator [95% CI]:')
d=(RL-Ag[:,ic,:])/sdg
for j in range(7): print(' ',N[j],P.ci(d[:,j]))
print('family means: αβγ',P.ci(d[:,:3].mean(1)),' δε',P.ci(d[:,3:].mean(1)))
print('\nstandardized RL − best catalog constant (chosen per evaluator on these same firms):')
for j in range(7):
    k=A9[:,:,j].mean(0).argmax(); print(' ',N[j],P.IDS[k],P.ci((RL[:,j]-A9[:,k,j])/sdg[j]))

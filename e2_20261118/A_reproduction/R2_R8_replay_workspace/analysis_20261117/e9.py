import sys; sys.path.insert(0,'.')
from prep2 import *
pd.set_option('display.width',260)
V9=np.load('V9.npy'); Vg=np.load('Vg.npy'); U=np.load('Ugrid.npy'); n=575; ar=np.arange(n); OEi=IDS.index('OE')
a0=np.where(np.abs(U).sum(1)<1e-12)[0][0]
def load(pref,armsf):
    G=np.load(pref+'grid_abs.npy'); A0=G[:,a0,:]; C=np.load(pref[0]+'9_abs.npy')-A0[:,None,:]; return G-A0[:,None,:],C,{k:v-A0 for k,v in np.load(armsf).items()}
Dg,D9,Da=load('D','Darms_abs.npz'); Eg,E9,Ea=load('E','Earms_abs.npz')
Da={k.replace('|high|1',''):v for k,v in Da.items() if k.endswith('|high|1') or 'E1' in k}
X9=np.concatenate([D9,E9],2); Xg=np.concatenate([Dg,Eg],2); CN=['δ1','δ2','ε1','ε2']
arm=lambda k: np.concatenate([Da[k],Ea[k]],1)
XOE=X9[:,OEi,:]; XRL=X9[ar,RLID,:]; VOE=V9[:,OEi,:]; VRL=V9[ar,RLID,:]
print('fixed policies under 4 challengers'); print(pd.DataFrame(X9.mean(0),index=IDS,columns=CN).round(3).T.to_string())
print('own optimum on grid:',dict(zip(CN,Xg.max(1).mean(0).round(3))))
def sel(Vs,idx,ref): return np.argmax(rnd(Vs[:,:,idx]-ref[:,None,idx]).min(2),1)
kg=sel(np.concatenate([VOE[:,None,:],Vg],1),[0,1,2],VOE); XV=np.concatenate([XOE[:,None,:],Xg],1)[ar,kg,:]
k9=sel(np.concatenate([VOE[:,None,:],V9],1),[0,1,2],VOE); XV9=np.concatenate([XOE[:,None,:],X9],1)[ar,k9,:]
rows=[]
def add(nm,d): rows.append(dict(policy=nm,**{c:d[:,j].mean() for j,c in enumerate(CN)},**{c+' vsOE':ci(d[:,j]-XOE[:,j]) for j,c in enumerate(CN)}))
add('Heuristic (OE)',XOE); add('RL (C3-E)',XRL)
for m in ['GPT','Gemini']:
    for c in ['C4','C4R','C6-E','C6-EX']: add(f'{m} {c}',arm(f'{m}|{c}'))
add('verified catalog-9 (sel αβγ)',XV9); add('verified grid-606 (sel αβγ)',XV)
t=pd.DataFrame(rows); print(t[['policy']+CN].round(3).to_string(index=False)); print(t[['policy']+[c+' vsOE' for c in CN]].to_string(index=False))
print('\nverified grid vs RL:',{c:ci(XV[:,j]-XRL[:,j]) for j,c in enumerate(CN)})
for m in ['GPT','Gemini']:
    print(m,'verified grid vs C6-E:',{c:ci(XV[:,j]-arm(f'{m}|C6-E')[:,j]) for j,c in enumerate(CN)})
    print(m,'C6-E − C4R:',{c:ci(arm(f'{m}|C6-E')[:,j]-arm(f'{m}|C4R')[:,j]) for j,c in enumerate(CN)})
    print(m,'E1 RL − C4R:',{c:ci(arm(f'{m}|E1-RL')[:,j]-arm(f'{m}|E1-C4R')[:,j]) for j,c in enumerate(CN)})
    print(m,'C4 − OE:',{c:ci(arm(f'{m}|C4')[:,j]-XOE[:,j]) for j,c in enumerate(CN)})
    print(m,'C6-E − RL:',{c:ci(arm(f'{m}|C6-E')[:,j]-XRL[:,j]) for j,c in enumerate(CN)})
print('RL − OE:',{c:ci(XRL[:,j]-XOE[:,j]) for j,c in enumerate(CN)})
# gains over OE: within-family (LOEO) vs challengers
allV=np.concatenate([np.concatenate([VOE[:,None,:],Vg],1),np.concatenate([XOE[:,None,:],Xg],1)],2); ref=np.concatenate([VOE,XOE],1); N7=E+CN
d=(allV[:,1:,:]-ref[:,None,:]).reshape(-1,7); print('\ncorrelation of action effects (vs OE):'); print(pd.DataFrame(np.corrcoef(d.T),index=N7,columns=N7).round(2).to_string())
T=np.zeros((7,7))
for i in range(7):
    k=np.argmax(rnd(allV[:,:,i]-ref[:,None,i]),1)
    for j in range(7): T[i,j]=allV[ar,k,j].mean()-ref[:,j].mean()
print('\ngain over Heuristic when the grid is searched with one evaluator (row) and scored on another (col):'); print(pd.DataFrame(T,index=N7,columns=N7).round(3).to_string())
np.save('X9.npy',X9); np.save('Xg.npy',Xg)

import sys; sys.path.insert(0,'.')
from prep2 import *
pd.set_option('display.width',250)
Vabs=np.load('Vgrid_abs.npy'); U=np.load('Ugrid.npy'); n,M,_=Vabs.shape
V9=np.load('V9.npy')                                   # catalog values vs A0
a0=np.where(np.abs(U).sum(1)<1e-12)[0][0]
Vg=Vabs-Vabs[:,[a0],:]                                 # grid policy values vs A0
OEi=IDS.index('OE'); VOE=V9[:,OEi,:]; RL=V9[np.arange(n),RLID,:]
# sanity: grid contains OE / RF / WC1 / MX1 points identical to catalog
def find(c): 
    u=norm(CAND[c]); d=np.abs(U-u[None,:]).sum(1); return int(d.argmin()), d.min()
for c in ['OE','RF','WC1','MX1','DL','CX']:
    j,dm=find(c); print(c,'grid idx',j,'dist %.1e'%dm,'max|dV| %.1e'%np.abs(Vg[:,j,:]-V9[:,IDS.index(c),:]).max())
budget=np.abs(U).sum(1)
print('grid size',M,'budget levels',np.unique(np.round(budget,2)))
# in-sample optimum (upper bound at quarter resolution) vs catalog upper
print('upper bound catalog-9:',V9.max(1).mean(0).round(3),' grid-606:',Vg.max(1).mean(0).round(3))
# transfer on the grid (ties -> mean over tied set)
T=np.zeros((3,3))
for i in range(3):
    best=rnd(Vg[:,:,i]); mx=best.max(1,keepdims=True); tie=(best==mx)
    for j in range(3): T[i,j]=((Vg[:,:,j]*tie).sum(1)/tie.sum(1)).mean()
print('grid transfer (rows optimised on, cols scored on)'); print(pd.DataFrame(T,index=E,columns=E).round(3).to_string())
print('share of own optimum'); print(pd.DataFrame(T/np.diag(T)[None,:],index=E,columns=E).round(3).to_string())
def select(Vsub, idx, ref):
    S_=rnd(Vsub[:,:,idx]-ref[:,None,idx]).min(2); return np.argmax(S_,1)
def loeo(Vsub, ref=VOE):
    out={}; ks={}
    for j,e in enumerate(E):
        idx=[i for i in range(3) if i!=j]; k=select(Vsub,idx,ref); out[e]=Vsub[np.arange(n),k,j]; ks[e]=k
    return out,ks
def insample(Vsub, ref=VOE):
    k=select(Vsub,[0,1,2],ref); return {e:Vsub[np.arange(n),k,j] for j,e in enumerate(E)},k
def line(name,Vsub):
    lo,ks=loeo(Vsub); ins,_=insample(Vsub)
    return dict(pool=name,**{e+'_held':lo[e].mean() for e in E},**{e+'_in':ins[e].mean() for e in E}), lo, ks
rows=[]
r,_,_=line('OE only',VOE[:,None,:]); rows.append(r)
r,_,_=line('RL only (C3-E)',RL[:,None,:]); rows.append(r)
r,lo_cat,_=line('OE + catalog-9',np.concatenate([VOE[:,None,:],V9],1)); rows.append(r)
r,lo_grid,ks_grid=line('OE + grid-606 (verifier-only search)',np.concatenate([VOE[:,None,:],Vg],1)); rows.append(r)
# random-K from the grid, mean over draws
rng=np.random.default_rng(0)
for Kk in [1,2,4,8,16,32,64,128,256]:
    acc={e+'_held':[] for e in E}; acc.update({e+'_in':[] for e in E})
    for d_ in range(60 if Kk<=32 else 20):
        cols=rng.integers(0,M,(n,Kk)); Vs=np.concatenate([VOE[:,None,:],Vg[np.arange(n)[:,None],cols,:]],1)
        lo,_=loeo(Vs); ins,_=insample(Vs)
        for e in E: acc[e+'_held'].append(lo[e].mean()); acc[e+'_in'].append(ins[e].mean())
    rows.append(dict(pool=f'OE + random-{Kk} grid pts',**{k:np.mean(v) for k,v in acc.items()}))
t=pd.DataFrame(rows); print(t.round(3).to_string(index=False))
for nm,lo in [('catalog-9',lo_cat),('grid-606',lo_grid)]:
    print(nm,'held-out vs RL:',{e:ci(lo[e]-RL[:,j]) for j,e in enumerate(E)})
    print(nm,'held-out vs OE:',{e:ci(lo[e]-VOE[:,j]) for j,e in enumerate(E)})
# what does verified grid search choose? (held-out alpha selection = chosen by beta+gamma etc.)
names=['CX','DL','RF','INV','AR','AP','COGS','SGA']
Ux=np.concatenate([norm(CAND['OE'])[None,:],U],0)
for e in E:
    k=ks_grid[e]; u=Ux[k]
    print('selected when',e,'held out: mean budget share by axis',dict(zip(names,np.abs(u).mean(0).round(3))),'| default kept %.3f'%(k==0).mean(),'| uses RF %.3f'%(u[:,2]>1e-9).mean(), '| mean budget %.2f'%np.abs(u).sum(1).mean())
np.save('Vg.npy',Vg)

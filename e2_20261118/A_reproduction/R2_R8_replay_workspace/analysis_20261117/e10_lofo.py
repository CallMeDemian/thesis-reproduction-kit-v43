import sys; sys.path.insert(0,'.')
from prep2 import *
pd.set_option('display.width',280)
V9=np.load('V9.npy'); Vg=np.load('Vg.npy'); X9=np.load('X9.npy'); Xg=np.load('Xg.npy'); U=np.load('Ugrid.npy'); n=575; ar=np.arange(n); OEi=IDS.index('OE')
A9=np.concatenate([V9,X9],2); Ag=np.concatenate([Vg,Xg],2); EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; FAM={'thesis(αβγ)':[0,1,2],'δ(leverage·coverage)':[3,4],'ε(cash flow·capital)':[5,6]}
ROE=A9[:,OEi,:]; RRL=A9[ar,RLID,:]
a0=np.where(np.abs(U).sum(1)<1e-12)[0][0]
Da={k.replace('|high|1',''):v for k,v in np.load('Darms_abs.npz').items() if k.endswith('|high|1')}; Ea=dict(np.load('Earms_abs.npz'))
D0=np.load('Dgrid_abs.npy')[:,a0,:]; E0=np.load('Egrid_abs.npy')[:,a0,:]
def arm7(m,c): return np.concatenate([vals(get('free8','B1','high',c,m)),Da[f'{m}|{c}']-D0,Ea[f'{m}|{c}']-E0],1)
LL={m:arm7(m,'C6-E') for m in ['GPT','Gemini']}; L4={m:arm7(m,'C4') for m in ['GPT','Gemini']}; LR={m:arm7(m,'C4R') for m in ['GPT','Gemini']}
sdg=(Ag-ROE[:,None,:]).reshape(-1,7).std(0)
def select(Vs,idx,z=False):
    D=Vs[:,:,idx]-ROE[:,None,idx]
    if z: D=D/sdg[idx]
    return np.argmax(rnd(D).min(2),1)
def run(pool_name,Vs,z):
    out=[]
    for fam,hold in FAM.items():
        idx=[i for i in range(7) if i not in hold]; k=select(Vs,idx,z); ch=Vs[ar,k,:]
        for j in hold:
            out.append(dict(pool=pool_name,rule='z-maximin' if z else 'raw maximin',heldout_family=fam,evaluator=EN[j],value=ch[:,j].mean(),OE=ROE[:,j].mean(),RL=RRL[:,j].mean(),
                 vsOE=ci(ch[:,j]-ROE[:,j]),vsRL=ci(ch[:,j]-RRL[:,j]),vsC6E_GPT=ci(ch[:,j]-LL['GPT'][:,j]),vsC6E_Gem=ci(ch[:,j]-LL['Gemini'][:,j]),default_kept=(k==0).mean()))
    return out
res=[]
G=np.concatenate([ROE[:,None,:],Ag],1); C=np.concatenate([ROE[:,None,:],A9],1)
PR=np.stack([ROE,RRL,L4['GPT'],LR['GPT'],LL['GPT'],L4['Gemini'],LR['Gemini'],LL['Gemini']],1)
for z in [False,True]:
    res+=run('OE+grid-606',G,z); res+=run('OE+catalog-9',C,z); res+=run('OE+RL+LLM(6 outputs)',PR,z)
t=pd.DataFrame(res)
for (p,r),g in t.groupby(['pool','rule'],sort=False):
    print('\n##',p,'|',r); print(g.drop(columns=['pool','rule']).round(3).to_string(index=False))
# what does family-diverse verified search choose?
names=['CX','DL','RF','INV','AR','AP','COGS','SGA']; Ux=np.concatenate([norm(CAND['OE'])[None,:],U],0)
k=select(G,list(range(7)),False); u=Ux[k]; print('\nall-7 selection (raw maximin): budget share',dict(zip(names,np.abs(u).mean(0).round(3))),'default kept %.3f'%(k==0).mean(),'value',dict(zip(EN,G[ar,k,:].mean(0).round(3))))
k=select(G,list(range(7)),True); u=Ux[k]; print('all-7 selection (z-maximin): budget share',dict(zip(names,np.abs(u).mean(0).round(3))),'default kept %.3f'%(k==0).mean(),'value',dict(zip(EN,G[ar,k,:].mean(0).round(3))))
print('reference values: OE',dict(zip(EN,ROE.mean(0).round(3))),'\n RL',dict(zip(EN,RRL.mean(0).round(3))),'\n GPT C6-E',dict(zip(EN,LL['GPT'].mean(0).round(3))),'\n Gemini C6-E',dict(zip(EN,LL['Gemini'].mean(0).round(3))))
np.save('A9.npy',A9); np.save('Ag.npy',Ag)

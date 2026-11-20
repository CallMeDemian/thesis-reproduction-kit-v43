"""e15: structural reason behind 'mix does not transfer, assignment does'.
(A) two-way decomposition of V[firm, action, evaluator] into action main effect and firm x action interaction; cross-evaluator agreement of each part
(B) RL assignment value after controlling for applicability (which catalog actions do anything for the firm)
(C) firm-level dominance across all 7 evaluators for key policy pairs
(D) headroom: actions that dominate the Heuristic / RL action on all 7 evaluators (in-sample upper bound)
"""
import sys; sys.path.insert(0,'.')
from prep2 import *
from itertools import combinations
pd.set_option('display.width',280)
A9=np.load('A9.npy'); Ag=np.load('Ag.npy'); U=np.load('Ugrid.npy'); Z=dict(np.load('e12_assign.npz'))
n=575; ar=np.arange(n); EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; FAMOF=[0,0,0,1,1,2,2]; OEi=IDS.index('OE')
OE=A9[:,OEi,:]; RL=A9[ar,RLID,:]
c2=np.array([IDS.index(c) for c in a3.C2_action]); BC=A9[ar,c2,:]

# ---------- (A) two-way decomposition ----------
def decomp(V):                      # V: firms x actions (one evaluator)
    g=V.mean(); a=V.mean(0)-g; f=V.mean(1)-g; R=V-g-a[None,:]-f[:,None]
    return a,f,R
def pairstats(V3,label,mask=None):
    parts=[decomp(V3[:,:,e]) for e in range(7)]
    rows=[]
    for e in range(7):
        a,f,R=parts[e]; tot=((V3[:,:,e]-V3[:,:,e].mean())**2).mean()
        rows.append(dict(evaluator=EN[e],share_action=(a**2).mean()/tot,share_firm=(f**2).mean()/tot,share_interaction=(R**2).mean()/tot))
    print(f'\n[{label}] variance shares of the action effect surface'); print(pd.DataFrame(rows).round(3).to_string(index=False))
    cm=np.zeros((7,7)); ci_=np.zeros((7,7)); cr=np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            cm[i,j]=np.corrcoef(parts[i][0],parts[j][0])[0,1]
            ci_[i,j]=np.corrcoef(parts[i][2].ravel(),parts[j][2].ravel())[0,1]
            cr[i,j]=np.corrcoef(V3[:,:,i].ravel(),V3[:,:,j].ravel())[0,1]
    wi=[(i,j) for i,j in combinations(range(7),2) if FAMOF[i]==FAMOF[j]]; bt=[(i,j) for i,j in combinations(range(7),2) if FAMOF[i]!=FAMOF[j]]
    f=lambda M,P:(np.mean([M[i,j] for i,j in P]),np.min([M[i,j] for i,j in P]),np.max([M[i,j] for i,j in P]))
    print('cross-evaluator Pearson correlation (mean / min / max)')
    for nm,M in [('action main effect (-> mix value)',cm),('firm x action interaction (-> assignment value)',ci_),('raw effect surface',cr)]:
        print(f'  {nm:48s} same family %.2f / %.2f / %.2f | different family %.2f / %.2f / %.2f'%(*f(M,wi),*f(M,bt)))
    return cm,ci_
cmg,cig=pairstats(Ag,'grid-606')
cm9,ci9=pairstats(A9[:,1:,:],'catalog-8 (without no-action)')
print('\ninteraction correlation matrix, grid-606'); print(pd.DataFrame(cig,index=EN,columns=EN).round(2).to_string())
print('main-effect correlation matrix, grid-606'); print(pd.DataFrame(cmg,index=EN,columns=EN).round(2).to_string())
# bootstrap over firms for the between-family gap (interaction corr − main-effect corr)
rng=np.random.default_rng(0); bt=[(i,j) for i,j in combinations(range(7),2) if FAMOF[i]!=FAMOF[j]]; gaps=[]
for b in range(300):
    s=rng.integers(0,n,n); V=Ag[s]; parts=[decomp(V[:,:,e]) for e in range(7)]
    m=np.mean([np.corrcoef(parts[i][0],parts[j][0])[0,1] for i,j in bt]); r=np.mean([np.corrcoef(parts[i][2].ravel(),parts[j][2].ravel())[0,1] for i,j in bt]); gaps.append((m,r,r-m))
gaps=np.array(gaps); print('between-family mean corr, firm bootstrap (300): main %.2f [%.2f, %.2f] | interaction %.2f [%.2f, %.2f] | gap %.2f [%.2f, %.2f]'%tuple(x for k in range(3) for x in (gaps[:,k].mean(),np.percentile(gaps[:,k],2.5),np.percentile(gaps[:,k],97.5))))
# best action per firm: agreement across evaluators (grid)
best=Ag.argmax(1)   # firms x 7
agree=np.array([[np.mean(best[:,i]==best[:,j]) for j in range(7)] for i in range(7)])
print('share of firms whose own-optimum grid action coincides between two evaluators: same family %.3f | different family %.3f'%(np.mean([agree[i,j] for i,j in combinations(range(7),2) if FAMOF[i]==FAMOF[j]]),np.mean([agree[i,j] for i,j in bt])))
# regret transfer: take evaluator i's best action per firm, score on j, relative to j's own best and to OE
rows=[]
for i in range(7):
    for j in range(7):
        if i==j: continue
        got=Ag[ar,best[:,i],j]; own=Ag.max(1)[:,j]; rows.append(dict(sel=EN[i],score=EN[j],same=FAMOF[i]==FAMOF[j],captured=(got.mean()-OE[:,j].mean())/(own.mean()-OE[:,j].mean())))
t=pd.DataFrame(rows); print('share of the (own optimum − Heuristic) gap captured when the action is optimised on another evaluator: same family %.2f | different family %.2f (min %.2f, max %.2f)'%(t[t.same].captured.mean(),t[~t.same].captured.mean(),t[~t.same].captured.min(),t[~t.same].captured.max()))

# ---------- (B) applicability-controlled assignment value ----------
tol=1e-9
app=(np.abs(A9)>tol).any(2)            # firms x 9 : action changes at least one of the 7 scores
print('\n[applicability] share of firms for which the catalog action moves any of the 7 scores:',dict(zip(IDS,app.mean(0).round(3))))
print('RL action inert for %.3f of firms; BC action inert for %.3f; Heuristic inert for %.3f'%((~app[ar,RLID]).mean(),(~app[ar,c2]).mean(),(~app[:,OEi]).mean()))
def assign(ids,name):
    w=np.bincount(ids,minlength=9)/n; own=A9[ar,ids,:]; mix=(A9*w[None,:,None]).sum(1)
    # (1) applicability-aware random: policy mix restricted to actions applicable to the firm
    q=w[None,:]*app; s=q.sum(1,keepdims=True); q=np.divide(q,s,out=np.zeros_like(q),where=s>0); mixapp=(A9*q[:,:,None]).sum(1)
    # (2) stratified by the firm's full applicability pattern: mix of the policy inside the stratum
    key=pd.Series([''.join('1' if x else '0' for x in r) for r in app[:,1:]]); strat=np.zeros((n,7));
    for k,idx in key.groupby(key).groups.items():
        idx=np.array(list(idx)); ws=np.bincount(ids[idx],minlength=9)/len(idx); strat[idx]=(A9[idx]*ws[None,:,None]).sum(1)
    print(f'\n{name}: strata={key.nunique()} (largest {key.value_counts().iloc[0]}, singletons {(key.value_counts()==1).sum()})')
    for lab,ref in [('assignment value (firm-blind mix)',mix),('  after applicability-aware mix',mixapp),('  within applicability strata',strat)]:
        print(f'  {lab:36s}',' | '.join(f'{e} {ci((own-ref)[:,j])}' for j,e in enumerate(EN)))
    # applicable-only firms
    ok=app[ar,ids]; print('  on firms where the chosen action is not inert (n=%d): own − firm-blind mix'%ok.sum(),' | '.join(f'{e} {ci((own-mix)[ok,j])}' for j,e in enumerate(EN)))
assign(RLID,'RL policy (C3-E)'); assign(c2,'BC policy (C2)')

# ---------- (C) firm-level dominance across 7 evaluators ----------
P={'Heuristic':OE,'RL':RL,'BC':BC}
for k in sorted({x.split('|')[0] for x in Z}): P[k]=Z[k+'|own']
def dom(a,b,t=1e-6):
    d=np.round(P[a]-P[b],9); win=((d>=-t).all(1))&((d>t).any(1)); loss=((d<=t).all(1))&((d<-t).any(1)); tie=(np.abs(d)<=t).all(1)
    x=win.astype(float)-loss.astype(float); m,l,h=boot(x); maj=np.sign(d).sum(1)
    return dict(pair=f'{a} vs {b}',dominates=win.mean(),dominated=loss.mean(),identical=tie.mean(),mixed=1-win.mean()-loss.mean()-tie.mean(),net=f'{m:+.3f} [{l:+.3f},{h:+.3f}]',majority_better=(maj>0).mean(),majority_worse=(maj<0).mean())
pairs=[('RL','Heuristic'),('RL','BC'),('GPT C4','Heuristic'),('Gemini C4','Heuristic'),('GPT C4R','GPT C4'),('Gemini C4R','Gemini C4'),('GPT C6-E','GPT C4R'),('Gemini C6-E','Gemini C4R'),('GPT C6-E','GPT C6-EX'),('Gemini C6-E','Gemini C6-EX'),
       ('GPT E1-RL','GPT E1-C4R'),('Gemini E1-RL','Gemini E1-C4R'),('GPT E1-OE','GPT E1-C4R'),('Gemini E1-OE','Gemini E1-C4R'),('GPT C6-E','RL'),('Gemini C6-E','RL'),('GPT C6-E','Heuristic'),('Gemini C6-E','Heuristic'),('verified grid (sel αβγ)','RL'),('verified grid (sel αβγ)','Heuristic')]
print('\n[firm-level dominance over all 7 evaluators] dominates = no evaluator worse and at least one better')
print(pd.DataFrame([dom(a,b) for a,b in pairs]).round(3).to_string(index=False))

# ---------- (D) headroom under all 7 evaluators (in-sample upper bound, not a policy) ----------
G=np.concatenate([OE[:,None,:],Ag],1)
def exists_dom(ref,t=1e-6):
    d=np.round(G-ref[:,None,:],9); return (((d>=-t).all(2))&((d>t).any(2))).any(1)
print('\n[headroom] share of firms with a grid action that dominates the reference on all 7 evaluators: vs Heuristic %.3f | vs RL action %.3f | vs GPT C6-E %.3f | vs Gemini C6-E %.3f'%(exists_dom(OE).mean(),exists_dom(RL).mean(),exists_dom(P['GPT C6-E']).mean(),exists_dom(P['Gemini C6-E']).mean()))
sd=(Ag-OE[:,None,:]).reshape(-1,7).std(0)
k=np.argmax(rnd((G-OE[:,None,:])/sd).min(2),1); ch=G[ar,k,:]
print('all-7 z-maximin selection (in-sample): value',dict(zip(EN,ch.mean(0).round(3))),'| default kept %.3f'%(k==0).mean())
print('  minus RL:',' | '.join(f'{e} {ci((ch-RL)[:,j])}' for j,e in enumerate(EN)))
print('  RL:',dict(zip(EN,RL.mean(0).round(3))),' Heuristic:',dict(zip(EN,OE.mean(0).round(3))))

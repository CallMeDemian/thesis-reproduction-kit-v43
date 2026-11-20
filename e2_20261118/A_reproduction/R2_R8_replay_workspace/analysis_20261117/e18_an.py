"""e18 analysis: three inference-time routes to improve a frozen LLM's decision, compared under evaluator shift.
 route 1  more computation without outside signal : self-review of the draft
 route 2  verifier-guided selection (best-of-N)   : choose among the LLM's own no-advice outputs with a set of evaluators
 route 3  policy guidance                         : one revision with the RL policy's action as reference
Every gain is measured against the same draft (free8, high reasoning, <=1 budget, C4)."""
import sys; sys.path.insert(0,'.')
from prep2 import *
pd.set_option('display.width',300)
Z=dict(np.load('e18_b1_all7.npz')); A=dict(np.load('e12_assign.npz')); A9=np.load('A9.npy'); n=575; ar=np.arange(n)
EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; FAM={'T':[0,1,2],'D':[3,4],'E':[5,6]}; RL=A9[ar,RLID,:]; OE=A9[:,IDS.index('OE'),:]
def pools(m):
    k=lambda mode,r,c,info='IC-b',rep=1: Z[f'{m}|{mode}|{r}|{c}|{info}|{rep}']
    draft=k('free8','high','C4'); selfrev=k('free8','high','C4R'); adv=k('free8','high','C6-E'); mis=k('free8','high','C6-EX')
    hi=[draft,selfrev,k('free8','high','C4',rep=2),k('free8','high','C4R',rep=2),k('free8','high','C4','IC-a'),k('free8','high','C4','IC-c'),k('free8','high','C5'),k('candidate9','high','C4'),k('candidate9','high','C4R'),A[f'{m} E1-C4R|own']]
    lo=[k('free8','low','C4'),k('free8','low','C4R'),k('free8','low','C4',rep=2),k('free8','low','C4R',rep=2),k('free8','low','C4','IC-a'),k('free8','low','C4','IC-c'),k('free8','low','C5'),k('candidate9','low','C4'),k('candidate9','low','C4R')]
    advp=[adv,k('free8','high','C6-E',rep=2),k('candidate9','high','C6-E'),A[f'{m} E1-RL|own']]
    return draft,selfrev,adv,mis,np.stack(hi,1),np.stack(hi+lo,1),np.stack(advp,1)
def bon(P,sel,sd):                 # P: firms x N x 7 ; choose by mean z over selection evaluators, ties -> first (the draft)
    s=(P[:,:,sel]/sd[sel]).mean(2); return P[ar,np.argmax(rnd(s),1),:]
def line(d): return ' | '.join(f'{EN[j]} {ci(d[:,j])}' for j in range(7))
rows=[]
for m in ['GPT','Gemini']:
    draft,selfrev,adv,mis,HI,ALL,ADV=pools(m); sd=HI.reshape(-1,7).std(0)
    print(f'\n================ {m}: draft value',dict(zip(EN,draft.mean(0).round(3))),'| pool sizes high',HI.shape[1],'all',ALL.shape[1])
    print('distinct actions per firm in the high pool (by 7-score signature): mean %.2f'%np.mean([len({tuple(np.round(HI[i,k],6)) for k in range(HI.shape[1])}) for i in range(n)]))
    print('route 1  self-review − draft        :',line(selfrev-draft))
    print('route 3  RL reference − draft       :',line(adv-draft))
    print('control  other-firm reference − draft:',line(mis-draft))
    for nm,sel in [('thesis family (α,β,γ)',FAM['T']),('δ family',FAM['D']),('ε family',FAM['E']),('δ+ε (no variable shared with α,β,γ)',FAM['D']+FAM['E']),('β,γ',[1,2]),('α only',[0]),('all seven (in-sample)',list(range(7)))]:
        b=bon(HI,sel,sd); print(f'route 2  best-of-{HI.shape[1]} by {nm:38s}− draft:',line(b-draft))
        for j in range(7):
            fam=[k for k,v in FAM.items() if j in v][0]; selfam={k for k,v in FAM.items() if set(v)&set(sel)}
            kind='in-sample' if j in sel else ('same family, held out' if fam in selfam else 'other family, held out')
            g=b[:,j]-draft[:,j]; ga=adv[:,j]-draft[:,j]; rows.append(dict(model=m,selector=nm,evaluator=EN[j],kind=kind,bon_gain=g.mean(),advice_gain=ga.mean(),bon_minus_advice=ci(b[:,j]-adv[:,j])))
    b=bon(HI,FAM['D']+FAM['E'],sd); print('route 2 (δ+ε selector) − route 3 :',line(b-adv))
    b=bon(HI,FAM['T'],sd); print('route 2 (α,β,γ selector) − route 3:',line(b-adv))
    # scaling with N: draft + (N-1) random further no-advice outputs, selector = δ+ε, scored on α (the thesis's main evaluator) and in-sample
    rng=np.random.default_rng(0); print('N-scaling (pool = high+low no-advice outputs, 200 random subsets): value on α when selecting by δ+ε | by α itself (in-sample) | on ε1 when selecting by α,β,γ | by ε1 itself')
    for N in [1,2,4,8,12,19]:
        acc=np.zeros(4)
        for _ in range(200 if N<19 else 1):
            idx=np.concatenate([[0],1+rng.choice(ALL.shape[1]-1,N-1,replace=False)]) if N>1 else np.array([0]); Pn=ALL[:,idx,:]
            acc+=[bon(Pn,FAM['D']+FAM['E'],sd)[:,0].mean(),bon(Pn,[0],sd)[:,0].mean(),bon(Pn,FAM['T'],sd)[:,5].mean(),bon(Pn,[5],sd)[:,5].mean()]
        acc/= (200 if N<19 else 1); print(f'   N={N:2d}: α {acc[0]:.3f} | α in-sample {acc[1]:.3f} | ε1 {acc[2]:.3f} | ε1 in-sample {acc[3]:.3f}')
    print('   reference on α: draft %.3f | self-review %.3f | RL reference %.3f | RL policy %.3f ; on ε1: draft %.3f | RL reference %.3f | RL policy %.3f'%(draft[:,0].mean(),selfrev[:,0].mean(),adv[:,0].mean(),RL[:,0].mean(),draft[:,5].mean(),adv[:,5].mean(),RL[:,5].mean()))
    # guidance + verifier: add the advised outputs to the pool
    both=np.concatenate([HI,ADV],1)
    for nm,sel in [('δ+ε',FAM['D']+FAM['E']),('α,β,γ',FAM['T'])]:
        b=bon(both,sel,sd); print(f'route 2+3 best-of-{both.shape[1]} (no-advice + advised outputs) by {nm} − draft:',line(b-draft)); print(f'          minus route 3 alone:',line(b-adv)); print(f'          minus RL policy   :',line(b-RL))
t=pd.DataFrame(rows)
print('\n=== summary: verifier-selected gain over the draft, by relation between selector and scoring evaluator (mean over evaluators and selectors, raw score units) ===')
s=t[~t.selector.str.contains('seven')].groupby(['model','kind']).agg(bon_gain=('bon_gain','mean'),advice_gain=('advice_gain','mean'),cells=('bon_gain','size')).reset_index(); s['ratio']=s.bon_gain/s.advice_gain; print(s.round(3).to_string(index=False))
print(t[t.selector.isin(['thesis family (α,β,γ)','δ+ε (no variable shared with α,β,γ)'])].round(3).to_string(index=False))

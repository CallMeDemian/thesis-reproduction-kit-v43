import sys; sys.path.insert(0,'.')
from prep2 import *
import pickle
pd.set_option('display.width',260)
st=pickle.load(open('e7_store.pkl','rb')); EN=['alpha','beta','gamma','δ1','δ2']; SCN=['RF premium 0%p','RF premium 1%p','RF premium 2%p','cost-cut realised 50%']
def sig(x):
    m,l,h=boot(x); return m,(l>0 or h<0)
print('=== (d) reference increment C6-E − C4R: sign and significance across 5 evaluators x 4 scenarios x 2 models ===')
pos=0;sg=0;tot=0;mn=9
for m in ['GPT','Gemini']:
    for s in SCN:
        for j,e in enumerate(EN):
            d=st[(s,f'{m} C6-E')][:,j]-st[(s,f'{m} C4R')][:,j]; mu,si=sig(d); tot+=1; pos+=mu>0; sg+=(si and mu>0); mn=min(mn,mu)
print(f'positive {pos}/{tot}, significant positive {sg}/{tot}, smallest mean {mn:+.3f}')
print('\n=== (c) level comparisons vs Heuristic: range across 5 evaluators x 4 scenarios ===')
for nm in ['RL (C3-E)','GPT C4','Gemini C4','GPT C6-E','Gemini C6-E']:
    cells=[]
    for s in SCN:
        for j,e in enumerate(EN):
            mu,si=sig(st[(s,nm)][:,j]-st[(s,'OE')][:,j]); cells.append((mu,si,s,e))
    sp=sum(1 for c in cells if c[1] and c[0]>0); sn=sum(1 for c in cells if c[1] and c[0]<0)
    lo=min(cells); hi=max(cells)
    print(f'{nm:12s} − OE: sig. positive {sp}/20, sig. negative {sn}/20, min {lo[0]:+.3f} ({lo[3]}, {lo[2]}), max {hi[0]:+.3f} ({hi[3]}, {hi[2]})')
hold={'alpha':'verified grid (sel βγ)','beta':'verified grid (sel αγ)','gamma':'verified grid (sel αβ)','δ1':'verified grid (sel αβγ)','δ2':'verified grid (sel αβγ)'}
for base in ['OE','RL (C3-E)','GPT C6-E','Gemini C6-E']:
    cells=[]
    for s in SCN:
        for j,e in enumerate(EN):
            mu,si=sig(st[(s,hold[e])][:,j]-st[(s,base)][:,j]); cells.append((mu,si,s,e))
    sp=sum(1 for c in cells if c[1] and c[0]>0); sn=sum(1 for c in cells if c[1] and c[0]<0); lo=min(cells); hi=max(cells)
    print(f'verified search − {base:12s}: sig. positive {sp}/20, sig. negative {sn}/20, min {lo[0]:+.3f} ({lo[3]}, {lo[2]}), max {hi[0]:+.3f} ({hi[3]}, {hi[2]})')
print('\n=== worst-case regret vs the best single baseline (OE, RL) per evaluator, baseline scenario ===')
s='RF premium 0%p'
for nm in ['OE','RL (C3-E)','GPT C4','GPT C6-E','Gemini C4','Gemini C6-E','verified(held-out)']:
    reg=[]
    for j,e in enumerate(EN):
        v=st[(s,hold[e])][:,j] if nm=='verified(held-out)' else st[(s,nm)][:,j]
        best=max(st[(s,'OE')][:,j].mean(),st[(s,'RL (C3-E)')][:,j].mean()); reg.append(v.mean()-best)
    print(f'{nm:20s}',' '.join(f'{e}:{r:+.3f}' for e,r in zip(EN,reg)),'| worst %+.3f'%min(reg))
print('\n=== (a) per-protocol vs ITT (alpha, free-form ≤1, reasoning low) ===')
rows=[]
for m in ['GPT','Gemini']:
    for c in ['C4','C5','C4R','C6-E','C6-EX']:
        x=get('free8','B1','low',c,m); u=x.strict_usable.values; v=x.delta_R_score_alpha.values
        rows.append(dict(model=m,cond=c,valid_rate=u.mean(),per_protocol=v[u].mean(),ITT=v.mean()))
t=pd.DataFrame(rows); print(t.round(3).to_string(index=False)); print('reference: random +0.353, Heuristic +0.765, C3-E +0.923')
for m in ['GPT','Gemini']:
    for c in ['C4','C6-E']:
        x=get('free8','B1','high',c,m); u=x.strict_usable.values; v=x.delta_R_score_alpha.values; print(m,c,'high: valid %.3f PP %.3f ITT %.3f'%(u.mean(),v[u].mean(),v.mean()))

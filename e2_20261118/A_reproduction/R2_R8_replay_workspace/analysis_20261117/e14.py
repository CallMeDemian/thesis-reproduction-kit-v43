import sys; sys.path.insert(0,'.')
from prep2 import *
from statsmodels.stats.proportion import proportion_confint
pd.set_option('display.width',280)
for nm,k,nn in [('α',117,165),('δ1',round(.733*165),165),('δ2',round(.655*165),165),('ε1',round(.661*165),165),('ε2',round(.630*165),165)]:
    lo,hi=proportion_confint(k,nn,method='wilson'); print(f'{nm}: {k}/{nn} = {100*k/nn:.1f}% Wilson [{100*lo:.1f}, {100*hi:.1f}]')
Z=dict(np.load('e12_assign.npz')); EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']
print('\nreference effect split into mix shift and assignment shift (same batch E1; free-form, high, ≤1)')
rows=[]
for m in ['GPT','Gemini']:
    for ref in ['E1-RL','E1-OE']:
        o=Z[f'{m} {ref}|own']-Z[f'{m} E1-C4R|own']; mx=Z[f'{m} {ref}|mix']-Z[f'{m} E1-C4R|mix']; av=o-mx
        for j,e in enumerate(EN):
            mu,l,h=boot(av[:,j]); rows.append(dict(model=m,reference='RL policy' if ref=='E1-RL' else 'Heuristic',evaluator=e,total=o[:,j].mean(),mix_shift=mx[:,j].mean(),assignment_shift=mu,assign_CI=f'[{l:+.3f},{h:+.3f}]'))
t=pd.DataFrame(rows); print(t.round(3).to_string(index=False))
for ref in ['RL policy','Heuristic']:
    g=t[t.reference==ref]; print(ref,': assignment shift positive in %d/14, mix shift positive in %d/14, total positive in %d/14'%((g.assignment_shift>0).sum(),(g.mix_shift>0).sum(),(g.total>0).sum()))
# main-run arms too
rows=[]
for m in ['GPT','Gemini']:
    for arm in ['C6-E','C6-EX']:
        o=Z[f'{m} {arm}|own']-Z[f'{m} C4R|own']; mx=Z[f'{m} {arm}|mix']-Z[f'{m} C4R|mix']; av=o-mx
        rows.append(dict(model=m,arm=arm+' − C4R',**{e+'_tot':o[:,j].mean() for j,e in enumerate(EN)},**{e+'_asg':av[:,j].mean() for j,e in enumerate(EN)}))
print(pd.DataFrame(rows).round(3).to_string(index=False))

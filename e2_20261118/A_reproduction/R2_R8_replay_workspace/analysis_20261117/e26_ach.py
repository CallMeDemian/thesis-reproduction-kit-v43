"""e26i: achievability — how often do firms actually achieve a one-unit improvement on each lever in a year, and how firm-specific is it?"""
import sys; sys.path.insert(0,'/home/claude/harness')
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(open('e26_out.py').read().split("d=r.merge")[0])
Lh=L.copy(); Lh=Lh[(Lh.year>=2010)&(Lh.year<=2023)]
# firm's own past volatility of the COGS ratio (3 prior years) as a simple achievability predictor
h2=h[['firm_id','fiscal_year','cogsr','sgar']].copy()
for c in ['cogsr','sgar']:
    dl=h2.groupby('firm_id')[c].diff(); h2['vol_'+c]=dl.groupby(h2.firm_id).transform(lambda s: s.shift(1).rolling(3,min_periods=2).std())
Lh=Lh.merge(h2.rename(columns={'fiscal_year':'year'})[['firm_id','year','vol_cogsr','vol_sgar','cogsr']],on=['firm_id','year'],how='left')
print('share of firm-years achieving >= 1 action unit of improvement in a year (all listed firms, 2010-2023):')
for c in LEV:
    x=Lh[c].dropna(); print(f'  {c:<16} {np.mean(x>=1):.3f}  (n={len(x)})')
for c,v in [('COGS cut','vol_cogsr'),('SGA cut','vol_sgar')]:
    z=Lh[[c,v]].dropna(); q=pd.qcut(z[v],5,labels=False)
    print(f'\n{c}: share achieving >= 1 unit, by quintile of the firm\'s own past volatility of that ratio:',' '.join(f'Q{i+1} {np.mean(z[c][q==i]>=1):.2f}' for i in range(5)))
# test firms: where would 'max COGS cut for everyone' be least achievable?

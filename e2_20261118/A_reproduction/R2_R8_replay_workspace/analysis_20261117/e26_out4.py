"""e26g: pre-specified heterogeneity — do structure / working-capital levers precede upgrades where they should matter most?"""
import sys; sys.path.insert(0,'/home/claude/harness')
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(open('e26_out2.py').read().split("specs=[")[0])
inv0=(f('raw__inventory')/f('raw__total_assets')).groupby(h.firm_id).shift(1)
base=base.merge(pd.DataFrame({'firm_id':h.firm_id.values,'year':h.fiscal_year.values,'inv0':inv0.values}),on=['firm_id','year'])
dd=base.dropna(subset=['up']+LEV+['sshare0','lev0','inv0','rating_num_10','log_assets']).copy()
hi=lambda c: (dd[c]>dd[c].median()).astype(float)
inter={'RF × high short-term debt share':dd['RF refinance']*hi('sshare0'),'DL × high leverage':dd['DL deleverage']*hi('lev0'),
       'INV × high inventory intensity':dd['INV turnover up']*hi('inv0'),'DL × weak rating (num ≥ 5)':dd['DL deleverage']*(dd.rating_num_10>=5).astype(float),
       'RF × weak rating (num ≥ 5)':dd['RF refinance']*(dd.rating_num_10>=5).astype(float)}
X=pd.concat([dd[LEV],pd.DataFrame(inter),dd[['rating_num_10','log_assets','sshare0','lev0','inv0']].astype(float),
             pd.get_dummies(dd.year,prefix='y',drop_first=True,dtype=float),pd.get_dummies(dd.sector_7,prefix='s',drop_first=True,dtype=float)],axis=1)
X.insert(0,'const',1.0); y=dd.up.astype(float).values; Xv=X.values; b=np.linalg.lstsq(Xv,y,rcond=None)[0]
names=LEV+list(inter); k=[X.columns.get_loc(c) for c in names]
fm=dd.firm_id.values; uf=np.unique(fm); ix={u:np.where(fm==u)[0] for u in uf}; rng=np.random.default_rng(1); BB=[]
for _ in range(800):
    s=np.concatenate([ix[u] for u in rng.choice(uf,len(uf))]); BB.append(np.linalg.lstsq(Xv[s],y[s],rcond=None)[0][k])
BB=np.array(BB); print('n',len(dd),' rating changes',(dd.up!=0).sum())
for i,c in enumerate(names): print(f'  {c:<34} {b[k[i]]:+.4f} [{np.percentile(BB[:,i],2.5):+.4f},{np.percentile(BB[:,i],97.5):+.4f}]')
# where does RL choose RF on the test firms?
ev=pd.read_parquet(DR+'/stage2_candidate_projection/phase_eval_candidate.parquet')
ss=(pd.to_numeric(ev.raw__short_term_debt,errors='coerce').fillna(0)/(pd.to_numeric(ev.raw__short_term_debt,errors='coerce').fillna(0)+pd.to_numeric(ev.raw__long_term_debt,errors='coerce').fillna(0)+pd.to_numeric(ev.raw__bonds,errors='coerce').fillna(0))).values
med=np.nanmedian(base.sshare0); isRF=(P.a3.action_id.values=='RF')
print(f'\nRL picks RF for {isRF.sum()} firms; share with short-term debt share above the historical median ({med:.2f}): RF firms {np.nanmean(ss[isRF]>med):.2f} vs others {np.nanmean(ss[~isRF]>med):.2f}')

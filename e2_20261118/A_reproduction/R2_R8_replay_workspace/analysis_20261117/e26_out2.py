"""e26f: robustness of the outcome-anchored lever check (timing, controls, outcome form, period) + relative-weight test vs evaluators."""
import sys; sys.path.insert(0,'/home/claude/harness')
src=open('e26_out.py').read(); exec(src.split("X=pd.concat")[0])          # data prep: r, h, L, LEV, d (lead convention, winsorized)
# extra columns: contemporaneous rating change and two-year lead, level controls at t-1
r2=r.copy(); pv=r2.groupby('firm_id').shift(1); nx2=r2.groupby('firm_id').shift(-2)
r2['up_contemp']=np.where((pv.year.astype(float)==r2.year-1).fillna(False).values,(pv.rating_num_10-r2.rating_num_10).values,np.nan)
r2['up_2y']=np.where((nx2.year.astype(float)==r2.year+2).fillna(False).values,(r2.rating_num_10-nx2.rating_num_10).values,np.nan)
lvl=pd.DataFrame({'firm_id':h.firm_id.values,'year':h.fiscal_year.values,'cogsr0':lag.cogsr.values,'sgar0':lag.sgar.values,'sshare0':lag.sshare.values,
                  'lev0':(lag.debt/f('raw__total_assets').groupby(h.firm_id).shift(1)).values})
base=r2.merge(L,on=['firm_id','year']).merge(lvl,on=['firm_id','year'])
for c in LEV: lo_,hi_=base[c].quantile([.01,.99]); base[c]=base[c].clip(lo_,hi_)
for c in ['cogsr0','sgar0','sshare0','lev0']: lo_,hi_=base[c].quantile([.01,.99]); base[c]=base[c].clip(lo_,hi_)
def fit(dd,yname,controls=(),B=600,seed=0):
    dd=dd.dropna(subset=[yname]+LEV+list(controls)+['rating_num_10','log_assets'])
    X=pd.concat([dd[LEV],dd[['rating_num_10','log_assets',*controls]].astype(float),pd.get_dummies(dd.year,prefix='y',drop_first=True,dtype=float),
                 pd.get_dummies(dd.sector_7,prefix='s',drop_first=True,dtype=float)],axis=1); X.insert(0,'const',1.0)
    y=dd[yname].astype(float).values; Xv=X.values; b=np.linalg.lstsq(Xv,y,rcond=None)[0]; k=[X.columns.get_loc(c) for c in LEV]
    fm=dd.firm_id.values; uf=np.unique(fm); ix={u:np.where(fm==u)[0] for u in uf}; rng=np.random.default_rng(seed); BB=[]
    for _ in range(B):
        s=np.concatenate([ix[u] for u in rng.choice(uf,len(uf))]); BB.append(np.linalg.lstsq(Xv[s],y[s],rcond=None)[0][k])
    return b[k],np.array(BB),len(dd)
specs=[('lead (main)','up',()),('lead + level controls','up',('cogsr0','sgar0','sshare0','lev0')),('contemporaneous','up_contemp',()),
       ('2-year lead','up_2y',()),('lead, upgrade indicator','upbin',())]
base['upbin']=(base.up>0).astype(float).where(base.up.notna())
print(f'{"spec":<26}{"n":>6} '+' '.join(f'{c.split()[0]:>15}' for c in LEV))
res={}
for nm,y,ctl in specs:
    b,BB,nn=fit(base,y,ctl); res[nm]=(b,BB)
    print(f'{nm:<26}{nn:>6} '+' '.join(f'{b[i]:+.4f}{"*" if (np.percentile(BB[:,i],2.5)>0 or np.percentile(BB[:,i],97.5)<0) else " ":1}{"":>5}' for i in range(len(LEV))))
for nm,cond in [('lead, 2008-2016',base.year<=2016),('lead, 2017-2022',base.year>=2017)]:
    b,BB,nn=fit(base[cond],'up'); print(f'{nm:<26}{nn:>6} '+' '.join(f'{b[i]:+.4f}{"*" if (np.percentile(BB[:,i],2.5)>0 or np.percentile(BB[:,i],97.5)<0) else " ":1}{"":>5}' for i in range(len(LEV))))
# evaluator sensitivities (same as e26_out)
U=np.load('Ugrid.npy'); Ag=np.load('Ag.npy'); A9=np.load('A9.npy'); OE=A9[:,6,:]; sdg=(Ag-OE[:,None,:]).reshape(-1,7).std(0)
unit={'DL deleverage':(1,1),'RF refinance':(2,1),'INV turnover up':(3,1),'AR turnover up':(4,1),'AP stretch':(5,-1),'COGS cut':(6,-1),'SGA cut':(7,-1)}
sens=np.zeros((len(LEV),7))
for li,c in enumerate(LEV):
    ax,sg=unit[c]; v=np.zeros(8); v[ax]=sg; w=np.where((np.abs(U-v)<1e-9).all(1))[0]; sc_=1
    if len(w)==0: v[ax]=sg*0.75; w=np.where((np.abs(U-v)<1e-9).all(1))[0]; sc_=1/0.75
    sens[li]=Ag[:,w[0],:].mean(0)*sc_/sdg
# relative weight test: lever effect / COGS effect, real (bootstrap CI) vs each evaluator
b,BB=res['lead (main)']; ic=LEV.index('COGS cut'); ratio=BB/BB[:,[ic]]
print('\nrelative weight vs COGS cut (1.00): real [95% CI] | evaluators α β γ δ1 δ2 ε1 ε2  (x = evaluator outside the real CI)')
for i,c in enumerate(LEV):
    if i==ic: continue
    lo_,hi_=np.percentile(ratio[:,i],[2.5,97.5]); ev=sens[i]/sens[ic]
    print(f'  {c:<16} {b[i]/b[ic]:+.2f} [{lo_:+.2f},{hi_:+.2f}] | '+' '.join(f'{e:+.2f}{"x" if (e<lo_ or e>hi_) else " "}' for e in ev))

"""Challenger evaluator delta: different variables (leverage / coverage / liquidity structure), pre-specified recipe."""
import sys; sys.path.insert(0,'.')
from hz import *
from hz import _row_to_firm_state
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib
FEATS=['opm','roa','lev','borrow','icr','cur','cashr','stshare','turn','size']
def sdiv(a,b):
    try:
        a=float(a); b=float(b)
        if not np.isfinite(a) or not np.isfinite(b) or abs(b)<1e-9: return np.nan
        return a/b
    except Exception: return np.nan
def g(s,k):
    v=getattr(s,k,None)
    try: v=float(v)
    except Exception: return np.nan
    return v if np.isfinite(v) else np.nan
def feats_from_state(s):
    debt=np.nansum([g(s,'short_term_debt'),g(s,'current_portion_long_debt'),g(s,'long_term_debt'),g(s,'bonds')])
    st=np.nansum([g(s,'short_term_debt'),g(s,'current_portion_long_debt')])
    fc=g(s,'financial_cost'); oi=g(s,'operating_income')
    icr=(oi/fc) if (np.isfinite(fc) and fc>1e-9 and np.isfinite(oi)) else (50.0 if np.isfinite(oi) and oi>0 else np.nan)
    ta=g(s,'total_assets')
    return [sdiv(oi,g(s,'revenue')), sdiv(g(s,'net_income'),ta), sdiv(g(s,'total_liabilities'),ta), sdiv(debt,ta),
            float(np.clip(icr,-20,50)) if np.isfinite(icr) else np.nan, sdiv(g(s,'current_assets'),g(s,'current_liabilities')),
            sdiv(np.nansum([g(s,'cash'),g(s,'short_term_investments')]),g(s,'current_liabilities')),
            (st/debt if debt>1e-9 else 0.0), sdiv(g(s,'revenue'),ta), (np.log(ta) if np.isfinite(ta) and ta>0 else np.nan)]
if __name__=='__main__':
    hist=pd.read_parquet(DR/'stage2_candidate_projection/input_splits/canonical_business_plan_history.parquet')
    rat=pd.read_parquet(DR/'stage1_oracle_backends/alpha/oracle_firm_year_output_alpha.parquet')[['거래소코드','year','split_stage4','rating_num_10','R_score_alpha']]
    rat=rat.rename(columns={'거래소코드':'firm_id','year':'fiscal_year'}); rat['firm_id']=rat.firm_id.astype(str)
    hist['firm_id']=hist.firm_id.astype(str)
    hist=hist.drop(columns=[c for c in ['rating_num_10','split_stage4','R_score_alpha'] if c in hist.columns])
    m=rat.merge(hist,on=['firm_id','fiscal_year'],how='inner'); print('rated firm-years',len(rat),'matched',len(m), m.split_stage4.value_counts().to_dict())
    X=np.array([feats_from_state(_row_to_firm_state(r)) for _,r in m.iterrows()],float)
    y=m.rating_num_10.astype(float).values; dev=(m.split_stage4=='dev').values; oot=(m.split_stage4=='oot').values
    print('missing share by feature',dict(zip(FEATS,np.isnan(X).mean(0).round(3))))
    lo,hi=np.nanpercentile(X[dev],1,axis=0),np.nanpercentile(X[dev],99,axis=0); med=np.nanmedian(X[dev],axis=0)
    def prep(X_):
        Z=np.where(np.isnan(X_),med[None,:],X_); return np.clip(Z,lo[None,:],hi[None,:])
    Z=prep(X); mu,sd=Z[dev].mean(0),Z[dev].std(0)
    d1=Ridge(alpha=1.0).fit((Z[dev]-mu)/sd,y[dev]); d2=HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.05,random_state=0).fit(Z[dev],y[dev])
    def sc(pred): return np.clip(100*(10-pred)/9,0,100)
    s1=sc(d1.predict((Z-mu)/sd)); s2=sc(d2.predict(Z))
    for nm,s in [('delta1 ridge',s1),('delta2 gbm',s2),('alpha (reference)',m.R_score_alpha.values)]:
        print(nm,'dev rho %.3f | OOT rho %.3f (n=%d)'%(abs(spearmanr(s[dev],y[dev])[0]),abs(spearmanr(s[oot],y[oot])[0]),oot.sum()))
    print('rho(delta1,alpha) OOT %.3f  rho(delta2,alpha) OOT %.3f  rho(delta1,delta2) %.3f'%(spearmanr(s1[oot],m.R_score_alpha.values[oot])[0],spearmanr(s2[oot],m.R_score_alpha.values[oot])[0],spearmanr(s1[oot],s2[oot])[0]))
    print('ridge std coefs (positive = worse rating):',dict(zip(FEATS,d1.coef_.round(3))))
    # direction agreement on rating changes (OOT): score change vs next-year rating change
    mm=m.assign(s1=s1,s2=s2,sa=m.R_score_alpha.values).sort_values(['firm_id','fiscal_year'])
    nx=mm.groupby('firm_id').shift(-1); ok=(nx.fiscal_year==mm.fiscal_year+1)&(mm.fiscal_year>=2019)&(nx.split_stage4=='oot')
    dr=(nx.rating_num_10-mm.rating_num_10)[ok]; ch=dr!=0
    for nm in ['s1','s2','sa']:
        ds=(nx[nm]-mm[nm])[ok]; agree=(np.sign(ds[ch])==-np.sign(dr[ch])).mean(); print(nm,'direction agreement on %d rating changes: %.3f'%(ch.sum(),agree))
    joblib.dump(dict(feats=FEATS,lo=lo,hi=hi,med=med,mu=mu,sd=sd,d1=d1,d2=d2),'delta_models.joblib')

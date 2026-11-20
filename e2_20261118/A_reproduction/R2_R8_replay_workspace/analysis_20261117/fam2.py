"""Second challenger family (cash-flow / capital-structure variables). Same pre-specified recipe as delta_train."""
import sys; sys.path.insert(0,'.')
from hz import *
from hz import _row_to_firm_state
from delta_train import g, sdiv
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
import joblib, prep2 as P
F2=['ebitda_m','eq_ta','debt_ebitda','cash_std','ocf_tl','npm','nwc_rev','ncl_share','size']
def feats2(s):
    ebitda=np.nansum([g(s,'operating_income'),g(s,'depreciation'),g(s,'amortization')]) if np.isfinite(g(s,'operating_income')) else np.nan
    debt=np.nansum([g(s,'short_term_debt'),g(s,'current_portion_long_debt'),g(s,'long_term_debt'),g(s,'bonds')])
    st=np.nansum([g(s,'short_term_debt'),g(s,'current_portion_long_debt')]); cash=np.nansum([g(s,'cash'),g(s,'short_term_investments')])
    de=(debt/ebitda if (np.isfinite(ebitda) and ebitda>1e-9) else 20.0) if debt>1e-9 else 0.0
    cs=(cash/st) if st>1e-9 else 10.0
    ta=g(s,'total_assets'); rev=g(s,'revenue'); tl=g(s,'total_liabilities')
    return [sdiv(ebitda,rev), sdiv(g(s,'total_equity'),ta), float(np.clip(de,0,20)), float(np.clip(cs,0,10)), sdiv(g(s,'operating_cf'),tl), sdiv(g(s,'net_income'),rev),
            sdiv(np.nansum([g(s,'receivables'),g(s,'inventory')])-(g(s,'payables') if np.isfinite(g(s,'payables')) else 0.0),rev), sdiv(g(s,'non_current_liabilities'),tl), (np.log(ta) if np.isfinite(ta) and ta>0 else np.nan)]
def make_scorer(M):
    def f(X):
        X=np.asarray(X,float); Z=np.clip(np.where(np.isnan(X),M['med'][None,:],X),M['lo'][None,:],M['hi'][None,:]); sc=lambda p: np.clip(100*(10-p)/9,0,100)
        return np.stack([sc(M['d1'].predict((Z-M['mu'])/M['sd'])), sc(M['d2'].predict(Z))],1)
    return f
if __name__=='__main__':
    hist=pd.read_parquet(DR/'stage2_candidate_projection/input_splits/canonical_business_plan_history.parquet'); hist['firm_id']=hist.firm_id.astype(str)
    rat=pd.read_parquet(DR/'stage1_oracle_backends/alpha/oracle_firm_year_output_alpha.parquet')[['거래소코드','year','split_stage4','rating_num_10']].rename(columns={'거래소코드':'firm_id','year':'fiscal_year'}); rat['firm_id']=rat.firm_id.astype(str)
    hist=hist.drop(columns=[c for c in ['rating_num_10','split_stage4'] if c in hist.columns]); m=rat.merge(hist,on=['firm_id','fiscal_year'],how='inner')
    X=np.array([feats2(_row_to_firm_state(r)) for _,r in m.iterrows()],float); y=m.rating_num_10.astype(float).values; dev=(m.split_stage4=='dev').values; oot=~dev
    print('missing',dict(zip(F2,np.isnan(X).mean(0).round(3))))
    lo,hi,med=np.nanpercentile(X[dev],1,axis=0),np.nanpercentile(X[dev],99,axis=0),np.nanmedian(X[dev],axis=0)
    Z=np.clip(np.where(np.isnan(X),med[None,:],X),lo[None,:],hi[None,:]); mu,sd=Z[dev].mean(0),Z[dev].std(0)
    d1=Ridge(alpha=1.0).fit((Z[dev]-mu)/sd,y[dev]); d2=HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.05,random_state=0).fit(Z[dev],y[dev])
    M=dict(lo=lo,hi=hi,med=med,mu=mu,sd=sd,d1=d1,d2=d2); joblib.dump(M,'delta2_models.joblib'); S=make_scorer(M)(X)
    mm=m.assign(s1=S[:,0],s2=S[:,1])[['firm_id','fiscal_year','split_stage4','rating_num_10','s1','s2']]
    for nm,c in [('ε1 ridge','s1'),('ε2 gbm','s2')]:
        rows=[]
        for _,gg in mm.sort_values(['firm_id','fiscal_year']).groupby('firm_id'):
            yv=gg.fiscal_year.values; r=gg.rating_num_10.values.astype(float); s=gg[c].values; sp=gg.split_stage4.values
            for i in range(1,len(gg)-1):
                if yv[i+1]-yv[i]==1 and yv[i]-yv[i-1]==1 and r[i+1]!=r[i]: rows.append((sp[i+1],s[i]-s[i-1],-(r[i+1]-r[i])))
        d=pd.DataFrame(rows,columns=['split','ds','imp']); o=d[d.split=='oot']; ag=((np.sign(o.ds)==np.sign(o.imp))&(o.ds!=0)).mean()
        print(nm,'OOT rho %.3f | lead direction agreement %.1f%% (n=%d) AUC %.3f'%(abs(spearmanr(mm[c][oot],y[oot])[0]),100*ag,len(o),roc_auc_score((o.imp>0).astype(int),o.ds)))
    print('ridge coefs (+ = worse):',dict(zip(F2,d1.coef_.round(3))))
    # score catalog, grid, arms
    H=Harness(); n=575; SC=make_scorer(M)
    def fe(i,a): r,fs,hi_,exo=H.firm(i); return feats2(H.bundle.simulate_action(fs,hi_,dict(a),action_label='X')[0].state_t1)
    np.save('E9_abs.npy',np.stack([SC([fe(i,H.cands[k]) for i in range(n)]) for k in P.IDS],1))
    U=np.load('Ugrid.npy'); ga=[{k:float(u[j]*H.scale[k]) for j,k in enumerate(AX)} for u in U]
    t=time.time(); D=np.zeros((n,len(ga),2))
    for i in range(n): D[i]=SC([fe(i,a) for a in ga])
    np.save('Egrid_abs.npy',D); out={}
    for mdl in ['GPT','Gemini']:
        for c in ['C4','C4R','C6-E','C6-EX']: out[f'{mdl}|{c}']=SC([fe(i,a) for i,a in enumerate(P.acts(P.get('free8','B1','high',c,mdl)))])
        for c in ['E1-C4R','E1-OE','E1-RL']: out[f'{mdl}|{c}']=SC([fe(i,a) for i,a in enumerate([json.loads(s) for s in P.gete1(c,mdl).strict_itt_action_json])])
    np.savez('Earms_abs.npz',**out); print('done %.0fs'%(time.time()-t))

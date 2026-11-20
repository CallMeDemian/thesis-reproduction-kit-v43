import sys; sys.path.insert(0,'.')
from hz import *
from hz import _row_to_firm_state
from delta_train import feats_from_state, FEATS
import joblib
from scipy.stats import spearmanr
M=joblib.load('delta_models.joblib')
def delta_scores(X):
    Z=np.clip(np.where(np.isnan(X),M['med'][None,:],X),M['lo'][None,:],M['hi'][None,:])
    sc=lambda p: np.clip(100*(10-p)/9,0,100)
    return sc(M['d1'].predict((Z-M['mu'])/M['sd'])), sc(M['d2'].predict(Z))
hist=pd.read_parquet(DR/'stage2_candidate_projection/input_splits/canonical_business_plan_history.parquet'); hist['firm_id']=hist.firm_id.astype(str)
rat=pd.read_parquet(DR/'stage1_oracle_backends/alpha/oracle_firm_year_output_alpha.parquet')[['거래소코드','year','split_stage4','rating_num_10','R_score_alpha']].rename(columns={'거래소코드':'firm_id','year':'fiscal_year'}); rat['firm_id']=rat.firm_id.astype(str)
hist=hist.drop(columns=[c for c in ['rating_num_10','split_stage4','R_score_alpha'] if c in hist.columns])
m=rat.merge(hist,on=['firm_id','fiscal_year'],how='inner')
X=np.array([feats_from_state(_row_to_firm_state(r)) for _,r in m.iterrows()],float)
s1,s2=delta_scores(X); m=m.assign(s1=s1,s2=s2,sa=m.R_score_alpha.values)[['firm_id','fiscal_year','split_stage4','rating_num_10','s1','s2','sa']]
def lead(col):
    rows=[]
    for _,g in m.sort_values(['firm_id','fiscal_year']).groupby('firm_id'):
        y=g.fiscal_year.values; r=g.rating_num_10.values.astype(float); s=g[col].values; sp=g.split_stage4.values
        for i in range(1,len(g)-1):
            if y[i+1]-y[i]==1 and y[i]-y[i-1]==1 and not np.isnan([r[i],r[i+1],s[i],s[i-1]]).any() and r[i+1]!=r[i]:
                rows.append((sp[i+1], s[i]-s[i-1], -(r[i+1]-r[i])))
    d=pd.DataFrame(rows,columns=['split','ds','imp']); o=d[d.split=='oot']; match=(np.sign(o.ds)==np.sign(o.imp))&(o.ds!=0)
    from sklearn.metrics import roc_auc_score
    return len(o), match.mean(), roc_auc_score((o.imp>0).astype(int),o.ds)
for nm,c in [('alpha',  'sa'),('delta1 ridge','s1'),('delta2 gbm','s2')]:
    n_,ag,auc=lead(c); print(f'{nm:13s} OOT movers {n_}  lead direction agreement {100*ag:.1f}%  AUC {auc:.3f}')

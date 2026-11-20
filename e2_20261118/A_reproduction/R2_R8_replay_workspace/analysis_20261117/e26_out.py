"""e26e (pilot): outcome-anchored check — do the levers that evaluators reward precede real rating upgrades?
Associational (not causal). Realized lever changes FY t-1 -> t (in action units) vs realized rating change t -> t+1 (lead convention used in δ/ε validation)."""
import sys; sys.path.insert(0,'/home/claude/harness')
import numpy as np, pandas as pd, prep2 as P
from scipy.stats import spearmanr
DR='/home/claude/kitzip/THESIS_REPRO_KIT_v2.1.1_FINAL/archive/DEPLOYED_RELEASE'
r=pd.read_parquet(DR+'/stage1_oracle_backends/alpha/oracle_firm_year_output_alpha.parquet').rename(columns={'거래소코드':'firm_id'})
r['firm_id']=r.firm_id.astype(str); r=r[['firm_id','year','sector_7','rating_num_10','log_assets','split_stage4']].sort_values(['firm_id','year'])
r["rating_num_10"]=r.rating_num_10.astype(float); r["year"]=r.year.astype(int); nx=r.groupby("firm_id").shift(-1); r["up"]=np.where((nx.year.astype(float)==r.year+1).fillna(False).values, (r.rating_num_10-nx.rating_num_10).values, np.nan)  # >0 = upgrade
h=pd.read_parquet(DR+'/stage2_candidate_projection/input_splits/canonical_business_plan_history.parquet'); h['firm_id']=h.firm_id.astype(str)
h=h.sort_values(['firm_id','fiscal_year']); f=lambda c: pd.to_numeric(h[c],errors='coerce')
h['debt']=f('raw__short_term_debt').fillna(0)+f('raw__long_term_debt').fillna(0)+f('raw__bonds').fillna(0)
h['sshare']=np.where(h.debt>0,f('raw__short_term_debt').fillna(0)/h.debt,np.nan)
h['cogsr']=f('raw__cogs')/f('raw__revenue'); h['sgar']=f('raw__sga')/f('raw__revenue')
h['invt']=f('raw__cogs')/f('raw__inventory').where(f('raw__inventory')>0); h['art']=f('raw__revenue')/f('raw__receivables').where(f('raw__receivables')>0)
h['apt']=f('raw__cogs')/f('raw__payables').where(f('raw__payables')>0)
for c in ['cogsr','sgar','invt','art','apt']: h[c]=h[c].replace([np.inf,-np.inf],np.nan)
lag=h.groupby('firm_id')[['fiscal_year','debt','sshare','cogsr','sgar','invt','art','apt']].shift(1)
okl=(lag.fiscal_year==h.fiscal_year-1).values; SC=P.C['normalization_scale_by_dimension']
L=pd.DataFrame({'firm_id':h.firm_id.values,'year':h.fiscal_year.values})
g=lambda x: np.where(okl,x,np.nan)
L['DL deleverage']=g(np.where(lag.debt>0,-(h.debt/lag.debt-1),np.nan))/SC['deleveraging_total_debt_pct']
L['RF refinance']=g(np.where(lag.sshare>0,-(h.sshare-lag.sshare)/lag.sshare,np.nan))/SC['refinancing_short_debt_pct']
L['INV turnover up']=g(h.invt-lag.invt)/SC['inv_turnover_chg']
L['AR turnover up']=g(h.art-lag.art)/SC['ar_turnover_chg']
L['AP stretch']=g(-(h.apt-lag.apt))/SC['ap_turnover_chg']
L['COGS cut']=g(-(h.cogsr-lag.cogsr))/SC['cogs_ratio_chg']
L['SGA cut']=g(-(h.sgar-lag.sgar))/SC['sga_ratio_chg']
LEV=[c for c in L.columns if c not in ('firm_id','year')]
d=r.merge(L,on=['firm_id','year'],how='inner').dropna(subset=['up']+LEV)
for c in LEV: lo,hi=d[c].quantile([.01,.99]); d[c]=d[c].clip(lo,hi)
print('transitions used',len(d),'firms',d.firm_id.nunique(),'rating changes',(d.up!=0).sum(),'years',d.year.min(),'-',d.year.max())
X=pd.concat([d[LEV],d[['rating_num_10','log_assets']].astype(float),pd.get_dummies(d.year,prefix='y',drop_first=True,dtype=float),pd.get_dummies(d.sector_7,prefix='s',drop_first=True,dtype=float)],axis=1)
X.insert(0,'const',1.0); y=d.up.values
def ols(Xm,yv): return np.linalg.lstsq(Xm,yv,rcond=None)[0]
b=ols(X.values,y); firms=d.firm_id.values; uf=np.unique(firms); idx={u:np.where(firms==u)[0] for u in uf}
rng=np.random.default_rng(0); B=[]
for _ in range(1000):
    s=np.concatenate([idx[u] for u in rng.choice(uf,len(uf))]); B.append(ols(X.values[s],y[s]))
B=np.array(B); k=[X.columns.get_loc(c) for c in LEV]
real=b[k]; lo=np.percentile(B[:,k],2.5,0); hi=np.percentile(B[:,k],97.5,0)
# evaluator sensitivities: single-axis unit actions on the 575 test firms
U=np.load('Ugrid.npy'); Ag=np.load('Ag.npy'); A9=np.load('A9.npy'); OE=A9[:,6,:]; sdg=(Ag-OE[:,None,:]).reshape(-1,7).std(0)
unit={'DL deleverage':(1,1),'RF refinance':(2,1),'INV turnover up':(3,1),'AR turnover up':(4,1),'AP stretch':(5,-1),'COGS cut':(6,-1),'SGA cut':(7,-1)}
sens=np.zeros((len(LEV),7)); 
for li,c in enumerate(LEV):
    ax,sg=unit[c]; v=np.zeros(8); v[ax]=sg
    w=np.where((np.abs(U-v)<1e-9).all(1))[0]
    if len(w)==0: v[ax]=sg*0.75; w=np.where((np.abs(U-v)<1e-9).all(1))[0]; scale=1/0.75
    else: scale=1
    sens[li]=Ag[:,w[0],:].mean(0)*scale/sdg
N=['α','β','γ','δ1','δ2','ε1','ε2']
print('\nlever (per action unit) | real: rating notches up next year [95% CI, firm bootstrap] | evaluators: standardized value of a unit action')
print(f'{"":<16}{"real β":>8} {"CI":>17} '+' '.join(f'{x:>6}' for x in N))
for li,c in enumerate(LEV): print(f'{c:<16}{real[li]:+8.4f} [{lo[li]:+.4f},{hi[li]:+.4f}] '+' '.join(f'{x:+6.2f}' for x in sens[li]))
print('\nSpearman across 8 levers, real β vs evaluator sensitivity:',' '.join(f'{N[j]} {spearmanr(real,sens[:,j])[0]:+.2f}' for j in range(7)))
sig=(lo>0)|(hi<0); print('levers with CI excluding 0:',[c for c,s_ in zip(LEV,sig) if s_])
print('sign agreement on those levers:',' '.join(f'{N[j]} {int((np.sign(real[sig])==np.sign(sens[sig,j])).sum())}/{sig.sum()}' for j in range(7)))

# ---- policy lever composition and a crude linear outcome-anchored score (action units x real β) ----
AXI={'DL deleverage':1,'RF refinance':2,'INV turnover up':3,'AR turnover up':4,'AP stretch':5,'COGS cut':6,'SGA cut':7}
SIGN={'DL deleverage':1,'RF refinance':1,'INV turnover up':1,'AR turnover up':1,'AP stretch':-1,'COGS cut':-1,'SGA cut':-1}
def lev(Avec):  # n x 8 normalized actions -> n x 7 lever units in the regression's orientation
    return np.stack([SIGN[c]*Avec[:,AXI[c]] for c in LEV],1)
rlv=np.array([P.norm(P.CAND[P.IDS[k]]) for k in P.RLID])
pols={'RL C3-E':rlv,'OE heuristic':np.tile(P.norm(P.CAND['OE']),(575,1)),'const max COGS cut':np.tile(np.array([0,0,0,0,0,0,-1,0.]),(575,1))}
for m in ['GPT','Gemini']:
    for c in ['C4','C6-E']:
        pols[f'LLM {m} {c}']=np.array([P.norm(d) for d in P.acts(P.get('free8','B1','high',c,m,rep=1))])
print('\nRL action counts:',dict(zip(*np.unique(P.a3.action_id,return_counts=True))))
print(f'\n{"policy":<22}'+' '.join(f'{c.split()[0]:>6}' for c in LEV)+'  op.share  outcome-anchored score [95% CI over β draws]')
for nm,A in pols.items():
    Lv=lev(A); comp=np.abs(Lv).mean(0); share=comp[[LEV.index('COGS cut'),LEV.index('SGA cut')]].sum()/max(comp.sum(),1e-9)
    sc=Lv@real; draws=(Lv@B[:,k].T).mean(0)
    print(f'{nm:<22}'+' '.join(f'{x:6.2f}' for x in comp)+f'  {share:7.2f}   {sc.mean():+.4f} [{np.percentile(draws,2.5):+.4f},{np.percentile(draws,97.5):+.4f}]')

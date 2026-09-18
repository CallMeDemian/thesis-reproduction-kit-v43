import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];D=pd.read_parquet(ROOT/'frozen_inputs/stage8/canonical_itt_observations.parquet');R=pd.read_csv(ROOT/'canonical_v13/V13_SUPPLEMENTAL_RESULTS.csv');B=10000
def oc(o):return 'delta_R_score_'+str(o).lower()
def model(x):return {'GEMINI':'google_gemini31flashlite','GPT':'openai_gpt54mini'}[x]
def get(reg,ep,space,m,budget,info,rep,cond,o):
 return D[(D.reasoning_regime==reg)&(D.execution_policy==ep)&(D.action_space==space)&(D.model_key==m)&(D.budget==budget)&(D.information_condition==info)&(D.replicate==rep)&(D.policy_condition==cond)].set_index('firm_key')
def calc(row):
 k=json.loads(row.key_json);m=model(row.model);l=k['lhs'];rr=k['rhs'];o=k['oracle'];rep=int(k['replicate'].replace('RUN',''))
 if row.analysis_id=='RQ4_ACTION_SPACE':
  a=get('BASELINE','STRICT','FREE8',m,'B1','IC-b',1,l,o);b=get('BASELINE','STRICT','CANDIDATE9',m,'B1','IC-b',1,l,o)
  ids=a.index.intersection(b.index);x=(a.loc[ids,oc(o)]-b.loc[ids,oc(o)]).to_numpy(float);return verify(row,x)
 elif row.analysis_id=='RQ5_INFORMATION':
  a=get('BASELINE','STRICT','FREE8',m,'B1',l,1,'C4',o);b=get('BASELINE','STRICT','FREE8',m,'B1',rr,1,'C4',o)
  ids=a.index.intersection(b.index);x=(a.loc[ids,oc(o)]-b.loc[ids,oc(o)]).to_numpy(float);return verify(row,x)
 elif row.analysis_id=='BUDGET_INTERACTION':
  a=get('BASELINE','STRICT','FREE8',m,'B1','IC-b',1,l,o);b=get('BASELINE','STRICT','FREE8',m,'B1','IC-b',1,rr,o);c=get('BASELINE','STRICT','FREE8',m,'BINF','IC-b',1,l,o);e=get('BASELINE','STRICT','FREE8',m,'BINF','IC-b',1,rr,o)
  ids=a.index.intersection(b.index).intersection(c.index).intersection(e.index);x=(a.loc[ids,oc(o)]-b.loc[ids,oc(o)]-c.loc[ids,oc(o)]+e.loc[ids,oc(o)]).to_numpy(float);return verify(row,x)
 elif row.analysis_id=='REASONING_REGIME_INTERACTION':
  a=get('HIGH','STRICT','FREE8',m,'B1','IC-b',1,l,o);b=get('HIGH','STRICT','FREE8',m,'B1','IC-b',1,rr,o);c=get('BASELINE','STRICT','FREE8',m,'B1','IC-b',1,l,o);e=get('BASELINE','STRICT','FREE8',m,'B1','IC-b',1,rr,o)
  ids=a.index.intersection(b.index).intersection(c.index).intersection(e.index);x=(a.loc[ids,oc(o)]-b.loc[ids,oc(o)]-c.loc[ids,oc(o)]+e.loc[ids,oc(o)]).to_numpy(float);return verify(row,x)
 elif row.analysis_id=='EXECUTION_SENSITIVITY':
  a=get(k['reasoning_regime'],'REPAIRED','FREE8',m,'B1','IC-b',1,l,o);b=get(k['reasoning_regime'],'REPAIRED','FREE8',m,'B1','IC-b',1,rr,o);c=get(k['reasoning_regime'],'STRICT','FREE8',m,'B1','IC-b',1,l,o);e=get(k['reasoning_regime'],'STRICT','FREE8',m,'B1','IC-b',1,rr,o)
  ids=a.index.intersection(b.index).intersection(c.index).intersection(e.index);x=(a.loc[ids,oc(o)]-b.loc[ids,oc(o)]-c.loc[ids,oc(o)]+e.loc[ids,oc(o)]).to_numpy(float);return verify(row,x)
 else:
  a=get(k['reasoning_regime'],k['execution_policy'],k['action_space'],m,k['budget'],k['information_condition'],rep,l,o);b=get(k['reasoning_regime'],k['execution_policy'],k['action_space'],m,k['budget'],k['information_condition'],rep,rr,o);ids=a.index.intersection(b.index)
  if row.population=='PP':ids=ids[a.loc[ids,'strict_usable'].astype(bool)&b.loc[ids,'strict_usable'].astype(bool)]
  x=(a.loc[ids,oc(o)]-b.loc[ids,oc(o)]).to_numpy(float);return verify(row,x)
def verify(row,x):
 rng=np.random.Generator(np.random.PCG64(int(row.seed_uint64)))
 # Bounded chunks preserve the exact PCG64 draw order while avoiding a
 # multi-gigabyte temporary matrix for the full supplemental package.
 chunks=[]
 for start in range(0,B,128):
  n=min(128,B-start)
  idx=rng.integers(0,len(x),size=(n,len(x)))
  chunks.append(x[idx].mean(1))
 y=np.concatenate(chunks)
 q=np.quantile(y,[.025,.975],method='linear');a=(y<0).sum();z=(y==0).sum();b=(y>0).sum();p=min(1.,2*(min(a+z,b+z)+1)/(B+1));dh=hashlib.sha256(np.asarray(y,dtype='<f8').tobytes()).hexdigest();return len(x),float(x.mean()),float(q[0]),float(q[1]),p,dh

bad=[]
for i,r in R.iterrows():
 try:
  n,e,lo,hi,p,dh=calc(r)
  if n!=int(r.N) or not np.isclose(e,r.estimate,atol=1e-12) or not np.isclose(lo,r.ci_low,atol=1e-12) or not np.isclose(hi,r.ci_high,atol=1e-12) or not np.isclose(p,r.raw_p,atol=1e-15) or dh!=r.bootstrap_draw_sha256:bad.append((i,'mismatch'))
 except Exception as ex:bad.append((i,type(ex).__name__+':'+str(ex)))
print('V13_REPRODUCTION_PASS' if not bad else 'V13_REPRODUCTION_FAIL');print('rows',len(R),'mismatches',len(bad))
if bad:print(bad[:5]);sys.exit(1)






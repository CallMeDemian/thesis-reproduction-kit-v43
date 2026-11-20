import sys; sys.path.insert(0,'.')
from hz import *
import prep2 as P
H=Harness(); n=575
cls=type(H.bundle._simulator); orig=cls.__dict__['_interest'].__func__
PREM={'p':0.0,'tau':1.0}
def patched(bp,**kw):
    total,parts=orig(bp,**kw)
    extra=PREM['tau']*(kw['new_long_tranche']+kw['new_bond_tranche'])*PREM['p']
    parts=dict(parts); parts['rf_term_premium']=extra
    return total+extra,parts
cls._interest=staticmethod(patched)
def score_actions(actlist):
    out=H.evaluate([(i,a) for i,a in actlist]); return out[['alpha','beta','gamma']].values
def scaled(a,f):
    b=dict(a); b['cogs_ratio_chg']=a['cogs_ratio_chg']*f; b['sga_ratio_chg']=a['sga_ratio_chg']*f; return b
# policies: catalog (9) + stored LLM arms
arms={}
for m in ['GPT','Gemini']:
    for c in ['C4','C4R','C6-E','C6-EX']: arms[(m,c)]=P.acts(P.get('free8','B1','high',c,m))
    for c in ['E1-C4R','E1-OE','E1-RL']: arms[(m,c)]=[json.loads(s) for s in P.gete1(c,m).strict_itt_action_json]
res=[]
def run(tag):
    A0=score_actions([(i,H.cands['A0']) for i in range(n)])
    cat={k:score_actions([(i,H.cands[k]) for i in range(n)])-A0 for k in P.IDS}
    rl=np.stack([cat[P.IDS[P.RLID[i]]][i] for i in range(n)])
    row=dict(scenario=tag)
    for j,e in enumerate(P.E):
        row[f'RF_{e}']=cat['RF'][:,j].mean(); row[f'OE_{e}']=cat['OE'][:,j].mean(); row[f'RL_{e}']=rl[:,j].mean()
        row[f'RL-OE_{e}']=P.ci(rl[:,j]-cat['OE'][:,j])
    for (m,c),al in arms.items():
        v=score_actions(list(enumerate(al)))-A0
        for j,e in enumerate(P.E): row[f'{m}|{c}|{e}']=v[:,j].mean()
        arms_v[(tag,m,c)]=v
    for m in ['GPT','Gemini']:
        for j,e in enumerate(P.E):
            row[f'{m} C6-E−C4R {e}']=P.ci(arms_v[(tag,m,'C6-E')][:,j]-arms_v[(tag,m,'C4R')][:,j])
            row[f'{m} E1RL−E1OE(343) {e}']=P.ci((arms_v[(tag,m,'E1-RL')][:,j]-arms_v[(tag,m,'E1-OE')][:,j])[P.RLID!=P.IDS.index('OE')])
    res.append(row)
arms_v={}
t=time.time()
for p in [0.0,0.005,0.01,0.02,0.03]:
    PREM['p']=p; run(f'RF premium {p*100:.1f}%p'); print('done',p,'%.0fs'%(time.time()-t),flush=True)
PREM['p']=0.0
# cost-cut realisation haircut (applied to every policy's cost-ratio components)
base_arms=dict(arms); base_cands={k:dict(v) for k,v in H.cands.items()}
for f in [0.5,0.25]:
    for k in H.cands: H.cands[k]=scaled(base_cands[k],f)
    for key in arms: arms[key]=[scaled(a,f) for a in base_arms[key]]
    run(f'cost-cut realised {int(f*100)}%'); print('done haircut',f,flush=True)
t_=pd.DataFrame(res).set_index('scenario').T
pd.set_option('display.width',280); pd.set_option('display.max_rows',400)
print(t_.to_string())
t_.to_csv('e5_stress.csv')

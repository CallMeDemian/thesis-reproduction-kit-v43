import sys; sys.path.insert(0,'.')
from prep2 import *
pd.set_option('display.width',280)
A9=np.load('A9.npy'); n=575; ar=np.arange(n); EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; OEi=IDS.index('OE')
a0i=None; U=np.load('Ugrid.npy'); a0=np.where(np.abs(U).sum(1)<1e-12)[0][0]
Da={k.replace('|high|1',''):v for k,v in np.load('Darms_abs.npz').items() if k.endswith('|high|1') or 'E1' in k}; Ea=dict(np.load('Earms_abs.npz'))
D0=np.load('Dgrid_abs.npy')[:,a0,:]; E0=np.load('Egrid_abs.npy')[:,a0,:]
def arm7(m,c):
    base=vals(gete1(c,m)) if c.startswith('E1') else vals(get('free8','B1','high',c,m))
    return np.concatenate([base,Da[f'{m}|{c}']-D0,Ea[f'{m}|{c}']-E0],1)
print('=== LLM-level contrasts on 7 evaluators (free-form, reasoning high, ≤1) ===')
for m in ['GPT','Gemini']:
    A=arm7(m,'C6-E'); R=arm7(m,'C4R'); X=arm7(m,'C6-EX'); D=arm7(m,'C4')
    x=get('free8','B1','high','C6-EX',m); e=get('free8','B1','high','C6-E',m); diff=(x.reference_candidate_id.values!=e.reference_candidate_id.values)
    print(m,'C6-E − C4R      :',' | '.join(f'{en} {ci((A-R)[:,j])}' for j,en in enumerate(EN)))
    print(m,'C6-E − C6-EX    :',' | '.join(f'{en} {ci((A-X)[:,j])}' for j,en in enumerate(EN)))
    print(m,'  (434 diff ref):',' | '.join(f'{en} {ci((A-X)[diff,j])}' for j,en in enumerate(EN)))
    print(m,'  (141 same ref):',' | '.join(f'{en} {ci((A-X)[~diff,j])}' for j,en in enumerate(EN)))
    print(m,'C4R − C4        :',' | '.join(f'{en} {ci((R-D)[:,j])}' for j,en in enumerate(EN)))
    EL=arm7(m,'E1-RL'); EO=arm7(m,'E1-OE'); ER=arm7(m,'E1-C4R'); sub=RLID!=OEi
    print(m,'E1: RL − C4R    :',' | '.join(f'{en} {ci((EL-ER)[:,j])}' for j,en in enumerate(EN)))
    print(m,'E1: OE − C4R    :',' | '.join(f'{en} {ci((EO-ER)[:,j])}' for j,en in enumerate(EN)))
    print(m,'E1: RL−OE (343) :',' | '.join(f'{en} {ci((EL-EO)[sub,j])}' for j,en in enumerate(EN)))
    RL=A9[ar,RLID,:]; print(m,'C6-E − RL       :',' | '.join(f'{en} {ci((A-RL)[:,j])}' for j,en in enumerate(EN)))
    OE=A9[:,OEi,:]; print(m,'C4 − OE         :',' | '.join(f'{en} {ci((D-OE)[:,j])}' for j,en in enumerate(EN)))

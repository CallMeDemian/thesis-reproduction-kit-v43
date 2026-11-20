"""e24: close the loops opened by the GPT/Codex follow-ups, on the same frozen outputs.
 (1) how many firms have any score difference between C6-E and C4R, per evaluator
 (2) where the advice increment comes from: firms whose revision copied the RL action vs moved toward it vs not
 (3) negative control: the 141 firms whose 'other firm' reference is the same action
 (4) reliability ceiling: do per-firm advice increments replicate when the same prompts are run again?"""
import sys; sys.path.insert(0,'.')
from prep2 import *
from scipy.stats import spearmanr, pearsonr
Z=dict(np.load('e18_b1_all7.npz')); A9=np.load('A9.npy'); n=575; ar=np.arange(n)
EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; RL=A9[ar,RLID,:]
R=np.stack([norm(CAND[IDS[k]]) for k in RLID])
def V(m,c,rep=1): return Z[f'{m}|free8|high|{c}|IC-b|{rep}']
def U(m,c,rep=1): return np.stack([norm(a) for a in acts(get('free8','B1','high',c,m,'IC-b',rep))])
for m in ['GPT','Gemini']:
    print(f'\n==================== {m} ====================')
    d=V(m,'C6-E')-V(m,'C4R')
    print('(1) firms with any difference C6-E vs C4R:',' | '.join(f'{EN[j]} {np.mean(np.abs(d[:,j])>1e-9):.3f}' for j in range(7)))
    u0,uN,uE=U(m,'C4'),U(m,'C4R'),U(m,'C6-E')
    dE=np.abs(uE-R).sum(1); dN=np.abs(uN-R).sum(1)
    copy=dE<1e-6; already=(dN<1e-6)
    toward=(~copy)&(dE<dN-1e-6); same=(~copy)&(np.abs(dE-dN)<=1e-6); away=(~copy)&(dE>dN+1e-6)
    print(f'(2) C4R already equal to the RL action: {already.mean():.3f} | C6-E copies RL: {copy.mean():.3f} (of which C4R already equal: {(copy&already).sum()}) | moved toward: {toward.mean():.3f} | same distance: {same.mean():.3f} | away: {away.mean():.3f}')
    chk=np.abs(V(m,"C6-E")[copy]-RL[copy]).max(); print(f'    copies score exactly as the RL policy: max |diff| = {chk:.1e}')
    for nm,g in [('copy (new copy, C4R differed)',copy&~already),('copy (C4R already equal)',copy&already),('moved toward, not copy',toward),('same distance',same),('away',away)]:
        print(f'    {nm:30s} n={g.sum():3d} | share of total increment:',' | '.join(f'{EN[j]} {d[g,j].sum()/d[:,j].sum() if abs(d[:,j].sum())>1e-12 else np.nan:+.2f}' for j in range(7)),'| mean α',f'{d[g,0].mean():+.3f}' if g.sum() else '')
    print('    total increment:',' | '.join(f'{EN[j]} {d[:,j].mean():+.3f}' for j in range(7)))
    # (3) negative control
    X=get('free8','B1','high','C6-EX',m); xid=np.array([IDS.index(c) for c in X.reference_candidate_id]); sameref=xid==RLID
    dx=V(m,'C6-E')-V(m,'C6-EX')
    print(f'(3) C6-E − C6-EX, different reference (n={(~sameref).sum()}):',' | '.join(f'{EN[j]} {ci(dx[~sameref,j])}' for j in range(7)))
    print(f'    C6-E − C6-EX, same reference (n={sameref.sum()}):',' | '.join(f'{EN[j]} {ci(dx[sameref,j])}' for j in range(7)))
    # (4) reliability across replicate runs
    d2=V(m,'C6-E',2)-V(m,'C4R',2)
    print('(4) per-firm increment, run 1 vs run 2: Pearson / Spearman:',' | '.join(f'{EN[j]} {pearsonr(d[:,j],d2[:,j])[0]:.2f}/{spearmanr(d[:,j],d2[:,j])[0]:.2f}' for j in range(7)))
    nz=(np.abs(d[:,0])>1e-9)|(np.abs(d2[:,0])>1e-9); both=(np.abs(d[:,0])>1e-9)&(np.abs(d2[:,0])>1e-9)
    print(f'    α: firms nonzero in either run {nz.sum()}, in both runs {both.sum()}, same sign when both nonzero {np.mean(np.sign(d[both,0])==np.sign(d2[both,0])):.2f}')
    print('    mean increment run 1 / run 2:',' | '.join(f'{EN[j]} {d[:,j].mean():+.3f}/{d2[:,j].mean():+.3f}' for j in range(7)))
    hg=np.maximum(V(m,'C6-E'),V(m,'C4R')).mean(0)-V(m,'C6-E').mean(0)
    hn=np.maximum(V(m,'C6-E'),V(m,'C6-E',2)).mean(0)-V(m,'C6-E').mean(0)
    print('    hindsight gain of choosing C4R vs C6-E per firm (gate ceiling):',' | '.join(f'{EN[j]} {hg[j]:+.3f}' for j in range(7)))
    print('    hindsight gain of choosing between two runs of the SAME C6-E prompt:',' | '.join(f'{EN[j]} {hn[j]:+.3f}' for j in range(7)))

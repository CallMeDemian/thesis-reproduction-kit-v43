"""e26h: where does RL's advantage over the OE heuristic come from, by RL's chosen action? (α vs β vs outcome anchor)"""
import sys; sys.path.insert(0,'/home/claude/harness')
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(open('e26_out.py').read())
OEv=A9[:,6,:]; RLv=A9[np.arange(575),P.RLID,:]; d=(RLv-OEv)/sdg
anc=(lev(rlv)-lev(pols['OE heuristic']))@real
print(f'{"RL action":<8}{"n":>5}  contribution to mean(RL − OE), standardized:  α      β      γ     δ1     δ2     ε1     ε2   | anchor (notches)')
for a in P.IDS:
    m=(P.a3.action_id.values==a)
    if m.sum()==0: continue
    c=d[m].sum(0)/575; print(f'{a:<8}{m.sum():>5}  {"":>44}'.replace(' '*44,'')+'  '+' '.join(f'{x:+6.3f}' for x in c)+f'   | {anc[m].sum()/575:+.4f}')
print(f'{"total":<8}{575:>5}  '+' '.join(f'{x:+6.3f}' for x in d.mean(0))+f'   | {anc.mean():+.4f}')

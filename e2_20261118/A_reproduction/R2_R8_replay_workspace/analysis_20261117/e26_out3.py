import sys; sys.path.insert(0,'/home/claude/harness')
import io, contextlib
buf=io.StringIO()
with contextlib.redirect_stdout(buf): exec(open('e26_out.py').read())
def anchor(A):   # mean anchor score per policy for each bootstrap draw of β
    return (lev(A)@B[:,k].T).mean(0)
def pol(m,c,rep=1): return np.array([P.norm(d) for d in P.acts(P.get('free8','B1','high',c,m,rep=rep))])
print('paired contrasts on the outcome-anchored linear score (notches per firm) [95% CI over β draws]')
for nm,a,b_ in [('RL − OE heuristic',rlv,pols['OE heuristic']),('RL − const max COGS',rlv,pols['const max COGS cut']),
                ('GPT C6-E − C4R',pol('GPT','C6-E'),pol('GPT','C4R')),('Gemini C6-E − C4R',pol('Gemini','C6-E'),pol('Gemini','C4R')),
                ('GPT C6-E − RL',pol('GPT','C6-E'),rlv),('Gemini C6-E − RL',pol('Gemini','C6-E'),rlv)]:
    dd=anchor(a)-anchor(b_); print(f'  {nm:<22} {np.median(dd):+.4f} [{np.percentile(dd,2.5):+.4f},{np.percentile(dd,97.5):+.4f}]')

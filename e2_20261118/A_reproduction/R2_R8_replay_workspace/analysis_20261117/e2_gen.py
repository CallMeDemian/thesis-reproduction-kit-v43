import sys; sys.path.insert(0,'.')
from hz import *
import itertools
H=Harness()
# directed axes: (axis, sign); AP has two directions
DAX=[('growth_capex_reduction_pct',1),('deleveraging_total_debt_pct',1),('refinancing_short_debt_pct',1),('inv_turnover_chg',1),
     ('ar_turnover_chg',1),('ap_turnover_chg',-1),('ap_turnover_chg',1),('cogs_ratio_chg',-1),('sga_ratio_chg',-1)]
STEP=4
comps=[c for c in itertools.product(range(STEP+1),repeat=len(DAX)) if sum(c)<=STEP]
comps=[c for c in comps if not (c[1]>0 and c[2]>0) and not (c[5]>0 and c[6]>0)]
print('grid points',len(comps))
def to_action(c):
    a={k:0.0 for k in AX}
    for (ax,sg),w in zip(DAX,c):
        if w: a[ax]+=sg*(w/STEP)*H.scale[ax]
    return a
acts=[to_action(c) for c in comps]
U=np.array([[a[k]/H.scale[k] for k in AX] for a in acts])     # normalised vectors
n=len(H.base); M=len(acts)
t=time.time()
V=np.full((n,M,3),np.nan)
for i in range(n):
    out=H.evaluate([(i,a) for a in acts])
    V[i,:,0]=out.alpha.values; V[i,:,1]=out.beta.values; V[i,:,2]=out.gamma.values
    if i%100==0: print(i,'%.0fs'%(time.time()-t),flush=True)
np.save('Vgrid_abs.npy',V); np.save('Ugrid.npy',U)
print('done %.0fs'%(time.time()-t),'fail share',np.isnan(V[:,:,0]).mean())

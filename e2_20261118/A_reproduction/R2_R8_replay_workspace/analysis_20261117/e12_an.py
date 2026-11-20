import sys; sys.path.insert(0,'.')
from prep2 import *
pd.set_option('display.width',280)
Z=dict(np.load('e12_assign.npz')); A9=np.load('A9.npy'); n=575; ar=np.arange(n); EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; OEi=IDS.index('OE')
w=np.bincount(RLID,minlength=9)/n; RL=A9[ar,RLID,:]; RLmix=(A9*w[None,:,None]).sum(1)
c2=np.array([IDS.index(c) for c in a3.C2_action]); BC=A9[ar,c2,:]; w2=np.bincount(c2,minlength=9)/n; BCmix=(A9*w2[None,:,None]).sum(1)
pol={'RL policy (C3-E)':(RL,RLmix),'BC policy (C2)':(BC,BCmix)}
for k in sorted({x.split('|')[0] for x in Z}): pol[k]=(Z[k+'|own'],Z[k+'|mix'])
rows=[];rows2=[];rows3=[]
for nm,(own,mix) in pol.items():
    av=own-mix
    rows.append(dict(policy=nm,**{e:av[:,j].mean() for j,e in enumerate(EN)}))
    rows2.append(dict(policy=nm,**{e:ci(av[:,j]) for j,e in enumerate(EN)}))
    rows3.append(dict(policy=nm,**{e+'_tot':own[:,j].mean() for j,e in enumerate(EN)},**{e+'_mix':mix[:,j].mean() for j,e in enumerate(EN)}))
order=['BC policy (C2)','RL policy (C3-E)','GPT C4','GPT C4R','GPT C6-EX','GPT C6-E','GPT E1-C4R','GPT E1-OE','GPT E1-RL','Gemini C4','Gemini C4R','Gemini C6-EX','Gemini C6-E','Gemini E1-C4R','Gemini E1-OE','Gemini E1-RL','verified grid (sel αβγ)']
t=pd.DataFrame(rows).set_index('policy').loc[order]; print('assignment value = V(own action) − V(another firm\'s action from the same policy), mean over 20 permutations'); print(t.round(3).to_string())
t2=pd.DataFrame(rows2).set_index('policy').loc[order]; print(t2.to_string())
t3=pd.DataFrame(rows3).set_index('policy').loc[order]; print('\ntotal value and mix value'); print(t3.round(3).to_string())
# rank consistency of policies across evaluators: total value vs assignment value (LLM arms + RL)
from scipy.stats import spearmanr, kendalltau
keys=[k for k in order if k.startswith(('GPT C','Gemini C','RL','BC'))]
TOT=np.array([pol[k][0].mean(0) for k in keys]); AV=np.array([(pol[k][0]-pol[k][1]).mean(0) for k in keys])
def meancorr(M): 
    c=[spearmanr(M[:,a],M[:,b])[0] for a in range(7) for b in range(a+1,7)]; return np.mean(c),np.min(c)
print('\npolicy-ranking agreement across the 7 evaluators (mean / min pairwise Spearman over %d policies): total value %.2f / %.2f ; assignment value %.2f / %.2f'%(len(keys),*meancorr(TOT),*meancorr(AV)))
# increments in assignment value
for m in ['GPT','Gemini']:
    f=lambda k: Z[f'{m} {k}|own']-Z[f'{m} {k}|mix']
    print(m,'AV(C6-E) − AV(C4R):',' | '.join(f'{e} {ci((f("C6-E")-f("C4R"))[:,j])}' for j,e in enumerate(EN)))
    print(m,'AV(C6-E) − AV(C6-EX):',' | '.join(f'{e} {ci((f("C6-E")-f("C6-EX"))[:,j])}' for j,e in enumerate(EN)))
    print(m,'AV(RL) − AV(C6-E):',' | '.join(f'{e} {ci(((RL-RLmix)-f("C6-E"))[:,j])}' for j,e in enumerate(EN)))
    print(m,'AV(RL) − AV(C4):',' | '.join(f'{e} {ci(((RL-RLmix)-f("C4"))[:,j])}' for j,e in enumerate(EN)))

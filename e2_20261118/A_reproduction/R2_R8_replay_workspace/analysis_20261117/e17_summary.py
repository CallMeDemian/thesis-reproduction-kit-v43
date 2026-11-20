"""e17: consolidated contrasts on the 7 evaluators with 95% bootstrap CIs (numbers for the summary figure and the record)."""
import sys, json; sys.path.insert(0,'.')
from prep2 import *
A9=np.load('A9.npy'); Z=dict(np.load('e12_assign.npz')); n=575; ar=np.arange(n)
EN=['alpha','beta','gamma','δ1','δ2','ε1','ε2']; OE=A9[:,IDS.index('OE'),:]; RL=A9[ar,RLID,:]
P={'OE':OE,'RL':RL}
for k in sorted({x.split('|')[0] for x in Z}): P[k]=Z[k+'|own']
pairs=[('GPT C6-E','GPT C4R'),('Gemini C6-E','Gemini C4R'),('GPT E1-RL','GPT E1-C4R'),('Gemini E1-RL','Gemini E1-C4R'),('GPT C4R','GPT C4'),('Gemini C4R','Gemini C4'),('GPT C6-E','GPT C6-EX'),('Gemini C6-E','Gemini C6-EX'),
       ('GPT E1-OE','GPT E1-C4R'),('Gemini E1-OE','Gemini E1-C4R'),('RL','OE'),('GPT C4','OE'),('Gemini C4','OE'),('GPT C6-E','OE'),('Gemini C6-E','OE'),('GPT C6-E','RL'),('Gemini C6-E','RL')]
out={}
for a,b in pairs:
    d=P[a]-P[b]; r=[boot(d[:,j]) for j in range(7)]; out[f'{a} − {b}']=[[round(float(x),6) for x in t] for t in r]
    print(f'{a} − {b}:',' | '.join(f'{EN[j]} {r[j][0]:+.3f} [{r[j][1]:+.3f},{r[j][2]:+.3f}]' for j in range(7)),'| pos',sum(t[0]>0 for t in r),'sig+',sum(t[1]>0 for t in r),'sig-',sum(t[2]<0 for t in r))
json.dump(out,open('e17_summary.json','w'),ensure_ascii=False,indent=0)

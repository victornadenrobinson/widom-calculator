"""Plot lambda overlap for the actual power-four quadratures used by TDI."""
import csv
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mlp_pipeline'))
from tdi.analyze import rule_nodes
keys=['GL4','GK9','GL8','GK17','GL10','GK21','GL15','GK31','GL32']
rules={}
for key in keys:
    rule=key[:2].lower();n=int(key[2:])
    rules[key]=rule_nodes(dict(rule=rule,nodes=n,ngauss=(n-1)//2,power=4))[0]
shared=np.array([[sum(np.any(np.isclose(v,rules[b],rtol=1e-12,atol=0)) for v in rules[a]) for b in keys] for a in keys])
fig,(ax,strip)=plt.subplots(1,2,figsize=(13,5.5),layout='constrained',gridspec_kw={'width_ratios':[1,1.3]})
ax.imshow(shared,cmap='Blues',vmin=0,vmax=32)
for i in range(len(keys)):
 for j in range(len(keys)):
  ax.text(j,i,str(shared[i,j]),ha='center',va='center',color='white' if shared[i,j]>17 else 'black',fontsize=9)
ax.set_xticks(range(len(keys)),keys,rotation=45,ha='right');ax.set_yticks(range(len(keys)),keys)
ax.set_title('Number of shared lambda values')
for i,key in enumerate(keys):
 strip.scatter(np.sort(rules[key]),np.full(len(rules[key]),i),s=28,color=f'C{(i//2)%10}')
strip.axvline(1/16,color='gray',ls='--',lw=1,label='Shared midpoint: λ = 1/16')
strip.set_xscale('log');strip.set_yticks(range(len(keys)),keys);strip.invert_yaxis();strip.set_xlabel('λ = x⁴ (log scale)');strip.set_title('Actual sampled lambda values');strip.legend(fontsize=8,loc='lower left')
fig.suptitle('Gauss / Kronrod overlap — current repository implementation',fontsize=14)
for ext in ('png','pdf'):fig.savefig(ROOT/f'docs/figures/quadrature_overlap.{ext}',dpi=180)
with (ROOT/'docs/data/quadrature_lambda_values.csv').open('w') as fh:
 wr=csv.writer(fh);wr.writerow(['rule','node_index','lambda','x_lambda_quarter'])
 for key,values in rules.items():
  for i,v in enumerate(values):wr.writerow([key,i,v,v**.25])
with (ROOT/'docs/data/quadrature_overlap.csv').open('w') as fh:
 wr=csv.writer(fh);wr.writerow(['rule',*keys]);wr.writerows([[key,*row] for key,row in zip(keys,shared)])
print('Wrote overlap figure and lambda/overlap CSVs.')

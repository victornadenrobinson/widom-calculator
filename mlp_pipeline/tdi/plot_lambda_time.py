"""Plot block and cumulative integrand per node and weighted contribution vs time."""
import argparse
import math
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quad_diagnostics import inspect


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dirs',nargs='+')
    ap.add_argument('--out',required=True,help='output prefix')
    ap.add_argument('--block-ps',type=float,default=.25)
    a=ap.parse_args()
    if a.block_ps<=0:ap.error('block length must be positive')
    runs=[inspect(d) for d in a.dirs]
    for r in runs:
        n_nodes=len(r['lam']); cols=4; rows=math.ceil(n_nodes/cols)
        fig,axes=plt.subplots(rows,cols,figsize=(16,3*rows),squeeze=False,layout='constrained')
        block=max(1,round(a.block_ps*r['spp'])); scale=1000/r['st']['N']
        for k,ax in enumerate(axes.flat):
            if k>=n_nodes:ax.set_visible(False);continue
            s=r['series'][k]*scale; nb=len(s)//block
            if not nb:ax.set_title(f'node {k}: too short');continue
            ends=np.arange(1,nb+1)*block
            t=ends/r['spp']
            means=np.array([np.mean(s[i*block:(i+1)*block]) for i in range(nb)])
            cumulative=np.cumsum(s)[ends-1]/ends
            ax.plot(t,means,'.-',color='0.7',lw=.8,ms=3,label='block mean')
            ax.plot(t,cumulative,color='tab:blue',lw=1.4,label='cumulative mean')
            ax.set_title(f"node {k}  λ={r['lam'][k]:.4g}  w={r['w'][k]:.4g}",fontsize=9)
            ax.set_xlabel('Production time (ps)');ax.set_ylabel('Integrand (meV/atom)')
            ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7)
        label=f"{r['A'].get('rule','gl').upper()}{n_nodes}"
        fig.suptitle(f"{label}: {r['dir']} | block = {block/r['spp']:.3g} ps | checkpoint validation passed",fontsize=11)
        prefix=Path(a.out);prefix.parent.mkdir(parents=True,exist_ok=True)
        for ext in ('png','pdf'):fig.savefig(f'{prefix}_{label}_nodes.{ext}',dpi=150)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
        for k,s in enumerate(r['series']):
            nb=len(s)//block
            if not nb:continue
            ends=np.arange(1,nb+1)*block
            y=np.cumsum(s)[ends-1]/ends*scale*r['w'][k]
            ax.plot(ends/r['spp'],y,label=f"{k}: λ={r['lam'][k]:.3g}, w={r['w'][k]:.3g}")
        ax.set(xlabel='Production time (ps)',ylabel='Cumulative weighted contribution (meV/atom)',title=f'{label}: per-node contributions')
        ax.grid(alpha=.2);ax.legend(fontsize=7,ncol=2)
        for ext in ('png','pdf'):fig.savefig(f'{prefix}_{label}_weighted.{ext}',dpi=150)
        plt.close(fig)
        print(f'Wrote {prefix}_{label}_nodes and _weighted (.png, .pdf)')
    print('Cumulative curves use overlapping prefixes and are correlated. Each node can have a different y-axis scale; weighted curves sum to the integral.')

if __name__=='__main__':main()

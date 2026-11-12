"""Compare reflectivity spectra from ice_x_baseline.py CSV files."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv',nargs='+');p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--n0',type=float,default=1.5);p.add_argument('--out',required=True)
    a=p.parse_args()
    if len(a.labels)!=len(a.csv):p.error('Supply one label per CSV')
    key=f'R_n0_{a.n0:g}'
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for path,label in zip(a.csv,a.labels):
        with open(path,newline='') as f: rows=list(csv.DictReader(f))
        if not rows or key not in rows[0]:raise SystemExit(f'{path}: missing {key} or empty CSV')
        E=np.array([float(r['energy_eV']) for r in rows]);R=np.array([float(r[key]) for r in rows])
        ax.plot(E,100*R,label=label)
        target=1239.841984/532
        if E[0]<=target<=E[-1]:print(f'{label}: R(532 nm, n0={a.n0:g}) = {100*np.interp(target,E,R):.3f}%')
    ax.axvline(1239.841984/532,color='black',ls=':',lw=1,label='532 nm')
    ax.set(xlim=(.5,5),xlabel='Photon energy (eV)',ylabel='Normal-incidence reflectivity (%)',title=f'Perfect ice X: incident index {a.n0:g}')
    ax.grid(alpha=.2);ax.legend()
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(str(out)+'.'+ext,dpi=160)
    print(f'Wrote {out}.png and .pdf. Labels are user supplied; verify measured pressures and matching response/settings.')

if __name__=='__main__':main()

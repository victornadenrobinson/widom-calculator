"""Pure-H EOS dissociation diagnostics, not a certification of an LLPT.

With --trajectories, recover missing f_H2 from the last half of saved trajectory
frames using mutual-nearest H pairs below 0.85 A. No model or GPU is required.
"""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def number(row,key):
    try: return float(row.get(key,'nan'))
    except (ValueError,TypeError): return float('nan')


def pair_fraction(at,cutoff=.85):
    from ase.neighborlist import neighbor_list
    if not np.all(at.numbers==1):raise ValueError('Not pure H')
    i,j,d=neighbor_list('ijd',at,cutoff)
    nearest=np.full(len(at),-1,dtype=int);distance=np.full(len(at),np.inf)
    for a,b,r in zip(i,j,d):
        if a!=b and r<distance[a]:nearest[a]=b;distance[a]=r
    paired=[a for a,b in enumerate(nearest) if b>=0 and nearest[b]==a]
    return len(paired)/len(at)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('roots',nargs='+')
    ap.add_argument('--model',default='ft_mfinkc_cont');ap.add_argument('--out',required=True)
    ap.add_argument('--trajectories',action='store_true');args=ap.parse_args()
    records=[]
    for root in args.roots:
        for folder in sorted(Path(root).glob(f'{args.model}_h_*K_*at')):
            table=folder/(folder.name+'.csv')
            if not table.exists():continue
            with table.open() as f:rows=list(csv.DictReader(f))
            count=0
            for row in rows:
                if number(row,'nO')!=0:continue
                P,T,N,v=[number(row,k) for k in ('P_GPa','T_set','N','vpa')]
                if not all(math.isfinite(x) for x in (P,T,N,v)):continue
                fraction=number(row,'f_H2');source='summary';frames=0
                if not math.isfinite(fraction) and args.trajectories:
                    path=folder/f'{folder.name}_v{v:.4f}'/'traj.extxyz'
                    try:
                        from ase.io import read
                        traj=read(path,index=':');traj=traj[len(traj)//2:]
                        vals=[pair_fraction(at) for at in traj]
                        fraction=float(np.mean(vals)) if vals else float('nan');frames=len(vals)
                        source='last_half_saved_trajectory'
                    except (OSError,ValueError,IndexError) as e:print(f'  Cannot recover {path}: {e}')
                if math.isfinite(fraction):count+=1
                records.append(dict(P_GPa=P,T_K=T,N=N,vpa_A3=v,f_H2=fraction,
                    epot_eV_atom=number(row,'epot_per_atom'),D_H_cm2s=number(row,'D_H_cm2s'),
                    dmin_HH_A=number(row,'dmin_HH'),fraction_source=source,frames_reanalysed=frames,source=str(table)))
            print(f'{table}: {len(rows)} EOS rows; {count} have molecular-fraction data')
    if not records:raise SystemExit('No pure-H EOS rows found here. Check the pure-H job status or run this on the cluster holding its EOS files.')
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with open(str(out)+'.csv','w') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    panels=[('f_H2','H atoms in H2-like pairs (%)',100),('vpa_A3','Volume (A³/atom)',1),
            ('epot_eV_atom','Potential energy (eV/atom)',1),('D_H_cm2s','H diffusion (cm²/s)',1)]
    for ax,(key,label,scale) in zip(axes.flat,panels):
        anypoints=False
        for source in sorted({r['source'] for r in records}):
            rows=sorted([r for r in records if r['source']==source and math.isfinite(r[key])],key=lambda r:r['P_GPa'])
            if not rows:continue
            anypoints=True
            ax.plot([r['P_GPa'] for r in rows],[r[key]*scale for r in rows],'.-',label=f"{rows[0]['T_K']:g} K, N={rows[0]['N']:g}")
        ax.set(xlabel='Pressure (GPa)',ylabel=label);ax.grid(alpha=.2)
        if key=='f_H2':ax.set_ylim(-2,102)
        if anypoints:ax.legend(fontsize=8)
        else:ax.text(.5,.5,'Observable absent from saved tables',ha='center',transform=ax.transAxes)
    fig.suptitle('Pure hydrogen: molecular fraction and EOS\nGeometric pair cutoff 0.85 A; connecting lines guide the eye, not phase boundaries')
    fig.savefig(str(out)+'.png',dpi=160);fig.savefig(str(out)+'.pdf')
    print(f'Wrote {out}.png, .pdf and .csv. Inspect short contacts and convergence before interpreting dissociation.')

if __name__=='__main__':main()

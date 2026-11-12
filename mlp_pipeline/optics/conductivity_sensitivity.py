"""Inspect optical conductivity and prepare controlled CSHIFT/k-point tests.

Uses VASP dielectric CSVs produced by ice_x_baseline.py; no DC extrapolation.
"""
import argparse
import csv
import json
import re
import shutil
from pathlib import Path


def prepare(source, out):
    source=Path(source);out=Path(out)
    if out.exists():raise SystemExit(f'Output already exists: {out}')
    for name in ('INCAR','POSCAR','POTCAR','KPOINTS'):
        if not (source/name).is_file():raise SystemExit(f'Missing {source/name}')
    incar=(source/'INCAR').read_text()
    if len(re.findall(r'(?im)^\s*CSHIFT\s*=.*$',incar))!=1:raise SystemExit('Expected one CSHIFT line')
    cases=[('cshift_0p05','gam',.05,None),('cshift_0p20','gam',.20,None),('kpoint_quarter','std',None,'Single shifted k point (fractional reciprocal coordinates)\n1\nReciprocal\n0.25 0.25 0.25 1\n')]
    out.mkdir(parents=True)
    for name,executable,shift,kpoints in cases:
        d=out/name;d.mkdir()
        for f in ('INCAR','POSCAR','POTCAR','KPOINTS','provenance.json'):
            if (source/f).is_file():shutil.copy2(source/f,d/f)
        if shift is not None:
            (d/'INCAR').write_text(re.sub(r'(?im)^\s*CSHIFT\s*=.*$',f'CSHIFT = {shift:.2f}',incar))
        if kpoints:(d/'KPOINTS').write_text(kpoints)
        (d/'test.json').write_text(json.dumps(dict(source=str(source.resolve()),vasp='vasp_'+executable,change=name),indent=2)+'\n')
        print(f'{name}: vasp_{executable}; {d}')
    print('Prepared inputs only. Shifted k-point changes sampling; it is not a k-mesh convergence test.')


def plot(specs, out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[]
    for spec in specs:
        if '=' not in spec:raise SystemExit('Use label=directory for every source')
        label,dirname=spec.split('=',1);directory=Path(dirname)
        for tensor in ('density','current'):
            path=directory/f'reflectivity_{tensor}.csv'
            if not path.is_file():print(f'Missing {path}; skipping');continue
            with path.open(newline='') as f: rows=list(csv.DictReader(f))
            e=np.array([float(r['energy_eV']) for r in rows]);eps1=np.array([float(r['epsilon_real_mean']) for r in rows]);eps2=np.array([float(r['epsilon_imag_mean']) for r in rows]);r=np.array([float(x['R_n0_1.52']) for x in rows]);
            if len(e)<2 or np.any(np.diff(e)<=0) or not all(np.all(np.isfinite(x)) for x in (e,eps1,eps2,r)):raise SystemExit(f'Invalid spectrum {path}')
            factor=8.8541878128e-12 / 6.582119569e-16 / 100 # S/cm per eV
            records.append((label,tensor,e,factor*e*eps2,factor*e*(1-eps1),r))
    if not records:raise SystemExit('No readable reflectivity CSVs')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    for label,tensor,e,s1,s2,r in records:
        style='-' if tensor=='density' else '--'
        for ax,y in zip(axes,(s1,s2,100*r)):ax.plot(e,y,style,label=f'{label}: {tensor}')
        en=1239.841984/532
        if e[0]<=en<=e[-1]:
            print(f'{label:24} {tensor:8} 532 nm: sigma1={np.interp(en,e,s1):.1f} S/cm; sigma2={np.interp(en,e,s2):.1f} S/cm; R={np.interp(en,e,100*r):.3f}%')
        else:print(f'{label} {tensor}: 532 nm outside grid')
    for ax in axes:ax.set_xlim(.1,5);ax.grid(alpha=.25);ax.axvline(1239.841984/532,color='grey',lw=.8,alpha=.7);ax.set_xlabel('Photon energy (eV)')
    axes[0].set_ylabel('σ₁ (S/cm)');axes[1].set_ylabel('σ₂ (S/cm)');axes[2].set_ylabel('R at n₀=1.52 (%)')
    axes[0].legend(fontsize=8);fig.suptitle('Electronic response from VASP dielectric tensors; no DC extrapolation')
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(str(out)+'.'+ext,dpi=170)
    print(f'Wrote {out}.png and .pdf. σ₂ uses ε=1+iσ/(ε₀ω); no separate background subtraction.')
    print('The density/current tensors are alternatives, not additive contributions. Optical convergence and explicit intraband treatment remain to be assessed.')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('source');q.add_argument('out')
    q=sub.add_parser('plot');q.add_argument('sources',nargs='+');q.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.source,a.out)
    else:plot(a.sources,a.out)

if __name__=='__main__':main()

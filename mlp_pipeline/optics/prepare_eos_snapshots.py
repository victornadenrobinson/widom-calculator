"""Prepare exploratory VASP-optics inputs for existing pure-H2O/HxO EOS frames.

Selects the screened EOS point nearest the requested pressure per composition;
no extrapolation, MD, job submission, or phase certification is performed.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
from ase.io import read, write

KB=8.617333262e-5


def number(row,key):
    try:return float(row.get(key,''))
    except (ValueError,TypeError):return float('nan')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='runs/eos_live');p.add_argument('--potcar',required=True)
    p.add_argument('--out',required=True);p.add_argument('--temperature',type=int,default=7000)
    p.add_argument('--pressure',type=float,default=350);p.add_argument('--pmin',type=float,default=250)
    p.add_argument('--pmax',type=float,default=450);p.add_argument('--frames',type=int,default=3)
    p.add_argument('--compositions',nargs='+',type=float,default=[2,2.25,3,5,7])
    a=p.parse_args()
    def label(comp): return 'H' if comp == 0 else f'H{comp:g}O'
    if a.temperature<=0 or a.frames<1 or not (0<a.pmin<a.pmax) or a.pressure<=0:p.error('Invalid temperature, pressure or number of frames')
    pot=Path(a.potcar).read_bytes();species=[x.decode().capitalize() for x in re.findall(rb'VRHFIN\s*=\s*([A-Za-z]+)',pot)]
    if species not in (['H','O'],['O','H']):raise SystemExit('Expected exactly H and O POTCAR datasets; got '+str(species))
    output=Path(a.out)
    if output.exists():raise SystemExit('Refusing to overwrite existing output: '+str(output))
    groups={c:[] for c in a.compositions}
    for folder in sorted(Path(a.root).glob('ft_mfinkc_cont_*K_*at')):
        table=folder/(folder.name+'.csv')
        if not table.is_file():continue
        with table.open(newline='') as stream:
            for row in csv.DictReader(stream):
                P,T,N,nO,V=(number(row,k) for k in ('P_GPa','T_set','N','nO','vpa'))
                if not all(math.isfinite(x) for x in (P,T,N,nO,V)) or nO<0 or T!=a.temperature or not a.pmin<=P<=a.pmax:continue
                ratio=(N-nO)/nO if nO else 0.0
                comp=next((c for c in a.compositions if abs(c-ratio)<1e-5),None)
                if comp is None:continue
                temp,dh,do=(number(row,k) for k in ('T_mean','dmin_HH','D_O_cm2s'))
                reason=[]
                if not math.isfinite(temp) or abs(temp-T)>.05*T:reason.append('temperature')
                if nO>0 and (not math.isfinite(dh) or dh<.4):reason.append('H-H contact')
                if nO>0 and (not math.isfinite(do) or do<2e-5):reason.append('O diffusion')
                if nO>0 and row.get('O_struct','').lower() in ('bcc','fcc','hcp'):reason.append('O order')
                drift,sd=number(row,'P_drift_GPa'),number(row,'P_block_sd')
                if not (math.isfinite(drift) and math.isfinite(sd)):reason.append('pressure drift unavailable')
                elif abs(drift)>max(.01*abs(P),3*sd):reason.append('pressure drift')
                traj=folder/f'{folder.name}_v{V:.4f}'/'traj.extxyz'
                if reason or not traj.is_file():
                    print(f'Skip {label(comp)} P={P:.1f}: {", ".join(reason) or "missing trajectory"} [{traj}]')
                    continue
                groups[comp].append((abs(P-a.pressure),P,V,row,table,traj))
    selected=[]
    for comp,candidates in groups.items():
        if not candidates:
            print(f'No screened {label(comp)} EOS point in {a.pmin:g}-{a.pmax:g} GPa at {a.temperature} K.');continue
        _,P,V,row,table,traj=min(candidates,key=lambda t:t[0])
        frames=read(str(traj),index=':')
        if len(frames)<a.frames:print(f'Skip {traj}: only {len(frames)} saved frames');continue
        first=len(frames)//2
        indices=np.unique(np.linspace(first,len(frames)-1,min(a.frames,len(frames)-first),dtype=int))
        if len(indices)<a.frames:print(f'Skip {traj}: last half has fewer than {a.frames} frames');continue
        selected.append((comp,P,V,row,table,traj,frames,indices))
    if not selected:raise SystemExit('No matching screened trajectories found; no directory created.')
    output.mkdir(parents=True)
    report=[]
    for comp,P,V,row,table,traj,frames,indices in selected:
        for seq,idx in enumerate(indices):
            atoms=frames[int(idx)]
            numbers=atoms.numbers
            if set(numbers)-{1,8}:raise ValueError(f'Unexpected species in {traj}')
            actual_species=[sp for sp in species if sp != 'O' or (numbers==8).any()]
            ordered=atoms[[i for sp in actual_species for i,z in enumerate(numbers) if z=={'H':1,'O':8}[sp]]]
            nH=int((numbers==1).sum());nO=int((numbers==8).sum()); ne=6*nO+nH
            if len(atoms)!=int(float(row['N'])) or nO!=int(float(row['nO'])):
                raise ValueError('Frame composition does not match EOS summary: '+str(traj))
            # Conservative starting guess; convergence and highest-band occupations must be checked.
            bands=int(math.ceil(max(.75*ne,ne/2+40)/32)*32)
            run=output/f'h{comp:g}o_P{P:.1f}GPa_frame{seq:02d}'
            run.mkdir();write(str(run/'POSCAR'),ordered,format='vasp',direct=True,sort=False,vasp5=True)
            if nO:
                shutil.copyfile(a.potcar,run/'POTCAR')
                selected_pot=pot
            else:
                datasets=re.findall(rb'.*?End of Dataset[^\n]*(?:\r?\n|$)',pot,re.S)
                hsets=[block for block in datasets if re.search(rb'VRHFIN\s*=\s*H\s*:',block)]
                if len(hsets)!=1 or len(datasets)!=2:raise ValueError('Could not safely isolate unique H POTCAR dataset')
                selected_pot=hsets[0]
                (run/'POTCAR').write_bytes(selected_pot)
            (run/'KPOINTS').write_text('Gamma-point exploratory timing only\n0\nGamma\n1 1 1\n0 0 0\n')
            (run/'INCAR').write_text(f'''SYSTEM = GRACE EOS snapshot {label(comp)} optical timing pilot
ENCUT = 1100
PREC = Accurate
EDIFF = 1E-6
NELM = 160
ISMEAR = -1
SIGMA = {a.temperature*KB:.9f}
IBRION = -1
NSW = 0
ISPIN = 1
ISYM = 0
LREAL = .FALSE.
ALGO = Normal
NBANDS = {bands}
LOPTICS = .TRUE.
NEDOS = 3000
CSHIFT = 0.10
LWAVE = .FALSE.
LCHARG = .FALSE.
NCORE = 4
KPAR = 1
''')
            provenance=dict(temperature_set_K=a.temperature,temperature_mean_K=number(row,'T_mean'),pressure_mean_GPa=P,
                pressure_sem_GPa=number(row,'P_sem'),vpa_A3=V,composition=label(comp),nH=nH,nO=nO,
                eos_table=str(table.resolve()),trajectory=str(traj.resolve()),frame_index_zero_based=int(idx),
                frame_count=len(frames),selected_from='last half of saved production frames',source_row=row,
                potential='ft_mfinkc_cont; exact checkpoint identity not verified from path alone',
                potcar_sha256=hashlib.sha256(selected_pot).hexdigest(),potcar_species=actual_species,nbands=bands,
                notes='Short screening trajectory; phase and pressure not certified. Gamma-point and NBANDS are initial timing choices, not converged optical settings. Distinct frames not necessarily independent.')
            (run/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
            report.append(dict(composition=label(comp),P_GPa=P,T_mean_K=number(row,'T_mean'),frame=int(idx),N=len(atoms),NBANDS=bands,path=str(run)))
            print(f'{label(comp)} P={P:.1f} GPa N={len(atoms)} frame={idx}/{len(frames)} NBANDS={bands} -> {run}')
    with (output/'manifest.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
    print(f'Prepared {len(report)} snapshots; no VASP runs started. Paths/phase/pressure/weights require review.')

if __name__=='__main__':main()

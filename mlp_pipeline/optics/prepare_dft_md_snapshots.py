"""Prepare exploratory optics from existing DFT-MD OUTCAR segments.

Select good inventory rows at given temperatures and pressures; select at most
one source segment per composition and temperature. A path or composition
label does not prove liquid phase or independent sampling. Never overwrite.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from ase.io import read, write

KB=8.617333262e-5

def num(r,k):
    try:
        v=float(r.get(k,''));return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',default='inventory_binned.csv');p.add_argument('--source-root',action='append',required=True)
    p.add_argument('--potcar',required=True);p.add_argument('--out',required=True)
    p.add_argument('--temps',type=int,nargs='+',default=[5000,7000]);p.add_argument('--pmin',type=float,default=250);p.add_argument('--pmax',type=float,default=450)
    p.add_argument('--pressure',type=float,default=350);p.add_argument('--frames',type=int,default=2)
    p.add_argument('--comps',nargs='+',default=['h','h2.25o','h5o','h7o'])
    p.add_argument('--exclude-root',action='append',default=[],help='skip source trajectories and run folders already present in these prepared-case roots')
    p.add_argument('--screen-source-paths',action='store_true',help='exclude TDI lambda windows and explicitly solid/crystal-labelled paths; liquid phase still requires validation')
    p.add_argument('--min-new-pressure-gap',type=float,default=0,help='minimum GPa from an already prepared same-composition, same-temperature DFT-MD state')
    p.add_argument('--allow-empty',action='store_true',help='exit successfully without creating output when no candidate is available')
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise SystemExit('Refusing to overwrite '+str(out))
    if a.frames<1:raise SystemExit('frames must be positive')
    if not math.isfinite(a.min_new_pressure_gap) or a.min_new_pressure_gap<0:raise SystemExit('min-new-pressure-gap must be finite and nonnegative')
    pot=Path(a.potcar).read_bytes();species=[x.decode().capitalize() for x in re.findall(rb'VRHFIN\s*=\s*([A-Za-z]+)',pot)]
    if species not in (['H','O'],['O','H']):raise SystemExit('POTCAR must contain exactly H and O datasets')
    with open(a.inventory,newline='') as f: rows=list(csv.DictReader(f))
    def run_key(path):
        path=Path(path).resolve(); parent=path.parent
        if re.fullmatch(r'out\d+',parent.name,re.I):parent=parent.parent
        return str(parent)
    excluded=set(); existing_states=[]
    for root in a.exclude_root:
        for f in Path(root).glob('h*o_P*_frame*/provenance.json'):
            try:
                info=json.loads(f.read_text());traj=info.get('trajectory')
                if traj and info.get('source_kind')=='DFT-MD OUTCAR segment':
                    excluded.add(run_key(traj))
                    comp=str(info.get('source_row',{}).get('comp_snap','')).lower()
                    t=num(info,'temperature_set_K');pressure=num(info,'pressure_mean_GPa')
                    if comp and t is not None and pressure is not None:existing_states.append((comp,t,pressure))
            except (OSError,ValueError):raise SystemExit('Cannot inspect provenance '+str(f))
    print(f'Excluding {len(excluded)} previously prepared DFT-MD source-run folders.')
    groups={}
    for r in rows:
        c=r.get('comp_snap','').lower();t=num(r,'True_T_K');P=num(r,'True_P_GPa');path=r.get('Path','')
        if c not in a.comps or t is None or P is None or not a.pmin<=P<=a.pmax or not path:continue
        if a.screen_source_paths and (re.search(r'lambda_',path,re.I) or re.search(r'(?:^|[/_-])(?:solid|crystal)(?:[/_-]|$)',path,re.I)):continue
        if str(r.get('good','')).strip().lower() not in ('true','1','yes'):continue
        if any(c==old_c and abs(t-old_t)<=250 and abs(P-old_p)<a.min_new_pressure_gap for old_c,old_t,old_p in existing_states):continue
        matched=min(a.temps,key=lambda x:abs(x-t))
        if abs(t-matched)>250:continue
        choices=[Path(path)] if Path(path).is_absolute() else [Path(x)/path for x in a.source_root]
        source=next((x for x in choices if x.is_file()),None)
        if source is None or run_key(source) in excluded:continue
        # Liquid-labelled trajectories preferred, but not certified by name.
        preference=0 if re.search(r'liquid|liq',path,re.I) else 1
        groups.setdefault((c,matched),[]).append((preference,abs(P-a.pressure),str(source),r))
    selected=[]
    for key in [(c,t) for t in a.temps for c in a.comps]:
        candidates=sorted(groups.get(key,[]),key=lambda x:(x[0],x[1],x[2]))
        if not candidates:print(f'No readable good inventory source for {key}');continue
        for _,_,source,r in candidates:
            try:
                frames=read(source,index=':',format='vasp-out')
                if len(frames)//2<a.frames:raise ValueError('too few frames in second half')
                idx=np.unique(np.linspace(len(frames)//2,len(frames)-1,a.frames,dtype=int))
                selected.append((key,source,r,frames,idx));break
            except Exception as exc:print('Cannot use',source,':',exc)
        else:print('No readable MD trajectory for',key)
    if not selected:
        if a.allow_empty:
            print('No candidate DFT-MD trajectories readable; no output created.')
            return
        raise SystemExit('No candidate DFT-MD trajectories readable; no output created.')
    out.mkdir(parents=True);report=[]
    for (comp,T),source,r,frames,idx in selected:
        P=num(r,'True_P_GPa')
        for seq,i in enumerate(idx):
            atoms=frames[int(i)].copy();atoms.calc=None;nH=int((atoms.numbers==1).sum());nO=int((atoms.numbers==8).sum())
            if nH+nO!=len(atoms) or not nH:raise ValueError('Non-H/O composition in '+source)
            actual=[sp for sp in species if sp!='O' or nO]
            atoms=atoms[[i for sp in actual for i,z in enumerate(atoms.numbers) if z=={'H':1,'O':8}[sp]]]
            ne=nH+6*nO;bands=int(math.ceil(max(.75*ne,ne/2+40)/32)*32)
            name=f'{comp if comp!="h" else "h0o"}_P{P:.1f}GPa_T{T}_frame{seq:02d}'
            d=out/name;d.mkdir();write(str(d/'POSCAR'),atoms,format='vasp',direct=True,sort=False,vasp5=True)
            if nO: chosen=pot
            else:
                sets=re.findall(rb'.*?End of Dataset[^\n]*(?:\r?\n|$)',pot,re.S)
                h=[b for b in sets if re.search(rb'VRHFIN\s*=\s*H\s*:',b)]
                if len(sets)!=2 or len(h)!=1:raise ValueError('Cannot safely isolate hydrogen POTCAR')
                chosen=h[0]
            (d/'POTCAR').write_bytes(chosen)
            (d/'KPOINTS').write_text('Gamma-point exploratory optics\n0\nGamma\n1 1 1\n0 0 0\n')
            (d/'INCAR').write_text(f'''SYSTEM = DFT-MD snapshot {comp} optical pilot
ENCUT = 1100
PREC = Accurate
EDIFF = 1E-6
NELM = 160
ISMEAR = -1
SIGMA = {num(r,'True_T_K')*KB:.9f}
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
            label='H' if not nO else f'H{nH/nO:.4g}O'
            info=dict(composition=label,nH=nH,nO=nO,temperature_set_K=num(r,'True_T_K'),pressure_mean_GPa=P,
                      trajectory=str(Path(source).resolve()),frame_index_zero_based=int(i),frame_count=len(frames),
                      source_kind='DFT-MD OUTCAR segment',inventory_path=str(Path(a.inventory).resolve()),source_row=r,
                      pressure_note='inventory corrected pressure; source-run mean, not frame-specific pressure',
                      phase='unverified',potcar_sha256=hashlib.sha256(chosen).hexdigest(),potcar_species=actual,
                      notes='Existing DFT-MD frames; source functional, phase, equilibration and independence require verification; gamma/NBANDS unconverged.')
            (d/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
            report.append(dict(composition=label,T_K=T,P_GPa=P,nH=nH,nO=nO,N=len(atoms),frame=int(i),source=source,path=str(d),NBANDS=bands))
            print(f'{label} T={T} P={P:.1f} N={len(atoms)} frame={i}/{len(frames)} -> {d}')
    with (out/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(report[0]));w.writeheader();w.writerows(report)
    print(f'Prepared {len(report)} snapshots. Phase and source ensemble not certified; no VASP started.')

if __name__=='__main__':main()

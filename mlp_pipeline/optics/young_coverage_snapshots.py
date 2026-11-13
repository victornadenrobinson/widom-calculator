"""Export screened Young EOS configurations, or prepare portable VASP inputs.

No pseudopotential is included in the export. Source pressure is a GRACE MD
mean, and selection is a screening heuristic, not phase certification.
"""
import argparse,csv,hashlib,json,math,re
from pathlib import Path
from ase.io import read,write

KB=8.617333262e-5

def num(row,key):
    try:
        value=float(row.get(key,''))
        return value if math.isfinite(value) else float('nan')
    except (ValueError,TypeError):return float('nan')

def export(a):
    out=Path(a.out)
    if out.exists():raise SystemExit(f'Output exists: {out}; choose a new directory')
    candidates={};audit=[]
    for folder in sorted(Path(a.root).glob('ft_mfinkc_cont_*K_*at')):
        table=folder/(folder.name+'.csv')
        if not table.is_file():continue
        with table.open(newline='') as f:
            for row in csv.DictReader(f):
                P,T,N,nO,V=(num(row,k) for k in ('P_GPa','T_set','N','nO','vpa'))
                if not all(map(math.isfinite,(P,T,N,nO,V))) or nO!=128 or T not in a.temps or not a.pmin<=P<=a.pmax:continue
                ratio=(N-nO)/nO
                if not any(abs(ratio-c)<1e-5 for c in a.compositions):continue
                traj=folder/f'{folder.name}_v{V:.4f}'/'traj.extxyz'; reasons=[]
                temp,hh,diff=(num(row,k) for k in ('T_mean','dmin_HH','D_O_cm2s'))
                if not math.isfinite(temp) or abs(temp-T)>.05*T:reasons.append('temperature')
                if not math.isfinite(hh) or hh<.4:reasons.append('H-H contact')
                if not math.isfinite(diff) or diff<2e-5:reasons.append('O diffusion')
                if row.get('O_struct','').lower() in ('bcc','fcc','hcp'):reasons.append('O order')
                drift,sd=num(row,'P_drift_GPa'),num(row,'P_block_sd')
                if not math.isfinite(drift) or not math.isfinite(sd):reasons.append('pressure drift unavailable')
                elif abs(drift)>max(.01*abs(P),3*sd):reasons.append('pressure drift')
                if not traj.is_file():reasons.append('trajectory missing')
                audit.append(dict(T=T,composition=ratio,P_GPa=P,vpa=V,trajectory=str(traj),screen='; '.join(reasons) or 'pass'))
                if reasons:continue
                # 250, 275, ... 450 GPa bins, centred on the stated value.
                b=min(int((P-a.pmin+a.spacing/2)//a.spacing),int((a.pmax-a.pmin)//a.spacing))
                target=a.pmin+b*a.spacing
                key=(int(T),ratio,b)
                candidate=(abs(P-target),str(traj),row,str(table),P,V)
                if key not in candidates or candidate[:2]<candidates[key][:2]:candidates[key]=candidate
    out.mkdir(parents=True)
    with (out/'screening.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['T','composition','P_GPa','vpa','trajectory','screen']);w.writeheader();w.writerows(audit)
    manifest=[]
    for (T,ratio,b),(_,path,row,table,P,V) in sorted(candidates.items()):
        try:
            frames=read(path,index=':')
            if len(frames)<2:raise ValueError('fewer than two saved production frames')
            index=(len(frames)-1+len(frames)//2)//2
            atoms=frames[index]
            nO=int((atoms.numbers==8).sum());nH=int((atoms.numbers==1).sum())
            if nO!=128 or nH+nO!=len(atoms) or abs(nH/nO-ratio)>1e-5:raise ValueError('frame composition does not match EOS row')
            name=f'h{ratio:g}o_P{P:.1f}GPa_T{T}_frame00'; case=out/name;case.mkdir()
            atoms=atoms.copy();atoms.calc=None;write(case/'frame.extxyz',atoms,format='extxyz')
            info=dict(composition=f'H{ratio:g}O',nH=nH,nO=nO,temperature_set_K=T,temperature_mean_K=num(row,'T_mean'),pressure_mean_GPa=P,vpa_A3=V,trajectory=str(Path(path).resolve()),eos_table=table,frame_index_zero_based=index,frame_count=len(frames),source_kind='GRACE-MD EOS',source_row=row,notes='Selected from second half of saved production frames; screen is not phase or convergence certification; checkpoint identity not independently verified.')
            (case/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
            manifest.append(dict(case=name,T_K=T,composition=info['composition'],P_GPa=P,N=len(atoms),trajectory=path,frame=index))
            print(f"{name}: {len(atoms)} atoms, frame {index}/{len(frames)}")
        except Exception as exc:print(f'FAILED {path}: {exc}')
    with (out/'manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case','T_K','composition','P_GPa','N','trajectory','frame']);w.writeheader();w.writerows(manifest)
    print(f'Exported {len(manifest)} cases. Candidate screening: {out}/screening.csv')

def prepare(a):
    src=Path(a.bundle);out=Path(a.out)
    if out.exists():raise SystemExit(f'Output exists: {out}; choose a new directory')
    pot=Path(a.potcar).read_bytes();species=[x.decode().capitalize() for x in re.findall(rb'VRHFIN\s*=\s*([A-Za-z]+)',pot)]
    if species not in (['H','O'],['O','H']):raise SystemExit('POTCAR must contain exactly H and O datasets')
    cases=sorted(src.glob('h*o_P*_frame*/provenance.json'))
    if not cases:raise SystemExit('No exported cases found')
    out.mkdir(parents=True)
    for f in cases:
        info=json.loads(f.read_text());atoms=read(f.parent/'frame.extxyz');nH=int((atoms.numbers==1).sum());nO=int((atoms.numbers==8).sum())
        if (nH,nO)!=(info['nH'],info['nO']) or nH+nO!=len(atoms):raise ValueError(f'composition mismatch: {f}')
        atoms=atoms[[i for sp in species for i,z in enumerate(atoms.numbers) if z=={'H':1,'O':8}[sp]]]
        case=out/f.parent.name;case.mkdir();write(case/'POSCAR',atoms,format='vasp',direct=True,sort=False,vasp5=True)
        (case/'POTCAR').write_bytes(pot)
        (case/'KPOINTS').write_text('Gamma-point exploratory optics\n0\nGamma\n1 1 1\n0 0 0\n')
        bands=int(math.ceil(max(.75*(nH+6*nO),(nH+6*nO)/2+40)/32)*32)
        (case/'INCAR').write_text(f'''SYSTEM = GRACE-MD snapshot optical pilot
ENCUT = 1100
PREC = Accurate
EDIFF = 1E-6
NELM = 160
ISMEAR = -1
SIGMA = {info['temperature_set_K']*KB:.9f}
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
        info.update(potcar_sha256=hashlib.sha256(pot).hexdigest(),potcar_species=species,nbands=bands)
        (case/'provenance.json').write_text(json.dumps(info,indent=2)+'\n')
        print(f"{case}: {len(atoms)} atoms, {bands} bands")
    print(f'Prepared {len(cases)} cases; no VASP started.')

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True)
    e=s.add_parser('export');e.add_argument('--root',default='runs/eos_live');e.add_argument('--out',required=True);e.add_argument('--temps',nargs='+',type=int,default=[5000,6000,7000]);e.add_argument('--compositions',nargs='+',type=float,default=[2,2.25,3,4,5,7]);e.add_argument('--pmin',type=float,default=250);e.add_argument('--pmax',type=float,default=450);e.add_argument('--spacing',type=float,default=25)
    q=s.add_parser('prepare');q.add_argument('--bundle',required=True);q.add_argument('--potcar',required=True);q.add_argument('--out',required=True)
    a=p.parse_args()
    if a.mode=='export':
        if a.spacing<=0 or a.pmax<=a.pmin:raise SystemExit('invalid pressure grid')
        export(a)
    else:prepare(a)
if __name__=='__main__':main()

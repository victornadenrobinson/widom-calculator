"""Prepare three new frames between two existing DFT-MD optical snapshots.

Reuses the exact source segment and original optical settings. Validates the
source trajectory against both existing POSCARs before preparing new cases.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from ase.io import read, write


def reorder(atoms, species):
    if set(atoms.get_chemical_symbols()) - {'H','O'}:
        raise ValueError('source contains non-H/O atoms')
    ordered=atoms[[i for sp in species for i,z in enumerate(atoms.get_chemical_symbols()) if z==sp]]
    if len(ordered)!=len(atoms):raise ValueError('POTCAR species do not cover source atoms')
    ordered.calc=None
    return ordered


def check_frame(source, case, info):
    reference=read(str(case/'POSCAR'),format='vasp')
    atoms=reorder(source,info['potcar_species'])
    if atoms.get_chemical_symbols()!=reference.get_chemical_symbols() or not np.allclose(atoms.cell,reference.cell,atol=1e-6,rtol=1e-7):
        raise ValueError(f'{case}: source species/cell do not match POSCAR')
    diff=atoms.get_scaled_positions(wrap=False)-reference.get_scaled_positions(wrap=False)
    diff-=np.rint(diff)
    if not np.allclose(diff,0,atol=2e-5,rtol=0):raise ValueError(f'{case}: source frame does not match POSCAR')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('source_root',type=Path);ap.add_argument('out',type=Path);a=ap.parse_args()
    if a.out.exists():raise SystemExit(f'Output already exists: {a.out}; preserving it')
    originals=sorted(a.source_root.glob('h*o_P*_T*_frame00'))
    if len(originals)!=8:raise SystemExit(f'Expected 8 original state directories; found {len(originals)}')
    a.out.mkdir(parents=True);report=[]
    for first in originals:
        second=first.with_name(first.name[:-2]+'01')
        info=json.loads((first/'provenance.json').read_text());info2=json.loads((second/'provenance.json').read_text())
        for f in ('INCAR','POTCAR','KPOINTS','POSCAR'):
            if not (first/f).is_file() or not (second/f).is_file():raise SystemExit(f'Missing {f} in original state {first}')
        if info.get('source_kind')!='DFT-MD OUTCAR segment' or info.get('trajectory')!=info2.get('trajectory'):
            raise SystemExit(f'{first}: source provenance is not a shared DFT-MD segment')
        for f in ('INCAR','POTCAR','KPOINTS'):
            if (first/f).read_bytes()!=(second/f).read_bytes():raise SystemExit(f'{first}: {f} differs between original frames')
        source=Path(info['trajectory'])
        frames=read(str(source),index=':',format='vasp-out')
        if len(frames)!=int(info['frame_count']) or len(frames)!=int(info2['frame_count']):
            raise SystemExit(f'{source}: trajectory length changed')
        i0,i1=int(info['frame_index_zero_based']),int(info2['frame_index_zero_based'])
        if not 0<=i0<i1<len(frames):raise SystemExit(f'{source}: invalid original frame indices')
        check_frame(frames[i0],first,info);check_frame(frames[i1],second,info2)
        indices=np.linspace(i0,i1,5,dtype=int)[1:-1].tolist()
        if len(set(indices+[i0,i1]))!=5:raise SystemExit(f'{source}: not enough distinct frames')
        for seq,idx in enumerate(indices,2):
            target=a.out/(first.name[:-2]+f'{seq:02d}');target.mkdir()
            for f in ('INCAR','POTCAR','KPOINTS'):shutil.copy2(first/f,target/f)
            write(str(target/'POSCAR'),reorder(frames[idx],info['potcar_species']),format='vasp',direct=True,sort=False,vasp5=True)
            new=dict(info);new.update(frame_index_zero_based=idx,selected_between=[i0,i1],original_case=str(first.resolve()),notes='Additional source-segment frame; optical inputs match original. Phase and independence not certified.')
            (target/'provenance.json').write_text(json.dumps(new,indent=2)+'\n')
            report.append(dict(composition=info['composition'],T_K=info['temperature_set_K'],P_GPa=info['pressure_mean_GPa'],frame=idx,source=str(source),path=str(target)))
            print(f"{info['composition']} {info['temperature_set_K']} K {info['pressure_mean_GPa']:.2f} GPa: frame {idx}/{len(frames)} -> {target}")
    with (a.out/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(report[0]));writer.writeheader();writer.writerows(report)
    print(f'Prepared {len(report)} new snapshots. Same trajectory segments and settings; phase and independence remain to be checked.')

if __name__=='__main__':main()

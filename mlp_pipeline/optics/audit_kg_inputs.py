"""Read-only audit of saved VASP files for an independent Kubo–Greenwood pilot."""
import argparse
import importlib.util
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dirs',nargs='+',type=Path);a=p.parse_args()
    for package in ('numpy','scipy','pymatgen','ase'):
        try:present=importlib.util.find_spec(package) is not None
        except (ImportError,ValueError):present=False
        print(f'Python {package}: {"available" if present else "not available"}')
    for d in a.dirs:
        print('\n'+str(d), flush=True)
        if not d.is_dir():print('  directory missing');continue
        for name in ('INCAR','KPOINTS','POTCAR','POSCAR','OUTCAR','vasprun.xml','EIGENVAL','WAVEDER','WAVEDERF','WAVECAR','provenance.json'):
            f=d/name
            print(f'  {name:16} {f.stat().st_size if f.is_file() else "missing"}', flush=True)
        incar=d/'INCAR'
        if incar.is_file():
            for line in incar.read_text(errors='replace').splitlines():
                if re.match(r'^\s*(LOPTICS|LWAVE|LVEL|ISMEAR|SIGMA|NBANDS|CSHIFT|NEDOS|ISYM|ISPIN|WPLASMAI)\s*=',line,re.I):
                    print('  INCAR:',line.strip())
        out=d/'OUTCAR'
        if out.is_file():
            patterns=('WPLASMAI','E-fermi','aborting loop because EDIFF','General timing and accounting','WAVEDER')
            print('  Reading first 2 MiB and last 4 MiB of OUTCAR...', flush=True)
            with out.open('rb') as f:
                head=f.read(2*1024*1024)
                size=out.stat().st_size
                f.seek(max(len(head),size-4*1024*1024))
                tail=f.read(4*1024*1024)
            chunks=(head+b'\n'+tail).decode(errors='replace').splitlines()
            for key in patterns:
                hits=[line.strip() for line in chunks if key.lower() in line.lower()]
                if hits:print(f'  OUTCAR {key}: {hits[-1][:220]}', flush=True)
                else:print(f'  OUTCAR {key}: not found in sampled sections', flush=True)
        print('  Nonempty matrix-element files are a candidate input, not proof of finite-T completeness or compatibility.')

if __name__=='__main__':main()

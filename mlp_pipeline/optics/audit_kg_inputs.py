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
        print('\n'+str(d))
        if not d.is_dir():print('  directory missing');continue
        for name in ('INCAR','KPOINTS','POTCAR','POSCAR','OUTCAR','vasprun.xml','EIGENVAL','WAVEDER','WAVEDERF','WAVECAR','provenance.json'):
            f=d/name
            print(f'  {name:16} {f.stat().st_size if f.is_file() else "missing"}')
        incar=d/'INCAR'
        if incar.is_file():
            for line in incar.read_text(errors='replace').splitlines():
                if re.match(r'^\s*(LOPTICS|LWAVE|LVEL|ISMEAR|SIGMA|NBANDS|CSHIFT|NEDOS|ISYM|ISPIN|WPLASMAI)\s*=',line,re.I):
                    print('  INCAR:',line.strip())
        out=d/'OUTCAR'
        if out.is_file():
            patterns=('WPLASMAI','E-fermi','aborting loop because EDIFF','General timing and accounting','WAVEDER')
            hits={key:[] for key in patterns}
            with out.open(errors='replace') as f:
                for line in f:
                    for key in patterns:
                        if key.lower() in line.lower():hits[key].append(line.strip())
            for key in patterns:
                if hits[key]:print(f'  OUTCAR {key}: {hits[key][-1][:220]}')
                else:print(f'  OUTCAR {key}: not found')
        print('  Nonempty matrix-element files are a candidate input, not proof of finite-T completeness or compatibility.')

if __name__=='__main__':main()

"""Prepare independent fixed-geometry VASP convergence tests from an ice-X pilot.

Only inputs are copied; no prior outputs or licensed POTCAR contents are printed.
All output directories must be new. Run VASP separately in each directory.
"""
import argparse
import json
from pathlib import Path
import re
import shutil

TESTS = {
    'cold300K': {'SIGMA': '0.025852'},
    'cold300K_bands128': {'SIGMA': '0.025852', 'NBANDS': '128'},
    'cold300K_mesh10_bands128': {'SIGMA': '0.025852', 'NBANDS': '128', 'mesh': 10},
    'cold300K_encut1200_mesh10_bands128': {'SIGMA': '0.025852', 'NBANDS': '128', 'mesh': 10, 'ENCUT': '1200'},
    'cold300K_cshift005_mesh10_bands128': {'SIGMA': '0.025852', 'NBANDS': '128', 'mesh': 10, 'CSHIFT': '0.05'},
    'bands128': {'NBANDS': '128'},
    'mesh10_bands128': {'NBANDS': '128', 'mesh': 10},
    'encut1200_mesh10_bands128': {'NBANDS': '128', 'mesh': 10, 'ENCUT': '1200'},
    'cshift005_mesh10_bands128': {'NBANDS': '128', 'mesh': 10, 'CSHIFT': '0.05'},
}


def set_tag(text,key,value):
    pat=re.compile(r'^\s*'+re.escape(key)+r'\s*=.*$',re.I|re.M)
    if len(pat.findall(text))!=1:
        raise ValueError('Expected exactly one '+key+' setting')
    return pat.sub(key+' = '+str(value),text)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',required=True)
    ap.add_argument('--out-root',required=True)
    ap.add_argument('--tests',nargs='+',choices=list(TESTS),default=list(TESTS))
    a=ap.parse_args();base=Path(a.baseline);outroot=Path(a.out_root)
    for name in ('POSCAR','POTCAR','INCAR','KPOINTS','setup.json'):
        if not (base/name).is_file():raise SystemExit('Missing baseline input '+str(base/name))
    dests=[outroot/name for name in a.tests]
    if len(set(a.tests))!=len(a.tests):raise SystemExit('Duplicate test names')
    for dest in dests:
        if dest.exists():raise SystemExit('Refusing to overwrite existing directory '+str(dest))
    incar=(base/'INCAR').read_text();meta=json.loads((base/'setup.json').read_text())
    for name,dest in zip(a.tests,dests):
        changes=TESTS[name];updated=incar
        for key,value in changes.items():
            if key!='mesh':updated=set_tag(updated,key,value)
        dest.mkdir(parents=True)
        for f in ('POSCAR','POTCAR'):shutil.copyfile(base/f,dest/f)
        (dest/'INCAR').write_text(updated)
        if 'mesh' in changes:
            mesh=changes['mesh'];(dest/'KPOINTS').write_text(f'Ice X convergence test\n0\nGamma\n{mesh} {mesh} {mesh}\n0 0 0\n')
        else:shutil.copyfile(base/'KPOINTS',dest/'KPOINTS')
        record={'baseline':str(base.resolve()),'test':name,'changes':changes,'baseline_setup':meta,
                'notes':'Same fixed cell. No extrapolation of R; compare measured stress and both dielectric responses. Cold-electron case changes physical temperature, not merely numerics.'}
        (dest/'convergence_setup.json').write_text(json.dumps(record,indent=2)+'\n')
        print(f'Prepared {dest}: {changes}')
    print('No VASP runs started. Preserve all directories and check SCF convergence before comparing.')

if __name__=='__main__':main()

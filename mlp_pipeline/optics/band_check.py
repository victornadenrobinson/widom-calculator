"""Prepare a higher-band snapshot and inspect VASP EIGENVAL occupations.

Use identical inputs except NBANDS; never modify the original calculation.
Occupation and spectral coverage are necessary diagnostics, not convergence tests.
"""
import argparse
from pathlib import Path
import re
import shutil


def prepare(src,out,bands):
    src=Path(src);out=Path(out)
    for name in ('INCAR','KPOINTS','POSCAR','POTCAR'):
        if not (src/name).is_file(): raise SystemExit('Missing input: '+str(src/name))
    incar=(src/'INCAR').read_text()
    pattern=r'(?im)^\s*NBANDS\s*=.*$'
    if len(re.findall(pattern,incar))!=1: raise SystemExit('Expected exactly one NBANDS line')
    old=int(re.search(r'(?im)^\s*NBANDS\s*=\s*(\d+)',incar).group(1))
    if bands<=old: raise SystemExit('New NBANDS must exceed original')
    if out.exists(): raise SystemExit('Refusing to overwrite '+str(out))
    out.mkdir(parents=True)
    for name in ('KPOINTS','POSCAR','POTCAR','provenance.json'):
        if (src/name).is_file():shutil.copy2(src/name,out/name)
    (out/'INCAR').write_text(re.sub(pattern,'NBANDS = '+str(bands),incar))
    print(f'Prepared {out}: NBANDS {old} -> {bands}; all other VASP inputs copied.')


def inspect(path):
    d=Path(path); e=d/'EIGENVAL'; o=d/'OUTCAR'
    if not e.is_file() or not o.is_file(): print(f'{d}: missing EIGENVAL or OUTCAR');return
    lines=e.read_text(errors='replace').splitlines()
    try:
        header=lines[5].split(); nk=int(header[1]); nb=int(header[2]); cursor=6; results=[]
        for _ in range(nk):
            while cursor<len(lines) and not lines[cursor].strip():cursor+=1
            kline=lines[cursor].split();cursor+=1
            if len(kline)!=4:raise ValueError('unexpected k-point line')
            states=[]
            for i in range(nb):
                parts=lines[cursor].split();cursor+=1
                if len(parts)!=3 or int(parts[0])!=i+1:raise ValueError('expected non-spin-polarized EIGENVAL bands')
                states.append((float(parts[1]),float(parts[2])))
            low=[en for en,occ in states if occ<1e-5]
            results.append((states[-1][0],states[-1][1],min(low) if low else None))
        txt=o.read_text(errors='replace');matches=re.findall(r'E-fermi\s*:\s*([-+0-9.Ee]+)',txt)
        if not matches:raise ValueError('no E-fermi in OUTCAR')
        ef=float(matches[-1]);maxocc=max(x[1] for x in results)
        spans=[top-first for top,occ,first in results if first is not None]
        print(f'{d}: {nk} k-point(s), {nb} bands, E-fermi={ef:.3f} eV')
        print(f'  largest highest-band occupation: {maxocc:.3g}; smallest top-band energy above E-fermi: {min(t-ef for t,occ,first in results):.2f} eV')
        if len(spans)==nk: print(f'  smallest energy span from first occupation <1e-5 to highest band: {min(spans):.2f} eV')
        else:print('  At least one k-point has no band with occupation <1e-5.')
        print('  Check occupations and compare R directly; definitions and convergence are method-dependent.')
    except (ValueError,IndexError) as exc:print(f'{d}: could not parse EIGENVAL: {exc}')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('source');q.add_argument('out');q.add_argument('--nbands',type=int,required=True)
    q=sub.add_parser('inspect');q.add_argument('dirs',nargs='+')
    a=p.parse_args()
    if a.command=='prepare':prepare(a.source,a.out,a.nbands)
    else:
        for d in a.dirs:inspect(d)
if __name__=='__main__':main()

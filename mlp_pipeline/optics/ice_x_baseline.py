"""Prepare and inspect a six-atom finite-electron-T ice-X VASP optics pilot.

Fixed cubic geometry, no ions at finite T, no phase certification. Results are
independent-particle dielectric spectra, not validated liquid reflectivities.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET

KB=8.617333262e-5
HC=1239.841984


def prepare(a):
    data=Path(a.potcar).read_bytes()
    elements=re.findall(rb'VRHFIN\s*=\s*([A-Za-z]+)', data)
    species=[x.decode().capitalize() for x in elements]
    if species not in (['O','H'], ['H','O']):
        raise SystemExit(f'POTCAR must contain exactly O and H; parsed species {species}. Select the matching project POTCAR.')
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    lattice=(6*a.vpa)**(1/3)
    coords = {
        'O': ['0 0 0', '0.5 0.5 0.5'],
        'H': ['0.25 0.25 0.25', '0.75 0.75 0.25',
              '0.75 0.25 0.75', '0.25 0.75 0.75'],
    }
    positions = '\n'.join(pos for element in species for pos in coords[element])
    counts = ' '.join(str(len(coords[element])) for element in species)
    (out/'POSCAR').write_text(f'''Perfect ice X, fixed cubic 6-atom cell, V/atom={a.vpa:.10f} A3
1.0
{lattice:.12f} 0 0
0 {lattice:.12f} 0
0 0 {lattice:.12f}
{' '.join(species)}
{counts}
Direct
{positions}
''')
    (out/'INCAR').write_text(f'''SYSTEM = ice X fixed-geometry finite-T electronic optics pilot
ENCUT = 1000
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
NBANDS = {a.nbands}
LOPTICS = .TRUE.
NEDOS = 3000
CSHIFT = 0.10
LWAVE = .FALSE.
LCHARG = .FALSE.
NCORE = 4
''')
    (out/'KPOINTS').write_text(f'Ice X convergence pilot\n0\nGamma\n{a.mesh} {a.mesh} {a.mesh}\n0 0 0\n')
    shutil.copyfile(a.potcar,out/'POTCAR')
    meta=dict(vpa_A3=a.vpa,electronic_T_K=a.temperature,geometry='perfect 6-atom cubic ice X',ionic_T='static geometry',pressure='not fixed; inspect VASP stress',mesh=[a.mesh]*3,nbands=a.nbands,encut_eV=1000,potcar_sha256=hashlib.sha256(data).hexdigest(),potcar_species=species,sigma_eV=a.temperature*KB,cshift_eV=.1,notes='Initial exploratory PBE-PAW baseline; verify POTCAR functional/version, optics broadening, bands, k mesh, occupations and electronic pressure; no explicit Drude term.')
    (out/'setup.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(f"Prepared {out.resolve()} | a={lattice:.6f} A | SIGMA={a.temperature*KB:.6f} eV | POTCAR {','.join(species)} | {a.nbands} bands")
    print('Fixed geometry is a baseline, not liquid or a proven 350-400 GPa / 7000 K state.')


def rows(element, name):
    root=element.find(name)
    if root is None:raise ValueError(f'Missing dielectricfunction/{name}')
    result=[]
    for row in root.findall('.//r'):
        try: nums=[float(x) for x in (row.text or '').split()]
        except ValueError:continue
        if len(nums)>=4:result.append(nums)
    if not result:raise ValueError(f'No tensor rows in {name}')
    return result


def vasp_pressure(path):
    """Last OUTCAR stress in kB, with VASP positive compression convention."""
    pressure = None
    try:
        with Path(path).open(errors='replace') as stream:
            for line in stream:
                match = re.match(r'^\s*in kB\s+(.*)$', line)
                if not match:
                    continue
                try:
                    values = [float(x) for x in match.group(1).split()[:3]]
                except ValueError:
                    continue
                if len(values) == 3 and all(math.isfinite(x) for x in values):
                    pressure = sum(values) / 30.0  # kbar -> GPa; mean of xx,yy,zz
    except OSError:
        pass
    return pressure

def plot(a):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matches=[]
    for _,e in ET.iterparse(a.xml,events=('end',)):
        if e.tag=='dielectricfunction':
            try:
                real=rows(e,'real');imag=rows(e,'imag')
                matches.append((e.get('comment',''),real,imag))
            except ValueError:pass
            e.clear()
    if not matches:raise SystemExit('No complete dielectricfunction tensor found in vasprun.xml; inspect the VASP run and LOPTICS output.')
    if len(matches)==2 and all(not m[0] for m in matches):
        if not a.tensor:raise SystemExit('Two unlabelled tensors: VASP 5.4.4 convention is density first, current second; verify OUTCAR and pass --tensor density or current.')
        index_selected = {'density': 0, 'current': 1}[a.tensor]
        comment,re_,im_=matches[index_selected]
        comment = a.tensor + '-response (unlabelled XML; verify OUTCAR ordering)'
    elif len(matches)==1:
        comment,re_,im_=matches[0]
        if a.tensor: print('Only one dielectric tensor; selected response:', a.tensor)
    else:
        raise SystemExit(f'Found {len(matches)} dielectric tensors ({[m[0] for m in matches]}); inspect XML before processing.')
    re_=np.array(re_);im_=np.array(im_)
    if re_.shape!=im_.shape or not np.allclose(re_[:,0],im_[:,0]):raise SystemExit('Real and imaginary dielectric grids differ')
    energy=re_[:,0]; mask=(energy>0)&np.isfinite(energy)
    energy=energy[mask];re_=re_[mask];im_=im_[mask]
    if len(energy)<2 or np.any(np.diff(energy)<=0):raise SystemExit('Dielectric energy grid is too short or unordered')
    eps=np.mean(re_[:,1:4],axis=1)+1j*np.mean(im_[:,1:4],axis=1)
    index=np.sqrt(eps.astype(complex)); index=np.where(index.imag<0,-index,index)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    outcar=Path(a.outcar) if a.outcar else Path(a.xml).with_name('OUTCAR')
    pressure=vasp_pressure(outcar)
    p_text=f'{pressure:.2f} GPa' if pressure is not None else 'pressure unavailable'
    labels={1.0:'Vacuum',1.3:'Clathrate scenario: lower guess',1.5:'Clathrate scenario: upper guess',1.52:'Ice VII (Millot)',1.7:'High-index sensitivity'}
    ns=list(labels)
    refl={n:np.abs((index-n)/(index+n))**2 for n in ns}
    with Path(str(out)+'.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['pressure_GPa','energy_eV','wavelength_nm','epsilon_real_mean','epsilon_imag_mean','n','k']+[f'R_n0_{n:g}' for n in ns])
        for i,E in enumerate(energy):w.writerow([pressure if pressure is not None else '',E,HC/E,eps[i].real,eps[i].imag,index[i].real,index[i].imag]+[refl[n][i] for n in ns])
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    for n in ns:ax.plot(energy,refl[n],label=f'{labels[n]} (n₀={n:g})')
    ax.set(xlim=(.5,min(5,float(energy[-1]))),ylim=(0,1),xlabel='Photon energy (eV)',ylabel='Normal-incidence reflectivity',title=f'Perfect ice X, {p_text}: {comment or "dielectric"} optics')
    ax.grid(alpha=.2);ax.legend(title='Incident index')
    for ext in ('png','pdf'):fig.savefig(str(out)+'.'+ext,dpi=160)
    print(f'Wrote {out}.csv/.png/.pdf; dielectric response: {comment or "unspecified"}.')
    print(f'VASP pressure: {p_text} (mean diagonal stress, positive compression; source: {outcar}).')
    for wavelength in (532,1064):
        e=HC/wavelength
        if energy[0]<=e<=energy[-1]:
            print(f'  {wavelength} nm ({e:.3f} eV), at {p_text}:')
            for n in ns: print(f'    {labels[n]} (n0={n:g}): R={100*np.interp(e,energy,refl[n]):.3f}%')
    print('Clathrate indices are illustrative hypotheses, not independently calculated or measured. Verify experimental wavelength, convergence, intraband response and VASP completion.')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--potcar',required=True);q.add_argument('--out',required=True);q.add_argument('--vpa',type=float,default=2.24);q.add_argument('--temperature',type=float,default=7000);q.add_argument('--mesh',type=int,default=8);q.add_argument('--nbands',type=int,default=64)
    q=sub.add_parser('plot');q.add_argument('xml');q.add_argument('--out',required=True);q.add_argument('--tensor',choices=['density','current'],help='select VASP 5.4.4 density or current response');q.add_argument('--outcar',help='OUTCAR for pressure (default: alongside XML)')
    a=p.parse_args()
    if a.cmd=='prepare':
        if a.vpa<=0 or a.temperature<=0 or a.mesh<1 or a.nbands<16:p.error('Invalid preparation arguments')
        prepare(a)
    else:plot(a)

if __name__=='__main__':main()

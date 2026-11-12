"""List final-model EOS snapshots around requested P,T; no phase certification."""
import argparse,csv,math
from pathlib import Path

def val(row,key):
    try:return float(row.get(key,''))
    except (ValueError,TypeError):return float('nan')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='runs/eos_live');p.add_argument('--pmin',type=float,default=250);p.add_argument('--pmax',type=float,default=450);p.add_argument('--temps',type=int,nargs='+',default=[5000,6000,7000]);a=p.parse_args()
    for d in sorted(Path(a.root).glob('ft_mfinkc_cont_*K_*at')):
        f=d/(d.name+'.csv')
        if not f.is_file():continue
        with f.open(newline='') as stream:
            for r in csv.DictReader(stream):
                T=val(r,'T_set');P=val(r,'P_GPa');V=val(r,'vpa')
                if not (T in a.temps and a.pmin<=P<=a.pmax and math.isfinite(V)):continue
                s=d/f'{d.name}_v{V:.4f}'
                print(f'{d.name} P={P:.1f} GPa Tmean={val(r,"T_mean"):.0f} K V={V:.4f} D_O={val(r,"D_O_cm2s"):.2g} O={r.get("O_struct","")} HHmin={val(r,"dmin_HH"):.3g} traj={"yes" if (s/"traj.extxyz").is_file() else "NO"} | {s}')
    print('Rows are screening observations; check pressure/temperature drift, close contacts, phase and frame independence before DFT optics.')

if __name__=='__main__':main()

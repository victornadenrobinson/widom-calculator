"""Compare production energy windows for completed mixture EOS table rows.

Uses explicit equilibration time; no inference of convergence or phase. Nuclear
kinetic plus model energy is not automatically the DFT internal energy.
"""
import argparse
import csv
import math
from pathlib import Path
import numpy as np


def finite(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (ValueError,TypeError):return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='runs/eos_live');p.add_argument('--out',required=True)
    p.add_argument('--equil-fs',type=float,required=True,help='equilibration length for these runs, fs; verify source logs')
    p.add_argument('--pmin',type=float,default=250);p.add_argument('--pmax',type=float,default=450)
    p.add_argument('--tmin',type=float,default=5000);p.add_argument('--tmax',type=float,default=7000)
    a=p.parse_args();root=Path(a.root);out=Path(a.out)
    if not root.is_dir():raise SystemExit('Missing root '+str(root))
    if out.exists():raise SystemExit('Refusing to overwrite '+str(out))
    records=[]
    for d in sorted(root.glob('ft_mfinkc_cont_*K_*at')):
        table=d/(d.name+'.csv')
        if not table.is_file():continue
        with table.open(newline='') as f: rows=list(csv.DictReader(f))
        for r in rows:
            P,T,v,N,nO=(finite(r.get(x)) for x in ('P_GPa','T_set','vpa','N','nO'))
            if None in (P,T,v,N,nO) or nO<=0 or not(a.pmin<=P<=a.pmax and a.tmin<=T<=a.tmax):continue
            run=d/(d.name+'_v%.4f'%v);thermo=run/'thermo.csv'
            if not thermo.is_file() or not (run/'final.extxyz').is_file():
                print('Missing thermo or final structure:',run);continue
            with thermo.open(newline='') as f: tr=list(csv.DictReader(f))
            vals=[]
            for trrow in tr:
                ts,ep,ek,temp,pres=(finite(trrow.get(k)) for k in ('t_fs','epot','ekin','T','P_GPa'))
                if None not in (ts,ep,ek,temp,pres) and ts>=a.equil_fs: vals.append((ts,ep/N,ek/N,temp,pres))
            if len(vals)<8:print('Too few valid production samples:',run);continue
            ar=np.array(vals);end=ar[:,0].max();start=a.equil_fs
            if end<=start:continue
            chunks=[]
            for i in range(4):
                lo=start+(end-start)*i/4;hi=start+(end-start)*(i+1)/4
                z=ar[(ar[:,0]>=lo)&((ar[:,0]<hi) if i<3 else (ar[:,0]<=hi))]
                if not len(z):break
                chunks.append(z)
            if len(chunks)<4:print('Empty production quarter:',run);continue
            last=chunks[-1]; mass=(N-nO)*1.008+nO*15.999
            rec=dict(tag=d.name,nH=int(N-nO),nO=int(nO),N=int(N),T_set_K=T,P_summary_GPa=P,
                vpa_A3=v,rho_g_cm3=mass/(N*v)*1.66053906660,production_start_fs=start,production_end_fs=end,
                samples_full=len(ar),samples_last_quarter=len(last),last_quarter_start_fs=start+.75*(end-start),
                P_last_quarter_GPa=float(last[:,4].mean()),T_last_quarter_K=float(last[:,3].mean()),
                epot_full_eV_atom=float(ar[:,1].mean()),ekin_full_eV_atom=float(ar[:,2].mean()),
                epot_last_quarter_eV_atom=float(last[:,1].mean()),ekin_last_quarter_eV_atom=float(last[:,2].mean()),
                model_energy_plus_nuclear_kinetic_last_eV_atom=float((last[:,1]+last[:,2]).mean()),
                epot_last_minus_first_quarter_meV_atom=float(1000*(last[:,1].mean()-chunks[0][:,1].mean())),
                source_thermo=str(thermo),note='reported; time windows not convergence or phase certification')
            records.append(rec)
            print(f"{d.name} P={P:.1f} GPa rho={rec['rho_g_cm3']:.3f} g/cm3 | last Epot={rec['epot_last_quarter_eV_atom']:.5f} eV/atom | last-first={rec['epot_last_minus_first_quarter_meV_atom']:+.1f} meV/atom")
    if not records:raise SystemExit('No matching completed mixture rows with usable thermo.csv; no file written.')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print(f'Wrote {len(records)} rows to {out}; equilibration time must match recorded runs.')

if __name__=='__main__':main()

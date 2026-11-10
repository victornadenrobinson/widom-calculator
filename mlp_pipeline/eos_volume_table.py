"""Inventory final-model EOS points and interpolate candidate TDI volumes (never extrapolate)."""
import argparse,csv,math
from pathlib import Path
import numpy as np

TARGETS=[(p,t) for t,pressures in ((7000,(100,200,350,500)),(5000,(100,200,350,500)),(3000,(100,200))) for p in pressures]

def usable(row,system,T):
    try:
        p=float(row['P_GPa']);v=float(row['vpa']);temp=float(row['T_mean']);hh=float(row['dmin_HH'])
        if not all(math.isfinite(x) for x in (p,v,temp,hh)) or p<=0 or v<=0 or abs(temp-T)>.05*T or hh<.4:return None
        if system!='h':
            D=float(row['D_O_cm2s'])
            if not math.isfinite(D) or D<2e-5 or row.get('O_struct','') in ('bcc','fcc','hcp'):return None
        drift=float(row.get('P_drift_GPa') or 'nan');sd=float(row.get('P_block_sd') or 'nan')
        if math.isfinite(drift) and math.isfinite(sd) and abs(drift)>max(.01*abs(p),3*sd):return None
        return p,v
    except (KeyError,ValueError,TypeError):return None


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',nargs='?',default='runs/eos_live');ap.add_argument('--model',default='ft_mfinkc_cont');ap.add_argument('--out',required=True);a=ap.parse_args()
    rows=[]
    for system in ('h2o','h3o','h'):
        for T in (7000,5000,3000):
            dirs=sorted(Path(a.root).glob(f'{a.model}_{system}_{T}K_*at'))
            groups=[]
            for d in dirs:
                path=d/f'{d.name}.csv'
                if not path.is_file():continue
                with path.open(newline='') as f:points=[z for r in csv.DictReader(f) if (z:=usable(r,system,T)) is not None]
                if points:groups.append((d,sorted(points)))
            if len(groups)>1:
                print(f'{system} {T}: multiple eligible runs: {[str(g[0]) for g in groups]}; choose one before interpolation');continue
            if not groups:print(f'{system} {T}: no usable EOS rows');continue
            d,pts=groups[0];p=np.array([v[0] for v in pts]);v=np.array([v[1] for v in pts])
            print(f'{system} {T}: {len(pts)} candidate liquid rows, P range {min(p):.1f}–{max(p):.1f} GPa, source {d}')
            for target,t in TARGETS:
                if t!=T or not min(p)<=target<=max(p) or len(pts)<2:continue
                i=np.searchsorted(p,target);lo=max(0,i-1);hi=min(len(p)-1,i)
                if lo==hi:continue
                if v[hi]>=v[lo]:print(f'  {target} GPa: nonmonotone bracketing EOS; skipped');continue
                vpa=float(np.interp(math.log(target),np.log(p[[lo,hi]]),v[[lo,hi]]))
                rows.append(dict(system=system,P_GPa=target,T_K=T,vpa_A3=f'{vpa:.8f}',source=str(d),bracket_GPa=f'{p[lo]:.3f}..{p[hi]:.3f}'))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['system','P_GPa','T_K','vpa_A3','source','bracket_GPa']);w.writeheader();w.writerows(rows)
    print(f'Wrote {len(rows)} candidate volumes to {out}; inspect source EOS/phase/pressure before using. No extrapolation.')

if __name__=='__main__':main()

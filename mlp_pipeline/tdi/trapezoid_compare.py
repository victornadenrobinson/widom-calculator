"""Compare GK integration with a trapezoid estimate in x=lambda**(1/p).

Estimates use identical saved samples. The transformed integrand is forced to
zero at x=0 for p>1 and linearly extrapolated from the last two nodes to x=1.
Endpoint extrapolation and quadrature error are not included in block errors.
"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tdi import analyze
from quad_diagnostics import inspect


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--out', default=None)
    args=ap.parse_args()
    runs=[inspect(d) for d in args.dirs]
    identities={(r['st']['T'],r['st']['P_target'],r['st']['N'],r['A']['model'],r['A'].get('hper',2),r['st']['V']) for r in runs}
    if len(identities)!=1:
        raise SystemExit('Incompatible recorded model path, P, T, N, composition or volume.')
    cutoff=min(len(s)/r['spp'] for r in runs for s in r['series'])
    rows=[]
    for r in runs:
        power=float(r['A'].get('power',4))
        if power<=1: raise SystemExit('Endpoint rule requires power > 1.')
        n=int(round(cutoff*r['spp'])); scale=1000/r['st']['N']
        x=np.asarray(r['lam'])**(1/power)
        means=np.array([np.mean(s[:n]) for s in r['series']])*scale
        transformed=means*power*x**(power-1)
        order=np.argsort(x);x=x[order];transformed=transformed[order]
        endpoint=transformed[-1]+(1-x[-1])*(transformed[-1]-transformed[-2])/(x[-1]-x[-2])
        trap=float(np.trapz(np.r_[0,transformed,endpoint],np.r_[0,x,1]))
        gk=float(sum(r['w']*means))
        rows.append(dict(run=r['dir'],rule=f"{r['A'].get('rule','gl').upper()}{len(x)}",common_ps=cutoff,
                         quadrature_meV_atom=gk,trapezoid_meV_atom=trap,trap_minus_rule_meV_atom=trap-gk,
                         endpoint_x1_meV_atom=endpoint))
        print(f"{r['dir']} | {rows[-1]['rule']}: integral {gk:.3f}; trapezoid {trap:.3f}; difference {trap-gk:+.3f} meV/atom; extrapolated x=1 integrand {endpoint:.3f}")
    print(f'Common first {cutoff:g} ps/node; checkpoint lambda/model/series verified.')
    print('Trapezoid integrates x=lambda^(1/p), including its Jacobian; x=0 set to 0, x=1 linearly extrapolated from the two largest x nodes.')
    print('The values use the same correlated samples; neither differences nor endpoint extrapolation are uncertainty estimates.')
    if args.out:
        p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print('Wrote',p)

if __name__=='__main__':main()

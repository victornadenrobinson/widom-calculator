"""CPU-only diagnostic dashboard from saved TDI samples and compact thermo CSVs.
Example: python mlp_pipeline/tdi/hull_dashboard.py runs/tdi_final --P 500 --T 5000 --out runs/plots/tdi_final
No trajectory loading or model evaluations. Hulls are provisional, not phase-certified.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tdi import analyze
from tdi.gcalc import G_from_nodes


def ideal(x, T):
    # Atomic O/H compositional entropy, relative to H2O and H endmembers.
    # x is the fraction of all atoms contributed by added H, not H2O/H formula units.
    def entropy(z):
        z = np.asarray(z)
        return np.where(z > 0, z * np.log(np.maximum(z, 1e-300)), 0)
    o = (1 - np.asarray(x)) / 3
    return .08617333 * T * (entropy(o) + entropy(1-o) - (1-x)*(entropy(1/3)+entropy(2/3)))


def lower_hull(x, y):
    ids = []
    for i in np.argsort(x):
        while len(ids) >= 2:
            a, b = ids[-2:]
            if (x[b]-x[a])*(y[i]-y[b]) - (y[b]-y[a])*(x[i]-x[b]) > 0:
                break
            ids.pop()
        ids.append(i)
    return ids


def load(path):
    st = json.loads(path.read_text()); a = st['args']
    lam, w, sub = analyze.rule_nodes(a)
    series = []
    for k in range(len(lam)):
        f = path.parent / f'{path.name[:-12]}_node{k}.npy'
        series.append(np.load(f) if f.exists() else np.array([]))
        state = path.parent / f'{path.name[:-12]}_node{k}_state.npz'
        if state.exists():
            with np.load(state) as z:
                for key, expected in [('lam',lam[k]),('T',st['T']),('dt',a['dt'])]:
                    if key in z and not np.isclose(z[key].item(), expected):
                        raise ValueError(f'{state}: {key} inconsistent with stage0')
                if 'model' in z and z['model'].item() != a['model']:
                    raise ValueError(f'{state}: inconsistent model')
    n = min(map(len, series)); spp = 1000/a['dt']/a['every']
    good = n >= 25 and all(np.isfinite(s).all() for s in series)
    value = G_from_nodes(st,lam,w,[s[:n] for s in series],st['N']) if good else None
    q = None
    if value and sub is not None:
        q = value[0]-G_from_nodes(st,*sub[:2],[s[:n] for s in series[:len(sub[0])]],st['N'])[0]
    return dict(path=path,tag=path.name[:-12],st=st,a=a,lam=lam,w=w,series=series,n=n,spp=spp,value=value,q=q,
                complete=good and n/spp >= a['ps']-1e-8)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dirs',nargs='+'); ap.add_argument('--P',type=float,required=True);ap.add_argument('--T',type=float,required=True)
    ap.add_argument('--out',required=True);args=ap.parse_args()
    runs=[]
    for directory in args.dirs:
        for p in sorted(Path(directory).glob('*_stage0.json')):
            st=json.loads(p.read_text())
            if st['phase'] not in ('liquid','H') or not np.isclose(st['T'],args.T) or not np.isclose(st['P_target'],args.P): continue
            runs.append(load(p))
    if not runs: raise SystemExit('No matching liquid / H runs.')
    models={r['a']['model'] for r in runs}
    if len(models)!=1: raise SystemExit(f'Multiple model specs: {models}. Select separate input directories.')
    runs.sort(key=lambda r: r['a'].get('hper',2) if r['st']['phase']=='liquid' else np.inf)
    comps=[r['a'].get('hper',2) if r['st']['phase']=='liquid' else np.inf for r in runs]
    if len(set(comps))!=len(comps): raise SystemExit('Duplicate compositions: select directories containing one run per composition.')
    nc=max(2,len(runs));fig=plt.figure(figsize=(max(10,3.5*nc),12),layout='constrained')
    grid=fig.add_gridspec(4,nc,height_ratios=[1.2,1,1,1])
    hx=fig.add_subplot(grid[0,:nc//2]);ex=fig.add_subplot(grid[0,nc//2:])
    water=next((r for r in runs if r['st']['phase']=='liquid' and r['a'].get('hper',2)==2),None)
    hydrogen=next((r for r in runs if r['st']['phase']=='H'),None)
    points=[];summary=[]
    for r in runs:
        st,a,v=r['st'],r['a'],r['value'];x=1. if st['phase']=='H' else (a.get('hper',2)-2)/(a.get('hper',2)+1)
        row=dict(tag=r['tag'],model=a['model'],N=st['N'],x_added_H_atom=x,ps_common=r['n']/r['spp'],complete=r['complete'],G_meV_atom=v[0] if v else None,stat_sem=v[1] if v else None,quad_diff=r['q'],mix_meV_atom=None,mix_stat_sem=None)
        if r['complete'] and water and hydrogen and water['complete'] and hydrogen['complete']:
            grids=[(z['lam'],z['w']) for z in (r,water,hydrogen)]
            if any(len(l)!=len(grids[0][0]) or not np.allclose(l,grids[0][0]) or not np.allclose(w,grids[0][1]) for l,w in grids):
                raise SystemExit('Mixing requires matching quadrature grids in this dashboard.')
            y=v[0]-(1-x)*water['value'][0]-x*hydrogen['value'][0]
            e=0. if r is water or r is hydrogen else np.sqrt(v[1]**2+(1-x)**2*water['value'][1]**2+x*x*hydrogen['value'][1]**2)
            row.update(mix_meV_atom=y,mix_stat_sem=e);points.append((x,y,e))
        summary.append(row)
    if points:
        points.sort();x,y,e=np.array(points).T;ids=lower_hull(x,y)
        hx.errorbar(x,y,yerr=e,fmt='o',label='TDI ± statistical SEM');hx.plot(x[ids],y[ids],label='sampled lower hull')
        xx=np.linspace(0,1,301);hx.plot(xx,ideal(xx,args.T),'--',label='ideal atomic entropy reference')
        ex.errorbar(x,y-ideal(x,args.T),yerr=e,fmt='o');ex.axhline(0,color='gray',lw=.7)
    else:
        hx.text(.5,.5,'Hull needs completed liquid H2O and H references',ha='center',wrap=True,transform=hx.transAxes)
    hx.set(title='Provisional liquid mixing hull',xlabel='Added-H atom fraction',ylabel='ΔG mixing (meV/atom)');hx.legend(fontsize=7) if points else None
    ex.set(title='Excess relative to ideal atomic entropy reference',xlabel='Added-H atom fraction',ylabel='ΔG − reference (meV/atom)')
    for j,r in enumerate(runs):
        st,a=r['st'],r['a'];title='H' if st['phase']=='H' else f'H{a.get("hper",2):g}O'
        ax=fig.add_subplot(grid[1,j]);order=np.argsort(r['lam'])
        means=np.array([np.mean(s) if len(s) else np.nan for s in r['series']])*1000/st['N']
        err=np.array([analyze.block_stats(s,5)[1] if len(s)>=25 else np.nan for s in r['series']])*1000/st['N']
        ax.errorbar(r['lam'][order],means[order],yerr=err[order],fmt='o-');ax.set_xscale('log')
        ax.set(title=f'{title}, N={st["N"]}: integrand',xlabel='λ (log)',ylabel=r'$\langle \partial U/\partial\lambda \rangle$ (meV/atom)')
        ax=fig.add_subplot(grid[2,j]);ns=np.unique(np.linspace(25,r['n'],30).astype(int)) if r['n']>=25 else []
        vals=[G_from_nodes(st,r['lam'],r['w'],[s[:n] for s in r['series']],st['N'])[:2] for n in ns]
        if vals:
            g,e=np.array(vals).T;t=np.array(ns)/r['spp'];ax.plot(t,g);ax.fill_between(t,g-e,g+e,alpha=.2)
        quad = f'\nK−G {r["q"]:.2f} meV/atom' if r['q'] is not None else '\nquadrature error unmeasured'
        ax.set(title=f'{title}: cumulative G'+quad,xlabel='Production ps / lambda',ylabel='meV/atom')
        ax=fig.add_subplot(grid[3,j]);ax2=ax.twinx();k=int(np.argmax(r['lam']));f=r['path'].parent/f'{r["tag"]}_node{k}_thermo.csv'
        if not f.exists(): ax.text(.5,.5,'No thermo CSV',ha='center',transform=ax.transAxes)
        if f.exists():
            with f.open() as fh: rows=list(csv.DictReader(fh))
            if rows:
                t=[float(z['t_ps']) for z in rows];ax.plot(t,[float(z['T_inst_K']) for z in rows],color='C0');ax2.plot(t,[float(z['P_MLP_GPa']) for z in rows],color='C3')
        ax.axhline(args.T,color='C0',ls=':',lw=.7)
        ax.set(title=f'{title}: λ={r["lam"][k]:.4g}',xlabel='Production ps',ylabel='Temperature (K)');ax2.set_ylabel('Full-MLP pressure (GPa)',color='C3')
    fig.suptitle(f'{args.P:g} GPa / {args.T:g} K — {next(iter(models))}\nStatistical diagnostics only; phase, finite-size and model errors not certified. High-λ thermo is not necessarily λ=1.',fontsize=11)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(str(out)+'.'+ext,dpi=160)
    with Path(str(out)+'.csv').open('w') as fh:
        wr=csv.DictWriter(fh,fieldnames=list(summary[0]));wr.writeheader();wr.writerows(summary)
    print(f'Wrote {out}.png / .pdf / .csv (no trajectory reads or model calls).')

if __name__=='__main__':main()

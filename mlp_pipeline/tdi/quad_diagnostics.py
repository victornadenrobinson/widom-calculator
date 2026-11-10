"""Inspect saved per-lambda TDI data and embedded Gauss–Kronrod differences.
Usage: python mlp_pipeline/tdi/quad_diagnostics.py RUN_DIR [RUN_DIR ...]
Computes descriptive first/second-half diagnostics and five-block errors; no model calls.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tdi import analyze
from tdi.gcalc import G_from_nodes


def inspect(directory):
    paths=sorted(Path(directory).glob('*_stage0.json'))
    if len(paths)!=1:
        raise ValueError(f'{directory}: expected one stage0 JSON, found {len(paths)}')
    path=paths[0];st=json.loads(path.read_text());A=st['args'];tag=path.name[:-12]
    lam,w,sub=analyze.rule_nodes(A);spp=1000/A['dt']/A['every'];series=[]
    for k,l in enumerate(lam):
        f=path.parent/f'{tag}_node{k}.npy';state=path.parent/f'{tag}_node{k}_state.npz'
        if not f.exists():raise ValueError(f'missing {f}')
        x=np.load(f)
        if len(x)<50 or not np.isfinite(x).all():raise ValueError(f'{f}: fewer than 50 or non-finite samples')
        if not state.exists():raise ValueError(f'missing checkpoint {state}')
        with np.load(state) as z:
            if 'lam' not in z or not np.isclose(z['lam'].item(),l,rtol=1e-12,atol=0):raise ValueError(f'{state}: lambda mismatch')
            if 'model' not in z or z['model'].item()!=A['model']:raise ValueError(f'{state}: model mismatch')
            if not np.array_equal(z['series'],x):raise ValueError(f'{state}: checkpoint and series differ')
        series.append(x)
    return dict(dir=str(path.parent),tag=tag,st=st,A=A,lam=lam,w=w,sub=sub,spp=spp,series=series)


def calc(r, cutoff_ps, discard):
    n=int(round(cutoff_ps*r['spp']));lo=int(n*discard)
    samples=[s[lo:n] for s in r['series']]
    if min(map(len,samples))<25:raise ValueError('Too few samples after discard')
    return G_from_nodes(r['st'],r['lam'],r['w'],samples,r['st']['N']),samples


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('dirs',nargs='+');ap.add_argument('--out',default=None,help='optional CSV of per-lambda diagnostics');a=ap.parse_args()
    runs=[inspect(d) for d in a.dirs]
    ids={(r['st']['T'],r['st']['P_target'],r['st']['N'],r['A']['model'],r['A'].get('hper',2),r['st']['V']) for r in runs}
    if len(ids)!=1:raise SystemExit('Incompatible run metadata: select same P, T, model path, composition, volume and size.')
    cutoff=min(len(s)/r['spp'] for r in runs for s in r['series'])
    rows=[]
    print(f'Common cutoff: {cutoff:.4f} ps/lambda. Energies are meV/atom; error bars are five-block SEM, not total uncertainty.')
    print('Checkpoint lambda/model/series checked; same model path does not prove identical weights.\n')
    for r in runs:
        (full,ef,_,_,_),samples=calc(r,cutoff,0);(late,el,_,_,_),_=calc(r,cutoff,.5)
        A=r['A'];print(f"{r['dir']} | {A.get('rule','gl').upper()}{len(r['lam'])} | all {full:.2f} ± {ef:.2f}; last half {late:.2f} ± {el:.2f}; late − all {late-full:+.2f}")
        if r['sub']:
            lg,wg=r['sub'];g,eg,*_=G_from_nodes(r['st'],lg,wg,samples[:len(lg)],r['st']['N'])
            embedded=np.r_[wg,np.zeros(len(r['w'])-len(wg))]
            sem=np.array([analyze.block_stats(s,5)[1] for s in samples])
            diff_sem=np.sqrt(np.sum(((r['w']-embedded)*sem)**2))*1000/r['st']['N']
            print(f'  Embedded GL{len(lg)}: {g:.2f} ± {eg:.2f}; GK−GL = {full-g:+.2f} ± {diff_sem:.2f} (shared-sample SEM; assumes independent lambdas)')
        print('  k   lambda        weight     mean       first half  second half   weighted drift  (all energies meV/atom)')
        for k,(l,w,s) in enumerate(zip(r['lam'],r['w'],samples)):
            n=len(s)//2;scale=1000/r['st']['N'];m=s.mean()*scale;first=s[:n].mean()*scale;second=s[n:].mean()*scale
            drift=w*(second-first)
            print(f'  {k:2d}  {l:11.4g}  {w:10.4g}  {m:10.2f}  {first:10.2f}  {second:10.2f}   {drift:+10.2f}')
            rows.append(dict(run=r['dir'],k=k,lambda_value=l,weight=w,mean_meV_atom=m,first_half_meV_atom=first,second_half_meV_atom=second,weighted_second_minus_first_meV_atom=drift))
        print()
    if a.out:
        import csv
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
        print('Wrote',path)

if __name__=='__main__':main()

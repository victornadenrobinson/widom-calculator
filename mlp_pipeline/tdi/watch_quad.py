"""Read-only progress viewer for an already running Young quadrature sweep.
Works on a login node; shows per-lambda status from progress files, not GPU ownership.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from quad_eta import estimate, format_eta


def snapshot(root, now=None):
    now=time.time() if now is None else now
    lines=[f'{time.strftime("%H:%M:%S")} | TDI quadrature viewer: {root}',
           'GPU assignments unavailable in saved progress; worker IDs below are family and lambda.']
    dirs=sorted(root.glob('gl*_gk*'))
    if not dirs:return lines+['No family directories found.']
    seconds,reason=estimate(root,now=now)
    lines.append(format_eta(seconds,reason))
    for d in dirs:
        files=list(d.glob('*_stage0.json'))
        if len(files)!=1:
            lines.append(f'{d.name}: waiting for stage0 metadata');continue
        try:
            st=json.loads(files[0].read_text());a=st['args'];tag=files[0].name[:-12];nodes=a['nodes'];rule=a.get('rule') or 'gl'
            target=float(a['ps']); per_ps=1000/float(a['dt'])/int(a['every']);parts=[];done=0
            for k in range(nodes):
                pf=d/f'{tag}_node{k}_progress.json';nf=d/f'{tag}_node{k}.npy'
                try:
                    p=json.loads(pf.read_text());age=now-p['t'];fraction=min(1.,p['done']/p['total']) if p['total'] else 0.
                    rate=float(p.get('rate',float('nan')))
                    if p.get('state')=='done' and nf.exists() and len(np.load(nf,mmap_mode='r'))/per_ps>=target-1e-8:
                        done+=1;continue
                    if age>180:
                        state=f'stale {age/60:.0f}m'
                    elif np.isfinite(rate) and rate>0:
                        state=f'{100*fraction:.0f}%, {rate:.1f} steps/s, {1000/rate:.0f} ms/step'
                    else:state=f'{100*fraction:.0f}%, starting'
                    parts.append(f'{rule.upper()}{nodes} λ{k}: {state}')
                except (OSError,ValueError,KeyError,TypeError,ZeroDivisionError):
                    if nf.exists() and len(np.load(nf,mmap_mode='r'))/per_ps>=target-1e-8:done+=1
                    else:parts.append(f'{rule.upper()}{nodes} λ{k}: waiting or loading')
            lines.append(f'{d.name}: {rule.upper()}{nodes} | {done}/{nodes} complete | target {target:g} ps')
            lines.extend('  '+line for line in parts)
        except (ValueError,OSError,KeyError) as e:lines.append(f'{d.name}: metadata unavailable ({e})')
    return lines


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');p.add_argument('--interval',type=float,default=5);p.add_argument('--once',action='store_true');a=p.parse_args()
    if a.interval<=0:p.error('--interval must be positive')
    previous=0
    try:
        while True:
            lines=snapshot(Path(a.root))
            if previous and sys.stdout.isatty():sys.stdout.write(f'\033[{previous}A\033[J')
            print('\n'.join(lines),flush=True);previous=len(lines)
            if a.once:return
            time.sleep(a.interval)
    except KeyboardInterrupt:print('\nViewer stopped; running jobs are unaffected.')

if __name__=='__main__':main()

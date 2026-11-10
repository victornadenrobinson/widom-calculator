"""Estimate a whole quadrature campaign from its saved progress (read-only).
Assumes recent aggregate throughput persists; excludes startup, contention changes,
queue dependencies, final tails and convergence extensions. Do not treat as a guarantee.
"""
import math
import re
import time
from pathlib import Path
import json
import numpy as np


def estimate(root, now=None, directories=None, min_workers=2):
    now=time.time() if now is None else now
    directories=sorted(Path(root).glob('gl*_gk*')) if directories is None else list(directories)
    if not directories:return None, 'no families found'
    remaining=0.;rates=[];pending_metadata=[];stale=0;failures=[]
    for d in directories:
        match=re.fullmatch(r'gl(\d+)_gk(\d+)',d.name)
        if not match:continue
        files=list(d.glob('*_stage0.json'))
        if len(files)!=1:
            pending_metadata.append(d.name);continue
        try:
            st=json.loads(files[0].read_text());a=st['args'];tag=files[0].name[:-12]
            final_nodes=int(match[2]);nodes=int(a['nodes']);dt=float(a['dt']);every=int(a['every'])
            target=float(a['ps']);eq=float(a['equil_ps']);hot=float(a.get('hot_ps') or 0.)
            prod_steps=target*1000/dt;eq_steps=eq*1000/dt;hot_steps=hot*1000/dt
            for k in range(final_nodes):
                if k>=nodes:
                    remaining+=prod_steps+eq_steps+hot_steps;continue
                pfile=d/f'{tag}_node{k}_progress.json';f=d/f'{tag}_node{k}.npy'
                saved=0
                if f.exists():
                    try:saved=len(np.load(f,mmap_mode='r'))*every
                    except (ValueError,OSError):pass
                if saved>=prod_steps-1e-8:continue
                p={}
                try:p=json.loads(pfile.read_text())
                except (OSError,ValueError):pass
                age=now-p.get('t',0)
                if p.get('state')=='failed':failures.append(f'{d.name} λ{k}')
                if p.get('state')=='running' and age<=120 and p.get('total',0)>0:
                    phase=p.get('phase','MD')
                    if phase == 'heating':
                        progress=min(hot_steps,max(0,float(p.get('done',0))))
                        remaining+=max(0,hot_steps-progress)+eq_steps+prod_steps
                        phase_total=max(hot_steps,1)
                    else:
                        progress=min(eq_steps+prod_steps,max(0,float(p.get('done',0))))
                        remaining+=max(0,eq_steps+prod_steps-progress)+(hot_steps if progress==0 and not saved and phase=='MD' else 0)
                        phase_total=eq_steps+prod_steps
                    rate=float(p.get('rate',float('nan')))
                    if math.isfinite(rate) and rate>0 and now-p.get('t_start',now)>=120 and progress>=.05*phase_total:rates.append(rate)
                else:
                    if p.get('state')=='running' and age>120:stale+=1
                    remaining+=max(0,prod_steps-saved)+(0 if saved else eq_steps+hot_steps)
        except (OSError,ValueError,KeyError,TypeError,ZeroDivisionError) as e:
            return None,f'cannot read {d.name}: {e}'
    if pending_metadata:return None,'waiting for metadata: '+', '.join(pending_metadata)
    if failures:return None,'failed workers: '+', '.join(failures)
    if remaining<=1e-6:return 0.,'all planned MD samples recorded; final collection may remain'
    if stale:return None,f'{stale} stale running workers; check logs before trusting an ETA'
    if len(rates)<min_workers:return None,f'waiting for {min_workers} warmed worker rate(s) (≥2 min, ≥5% of phase)'
    speed=sum(rates)
    return remaining/speed, f'{len(rates)} warmed workers, {speed:.1f} steps/s; assumes current aggregate throughput'


def format_eta(seconds,reason,label='Campaign ETA'):
    if seconds is None:return label+': unavailable — '+reason
    if seconds==0:return label+': MD complete — '+reason
    mins=seconds/60
    duration=f'{mins:.0f} min' if mins<120 else f'{mins/60:.1f} h'
    return f'{label}: ~{duration} (provisional; {reason}; startup/tail not included)'

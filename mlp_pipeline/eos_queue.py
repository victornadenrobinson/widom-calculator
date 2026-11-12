"""Run successive EOS temperature/composition groups with a visible campaign queue.

Preview by default; --run executes within the current GPU allocation. Completed
volume rows are skipped by eos_live; interrupted points restart normally.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import numpy as np
from eos_volume_table import usable
ROOT=Path(os.environ.get('EOS_REPO_ROOT', str(Path(__file__).resolve().parents[1]))).resolve()
CODE_DIR=Path(os.environ.get('EOS_CODE_DIR', str(ROOT/'mlp_pipeline'))).resolve()
COMPS=('h2o','h2.25','h3o','h4o','h5o','h7o','h')


def resolve(group, model):
    """Explicit pressures require same-temperature liquid EOS brackets."""
    if 'pressures_GPa' not in group: return group.get('volumes',{})
    grids={}
    for c in group['compositions']:
        pts=[]
        hper={'h2o':2,'h2.25':2.25,'h3o':3,'h4o':4,'h5o':5,'h7o':7}.get(c)
        nrep=min((3,4),key=lambda n:abs(2*n**3*(1+hper)-384)) if hper else 0
        expected_N=round(2*nrep**3*(1+hper)) if hper else 729
        for d in (ROOT/'runs/eos_live').glob(f'{model}_{c}_{group["T"]}K_*at'):
            f=d/(d.name+'.csv')
            if not f.exists():continue
            with f.open() as stream:
                pts.extend(p for row in csv.DictReader(stream) if row.get('N') and float(row['N'])==expected_N and (p:=usable(row,c,group['T'])) is not None)
        pts=sorted(pts)
        if len(pts)<2:raise ValueError(f'{c} {group["T"]} K: need liquid EOS brackets first')
        grids[c]=[]
        for P in group['pressures_GPa']:
            if P<pts[0][0] or P>pts[-1][0]:raise ValueError(f'{c}: {P} GPa outside screened EOS range')
            k=max(1,min(len(pts)-1,int(np.searchsorted([p for p,v in pts],P))))
            lo,hi=pts[k-1],pts[k]
            if hi[1]>=lo[1] or hi[0]<=lo[0]:raise ValueError(f'{c}: nonmonotone pressure bracket')
            grids[c].append(round(float(np.interp(math.log(P),np.log([lo[0],hi[0]]),[lo[1],hi[1]])),4))
    return grids


def estimate(group, model, slots):
    # Historical per-worker timings include GPU sharing and analysis overhead.
    costs=[]
    for c in group['compositions']:
        times=[]
        for d in (ROOT/'runs/eos_live').glob(f'{model}_{c}_*K_*at'):
            f=d/(d.name+'.csv')
            if not f.exists():continue
            with f.open() as stream:
                for row in csv.DictReader(stream):
                    try:
                        t=float(row['ms_per_step'])
                        if math.isfinite(t) and t>0:times.append(t)
                    except (KeyError,ValueError):pass
        if not times:return None
        n=len(group.get('volumes',{}).get(c,group.get('pressures_GPa',[]))) or group.get('nv',16)
        costs.extend([np.median(times)/1000 * (2000 * (group.get('ps', .75 if c!='h' else .5) + group.get('eq', .25 if c!='h' else .2) + (.25 if c!='h' else 0)))] * n)
    return float(max(sum(costs)/slots,max(costs,default=0)))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('plan');ap.add_argument('--run',action='store_true')
    ap.add_argument('--gpus',type=int,default=None);ap.add_argument('--per-gpu',type=int,default=4)
    ap.add_argument('--model',default='grace-ft:runs/grace/ft_mfinkc_cont');a=ap.parse_args()
    os.chdir(ROOT);groups=json.loads(Path(a.plan).read_text())
    if not isinstance(groups,list) or not groups:ap.error('Plan must be a nonempty list')
    for g in groups:
        if not isinstance(g.get('T'),int) or g['T']<=0 or not g.get('compositions') or set(g['compositions'])-set(COMPS):ap.error('Each group needs positive integer T and supported compositions')
        if any(not isinstance(g[k], (int, float)) or not math.isfinite(g[k]) or g[k] <= 0 for k in ('ps', 'eq') if k in g):ap.error('ps and eq must be positive finite numbers')
        if 'volumes' in g and 'pressures_GPa' in g:ap.error('Choose volumes or pressures_GPa, not both')
        if 'pressures_GPa' in g and (not g['pressures_GPa'] or any(not math.isfinite(p) or p<=0 for p in g['pressures_GPa'])):ap.error('Pressures must be positive and finite')
    tokens=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    ng=a.gpus or (len(tokens) if tokens!=[''] else 1)
    if ng<1 or a.per_gpu<1:ap.error('Positive GPU and worker counts required')
    name=Path(a.model.split(':',1)[-1]).name
    schedule=[dict(T=g['T'],comp=' '.join(g['compositions']),
        pressure=str(g.get('pressures_GPa','explicit V' if 'volumes' in g else 'volume scan')),
        state='waiting',seconds=estimate(g,name,ng*a.per_gpu)) for g in groups]
    for r in schedule:print(r)
    print('Estimates are full-group historical costs, not deadlines; saved points shorten reruns. Startup/tail and changed GPU load add uncertainty.')
    if not a.run: print('Preview only; add --run in a GPU allocation.');return
    child=None
    def stop(*_):
        if child and child.poll() is None:child.send_signal(signal.SIGTERM);child.wait()
        raise SystemExit('Stopped; completed rows retained. Rerun to continue.')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for idx,g in enumerate(groups):
        locks=[]
        try:
            # Same locks as the existing batch EOS campaign, preventing concurrent writers.
            lockdir=ROOT/'runs/eos_locks';lockdir.mkdir(parents=True,exist_ok=True)
            for c in sorted(set(g['compositions'])):
                f=(lockdir/f'{name}_{c}_{g["T"]}.lock').open('a');locks.append(f)
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            volumes=resolve(g,name);schedule[idx]['state']='running'
            env=os.environ.copy()
            for key in ('VMAX','VMIN','PS','EQ','NREP','NATOMS','GRACE_MODE','HHWALL','GRACE_UQ'):env.pop(key,None)
            env.update(PY=sys.executable,MODEL=a.model,T=str(g['T']),COMP=' '.join(g['compositions']),
                NGPU=str(ng),PER_GPU=str(a.per_gpu),NV=str(g.get('nv',16)),VPA_JSON=json.dumps(volumes),
                NEARBY_START='1',PLOT='braille',REFRESH='10',OMP_NUM_THREADS='1',EOS_QUEUE=json.dumps(schedule),
                EOS_REPO_ROOT=str(ROOT),EOS_CODE_DIR=str(CODE_DIR))
            for key in ('ps', 'eq'):
                if key in g: env[key.upper()]=str(g[key])
            child=subprocess.Popen(['bash',str(CODE_DIR/'eos_live.sh')],env=env)
            rc=child.wait();schedule[idx]['state']='done' if rc==0 else 'failed'
            if rc:raise SystemExit('Group failed; inspect its logs before resuming the queue.')
        except BlockingIOError:
            raise SystemExit(f'{g["T"]} K group overlaps an active batch/queue job; no duplicate writer launched.')
        except ValueError as e:raise SystemExit(str(e))
        finally:
            for f in locks:f.close()
    print('EOS queue complete.')

if __name__=='__main__':main()

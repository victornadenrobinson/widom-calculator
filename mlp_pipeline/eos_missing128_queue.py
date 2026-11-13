"""Run planned missing H–O EOS volumes as one refillable 128-oxygen queue.

Locks planned composition/temperature states against the original queue. Preview by default.
Temperature priority is the order in the plan, then smallest volume first.
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

ROOT = Path(os.environ.get('EOS_REPO_ROOT', Path(__file__).resolve().parents[1])).resolve()
CODE = Path(os.environ.get('EOS_CODE_DIR', ROOT / 'mlp_pipeline')).resolve()
HPER={'h2o':2.,'h2.25':2.25,'h3o':3.,'h5o':5.,'h7o':7.}


def saved(path):
    if not path.exists():
        return set()
    # Other workers may append the same EOS table concurrently.
    with Path(str(path)+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH)
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or 'vpa' not in reader.fieldnames:
                raise ValueError(f'{path}: missing vpa header; inspect table')
            result = set()
            for row in reader:
                try:
                    result.add(round(float(row['vpa']),4))
                except (TypeError,ValueError,KeyError) as exc:
                    raise ValueError(f'{path}: invalid row; inspect table') from exc
        fcntl.flock(lock,fcntl.LOCK_UN)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('plan')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--gpus', type=int, default=4)
    ap.add_argument('--per-gpu', type=int, default=4, help='scheduling slots per GPU')
    ap.add_argument('--h7-slots', type=int, default=2, help='slots occupied by each H7O job')
    ap.add_argument('--model', default='grace-ft:runs/grace/ft_mfinkc_cont')
    ap.add_argument('--refresh', type=float, default=10)
    ap.add_argument('--max-failures',type=int,default=3,help='stop launching after this many failures; running jobs finish')
    ap.add_argument('--skip',action='append',default=[],metavar='COMP:T:VPA',help='skip this volume for this invocation; can be repeated')
    a = ap.parse_args()
    if a.gpus < 1 or a.per_gpu < 1 or not 1 <= a.h7_slots <= a.per_gpu or a.refresh <= 0 or a.max_failures < 1:
        ap.error('GPU counts and refresh must be positive')
    skipped_keys=set()
    for value in a.skip:
        try:
            c,t,v=value.split(':');t=int(t);v=float(v)
            if c not in HPER or t<=0 or not np.isfinite(v) or v<=0:raise ValueError()
            skipped_keys.add((c,t,round(v,4)))
        except ValueError:ap.error('--skip must be COMP:T:VPA, e.g. h3o:4000:1.7365')
    os.chdir(ROOT)
    groups = []
    for g in json.loads(Path(a.plan).read_text()):
        comps=[c for c in g.get('compositions',[]) if c in HPER]
        if comps:
            if g.get('nrep') != 4 or 'pressures_GPa' in g:
                raise SystemExit('Expected nrep=4 and volume grids, not pressure targets')
            groups.append(dict(g,compositions=comps))
    pairs=[(g['T'],c) for g in groups for c in g['compositions']]
    if not groups or len(set(pairs))!=len(pairs):
        raise SystemExit('Missing states or duplicate composition/temperature pairs in plan')
    ids = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if os.environ.get('CUDA_VISIBLE_DEVICES') else list(map(str, range(a.gpus)))
    if len(ids) < a.gpus:
        raise SystemExit('Fewer visible GPU IDs than requested')
    name = Path(a.model.split(':',1)[-1].rstrip('/')).name
    locks=[]; running={}; failed=[]; finished=0
    if a.run:
        if not os.environ.get('SLURM_JOB_ID'):
            raise SystemExit('Run inside a Slurm GPU allocation')
        lockdir=ROOT/'runs/eos_locks';lockdir.mkdir(parents=True,exist_ok=True)
        try:
            for temp, comp in sorted(pairs):
                f=(lockdir/f'{name}_{comp}_{temp}.lock').open('a');locks.append(f)
                fcntl.flock(f,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A planned composition/temperature lock is held: stop or finish the overlapping queue first.')
    tasks=[]; completed=0; state_specs=[]; skipped=[]; matched_skips=set()
    for group in groups:
        temp=group['T']; batch=[]
        for comp in group['compositions']:
            h=HPER[comp]; nat=128+round(h*128)
            tag=f'{name}_{comp}_{temp}K_{nat}at'; directory=ROOT/'runs/eos_live'/tag
            done=saved(directory/(tag+'.csv'))
            if 'volumes' in group:
                if comp not in group['volumes']:raise SystemExit(f'Missing explicit volume grid for {comp}')
                grid=group['volumes'][comp]
            else:
                vmax=round((30+(h-2)*8)/(1+h),3);vmin=round((5.25+(h-2)*.9)/(1+h),3)
                grid=np.geomspace(vmax,vmin,group.get('nv',16))
            volumes=sorted(round(float(v),4) for v in grid)
            if not volumes or any(not np.isfinite(v) or v<=0 for v in volumes) or len(volumes)!=len(set(volumes)):
                raise SystemExit(f'{temp} K {comp}: invalid or duplicate volumes')
            skip_volumes=[v for v in volumes if (comp,temp,v) in skipped_keys and v not in done]
            matched_skips.update((comp,temp,v) for v in volumes if (comp,temp,v) in skipped_keys)
            state_specs.append(dict(T=temp,comp=comp,tag=tag,directory=directory,volumes=volumes,skipped=skip_volumes))
            for vol in volumes:
                if vol in done:completed+=1;continue
                if vol in skip_volumes:
                    skipped.append((comp,temp,vol));continue
                batch.append(dict(T=temp,comp=comp,hper=h,nat=nat,vpa=vol,directory=directory,tag=tag,eq=group.get('eq',.25),ps=group.get('ps',.75),cost=nat*(.5+4/vol)))
            print(f'{temp} K {comp} ({nat} atoms, 128 O): {sum(v not in done and v not in skip_volumes for v in volumes)} to run, {sum(v in done for v in volumes)} saved, {len(skip_volumes)} skipped',flush=True)
        tasks.extend(sorted(batch,key=lambda t:-t['cost']))
    if skipped_keys-matched_skips:raise SystemExit(f'Skip does not match planned grid: {sorted(skipped_keys-matched_skips)}')
    print(f'{len(tasks)} planned volumes pending, {completed} saved, {len(skipped)} skipped; {a.gpus} GPUs x {a.per_gpu} slots; H7O uses {a.h7_slots} slots, others 1. H4O and pure H excluded.',flush=True)
    if not a.run:
        print('Preview only; add --run within the GPU allocation.')
        return
    stop=False
    def interrupt(*_):
        nonlocal stop
        stop=True
        for item in running.values():
            if item['proc'].poll() is None: item['proc'].terminate()
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    dashboard = sys.stdout.isatty()
    if dashboard:
        from eos_missing128_dashboard import render
        from eos_h4_dashboard import refresh_screen
    started=time.monotonic(); last=0; previous_lines=0
    envbase=os.environ.copy()
    for key in ('VMAX','VMIN','PS','EQ','NREP','NATOMS','GRACE_MODE','HHWALL','GRACE_UQ','EOS_QUEUE'):
        envbase.pop(key,None)
    envbase.update(PYTHONPATH=str(CODE),TF_FORCE_GPU_ALLOW_GROWTH='true',TF_CPP_MIN_LOG_LEVEL='2',OMP_NUM_THREADS='1')
    try:
        while tasks or running:
            for pid,item in list(running.items()):
                rc=item['proc'].poll()
                if rc is None:continue
                item['log'].close(); del running[pid]
                if rc != 0 or item['task']['vpa'] not in saved(item['table']):
                    failed.append((item['task'],rc,item['logpath']))
                    if not dashboard: print(f"FAILED {item['task']['T']} K vpa={item['task']['vpa']:.4f}, exit={rc}; {item['logpath']}",flush=True)
                else:finished+=1
            while tasks and not stop and len(failed)<a.max_failures:
                used=[sum(item['slots'] for item in running.values() if item['gpu']==g) for g in range(a.gpus)]
                # Find the first queued task that fits, so a large H7O task
                # cannot block a smaller task from filling an available slot.
                choice=None
                for index, candidate in enumerate(tasks):
                    slots=a.h7_slots if candidate['comp']=='h7o' else 1
                    free=[g for g in range(a.gpus) if used[g]+slots <= a.per_gpu]
                    if free:
                        choice=(index,slots,min(free,key=lambda g:used[g]));break
                if choice is None:break
                index,slots,gpu=choice; task=tasks.pop(index); task['directory'].mkdir(parents=True,exist_ok=True)
                logpath=task['directory']/f"flat_log_v{task['vpa']:.4f}.txt"
                log=logpath.open('a'); log.write(f"\nStarting {time.strftime('%Y-%m-%d %H:%M:%S')} GPU {ids[gpu]}\n");log.flush()
                cmd=[sys.executable,str(CODE/'17_eos_md.py'),a.model,'--phase','bcc','--branch','liquid','--hper',str(task['hper']),'--nrep','4',
                     '--premelt-T',os.environ.get('PM_T','9000'),'--premelt-ps',os.environ.get('PM_PS','0.25'),
                     '--nearby-start','--seed-root',os.environ.get('SEED_ROOT','runs/eos_live'),
                     '--T',str(task['T']),'--vpa',str(task['vpa']),'--equil-ps',str(task['eq']),'--ps',str(task['ps']),
                     '--report-fs','50','--tag',task['tag'],'--outdir',str(task['directory'])]
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(envbase,CUDA_VISIBLE_DEVICES=ids[gpu]))
                running[proc.pid]=dict(proc=proc,gpu=gpu,slots=slots,task=task,log=log,logpath=logpath,table=task['directory']/(task['tag']+'.csv'))
                if not dashboard:
                    print(f"START {task['T']} K vpa={task['vpa']:.4f} GPU={gpu}; {logpath}",flush=True)
            now=time.monotonic()
            if now-last>=a.refresh:
                counts=[sum(item['gpu']==g for item in running.values()) for g in range(a.gpus)]
                summary=f'elapsed {int(now-started)}s | done {finished} (+{completed} saved) | running {len(running)} {counts}/GPU | pending {len(tasks)} | failed {len(failed)} | skipped {len(skipped)}'
                if failed:
                    ft,rc,_=failed[-1];summary+=f" | last: {ft['comp']} {ft['T']}K v={ft['vpa']:.4f}"
                if dashboard:
                    try:
                        previous_lines=refresh_screen(render(ROOT, state_specs, running, tasks, summary=summary), previous_lines)
                    except Exception as exc:
                        # A display problem must never stop or kill MD workers.
                        print(summary+f' | dashboard unavailable: {exc}',flush=True)
                else: print(summary,flush=True)
                last=now
            if (stop or len(failed)>=a.max_failures) and not running:break
            if tasks or running:time.sleep(2)
    finally:
        for item in running.values():
            if item['proc'].poll() is None:item['proc'].terminate()
        for item in running.values():
            item['proc'].wait();item['log'].close()
        for f in locks:f.close()
    for task,rc,logpath in failed:
        print(f"FAILED {task['T']} K vpa={task['vpa']:.4f}, exit={rc}; {logpath}",flush=True)
    for comp,temp,vol in skipped:
        print(f'SKIPPED {comp} {temp} K vpa={vol:.4f}; not marked complete',flush=True)
    if stop or failed:
        raise SystemExit('Stopped or finished with failures: completed rows retained; inspect failures before resuming.')
    if skipped:
        print('Queue finished with explicitly skipped points; these remain incomplete.',flush=True)
    else:
        print('128-O missing-state queue complete. Screen phase and convergence before using as liquid EOS.')

if __name__=='__main__':main()

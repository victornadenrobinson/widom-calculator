"""Run the H4O plan as one refillable volume queue on an interactive GPU node.

Locks all H4O temperatures against the original group queue. Preview by default.
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


def saved(path):
    if not path.exists():
        return set()
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or 'vpa' not in reader.fieldnames:
            raise ValueError(f'{path}: missing vpa header; inspect table')
        result = set()
        for row in reader:
            try:
                result.add(round(float(row['vpa']), 4))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f'{path}: invalid row; inspect table') from exc
        return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('plan')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--gpus', type=int, default=4)
    ap.add_argument('--per-gpu', type=int, default=4)
    ap.add_argument('--model', default='grace-ft:runs/grace/ft_mfinkc_cont')
    ap.add_argument('--refresh', type=float, default=10)
    a = ap.parse_args()
    if a.gpus < 1 or a.per_gpu < 1 or a.refresh <= 0:
        ap.error('GPU counts and refresh must be positive')
    os.chdir(ROOT)
    groups = [g for g in json.loads(Path(a.plan).read_text()) if g.get('compositions') == ['h4o']]
    if not groups:
        raise SystemExit('No H4O-only groups in plan')
    if len({g['T'] for g in groups}) != len(groups):
        raise SystemExit('Duplicate H4O temperatures in plan')
    if any(g.get('nrep') != 4 or 'pressures_GPa' in g or 'volumes' in g for g in groups):
        raise SystemExit('Expected H4O nrep=4 volume-scan groups only')
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
            for temp in sorted(g['T'] for g in groups):
                f=(lockdir/f'{name}_h4o_{temp}.lock').open('a');locks.append(f)
                fcntl.flock(f,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('H4O campaign lock held: stop or finish the existing queue before starting this one.')
    tasks=[]; completed=0
    for group in groups:
        temp=group['T']; tag=f'{name}_h4o_{temp}K_640at'; directory=ROOT/'runs/eos_live'/tag
        done=saved(directory/(tag+'.csv'))
        volumes=sorted(round(float(v),4) for v in np.geomspace(9.2,1.41,group.get('nv',16)))
        if len(volumes)!=len(set(volumes)):raise SystemExit('Volume rounding produced duplicates')
        for vol in volumes:
            if vol in done:completed+=1;continue
            tasks.append(dict(T=temp, vpa=vol, directory=directory, tag=tag, eq=group.get('eq',.25), ps=group.get('ps',.75)))
        print(f'{temp} K: {len(volumes)-sum(v in done for v in volumes)} to run, {sum(v in done for v in volumes)} already saved',flush=True)
    print(f'{len(tasks)} H4O volumes pending, {completed} saved; {a.gpus} GPUs x {a.per_gpu} slots; other compositions are not included.',flush=True)
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
        from eos_h4_dashboard import render, refresh_screen
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
            while tasks and not stop and not failed:
                counts=[sum(item['gpu']==g for item in running.values()) for g in range(a.gpus)]
                free=[g for g in range(a.gpus) if counts[g] < a.per_gpu]
                if not free:break
                gpu=min(free,key=lambda g:counts[g]); task=tasks.pop(0); task['directory'].mkdir(parents=True,exist_ok=True)
                logpath=task['directory']/f"flat_log_v{task['vpa']:.4f}.txt"
                log=logpath.open('a'); log.write(f"\nStarting {time.strftime('%Y-%m-%d %H:%M:%S')} GPU {ids[gpu]}\n");log.flush()
                cmd=[sys.executable,str(CODE/'17_eos_md.py'),a.model,'--phase','bcc','--branch','liquid','--hper','4','--nrep','4',
                     '--premelt-T',os.environ.get('PM_T','9000'),'--premelt-ps',os.environ.get('PM_PS','0.25'),
                     '--nearby-start','--seed-root',os.environ.get('SEED_ROOT','runs/eos_live'),
                     '--T',str(task['T']),'--vpa',str(task['vpa']),'--equil-ps',str(task['eq']),'--ps',str(task['ps']),
                     '--report-fs','50','--tag',task['tag'],'--outdir',str(task['directory'])]
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(envbase,CUDA_VISIBLE_DEVICES=ids[gpu]))
                running[proc.pid]=dict(proc=proc,gpu=gpu,task=task,log=log,logpath=logpath,table=task['directory']/(task['tag']+'.csv'))
                if not dashboard:
                    print(f"START {task['T']} K vpa={task['vpa']:.4f} GPU={gpu}; {logpath}",flush=True)
            now=time.monotonic()
            if now-last>=a.refresh:
                counts=[sum(item['gpu']==g for item in running.values()) for g in range(a.gpus)]
                summary=f'elapsed {int(now-started)}s | done {finished} (+{completed} saved) | running {len(running)} {counts}/GPU | pending {len(tasks)} | failed {len(failed)}'
                if dashboard:
                    try:
                        previous_lines=refresh_screen(render(ROOT, viewer=False, summary=summary), previous_lines)
                    except Exception as exc:
                        # A display problem must never stop or kill MD workers.
                        print(summary+f' | dashboard unavailable: {exc}',flush=True)
                else: print(summary,flush=True)
                last=now
            if (stop or failed) and not running:break
            if tasks or running:time.sleep(2)
    finally:
        for item in running.values():
            if item['proc'].poll() is None:item['proc'].terminate()
        for item in running.values():
            item['proc'].wait();item['log'].close()
        for f in locks:f.close()
    for task,rc,logpath in failed:
        print(f"FAILED {task['T']} K vpa={task['vpa']:.4f}, exit={rc}; {logpath}",flush=True)
    if stop or failed:raise SystemExit('Stopped: completed rows retained; interrupted volumes restart. Check failures before resuming.')
    print('H4O queue complete. Screen phase and convergence before using as liquid EOS.')

if __name__=='__main__':main()

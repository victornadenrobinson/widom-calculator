"""Dynamic GPU queue for liquid-H2O quadrature families GL4/GK9 ... GL15/GK31.
Uses all Slurm-assigned GPUs, with a configurable number of lambda workers per GPU.
"""
import argparse
from collections import deque
import datetime
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mlp_pipeline'))
from tdi.quadrature_plan import plan
from tdi.analyze import rule_nodes

@lru_cache(maxsize=64)
def lambda_grid(serialized_args):
    return rule_nodes(json.loads(serialized_args))[0]


def choose_gpu(gpus, active, slots):
    """Least occupied available GPU; stable order breaks ties."""
    loads={g:sum(t['gpu']==g for t in active) for g in gpus}
    available=[g for g in gpus if loads[g]<slots]
    return min(available,key=lambda g:loads[g]) if available else None


def node_complete(directory,tag,k,target):
    sf=directory/f'{tag}_node{k}_state.npz';f=directory/f'{tag}_node{k}.npy'
    if not sf.exists() or not f.exists():return False
    try:
        a=json.loads((directory/f'{tag}_stage0.json').read_text())['args']
        needed=round(target*1000/a['dt']/a['every']);s=np.load(f)
        with np.load(sf) as z:
            expected = dict(model=a['model'], dt=a['dt'], T=json.loads((directory/f'{tag}_stage0.json').read_text())['T'], lam=lambda_grid(json.dumps(a,sort_keys=True))[k])
            for key,value in expected.items():
                if key not in z: return False
                actual=z[key].item()
                if (actual!=value) if isinstance(value,str) else not np.isclose(actual,value,rtol=1e-12,atol=0): return False
            return len(s)>=needed and np.isfinite(s).all() and np.array_equal(s,z['series'])
    except (ValueError,OSError,KeyError):return False


def worker_status(task, now=None):
    """Describe a running task, ignoring stale progress from earlier processes."""
    now=time.time() if now is None else now
    elapsed=max(0,now-task['launched'])
    j=task['job']; k=task['k'];label=f"{j['rule'].upper()}{j['nodes']}"
    if k is None:
        return f"{label} setup | {elapsed:.0f}s (preparing)"
    f=j['directory']/f"{j['tag']}_node{k}_progress.json"
    try:
        p=json.loads(f.read_text())
        if p.get('pid')!=task['proc'].pid or p.get('t',0)<task['launched']:
            raise ValueError('stale progress')
        frac=max(0.,min(1.,p['done']/p['total'])) if p['total'] else 0.
        rate=float(p.get('rate',float('nan')))
        speed=f'{rate:.1f} steps/s, {1000/rate:.0f} ms/step' if np.isfinite(rate) and rate>0 else 'warming up'
        eta=f", ETA {(p['total']-p['done'])/rate/60:.1f} min" if np.isfinite(rate) and rate>0 else ''
        stale=f", last update {now-p['t']:.0f}s ago" if now-p['t']>90 else ''
        return f"{label} λ{k}: {100*frac:.1f}% | {speed}{eta}{stale}"
    except (OSError,ValueError,KeyError,TypeError,ZeroDivisionError):
        return f"{label} λ{k}: starting/model loading ({elapsed:.0f}s)"


def cell_size(oxygen, hper):
    """Represent composition using available cubic FCC/BCC or 64-O rectangular BCC starts."""
    if oxygen < 1 or not np.isfinite(hper) or not 2 <= hper <= 7:
        raise ValueError('Use a positive oxygen count and H/O between 2 and 7 (available EOS coverage).')
    nrep = {32: 2, 64: 4, 108: 3}.get(oxygen, round((oxygen / 2) ** (1 / 3)))
    if oxygen not in (32,64,108) and (nrep < 1 or 2 * nrep ** 3 != oxygen):
        raise ValueError('For the current cubic bcc builder --oxygen must be 2*n^3: e.g. 16, 32, 54, 64, 108, 128, 250.')
    hydrogen = round(oxygen * hper)
    if not np.isclose(hydrogen, oxygen * hper, rtol=0, atol=1e-8):
        raise ValueError('H/O times oxygen count must be an integer; choose a representable composition.')
    return nrep, hydrogen


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--P',type=float,default=350);ap.add_argument('--T',type=float,default=7000)
    ap.add_argument('--ps',type=float,default=6.25);ap.add_argument('--root',default=None)
    ap.add_argument('--oxygen',type=int,default=128,help='O atoms / HxO formula units (supported: 16,32,54,64,108,128,250...; default 128)')
    ap.add_argument('--hper',type=float,default=2.0,help='hydrogen per oxygen, e.g. 2.25 (default 2)')
    ap.add_argument('--per-gpu',type=int,default=4);ap.add_argument('--families',type=int,nargs='+',default=[4,8,10,15],choices=[4,8,10,15])
    ap.add_argument('--model',default='grace-ft:runs/grace/ft_mfinkc_cont/seed/1/final_model')
    args=ap.parse_args();os.chdir(ROOT)
    try: nrep, hydrogen = cell_size(args.oxygen,args.hper)
    except ValueError as e: ap.error(str(e))
    if args.root is None:
        args.root = ('runs/quad384_350GPa_7000K' if (args.oxygen,args.hper,args.P,args.T)==(128,2.,350.,7000.)
                     else f'runs/quad_O{args.oxygen}_H{args.hper:g}_{args.P:g}GPa_{args.T:g}K')
    if not os.environ.get('SLURM_JOB_ID'):ap.error('Run inside a Slurm GPU allocation.')
    gpus=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if not all(gpus) or len(set(gpus))!=len(gpus):ap.error('Need valid Slurm-assigned CUDA_VISIBLE_DEVICES.')
    if args.per_gpu<1:ap.error('--per-gpu must be positive.')
    if len(set(args.families))!=len(args.families):ap.error('Duplicate families.')
    if args.ps<=0 or abs(args.ps/.125-round(args.ps/.125))>1e-8:ap.error('--ps must be a positive multiple of 0.125 ps.')
    if args.model.startswith('grace-ft:') and not Path(args.model.split(':',1)[1],'saved_model.pb').is_file():ap.error('Missing model export.')
    root=Path(args.root).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'sweep.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:ap.error('Another sweep is already using this root.')
    liquid_start = 'fcc' if args.oxygen in (32,108) else 'bcc'
    config=dict(P=args.P,T=args.T,model=args.model,nrep=nrep,liquid_start=liquid_start,dt=.5,chunk_ps=.125,equil_ps=.5,hot_ps=.5)
    if args.hper != 2.: config['hper'] = args.hper
    shape = (4,4,2) if args.oxygen == 64 else None
    if shape: config['liquid_shape'] = list(shape)
    cf=root/'campaign.json'
    if cf.exists() and json.loads(cf.read_text())!=config:ap.error('Campaign settings differ; use a new --root.')
    cf.write_text(json.dumps(config,indent=2))
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    snap=root/'code'/stamp; snap.mkdir(parents=True);shutil.copytree(ROOT/'mlp_pipeline',snap/'mlp_pipeline',ignore=shutil.ignore_patterns('__pycache__'))
    code=snap/'mlp_pipeline';active=[];pending=deque();serial=0
    env=os.environ.copy()
    for key in ('GRACE_MODE','HHWALL','GRACE_UQ'):env.pop(key,None)
    env.update(PYTHONPATH=str(code),OMP_NUM_THREADS='1',TF_USE_LEGACY_KERAS='1',TF_FORCE_GPU_ALLOW_GROWTH='true',H3O_RUNS=str(ROOT/'runs'))
    env['H3O_GIT']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    jobs=[dict(n=n,stage=0,status='preparing',directory=root/f'gl{n}_gk{2*n+1}') for n in args.families]
    for j in jobs:j['directory'].mkdir(exist_ok=True)
    def diary(j,event,rc=0):
        de=env.copy();de.update(NODES=str(j['nodes']),RULE=j['rule'],NGAUSS=str(j['ng']))
        subprocess.run([sys.executable,str(code/'tdi/run_diary.py'),str(j['directory']),event,str(args.ps),str(rc)],env=de,check=True,stdout=subprocess.DEVNULL)
    def prepare(j):
        files=list(j['directory'].glob('*_stage0.json'));prev=json.loads(files[0].read_text())['args'] if files else None
        if len(files)>1:raise ValueError('Multiple runs in '+str(j['directory']))
        if prev and prev.get('rule')=='gk':j['stage']=1
        requested=f'gl{j["n"]}' if j['stage']==0 else f'gk{2*j["n"]+1}'
        rule,nodes,ng=plan(prev,requested);j.update(rule=rule,nodes=nodes,ng=ng,status='stage0',stage_started=time.time())
        j['tag']=f'liquid_{args.P:.0f}GPa_{args.T:.0f}K_h{args.hper:g}o_rep{nrep}' + ('_startbcc' if liquid_start == 'bcc' else '')
        if shape: j['tag'] += '_shape4x4x2'
        j['cmd']=[sys.executable,'-u',str(code/'tdi/run_real.py'),args.model,'--phase','liquid','--liquid-start',liquid_start,'--nrep',str(nrep),
                  '--P',str(args.P),'--T',str(args.T),'--hper',str(args.hper),'--nodes',str(nodes),'--rule',rule,'--ngauss',str(ng),
                  '--power','4','--ps',str(args.ps),'--ps1','.5','--equil-ps','.5','--hot-ps','.5','--hot-T','12000',
                  '--chunk-ps','.125','--dt','.5','--outdir',str(j['directory'])]
        if shape: j['cmd'] += ['--liquid-shape',*map(str,shape)]
        action = ('RESUME/EXTEND saved samples; missing lambdas will run' if files else 'NEW run directory')
        if files and all(node_complete(j['directory'],j['tag'],k,args.ps) for k in range(nodes)):
            action = 'ALL lambdas already at target; skip production (grid may advance)'
        color = ('\033[36m' if files else '\033[32m') if sys.stdout.isatty() else ''
        reset = '\033[0m' if color else ''
        print(f'{color}{j["directory"]}: {action}; no samples will be cleaned{reset}',flush=True)
        diary(j,'start');pending.append(dict(job=j,k=None))
    def finish(j):
        if not all(node_complete(j['directory'],j['tag'],k,args.ps) for k in range(j['nodes'])):
            j['status']='FAILED';diary(j,'finish',1);return
        with (j['directory']/f'{stamp}_{j["rule"]}{j["nodes"]}_collection.log').open('w') as fh:
            rc=subprocess.run([sys.executable,str(code/'tdi/collect.py'),str(j['directory']),j['tag']],env=env,stdout=fh,stderr=subprocess.STDOUT).returncode
        diary(j,'finish',rc)
        if rc:j['status']='FAILED'
        elif j['stage']==0:j['stage']=1;prepare(j)
        else:j['status']='done'
    def terminal_lines():
        lines=[f'{time.strftime("%H:%M:%S")} | Liquid H{args.hper:g}O: {args.oxygen} O + {hydrogen} H = {args.oxygen+hydrogen} atoms | {args.P:g} GPa / {args.T:g} K | {args.ps:g} ps/lambda | {len(gpus)} GPUs, {args.per_gpu} slots/GPU',
               'Shared lambda queue; log files: '+str(root)]
        for gpu in gpus:
            tasks=[t for t in active if t['gpu']==gpu]
            lines.append(f'GPU {gpu}: '+(f'{len(tasks)} running' if tasks else 'idle'))
            for task in tasks: lines.append('  '+worker_status(task))
        for j in jobs:
            fraction=0.;done=0
            if 'nodes' in j:
                for k in range(j['nodes']):
                    if node_complete(j['directory'],j['tag'],k,args.ps):fraction+=1;done+=1;continue
                    f=j['directory']/f'{j["tag"]}_node{k}_progress.json'
                    try:
                        p=json.loads(f.read_text())
                        if p['t']>=j['stage_started']:fraction+=min(1.,p['done']/(2000*(args.ps+.5)))
                    except (ValueError,OSError,KeyError):pass
                fraction/=j['nodes']
            bar='#'*int(fraction*25)+'-'*(25-int(fraction*25))
            lines.append(f'GL{j["n"]}→GK{2*j["n"]+1}: [{bar}] {100*fraction:5.1f}%  {j["status"]} ({done}/{j.get("nodes",0)} lambdas saved)')
        return lines
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    previous=0;last_print=0
    try:
        for j in jobs:prepare(j)
        while pending or active:
            # Fill free slots across every assigned GPU; stage0 dependencies are queued first.
            while pending:
                gpu=choose_gpu(gpus,active,args.per_gpu)
                if gpu is None:break
                t=pending.popleft();j=t['job']
                if j['status']=='FAILED':continue
                serial+=1;label='stage0' if t['k'] is None else f'node{t["k"]}'
                t['log']=j['directory']/f'{stamp}_{serial}_{j["rule"]}{j["nodes"]}_{label}.log'
                t['fh']=t['log'].open('a');te=env.copy();te['CUDA_VISIBLE_DEVICES']=gpu
                cmd=j['cmd']+(['--stage0'] if t['k'] is None else ['--node',str(t['k'])])
                t.update(gpu=gpu,launched=time.time(),proc=subprocess.Popen(cmd,env=te,stdout=t['fh'],stderr=subprocess.STDOUT,start_new_session=True));active.append(t)
            for t in list(active):
                rc=t['proc'].poll()
                if rc is None:continue
                active.remove(t);t['fh'].close();j=t['job']
                if rc or (t['k'] is not None and not node_complete(j['directory'],j['tag'],t['k'],args.ps)):
                    if j['status']!='FAILED':diary(j,'finish',rc or 1)
                    j['status']='FAILED';print(f'Failed task: {t["log"]}',flush=True);continue
                if t['k'] is None:
                    j['status']='sampling'
                    for k in range(j['nodes']):
                        if not node_complete(j['directory'],j['tag'],k,args.ps):pending.append(dict(job=j,k=k))
                if j['status']=='sampling' and not any(x['job'] is j for x in list(pending)+active):finish(j)
            lines=terminal_lines()
            if sys.stdout.isatty():
                if previous:sys.stdout.write(f'\033[{previous}A\033[J')
                for line in lines:sys.stdout.write('\r\033[2K'+line+'\n')
                sys.stdout.flush();previous=len(lines)
            elif time.time()-last_print>60 or not(pending or active):print('\n'.join(lines),flush=True);last_print=time.time()
            if pending or active:time.sleep(3)
    finally:
        for t in active:
            if t['proc'].poll() is None:os.killpg(t['proc'].pid,signal.SIGTERM)
        for t in active:
            try:t['proc'].wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(t['proc'].pid,signal.SIGKILL);t['proc'].wait()
            t['fh'].close()
    if any(j['status']!='done' for j in jobs):raise SystemExit('Incomplete/failed tasks; inspect logs. Rerun to resume.')
    print('All selected families completed. Increase --ps to extend; rerun unchanged to verify saved results.')

if __name__=='__main__':main()

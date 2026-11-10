"""Four GPU liquid-H2O quadrature study: independent GL4/GK9, GL8/GK17,
GL10/GK21 and GL15/GK31 families. Updating terminal, full logs, restartable.
Run with the Young GRACE Python inside a four-GPU Slurm allocation.
"""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mlp_pipeline'))
from tdi.quadrature_plan import plan


def completed(directory, nodes, target):
    files=list(directory.glob('*_stage0.json'))
    if not files: return False
    st=json.loads(files[0].read_text());a=st['args'];tag=files[0].name[:-12]
    if int(a['nodes'])!=nodes:return False
    needed=round(target*1000/a['dt']/a['every'])
    for k in range(nodes):
        f=directory/f'{tag}_node{k}.npy'
        if not f.exists():return False
        s=np.load(f)
        if len(s)<needed or not np.isfinite(s).all():return False
    return True


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--P',type=float,default=350);ap.add_argument('--T',type=float,default=7000)
    ap.add_argument('--ps',type=float,default=6.25);ap.add_argument('--root',default='runs/quad384_350GPa_7000K')
    ap.add_argument('--model',default='grace-ft:runs/grace/ft_mfinkc_cont/seed/1/final_model')
    args=ap.parse_args();os.chdir(ROOT)
    if not os.environ.get('SLURM_JOB_ID'):ap.error('Run inside a Slurm GPU allocation.')
    gpu_ids=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(gpu_ids)<4 or not all(gpu_ids):ap.error('Need four Slurm-assigned GPUs in CUDA_VISIBLE_DEVICES.')
    if args.ps<=0 or abs(args.ps/.125-round(args.ps/.125))>1e-8:ap.error('--ps must be a positive multiple of 0.125 ps.')
    if args.model.startswith('grace-ft:') and not Path(args.model.split(':',1)[1],'saved_model.pb').is_file():ap.error('Missing model export.')
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'sweep.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:ap.error('Another sweep is already using this root.')
    config=dict(P=args.P,T=args.T,model=args.model,nrep=4,liquid_start='bcc',dt=.5,chunk_ps=.125,equil_ps=.5,hot_ps=.5)
    cf=root/'campaign.json'
    if cf.exists() and json.loads(cf.read_text())!=config:ap.error('Campaign settings differ; use a new --root.')
    cf.write_text(json.dumps(config,indent=2))
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    jobs=[dict(n=n,gpu=gpu_ids[i],stage=0,proc=None,status='waiting',directory=root/f'gl{n}_gk{2*n+1}') for i,n in enumerate((4,8,10,15))]
    for j in jobs:j['directory'].mkdir(exist_ok=True)
    def launch(j):
        files=list(j['directory'].glob('*_stage0.json'))
        prev=json.loads(files[0].read_text())['args'] if files else None
        if prev and prev.get('rule')=='gk':j['stage']=1
        rule=f'gl{j["n"]}' if j['stage']==0 else f'gk{2*j["n"]+1}'
        planned,nodes,ng=plan(prev,rule)
        env=os.environ.copy()
        for key in ('FRESH','ONLY0','ONLYNODES','INIT','VPA','BULK','K','PREPHOLD','PREPFREE','ENGINE','TAG','RULE','NGAUSS','UQ','GRACE_UQ','SEED','HOSTPHASE','NATOMS','MINRUNS','GRACE_MODE','HHWALL'):
            env.pop(key,None)
        env.update(PY=sys.executable,PYTHONPATH=str(ROOT/'mlp_pipeline'),CUDA_VISIBLE_DEVICES=j['gpu'],MODEL=args.model,
                   OUTDIR=str(j['directory']),P=str(args.P),T=str(args.T),PS=str(args.ps),NREP='4',LIQUID_START='bcc',HPER='2',
                   NODES=str(nodes),NPAR='4',POWER='4',DT='.5',EQUIL='.5',HOT='.5',HOTT='12000',CHUNK='.125',PS1='.5',
                   OMP_NUM_THREADS='1',TF_USE_LEGACY_KERAS='1',TF_FORCE_GPU_ALLOW_GROWTH='true')
        if planned=='gk':env.update(RULE='gk',NGAUSS=str(ng))
        log=j['directory']/f'{stamp}_{rule}.log';j.update(rule=rule,nodes=nodes,env=env,log=log)
        subprocess.run([sys.executable,'mlp_pipeline/tdi/run_diary.py',str(j['directory']),'start',str(args.ps)],env=env,check=True,stdout=subprocess.DEVNULL)
        j['fh']=log.open('a');j['proc']=subprocess.Popen(['bash','mlp_pipeline/young_tdi_par.sh','liquid'],env=env,stdout=j['fh'],stderr=subprocess.STDOUT,start_new_session=True)
        j['status']='running'
    def finish(j,code):
        j['fh'].close()
        subprocess.run([sys.executable,'mlp_pipeline/tdi/run_diary.py',str(j['directory']),'finish',str(args.ps),str(code)],env=j['env'],check=True,stdout=subprocess.DEVNULL)
        j['proc']=None
        if code or not completed(j['directory'],j['nodes'],args.ps):j['status']='FAILED';return
        if j['stage']==0:j['stage']=1;launch(j)
        else:j['status']='done'
    def display():
        lines=[f'Liquid H2O: 128 O + 256 H = 384 atoms | {args.P:g} GPa / {args.T:g} K | {args.ps:g} ps/lambda',
               'One quadrature family per GPU; four lambda processes per GPU. Logs: '+str(root)]
        for j in jobs:
            fraction=0.;counts=[]
            files=list(j['directory'].glob('*_stage0.json'))
            if files:
                try:
                    st=json.loads(files[0].read_text());a=st['args'];tag=files[0].name[:-12]
                    total=1000/.5*(.5+args.ps)
                    for k in range(j.get('nodes',a['nodes'])):
                        f=j['directory']/f'{tag}_node{k}_progress.json'
                        if f.exists():counts.append(min(1.,json.loads(f.read_text())['done']/total))
                        else:counts.append(0.)
                    fraction=sum(counts)/max(len(counts),1)
                except (ValueError,OSError,KeyError):pass
            if j['status']=='done':fraction=1.
            bar='#'*int(25*fraction)+'-'*(25-int(25*fraction))
            lines.append(f'GPU {j["gpu"]:>3} {j.get("rule","starting"):>5} [{bar}] {100*fraction:5.1f}%  {j["status"]:7}  {j.get("log","")}')
        return lines
    previous=0;last_print=0
    try:
        for j in jobs:launch(j)
        while True:
            for j in jobs:
                if j['proc'] and j['proc'].poll() is not None:finish(j,j['proc'].returncode)
            lines=display()
            if sys.stdout.isatty():
                if previous:sys.stdout.write(f'\033[{previous}A')
                for line in lines:sys.stdout.write('\r\033[2K'+line+'\n')
                sys.stdout.flush();previous=len(lines)
            elif time.time()-last_print>60 or all(j['status'] in ('done','FAILED') for j in jobs):
                print('\n'.join(lines),flush=True);last_print=time.time()
            if all(j['status'] in ('done','FAILED') for j in jobs):break
            time.sleep(5)
    finally:
        for j in jobs:
            if j['proc'] and j['proc'].poll() is None:
                os.killpg(j['proc'].pid,signal.SIGTERM);j['proc'].wait();j['fh'].close()
    if any(j['status']=='FAILED' for j in jobs):raise SystemExit('Some families failed or remain incomplete; inspect their logs. Rerun to resume.')
    print('All families completed. Rerun the same command to extend to a larger --ps.')

if __name__=='__main__':main()

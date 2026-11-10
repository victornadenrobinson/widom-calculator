"""Plan/submit one-GPU Young liquid TDI jobs; runs are resumable and volume-checked.
Default: print a plan. --submit sends only states with covered/supplied volumes.
"""
import argparse
import contextlib
import csv
import datetime
import fcntl
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(Path(__file__).resolve().parent))
MODEL='grace-ft:runs/grace/ft_mfinkc_cont/seed/1/final_model'
STATES=[(p,t) for t,ps in ((7000,(100,200,350,500)),(5000,(100,200,350,500)),(3000,(100,200))) for p in ps]
SYSTEMS=('h2o','h3o','h')


def volume_overrides(path):
    if not path:return {}
    out={}
    with open(path,newline='') as f:
        for row in csv.DictReader(f):
            key=(row['system'].strip().lower(),int(row['P_GPa']),int(row['T_K']))
            if key in out:raise ValueError(f'duplicate volume row: {key}')
            value=float(row['vpa_A3'])
            if key[0] not in SYSTEMS or not math.isfinite(value) or value<=0:raise ValueError(f'invalid volume row: {row}')
            out[key]=value
    return out


def build_plan(inventory,overrides):
    from tdi.run_real import hull_vpa, h_vpa
    result=[]
    for p,t in STATES:
        for system in SYSTEMS:
            key=(system,p,t);v=overrides.get(key);source='user-supplied volume table' if v is not None else ''
            reason=''
            if v is None:
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        if system=='h':v=h_vpa(str(inventory),t,p);source='DFT pure-H inventory fit'
                        else:v=hull_vpa('data-h3o-convex-hull.csv',2. if system=='h2o' else 3.,t,p);source='DFT hull EOS'
                    if not math.isfinite(v) or v<=0:raise ValueError('pressure outside EOS range')
                except (Exception,SystemExit) as e:
                    v=None;source='';reason=str(e).replace('\n',' ')[:180]
            result.append(dict(system=system,P_GPa=p,T_K=t,vpa_A3=v,source=source,blocked_reason=reason))
    return result


def plan_cli(a):
    overrides=volume_overrides(a.volumes);rows=build_plan(Path(a.inventory),overrides)
    print('System      P   T       V/atom      source / missing data')
    for r in rows:
        v=f"{r['vpa_A3']:.5f}" if r['vpa_A3'] is not None else 'MISSING'
        print(f"{r['system']:6} {r['P_GPa']:5} {r['T_K']:5} {v:>11}   {r['source'] or r['blocked_reason']}")
    ready=[r for r in rows if r['vpa_A3'] is not None];blocked=[r for r in rows if r['vpa_A3'] is None]
    print(f'\n{len(ready)} ready; {len(blocked)} blocked. One GPU/job; GK17, {a.ps:g} ps/lambda, max {a.concurrency} simultaneous jobs.')
    if blocked:print('Supply missing volumes with --volumes CSV: system,P_GPa,T_K,vpa_A3. No extrapolation is performed.')
    if not a.submit:print('PREVIEW ONLY; --submit queues the ready subset.');return
    if not ready:raise SystemExit('No covered states to submit.')
    if a.ps<=0 or abs(a.ps/.125-round(a.ps/.125))>1e-8:raise SystemExit('--ps must be a positive multiple of .125')
    if a.concurrency<1:raise SystemExit('--concurrency must be positive')
    if not Path(a.model.split(':',1)[-1],'saved_model.pb').is_file():raise SystemExit('Model export not found: '+a.model)
    if os.environ.get('SLURM_JOB_ID'):raise SystemExit('Submit from the Young login node, not inside an allocation.')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');campaign=Path('runs/liquid_campaigns')/stamp;campaign.mkdir(parents=True)
    shutil.copytree('mlp_pipeline',campaign/'code',ignore=shutil.ignore_patterns('__pycache__'))
    manifest=dict(created=stamp,git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),model=a.model,ps=a.ps,rows=ready,blocked=blocked,root=str(ROOT))
    mp=campaign/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    (campaign/'logs').mkdir();array=f'0-{len(ready)-1}%{a.concurrency}'
    cmd=['sbatch','--parsable',f'--array={array}',f'--time={a.time}',f'--output={campaign}/logs/%A_%a.out',str(campaign/'code/young_liquid_campaign.sbatch'),str(mp.resolve())]
    print('Submitting:', ' '.join(cmd),flush=True)
    result=subprocess.run(cmd,text=True,capture_output=True)
    (campaign/'submission.json').write_text(json.dumps(dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
    if result.returncode:raise SystemExit('sbatch failed: '+result.stderr)
    print('Submitted job array:',result.stdout.strip(),'manifest:',mp)


def run_task(manifest_path,index):
    m=json.loads(Path(manifest_path).read_text());r=m['rows'][index];os.chdir(m['root']);code=Path(manifest_path).parent/'code'
    sys.path.insert(0,str(code));from young_quad_sweep import node_complete
    P,T,system=r['P_GPa'],r['T_K'],r['system'];phase='H' if system=='h' else 'liquid';hper=3 if system=='h3o' else 2
    root=Path('runs/liquid_tdi_mfinkc')/f'{system}_{P}GPa_{T}K';root.mkdir(parents=True,exist_ok=True)
    lock=(root/'campaign_job.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit(f'Another job holds {root}; refusing a concurrent run.')
    model=m['model'];ps=float(m['ps']);tag=(f'H_{P}GPa_{T}K_N729' if phase=='H' else f'liquid_{P}GPa_{T}K_h{hper}o_rep4_startbcc')
    env=os.environ.copy();env.update(PYTHONPATH=str(code),OMP_NUM_THREADS='1',TF_FORCE_GPU_ALLOW_GROWTH='true',TF_USE_LEGACY_KERAS='1',H3O_GIT=m['git'])
    common=[sys.executable,'-u',str(code/'tdi/run_real.py'),model,'--phase',phase,'--P',str(P),'--T',str(T),'--vpa',str(r['vpa_A3']),
            '--nodes','17','--rule','gk','--ngauss','8','--power','4','--ps',str(ps),'--ps1','.5','--equil-ps','.5','--chunk-ps','.125','--dt','.5','--outdir',str(root)]
    common+=['--natoms','729'] if phase=='H' else ['--hper',str(hper),'--liquid-start','bcc','--nrep','4','--hot-ps','.5','--hot-T','12000']
    print('Explicit volume: bulk modulus is unknown unless supplied separately; pressure correction is not automatically applied.',flush=True)
    print(f"{system} at {P} GPa, {T} K | V/atom {r['vpa_A3']:.5f} ({r['source']}) | {ps:g} ps/lambda | GPU {os.environ.get('CUDA_VISIBLE_DEVICES','unknown')}\nOutput: {root}",flush=True)
    stage=subprocess.run(common+['--stage0'],env=env)
    if stage.returncode:raise SystemExit(stage.returncode)
    def work(k):
        if node_complete(root,tag,k,ps):return k,0,'already complete'
        path=root/f'{tag}_node{k}.log'
        with path.open('a') as log:
            rc=subprocess.run(common+['--node',str(k)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc==0 and not node_complete(root,tag,k,ps):rc=1
        return k,rc,str(path)
    failed=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(work,k) for k in range(17)]
        for future in as_completed(futures):
            k,rc,path=future.result();print(f'lambda {k}: '+('done' if rc==0 else f'FAILED; see {path}'),flush=True)
            if rc:failed.append(k)
    if failed:raise SystemExit(f'Incomplete lambdas {failed}; rerun to resume.')
    rc=subprocess.run([sys.executable,str(code/'tdi/collect.py'),str(root),tag],env=env).returncode
    if rc:raise SystemExit(rc)
    print('Completed; verify phase, pressure and sampling before hull analysis.',flush=True)


def main():
    os.chdir(ROOT)
    if len(sys.argv)>1 and sys.argv[1]=='run':run_task(sys.argv[2],int(sys.argv[3]));return
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory',default='runs/inventory_binned.csv');ap.add_argument('--volumes',default=None)
    ap.add_argument('--model',default=MODEL);ap.add_argument('--ps',type=float,default=6.25);ap.add_argument('--concurrency',type=int,default=4)
    ap.add_argument('--time',default='12:00:00');ap.add_argument('--submit',action='store_true')
    plan_cli(ap.parse_args())

if __name__=='__main__':main()

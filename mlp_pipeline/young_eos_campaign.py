"""Submit Young single-GPU H2O/H3O/H EOS scans at 7000, 5000, 3000 K."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parent.parent
STATES=[(c,t) for t in (7000,5000,3000) for c in ('h2o','h3o','h')]
MODEL='grace-ft:runs/grace/ft_mfinkc_cont'


def run_task(manifest,index):
    m=json.loads(Path(manifest).read_text());os.chdir(m['root']);code=Path(manifest).parent/'code'
    comp,T=m['states'][index];lockdir=Path('runs/eos_locks');lockdir.mkdir(parents=True,exist_ok=True)
    lock=(lockdir/f'{Path(m["model"].split(":",1)[1]).name}_{comp}_{T}.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit(f'Another EOS job is running for {comp} {T} K.')
    env=os.environ.copy();env.update(PY=sys.executable,MODEL=m['model'],COMP=comp,T=str(T),NGPU='1',PER_GPU='4',
                                     NV=str(m['nv']),PLOT='none',REFRESH='120',OMP_NUM_THREADS='1',EOS_REPO_ROOT=m['root'],EOS_CODE_DIR=str(code))
    print(f'{comp} {T} K; 1 GPU, four processes; {m["nv"]} volume points; model {m["model"]}',flush=True)
    raise SystemExit(subprocess.run(['bash',str(code/'eos_live.sh')],env=env).returncode)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='run':run_task(sys.argv[2],int(sys.argv[3]));return
    os.chdir(ROOT);ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--submit',action='store_true');ap.add_argument('--nv',type=int,default=16)
    ap.add_argument('--concurrency',type=int,default=4);ap.add_argument('--time',default='12:00:00');ap.add_argument('--model',default=MODEL);a=ap.parse_args()
    if a.nv<2 or a.concurrency<1:ap.error('NV >=2 and concurrency >=1 required')
    print('Nine one-GPU EOS jobs; model',a.model)
    for c,t in STATES:print(f'  {c:4} {t} K: {a.nv} volume points')
    print('Existing volume rows in matching model/composition/temperature directories are skipped; no deletion.')
    if not a.submit:print('PREVIEW ONLY; --submit starts the array.');return
    if os.environ.get('SLURM_JOB_ID'):ap.error('Submit from the login node.')
    if not (Path(a.model.split(':',1)[1])/'seed/1/final_model/saved_model.pb').is_file():ap.error('Final model export not found under model run directory.')
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');camp=Path('runs/eos_campaigns')/stamp;camp.mkdir(parents=True)
    shutil.copytree('mlp_pipeline',camp/'code',ignore=shutil.ignore_patterns('__pycache__'));(camp/'logs').mkdir()
    manifest=dict(root=str(ROOT),states=STATES,model=a.model,nv=a.nv,git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    mp=camp/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    cmd=['sbatch','--parsable',f'--array=0-{len(STATES)-1}%{a.concurrency}',f'--time={a.time}',f'--output={camp}/logs/%A_%a.out',str(camp/'code/young_eos_campaign.sbatch'),str(mp.resolve())]
    r=subprocess.run(cmd,text=True,capture_output=True);(camp/'submission.json').write_text(json.dumps(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    if r.returncode:raise SystemExit('sbatch failed: '+r.stderr)
    print('Submitted EOS array:',r.stdout.strip(),'manifest:',mp)

if __name__=='__main__':main()

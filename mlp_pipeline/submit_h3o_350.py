"""Submit the missing single 350 GPa / 7000 K H3O liquid-TDI array task.

Uses the previously submitted liquid campaign's immutable code snapshot/settings.
The volume is a screening guess from rounded final-model Perlmutter EOS points;
measured pressure, phase and model identity require verification.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import subprocess


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',default='runs/liquid_campaigns/20261110_205128_652967')
    ap.add_argument('--vpa',type=float,default=2.27088342)
    ap.add_argument('--submit',action='store_true')
    a=ap.parse_args()
    if a.vpa<=0:ap.error('Volume must be positive')
    base=Path(a.base); original=json.loads((base/'manifest.json').read_text())
    code=base/'code'
    if not (code/'young_liquid_campaign.sbatch').is_file():raise SystemExit('Missing original campaign code snapshot')
    target=Path('runs/liquid_tdi_mfinkc/h3o_350GPa_7000K')
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f'{target} already contains files; inspect before submitting a duplicate.')
    jobs=subprocess.run(['squeue','-h','-u',__import__('os').environ['USER'],'-o','%A %j'],capture_output=True,text=True,check=True)
    live={line.split()[0] for line in jobs.stdout.splitlines() if line.split()}
    for sf in Path('runs/liquid_campaigns').glob('*/submission.json'):
        try:
            s=json.loads(sf.read_text());job=s.get('stdout','').strip().split(';')[0]
            m=json.loads((sf.parent/'manifest.json').read_text())
            if job in live and any(r.get('system')=='h3o' and r.get('P_GPa')==350 and r.get('T_K')==7000 for r in m.get('rows',[])):
                raise SystemExit(f'Live job {job} already includes H3O 350 GPa 7000 K: {sf.parent}')
        except (OSError,ValueError,KeyError):
            continue
    row=dict(system='h3o',P_GPa=350,T_K=7000,vpa_A3=a.vpa,
             source='log-pressure interpolation of rounded Perlmutter final-model 7000 K EOS: 271.4 GPa at 2.4994 and 373.0 GPa at 2.2137 A3/atom; screening estimate',blocked_reason='')
    print(f'H3O 350 GPa / 7000 K, V/atom={a.vpa:.8f}; GK17, {original["ps"]} ps/lambda, one GPU, 12 h.')
    print('Volume is a screening guess from rounded short-trajectory EOS data; verify measured pressure and liquid phase.')
    print('Uses immutable code snapshot and model recorded in',base)
    if not a.submit:print('Preview only; use --submit to queue.');return
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    camp=Path('runs/liquid_campaigns')/stamp;camp.mkdir(parents=True)
    shutil.copytree(code,camp/'code',ignore=shutil.ignore_patterns('__pycache__'))
    (camp/'logs').mkdir()
    manifest=dict(created=stamp,git=original['git'],model=original['model'],ps=original['ps'],rows=[row],blocked=[],root=original['root'],parent_manifest=str((base/'manifest.json').resolve()))
    mp=camp/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n')
    cmd=['sbatch','--parsable','--array=0-0','--time=12:00:00',f'--output={camp}/logs/%A_%a.out',str(camp/'code/young_liquid_campaign.sbatch'),str(mp.resolve())]
    result=subprocess.run(cmd,capture_output=True,text=True)
    (camp/'submission.json').write_text(json.dumps(dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2)+'\n')
    if result.returncode:raise SystemExit(f'sbatch failed: {result.stderr}')
    print('Submitted array',result.stdout.strip(),'manifest',mp)

if __name__=='__main__':main()

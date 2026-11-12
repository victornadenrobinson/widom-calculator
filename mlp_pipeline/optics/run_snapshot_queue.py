"""Run prepared optical snapshots serially inside an existing Slurm allocation.

Each try has a separate attempt directory. Finished cases are skipped; failed or
interrupted attempts are preserved and block automatic retries. The final R is
preliminary and uses source EOS mean pressure, not a measured DFT pressure.
"""
import argparse
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from itertools import zip_longest


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def remaining():
    job=os.environ.get('SLURM_JOB_ID')
    if not job:raise RuntimeError('No SLURM_JOB_ID: run inside the allocated CPU job')
    text=subprocess.check_output(['squeue','-h','-j',job,'-o','%L'],text=True).strip().splitlines()
    if not text:raise RuntimeError('Slurm returned no remaining time')
    value=text[0].strip()
    if value.upper() in ('UNLIMITED','INFINITE'):return float('inf')
    days=0
    if '-' in value:
        d,value=value.split('-',1);days=int(d)
    parts=[int(x) for x in value.split(':')]
    if len(parts)==3:h,m,s=parts
    elif len(parts)==2:h,m,s=0,*parts
    elif len(parts)==1:h,m,s=0,0,parts[0]
    else:raise ValueError('Unexpected Slurm remaining-time format')
    return days*86400+h*3600+m*60+s


def interpolated(path,n0):
    with open(path,newline='') as f:r=list(csv.DictReader(f))
    key='R_n0_%g'%n0;E=1239.841984/532
    for a,b in zip(r,r[1:]):
        lo,hi=float(a['energy_eV']),float(b['energy_eV'])
        if lo<=E<=hi:
            t=(E-lo)/(hi-lo);return (1-t)*float(a[key])+t*float(b[key])
    raise ValueError('532 nm lies outside dielectric grid')


def valid(outcar):
    text=outcar.read_text(errors='replace')
    return 'aborting loop because EDIFF is reached' in text and 'General timing and accounting' in text


def analyse(case,work,plotter,py,origin):
    results={}
    for tensor in ('density','current'):
        prefix=work/('reflectivity_'+tensor)
        cmd=[py,str(plotter),'plot',str(work/'vasprun.xml'),'--tensor',tensor,'--out',str(prefix)]
        if not (prefix.with_suffix('.csv')).exists():subprocess.run(cmd,check=True)
        results[tensor]=interpolated(str(prefix)+'.csv',1.52)
    info=json.loads((case/'provenance.json').read_text())
    print(f"RESULT {info['composition']} GRACE mean P={info['pressure_mean_GPa']:.2f} GPa frame={info['frame_index_zero_based']} | 532nm n0=1.52 density={100*results['density']:.3f}% current={100*results['current']:.3f}% | {origin}",flush=True)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('roots',nargs='+');ap.add_argument('--plotter',required=True);ap.add_argument('--reuse',default=None)
    ap.add_argument('--start-buffer-min',type=float,default=12)
    ap.add_argument('--vasp', choices=('vasp_std','vasp_gam'), default='vasp_std', help='VASP executable for new attempts')
    a=ap.parse_args();py=sys.executable;plotter=Path(a.plotter).resolve()
    if not plotter.is_file():raise SystemExit('Missing plotter '+str(plotter))
    if a.start_buffer_min<=0:ap.error('Start buffer must be positive')
    if not os.environ.get('SLURM_JOB_ID'):raise SystemExit('Run inside your Perlmutter allocation')
    exe=shutil.which(a.vasp)
    if not exe:raise SystemExit('Executable not found on PATH: '+a.vasp)
    roots=[Path(x) for x in a.roots]
    for root in roots:
        if not root.is_dir():raise SystemExit('Missing prepared root '+str(root))
    groups=[sorted(r.glob('h*o_P*_frame*')) for r in roots]
    if any(not g for g in groups):raise SystemExit('At least one root has no prepared H–O snapshots')
    cases=[item for batch in zip_longest(*groups) for item in batch if item is not None]
    lockpath=Path('runs/optics/liquid_snapshot_queue.lock');lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another liquid optical snapshot queue holds the lock')
        reuse=Path(a.reuse) if a.reuse else None
        for case in cases:
            info=json.loads((case/'provenance.json').read_text())
            previous=sorted((case/'attempts').glob('*/DONE.json')) if (case/'attempts').is_dir() else []
            if previous:
                print('Already complete:',case,flush=True);analyse(case,previous[-1].parent,plotter,py,'completed run');continue
            if reuse and (reuse/'provenance.json').is_file() and (reuse/'OUTCAR').is_file() and (reuse/'vasprun.xml').is_file():
                old=json.loads((reuse/'provenance.json').read_text())
                same=old.get('trajectory')==info.get('trajectory') and old.get('frame_index_zero_based')==info.get('frame_index_zero_based')
                same=same and all((reuse/f).is_file() and digest(reuse/f)==digest(case/f) for f in ('INCAR','POSCAR','KPOINTS','POTCAR'))
                if same and valid(reuse/'OUTCAR'):
                    print('Reusing matching completed benchmark:',case,flush=True)
                    analyse(case,reuse,plotter,py,'reused benchmark');continue
            attempts=case/'attempts'
            if attempts.is_dir() and any(attempts.iterdir()):
                print('Preserving previous incomplete attempt; skipping pending manual review:',case,flush=True);continue
            try:seconds=remaining()
            except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as e:raise SystemExit('Cannot check remaining wall time: '+str(e))
            if seconds<a.start_buffer_min*60:
                print(f'Stopping with {seconds/60:.1f} min remaining; rerun to continue.',flush=True);break
            if a.vasp == 'vasp_gam':
                kpoints=(case/'KPOINTS').read_text().splitlines()
                if len(kpoints)<5 or kpoints[1].strip()!='0' or not kpoints[2].strip().lower().startswith('g') or kpoints[3].split()!=['1','1','1'] or any(float(x)!=0 for x in kpoints[4].split()):
                    raise SystemExit('vasp_gam requires the prepared Gamma 1x1x1 unshifted KPOINTS: '+str(case))
            attempt=attempts/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');attempt.mkdir(parents=True)
            for name in ('INCAR','KPOINTS','POSCAR','POTCAR','provenance.json'):shutil.copyfile(case/name,attempt/name)
            (attempt/'execution.json').write_text(json.dumps(dict(executable=a.vasp, resolved_path=exe),indent=2)+'\n')
            print(f'START {case} executable={exe} ({seconds/60:.1f} min remaining); log {attempt}/vasp.out',flush=True)
            t0=time.monotonic()
            with (attempt/'vasp.out').open('w') as f:
                rc=subprocess.run(['srun','-n','128','-c','2','--cpu-bind=cores',exe],cwd=attempt,stdout=f,stderr=subprocess.STDOUT).returncode
            elapsed=time.monotonic()-t0
            if rc or not (attempt/'OUTCAR').is_file() or not (attempt/'vasprun.xml').is_file() or not valid(attempt/'OUTCAR'):
                print(f'FAILED or incomplete (exit {rc}, {elapsed:.1f} s); inspect {attempt}; queue stopped.',flush=True);break
            result=analyse(case,attempt,plotter,py,'new run')
            (attempt/'DONE.json').write_text(json.dumps(dict(elapsed_s=elapsed,results=result,executable=a.vasp,resolved_executable=exe),indent=2)+'\n')
            print(f'DONE in {elapsed:.1f} s: {attempt}',flush=True)
    print('Queue stopped or finished. Source trajectories are short and nearby frames can be correlated; optical inputs need convergence.',flush=True)

if __name__=='__main__':main()

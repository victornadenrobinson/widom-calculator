"""Read-only live dashboard for eos_h4_flat_queue, on its allocated GPU node."""
import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import time

TEMPS=(7000,6000,5000,8000,9000,4000,3000,2000,1000)
STEP=re.compile(r'step\s+(\d+)/(\d+)\s+\[(.*?),\s*\d+ pts\]\s+T\s+(\d+) K\s+P\s+([-\d.]+)\s+\+-\s+([-\w.]+) GPa.*?([\d.]+) ms/step')
TAG=re.compile(r'ft_mfinkc_cont_h4o_(\d+)K_640at$')


def gpu_info():
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=5,check=True)
        return '   '.join(f'GPU{x[0].strip()} {x[1].strip()}% {int(x[2])/1024:.1f}/{int(x[3])/1024:.0f} GiB' for x in (line.split(',') for line in result.stdout.splitlines()))
    except (OSError,ValueError,subprocess.SubprocessError):return 'GPU statistics unavailable'


def active(root):
    jobs=[];proc=Path('/proc')
    for p in proc.iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if not any(s.endswith('17_eos_md.py') for s in args) or '--tag' not in args:continue
            tag=args[args.index('--tag')+1];match=TAG.fullmatch(tag)
            if not match:continue
            t=int(match.group(1));v=float(args[args.index('--vpa')+1])
            env=dict(x.split('=',1) for x in (p/'environ').read_bytes().decode(errors='replace').split('\0') if '=' in x)
            gpu=env.get('CUDA_VISIBLE_DEVICES','?')
            log=root/'runs/eos_live'/tag/f'flat_log_v{v:.4f}.txt'
            txt=log.read_text(errors='replace') if log.exists() else ''
            # Logs append between attempts: consider only the most recent attempt.
            txt=txt.rsplit('\nStarting ',1)[-1]
            stage='loading';progress='';temp='';pressure='';err='';ms='';eta=''
            if 'premelt' in txt:stage='premelt'
            if 'premelt done' in txt:stage='equil'
            matches=STEP.findall(txt)
            if matches:
                s,n,phase,temp,pressure,err,ms=matches[-1];s,n=int(s),int(n);stage='prod' if phase.startswith('prod') else 'equil'
                progress=f'{100*s/n:.0f}%';eta=f'{max(0,(n-s)*float(ms)/60000):.1f}m'
            jobs.append(dict(T=t,v=v,gpu=gpu,stage=stage,progress=progress,temp=temp,pressure=pressure,err=err,ms=ms,eta=eta,pid=p.name))
        except (OSError,ValueError,IndexError,ZeroDivisionError):continue
    return sorted(jobs,key=lambda j:(TEMPS.index(j['T']) if j['T'] in TEMPS else 99,j['v']))


def render(root):
    jobs=active(root);out=['H4O EOS: live read-only dashboard | 128 O, 640 atoms',gpu_info(),
       'MD progress and ETA cover equilibration + production only; premelt is separate.',
       f"{'target':>7} {'vpa':>6} {'GPU':>5} {'progress':>8} {'stage':>8} {'T K':>7} {'P GPa':>8} {'±':>6} {'ms/st':>6} {'ETA':>7}"]
    for j in jobs:
        out.append(f"{j['T']:7} {j['v']:6.3f} {j['gpu']:>5} {j['progress']:>8} {j['stage']:>8} {j['temp']:>7} {j['pressure']:>8} {j['err']:>6} {j['ms']:>6} {j['eta']:>7}")
    if not jobs:out.append('No H4O workers visible on this node. Check allocation and queue status.')
    out+=['','H4O P–T coverage (● saved EOS row, · running; log10 pressure in GPa)', '          1 GPa             10             100           1000']
    coverage={}
    for t in TEMPS:
        tag=f'ft_mfinkc_cont_h4o_{t}K_640at';path=root/'runs/eos_live'/tag/(tag+'.csv');done=[]
        try:
            with path.open(newline='') as f:
                for row in csv.DictReader(f):
                    try:
                        p=float(row['P_GPa']);v=round(float(row['vpa']),4)
                        if math.isfinite(p) and p>0:done.append((v,p))
                    except (KeyError,ValueError,TypeError):pass
        except OSError:pass
        line=[' ']*49
        def mark(p,symbol):
            if p>0:
                pos=max(0,min(48,round(math.log10(p)*16)));line[pos]=symbol
        for j in jobs:
            if j['T']==t and j['pressure']:
                try:mark(float(j['pressure']),'·')
                except ValueError:pass
        for _,p in done:mark(p,'●')
        coverage[t]=done
        running=sum(j['T']==t for j in jobs)
        out.append(f"{t:5} K |{''.join(line)}| saved {len(set(v for v,_ in done)):2}/16  running {running}")
    # P–V plot for the first currently running temperature, matching the old
    # dashboard's volume-per-formula-unit convention (5 atoms per H4O).
    selected=jobs[0]['T'] if jobs else None
    points=[(p,5*v,'●') for v,p in coverage.get(selected,[])]
    for job in jobs:
        if job['T']==selected and job['pressure']:
            try:points.append((float(job['pressure']),5*job['v'],'·'))
            except ValueError:pass
    if points:
        grid=[[' ']*49 for _ in range(9)]
        for p,v,symbol in points:
            if p<=0:continue
            x=max(0,min(48,round(math.log10(p)*16)))
            y=max(0,min(8,round((46-v)/39*8)))
            if symbol=='●' or grid[y][x]==' ':grid[y][x]=symbol
        out += ['',f'H4O at {selected} K: V per formula unit vs pressure (● saved, · running)']
        for i,row in enumerate(grid):out.append(f'{46-i*39/8:5.1f} |'+''.join(row))
        out.append('      +-------------------------------------------------')
        out.append('       1 GPa            10             100          1000')
    out+=['','Saved rows are not phase/convergence certification; blank pressures are not yet reported.',
          'This viewer does not control the queue. Ctrl-C closes only the viewer.']
    return '\n'.join(out)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path.cwd());ap.add_argument('--once',action='store_true');ap.add_argument('--refresh',type=float,default=10)
    a=ap.parse_args()
    if a.refresh<=0:ap.error('refresh must be positive')
    while True:
        if not a.once:print('\033[H\033[2J',end='')
        print(render(a.root.resolve()),flush=True)
        if a.once:return
        time.sleep(a.refresh)

if __name__=='__main__':main()

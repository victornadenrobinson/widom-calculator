"""Read-only live dashboard for eos_h4_flat_queue, on its allocated GPU node."""
import argparse
import csv
import math
import os
from pathlib import Path
import re
import shutil
import sys
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


VIS = re.compile(r"\x1b\[[0-9;]*m")


def fit(line, width):
    # As in eos_live.sh: prevent long lines from wrapping and breaking refresh.
    if len(VIS.sub('', line)) <= width:
        return line
    out = ''; visible = 0; i = 0
    while i < len(line) and visible < width:
        m = VIS.match(line, i)
        if m:
            out += m.group(); i = m.end()
        else:
            out += line[i]; i += 1; visible += 1
    return out + '\033[0m'


def plot(points, temp, rows, cols):
    # Same termplot canvas, symbols, axes and scaling as eos_live.sh.
    try:
        import numpy as np
        from termplot import Canvas, frame, fg, PAL
    except ImportError:
        return ['P–V plot unavailable: termplot or NumPy missing']
    if not points:
        return [f'EOS at {temp} K: waiting for first pressure values']
    pts=[(math.log10(p),5*v,kind) for p,v,kind in points if p>0 and math.isfinite(p)]
    if not pts: return ['Waiting for pressure values']
    xs=[p[0] for p in pts];ys=[p[1] for p in pts]
    xlo,xhi=min(xs)-.05,max(xs)+.05
    ylo,yhi=min(ys)*.95,max(ys)*1.03
    if xhi<=xlo:xhi=xlo+1
    if yhi<=ylo:yhi=ylo+1
    cv=Canvas(cols,rows)
    X=lambda x:(x-xlo)/(xhi-xlo)*(cv.w-1)
    Y=lambda y:(1-(y-ylo)/(yhi-ylo))*(cv.h-1)
    for x,y,kind in pts:
        dd=np.array([-1,0,1]) if kind=='done' else np.array([0])
        gx,gy=np.meshgrid(dd,dd);cv.dots(X(x)+gx.ravel(),Y(y)+gy.ravel(),1)
    legend=fg(PAL[1],'● H4O')+'  (block = saved, dot = running; no H4O DFT reference)'
    return frame(cv,(xlo,xhi),(ylo,yhi),'log10 P (GPa)   [1 = 10, 2 = 100, 3 = 1000 GPa]',
                 'V per formula unit (A^3)',f'EOS at {temp} K: H4O model',legend=legend).split('\n')


def render(root, viewer=True, summary=None):
    size=shutil.get_terminal_size((160,50));width=max(40,size.columns-1);height=max(12,size.lines-1)
    jobs=active(root);coverage={};queue=[]
    for t in TEMPS:
        tag=f'ft_mfinkc_cont_h4o_{t}K_640at';path=root/'runs/eos_live'/tag/(tag+'.csv');done={}
        try:
            with path.open(newline='') as f:
                for row in csv.DictReader(f):
                    try:
                        v=round(float(row['vpa']),4);p=float(row['P_GPa'])
                        done[v]=p if math.isfinite(p) and p>0 else None
                    except (KeyError,ValueError,TypeError):pass
        except OSError:pass
        coverage[t]=done
        nr=sum(j['T']==t for j in jobs); remaining=max(0,16-len(done)-nr)
        status='running' if nr else ('done' if len(done)>=16 else 'waiting')
        queue.append(f'{t}K H4O {status}: {len(done)}/16, run {nr}, wait {remaining}')
    selected=max(TEMPS,key=lambda t:sum(j['T']==t for j in jobs)) if jobs else TEMPS[0]
    out=[summary or 'H4O EOS queue: read-only dashboard',gpu_info(),
         f"{'target':>7} {'vpa':>6} {'GPU':>5} {'progress':>8} {'stage':>8} {'T K':>7} {'P GPa':>8} {'±':>6} {'ms/st':>6} {'ETA':>7}"]
    max_jobs=max(1,height-16)
    for j in jobs[:max_jobs]:
        out.append(f"{j['T']:7} {j['v']:6.3f} {j['gpu']:>5} {j['progress']:>8} {j['stage']:>8} {j['temp']:>7} {j['pressure']:>8} {j['err']:>6} {j['ms']:>6} {j['eta']:>7}")
    if len(jobs)>max_jobs:out.append(f'... {len(jobs)-max_jobs} additional workers hidden; enlarge terminal')
    if not jobs:out.append('No H4O workers visible on this node; check the allocation.')
    points=[(p,v,'done') for v,p in coverage[selected].items() if p is not None]
    for j in jobs:
        if j['T']==selected and j['pressure']:
            try:points.append((float(j['pressure']),j['v'],'run'))
            except ValueError:pass
    room=height-len(out)-1
    panel=['EOS QUEUE','Campaign ETA: unknown']+queue
    if room>=6:
        graph=plot(points,selected,max(1,min(18,room-6)),max(20,min(72,width-47 if width>=120 else width-12)))
        if width>=120:
            leftwidth=max(1,width-42)
            for i in range(max(len(graph),len(panel))):
                left=fit(graph[i],leftwidth-2) if i<len(graph) else ''
                right=panel[i] if i<len(panel) else ''
                out.append(left+' '*(leftwidth-len(VIS.sub('',left)))+right)
        else:
            out.extend(graph);out.extend(panel)
    if len(out)<height:
        out.append('Ctrl-C closes viewer.' if viewer else 'Ctrl-C stops queue; completed rows retained.')
    return '\n'.join(fit(line,width) for line in out[:height])


def refresh_screen(display, previous):
    lines=display.split('\n')
    if previous:sys.stdout.write(f'\033[{previous}A')
    sys.stdout.write('\n'.join(line+'\033[K' for line in lines)+'\n\033[J')
    sys.stdout.flush()
    return len(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--root',type=Path,default=Path.cwd());ap.add_argument('--once',action='store_true');ap.add_argument('--refresh',type=float,default=10)
    a=ap.parse_args()
    if a.refresh<=0:ap.error('refresh must be positive')
    previous=0
    while True:
        display=render(a.root.resolve())
        if a.once: print(display,flush=True);return
        previous=refresh_screen(display,previous)
        time.sleep(a.refresh)

if __name__=='__main__':main()

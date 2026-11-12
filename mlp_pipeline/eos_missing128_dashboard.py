"""Original-style live dashboard for the refillable 128-O missing-state queue."""
import csv
import math
import re
import shutil
import numpy as np
from eos_h4_dashboard import gpu_info, progress_bar, fit, VIS, STEP
from termplot import Canvas, frame, fg, PAL

H={'h2o':2.,'h2.25':2.25,'h3o':3.,'h5o':5.,'h7o':7.}
ORDER=list(H)


def state_rows(spec):
    path=spec['directory']/(spec['tag']+'.csv');out={}
    try:
        with path.open(newline='') as f:
            for row in csv.DictReader(f):
                try:
                    v=round(float(row['vpa']),4);p=float(row['P_GPa'])
                    out[v]=p if math.isfinite(p) and p>0 else None
                except (KeyError,ValueError,TypeError):pass
    except OSError:pass
    return out


def read_worker(item):
    j=item['task'];txt=''
    try:txt=item['logpath'].read_text(errors='replace').rsplit('\nStarting ',1)[-1]
    except OSError:pass
    stage='loading';progress='';temp='';pressure='';err='';ms='';eta=''
    if 'premelt' in txt:stage='premelt'
    if 'premelt done' in txt:stage='equil'
    matches=STEP.findall(txt)
    if matches:
        s,n,phase,temp,pressure,err,ms=matches[-1];s,n=int(s),int(n)
        if n>0:
            stage='prod' if phase.startswith('prod') else 'equil'
            progress=f'{100*s/n:.0f}%';eta=f'{max(0,(n-s)*float(ms)/60000):.1f}m'
    return dict(comp=j['comp'],T=j['T'],v=j['vpa'],gpu=item['gpu'],stage=stage,progress=progress,temp=temp,pressure=pressure,err=err,ms=ms,eta=eta)


def dft_points(root,T,comp):
    try:
        with (root/'data-h3o-convex-hull.csv').open(newline='') as f:
            reader=csv.DictReader(f);points=[]
            for row in reader:
                try:
                    p=float(row[f'P {T}K {comp}']);v=float(row[f'V {T}K {comp} norm'])
                    if p>0 and math.isfinite(p) and math.isfinite(v):points.append((math.log10(p),v))
                except (KeyError,ValueError,TypeError):pass
            return sorted(points)
    except OSError:return []


def graph(root,T,points,rows,cols):
    # Original eos_live.sh braille plot and DFT reference convention.
    dfts={c:dft_points(root,T,c) for c in ORDER}
    allpoints=[(x,y) for vals in dfts.values() for x,y in vals]+[(math.log10(p),(1+H[c])*v) for c,p,v,_ in points if p>0 and math.isfinite(p)]
    if not allpoints:return [f'EOS at {T} K: waiting for pressure values']
    xs=[x for x,y in allpoints];ys=[y for x,y in allpoints]
    xlo,xhi=min(xs)-.05,max(xs)+.05;ylo,yhi=min(ys)*.95,max(ys)*1.03
    if xhi<=xlo:xhi=xlo+1
    if yhi<=ylo:yhi=ylo+1
    cv=Canvas(cols,rows);X=lambda x:(x-xlo)/(xhi-xlo)*(cv.w-1);Y=lambda y:(1-(y-ylo)/(yhi-ylo))*(cv.h-1)
    for vals in dfts.values():
        for (x0,y0),(x1,y1) in zip(vals,vals[1:]):cv.line(X(x0),Y(y0),X(x1),Y(y1),11)
    for comp,p,v,kind in points:
        if p<=0 or not math.isfinite(p):continue
        dd=np.array([-1,0,1]) if kind=='done' else np.array([0]);gx,gy=np.meshgrid(dd,dd)
        cv.dots(X(math.log10(p))+gx.ravel(),Y((1+H[comp])*v)+gy.ravel(),ORDER.index(comp)+1)
    present=[c for c in ORDER if any(item[0]==c for item in points)]
    legend=' '.join(fg(PAL[ORDER.index(c)+1],'● '+c) for c in present)+'  '+fg(PAL[11],'— DFT')+'  (block=saved, dot=running)'
    return frame(cv,(xlo,xhi),(ylo,yhi),'log10 P (GPa)   [1 = 10, 2 = 100, 3 = 1000 GPa]', 'V per formula unit (A^3)',f'EOS at {T} K: model vs DFT',legend=legend).split('\n')


def render(root, specs, running, pending, summary):
    size=shutil.get_terminal_size((160,50));width=max(40,size.columns-1);height=max(12,size.lines-1)
    jobs=sorted((read_worker(v) for v in running.values()),key=lambda j:(-j['T'],ORDER.index(j['comp']),j['v']))
    records={(s['T'],s['comp']):state_rows(s) for s in specs}
    out=[summary,gpu_info(),f"{'comp':7} {'target':>6} {'vpa':>6} {'GPU':>3} {'progress (eq+prod)':22} {'stage':>7} {'T K':>6} {'P GPa':>8} {'±':>5} {'ms/st':>6} {'ETA':>6}"]
    max_jobs=max(1,height-16)
    for j in jobs[:max_jobs]:out.append(f"{j['comp']:7} {j['T']:6} {j['v']:6.3f} {j['gpu']:3} {progress_bar(j['progress']):22} {j['stage']:>7} {j['temp']:>6} {j['pressure']:>8} {j['err']:>5} {j['ms']:>6} {j['eta']:>6}")
    if len(jobs)>max_jobs:out.append(f'... {len(jobs)-max_jobs} workers hidden; enlarge terminal')
    if not jobs:out.append('Waiting for workers...')
    panel=['EOS QUEUE','Campaign ETA: unknown']
    for spec in specs:
        t,c=spec['T'],spec['comp'];done=records[(t,c)]
        nr=sum(j['T']==t and j['comp']==c for j in jobs)
        nw=sum(j['T']==t and j['comp']==c for j in pending)
        ns=sum(v in done for v in spec['volumes']);status='running' if nr else ('done' if ns==len(spec['volumes']) else 'waiting')
        panel.append(f'{t}K {c} {status}: {ns}/{len(spec["volumes"])} run {nr} wait {nw}')
    selected=max([s['T'] for s in specs],key=lambda t:sum(j['T']==t for j in jobs))
    points=[]
    for spec in specs:
        if spec['T']==selected:
            points.extend((spec['comp'],p,v,'done') for v,p in records[(selected,spec['comp'])].items() if p is not None)
    for j in jobs:
        if j['T']==selected and j['pressure']:
            try:points.append((j['comp'],float(j['pressure']),j['v'],'run'))
            except ValueError:pass
    room=height-len(out)-1
    if room>=6:
        drawing=graph(root,selected,points,max(1,min(18,room-6)),max(20,min(72,width-47 if width>=120 else width-12)))
        if width>=120:
            leftwidth=width-42
            for i in range(max(len(drawing),len(panel))):
                left=fit(drawing[i],leftwidth-2) if i<len(drawing) else ''
                right=panel[i] if i<len(panel) else ''
                out.append(left+' '*(leftwidth-len(VIS.sub('',left)))+right)
        else:out.extend(drawing);out.extend(panel)
    if len(out)<height:out.append('Ctrl-C stops the queue; completed rows retained.')
    return '\n'.join(fit(line,width) for line in out[:height])

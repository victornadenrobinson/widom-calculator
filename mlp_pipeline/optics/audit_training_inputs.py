"""Sample unique original OUTCAR headers referenced by GRACE training frames.

Read-only except for an optional summary CSV. Samples are not a complete audit.
K-corrected labels are derived from original Gamma labels; header settings alone
cannot validate the correction or prove model weights match any DFT source.
"""
import argparse
from collections import Counter
import csv
import gzip
from itertools import islice
from pathlib import Path
import re

from ase.io import iread

KEYS=('ENCUT','EDIFF','ISMEAR','SIGMA','TEBEG','IBRION','NSW','ISIF','ISYM','LREAL','IALGO','NELM','NELMIN','NBANDS','NCORE','KPAR')


def locate(source,roots):
    p=Path(source).expanduser()
    candidates=[p] if p.is_absolute() else [root/p for root in roots]
    for candidate in candidates:
        for f in (candidate,candidate.with_name(candidate.name+'.gz')):
            if f.is_file():return f.resolve()
    return None


def header(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',errors='replace') as f:text=''.join(islice(f,12000))
    result={}
    for key in KEYS:
        m=re.search(r'\b'+re.escape(key)+r'\s*=\s*([-+\w.]+)',text,re.I)
        result[key]=m.group(1) if m else ''
    version=re.search(r'\bvasp\.[\w.]+',text,re.I)
    result['version']=version.group(0) if version else ''
    titles=re.findall(r'^\s*TITEL\s*=\s*(.+)$',text,re.M)
    result['potcar_titles']=' | '.join(dict.fromkeys(titles))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('frames',nargs='+');p.add_argument('--max-sources',type=int,default=40)
    p.add_argument('--out',default='runs/plots/training_source_input_audit.csv')
    a=p.parse_args()
    if a.max_sources<1:p.error('max-sources must be positive')
    roots=[Path.cwd(),Path.cwd().parent,Path.home()/'work']
    sources={};nframes=0;no_source=0;corrected=0
    for filename in a.frames:
        path=Path(filename)
        if not path.is_file():
            print('Missing frames file:',path);continue
        count=0
        for at in iread(str(path),index=':'):
            count+=1;nframes+=1
            if 'kcorr_meV_atom' in at.info:corrected+=1
            raw=at.info.get('source')
            if not raw:
                no_source+=1;continue
            raw=str(raw)
            if raw not in sources:sources[raw]=dict(nH=int((at.numbers==1).sum()),nO=int((at.numbers==8).sum()),N=len(at),frames=0)
            sources[raw]['frames']+=1
        print(f'{path}: {count} frames read')
    print(f'{nframes} frames, {len(sources)} unique source strings, {no_source} frames without source, {corrected} with k-correction metadata')
    # Spread chosen sources across their sorted list; explicit about non-random sampling.
    keys=sorted(sources)
    selected=keys if len(keys)<=a.max_sources else [keys[round(i*(len(keys)-1)/(a.max_sources-1))] for i in range(a.max_sources)] if a.max_sources>1 else keys[:1]
    rows=[]
    for source in selected:
        resolved=locate(source,roots);row=dict(source=source,resolved=str(resolved) if resolved else '',**sources[source])
        if resolved:
            try:row.update(header(resolved));row['status']='read'
            except (OSError,EOFError) as e:row['status']='read_error: '+str(e)
        else:row['status']='not_found'
        rows.append(row)
    if not rows:
        print('No source metadata found; cannot check original OUTCAR inputs from these frames.')
        return
    fields=['source','resolved','status','frames','N','nH','nO','version','potcar_titles']+list(KEYS)
    dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    read=[r for r in rows if r['status']=='read']
    print(f'Checked {len(read)}/{len(rows)} selected original source headers (evenly spaced through sorted source names; not a random or complete audit).')
    for key in ('ENCUT','ISMEAR','SIGMA','ISYM','LREAL','IALGO','version','potcar_titles'):
        print(f'  {key}: {dict(Counter(r.get(key,"") or "not found" for r in read))}')
    missing=[r['source'] for r in rows if r['status']!='read']
    if missing:
        print('Unresolved source examples:',missing[:8])
    print('CSV:',dest.resolve())
    print('SIGMA should be compared with the electronic temperature of each original source; template alone does not prove all source inputs match.')

if __name__=='__main__':main()

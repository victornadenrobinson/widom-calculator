"""Conservative reuse of completed nearby liquid EOS frames."""
import csv
import hashlib
import io
import math
from pathlib import Path
import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list
from eos_volume_table import usable


def nearby_liquid(root, model_name, N, nO, T, vpa, exclude=None):
    """Same model-tag/composition/size, T..T+2000 K, <=20% volume change.

Legacy EOS tags identify a model path, not a verified weight fingerprint.
Only completed summary rows with mobile species and basic quality checks qualify.
"""
    candidates = []
    for folder in Path(root).glob(f'{model_name}_*K_{N}at'):
        table = folder/(folder.name+'.csv')
        if not table.exists(): continue
        try:
            with table.open(newline='') as f: rows = list(csv.DictReader(f))
            for r in rows:
                try:
                    st, sv = float(r['T_set']), float(r['vpa'])
                    if int(float(r['N'])) != N or int(float(r['nO'])) != nO: continue
                    if not T <= st <= T+2000 or not .8 <= vpa/sv <= 1.2: continue
                    if usable(r, 'h' if nO==0 else 'mixture', st) is None: continue
                    if float(r.get('frac_HH_below_0p5') or 'nan') > .2: continue
                    dh=float(r['D_H_cm2s'])
                    if not math.isfinite(dh) or dh <= 2e-5: continue
                    if r.get('branch') not in ('liquid','fluid'): continue
                    frame=folder/f'{folder.name}_v{sv:.4f}'/'final.extxyz'
                    if exclude and frame.parent.resolve()==Path(exclude).resolve(): continue
                    if frame.exists(): candidates.append((abs(math.log(vpa/sv))+(st-T)/10000,frame,r))
                except (KeyError,ValueError,TypeError,ZeroDivisionError): continue
        except (OSError,csv.Error): continue
    for _,frame,r in sorted(candidates,key=lambda x:(x[0],str(x[1]))):
        try:
            raw=frame.read_bytes(); at=read(io.StringIO(raw.decode()),format='extxyz',index=-1)
            if len(at)!=N or int((at.numbers==8).sum())!=nO or set(at.numbers)-{1,8}: continue
            if not at.pbc.all(): continue
            lengths=at.cell.lengths()
            if not np.allclose(lengths,lengths[0],rtol=1e-5) or not np.allclose(at.cell.angles(),90,atol=1e-4): continue
            if not np.isclose(at.get_volume()/N,float(r['vpa']),rtol=1e-4): continue
            scale=(N*vpa/at.get_volume())**(1/3)
            at.set_constraint();at.set_cell(at.cell*scale,scale_atoms=True);at.wrap();at.calc=None
            i,j,d=neighbor_list('ijd',at,.6)
            threshold=np.where((at.numbers[i]==1)&(at.numbers[j]==1),.4,.6)
            if np.any(d<threshold): continue
            return at,dict(method='nearby_liquid',source=str(frame.resolve()),
                source_sha256=hashlib.sha256(raw).hexdigest(),source_T_K=float(r['T_set']),
                source_vpa_A3=float(r['vpa']),target_T_K=T,target_vpa_A3=vpa,
                cell_scale=scale,model_tag=model_name,model_identity='path/tag; weights not independently verified')
        except (OSError,ValueError,IndexError,KeyError,RuntimeError): continue
    return None,dict(method='fresh_build_premelt',reason='No qualifying nearby liquid frame')

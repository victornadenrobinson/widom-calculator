"""Collect original compact EOS tables and a reproducible, non-certifying inventory.

Run on the source host. Raw trajectories and labels are not copied or created.
The archive is an intermediate for review, not a training or liquid database.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import socket
import tarfile
from datetime import datetime, timezone


def number(value):
    try:
        v=float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='runs/eos_live')
    p.add_argument('--out', required=True, help='new .tar.gz archive; never overwritten')
    a=p.parse_args()
    root=Path(a.root)
    out=Path(a.out)
    if not root.is_dir(): p.error('Missing root: '+str(root))
    if out.exists(): p.error('Refusing to overwrite '+str(out))
    tables=sorted(d/(d.name+'.csv') for d in root.glob('ft_mfinkc_cont_*K_*at') if d.is_dir() and (d/(d.name+'.csv')).is_file())
    if not tables: raise SystemExit('No source EOS tables found; no archive created.')
    metadata=[]
    for path in tables:
        data=path.read_bytes()
        try:
            with path.open(newline='') as f: rows=list(csv.DictReader(f))
        except (OSError,UnicodeError,csv.Error) as exc:
            print('Unreadable:', path, exc)
            continue
        available=0
        for row in rows:
            v=number(row.get('vpa'))
            if v is not None and (path.parent/(path.parent.name+'_v%.4f'%v)/'traj.extxyz').is_file(): available+=1
        pressures=[v for v in (number(r.get('P_GPa')) for r in rows) if v is not None]
        temperatures=sorted({v for v in (number(r.get('T_set')) for r in rows) if v is not None})
        item=dict(source_path=str(path.resolve()),sha256=hashlib.sha256(data).hexdigest(),rows=len(rows),
                  finite_pressure_rows=len(pressures),pressure_range_GPa=[min(pressures),max(pressures)] if pressures else None,
                  T_set_K=temperatures,trajectory_files_present=available,
                  note='Presence and row count do not certify completion, equilibration, liquid phase or trajectory integrity.')
        metadata.append(item)
        pr=item['pressure_range_GPa']
        print(f"{path.parent.name}: rows={len(rows)}, finite P={len(pressures)}, trajectory files={available}, P range={pr}")
    if not metadata: raise SystemExit('No readable source tables; no archive created.')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),root=str(root.resolve()),
                  model_id_from_directory='ft_mfinkc_cont; exact checkpoint identity not established',
                  limitations='Full source CSV tables only; no raw trajectories, DFT labels, independent model hash or phase certification.',tables=metadata)
    out.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive creation prevents accidentally overwriting an earlier bundle.
    with out.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as tar:
            import io
            payload=(json.dumps(manifest,indent=2)+'\n').encode()
            info=tarfile.TarInfo('manifest.json');info.size=len(payload)
            tar.addfile(info,io.BytesIO(payload))
            for item in metadata:
                source=Path(item['source_path'])
                tar.add(source,arcname='tables/'+source.parent.name+'/'+source.name,recursive=False)
    print(f'Wrote {out}: {len(metadata)} tables; original values and per-table hashes retained.')
    print('Share the archive for review; raw trajectories remain on the source host.')


if __name__=='__main__':main()

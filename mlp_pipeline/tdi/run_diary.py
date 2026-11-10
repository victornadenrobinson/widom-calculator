"""Append pilot start/finish entries and verify retained sample prefixes on extension."""
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path
import sys
import numpy as np

root, event, target = Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
diary = root / 'RUN_DIARY.jsonl'
def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
entries = [json.loads(s) for s in diary.read_text().splitlines()] if diary.exists() else []
prior = next((e for e in reversed(entries) if e['event'] == 'start'), {}) if event == 'finish' else {}
saved = prior.get('samples', {})
counts = {}
for f in sorted(root.glob('*_node[0-9]*.npy')):
    a = np.load(f)
    counts[f.name] = dict(count=len(a), sha256=digest(a))
    if f.name in saved:
        old = saved[f.name]
        if len(a) < old['count'] or digest(a[:old['count']]) != old['sha256']:
            raise SystemExit(f'Existing sample prefix changed: {f}; inspect the run.')
entry = dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(), event=event,
             target_ps_per_lambda=target, samples=counts)
if event == 'start':
    entry['quadrature'] = {k: os.environ.get(k) for k in ('RULE','NODES','NGAUSS')}
    archive = root / 'run_history' / datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
    archive.mkdir(parents=True)
    for source in list(root.glob('*_stage0.json')) + list(root.glob('tdi_*.json')):
        shutil.copy2(source, archive / source.name)
    entry['previous_metadata'] = str(archive)
if event == 'finish':
    entry['exit_code'] = int(sys.argv[4])
    entry['retained_prefixes_verified'] = len(saved)
    missing = set(saved) - set(counts)
    if missing:
        raise SystemExit(f'Sample files disappeared: {sorted(missing)}')
with diary.open('a') as f:
    f.write(json.dumps(entry) + '\n')
with (root / 'RUN_DIARY.md').open('a') as f:
    f.write(f"- {entry['time']}: **{event}**, target {target:g} ps/lambda")
    if event == 'start':
        f.write(f"; quadrature {entry['quadrature']}")
    if event == 'finish':
        f.write(f"; exit {entry['exit_code']}; {len(saved)} previous sample prefixes verified")
    f.write('; samples: ' + ', '.join(f"{k}: {v['count']}" for k, v in counts.items()) + '\n')
print(f'Diary: {root / "RUN_DIARY.md"}')

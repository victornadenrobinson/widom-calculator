"""Read prepared snapshot queues without touching running calculations.

DONE.json is the queue's completion marker. A partial attempt might still be
running; check Slurm separately. SD describes snapshot spread, not uncertainty
in the mean or convergence of the optics calculation.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.digest()


def interp(path):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    energy = 1239.841984 / 532
    for a, b in zip(rows, rows[1:]):
        lo, hi = float(a['energy_eV']), float(b['energy_eV'])
        if lo <= energy <= hi and hi > lo:
            t = (energy-lo)/(hi-lo)
            return float(a['R_n0_1.52'])*(1-t)+float(b['R_n0_1.52'])*t
    raise ValueError('532 nm outside reflectivity grid')


def same_reuse(case, info, reuse):
    if not reuse or not (reuse/'provenance.json').is_file():
        return False
    old = json.loads((reuse/'provenance.json').read_text())
    if not info.get('trajectory') or (old.get('trajectory'), old.get('frame_index_zero_based')) != (info.get('trajectory'), info.get('frame_index_zero_based')):
        return False
    for name in ('INCAR', 'POSCAR', 'POTCAR', 'KPOINTS'):
        if not (case/name).is_file() or not (reuse/name).is_file() or digest(case/name) != digest(reuse/name):
            return False
    if not (reuse/'OUTCAR').is_file() or not (reuse/'vasprun.xml').is_file():
        return False
    text = (reuse/'OUTCAR').read_text(errors='replace')
    return 'aborting loop because EDIFF is reached' in text and 'General timing and accounting' in text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--reuse', type=Path, help='completed original benchmark directory')
    ap.add_argument('--out', type=Path, required=True, help='output prefix for case and state CSV tables')
    args = ap.parse_args()
    cases = []
    for root_text in args.roots:
        root = Path(root_text)
        if not root.is_dir():
            print('MISSING ROOT:', root)
            continue
        found = sorted(p for p in root.glob('h*o_P*_frame*') if p.is_dir())
        print(f'{root}: {len(found)} prepared cases')
        cases.extend(found)
    records = []
    for case in cases:
        row = dict(case=str(case), status='', composition='', T_K='', P_source_GPa='', source='', trajectory='', frame='', density_percent='', current_percent='', detail='')
        try:
            info = json.loads((case/'provenance.json').read_text())
            row.update(composition=info['composition'], T_K=info['temperature_set_K'], P_source_GPa=info['pressure_mean_GPa'], source=info.get('source_kind', 'GRACE EOS'), trajectory=info.get('trajectory', ''), frame=info.get('frame_index_zero_based', ''))
            done = sorted((case/'attempts').glob('*/DONE.json'))
            attempts = sorted(p for p in (case/'attempts').glob('*') if p.is_dir())
            if done:
                completed = json.loads(done[-1].read_text())
                row.update(status='DONE', detail=str(done[-1].parent))
                for tensor in ('density', 'current'):
                    value = float(completed['results'][tensor])
                    if not math.isfinite(value): raise ValueError('nonfinite result')
                    row[tensor+'_percent'] = 100*value
            elif same_reuse(case, info, args.reuse):
                row.update(status='REUSED', detail=str(args.reuse))
                for tensor in ('density', 'current'):
                    row[tensor+'_percent'] = 100*interp(args.reuse/('reflectivity_'+tensor+'.csv'))
            elif attempts:
                row.update(status='ATTEMPT_NO_DONE', detail=str(attempts[-1]))
            else:
                row.update(status='NOT_STARTED', detail='prepared; queue status unknown')
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            row.update(status='NEEDS_REVIEW', detail=str(e))
        records.append(row)
    groups = {}
    for r in records:
        key = tuple(r[x] for x in ('composition','T_K','P_source_GPa','source','trajectory'))
        groups.setdefault(key, []).append(r)
    summaries = []
    print('\nCases: 532 nm, n0=1.52; R in percent')
    for r in records:
        density = '--' if r['density_percent'] == '' else f"{r['density_percent']:.3f}"
        current = '--' if r['current_percent'] == '' else f"{r['current_percent']:.3f}"
        print(f"{r['status']:16} {r['composition']:7} T={str(r['T_K']):5} P={str(r['P_source_GPa']):8} frame={str(r['frame']):4} density={density:>7} current={current:>7}  {r['case']}")
    print('\nState summaries: mean ± sample SD, percentage points')
    for key, rows in groups.items():
        completed = [r for r in rows if r['status'] in ('DONE','REUSED')]
        summary = dict(zip(('composition','T_K','P_source_GPa','source','trajectory'), key))
        summary.update(prepared=len(rows), completed=len(completed), remaining=len(rows)-len(completed))
        for tensor in ('density', 'current'):
            vals = [r[tensor+'_percent'] for r in completed]
            summary[tensor+'_mean_percent'] = statistics.mean(vals) if vals else ''
            summary[tensor+'_sd_percent_points'] = statistics.stdev(vals) if len(vals) >= 2 else ''
        summaries.append(summary)
        def fmt(tensor):
            mean=summary[tensor+'_mean_percent']; sd=summary[tensor+'_sd_percent_points']
            return ('--' if mean == '' else f'{mean:.3f}') + (' ± --' if sd == '' else f' ± {sd:.3f}')
        print(f"{key[0]} T={key[1]} P={key[2]} {key[3]} | {len(completed)}/{len(rows)} | density {fmt('density')} | current {fmt('current')}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix, data in (('_cases.csv', records), ('_states.csv', summaries)):
        if data:
            path = Path(str(args.out)+suffix)
            with path.open('w', newline='') as f:
                writer=csv.DictWriter(f, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
            print('Wrote', path)
    print('ATTEMPT_NO_DONE may be active or interrupted; check Slurm. NOT_STARTED means prepared, not necessarily queued.')
    print('SD is snapshot spread, not SEM; frames can be correlated. Source pressure is a trajectory mean.')

if __name__ == '__main__':
    main()

"""Plot saved pure-H RDFs and geometric pairing against measured pressure.

No simulation is run. Missing pairing values can be recovered from the last half
of saved production frames. H2-like means mutual nearest neighbours below 0.85 A;
unpaired is its complement, not a lifetime-based atomic species measurement.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_pure_h import pair_fraction


def number(row, key):
    try:
        return float(row.get(key, 'nan'))
    except (TypeError, ValueError):
        return float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--temperature', type=int, required=True)
    ap.add_argument('--root', type=Path, default=Path('runs/eos_live'))
    ap.add_argument('--model', default='ft_mfinkc_cont')
    ap.add_argument('--atoms', type=int, default=729)
    ap.add_argument('--out', type=Path, default=None, help='output directory')
    ap.add_argument('--trajectories', action='store_true', help='recover missing fractions from saved frames')
    args = ap.parse_args()
    tag = f'{args.model}_h_{args.temperature}K_{args.atoms}at'
    folder = args.root / tag
    table = folder / f'{tag}.csv'
    out = args.out or Path(f'runs/plots/pure_h_{args.temperature}K')
    if not table.exists():
        raise SystemExit(f'EOS table not found: {table}')
    with table.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for row in rows:
        p, v = number(row, 'P_GPa'), number(row, 'vpa')
        if not (math.isfinite(p) and math.isfinite(v)):
            continue
        state = folder / f'{tag}_v{v:.4f}'
        fraction, source, frames = number(row, 'f_H2'), 'summary', 0
        if not math.isfinite(fraction) and args.trajectories:
            path = state / 'traj.extxyz'
            try:
                from ase.io import read
                traj = read(str(path), index=':')
                traj = traj[len(traj)//2:]
                vals = [pair_fraction(at) for at in traj]
                if vals:
                    fraction, source, frames = float(np.mean(vals)), 'last_half_saved_trajectory', len(vals)
            except (OSError, ValueError, IndexError) as exc:
                print(f'Cannot recover {path}: {exc}')
        records.append(dict(P_GPa=p, vpa_A3=v, f_H2_like=fraction,
                            fraction_source=source if math.isfinite(fraction) else 'missing',
                            frames_reanalysed=frames, rdf=state/'rdf.csv'))
    records.sort(key=lambda x: x['P_GPa'])
    if not records:
        raise SystemExit('No completed EOS rows with pressure and volume.')
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    count = 0
    for rec, color in zip(records, plt.cm.viridis(np.linspace(0, .9, len(records)))):
        try:
            with rec['rdf'].open(newline='') as stream:
                data = list(csv.DictReader(stream))
            r = [float(x['r_A']) for x in data]
            g = [float(x['g_HH']) for x in data]
            if not r:
                raise ValueError('empty RDF')
        except (OSError, KeyError, ValueError) as exc:
            print(f"Cannot read {rec['rdf']}: {exc}")
            continue
        ax.plot(r, g, color=color, label=f"{rec['P_GPa']:.1f} GPa")
        count += 1
    ax.set(xlabel='H–H distance (Å)', ylabel='gHH(r)',
           title=f'Pure hydrogen at {args.temperature} K: H–H RDF')
    ax.grid(alpha=.2)
    if count:
        ax.legend(title='Mean pressure', ncol=2, fontsize=8)
        for ext in ('png', 'pdf'):
            fig.savefig(out/f'rdf_{args.temperature}K.{ext}', dpi=180)
    else:
        print('No readable RDFs; RDF plot not written.')
    plt.close(fig)
    valid = [r for r in records if math.isfinite(r['f_H2_like'])]
    if valid:
        p = np.array([r['P_GPa'] for r in valid])
        f = np.array([r['f_H2_like'] for r in valid])
        fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
        ax.plot(p, 100*f, 'o-', label='H₂-like paired H')
        ax.plot(p, 100*(1-f), 's-', label='Unpaired H')
        ax.set(xlabel='Mean pressure (GPa)', ylabel='Fraction of H atoms (%)',
               title=f'Pure hydrogen at {args.temperature} K: geometric pairing', ylim=(-2, 102))
        ax.grid(alpha=.2)
        ax.legend()
        for ext in ('png', 'pdf'):
            fig.savefig(out/f'pair_fraction_{args.temperature}K.{ext}', dpi=180)
        plt.close(fig)
    else:
        print('No pairing fractions; fraction plot not written.')
    fields = ['P_GPa', 'vpa_A3', 'f_H2_like', 'fraction_source', 'frames_reanalysed']
    with (out/f'pair_fraction_{args.temperature}K.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    print(f'Read {len(records)} EOS rows; plotted {count} RDFs and {len(valid)} fractions.')
    print(f'Output: {out.resolve()}')
    for rec in records:
        f = rec['f_H2_like']
        value = f'{100*f:.1f}%' if math.isfinite(f) else 'missing'
        print(f"{rec['P_GPa']:8.1f} GPa  V={rec['vpa_A3']:.4f} Å³/atom  H₂-like={value:>8}  {rec['fraction_source']}")


if __name__ == '__main__':
    main()

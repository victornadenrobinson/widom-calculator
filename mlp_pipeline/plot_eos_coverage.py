#!/usr/bin/env python3
"""Plot saved final-model EOS rows and flag rows passing preliminary liquid checks."""
import argparse
import csv
import math
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eos_volume_table import usable

COMPS = ('h2o', 'h2.25', 'h3o', 'h5o', 'h7o', 'h')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('roots', nargs='+', help='one or more eos_live directories')
    ap.add_argument('--out', required=True, help='figure prefix')
    args = ap.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True, layout='constrained')
    counts = []
    for ax, comp in zip(axes.flat, COMPS):
        for root in args.roots:
            for d in sorted(Path(root).glob(f'ft_mfinkc_cont_{comp}_*K_*at')):
                f = d / (d.name + '.csv')
                if not f.is_file(): continue
                with f.open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                good, other = [], []
                for row in rows:
                    try:
                        p, T = float(row['P_GPa']), float(row['T_set'])
                        if not math.isfinite(p) or p <= 0 or not math.isfinite(T): continue
                        (good if usable(row, comp, T) is not None else other).append((p,T))
                    except (KeyError, ValueError, TypeError): continue
                if other: ax.scatter(*zip(*other), marker='x', color='tab:red', s=22)
                if good: ax.scatter(*zip(*good), facecolors='none', edgecolors='tab:blue', s=35)
                counts.append((str(f), len(good), len(other)))
        for p in (100, 200, 350, 500): ax.axvline(p, color='0.85', lw=.7)
        for T in (3000, 5000, 7000): ax.axhline(T, color='0.92', lw=.7)
        ax.set(title=comp, xscale='log', xlabel='Pressure (GPa)', ylabel='Temperature (K)')
    fig.suptitle('Final-model EOS: blue circles pass preliminary liquid screening; red crosses fail or lack data\nPoints do not prove phase or pressure coverage between samples')
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png','.pdf'): fig.savefig(str(out)+ext, dpi=160)
    with open(str(out)+'.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['source','candidate_liquid_rows','other_rows']);w.writerows(counts)
    for path, good, other in counts: print(f'{path}: {good} candidates, {other} failed/missing diagnostics')
    if not counts: print('No matching tables found; check the supplied directories.', file=sys.stderr)
    print(f'Wrote {out}.png, .pdf and .csv')

if __name__ == '__main__': main()

"""Sampled mutual-nearest-pair survival; frames cannot resolve between-frame breaks."""
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import iread
from ase.neighborlist import neighbor_list


def pair_sets(at, cutoffs):
    if not np.all(at.numbers == 1):
        raise ValueError('trajectory is not pure hydrogen')
    i, j, d = neighbor_list('ijd', at, max(cutoffs))
    nearest = np.full(len(at), -1, dtype=int)
    dist = np.full(len(at), np.inf)
    # A deterministic tie break; excludes periodic self images.
    for k in np.lexsort((j, d, i)):
        a, b, r = i[k], j[k], d[k]
        if a != b and nearest[a] < 0:
            nearest[a], dist[a] = b, r
    a = np.flatnonzero(nearest >= 0)
    b = nearest[a]
    mutual = (a < b) & (nearest[b] == a)
    return [set(zip(a[mutual & (dist[a] < c)].tolist(),
                    b[mutual & (dist[a] < c)].tolist())) for c in cutoffs]


def run_lengths(frame_sets):
    active, lengths = {}, []
    for pairs in frame_sets:
        for pair in active.keys() - pairs:
            lengths.append(active.pop(pair))
        for pair in pairs:
            active[pair] = active.get(pair, 0) + 1
    lengths.extend(active.values())  # boundary-censored runs remain identifiable below
    return np.asarray(lengths, dtype=int)


def metrics(frame_sets, n_atoms, frame_fs):
    """Occupancy weights all frames of observed runs; survival uses eligible origins."""
    lengths = run_lengths(frame_sets)
    nframes = len(frame_sets)
    lagmax = min(nframes - 1, int(400 / frame_fs))
    survival = []
    for lag in range(lagmax + 1):
        # Pairs present at each origin with enough frames remaining.
        denominator = sum(len(x) for x in frame_sets[:nframes-lag])
        numerator = int(np.maximum(lengths - lag, 0).sum())
        survival.append((lag * frame_fs, numerator / denominator if denominator else np.nan, denominator))
    occupancy = []
    for target in (0., 40., 80., 160.):
        lag = int(np.ceil(target / frame_fs))
        n = int(lengths[lengths >= lag + 1].sum())
        occupancy.append((lag * frame_fs, 2*n/(nframes*n_atoms)))
    return survival, occupancy


def analyse_campaign(records, temperature, out, frame_fs, cutoffs):
    summaries, curves = [], []
    for rec in records:
        path = Path(rec['rdf']).parent/'traj.extxyz'
        sets = [[] for _ in cutoffs]
        n_atoms = None
        try:
            for at in iread(str(path), index=':'):
                if n_atoms is None:
                    n_atoms = len(at)
                if len(at) != n_atoms:
                    raise ValueError('atom count changes')
                for dest, pairs in zip(sets, pair_sets(at, cutoffs)):
                    dest.append(pairs)
        except (OSError, ValueError, IndexError) as exc:
            print(f'Cannot analyse {path}: {exc}')
            continue
        if not sets[0]:
            print(f'Empty trajectory: {path}')
            continue
        for cutoff, pairs in zip(cutoffs, sets):
            survival, occupancy = metrics(pairs, n_atoms, frame_fs)
            for lag, val, origins in survival:
                curves.append(dict(P_GPa=rec['P_GPa'], vpa_A3=rec['vpa_A3'], cutoff_A=cutoff,
                                   lag_fs=lag, survival=val, eligible_pair_origins=origins,
                                   frames=len(pairs), frame_fs=frame_fs, source=str(path)))
            for lag, val in occupancy:
                summaries.append(dict(P_GPa=rec['P_GPa'], vpa_A3=rec['vpa_A3'], cutoff_A=cutoff,
                                      min_sampled_duration_fs=lag, persistent_fraction=val,
                                      frames=len(pairs), frame_fs=frame_fs, source=str(path)))
        print(f'Pair dynamics: {rec["P_GPa"]:.1f} GPa, {len(sets[0])} frames: {path}')
    if not summaries:
        print('No trajectories analysed; no dynamics plots written.')
        return
    for name, rows in [('pair_survival', curves), ('pair_persistence', summaries)]:
        with (out/f'{name}_{temperature}K.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    fig, axes = plt.subplots(1, len(cutoffs), figsize=(5*len(cutoffs), 4), squeeze=False, layout='constrained')
    for ax, cutoff in zip(axes[0], cutoffs):
        for target in sorted({r['min_sampled_duration_fs'] for r in summaries}):
            rows = sorted((r for r in summaries if r['cutoff_A']==cutoff and r['min_sampled_duration_fs']==target), key=lambda x:x['P_GPa'])
            ax.plot([r['P_GPa'] for r in rows], [100*r['persistent_fraction'] for r in rows], 'o-', label=f'{target:g} fs')
        ax.set(title=f'cutoff {cutoff:g} Å', xlabel='Mean pressure (GPa)', ylabel='H in sampled persistent pairs (%)', ylim=(-2,102))
        ax.grid(alpha=.2)
        ax.legend(title='Minimum duration')
    fig.suptitle(f'Pure H, {temperature} K: sampled pair persistence (frame spacing {frame_fs:g} fs)')
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'pair_persistence_{temperature}K.{ext}', dpi=180)
    plt.close(fig)
    # The centre cutoff is shown; all cutoffs and pressure points remain in CSV.
    cutoff = cutoffs[len(cutoffs)//2]
    states = sorted({r['P_GPa'] for r in curves})
    fig, ax = plt.subplots(figsize=(9,6), layout='constrained')
    for p, color in zip(states, plt.cm.viridis(np.linspace(0,.9,len(states)))):
        rows = [r for r in curves if r['cutoff_A']==cutoff and r['P_GPa']==p]
        ax.plot([r['lag_fs'] for r in rows], [r['survival'] for r in rows], color=color, label=f'{p:.1f} GPa')
    ax.set(xlabel='Lag (fs)', ylabel='Sampled continuous survival probability', ylim=(-.02,1.02),
           title=f'Pure H, {temperature} K: pair survival, {cutoff:g} Å cutoff')
    ax.grid(alpha=.2)
    ax.legend(title='Mean pressure', ncol=2, fontsize=8)
    for ext in ('png', 'pdf'):
        fig.savefig(out/f'pair_survival_{temperature}K.{ext}', dpi=180)
    plt.close(fig)
    print('Persistence counts full observed runs, including boundary-censored runs; survival uses eligible time origins.')
    print('A pair may break and reform between saved frames. Verify the specified frame spacing and atom ordering.')

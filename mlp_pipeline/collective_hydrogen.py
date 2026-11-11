#!/usr/bin/env python3
"""Exploratory collective molecularity from pure-H EOS trajectories (CPU only).

Synthetic references are not electronic bond labels. Read the fit diagnostics
before interpreting fractions. Requires numpy, scipy, matplotlib, and ASE.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

FRACTIONS = np.linspace(0, 1, 5)
BONDS = np.array([.70, .76, .82, .88])
WIDTHS = np.array([.035, .07, .12])
EDGES = np.linspace(.54, 1.30, 25)


def distances(x, cell):
    # EOS runs have cubic/orthorhombic cells. Reject unsupported cells explicitly.
    if not np.allclose(cell, np.diag(np.diag(cell)), atol=1e-8):
        raise ValueError('Only orthorhombic cells are supported by this analysis')
    lengths = np.diag(cell)
    if np.any(lengths <= 0):
        raise ValueError('Invalid periodic cell')
    d = x[:, None] - x[None, :]
    d -= lengths * np.round(d / lengths)
    return np.linalg.norm(d, axis=-1)


def features(x, cell):
    n = len(x)
    d = distances(x, cell)
    np.fill_diagonal(d, np.inf)
    pair = np.histogram(d[np.triu_indices(n, 1)], EDGES)[0] * 2 / n
    nn = np.partition(d, 1, axis=1)[:, :2]
    nn.sort(axis=1)
    ratio = nn[:, 0] / nn[:, 1]
    return np.r_[pair, np.histogram(ratio, np.linspace(0, 1, 11))[0] / n]


def generate(rng, density, fraction, bond, width, n=64, exclusion=.54):
    length = (n / density) ** (1 / 3)
    if length <= 2 * EDGES[-1]:
        raise ValueError('Calibration cell too small: increase --reference-atoms')
    pairs = round(n * fraction / 2)
    for restart in range(10):
        xyz = []
        for obj in range(n - pairs):
            for attempt in range(3000):
                center = rng.uniform(0, length, 3)
                if obj < pairs:
                    direction = rng.normal(size=3)
                    direction /= np.linalg.norm(direction)
                    r = np.clip(rng.normal(bond, width), .55, 1.15)
                    candidate = (center + np.array([-.5, .5])[:, None] * r * direction) % length
                else:
                    candidate = center[None]
                if xyz:
                    delta = candidate[:, None] - np.asarray(xyz)[None]
                    delta -= length * np.round(delta / length)
                    if np.any(np.sum(delta * delta, axis=-1) < exclusion ** 2):
                        continue
                xyz.extend(candidate)
                break
            else:
                break
        else:
            return np.asarray(xyz), np.eye(3) * length
    raise ValueError('Synthetic insertion jammed at this density; no estimate produced')


def calibration(density, args):
    spec = dict(version=1, density=float(density), reps=args.reference_reps,
                n=args.reference_atoms, seed=args.seed, exclusion=args.exclusion)
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]
    path = args.out / 'cache' / (key + '.npz')
    if path.exists():
        with np.load(path) as z:
            return z['samples']
    rng = np.random.default_rng(args.seed)
    samples = np.empty((len(BONDS), len(WIDTHS), len(FRACTIONS), args.reference_reps, 34))
    for bi, bond in enumerate(BONDS):
        for wi, width in enumerate(WIDTHS):
            for fi, fraction in enumerate(FRACTIONS):
                for rep in range(args.reference_reps):
                    x, cell = generate(rng, density, fraction, bond, width,
                                       args.reference_atoms, args.exclusion)
                    samples[bi, wi, fi, rep] = features(x, cell)
    np.savez_compressed(path, samples=samples, spec=json.dumps(spec))
    return samples


def fit_model(observed, samples):
    means = samples.mean(axis=3)
    scales = np.maximum(samples.std(axis=3, ddof=1).mean(axis=(0, 1, 2)), .025)
    grid = np.stack(np.meshgrid(np.linspace(.70, .88, 19), WIDTHS,
                               np.linspace(0, 1, 101), indexing='ij'), axis=-1).reshape(-1, 3)
    templates = RegularGridInterpolator((BONDS, WIDTHS, FRACTIONS), means)(grid)
    loss = np.mean(((templates - observed) / scales) ** 2, axis=1)
    best = np.argmin(loss)
    return grid[best], float(np.sqrt(loss[best])), templates[best]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('runs/eos_live'))
    p.add_argument('--model', default='ft_mfinkc_cont')
    p.add_argument('--temperature', type=int, default=2000)
    p.add_argument('--out', type=Path, default=Path('runs/collective_h_2000K'))
    p.add_argument('--reference-reps', type=int, default=12)
    p.add_argument('--reference-atoms', type=int, default=64)
    p.add_argument('--exclusion', type=float, default=.54)
    p.add_argument('--seed', type=int, default=73121)
    p.add_argument('--discard', type=float, default=.5, help='Discard this fraction of saved frames')
    p.add_argument('--block-frames', type=int, default=20, help='Contiguous saved frames per block')
    a = p.parse_args()
    if not 0 <= a.discard < 1 or a.reference_reps < 2 or a.block_frames < 1:
        p.error('Require 0 <= discard < 1, reference-reps >= 2 and block-frames >= 1')
    if a.reference_atoms < 4 or a.reference_atoms % 4:
        p.error('reference-atoms must be a positive multiple of four, at least four')
    from ase.io import iread
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    paths = sorted(a.root.glob(f'{a.model}_h_{a.temperature}K_*at/*/traj.extxyz'))
    if not paths:
        p.error(f'No matching trajectories under {a.root}')
    (a.out / 'cache').mkdir(parents=True, exist_ok=True)
    (a.out / 'settings.json').write_text(json.dumps(vars(a), default=str, indent=2))
    records = []
    for path in paths:
        print(f'Reading {path}', flush=True)
        values, densities, sizes = [], [], []
        for at in iread(path, index=':'):
            if not np.all(at.numbers == 1) or not np.all(at.pbc):
                raise ValueError(f'{path}: requires pure H and full periodic boundaries')
            values.append(features(at.positions, np.asarray(at.cell)))
            densities.append(len(at) / at.get_volume())
            sizes.append(len(at))
        start = int(len(values) * a.discard)
        values = np.asarray(values[start:])
        density = float(np.mean(densities[start:]))
        if not len(values) or np.ptp(densities[start:]) > 1e-4 * density or len(set(sizes)) != 1:
            raise ValueError(f'{path}: empty or nonconstant-density trajectory')
        print(f'Calibrating rho={density:.6f}; {len(values)} retained frames', flush=True)
        samples = calibration(density, a)
        estimate, mismatch, fitted = fit_model(values.mean(axis=0), samples)
        blocks = [fit_model(values[i:i+a.block_frames].mean(axis=0), samples)[0][2]
                  for i in range(0, len(values) - a.block_frames + 1, a.block_frames)]
        pressure = float('nan')
        table = path.parent.parent / (path.parent.parent.name + '.csv')
        if table.exists():
            with table.open() as f:
                matches = [r for r in csv.DictReader(f) if abs(float(r['vpa']) - 1/density) < 5e-4]
            if len(matches) == 1:
                pressure = float(matches[0]['P_GPa'])
        row = dict(path=str(path), N=sizes[0], density=density, P_GPa=pressure,
                   retained_frames=len(values), blocks=len(blocks), f_molecular=estimate[2],
                   block_sd=float(np.std(blocks, ddof=1)) if len(blocks)>1 else float('nan'),
                   bond_parameter_A=estimate[0], width_parameter_A=estimate[1],
                   standardized_mismatch=mismatch)
        records.append(row)
        with (a.out / 'summary.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=row.keys()); w.writeheader(); w.writerows(records)
        np.savez_compressed(a.out / (path.parent.name + '_fit.npz'), observed=values,
                            fitted=fitted, block_fractions=blocks, estimate=estimate)
        print(f'  f={estimate[2]:.3f}; mismatch={mismatch:.2f}; blocks={len(blocks)}', flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for n in sorted({r['N'] for r in records}):
        rows = sorted([r for r in records if r['N']==n and np.isfinite(r['P_GPa'])], key=lambda r:r['P_GPa'])
        if not rows:
            continue
        x = [r['P_GPa'] for r in rows]
        axes[0,0].errorbar(x,[r['f_molecular'] for r in rows],yerr=[r['block_sd'] for r in rows],fmt='o-',label=f'N={n}')
        for ax,key in [(axes[0,1],'standardized_mismatch'),(axes[1,0],'bond_parameter_A'),(axes[1,1],'width_parameter_A')]:
            ax.plot(x,[r[key] for r in rows],'o-',label=f'N={n}')
    for ax,label in zip(axes.flat,['Estimated molecular H fraction','Standardized template mismatch','Fitted bond parameter (Å)','Fitted width parameter (Å)']):
        ax.set(xlabel='Measured pressure (GPa)',ylabel=label); ax.grid(alpha=.2)
        if ax.lines: ax.legend()
    axes[0,0].set_ylim(-.05,1.05)
    fig.suptitle(f'Pure H, {a.temperature} K — exploratory synthetic-reference inference\nBars: contiguous-block SD, not confidence intervals; large mismatch indicates reference failure')
    fig.savefig(a.out/'collective.png',dpi=180); fig.savefig(a.out/'collective.pdf')
    if any(not np.isfinite(r['P_GPa']) for r in records):
        print('WARNING: some pressures missing/ambiguous; these rows remain in CSV but are omitted from pressure plots')
    print(f'Results: {a.out}/summary.csv and collective.png',flush=True)

if __name__ == '__main__':
    main()

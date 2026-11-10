#!/usr/bin/env python3
"""Plot documented aggregate model results, without requiring cluster run files."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
with (ROOT / 'docs/data/model_accuracy_reported.csv').open() as f:
    rows = list(csv.DictReader(f))
with (ROOT / 'docs/data/zero_shot_force_ranges.csv').open() as f:
    zero = list(csv.DictReader(f))
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
fig = plt.figure(figsize=(16, 12), constrained_layout=True)
grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.3])
labels = [r['label'] for r in rows]
y = np.arange(len(rows)); colors = ['#78889a'] * 5 + ['#e76f51']
for i, (key, title, unit) in enumerate([
    ('energy_rmse_meV_atom', 'Energy RMSE', 'meV / atom'),
    ('force_rmse_eV_A', 'Force RMSE', 'eV / Å'),
    ('stress_rmse_GPa', 'Stress RMSE', 'GPa'),
    ('sd_cell_eV', 'Energy residual spread', 'eV / cell')]):
    ax = fig.add_subplot(grid[0, i])
    for j, r in enumerate(rows):
        if r[key]:
            v = float(r[key]); ax.barh(j, v, color=colors[j]); ax.text(v, j, f' {v:g}', va='center', fontsize=8)
        else: ax.text(0.02, j, 'not reported', va='center', fontsize=8, color='#777')
    ax.set_yticks(y, labels if i == 0 else []); ax.set_ylim(len(rows) - .5, -.5); ax.set_title(title); ax.set_xlabel(unit)
    ax.set_xlim(0, max(float(r[key]) for r in rows if r[key]) * 1.35)
ax = fig.add_subplot(grid[1, :2]); ax.axvspan(-50, 50, color='#e5f2e9', label='provisional ±50 meV/H')
for j, r in enumerate(rows):
    if r['composition_min_meV_H']:
        a, b = float(r['composition_min_meV_H']), float(r['composition_max_meV_H'])
        ax.plot([a, b], [j, j], color=colors[j], lw=4, solid_capstyle='round')
    else: ax.text(-135, j, 'not reported on same basis', va='center', fontsize=8)
ax.set_yticks(y, labels); ax.set_ylim(len(rows) - .5, -.5); ax.set_title('Composition test: reported range'); ax.set_xlabel('c (meV / added H)'); ax.set_xlim(-145, 100); ax.axvline(0, color='#aaa', lw=.7); ax.legend(loc='upper left', fontsize=8)
ax = fig.add_subplot(grid[1, 2:]); ax.axvspan(-50, 50, color='#e5f2e9')
for j, r in enumerate(rows):
    if r['pair_meV_H']:
        v = float(r['pair_meV_H']); ax.scatter(v, j, color=colors[j], s=45); ax.text(v+3, j, f'{v:g}  ({r["pair_status"]})', va='center', fontsize=8)
    else: ax.text(-180, j, 'not reported', va='center', fontsize=8, color='#777')
ax.set_yticks(y, []); ax.set_ylim(len(rows) - .5, -.5); ax.set_xlim(-190, 110); ax.axvline(0, color='#aaa', lw=.7); ax.set_title('Pair test: interpretation depends on training'); ax.set_xlabel('c (meV / added H)')
ax = fig.add_subplot(grid[2, :]); zy = np.arange(len(zero))
for j, r in enumerate(zero):
    lo, hi = float(r['F_RMSE_min']), float(r['F_RMSE_max']); ax.plot([lo, hi], [j, j], color='#327a91', lw=4, solid_capstyle='round')
ax.set_yticks(zy, [r['model'] for r in zero]); ax.invert_yaxis(); ax.set_xscale('log'); ax.set_xlabel('Force RMSE range across reported states (eV / Å; log scale)'); ax.set_title('Earlier zero-shot screening — different samples; do not rank directly against fine-tunes'); ax.grid(axis='x', alpha=.2)
fig.suptitle('Reported MLP and fine-tuning accuracy', fontsize=17, fontweight='bold')
fig.text(.5, -.035, 'Top: reported valid_n comparison (629 frames); k-corrected continuation evaluated against Gamma labels, force RMSE not stated.\nPair scores on training pairs are not independent tests. Zero-shot sample sizes / states differ. Sources: docs/HANDOFF.md, PROJECT.md, docs/data/zero_shot_force_ranges.csv.', ha='center', fontsize=9)
for ext in ('png', 'pdf'):
    fig.savefig(ROOT / f'docs/figures/reported_model_accuracy.{ext}', dpi=180, bbox_inches='tight')

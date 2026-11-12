"""Illustrative 532 nm linear-reflectivity scenario; not a pressure model.

Hydrogen Drude parameters: Zaghoo and Silvera, PNAS 114, 11873 (2017),
https://pmc.ncbi.nlm.nih.gov/articles/PMC5692551/. Water: one project snapshot.
Frozen endpoint values do not establish P or T dependence. Mixture fraction
counts H2 formula units in H2O + H2, not volume fraction.
"""
import argparse
import cmath
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def hydrogen_r(plasma_ev, tau_fs, n0):
    energy = 1239.841984 / 532.0
    hbar_ev_fs = 0.6582119569
    epsilon = 1 - plasma_ev**2 / (energy**2 + 1j * energy * hbar_ev_fs / tau_fs)
    n = cmath.sqrt(epsilon)
    return abs((n0 - n) / (n0 + n))**2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='runs/plots/linear_R_7000K_scenario')
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    water = 0.16885
    hs = [hydrogen_r(20.4, 0.13, 1.52), hydrogen_r(21.8, 0.16, 1.52)]
    pressures = np.linspace(250, 450, 101)
    fig, ax = plt.subplots(figsize=(9, 6))
    rows = []
    for x, color in zip([2, 2.25, 3, 5, 7, float('inf')], plt.cm.viridis(np.linspace(0, 1, 6))):
        fraction = 1 if np.isinf(x) else (x - 2) / x
        lo, hi = [(1-fraction)*water + fraction*h for h in hs]
        label = 'pure H' if np.isinf(x) else f'H{x:g}O'
        ax.plot(pressures, np.full_like(pressures, 50*(lo+hi)), '--', color=color, label=label)
        if fraction:
            ax.fill_between(pressures, 100*lo, 100*hi, color=color, alpha=.08)
        for pressure in pressures:
            rows.append([pressure, label, fraction, lo, hi])
    # Millot SI table 1: observed shot pressure and reflectivity, 1 sigma.
    exp_p = [165.0,175.0,177.9,187.5,218.7,236.0,260.0,275.9,284.8,292.6,355.8]
    exp_dp = [2.5,2.5,2.6,2.7,3.2,3.2,3.9,3.9,3.8,3.9,4.9]
    exp_r = [1.0,1.1,1.6,2.0,1.7,1.8,3.1,4.6,5.9,5.8,11.0]
    exp_dr = [.5,.5,.5,.5,.5,.5,.7,.9,1.7,1.1,3.4]
    ax.errorbar(exp_p, exp_r, xerr=exp_dp, yerr=exp_dr, fmt='o',
                color='tab:red', markersize=3, capsize=1, elinewidth=.7,
                label='Millot H2O: experiment, shock Hugoniot')
    # SI tables 3, 5 and 6: infer P at each table-3 density by linear
    # interpolation of tabulated P at the same temperature. No extrapolation.
    # Each tuple: T, rho, R %, rho_low, P_low, rho_high, P_high.
    calculated = [(5000,3.345,.83692,3.25,177.51,3.4,203.10),
                  (6000,3.458,1.862,3.4,215.76,3.537,240.76),
                  (7000,3.557,3.27828,3.5,246.97,3.6,265.58),
                  (8000,3.637,5.53853,3.6,277.79,3.709,301.21),
                  (10000,3.758,10.23318,3.003,188.15,3.826,351.55)]
    calc_p = [p0+(rho-r0)/(r1-r0)*(p1-p0)
              for T,rho,R,r0,p0,r1,p1 in calculated]
    ax.plot(calc_p,[row[2] for row in calculated], 's', color='purple',
            markersize=4, label='Millot H2O: PBE MD / HSE06 optics, P interpolated')
    ax.plot([363.44], [100*water], 'ko', markersize=6, label='Our H2O: PBE optics, 7000 K')
    ax.plot([140, 170], [100*h for h in hs], linestyle='none', marker='D',
            markersize=6, markerfacecolor='white', markeredgecolor='black',
            label='H: experimental Drude fits, n0 converted')
    # Approximate visual readings of filled orange triangles in Rillo et al.
    # Fig. 3: 5000 K, n0=1.49, HSE optical response / vdW-DF1 PIMD.
    # These are manually read from a user-provided screenshot, not source data.
    rillo_p = np.array([70, 84, 100, 122, 149, 170, 182, 199, 210, 225, 280])
    rillo_r = np.array([25.0, 27.5, 31.5, 38.5, 43.0, 45.5, 46.8, 48.5, 49.5, 50.5, 53.0])
    ax.errorbar(rillo_p, rillo_r, xerr=4, yerr=1.5, fmt='^', markersize=4,
                color='darkorange', elinewidth=.7, capsize=1.5,
                label='Rillo H: 5000 K, HSE optics, n0=1.49 (digitized)')
    ax.axvspan(250, 450, color='grey', alpha=.035)
    ax.set(xlabel='Pressure (GPa)', ylabel='Reflectivity at 532 nm (%)', ylim=(0, 75),
           title='532 nm: mixture scenario and literature context')
    ax.text(.02, .02, 'Dashed lines: frozen endpoint scenario for 7000 K, not pressure predictions.\nMillot: changing shock T; calculated P interpolated from EOS tables.\nOrange: approximate figure readings (±4 GPa, ±1.5 percentage points).',
            transform=ax.transAxes, fontsize=9, va='bottom', bbox=dict(facecolor='white', alpha=.85, edgecolor='.8'))
    ax.legend(loc='upper center', ncol=2, fontsize=8); ax.set_xlim(55, 455); ax.grid(alpha=.2); fig.tight_layout()
    for ext in ('png', 'pdf'): fig.savefig(str(out)+'.'+ext, dpi=180)
    plt.close(fig)
    with open(str(out)+'.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['pressure_GPa', 'composition', 'added_H2_formula_fraction', 'R_H_140GPa_fit', 'R_H_170GPa_fit']); w.writerows(rows)
    print(f'Wrote {out}.png/.pdf/.csv; frozen sensitivity assumptions, not pressure isotherms.')
    print(f'H endpoints at n0=1.52: {100*hs[0]:.2f}%, {100*hs[1]:.2f}%; H2O snapshot {100*water:.3f}%.')


if __name__ == '__main__': main()

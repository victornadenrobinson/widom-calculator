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
    p.add_argument('--dft-csv', default='data/final_model/optics/dft_md_reflectivity_20261112.csv')
    p.add_argument('--french-csv', default=None, help='approximate digitized Fig. 3 CSV')
    a = p.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    water = (16.493 + 13.595 + 13.148) / 300.0
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
    # Faint points are frames; opaque points are arithmetic state means.
    ax.scatter([264.8,264.8,264.8,363.44,363.44,363.44], [11.624,9.471,7.801,16.493,13.595,13.148],
               marker='o', color='black', alpha=.23, s=42)
    ax.scatter([373.05,373.05,373.05], [16.715,18.737,18.523], marker='D', color='black', alpha=.23, s=42)
    ax.errorbar([264.8,363.44], [np.mean([11.624,9.471,7.801]),np.mean([16.493,13.595,13.148])],
               yerr=[np.std([11.624,9.471,7.801],ddof=1),np.std([16.493,13.595,13.148],ddof=1)], fmt='none', color='black', capsize=3)
    ax.scatter([264.8,363.44], [np.mean([11.624,9.471,7.801]),np.mean([16.493,13.595,13.148])],
               marker='o', color='black', s=45, label='Our H2O: mean ± 1 SD, 3 PBE snapshots')
    ax.errorbar([373.05], [np.mean([16.715,18.737,18.523])], yerr=[np.std([16.715,18.737,18.523],ddof=1)], fmt='none', color='black', capsize=3)
    ax.scatter([373.05], [np.mean([16.715,18.737,18.523])], marker='D', color='black', s=42,
               label='Our H3O: mean ± 1 SD, 3 PBE snapshots')
    ax.scatter([314.27,427.14], [56.858,60.797], color='black', marker='*', s=80, label='Our pure H: one PBE snapshot each')
    # DFT-MD snapshots: show only 7000 K here; pressure is the DFT-MD source mean.
    with open(a.dft_csv, newline='') as f:
        dft = list(csv.DictReader(f))
    for comp, marker in [('H','X'),('H2.25O','v'),('H5.024O','h'),('H7O','>')]:
        subset = [r for r in dft if r['composition']==comp and float(r['temperature_K'])==7000]
        if not subset: continue
        ps = [float(r['pressure_GPa']) for r in subset]
        vals = [float(r['density_R_percent']) for r in subset]
        ax.scatter(ps, vals, marker=marker, color='teal', alpha=.25, s=48)
        mean = np.mean(vals); sd = np.std(vals,ddof=1) if len(vals)>1 else None
        ax.errorbar(np.mean(ps), mean, yerr=sd, fmt=marker, color='teal',
                    markersize=5, capsize=2, label=f'{comp}: PBE DFT-MD/optics (7000 K)')
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
    if a.french_csv:
        with open(a.french_csv, newline='') as f:
            french = list(csv.DictReader(f))
        for series, marker, color, label in [
            ('PBE_MD_HSE_optics', 's', 'royalblue', 'French H2O: PBE MD / HSE optics'),
            ('HSE_MD_HSE_optics', 'D', 'deepskyblue', 'French H2O: HSE MD / HSE optics')]:
            subset = [r for r in french if r['series'] == series]
            ax.errorbar([float(r['pressure_GPa']) for r in subset],
                        [float(r['R_percent']) for r in subset],
                        xerr=[float(r['reading_P_GPa']) for r in subset],
                        yerr=[float(r['reading_R_percentage_points']) for r in subset],
                        fmt=marker, ms=4, color=color, capsize=1, elinewidth=.6,
                        label=label + ' (digitized; n0~1.33)')
    ax.axvspan(250, 450, color='grey', alpha=.035)
    ax.set(xlabel='Pressure (GPa)', ylabel='Reflectivity at 532 nm (%)', ylim=(0, 75),
           title='532 nm: mixture scenario and literature context')
    fig.text(.09, .015, 'Dashed lines: frozen endpoints; H2O endpoint = 363 GPa three-frame mean.\nMillot: changing shock T; calculated P interpolated from EOS tables.\nOrange and blue markers: approximate figure readings, not reported statistical errors.\nFrench: liquid-water interface ~1.33; other scenarios n0=1.52 unless noted.',
            fontsize=8, va='bottom')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5,.11), ncol=3, fontsize=6.5)
    ax.set_xlim(55, 455); ax.grid(alpha=.2); fig.tight_layout(rect=(0, .35, 1, 1))
    for ext in ('png', 'pdf'): fig.savefig(str(out)+'.'+ext, dpi=180)
    plt.close(fig)
    with open(str(out)+'.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['pressure_GPa', 'composition', 'added_H2_formula_fraction', 'R_H_140GPa_fit', 'R_H_170GPa_fit']); w.writerows(rows)
    print(f'Wrote {out}.png/.pdf/.csv; frozen sensitivity assumptions, not pressure isotherms.')
    print(f'H endpoints at n0=1.52: {100*hs[0]:.2f}%, {100*hs[1]:.2f}%; H2O 363 GPa three-frame mean {100*water:.3f}%.')


if __name__ == '__main__': main()

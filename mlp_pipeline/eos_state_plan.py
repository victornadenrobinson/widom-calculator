"""Plot a planning grid and actual EOS points; export model energy summaries.

No submission or implicit completion inference: measured pressures need not equal
planned pressures. Re-run with --root runs/eos_live to include local results.
"""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
COMPS = ('h2o', 'h2.25', 'h3o', 'h5o', 'h7o', 'h')
PRESSURES = (10, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500)
BOUND_P = (10, 100, 150, 200, 300, 400, 500)
BOUND_T = (1000, 1900, 3000, 3700, 4800, 5900, 6700)
KB = 8.617330337217213e-5  # ASE kB, eV/K (CODATA 2014)


def read_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', action='append', default=[], help='local eos_live root; can repeat')
    ap.add_argument('--out', default='runs/plots/eos_state_plan')
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    plan = [dict(composition=c, T_target_K=T, P_target_GPa=P, phase_requested='liquid',
                 priority='paper_window' if 300 <= P <= 400 and 5000 <= T <= 7000 else 'coverage',
                 status='target')
            for c in COMPS for T in range(1000, 9001, 1000) for P in PRESSURES
            if T >= np.interp(P, BOUND_P, BOUND_T)]
    write_rows(Path(str(out)+'_targets.csv'), plan)
    reported = read_rows(ROOT/'data/final_model/perlmutter_eos_7000K_reported.csv')
    observations = [dict(c=r['composition'], P=float(r['P_GPa_rounded']),
                         T=float(r['T_set_K']), structure=r['O_struct']) for r in reported]
    young = ROOT/'data/final_model/young_eos_7000K_reported.csv'
    if young.exists():
        observations.extend(dict(c=r['composition'], P=float(r['P_GPa_rounded']),
                                 T=float(r['T_set_K']), structure=r['O_struct'])
                            for r in read_rows(young))
    energies = []
    for root in a.root:
        for c in COMPS:
            for d in sorted(Path(root).glob(f'ft_mfinkc_cont_{c}_*K_*at')):
                f = d/(d.name+'.csv')
                if not f.exists(): continue
                for r in read_rows(f):
                    try:
                        P,T = float(r['P_GPa']), float(r['T_set'])
                        if not math.isfinite(P+T): continue
                        observations.append(dict(c=c,P=P,T=T,structure=r.get('O_struct','')))
                        # The MD writer defines T = 2 K / [3 (N-1) kB].
                        # This recovers mean nuclear K from the same mean T.
                        N, temp, epot = float(r['N']), float(r['T_mean']), float(r['epot_per_atom'])
                        if N <= 1 or not math.isfinite(N+temp+epot): continue
                        kinetic = 1.5*(N-1)/N*KB*temp
                        energies.append(dict(composition=c,T_set_K=T,P_GPa=P,
                            epot_eV_atom=epot,ekin_eV_atom=kinetic,
                            epot_plus_nuclear_kinetic_eV_atom=epot+kinetic,source=str(f)))
                    except (KeyError,ValueError,TypeError): continue
    fig, axes = plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True,layout='constrained')
    for ax,c in zip(axes.flat,COMPS):
        ax.fill_between(BOUND_P,BOUND_T,9000,color='aliceblue',zorder=0)
        targets=[r for r in plan if r['composition']==c]
        ax.scatter([r['P_target_GPa'] for r in targets],[r['T_target_K'] for r in targets],
                   s=32,facecolors='none',edgecolors='0.65',label='Planned target')
        rows=[r for r in observations if r['c']==c]
        normal=[r for r in rows if r['structure'] not in ('bcc','fcc','hcp')]
        ordered=[r for r in rows if r['structure'] in ('bcc','fcc','hcp')]
        ax.scatter([r['P'] for r in normal],[r['T'] for r in normal],s=24,color='tab:blue',label='Reported EOS point',zorder=3)
        ax.scatter([r['P'] for r in ordered],[r['T'] for r in ordered],s=45,marker='x',color='firebrick',label='Ordered O reported',zorder=4)
        ax.set(title=c.upper(),xlim=(0,520),ylim=(500,9500),xlabel='Pressure (GPa)',ylabel='Temperature (K)')
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,loc='lower right')
    fig.suptitle('Liquid EOS state-point plan: all six compositions\nFilled points = available reported data, not certification of liquid phase; live Young results require --root')
    for ext in ('.png','.pdf'): fig.savefig(str(out)+ext,dpi=160)
    if energies:
        write_rows(Path(str(out)+'_energies.csv'),energies)
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        for ax,c in zip(axes.flat,COMPS):
            for T in sorted({r['T_set_K'] for r in energies if r['composition']==c}):
                rows=sorted((r for r in energies if r['composition']==c and r['T_set_K']==T),key=lambda r:r['P_GPa'])
                ax.scatter([r['P_GPa'] for r in rows],[r['epot_plus_nuclear_kinetic_eV_atom'] for r in rows],label=f'{T:g} K')
            ax.set(title=c.upper(),xlabel='Pressure (GPa)',ylabel='MLP potential + nuclear kinetic (eV/atom)');ax.grid(alpha=.15)
            if ax.collections: ax.legend(fontsize=8)
        fig.suptitle('Recorded model energy; electronic entropy corrections not included; phase unscreened')
        fig.savefig(str(out)+'_energy.png',dpi=160)
    print(f'{len(plan)} targets written to {out}_targets.csv; {len(observations)} reported rows loaded.')
    print(f'Wrote {out}.png and .pdf; energy rows: {len(energies)} (requires full local EOS CSVs).')

if __name__=='__main__': main()

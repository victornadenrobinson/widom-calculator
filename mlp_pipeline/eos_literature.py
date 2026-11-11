"""Literature context for EOS P–T plots; no extrapolated phase boundaries."""
import csv
from pathlib import Path

DATA=Path(__file__).resolve().parents[1]/'docs/data'

def rows(name):
    with (DATA/name).open() as f:
        return list(csv.DictReader(line for line in f if not line.startswith('#')))

def overlay(ax, composition):
    if composition == 'h2o':
        d=rows('cheng2021_tm_fcc.csv')
        for key,label,style in [('T_K','Cheng 2021 fcc melt (classical)','-'),('T_K_nqe','Cheng 2021 fcc melt (NQE)','--')]:
            x,y=[],[]
            for row in d:
                try:x.append(float(row['P_GPa']));y.append(float(row[key]))
                except (ValueError,KeyError):continue
            ax.plot(x,y,style,color='darkorange',lw=1.5,label=label,zorder=5)
    if composition == 'h3o':
        # Approximate pressure at 5000 K, not a measured coexistence pressure.
        ax.scatter([540],[5250],marker='*',s=75,color='purple',label='Huang 2020 heating (~540 GPa)',zorder=6)
    if composition == 'h7o':
        d=rows('gupta2024_critical_curve.csv')
        ax.plot([float(r['P_GPa']) for r in d],[float(r['Tc_K']) for r in d],
                color='purple',ls=':',lw=1.7,label='Gupta 2024 critical locus (x varies)',zorder=5)

def dft_points(ax, composition):
    """Plot actual P,T entries from the project's DFT EOS table."""
    table=DATA.parents[1]/'data-h3o-convex-hull.csv'
    with table.open(newline='') as f:
        reader=csv.DictReader(f)
        fields=reader.fieldnames or []
        cols=[(f'P {T}K {composition}',T) for T in (5000,7000,9000) if f'P {T}K {composition}' in fields]
        points=[]
        for row in reader:
            for col,T in cols:
                try:
                    p=float(row[col])
                    if p>0:points.append((p,T))
                except (ValueError,TypeError):pass
    if points:
        ax.scatter([p for p,t in points],[t for p,t in points],marker='s',s=24,
                   facecolors='none',edgecolors='darkgreen',linewidths=.9,
                   label='Project DFT EOS',zorder=5)

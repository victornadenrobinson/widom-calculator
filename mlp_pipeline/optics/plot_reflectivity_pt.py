"""Plot 532 nm reflectivity observations versus pressure and temperature.

Millot 2018 SI Tables 1,3,5,6; Rillo 2019 Fig.3 screenshot manual readings;
project snapshot results. Different incident media and shock paths are labelled.
Temperatures absent from a source are never inferred from pressure.
"""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',default='runs/plots/reflectivity_PT')
    a=p.parse_args(); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    def add(label,method,n0,pres,temp,r):
        rows.append(dict(series=label,method=method,n0=n0,pressure_GPa=pres,temperature_K=temp,R_percent=r))
    # Millot SI Table 1: only shots with tabulated temperatures.
    for pressure,temp,r in [(355.8,10514,11),(292.6,8611,5.8),(284.8,8390,5.9),
                            (275.9,7775,4.6),(260,7264,3.1),(236,6510,1.8)]:
        add('Millot experiment','experiment','1.52',pressure,temp,r)
    # Table 3 R and density, pressure linearly interpolated at same T using SI Tables 5/6.
    for temp,rho,r,r0,p0,r1,p1 in [(5000,3.345,.83692,3.25,177.51,3.4,203.10),
           (6000,3.458,1.862,3.4,215.76,3.537,240.76),(7000,3.557,3.27828,3.5,246.97,3.6,265.58),
           (8000,3.637,5.53853,3.6,277.79,3.709,301.21),(10000,3.758,10.23318,3.003,188.15,3.826,351.55)]:
        add('Millot PBE MD / HSE optics','H2O calculation; P interpolated','1.52',p0+(rho-r0)*(p1-p0)/(r1-r0),temp,r)
    add('Our H2O','PBE snapshot frame 19','1.52',363.44,7000,16.493)
    add('Our H2O','PBE snapshot frame 28','1.52',363.44,7000,13.595)
    add('Our H2O','PBE snapshot frame 19','1.52',264.80,7000,11.624)
    add('Our H3O','PBE snapshot','1.52',373.05,7000,16.715)
    for pressure,r in [(70,25),(84,27.5),(100,31.5),(122,38.5),(149,43),(170,45.5),(182,46.8),(199,48.5),(210,49.5),(225,50.5),(280,53)]:
        add('Rillo H 5000 K','HSE optics; screenshot reading; 539 nm','1.49',pressure,5000,r)
    # French Fig. 3 has no per-point temperatures; excluded until sourced.
    styles={'Millot experiment':('o','Millot H₂O experiment (n₀=1.52)'),
            'Millot PBE MD / HSE optics':('s','Millot H₂O: PBE MD / HSE optics (n₀=1.52)'),
            'Our H2O':('D','Our H₂O: PBE optics (n₀=1.52)'),
            'Our H3O':('P','Our H₃O: PBE optics (n₀=1.52)'),
            'Rillo H 5000 K':('^','Rillo H: HSE optics (n₀=1.49; 539 nm)')}
    fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    tn=Normalize(4500,11000); rn=Normalize(0,60)
    for label,(marker,display) in styles.items():
        rr=[r for r in rows if r['series']==label]
        if not rr: continue
        pp=[r['pressure_GPa'] for r in rr];tt=[r['temperature_K'] for r in rr];vv=[r['R_percent'] for r in rr]
        axes[0].scatter(pp,vv,c=tt,cmap='plasma',norm=tn,marker=marker,s=50,edgecolor='black',linewidth=.5,label=display)
        # Multiple snapshots at exactly the same state overlap in P-T space.
        # Show their arithmetic mean there; keep individual rows in output CSV.
        states = sorted(set(zip(pp,tt)))
        means = [np.mean([r['R_percent'] for r in rr if (r['pressure_GPa'],r['temperature_K'])==state]) for state in states]
        axes[1].scatter([x[0] for x in states],[x[1] for x in states],c=means,cmap='viridis',norm=rn,marker=marker,s=50,edgecolor='black',linewidth=.5,label=display)
        for pressure,temp in states:
            count=sum((r['pressure_GPa'],r['temperature_K'])==(pressure,temp) for r in rr)
            if count>1: axes[1].annotate('n='+str(count),(pressure,temp),xytext=(5,7),textcoords='offset points',fontsize=8)
    fig.colorbar(plt.cm.ScalarMappable(norm=tn,cmap='plasma'),ax=axes[0],label='Temperature (K)')
    fig.colorbar(plt.cm.ScalarMappable(norm=rn,cmap='viridis'),ax=axes[1],label='Reflectivity (%)')
    axes[0].set(xlabel='Pressure (GPa)',ylabel='Reflectivity (%)',title='Reflectivity and temperature')
    axes[1].set(xlabel='Pressure (GPa)',ylabel='Temperature (K)',title='Reflectivity in P–T space')
    axes[1].axvspan(300,400, ymin=(5000-4000)/(11500-4000),ymax=(7000-4000)/(11500-4000),color='red',alpha=.08)
    axes[1].set_ylim(4000,11500)
    for ax in axes: ax.set_xlim(50,430);ax.grid(alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,fontsize=9)
    fig.suptitle('H₂O and H reflectivity: different media and thermodynamic paths')
    for ext in ('png','pdf'):fig.savefig(str(out)+'.'+ext,dpi=180)
    plt.close(fig)
    with open(str(out)+'.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(f'Wrote {out}.png/.pdf/.csv; {len(rows)} points with stated temperatures.')


if __name__=='__main__': main()

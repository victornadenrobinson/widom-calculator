"""Exploratory H2O + H2 reflectivity mixing at 7000 K near 350 GPa.

Uses state-mean PBE 532 nm density-response results; source trajectories have
slightly different pressures and MD methods. Outputs descriptive errors, not
statistical model selection or validated pressure corrections.
"""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Finite PBE optical data reported in the project conversation. Pressure in
# GPa is a trajectory mean, reflectivity is percent, at 532 nm and n0=1.52.
WATER=[16.493,13.595,13.148]
WATER_LOW=[11.624,9.471,7.801]
H3O=[16.715,18.737,18.523]
COMP={'H2.25O':2.25,'H3O':3.,'H5.024O':5.024,'H7O':7.}


def endpoint_volume(path,composition,target=350):
    pairs=[]
    with Path(path).open(newline='') as f:
        for r in csv.DictReader(f):
            try:
                p=float(r[f'P 7000K {composition}']);v=float(r[f'V 7000K {composition} norm'])
                if math.isfinite(p) and math.isfinite(v) and p>0 and v>0:pairs.append((p,v))
            except (KeyError,ValueError,TypeError):pass
    pairs.sort()
    if len(pairs)<2 or target<pairs[0][0] or target>pairs[-1][0]:
        raise ValueError(f'No EOS pressure bracket for {composition} at {target:g} GPa')
    return float(np.interp(target,*zip(*pairs))),pairs


def mean_sd(vals):return float(np.mean(vals)),float(np.std(vals,ddof=1)) if len(vals)>1 else float('nan')


def fractions(h,water_volume,h_volume):
    delta=np.asarray(h,dtype=float)-2
    return {'formula-unit':delta/(delta+2),
            'ideal volume':delta*h_volume/(water_volume+delta*h_volume),
            'valence-electron count':delta/(delta+8),
            'added-H mass':delta*1.008/(18.015+delta*1.008)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',default='data/final_model/optics/dft_md_reflectivity_20261112.csv')
    ap.add_argument('--eos',default='data-h3o-convex-hull.csv')
    ap.add_argument('--out',default='runs/plots/reflectivity_mixing_fit_7000K');a=ap.parse_args()
    grouped=defaultdict(list);pressures=defaultdict(set)
    with Path(a.csv).open(newline='') as f:
        for row in csv.DictReader(f):
            if float(row['temperature_K'])!=7000:continue
            key=row['composition'];grouped[key].append(float(row['density_R_percent']));pressures[key].add(float(row['pressure_GPa']))
    for c in ('H','H2.25O','H5.024O','H7O'):
        if len(pressures[c])!=1 or len(grouped[c])<2:raise SystemExit(f'{c}: need one pressure group with at least two frames')
    grouped['H3O']=H3O;pressures['H3O']={373.05}
    rw,sdw=mean_sd(WATER);rh,sdh=mean_sd(grouped['H']);vw,_=endpoint_volume(a.eos,'h2o');vh,_=endpoint_volume(a.eos,'h')
    labels=list(COMP);h=np.array([COMP[k] for k in labels]);y=np.array([mean_sd(grouped[k])[0] for k in labels]);sd=np.array([mean_sd(grouped[k])[1] for k in labels]);pp=np.array([next(iter(pressures[k])) for k in labels])
    f=fractions(h,vw,vh);f_formula=f['formula-unit'];model={name:rw+(rh-rw)*val for name,val in f.items()}
    alpha_grid=np.linspace(.25,5,10000);errors=np.array([np.mean((rw+(rh-rw)*f_formula**alpha-y)**2) for alpha in alpha_grid]);alpha=float(alpha_grid[np.argmin(errors)])
    model['power fit']=rw+(rh-rw)*f_formula**alpha
    # A partial sensitivity check only: interpolate water between measured GRACE
    # states; approximate H endpoint with two single-frame GRACE states.
    rw350=float(np.interp(350,[264.8,363.44],[np.mean(WATER_LOW),rw]))
    rh350=float(np.interp(350,[314.27,427.14],[56.858,60.797]))
    print(f'7000 K approximate P=350 GPa DFT EOS endpoint V: H2O={vw:.4f} Å³/formula, H={vh:.4f} Å³/atom')
    print(f'Baseline reflectivity endpoints: H2O {rw:.3f}% at 363.44 GPa; H {rh:.3f}% at {next(iter(pressures["H"])):.2f} GPa')
    print(f'Alternative endpoints: H2O {rw350:.3f}% (two-state interpolation), H {rh350:.3f}% (two single-frame GRACE states); mixtures uncorrected')
    for name,pred in model.items():
        print(f'{name:24} RMS = {np.sqrt(np.mean((y-pred)**2)):.3f} percentage points, four intermediate states')
    print(f'power exponent alpha = {alpha:.3f}, unweighted in-sample fit; no formal model selection')
    alt={name:rw350+(rh350-rw350)*val for name,val in f.items()}
    for name,pred in alt.items():print(f'  alternative endpoints, {name:22}: RMS {np.sqrt(np.mean((y-pred)**2)):.3f} points')
    x=np.linspace(0,1,400);h_curve=np.divide(2,1-x,out=np.full_like(x,np.inf),where=x<1);d=h_curve-2
    # Evaluate finite-composition fractions directly to avoid infinity/infinity.
    curves={'formula-unit':x,'ideal volume':(2*x*vh)/(vw*(1-x)+2*x*vh),
            'valence-electron count':(2*x)/(8*(1-x)+2*x),'added-H mass':(2*x*1.008)/(18.015*(1-x)+2*x*1.008)}
    colors={'formula-unit':'#8a559e','ideal volume':'#008c91','valence-electron count':'#d78025','added-H mass':'#6e797e','power fit':'#343434'}
    fig,(ax,ar)=plt.subplots(2,1,figsize=(10,9),height_ratios=(2,1),layout='constrained')
    for name,z in curves.items():ax.plot(x,rw+(rh-rw)*z,label=name,color=colors[name],lw=1.7)
    ax.plot(x,rw+(rh-rw)*x**alpha,'--',color=colors['power fit'],label=f'power fit, α={alpha:.2f}')
    ax.errorbar([0,1],[rw,rh],yerr=[sdw,sdh],fmt='s',color='black',capsize=3,label='endpoint mean ± snapshot SD',zorder=5)
    ax.errorbar(f_formula,y,yerr=sd,fmt='o',color='black',capsize=3,label='intermediate mean ± snapshot SD',zorder=5)
    for xx,yy,l,p in zip(f_formula,y,labels,pp):ax.annotate(f'{l}, {p:.0f} GPa',(xx,yy),xytext=(4,-15),textcoords='offset points',fontsize=8)
    ax.set(ylabel='532 nm reflectivity (%)',title='7000 K: mixing-variable comparison, approximately 350 GPa',ylim=(0,70));ax.grid(alpha=.2)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3,fontsize=8)
    for name,pred in model.items():ar.plot(f_formula,y-pred,marker='o',markersize=3,color=colors[name],label=name,linestyle='--' if name=='power fit' else '-')
    ar.axhline(0,color='black',lw=.8);ar.set(xlabel='Added H₂ formula-unit fraction in H₂O + H₂',ylabel='Observed − model\n(percentage points)');ar.grid(alpha=.2)
    fig.suptitle('Exploratory: unmatched pressures and mixed MD sources; PBE optics, density response, n₀=1.52',fontsize=10)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(str(out)+'.'+ext,dpi=180)
    with Path(str(out)+'.csv').open('w',newline='') as file:
        fields=['composition','pressure_GPa','n','mean_R_percent','sample_SD_pp','formula_fraction']+[name+'_pred_percent' for name in model]
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for i,label in enumerate(labels):
            row=dict(composition=label,pressure_GPa=pp[i],n=len(grouped[label]),mean_R_percent=y[i],sample_SD_pp=sd[i],formula_fraction=f_formula[i]);row.update({name+'_pred_percent':pred[i] for name,pred in model.items()});writer.writerow(row)
    print('Wrote',str(out)+'.png/.pdf/.csv')
    print('Caveat: pressure differs between states, single-segment frames may be correlated, phase/source functional remain to be checked.')

if __name__=='__main__':main()

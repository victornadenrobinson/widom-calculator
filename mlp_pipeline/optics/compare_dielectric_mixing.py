"""Compare saved 532-nm response components with end-member dielectric mixing.

Uses completed per-snapshot optical CSVs. The reference H2O GRACE-MD and pure-H
DFT-MD states are near, not exactly at, 350 GPa. Phase and convergence are not
certified. The dielectric interpolation is a model, not a phase claim.
"""
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_mixing_reflectivity import endpoint_volume

HC=1239.841984
E=HC/532
FACTOR=8.8541878128e-12/6.582119569e-16/100

def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.digest()

def valid_reuse(case,info,reference):
    if not reference:return False
    try:
        old=json.loads((reference/'provenance.json').read_text())
        if not info.get('trajectory') or (info['trajectory'],info['frame_index_zero_based']) != (old['trajectory'],old['frame_index_zero_based']):return False
        for name in ('INCAR','POSCAR','POTCAR','KPOINTS'):
            if hashfile(case/name)!=hashfile(reference/name):return False
        out=reference/'OUTCAR'
        with out.open('rb') as f:
            head=f.read(8*1024*1024);f.seek(max(0,out.stat().st_size-4*1024*1024));tail=f.read()
        return b'aborting loop because EDIFF is reached' in head+tail and b'General timing and accounting' in tail and (reference/'vasprun.xml').is_file()
    except (OSError,KeyError,ValueError):return False

def load_spectrum(path):
    with path.open(newline='') as f:rows=list(csv.DictReader(f))
    e=np.array([float(r['energy_eV']) for r in rows]);a=np.array([float(r['epsilon_real_mean']) for r in rows]);b=np.array([float(r['epsilon_imag_mean']) for r in rows])
    if len(e)<2 or np.any(np.diff(e)<=0) or not all(np.isfinite(x).all() for x in (e,a,b)) or not e[0]<=E<=e[-1]:raise ValueError(f'invalid spectral grid {path}')
    return e,a+1j*b

def components(z,n0):
    n=np.sqrt(complex(z))
    if n.imag < 0:n=-n
    r=abs((n-n0)/(n+n0))**2*100
    return np.array([z.real,z.imag,n.real,n.imag,FACTOR*E*z.imag,FACTOR*E*(1-z.real),r])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('roots',nargs='+',type=Path);p.add_argument('--reuse',type=Path);p.add_argument('--eos',default='data-h3o-convex-hull.csv');p.add_argument('--out',required=True);p.add_argument('--tensor',choices=['density','current'],default='density');a=p.parse_args()
    grouped=defaultdict(list);seen=set()
    for root in a.roots:
        if not root.is_dir():print('Missing root:',root);continue
        for case in sorted(root.glob('h*o_P*_frame*')):
            try:
                info=json.loads((case/'provenance.json').read_text());t=float(info['temperature_set_K']);pressure=float(info['pressure_mean_GPa']);comp=info['composition']
                if abs(t-7000)>250 or not 330<=pressure<=385:continue
                trajectory=info['trajectory'];frame=int(info['frame_index_zero_based']);unique=(trajectory,frame)
                if unique in seen:print('Skipping duplicate',case);continue
                done=sorted((case/'attempts').glob('*/DONE.json'))
                if done:
                    json.loads(done[-1].read_text());work=done[-1].parent
                elif valid_reuse(case,info,a.reuse):work=a.reuse
                else:print('Pending or unverified:',case);continue
                energies,spectrum=load_spectrum(work/f'reflectivity_{a.tensor}.csv')
                z=complex(float(np.interp(E,energies,spectrum.real)),float(np.interp(E,energies,spectrum.imag)))
                nH=int(info['nH']);nO=int(info['nO']);x=1.0 if nO==0 else (nH/nO-2)/(nH/nO)
                if not math.isfinite(x) or x < -1e-8 or x > 1:raise ValueError('invalid composition')
                source=info.get('source_kind','GRACE EOS')
                key=(comp,round(pressure,2),source,trajectory,nH,nO)
                grouped[key].append(dict(z=z,x=x,case=str(case),frame=frame,work=str(work),energies=energies,spectrum=spectrum))
                seen.add(unique)
            except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as e:print(f'Cannot use {case}: {e}')
    bycomp=defaultdict(list)
    for key,items in grouped.items():bycomp[key[0]].append((key,items))
    for comp,groups in bycomp.items():
        if len(groups)>1:raise SystemExit(f'Ambiguous groups for {comp}: {[x[0] for x in groups]}; choose narrower roots or pressure range')
    required=['H2O','H2.25O','H3O','H5.024O','H7O','H']
    if 'H2O' not in bycomp or 'H' not in bycomp:raise SystemExit('Need completed H2O and H endpoint spectra; review missing/pending cases above')
    for comp in required:
        if comp not in bycomp:print('No completed state:',comp)
    data=[]
    for comp in required:
        if comp not in bycomp:continue
        key,items=bycomp[comp][0];arr=np.array([components(v['z'],1.52) for v in items]);z=np.array([v['z'] for v in items]);mean=arr.mean(axis=0);sd=arr.std(axis=0,ddof=1) if len(items)>1 else np.full(7,np.nan)
        data.append(dict(comp=comp,P=key[1],source=key[2],trajectory=key[3],n=len(items),x=items[0]['x'],eps=z.mean(),mean=mean,sd=sd,items=items))
    water=next(x for x in data if x['comp']=='H2O');hydrogen=next(x for x in data if x['comp']=='H')
    vw,_=endpoint_volume(a.eos,'h2o');vh,_=endpoint_volume(a.eos,'h')
    grid=np.linspace(0,1,300);phi=2*grid*vh/(vw*(1-grid)+2*grid*vh)
    zvol=(1-phi)*water['eps']+phi*hydrogen['eps'];zf=(1-grid)*water['eps']+grid*hydrogen['eps']
    model=np.array([components(v,1.52) for v in zvol]);model_formula=np.array([components(v,1.52) for v in zf]);r_linear=(1-phi)*water['mean'][6]+phi*hydrogen['mean'][6]
    fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained');axes=axes.ravel();names=['ε₁','ε₂','n','k','σ₁ (S/cm)','σ₂ (S/cm)','Reflectivity (%)'];colors=plt.cm.viridis(np.linspace(0,1,len(data)))
    for i,ax in enumerate(axes[:7]):
        ax.plot(grid,model[:,i],color='#006f78',label='linear ε, ideal volume')
        ax.plot(grid,model_formula[:,i],color='#a35d16',ls='--',label='linear ε, formula fraction')
        if i==6:ax.plot(grid,r_linear,':',color='grey',label='linear R, ideal volume')
        for color,record in zip(colors,data):
            ax.errorbar(record['x'],record['mean'][i],yerr=record['sd'][i] if np.isfinite(record['sd'][i]) else None,fmt='o',color=color,markeredgecolor='black',markeredgewidth=.4,capsize=2,markersize=5)
        ax.set(xlim=(-.04,1.04),xlabel='Added H₂ formula-unit fraction',ylabel=names[i]);ax.grid(alpha=.2)
    axes[7].axis('off')
    legend=[plt.Line2D([],[],color='#006f78',label='linear ε: ideal volume'),plt.Line2D([],[],color='#a35d16',ls='--',label='linear ε: formula fraction'),plt.Line2D([],[],color='grey',ls=':',label='linear R: ideal volume')]
    legend += [plt.Line2D([],[],marker='o',ls='',color=c,markeredgecolor='black',label=f"{r['comp']} {r['P']:.0f} GPa (n={r['n']})") for c,r in zip(colors,data)]
    axes[7].legend(handles=legend,loc='center left',fontsize=8,frameon=False)
    fig.suptitle(f'532 nm response ingredients: {a.tensor} tensor, 7000 K near 350 GPa; n₀=1.52\nPoints: mean ± snapshot SD; pressures and MD sources differ',fontsize=13)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(str(out)+'.'+ext,dpi=170)
    # Full spectra: interpolate only over the grid common to every snapshot,
    # then average each observable per snapshot (the transforms are nonlinear).
    all_items=[v for record in data for v in record['items']]
    lo=max(.1,max(float(v['energies'][0]) for v in all_items))
    hi=min(5.,min(float(v['energies'][-1]) for v in all_items))
    if lo < hi:
        energy=np.linspace(lo,hi,500)
        fullfig,fullaxes=plt.subplots(2,4,figsize=(17,8),layout='constrained');fullaxes=fullaxes.ravel()
        for color,record in zip(colors,data):
            spectra=[]
            for v in record['items']:
                z=np.interp(energy,v['energies'],v['spectrum'].real)+1j*np.interp(energy,v['energies'],v['spectrum'].imag)
                n=np.sqrt(z);n=np.where(n.imag<0,-n,n)
                spectra.append(np.stack([z.real,z.imag,n.real,n.imag,FACTOR*energy*z.imag,FACTOR*energy*(1-z.real),100*np.abs((n-1.52)/(n+1.52))**2]))
            mean=np.mean(spectra,axis=0)
            for i,ax in enumerate(fullaxes[:7]):ax.plot(energy,mean[i],color=color,label=record['comp'])
        for i,ax in enumerate(fullaxes[:7]):
            ax.set(xlabel='Photon energy (eV)',ylabel=names[i]);ax.grid(alpha=.2);ax.axvline(E,color='grey',lw=.7,ls=':')
        fullaxes[7].axis('off');fullaxes[7].legend(handles=[plt.Line2D([],[],color=c,label=f"{r['comp']} {r['P']:.0f} GPa (n={r['n']})") for c,r in zip(colors,data)],loc='center left',frameon=False)
        fullfig.suptitle(f'Frequency-dependent response, {a.tensor} tensor; 532 nm marked\nState means; source pressures differ and trajectories may be correlated',fontsize=13)
        for ext in ('png','pdf'):fullfig.savefig(str(out)+'_spectra.'+ext,dpi=170)
        print(f'Wrote {out}_spectra.png/.pdf (common range {lo:.2f}–{hi:.2f} eV).')
    else:print('No common spectral range for frequency-dependent plot.')
    columns=['composition','pressure_GPa','source','trajectory','n_frames','added_H2_formula_fraction','epsilon1','epsilon2','n','k','sigma1_S_cm','sigma2_S_cm','R_percent']
    with Path(str(out)+'_states.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(columns)
        for r in data:w.writerow([r['comp'],r['P'],r['source'],r['trajectory'],r['n'],r['x'],*r['mean']])
    with Path(str(out)+'_frames.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['composition','pressure_GPa','trajectory','frame','case','work','epsilon1','epsilon2','n','k','sigma1_S_cm','sigma2_S_cm','R_percent'])
        for r in data:
            for v in r['items']:w.writerow([r['comp'],r['P'],r['trajectory'],v['frame'],v['case'],v['work'],*components(v['z'],1.52)])
    print(f'Using EOS endpoint volumes at 350 GPa: H2O {vw:.5f} Å³/f.u.; H {vh:.5f} Å³/atom')
    for r in data:print(f"{r['comp']:8} P={r['P']:7.2f} GPa n={r['n']} ε={r['eps'].real:.3f}+{r['eps'].imag:.3f}i R={r['mean'][6]:.3f}% source={r['source']}")
    print('Wrote',str(out)+'.png/.pdf, _states.csv and _frames.csv')
    print('σ is calculated from epsilon and is not an independent conductivity computation; no DC inference.')
    print('R is averaged per snapshot; curves use the mean complex endpoint dielectric values. Pressures are not matched.')

if __name__=='__main__':main()

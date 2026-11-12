"""Compare completed 532 nm snapshot reflectivities across H–O composition."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COMPS={'H2O':(0,'H₂O'),'H2.25O':(1-2/2.25,'H₂.₂₅O'),'H3O':(1-2/3,'H₃O'),'H5.024O':(1-2/5.024,'H₅.₀₂₄O'),'H7O':(1-2/7,'H₇O'),'H':(1,'H')}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dft-csv',default='data/final_model/optics/dft_md_reflectivity_20261112.csv');p.add_argument('--out',default='runs/plots/reflection_composition');a=p.parse_args()
 with open(a.dft_csv,newline='') as f: data=list(csv.DictReader(f))
 groups={}
 for r in data:
  key=(int(r['temperature_K']),r['composition'],'DFT-MD',float(r['pressure_GPa']))
  groups.setdefault(key,[]).append(float(r['density_R_percent']))
 for key,values in [((7000,'H2O','GRACE-MD',363.44),[16.493,13.595,13.148]),((7000,'H3O','GRACE-MD',373.05),[16.715,18.737,18.523])]:groups[key]=values
 fig,axes=plt.subplots(1,2,figsize=(12.5,5.5),sharey=True,layout='constrained')
 from matplotlib.lines import Line2D
 from plot_linear_r_scenario import hydrogen_r
 hs=[100*hydrogen_r(20.4,.13,1.52),100*hydrogen_r(21.8,.16,1.52)]
 water=np.mean(groups[(7000,'H2O','GRACE-MD',363.44)])
 for ax,temp in zip(axes,(5000,7000)):
  if temp==7000:
   x=np.linspace(0,1,100);lo=water+(hs[0]-water)*x;hi=water+(hs[1]-water)*x
   ax.fill_between(x,lo,hi,color='grey',alpha=.15);ax.plot(x,(lo+hi)/2,'--',color='grey',lw=1)
  for (T,comp,source,P),values in groups.items():
   if T!=temp:continue
   x,label=COMPS[comp];marker='o' if source=='DFT-MD' else 'D';color='#157b88' if source=='DFT-MD' else '#a03b67'
   mean=np.mean(values);sd=np.std(values,ddof=1) if len(values)>1 else 0
   ax.scatter([x]*len(values),values,c=color,marker=marker,s=55,alpha=.25,zorder=2)
   ax.errorbar(x,mean,yerr=sd,fmt=marker,color=color,markersize=7,capsize=3,lw=1.2,zorder=3)
   ax.annotate(f'{label}  {P:.0f} GPa', (x,mean), xytext=(0,9 if comp!='H5.024O' else -17),textcoords='offset points',ha='center',fontsize=8)
  ax.set(xlim=(-.08,1.08),ylim=(0,70),title=f'{temp} K',xlabel='Added H₂ formula-unit fraction  (H₂O + H₂)')
  ax.grid(alpha=.2)
 axes[0].set_ylabel('Reflectivity at 532 nm (%)')
 fig.legend(handles=[Line2D([],[],color='#157b88',marker='o',linestyle='none',label='PBE DFT-MD / PBE optics'),Line2D([],[],color='#a03b67',marker='D',linestyle='none',label='GRACE-MD / PBE optics'),Line2D([],[],color='grey',linestyle='--',label='7000 K frozen-endpoint scenario')],loc='lower center',bbox_to_anchor=(.5,-.07),ncol=3,fontsize=9)
 fig.suptitle('H–O reflectivity near 350 GPa | n₀ = 1.52 | density response\nFaint: snapshots; solid: mean ± sample SD (n = 2–3). Pressure labels are trajectory means.', fontsize=12)
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 for ext in ('png','pdf'):fig.savefig(str(out)+'.'+ext,dpi=180,bbox_inches='tight')
 print(f'Wrote {out}.png/.pdf; {len(groups)} state groups.')
if __name__=='__main__':main()

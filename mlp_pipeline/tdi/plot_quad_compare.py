"""Compare saved Gauss/Kronrod G(t) without MD, trajectory reads or reference endpoints."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tdi import analyze
from tdi.gcalc import G_from_nodes


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directories',nargs='+');ap.add_argument('--out',required=True);a=ap.parse_args()
    data=[];seen=set()
    for d in a.directories:
        for f in Path(d).glob('*_stage0.json'):
            import json
            st=json.loads(f.read_text());A=st['args'];tag=f.name[:-12]
            if st['phase']!='liquid':continue
            identity=(st['T'],st['P_target'],st['N'],A['model'],A.get('hper',2),st['V'])
            seen.add(identity);lam,w,sub=analyze.rule_nodes(A);spp=1000/A['dt']/A['every']
            for name,ll,ww in [(('GK' if A.get('rule')=='gk' else 'GL')+str(len(lam)),lam,w)]+([(f'GL{len(sub[0])}',sub[0],sub[1])] if sub else []):
                ser=[]
                for k in range(len(ll)):
                    p=f.parent/f'{tag}_node{k}.npy';ser.append(np.load(p) if p.exists() else np.array([]))
                n=min(map(len,ser))
                if n<25:print(f'{d}: {name} skipped (missing/too-short lambda series)');continue
                data.append(dict(label=name,st=st,lam=ll,w=ww,series=ser,n=n,spp=spp,source=str(f.parent),power=A.get('power',4)))
    if len(seen)>1:raise SystemExit('Runs differ in model path, P/T, atom count, composition or volume. Select consistent runs.')
    if not data:raise SystemExit('No complete-enough quadrature series found.')
    cutoff=min(r['n']/r['spp'] for r in data)
    fig,(ax,bx)=plt.subplots(2,1,figsize=(11,8),layout='constrained');rows=[]
    for r in data:
        ns=np.unique(np.linspace(25,r['n'],min(60,r['n']-24)).astype(int));vals=np.array([G_from_nodes(r['st'],r['lam'],r['w'],[s[:n] for s in r['series']],r['st']['N'])[:2] for n in ns]);t=ns/r['spp']
        ln=ax.plot(t,vals[:,0],label=f"{r['label']} ({r['source']})")[0];ax.fill_between(t,vals[:,0]-vals[:,1],vals[:,0]+vals[:,1],color=ln.get_color(),alpha=.1)
        n=int(round(cutoff*r['spp']));G,err,*_=G_from_nodes(r['st'],r['lam'],r['w'],[s[:n] for s in r['series']],r['st']['N']);rows.append(dict(rule=r['label'],source=r['source'],common_ps=cutoff,G_meV_atom=G,stat_sem_meV_atom=err))
    rows.sort(key=lambda r:(int(r['rule'][2:]),r['rule']))
    y=np.arange(len(rows));bx.errorbar([r['G_meV_atom'] for r in rows],y,xerr=[r['stat_sem_meV_atom'] for r in rows],fmt='o');bx.set_yticks(y,[r['rule'] for r in rows]);bx.invert_yaxis()
    ax.set(xlabel='Production ps per lambda',ylabel='Cumulative G (meV/atom)',title='Convergence — shaded bands are statistical SEM');ax.legend(fontsize=8)
    bx.set(xlabel='G (meV/atom)',title=f'Comparison at common cutoff {cutoff:g} ps')
    st=data[0]['st'];fig.suptitle(f"{st['N']} atoms, {st['P_target']:g} GPa, {st['T']:g} K | phase: {st['phase']}\nSame-model paths only; weights and phase not independently verified. Prefix estimates are correlated.",fontsize=10)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(str(out)+'.'+ext,dpi=160)
    with Path(str(out)+'.csv').open('w') as fh:
        wr=csv.DictWriter(fh,fieldnames=rows[0]);wr.writeheader();wr.writerows(rows)
    # Individual quadrature contributions cannot be compared node-by-node across
    # rules as if their weights or lambda locations were the same.
    families=[r for r in data if r['label'].startswith('GK')]
    if families:
        fig, axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
        node_rows=[]
        for r in families:
            n=int(round(cutoff*r['spp'])); scale=1000/r['st']['N']
            x=np.asarray(r['lam'])**(1/float(r['power']))
            means=np.array([np.mean(s[:n])*scale for s in r['series']])
            first=np.array([np.mean(s[:n//2])*scale for s in r['series']])
            second=np.array([np.mean(s[n//2:n])*scale for s in r['series']])
            order=np.argsort(x)
            line=axes[0,0].plot(x[order],means[order],'.-',label=r['label'])[0]
            color=line.get_color()
            axes[0,1].scatter(x,r['w'],color=color,label=r['label'],s=24)
            axes[1,0].scatter(x,r['w']*means,color=color,label=r['label'],s=24)
            axes[1,1].scatter(x,r['w']*(second-first),color=color,label=r['label'],s=24)
            for k in range(len(x)):
                node_rows.append(dict(rule=r['label'],source=r['source'],node=k,lambda_value=r['lam'][k],x=x[k],weight=r['w'][k],mean_meV_atom=means[k],weighted_contribution_meV_atom=r['w'][k]*means[k],weighted_second_minus_first_meV_atom=r['w'][k]*(second[k]-first[k]),common_ps=cutoff))
        axes[0,0].set(ylabel='Mean integrand (meV/atom)',title='Integrand')
        axes[0,1].set(ylabel='Weight (including Jacobian)',title='Quadrature weights')
        axes[1,0].set(ylabel='Weight × mean (meV/atom)',title='Individual contributions to integral')
        axes[1,1].set(ylabel='Weight × half-to-half shift (meV/atom)',title='Second half minus first half')
        for ax in axes.flat:
            ax.set_xlabel('Quadrature variable x = λ^(1/power)')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        for ax in axes[1]: ax.axhline(0,color='black',lw=.6)
        fig.suptitle(f'{st["N"]} atoms, {st["P_target"]:g} GPa, {st["T"]:g} K | first {cutoff:g} ps/node\nContributions sum within each rule; different rules have different nodes and weights',fontsize=10)
        for ext in ('png','pdf'):fig.savefig(str(out)+'_weights.'+ext,dpi=160)
        plt.close(fig)
        with Path(str(out)+'_weights.csv').open('w',newline='') as fh:
            wr=csv.DictWriter(fh,fieldnames=node_rows[0]);wr.writeheader();wr.writerows(node_rows)
        print(f'Wrote {out}_weights.png, .pdf, .csv')
    print(f'Wrote {out}.png, .pdf, .csv. Common cutoff {cutoff:g} ps.')

if __name__=='__main__':main()

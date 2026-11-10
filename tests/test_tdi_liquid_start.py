"""Liquid cell size and restart identity must agree with the chosen initial lattice."""
import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mlp_pipeline'))
from tdi.run_real import make_parser,setup,check_stage0_compatible


def test_liquid_128_molecules_and_restart_guard(tmp_path):
    args=make_parser().parse_args(['toyice','--phase','liquid','--liquid-start','bcc',
        '--nrep','4','--vpa','2.52','--T','7000','--P','350','--outdir',str(tmp_path)])
    ctx=setup(args);at=ctx['at0']
    assert len(at)==384 and (at.numbers==8).sum()==128
    assert np.isclose(at.get_volume()/len(at),2.52)
    assert ctx['tag'].endswith('_rep4_startbcc')
    saved=vars(args).copy();saved['liquid_start']='fcc'
    (tmp_path/(ctx['tag']+'_stage0.json')).write_text(json.dumps(dict(args=saved)))
    np.save(tmp_path/(ctx['tag']+'_node0.npy'),[0.])
    with pytest.raises(SystemExit,match='liquid_start'):
        check_stage0_compatible(ctx,0.)


def test_default_liquid_tags_unchanged():
    from tdi.run_real import make_tag
    a=make_parser().parse_args(['toyice','--phase','liquid','--P','350','--T','7000','--nrep','2'])
    assert make_tag(a)=='liquid_350GPa_7000K_h2o_rep2'


def test_64_oxygen_cubic_cell_and_separation(tmp_path):
    from ase.geometry import get_distances
    a=make_parser().parse_args(['toyice','--phase','liquid','--liquid-start','sc',
        '--nrep','4','--vpa','2.52','--T','7000','--P','350','--outdir',str(tmp_path)])
    c=setup(a);at=c['at0']
    assert len(at)==192 and (at.numbers==8).sum()==64
    assert np.isclose(at.get_volume()/len(at),2.52)
    assert c['tag'].endswith('_rep4_startsc')
    assert np.allclose(at.cell.lengths(),at.cell.lengths()[0])
    _,dist=get_distances(at.positions,cell=at.cell,pbc=True)
    np.fill_diagonal(dist,np.inf)
    assert dist.min()>0.75

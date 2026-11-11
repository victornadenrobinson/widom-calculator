import csv
import sys
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.io import write
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mlp_pipeline'))
from eos_seed import nearby_liquid
import eos_queue


def table(tmp_path, structure='other'):
    root=tmp_path/'eos_live';d=root/'ft_test_h2o_7000K_3at';d.mkdir(parents=True)
    sub=d/(d.name+'_v9.0000');sub.mkdir()
    atoms=Atoms('OHH',positions=[[0,0,0],[1.5,0,0],[0,1.5,0]],cell=[3,3,3],pbc=True)
    write(sub/'final.extxyz',atoms)
    row=dict(N=3,nO=1,T_set=7000,T_mean=7000,vpa=9,P_GPa=200,dmin_HH=.5,
             D_O_cm2s=.001,D_H_cm2s=.002,O_struct=structure,branch='liquid',
             P_drift_GPa=0,P_block_sd=1,frac_HH_below_0p5=0)
    with (d/(d.name+'.csv')).open('w') as f:
        w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
    return root,sub


def test_nearby_seed_and_source_preserved(tmp_path):
    root,sub=table(tmp_path);before=(sub/'final.extxyz').read_bytes()
    at,meta=nearby_liquid(root,'ft_test',3,1,5000,9.9)
    assert at is not None and np.isclose(at.get_volume()/3,9.9)
    assert meta['method']=='nearby_liquid' and len(meta['source_sha256'])==64
    assert (sub/'final.extxyz').read_bytes()==before


def test_ordered_and_remote_seeds_fall_back(tmp_path):
    root,sub=table(tmp_path,structure='bcc')
    assert nearby_liquid(root,'ft_test',3,1,5000,9)[0] is None
    assert nearby_liquid(root,'other_model',3,1,5000,9)[0] is None


def test_temperature_volume_size_and_self_excluded(tmp_path):
    root,sub=table(tmp_path)
    for N,nO,T,v in [(4,1,5000,9),(3,0,5000,9),(3,1,4000,9),(3,1,8000,9),(3,1,5000,12)]:
        assert nearby_liquid(root,'ft_test',N,nO,T,v)[0] is None
    assert nearby_liquid(root,'ft_test',3,1,5000,9,exclude=sub)[0] is None


def test_pressure_plan_needs_brackets(tmp_path,monkeypatch):
    monkeypatch.setattr(eos_queue,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='brackets'):
        eos_queue.resolve(dict(T=5000,compositions=['h2o'],pressures_GPa=[350]),'ft_test')
    assert eos_queue.resolve(dict(T=5000,compositions=['h2o'],volumes={'h2o':[3]}),'ft_test')=={'h2o':[3]}

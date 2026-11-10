import csv
import importlib.util
from pathlib import Path
import sys
import pytest
p=Path(__file__).resolve().parents[1]/'mlp_pipeline/young_liquid_campaign.py'
spec=importlib.util.spec_from_file_location('liquid_campaign',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_requested_grid_has_30_distinct_runs():
    assert len(m.STATES)==10 and len(set(m.STATES))==10
    assert {(p,t) for p,t in m.STATES if t==3000}=={(100,3000),(200,3000)}
    assert len(m.STATES)*len(m.SYSTEMS)==30


def test_explicit_volume_validation(tmp_path):
    p=tmp_path/'v.csv';p.write_text('system,P_GPa,T_K,vpa_A3\nh3o,200,3000,3.1\n')
    assert m.volume_overrides(p)=={('h3o',200,3000):3.1}
    p.write_text('system,P_GPa,T_K,vpa_A3\nh3o,200,3000,-1\n')
    with pytest.raises(ValueError):m.volume_overrides(p)


def test_plan_does_not_extrapolate_missing_low_pressure(tmp_path,monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    rows=m.build_plan(tmp_path/'missing.csv',{})
    by={(r['system'],r['P_GPa'],r['T_K']):r for r in rows}
    assert by['h2o',350,7000]['vpa_A3'] is not None
    assert by['h2o',100,7000]['vpa_A3'] is None
    assert by['h2o',200,3000]['vpa_A3'] is None
    assert by['h',500,7000]['vpa_A3'] is None
    assert by['h3o',200,3000]['vpa_A3'] is None

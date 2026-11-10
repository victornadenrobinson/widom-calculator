import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parents[1]/'mlp_pipeline/young_eos_campaign.py'
s=importlib.util.spec_from_file_location('eoscampaign',p);e=importlib.util.module_from_spec(s);s.loader.exec_module(e)


def test_requested_eos_jobs_are_distinct():
    assert len(e.STATES)==9 and len(set(e.STATES))==9
    assert set(e.STATES)=={(c,t) for c in ('h2o','h3o','h') for t in (3000,5000,7000)}


def test_eos_row_filter_distinguishes_liquid_and_bad_data():
    p=Path(__file__).resolve().parents[1]/'mlp_pipeline/eos_volume_table.py'
    s=importlib.util.spec_from_file_location('eosvol',p);v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
    row=dict(P_GPa='200',vpa='3.0',T_mean='7000',dmin_HH='.6',D_O_cm2s='.0002',O_struct='liquid',P_drift_GPa='1',P_block_sd='1')
    assert v.usable(row,'h2o',7000)==(200,3)
    assert v.usable(dict(row,O_struct='fcc'),'h2o',7000) is None
    assert v.usable(dict(row,dmin_HH='.2'),'h2o',7000) is None
    assert v.usable(dict(row,T_mean='6000'),'h2o',7000) is None

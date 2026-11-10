import json
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mlp_pipeline'))
from tdi.quad_eta import estimate


def test_eta_includes_future_kronrod_and_current_rates(tmp_path):
    d=tmp_path/'gl4_gk9';d.mkdir()
    args=dict(nodes=4,dt=.5,every=2,ps=1.,equil_ps=.5,hot_ps=.5)
    (d/'run_stage0.json').write_text(json.dumps(dict(args=args)))
    for k in range(4):
        (d/f'run_node{k}_progress.json').write_text(json.dumps(dict(state='running',t=995,t_start=700,done=2000,total=3000,rate=10)))
    seconds,why=estimate(tmp_path,now=1000)
    assert seconds==pytest.approx((4*1000+5*4000)/40)
    assert '4 warmed workers' in why
    stale=d/'run_node0_progress.json';p=json.loads(stale.read_text());p['t']=700;stale.write_text(json.dumps(p))
    seconds,why=estimate(tmp_path,now=1000)
    assert seconds is None and 'stale' in why


def test_missing_metadata_is_not_reported_as_done(tmp_path):
    (tmp_path/'gl8_gk17').mkdir()
    seconds,why=estimate(tmp_path,now=1000)
    assert seconds is None and 'metadata' in why

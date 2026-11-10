"""GPU scheduling and checkpoint completion must not waste slots or skip missing work."""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pytest
p=Path(__file__).resolve().parents[1]/'mlp_pipeline/young_quad_sweep.py'
spec=importlib.util.spec_from_file_location('quad_queue',p);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


@pytest.mark.parametrize('gpus', [['0'],['2','5'],['GPU-a','GPU-b','GPU-c','GPU-d']])
def test_fill_and_refill_available_gpus(gpus):
    active=[]
    for _ in range(len(gpus)*4):
        gpu=q.choose_gpu(gpus,active,4);assert gpu is not None;active.append(dict(gpu=gpu))
        counts=[sum(t['gpu']==g for t in active) for g in gpus]
        assert max(counts)-min(counts)<=1
    assert q.choose_gpu(gpus,active,4) is None
    freed=active.pop(0)['gpu'];assert q.choose_gpu(gpus,active,4)==freed


def test_saved_completion_requires_matching_state_and_full_sampling(tmp_path):
    a=dict(dt=.5,every=2,model='toyice',rule='gl',nodes=1,power=4)
    (tmp_path/'test_stage0.json').write_text(json.dumps(dict(args=a,T=7000)))
    data=np.arange(125.);np.save(tmp_path/'test_node0.npy',data)
    sf=tmp_path/'test_node0_state.npz'
    np.savez(sf,series=data,model='toyice',dt=.5,T=7000,lam=.0625)
    assert q.node_complete(tmp_path,'test',0,.125)
    assert not q.node_complete(tmp_path,'test',0,.25)
    np.savez(sf,series=data,model='wrong-model',dt=.5,T=7000,lam=.0625)
    assert not q.node_complete(tmp_path,'test',0,.125)


def test_exact_mixture_cell_size():
    assert q.cell_size(128,2)==(4,256)
    assert q.cell_size(128,2.25)==(4,288)
    assert q.cell_size(54,2.5)==(3,135)
    with pytest.raises(ValueError,match='integer'):
        q.cell_size(54,2.25)
    with pytest.raises(ValueError,match='2\\*n\\^3'):
        q.cell_size(100,2)


def test_worker_status_displays_rate_and_ignores_stale_process(tmp_path):
    from types import SimpleNamespace
    job=dict(rule='gl',nodes=8,tag='run',directory=tmp_path)
    task=dict(job=job,k=3,proc=SimpleNamespace(pid=42),launched=100)
    f=tmp_path/'run_node3_progress.json'
    f.write_text(json.dumps(dict(pid=42,t=110,done=200,total=1000,rate=20)))
    s=q.worker_status(task,now=120)
    assert 'GL8 λ3' in s and '20.0%' in s and '20.0 steps/s' in s and '50 ms/step' in s
    f.write_text(json.dumps(dict(pid=41,t=110,done=200,total=1000,rate=20)))
    assert 'starting/model loading' in q.worker_status(task,now=120)

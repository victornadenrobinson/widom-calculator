"""Prevent reusing saved MD series at different quadrature lambda values."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mlp_pipeline'))
from tdi.quadrature_plan import plan
from tdi.analyze import rule_nodes


def test_gl_to_gk_preserves_indices_and_integrates_constant():
    old=dict(nodes=4,rule='gl',power=4)
    rule,n,ng=plan(old,'gk9');new=dict(nodes=n,rule=rule,ngauss=ng,power=4)
    lo,_,_=rule_nodes(old);ln,w,_=rule_nodes(new)
    np.testing.assert_allclose(lo,ln[:4],rtol=1e-12)
    assert abs(w.sum()-1)<1e-12
    assert plan(new)==('gk',9,4)


def test_non_nested_refinement_and_downgrade_refused():
    old=dict(nodes=9,rule='gk',ngauss=4,power=4)
    for request in ('gk19','gk21','gl4','gl10'):
        with pytest.raises(ValueError,match='preserve'):plan(old,request)

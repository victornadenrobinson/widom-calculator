"""Plan a saved-sample-compatible GL -> GK extension. Prints RULE NODES NGAUSS SAVED_PS."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from tdi import analyze


def plan(previous=None, requested='auto'):
    if requested == 'auto':
        rule = (previous.get('rule') or 'gl') if previous else 'gl'
        nodes = int(previous['nodes']) if previous else 4
        ngauss = int(previous.get('ngauss') or 0) if previous and rule == 'gk' else 0
    else:
        m = re.fullmatch(r'(gl|gk)([1-9][0-9]*)', requested)
        if not m:
            raise ValueError('Use auto, gl<N> or gk<2n+1>.')
        rule,nodes = m[1],int(m[2]);ngauss=(nodes-1)//2 if rule=='gk' else 0
        if rule=='gk' and (nodes<3 or nodes%2!=1):
            raise ValueError('GK node count must be odd and at least 3.')
    current=dict(rule=rule,nodes=nodes,ngauss=ngauss,power=previous.get('power',4) if previous else 4)
    lam,_,_=analyze.rule_nodes(current)
    if previous:
        old=analyze.rule_nodes(previous)[0]
        if len(lam)<len(old) or not np.allclose(lam[:len(old)],old,rtol=1e-12,atol=0):
            raise ValueError('Requested grid does not preserve existing lambda indices. Use a separate OUTDIR. '
                             'This implementation supports GL(n) -> GK(2n+1), not repeated GK nesting.')
    return rule,nodes,ngauss


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('directory');ap.add_argument('rule',nargs='?',default='auto');a=ap.parse_args()
    files=list(Path(a.directory).glob('*_stage0.json'))
    if len(files)>1:ap.error('Expected one pilot run in this directory.')
    previous=json.loads(files[0].read_text())['args'] if files else None
    try:print(*plan(previous,a.rule), previous.get('ps',0.25) if previous else 0.25)
    except ValueError as e:ap.error(str(e))

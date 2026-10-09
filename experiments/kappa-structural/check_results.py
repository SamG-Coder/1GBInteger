"""Offline evidence and independent exact arithmetic checks (MIT)."""
import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def upper_log(x):
    """atanh series with a rational upper bound on the positive tail."""
    def small(y):
        z = (y-1)/(y+1)
        return 2*sum((z**k/k for k in range(1,65,2)), F(0)) + 2*z**65/(65*(1-z*z))
    power = 0
    while x >= 2:
        power += 1; x /= 2
    return round_up(power*small(F(2))+small(x))


def round_up(x):
    scale = 2**120
    return F(-(-x.numerator*scale//x.denominator),scale)


def upper_exp(x):
    assert 0 <= x < 3
    return round_up(1+x+x*x/(2*(1-x/3)))


def coarse_gap(row, c):
    m, w = row['m'], row['W_per_vertex']
    children = {int(r): n for r,n in row['child_histogram'].items()}
    ideal = sum(F(r*n,w*m)*upper_exp(c*upper_log(F(m,r))) for r,n in children.items())
    rare = F(32*m*m*sum(children.values()),10**16*w*m)*upper_exp(c*upper_log(F(m)))
    assert F(row['rank_per_vertex'])+F(32*m*m*sum(children.values()),10**16)<w*m
    return 1-ideal-rare


def validate():
    manifest_path = HERE/'manifest.json'
    if manifest_path.exists():
        for name, digest in json.loads(manifest_path.read_text())['files'].items():
            assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,name
    row = json.loads((HERE/'row.json').read_text())
    plan = json.loads((HERE/'plan.json').read_text())
    v = json.loads((HERE/'verification.json').read_text())
    h = Counter()
    for name in ['auxiliary_histogram','source_data_histogram','target_data_histogram','copied_center_histogram']:
        for rank,n in row[name].items(): h[int(rank)] += 3*n
    for rank,n in row['selected_rank_histogram'].items(): h[3*int(rank)] += n
    h[2] += 2*row['v']
    assert dict(sorted(h.items()))=={int(r):n for r,n in row['child_histogram'].items()}
    assert sum(r*n for r,n in h.items())==row['rank_per_vertex']
    assert row['W_per_vertex']*row['m']-row['rank_per_vertex']==2024
    assert row['R']==28866-len(plan['elim'])-len(plan['pairs'])
    assert row['selected_roles']==len(plan['retained'])
    c = F(v['refined_coarse']); k = F(v['refined_kappa']); grid = F(1,10**15)
    gap = coarse_gap(row,c)
    assert gap>0 and coarse_gap(row,c+grid)<=0
    for priced in v['prices']:
        assert coarse_gap(row,F(priced['coarse']))==F(priced['coarse_gap'])>0
        assert coarse_gap(row,F(priced['coarse_next_rejected']))<=0
    # Reuse our separately written, frozen 47-constraint checker without
    # importing any upstream package. Adapt the new receipt to its schema.
    path = HERE.parent/'kappa-refinement/check_candidate.py'
    spec = importlib.util.spec_from_file_location('independent_assembly',path)
    independent = importlib.util.module_from_spec(spec); spec.loader.exec_module(independent)
    atom=F(v['atom']); a=F(v['stopped']); old=F(v['finite_bridge']['bit_uniform']['old_atom_saving'])
    threshold=c/(1+c-old)
    doc=dict(refined_coarse_control=dict(coarse=str(c),coarse_gap=str(gap)),threshold=str(threshold),
        fixed_coarse_eta_zero_supremum=str(threshold/(1+2*threshold)),
        headline=v['prices'][0],secondary=v['prices'][1],best=dict(assembly=v['assembly'],finite_bridge=v['finite_bridge'],
        record=dict(atom=str(atom),eta=v['eta'],kappa=str(k),stopped=str(a),adapter_gap=str(atom-a),
            absorption_gap=v['assembly']['absorption_gap'],next_grid_rejected=v['next_grid_rejected'])))
    independent.validate(doc)
    rejected=[]
    for name,key,value in [('next-kappa','kappa',v['next_grid_rejected']),('zero-eta','eta','0'),('atom-equality','atom',str(threshold))]:
        bad=copy.deepcopy(doc);bad['best']['record'][key]=value
        try: independent.validate(bad)
        except AssertionError: rejected.append(name)
        else: raise AssertionError('corruption accepted: '+name)
    assert v['complete_formal_identity_F2'] and v['complete_formal_identity_Z']
    assert v['fresh_moves']==4878 and v['formal_variables']==row['W_per_vertex']
    assert v['rejected_corruptions']==['stale','early','vlast']
    search=json.loads((HERE/'search-results.json').read_text())
    assert len(search['added_gauges'])==187 and search['row_identical_to_PR151']
    assert not search['baseline_greedy_removals'] and not search['rounds']
    for prefix in ['cut','all-gauges-cut']:
        assert json.loads((HERE/(prefix+'-row.json')).read_text())==row
    match=json.loads((HERE/'rematch-results.json').read_text())
    assert match['edges']==666715 and len(match['trials'])==4
    for i,trial in enumerate(match['trials'],1):
        assert trial['pairs']==3338 and trial['child_histogram']==row['child_histogram']
        assert json.loads((HERE/f'rematch-{i}-row.json').read_text())==row
    provenance=json.loads((HERE/'provenance.json').read_text())
    assert provenance['working_tree_clean'] and provenance['HEAD']==v['upstream_commit']
    assert provenance['PR151_row_equal']
    assert hashlib.sha256((HERE/'row.json').read_bytes()).hexdigest()==provenance['local_row_sha256']
    published=F(472154791,10**12)
    assert k>published
    return dict(kappa=str(k),decimal=float(k),improvement_over_PR151_percent=float(100*(k/published-1)),
        coarse_gap=str(gap),strict_constraints=47,assembly_margins=7,negative_arithmetic_controls=rejected,
        scope='Saved geometry/replay receipts and independently recomputed arithmetic; geometry is not rerun by this offline check.')


if __name__=='__main__':
    print(json.dumps(validate(),indent=2))

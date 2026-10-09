"""Exact parameter-only search on the unchanged, pinned PR150 bit row.

MIT wrapper; all external checker code retains upstream Apache-2.0 notices.
This is an instantiation refinement, not a new circuit or unconditional theorem.
"""
import copy,hashlib,json,pathlib,subprocess,sys
from fractions import Fraction as Q
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
UP=ROOT/'.local-llm/integer-mult-research';PIN='40d4038760ebb6d6d3d702ce88fa3645de29f0c1'
assert subprocess.check_output(['git','-C',str(UP),'rev-parse','HEAD'],text=True).strip()==PIN
subprocess.run(['git','-C',str(UP),'diff','--exit-code'],check=True)
sys.path.insert(0,str(UP/'research/bit-reuse-147'))
import price147 as pricing
p=pricing.pcn
row=json.loads((UP/'research/bit-reuse-147/row.json').read_text())
base=pricing.price(row);secondary=pricing.price(row,Q(1,2000))
assert base['kappa']==Q(471809569,10**12) and secondary['kappa']==Q(472026274,10**12)
precise=pricing.price(row,Q(1,2000),grid=10**15)
coarse=precise['coarse'];old=p.OLD
threshold=coarse/(1+coarse-old)
# Strict atom > stopped(atom) is equivalent to atom > this threshold.
grid=10**15
smallest_grid_atom=Q((threshold*grid).__floor__()+1,grid)
crow=json.loads((UP/'certificates/paired-cube-complex-input.json').read_text())
phase=p.complex_certificate(crow);original_bridge=p.finite_bridge(phase,None,crow)
cases=[];full=[]
for atom in [Q(1,n) for n in [1000,2000,2050,2100,2110,2116,2117]]+[threshold,smallest_grid_atom]:
    a=(1-atom)*coarse+atom*old
    if not a<atom<1-a:
        cases.append(dict(atom=str(atom),accepted=False,reason='Strict subordinate adapter/row toll failed',stopped=str(a)));continue
    for eta in [Q(1,10**8),Q(1,10**14)]:
        bridge=copy.deepcopy(original_bridge)
        bridge['bit_uniform'].update(coarse_saving=coarse,atom_beta=atom,ordinary_saving=a)
        assert a<(1-p.PHASE_STOP)*p.AC-Q(1,10**10)
        def accepted(k):
            try:p.assembly(a,p.AC,bridge,Q(k,grid),eta=eta,beta=p.PHASE_STOP)
            except (AssertionError,ValueError):return False
            return True
        kk=pricing.largest(accepted,0,int(p.AC*grid)+1)
        result=p.assembly(a,p.AC,bridge,Q(kk,grid),eta=eta,beta=p.PHASE_STOP)
        assert not accepted(kk+1)
        assert len(result['strict_constraints'])==47 and len(result['margins'])==7
        assert all(x>0 for x in result['strict_constraints'].values())
        record=dict(atom=str(atom),eta=str(eta),accepted=True,stopped=str(a),kappa=str(Q(kk,grid)),kappa_decimal=float(Q(kk,grid)),
            adapter_gap=str(atom-a),next_grid_rejected=str(Q(kk+1,grid)),absorption_gap=str(result['absorption_gap']))
        cases.append(record);full.append(dict(record=record,assembly=p.js(result),finite_bridge=p.js(bridge)))
best=max(full,key=lambda x:Q(x['record']['kappa']))
ceiling=threshold/(1+2*threshold)
assert Q(best['record']['kappa'])<ceiling
data=dict(upstream_commit=PIN,scope='Parameter-only conditional finite witness using unchanged PR150 row and inherited interfaces; no structural circuit improvement.',
    headline=p.js(base),secondary=p.js(secondary),refined_coarse_control=p.js(precise),threshold=str(threshold),fixed_coarse_eta_zero_supremum=str(ceiling),
    cases=cases,best=best,source_sha256={f:hashlib.sha256((UP/f).read_bytes()).hexdigest() for f in [
        'research/bit-reuse-147/price147.py','research/bit-reuse-147/row.json','research/bit-reuse-147/plan.json',
        'scripts/paired_cube_network.py','scripts/structured_bulk_assembly.py','certificates/paired-cube-complex-input.json']})
(HERE/'candidate.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8',newline='\n')
print('PR150 headline:',float(Q(data['headline']['kappa'])),'secondary:',float(Q(data['secondary']['kappa'])))
print('Best arithmetic candidate:',best['record'])
print('Fixed-coarse eta-zero supremum:',float(ceiling))

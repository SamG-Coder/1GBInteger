"""MIT verification driver for the composed, pinned conditional circuit."""
import copy
import json
import random
import sys
import time
from collections import Counter
from fractions import Fraction as Q
import search as setup
import price147 as pricing
from frames147 import Exact


def main():
    assert not sys.flags.optimize
    start = time.perf_counter()
    def log(s):
        print(f'[{time.perf_counter()-start:.1f}s] {s}', flush=True)
    _, W, _, S, record, folder = setup.word147.load_schedule()
    plan = json.loads((setup.HERE / 'plan.json').read_text())
    word = setup.word147.Word(S, **plan)
    ledger = word.ledger(); row = word.row(ledger, record)
    assert row == json.loads((setup.HERE / 'row.json').read_text())
    log(f'Ledger rebuilt: {row["R"]} registers, {len(plan["pairs"])} handoffs')
    exact = Exact(S, W, folder)
    fresh = word.new_moves(ledger)
    for i, (reg, a, b) in enumerate(fresh):
        assert exact.inside(a, b), (reg, a, b)
        assert exact.nondegenerate(a) and exact.nondegenerate(b), (a, b)
        if (i+1) % 1000 == 0:
            log(f'Exact new frame checks {i+1}/{len(fresh)}')
    log(f'PASS all {len(fresh)} new moves nested over Q with nondegenerate endpoints')
    all_count = None
    if '--all' in sys.argv:
        all_count = len(ledger['moves'])
        for i, (reg, a, b) in enumerate(sorted(ledger['moves'], key=str)):
            assert exact.inside(a, b), (reg, a, b)
            if (i+1) % 20000 == 0:
                log(f'All frame checks {i+1}/{all_count}')
        log(f'PASS all {all_count} distinct frame moves nested over Q')
    outputs = {(c, tuple(T)): n for c, T, n in W['outputs']}
    for ring in [2, 0]:
        assert word.complete(ring, outputs)
        log(f'PASS complete formal identity over {"F2" if ring else "Z"}')
    rng = random.Random(20261009)
    for ring in [2, 2, 2, 0, 0]:
        assert word.replay(ring, rng, outputs)
    for bad in ['stale', 'early', 'vlast']:
        assert not word.replay(0, rng, outputs, bad)
    log('PASS five scalar replays; stale/early/vlast corruptions rejected')
    prices = [pricing.price(row), pricing.price(row, Q(1, 2000))]
    refined = pricing.price(row, Q(1, 2000), grid=10**15)
    coarse = refined['coarse']; p = pricing.pcn; grid = 10**15
    threshold = coarse / (1 + coarse - p.OLD)
    atom = Q((threshold * grid).__floor__() + 1, grid)
    stopped = (1-atom)*coarse + atom*p.OLD
    assert stopped < atom < 1-stopped
    eta = Q(1, 10**14)
    crow = json.loads((setup.UP / 'certificates/paired-cube-complex-input.json').read_text())
    bridge = p.finite_bridge(p.complex_certificate(crow), None, crow)
    bridge['bit_uniform'].update(coarse_saving=coarse, atom_beta=atom, ordinary_saving=stopped)
    assert stopped < (1-p.PHASE_STOP)*p.AC-Q(1, 10**10)
    def accepts(k):
        try:
            p.assembly(stopped, p.AC, bridge, Q(k, grid), eta=eta, beta=p.PHASE_STOP)
        except (AssertionError, ValueError):
            return False
        return True
    kk = pricing.largest(accepts, 0, int(p.AC*grid)+1)
    assembly = p.assembly(stopped, p.AC, bridge, Q(kk, grid), eta=eta, beta=p.PHASE_STOP)
    assert not accepts(kk+1)
    assert len(assembly['strict_constraints']) == 47 and len(assembly['margins']) == 7
    assert all(x > 0 for x in assembly['strict_constraints'].values())
    assert all(x > 0 for x in assembly['margins'].values())
    result = dict(upstream_commit=setup.PIN, fresh_moves=len(fresh), all_moves_checked=all_count,
        frame_kinds={str(k): n for k, n in Counter((reg, a[0], b[0]) for reg,a,b in fresh).items()},
        complete_formal_identity_F2=True, complete_formal_identity_Z=True, formal_variables=row['W_per_vertex'],
        scalar_trials=5, rejected_corruptions=['stale','early','vlast'], prices=p.js(prices),
        refined_coarse=str(coarse), refined_kappa=str(Q(kk,grid)), refined_kappa_decimal=float(Q(kk,grid)),
        atom=str(atom), eta=str(eta), stopped=str(stopped), next_grid_rejected=str(Q(kk+1,grid)),
        finite_bridge=p.js(bridge), assembly=p.js(assembly), elapsed_seconds=time.perf_counter()-start)
    setup.save('verification.json', result)
    log(f'PASS conditional kappa {Q(kk,grid)} = {float(Q(kk,grid)):.15f}')


if __name__ == '__main__':
    main()

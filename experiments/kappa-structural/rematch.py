"""MIT search driver: reconsider register matching after gauge composition.

Uses the PR150 legal-edge criterion and moment objective (DaysSky); expands
the recipient set with PR146/PR151 gauges. Exact verification is separate.
"""
import json
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
import search as setup
from frames147 import Exact


def main():
    started = time.perf_counter()
    def log(s):
        print(f'[{time.perf_counter()-started:.1f}s] {s}', flush=True)
    _, W, _, S, record, folder = setup.word147.load_schedule()
    plan = json.loads((setup.HERE / 'plan.json').read_text())
    selected = set(plan['retained']) | {b for b, d in plan['pairs']}
    level = [s for s in S.readout if s in selected]
    bare = setup.word147.Word(S, plan['elim'], level, [])
    exact = Exact(S, W, folder)
    donors = [d for d in range(S.R) if d not in S.out and d not in S.ret and d not in bare.elim]
    final = {d: [k for k in S.chain_keys(d) if k[0] != 'F'][-1] for d in donors}
    dims = np.array([S.dim(final[d]) for d in donors])
    deaths = np.array([bare.death[d] for d in donors])
    prime = 1048573
    rows, owners = [], []
    for i, d in enumerate(donors):
        for r in exact.basis(final[d]):
            rows.append([x % prime for x in r]); owners.append(i)
        if (i+1) % 5000 == 0:
            log(f'Donor bases {i+1}/{len(donors)}')
    matrix = np.asarray(rows, dtype=np.int64); owners = np.asarray(owners)
    lengths = np.bincount(owners, minlength=len(donors))
    log(f'Donor matrix {matrix.shape}, {matrix.nbytes/2**20:.1f} MiB')
    compatible = {}; edge_data = []
    for j, b in enumerate(level):
        sigma = tuple(map(tuple, S.sigma[b]))
        if sigma not in compatible:
            null = np.asarray([[x % prime for x in r] for r in exact.null(('sigma', b))], dtype=np.int64)
            # h=23 and entries < prime, so dot products cannot overflow int64.
            assert S.h * (prime-1)**2 < 2**63
            ok = ((matrix @ null.T) % prime == 0).all(axis=1)
            compatible[sigma] = np.flatnonzero(np.bincount(owners[ok], minlength=len(donors)) == lengths)
        idx = compatible[sigma]
        idx = idx[(deaths[idx] < bare.p[b]) & (dims[idx] <= S.f[b])]
        for i in idx:
            if donors[i] != b:
                edge_data.append((int(i), j, S.f[b], int(dims[i])))
        if (j+1) % 2000 == 0:
            log(f'Recipients {j+1}/{len(level)}, edges {len(edge_data)}')
    used_d = sorted({i for i,j,f,d in edge_data}); used_b = sorted({j for i,j,f,d in edge_data})
    dr = {d:i for i,d in enumerate(used_d)}; br = {b:j for j,b in enumerate(used_b)}
    weights = np.zeros((len(used_d), len(used_b)), dtype=np.float64)
    log(f'Assignment matrix {weights.shape}, {weights.nbytes/2**20:.1f} MiB')
    trials = []
    for saving in [0.000472806534, 0.000473, 0.00048, 0.00046]:
        phi = lambda r: setup.phi(r, saving)
        for i,j,f,d in edge_data:
            weights[dr[i], br[j]] = 3*phi(S.h-d)+phi(3*f)-3*phi(f-d)
        rr, cc = linear_sum_assignment(weights, maximize=True)
        pairs = sorted([level[used_b[c]], donors[used_d[r]]] for r,c in zip(rr,cc) if weights[r,c]>0)
        taken = {b for b,d in pairs}
        new_plan = dict(elim=plan['elim'], retained=[b for b in level if b not in taken], pairs=pairs)
        word = setup.word147.Word(S, **new_plan); row = word.row(word.ledger(), record)
        trial = dict(saving=saving, pairs=len(pairs), R=row['R'], child_histogram=row['child_histogram'])
        trials.append(trial)
        log(f'Saving {saving}: {len(pairs)} pairs, {row["R"]} registers')
        setup.save(f'rematch-{len(trials)}-plan.json', new_plan)
        setup.save(f'rematch-{len(trials)}-row.json', row)
    setup.save('rematch-results.json', dict(trials=trials, edges=len(edge_data), donor_matrix_bytes=matrix.nbytes,
        assignment_matrix_bytes=weights.nbytes, elapsed_seconds=time.perf_counter()-started))


if __name__ == '__main__':
    main()

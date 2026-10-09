"""MIT experiment: prune gauge reads on the pinned PR150 recycled circuit.

Imports upstream Apache-2.0 code without changing it. Floating point only
proposes candidates; verify.py rebuilds and certifies the resulting word.
"""
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UP = ROOT / '.local-llm/integer-mult-research'
PIN = '40d4038760ebb6d6d3d702ce88fa3645de29f0c1'
GAUGE_PIN = '70355a3192028852598e96624ae29b389e664451'
COMPOSITION_PIN = '5a46ac37ae2e2f8f75637cb2cf6b9d0b36afc436'
sys.path.insert(0, str(UP / 'research/bit-reuse-147'))
import word147


def save(name, value):
    options = dict(separators=(',', ':')) if name.endswith('plan.json') else dict(indent=2)
    (HERE / name).write_text(json.dumps(value, **options) + '\n', encoding='utf-8', newline='\n')


def phi(r, a=0.000473, m=69):
    return r * math.expm1(a * math.log(m / r)) / a if r else 0.0


def target_paths(word):
    """Label every target visit, including repeats at the same frame."""
    S = word.S
    paths = [[(None, 0)] for _ in range(S.v)]
    for kind, s in word.events():
        if kind == 'read':
            for t in word.adj[s]:
                paths[t].append((s, S.f[s]))
        elif kind == 'vred':
            paths[word.target_of[s]].append((None, S.dim(('v', s))))
        elif kind == 'op':
            op = S.ops[s]
            if op[0] == 'add':
                if op[1] in word.elim:
                    paths[word.target_of[op[1]]].append((None, S.dim(('n', op[3]))))
            else:
                for g in op[2]:
                    if g in word.elim:
                        paths[word.target_of[g]].append((None, S.dim(S.start_key(g))))
    for path in paths:
        path.append((None, S.h - 1))
        assert all(a[1] <= b[1] for a, b in zip(path, path[1:]))
    return paths


def prune(word, ledger):
    paths = target_paths(word)
    # Check that this independent target trace reconstructs the ledger histogram.
    from collections import Counter
    hist = Counter(b[1] - a[1] for path in paths for a, b in zip(path, path[1:]) if b[1] > a[1])
    assert hist == ledger['hist']['y']
    nodes = []; occurrences = {s: [] for s in word.retained}
    for path in paths:
        for i, (s, rank) in enumerate(path):
            j = len(nodes)
            nodes.append([rank, j - 1 if i else None, j + 1 if i + 1 < len(path) else None])
            if s in occurrences:
                occurrences[s].append(j)
    aux_gain = {}
    for s in word.retained:
        f = word.S.f[s]
        first = word.S.dim(ledger['seq']['a', s][1])
        aux_gain[s] = phi(3 * f) + 3 * phi(first - f) - 3 * phi(first)
    def gain(s):
        z = aux_gain[s]
        for j in occurrences[s]:
            r, prev, nxt = nodes[j]
            lo, hi = nodes[prev][0], nodes[nxt][0]
            z += 3 * (phi(r - lo) + phi(hi - r) - phi(hi - lo))
        return z
    active = set(word.retained); removed = []; total = 0.0
    while active:
        score, s = max((gain(s), s) for s in active)
        if score <= 1e-8:
            break
        active.remove(s); removed.append(s); total += score
        for j in occurrences[s]:
            _, prev, nxt = nodes[j]
            nodes[prev][2] = nxt; nodes[nxt][1] = prev
    return sorted(active), removed, total


def main():
    t0 = time.perf_counter()
    assert subprocess.check_output(['git', '-C', str(UP), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    subprocess.run(['git', '-C', str(UP), 'diff', '--exit-code'], check=True)
    _, W, D, S, record, folder = word147.load_schedule()
    plan = json.loads((UP / 'research/bit-reuse-147/plan.json').read_text())
    original = word147.Word(S, **plan)
    base = original.row(original.ledger(), record)
    assert base == json.loads((UP / 'research/bit-reuse-147/row.json').read_text())
    _, baseline_removed, baseline_gain = prune(original, original.ledger())
    assert not baseline_removed
    selection = json.loads(subprocess.check_output(['git', '-C', str(UP), 'show',
        GAUGE_PIN + ':references/paired-cube/pr97-subset/selection.json'], text=True))
    recipients = {b for b, d in plan['pairs']}
    old_retained = set(plan['retained'])
    plan['retained'] = sorted(set(selection['retained_readout_order']) - set(plan['elim']) - recipients)
    added = sorted(set(plan['retained']) - old_retained)
    composed = word147.Word(S, **plan)
    composed_row = composed.row(composed.ledger(), record)
    published = json.loads(subprocess.check_output(['git', '-C', str(UP), 'show',
        COMPOSITION_PIN + ':research/gauge-recycling/row.json'], text=True))
    assert composed_row == published, 'Rediscovered composition is exactly PR151, not a new construction'
    save('plan.json', plan); save('row.json', composed_row)
    print('PR146/PR150 composition:', len(added), 'added gauges; row identical to PR151', flush=True)
    trials = []
    for iteration in range(10):
        word = word147.Word(S, **plan); led = word.ledger()
        retained, removed, predicted = prune(word, led)
        print('round', iteration, 'removed', len(removed), 'predicted phi gain', predicted, flush=True)
        if not removed:
            break
        before = word.row(led, record)
        candidate = dict(plan, retained=retained)
        after_word = word147.Word(S, **candidate)
        after = after_word.row(after_word.ledger(), record)
        cost = lambda row: sum(n * phi(int(r)) for r, n in row['child_histogram'].items())
        actual = cost(before) - cost(after)
        trials.append(dict(iteration=iteration, removed=removed, predicted_gain=predicted, actual_gain=actual))
        print('actual phi gain', actual, 'roles', after['R'], flush=True)
        assert actual > 0, 'Fresh timetable must improve the actual ledger too'
        plan = candidate
        save('plan.json', plan); save('row.json', after)
    save('search-results.json', dict(upstream_commit=PIN, gauge_commit=GAUGE_PIN, composition_commit=COMPOSITION_PIN,
        method='PR146 gauges composed with fixed PR150 handoffs; greedy removals on both baseline and composed row',
        baseline_greedy_removals=baseline_removed, added_gauges=added, row_identical_to_PR151=True,
        rounds=trials, removed=sum(len(t['removed']) for t in trials), baseline_roles=base['R'], elapsed_seconds=time.perf_counter()-t0))


if __name__ == '__main__':
    main()

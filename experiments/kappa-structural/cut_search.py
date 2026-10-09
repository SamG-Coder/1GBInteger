"""MIT proposal search for jointly removable gauges on a recycled word.

Interval closure is inspired by Th0rgal's PR146 telescoping min-cut. Here
mandatory recycled reads and redirected writes split each target's path.
Rounded capacities are search heuristics, never an exact optimality claim.
"""
import json
import sys
import networkx as nx
import search as setup


def main():
    _, _, _, S, record, _ = setup.word147.load_schedule()
    plan = json.loads((setup.HERE / 'plan.json').read_text())
    prefix = 'cut'
    if '--all-gauges' in sys.argv:
        recipients = {b for b,d in plan['pairs']}
        plan['retained'] = sorted(set(S.sel)-set(plan['elim'])-recipients)
        prefix = 'all-gauges-cut'
    word = setup.word147.Word(S, **plan); ledger = word.ledger()
    paths = setup.target_paths(word)
    graph = nx.DiGraph(); source = 'source'; sink = 'sink'
    scale = 10**9
    benefit = {}
    for s in word.retained:
        f = S.f[s]; first = S.dim(ledger['seq']['a', s][1])
        benefit[s] = 3*setup.phi(first) - setup.phi(3*f) - 3*setup.phi(first-f)
    inf = sum(round(abs(b)*scale) for b in benefit.values()) + 1
    for s, b in benefit.items():
        if b >= 0:
            graph.add_edge(source, s, capacity=round(b*scale))
        else:
            graph.add_edge(s, sink, capacity=round(-b*scale))
    serial = 0
    for path in paths:
        left = 0; optional = []
        for s, rank in path[1:]:
            if s in word.retained:
                optional.append((s, rank))
                continue
            # The next required visit fixes this interval's right endpoint.
            ranks = sorted({r for _,r in optional if left < r < rank})
            dims = [left] + ranks + [rank]
            interval = {}
            for i in range(1, len(dims)-1):
                for j in range(i, len(dims)-1):
                    key = ('interval', serial); serial += 1; interval[i,j] = key
                    c = 3*(setup.phi(dims[j+1]-dims[i]) + setup.phi(dims[j]-dims[i-1])
                           - setup.phi(dims[j+1]-dims[i-1]) - setup.phi(dims[j]-dims[i]))
                    assert c >= -1e-8
                    graph.add_edge(key, sink, capacity=max(0,round(c*scale)))
            for (i,j), key in interval.items():
                if i < j:
                    graph.add_edge(interval[i,j-1], key, capacity=inf)
                    graph.add_edge(interval[i+1,j], key, capacity=inf)
            for candidate, r in optional:
                if left < r < rank:
                    i = ranks.index(r)+1
                    graph.add_edge(candidate, interval[i,i], capacity=inf)
            left = rank; optional = []
    flow, (kept, other) = nx.minimum_cut(graph, source, sink)
    retained = sorted(word.retained & kept)
    trial = dict(plan, retained=retained)
    candidate = setup.word147.Word(S, **trial)
    row = candidate.row(candidate.ledger(), record)
    original = word.row(ledger, record)
    cost = lambda r: sum(n*setup.phi(int(k)) for k,n in r['child_histogram'].items())
    gain = cost(original)-cost(row)
    result = dict(nodes=len(graph), edges=graph.number_of_edges(), rounded_scale=scale,
        cut_capacity=flow, removed=sorted(word.retained-set(retained)), actual_phi_gain=gain)
    setup.save(prefix+'-results.json', result)
    setup.save(prefix+'-plan.json', trial); setup.save(prefix+'-row.json', row)
    print('Cut:',len(graph),'nodes;',len(result['removed']),'removed; phi gain',gain)


if __name__ == '__main__':
    main()

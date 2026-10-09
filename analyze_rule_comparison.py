"""Verify fixed-suite evidence, summarize every seed, and render comparative studies."""
import argparse
import csv
import hashlib
import json
import pathlib
import statistics
from analyze_experiments import derive

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'experiments'/'rule-comparison'
RULES=('xor','add','local')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def span(rows,key,digits=6):
    values=[r[key] for r in rows]
    return f'{min(values):.{digits}f}–{max(values):.{digits}f}'


def write_csv(name,rows,keys):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)


def main(plots=False):
    manifest=json.loads((OUT/'manifest.json').read_text())
    for path,digest in manifest['source_sha256'].items():assert sha(ROOT/path)==digest,path
    stats=[];damage=[];graphs=[];bench=[];controls=[]
    for item in manifest['runs']:
        path=ROOT/item['file'];assert sha(path)==item['sha256'],path
        d=json.loads(path.read_text());identity={k:item[k] for k in ('group','rule','seed','file')}
        if item['kind']=='stats':
            for r in d['layouts']:
                assert sum(r['atom_ones'])==r['ones']
                assert sum(r['first'])==r['atoms']
                assert sum(r['blocks'])==r['atoms']*r['observations_per_atom']//8
                assert sum(r['transitions'])==r['atoms']*(r['observations_per_atom']-1)
                stats.append({**r,**derive(r),**identity,'bytes':d['bytes']})
        elif item['kind']=='damage':
            for r in d['damage']:
                assert len(r['hamming'])==len(r['changed_limbs'])==r['steps']+1
                assert r['hamming'][0]==r['changed_limbs'][0]==1
                assert all(0<=h<=64*r['limbs'] for h in r['hamming'])
                damage.append({**r,**identity,'final_hamming':r['hamming'][-1],
                               'final_changed_limbs':r['changed_limbs'][-1],
                               'max_hamming':max(r['hamming']),
                               'fraction_changed_bits':r['hamming'][-1]/(64*r['limbs'])})
        elif item['kind']=='graph':
            for r in d['graphs']:
                assert r['distinct_images']+r['collision_excess']==r['states']
                if d['rule'] in ('xor','add'):
                    assert r['collision_excess']==0 and r['cyclic_states']==r['states'] and r['max_transient']==0
                graphs.append({**r,**identity,'collision_fraction':r['collision_excess']/r['states'],
                               'cyclic_fraction':r['cyclic_states']/r['states']})
        else:
            assert d['steps']==3 and len(d['step_seconds'])==3
            bench.append({**d,**identity,'ms_per_step':1000*sum(d['step_seconds'])/3,
                          'cpu_percent_machine':100*d['evolution_cpu_seconds']/d['evolution_seconds']/manifest['logical_cpus'],
                          'peak_MB':d['peak_resident_bytes']/1e6,
                          'process_wall_seconds':item['process_wall_seconds']})
    for item in manifest['controls']:
        path=ROOT/item['file'];assert sha(path)==item['sha256'],path
        d=json.loads(path.read_text());controls.append({**d,**derive(d),'group':item['group']})
    for rule in RULES:
        for size in (1000000,16000000,1000000000):
            rs=[r for r in bench if r['rule']==rule and r['bytes']==size]
            assert len(rs)==3 and len({r['checksum'] for r in rs})==1
    common=['group','seed','rule']
    write_csv('temporal-summary.csv',stats,common+['bytes','layout','ones_fraction','atom_bias_min','atom_bias_max',
              'atom_bias_sd','block_entropy','conditional_entropy','max_abs_correlation','cross_correlation',
              'prediction_accuracy','survival_max_deviation','restricted_mean_trials','first_event_censored',
              'events','completed_gaps','tail_censored','run_length_one_fraction','file'])
    write_csv('damage-summary.csv',damage,common+['limbs','steps','first_return','final_hamming','max_hamming',
              'final_changed_limbs','fraction_changed_bits','file'])
    write_csv('graph-summary.csv',graphs,common+['word_bits','limbs','states','distinct_images','collision_excess',
              'collision_fraction','maximum_preimages','cycles','cyclic_states','cyclic_fraction',
              'min_cycle','max_cycle','max_transient','file'])
    write_csv('benchmark-summary.csv',bench,['rule','bytes','steps','ms_per_step','init_seconds','evolution_seconds',
              'evolution_cpu_seconds','cpu_percent_machine','peak_MB','process_wall_seconds','checksum','file'])
    write_csv('control-summary.csv',controls,['group','seed','ones_fraction','max_abs_correlation','block_entropy',
              'conditional_entropy','prediction_accuracy','survival_max_deviation'])
    # Compact derived data; raw counters remain the canonical evidence.
    (OUT/'derived.json').write_text(json.dumps({'stats':stats,'damage':damage,'graphs':graphs,'bench':bench,'controls':controls},separators=(',',':'))+'\n')
    (OUT/'analysis-provenance.json').write_text(json.dumps({'manifest_sha256':sha(OUT/'manifest.json'),
        'analysis_source_sha256':{f:sha(ROOT/f) for f in ('analyze_rule_comparison.py','analyze_experiments.py')}},indent=2)+'\n')
    report_temporal(stats,controls)
    report_structure(damage,graphs)
    report_performance(bench)
    index(stats,damage,graphs,bench)
    if plots:plot(stats,damage,graphs,bench,controls)
    print(f'PASS: verified {len(manifest["runs"])} native runs; wrote {len(stats)} temporal profiles, {len(damage)} damage cases, {len(graphs)} exhaustive graphs, {len(bench)} benchmark records.')


def report_temporal(stats,controls):
    s='''# Case study 5 — Temporal comparison of three rules

All rules were evaluated on the same materialized initial state and mask for
each seed/size. Both layouts observe bit zero. There are 1,024 atoms and 8,192
consecutive observations per profile, without burn-in. Unlike the first suite,
these are 32,768-byte and 1,000,000-byte states, not a logical sparse 1 GB state.

The two new equations and all constants were fixed before collection in the
[protocol](PROTOCOL.md). The formulas were not tuned after seeing results.

The old neighbor-history predictor is:
`b_i(t+1) = b_i(t) XOR b_(i-1)(t-63) XOR b_(i-1)(t-64)`.
Its exact derivation applies only to the XOR baseline. Here it is also used
unchanged as a diagnostic attack against the candidate rules.

## Every seed group, size, rule and layout

Each cell is the range across the four fixed seeds in that group. Lag statistic
means the maximum absolute uncentered sign-product correlation over the fixed
11 lags. Survival deviation is the maximum absolute distance from
`(255/256)^k`, k=0..1024, for nonoverlapping eight-one events.

| Group | Bytes | Rule | Layout | Ones fraction | Max lag statistic | Block entropy (bits/8) | Predictor accuracy | Survival deviation |
|---|---:|---|---|---:|---:|---:|---:|---:|
'''
    for stage in ('development','validation'):
        for size in (32768,1000000):
            for rule in RULES:
                for layout in ('spaced','adjacent'):
                    rs=[r for r in stats if (r['group'],r['bytes'],r['rule'],r['layout'])==(stage,size,rule,layout)]
                    s+=f'| {stage} | {size:,} | {rule} | {layout} | {span(rs,"ones_fraction")} | {span(rs,"max_abs_correlation")} | {span(rs,"block_entropy",5)} | {span(rs,"prediction_accuracy",5)} | {span(rs,"survival_max_deviation",5)} |\n'
    s+='\n| Control group | Ones fraction | Max lag statistic | Predictor accuracy | Survival deviation |\n|---|---:|---:|---:|---:|\n'
    for stage in ('development','validation'):
        rs=[r for r in controls if r['group']==stage]
        s+=f'| {stage} | {span(rs,"ones_fraction")} | {span(rs,"max_abs_correlation")} | {span(rs,"prediction_accuracy",5)} | {span(rs,"survival_max_deviation",5)} |\n'
    s+='''
## Findings

- **XOR:** The old predictor remains exact for adjacent limbs. Widely spaced
  observations look much closer to the control, particularly in the larger
  state. This replicates the geometry sensitivity in the first study.
- **Addition:** The old predictor is no longer exact, but its adjacent accuracy
  remains about 65–67%, well above chance in both seed groups. Adding carries
  therefore does not remove all of the tested predictability. Balanced bits
  and closer waiting-time curves do not repair that failure.
- **Local mixing:** The old predictor is near 50% and the measured low-order
  diagnostics resemble the control in both layouts. This is a failure of this
  particular attack, not proof that no other predictor exists. Case study 6
  finds a different weakness in the reduced-word analogues.

All conditional entropies, per-atom bias extrema/standard deviations, paired
correlations, run statistics, event counts and censored totals are in
[temporal-summary.csv](temporal-summary.csv). Raw JSON retains every lag count,
transition count, block/run histogram, first-event histogram and complete gap
histogram. [derived.json](derived.json) also includes all survival/hazard points.

First-event bin zero is censored, not discarded. Restricted mean waiting time
is E[min(T,1024)], not an uncensored mean. Completed gaps alone are biased by
the finite horizon. The first and last runs of each trace are excluded from
the completed internal-run histogram; bin 128 means >=128.

These are descriptive seed-level comparisons with dependent observations. No
uncorrected multiple-test p-values or confidence claims are made. The MT19937-64
control files are reused once per seed from the earlier experiment, not new
independent controls for every size/layout. The smaller ring's wider sampling
still shares substantial source history; observations cannot be treated as
independent solely because their initial limb indices differ.

There is no physical time calibration, isotope or quantum dataset. These
waiting-time experiments are mathematical event diagnostics, not radioactive
decay measurements or evidence of quantum equivalence.
'''
    (OUT/'05-temporal-comparison.md').write_text(s,encoding='utf-8')


def report_structure(damage,graphs):
    s='''# Case study 6 — Perturbations, collisions and cycles

## Full 64-bit word perturbations

For each rule, seed and ring size, two initial states differed only at bit zero
of limb zero and evolved synchronously under the **same mask**. Every step
through 4,096 is saved. A zero first-return field means no return was observed
within the finite horizon; it does not mean an infinite period.

Ranges below cover all eight fixed seeds. Complete seed/group data are in
[damage-summary.csv](damage-summary.csv).

| Rule | Limbs | Bits | Hamming distance at step 4096 | Changed limbs at step 4096 | Maximum distance during run | Returns within horizon / 8 |
|---|---:|---:|---:|---:|---:|---:|
'''
    for rule in RULES:
        for limbs in (1,3,17,256):
            rs=[r for r in damage if r['rule']==rule and r['limbs']==limbs]
            s+=f'| {rule} | {limbs} | {limbs*64:,} | {span(rs,"final_hamming",0)} | {span(rs,"final_changed_limbs",0)} | {span(rs,"max_hamming",0)} | {sum(bool(r["first_return"]) for r in rs)} |\n'
    s+='''
**XOR** keeps one changed bit. **Addition** can produce transient carry-related
bursts of differing bits, but the final perturbation on the 256-word ring is
still confined to one word in these tests. It does not show the broad sustained
spread seen with **local mixing**, which changes all 256 words and about half
the ring's bits at the final time. The full trajectories, not just the last
point, are retained. One flipped location across eight seeded states is a
bounded experiment, not an exhaustive avalanche characterization.

The zero mask for seed 0 in a one-word ring makes both XOR and addition reduce
to pure rotation. This predetermined degenerate case is retained, not removed
to improve the candidate's appearance.

## Exhaustive reduced-word state graphs

We enumerated every state and transition for each (word width, limb count),
rule and seed: **120 graphs, 1,683,840 edges in total**. Word widths are 4 or 8
bits; rotation counts are reduced modulo that width and mask words truncated.
These are explicitly reduced analogues of the 64-bit rule.

Collision excess = number of input states minus number of distinct next states.
It is not the number of colliding pairs. A permutation has zero collision
excess, one preimage per state, and every state lies on a cycle. Local mixing
can have transient trees leading into cycles; maximum transient is the longest
distance to a cycle. Ranges cover eight masks/seeds per row.

| Rule | Word bits | Limbs | State count | Collision excess | Largest cycle | Cyclic states | Max transient |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for rule in RULES:
        for w,limbs in ((4,1),(4,2),(4,3),(8,1),(8,2)):
            rs=[r for r in graphs if (r['rule'],r['word_bits'],r['limbs'])==(rule,w,limbs)]
            s+=f'| {rule} | {w} | {limbs} | {2**(w*limbs):,} | {span(rs,"collision_excess",0)} | {span(rs,"max_cycle",0)} | {span(rs,"cyclic_states",0)} | {span(rs,"max_transient",0)} |\n'
    s+='''
## Interpretation

- XOR and addition are permutations for any width, consistent with their
  algebraic definitions and zero collisions in every enumerated graph. Addition
  has substantially longer cycles than XOR in several small configurations,
  but that does not establish a long period at production width.
- Local mixing has collisions and short attracting cycles in the tested
  reduced-word configurations. In the 16-bit graphs, only a small fraction of
  states lie on cycles; other trajectories merge into those cycles. Improved
  marginal statistics therefore coexist with loss of state information in
  these analogues.
- The small-word collision counts do **not** prove non-injectivity of the
  64-bit-word implementation. Conversely, the absence of a return in the
  finite 64-bit tests does **not** prove reversibility or a long period.

No candidate is promoted to replace the baseline. A useful follow-up would be
an explicitly reversible local rule, which could test whether local spreading
can coexist with guaranteed information preservation. It would still need
independent structural and physical tests; reversibility alone is insufficient.

All graph metrics and raw paths: [graph-summary.csv](graph-summary.csv).
Small graphs with at most 256 states also store every edge. Independent Python
traversal verified every graph statistic for one seed at all five sizes and
all three rules, including the 65,536-state graphs.
'''
    (OUT/'06-structure-and-cycles.md').write_text(s,encoding='utf-8')


def report_performance(bench):
    s='''# Case study 7 — Matched native full-state cost

Hardware: AMD Ryzen 7 9800X3D, 8 cores / 16 logical CPUs; Windows, Clang 22.1.8,
`-O3 -std=c++17`, generic x86-64 target, one computation thread. Windows links
Psapi only for process measurements. No explicit AVX2/AVX-512 kernel is added.

All rules materialize three equally sized buffers: source, destination and a
cached mask. Each run performs three synchronous full-state updates, then
computes a checksum outside the timed evolution. There are three fresh-process
replicates per rule/size; rule order rotates between repetitions. Initialization
is timed separately. No CSV/trace/statistics work occurs inside benchmark steps.

Median per-step time is the median of the three run means. CPU and peak memory
below come from that median run. CPU% is process CPU time during evolution,
divided by evolution wall time and 16 logical processors; a fully occupied
single logical CPU would be about 6.25%. CPU accounting is quantized: short
runs can round to zero or imply more than one busy core. CPU columns are therefore
suppressed for evolution durations below 0.1 seconds; raw readings remain in CSV.
Longer-run CPU figures are still approximate. Peak working set is the OS process high-water mark, not a memory-bandwidth
counter; it includes allocation/initialization and the checksum pass.

| Logical bytes | Rule | Median ms/step | Median-run init ms | Evolution CPU seconds (3 steps) | Machine CPU % | Peak working set MB |
|---:|---|---:|---:|---:|---:|---:|
'''
    for size in (1000000,16000000,1000000000):
        for rule in RULES:
            r=sorted([r for r in bench if r['bytes']==size and r['rule']==rule],key=lambda r:r['ms_per_step'])[1]
            cpu=f'{r["evolution_cpu_seconds"]:.6f}' if r['evolution_seconds']>=.1 else '—'
            utilization=f'{r["cpu_percent_machine"]:.2f}' if r['evolution_seconds']>=.1 else '—'
            s+=f'| {size:,} | {rule} | {r["ms_per_step"]:.6f} | {1000*r["init_seconds"]:.3f} | {cpu} | {utilization} | {r["peak_MB"]:.2f} |\n'
    s+='''
## Limits of the comparison

This compares identical materialized workloads using the new common backend.
It is not a comparison against the fastest existing in-place multithreaded AVX2
baseline, and it is not comparable to billions of sparse observations/second.
The scalar carry dependency in whole-integer addition limits simple vectorization.
Local mixing reads neighboring words but can use independent destination words;
no thread-scaling optimization was attempted in this scientific comparison.

The 1 GB runs need about 3 GB of buffers, rather than the old sparse engine's
small worker-local state. All three requested sizes completed on the connected
machine. The largest tests are short, active-desktop measurements, not sustained
thermal or contention tests. No cache-miss or DRAM-bandwidth counters were
collected. Compiler/CPU/memory details and every replicate are retained in
[manifest.json](manifest.json) and [benchmark-summary.csv](benchmark-summary.csv).

Checksums match between all repeated runs for a rule/size; checksums are expected
to differ between different rules. Native transition correctness is separately
checked with an independent whole-integer/word oracle, including carries and
discarded overflow. Benchmark determinism alone is not that correctness proof.
'''
    (OUT/'07-native-performance.md').write_text(s,encoding='utf-8')


def index(stats,damage,graphs,bench):
    s='''# 1GBInteger — Evolution-rule comparison

Recorded 2026-10-09. All artifacts remain under the repository's MIT Licence.

**Neither candidate is an unqualified replacement for the baseline.**
Whole-integer addition preserves information but retains a strong form of the
old predictability and weak perturbation spreading. Local nonlinear mixing
spreads perturbations and defeats the old predictor, but its small-word
analogues lose information and funnel trajectories into short cycles.

## Case studies

5. [Temporal statistics, waiting times and prediction](05-temporal-comparison.md)
6. [Perturbation spreading, exhaustive collisions and cycles](06-structure-and-cycles.md)
7. [Matched full-state native performance](07-native-performance.md)

![Rule comparison](figures/rule-comparison.png)

## Scope and evidence

- 123 native process runs: 48 temporal runs, 24 perturbation runs,
  24 exhaustive graph runs and 27 benchmark runs.
- Each temporal run measures both layouts, yielding 96 profiles and
  805,306,368 observed bits. These profiles share the same evolved state and
  are not independent replicates. Eight old control profiles are referenced,
  not counted as newly generated evidence.
- 96 perturbation trajectories, each 4,097 time points, including t=0.
- 120 exhaustive small-state graphs, totaling 1,683,840 transitions.
- Materialized benchmarks at 1 MB, 16 MB and 1 GB, with three repetitions.
- Four development and four fixed validation seeds; no candidate constants
  or event rules were tuned after collecting results.

The original baseline engine and first studies are unchanged. This experiment
uses full-state computation; it does not assume the new rules have an exact
sparse shortcut. Good distribution diagnostics are not evidence of a physical
atomic or quantum model. No isotope, physical time, energy, measurement law,
entanglement experiment, or quantum dataset is represented.

## Reproduce

```powershell
clang++ -O3 -std=c++17 rule_compare.cpp -lpsapi -o rule_compare.exe
clang++ -O2 -std=c++17 test_rule_edges.cpp -o test_rule_edges.exe
python test_rule_comparison.py
python run_rule_comparison.py
python analyze_rule_comparison.py
# Optional chart rendering, with matplotlib installed:
python analyze_rule_comparison.py --plots
```

On Linux omit `-lpsapi` and `.exe`; `getrusage` supplies process measurements.
The existing first-suite control files are required for collection/analysis.
The runner uses the same rules and settings described in the [fixed protocol](PROTOCOL.md).

[manifest.json](manifest.json) preserves collection-source hashes, executable
hash, compiler version, all commands, exact raw-output hashes and process wall
times. [analysis-provenance.json](analysis-provenance.json) records the analysis
sources and input manifest. Raw files preserve their original bytes across Git
checkouts. [derived.json](derived.json) contains full derived curves; compact
CSV summaries link back to every raw file. No optional plotting package is
required to reproduce the numerical analysis.

All three kernels pass independent Python transition and diagnostic-counter
oracles, including whole-integer carry/overflow edges. Exhaustive graphs are
cross-checked against an independent traversal; damage trajectories are
cross-checked against independent full states. Sanitized native smoke tests and
all earlier baseline regressions provide additional validation.
'''
    (OUT/'README.md').write_text(s,encoding='utf-8')


def plot(stats,damage,graphs,bench,controls):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'xor':'#4477aa','add':'#dd9933','local':'#228855'}
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    labels=[];values=[]
    for rule in RULES:
        rs=[r for r in stats if r['rule']==rule and r['group']=='validation' and r['bytes']==1000000 and r['layout']=='adjacent']
        labels.append(rule);values.append(sum(r['predicted'] for r in rs)/sum(r['predictions'] for r in rs))
    bars=axes[0,0].bar(labels,values,color=[colors[r] for r in RULES]);axes[0,0].bar_label(bars,labels=[f'{v:.3%}' for v in values],padding=4)
    axes[0,0].axhline(.5,color='black',ls='--',lw=.8)
    axes[0,0].set(title='Old predictor: adjacent atoms, 1 MB',ylabel='Correct next-bit fraction',ylim=(0,1.12))
    for rule in RULES:
        rs=[r for r in damage if r['rule']==rule and r['group']=='validation' and r['limbs']==256]
        a=np.array([r['hamming'] for r in rs])/16384
        axes[0,1].plot(np.median(a,axis=0),label=rule,color=colors[rule])
        axes[0,1].fill_between(np.arange(4097),np.min(a,axis=0),np.max(a,axis=0),alpha=.13,color=colors[rule])
    axes[0,1].set(title='Perturbation on a 16,384-bit ring',xlabel='Evolution step',ylabel='Fraction of bits differing',ylim=(0,.6))
    axes[0,1].legend(title='Median; shade = seed range',fontsize=8)
    for rule in RULES:
        rs=[r for r in graphs if r['rule']==rule and r['word_bits']==8 and r['limbs']==2]
        x=RULES.index(rule)
        axes[1,0].scatter([x-.10]*len(rs),[100*r['collision_fraction'] for r in rs],color=colors[rule],marker='x')
        axes[1,0].scatter([x+.10]*len(rs),[100*r['cyclic_fraction'] for r in rs],color=colors[rule],marker='o',facecolors='none')
    axes[1,0].scatter([],[],c='black',marker='x',label='Collision excess / states')
    axes[1,0].scatter([],[],edgecolors='black',facecolors='none',marker='o',label='States on cycles')
    axes[1,0].set(xticks=range(3),xticklabels=RULES,title='Exhaustive 16-bit analogues: all seeds',ylabel='Percent of 65,536 states',ylim=(-4,108))
    axes[1,0].legend(fontsize=8)
    sizes=(1000000,16000000,1000000000)
    for rule in RULES:
        med=[statistics.median(r['ms_per_step'] for r in bench if r['rule']==rule and r['bytes']==size) for size in sizes]
        axes[1,1].plot([s/1e6 for s in sizes],med,'o-',label=rule,color=colors[rule])
    axes[1,1].set(xscale='log',yscale='log',title='Matched full-state cost: one CPU thread',xlabel='Materialized state (decimal MB)',ylabel='Median milliseconds per step')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('1GBInteger | More mixing brings different tradeoffs\nTop panels use the four fixed validation seeds',fontsize=14)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'rule-comparison.png',dpi=180);fig.savefig(folder/'rule-comparison.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plots',action='store_true')
    main(parser.parse_args().plots)

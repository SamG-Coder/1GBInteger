"""Analyze native aggregate counters, verify provenance, and write case studies."""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT/'experiments'


def entropy(counts):
    total = sum(counts)
    return -sum((c/total)*math.log2(c/total) for c in counts if c)


def derive(d):
    n = d['atoms']*d['observations_per_atom']
    transitions = d['transitions']
    conditional = sum(sum(transitions[i:i+2])*entropy(transitions[i:i+2])
                      for i in (0, 2))/sum(transitions)
    survival = [1.0]
    risk = d['atoms']
    hazard = []
    for count in d['first'][1:]:
        hazard.append(count/risk if risk else None)
        risk -= count
        survival.append(risk/d['atoms'])
    assert risk == d['first'][0]
    correlations = [2*a/b-1 for a, b in zip(d['lag_equal'], d['lag_n'])]
    reference = [(255/256)**k for k in range(len(survival))]
    bias = [x/d['observations_per_atom'] for x in d['atom_ones']]
    return {'ones_fraction': d['ones']/n, 'atom_bias_min': min(bias), 'atom_bias_max': max(bias),
            'atom_bias_sd': statistics.stdev(bias), 'block_entropy': entropy(d['blocks']),
            'conditional_entropy': conditional, 'correlations': correlations,
            'max_abs_correlation': max(map(abs, correlations)),
            'cross_correlation': 2*d['cross_equal']/d['cross_n']-1,
            'prediction_accuracy': d['predicted']/d['predictions'],
            'survival_max_deviation': max(abs(a-b) for a,b in zip(survival, reference)),
            'survival': survival, 'hazard': hazard, 'restricted_mean_trials': sum(survival[:-1]),
            'first_event_censored': d['first'][0], 'events': d['blocks'][255],
            'run_length_one_fraction': d['runs'][1]/sum(d['runs']) if sum(d['runs']) else None,
            'completed_gaps': sum(d['gaps']), 'tail_censored': d['tail_censored']}


def interval(rows, key, digits=6):
    values = [r[key] for r in rows]
    return f'{min(values):.{digits}f}–{max(values):.{digits}f}'


def main(plots=False):
    manifest = json.loads((OUT/'manifest.json').read_text())
    for file, digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == digest, f'Source drift: {file}'
    results = []
    for item in manifest['runs']:
        path = ROOT/item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], path
        d = json.loads(path.read_text())
        assert sum(d['atom_ones']) == d['ones']
        assert sum(d['first']) == d['atoms']
        assert sum(d['blocks']) == d['atoms']*d['observations_per_atom']//8
        assert sum(d['transitions']) == d['atoms']*(d['observations_per_atom']-1)
        results.append({**d, **derive(d), 'group': item['group'], 'file': item['file']})
    structure_path = OUT/'raw'/'structure.csv'
    assert hashlib.sha256(structure_path.read_bytes()).hexdigest() == manifest['structure_sha256']
    with structure_path.open() as f:
        structure = [{k:int(v) for k,v in row.items()} for row in csv.DictReader(f)]
    groups = [(stage, layout, source, [r for r in results if
               (r['group'],r['layout'],r['source']) == (stage,layout,source)])
              for stage in ('development','validation') for layout in ('spaced','adjacent')
              for source in ('native','control')]
    (OUT/'summary.json').write_text(json.dumps({'runs': results, 'structure': structure}, indent=2)+'\n')
    scalar_keys = ['group','seed','layout','source','ones_fraction','atom_bias_min','atom_bias_max',
                   'atom_bias_sd','block_entropy','conditional_entropy','max_abs_correlation',
                   'cross_correlation','prediction_accuracy','survival_max_deviation',
                   'restricted_mean_trials','first_event_censored','events','completed_gaps',
                   'run_length_one_fraction','seconds']
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=scalar_keys,extrasaction='ignore');writer.writeheader();writer.writerows(results)
    common = ('\n\nDesign and limitations: [prospective protocol](PROTOCOL.md). All seeds and raw counters: '
              '[summary.csv](summary.csv), [manifest](manifest.json), and [raw directory](raw/). '
              'Development and validation use four seeds each. Range tables summarize seed-level results; '
              'the prediction table pools exact counts.\n')
    s = '# Case study 1 — Temporal appearance and sampling geometry\n\n'
    s += ('**Question:** Do individual temporal traces and paired atoms look like independent fair bits? '
          'We measured 8,388,608 bits per run, observing bit zero over 8,192 consecutive times for 1,024 atoms. '
          'The native runs use a logical 1 GB state. The control is deterministic MT19937-64, not physical randomness.\n')
    s += '\n| Group | Layout | Source | Ones fraction | Maximum absolute lag statistic | 8-bit entropy | H(next bit given current bit) |\n|---|---|---|---:|---:|---:|---:|\n'
    for stage,layout,source,rs in groups:
        s += f'| {stage} | {layout} | {source} | {interval(rs,"ones_fraction")} | {interval(rs,"max_abs_correlation")} | {interval(rs,"block_entropy",5)} | {interval(rs,"conditional_entropy",7)} |\n'
    s += ('\n| Group | Layout | Source | Per-atom bias SD | Paired-atom sign correlation | Fraction of complete internal runs with length 1 |\n'
          '|---|---|---|---:|---:|---:|\n')
    for stage,layout,source,rs in groups:
        s += f'| {stage} | {layout} | {source} | {interval(rs,"atom_bias_sd")} | {interval(rs,"cross_correlation")} | {interval(rs,"run_length_one_fraction")} |\n'
    s += ('\nUnder independent fair bits the per-atom bias standard deviation is '
          '`sqrt(0.25/8192) = 0.005524`, the expected sign correlation is zero, and '
          'the untruncated run-length probability P(L=1) is 1/2. The completed-run '
          'histogram is subject to finite-window boundary selection; the matched control '
          'uses the same selection. These are reference values, not confidence intervals.\n')
    s += ('\n**Finding:** Widely spaced native samples resemble the control on these low-order diagnostics. '
          'Adjacent native samples retain nearly balanced bits but exhibit larger temporal deviations and lower '
          '8-bit block entropy. The same direction appears in the fixed validation seeds. This is a descriptive '
          'comparison, not a calibrated multiple-testing rejection threshold.\n\n'
          'The lag statistic is the average product of signs (zero -> -1, one -> +1), not centered Pearson '
          'correlation. Entropies are plug-in histogram estimates. High first-order conditional entropy does '
          'not rule out predictability using more history or neighboring atoms; case study 3 demonstrates exactly that.\n\n'
          'Adjacent traces reuse overlapping source regions, so their bits are not independent replicates. '
          'Spaced limbs are about 122,070 words apart, while an 8,192-step observation window accesses about '
          '128 preceding words per limb. This difference in source overlap helps explain sensitivity to geometry; '
          'it is not evidence of a physical interaction law.\n\n'
          'Per-atom bias ranges/standard deviations and paired-atom correlation appear for every run in '
          '`summary.csv`. Completed internal run-length histograms and all 256 block counts are preserved '
          'in raw JSON. The first and final run per trace are boundary-censored; run bin 128 means >=128.\n')
    s += common
    (OUT/'01-temporal-statistics.md').write_text(s,encoding='utf-8')

    s = '# Case study 2 — Event waiting times and a discrete decay analogue\n\n'
    s += ('**Question:** Does a fixed rare-event rule yield independent-event waiting times? '
          'One trial consists of eight nonoverlapping temporal bits; `11111111` is an event. '
          'For independent fair bits, p=1/256, first-event survival is S(k)=(255/256)^k, '
          'and the hazard per trial is 1/256. Each atom has 1,024 trials. '
          'No time scale or event probability was fitted to the observations.\n\n'
          '| Group | Layout | Source | Maximum absolute survival deviation | First-event censored atoms / 1024 | Event counts |\n'
          '|---|---|---|---:|---:|---:|\n')
    for stage,layout,source,rs in groups:
        s += f'| {stage} | {layout} | {source} | {interval(rs,"survival_max_deviation",5)} | {interval(rs,"first_event_censored",0)} | {interval(rs,"events",0)} |\n'
    s += ('\n**Finding:** Spaced native traces give first-event survival curves relatively close to the '
          'independent-event reference at this sample size. Adjacent native traces show seed-dependent deviations, '
          'including larger validation deviations than the matched controls. A balanced overall bit count '
          'therefore does not ensure the selected event behaves independently.\n\n'
          'First-event bin zero stores right-censored atoms. Survival retains these atoms in the denominator; '
          'it never renormalizes to only the atoms that had events. `summary.json` contains every survival '
          'and hazard point. The restricted mean is the sum of S(k) for k=0..1023, estimating '
          'E[min(T,1024)] rather than an uncensored mean lifetime. Completed inter-event gaps and censored '
          'tails are recorded separately. A histogram of completed gaps alone is horizon-biased and is not '
          'used to estimate an uncensored mean.\n\n'
          'The figure uses the first predefined validation seed, 20261009. Its shaded band is a pointwise '
          'normal IID reference band for 1,024 atoms, not a simultaneous band or a coverage guarantee for '
          'dependent native atoms. We do not derive formal significance from crossing it.\n\n'
          '**Physical limit:** This is a mathematical event/first-passage experiment. No isotope, energy, '
          'coupling law, or experimental decay dataset has been specified. It demonstrates neither radioactive '
          'decay nor an explanation of a measured half-life. The independent-event reference is geometric '
          'in discrete trials; an exponential is a continuous analogue, not the distribution fitted here.\n\n'
          'References: [NIST geometric distribution](https://www.itl.nist.gov/div898/software/dataplot/refman2/ch8/geopdf.pdf), '
          '[NIST exponential survival](https://itl.nist.gov/div898/handbook/eda/section3/eda3667.htm).\n')
    (OUT/'02-waiting-times.md').write_text(s+common,encoding='utf-8')

    s = '# Case study 3 — Exact prediction from neighboring history\n\n'
    s += ('**Question:** Can observed history reveal deterministic structure missed by low-order randomness tests?\n\n'
          'Let R rotate the entire integer by one bit and define D(t)=S(t+1) XOR S(t). '
          'The fixed masks cancel, giving D(t+1)=R D(t). After 64 steps, limb i receives '
          'the old difference at limb i-1. For bit zero of those limbs, this yields:\n\n'
          '```text\nb_i(t+1) = b_i(t) XOR b_(i-1)(t-63) XOR b_(i-1)(t-64), t >= 64\n```\n\n'
          'Every input is available at or before time t. This is a direct algebraic predictor, '
          'with no training, fitted parameters, seed access, or mask access. We use 512 disjoint '
          'pairs per run and 8,127 predictions per pair.\n\n'
          '| Group | Layout | Source | Correct / total predictions | Accuracy |\n|---|---|---|---:|---:|\n')
    for stage,layout,source,rs in groups:
        correct=sum(r['predicted'] for r in rs);total=sum(r['predictions'] for r in rs)
        s += f'| {stage} | {layout} | {source} | {correct:,} / {total:,} | {correct/total:.8%} |\n'
    s += ('\n**Finding:** The predictor is exact on adjacent native limbs in both seed groups, while '
          'the control remains near chance. Applying the same formula to widely spaced pairs, where '
          'the neighbor assumption is false, also gives near-chance accuracy. This is a geometry-dependent '
          'structural failure of an IID interpretation. It does not contradict the high marginal entropy '
          'or almost balanced bits in case study 1.\n\n'
          'The predictor requires neighboring histories; it does not establish that an observer restricted '
          'to a single isolated bit can always predict that bit. The proof applies to this recurrence, '
          'not to all deterministic models. Adjacent native predictions are mutually dependent, so the '
          'count is not treated as millions of independent statistical trials.\n')
    (OUT/'03-structural-prediction.md').write_text(s+common,encoding='utf-8')

    assert all(r['final_return']==r['midpoint_correct']==1 and r['hamming_min']==r['hamming_max']==1 for r in structure)
    s = '# Case study 4 — Perturbation propagation and a cycle bound\n\n'
    s += ('**Question:** Does a large state produce spreading perturbations or a correspondingly enormous orbit?\n\n'
          'We evolved complete rings of 1, 2, 3, 7, and 17 limbs for all eight fixed seeds. '
          'For each, a second initial state differs by one bit and evolves under the **same mask**. '
          'Across all 40 cases and every tested time through 2N steps, Hamming distance remains exactly one. '
          'There is no spreading of this perturbation. Changing the seed would also change the mask, '
          'and would be a different experiment.\n\n'
          '| Ring bits N | Cases | Measured first-return times | Distance range | S(N) parity rule | S(2N)=S(0) |\n'
          '|---:|---:|---|---|---|---|\n')
    for n in sorted({r['bits'] for r in structure}):
        rs=[r for r in structure if r['bits']==n]
        s+=f'| {n} | {len(rs)} | {", ".join(map(str,sorted({r["first_return"] for r in rs})))} | 1–1 | all pass | all pass |\n'
    s += ('\n**General proof:** R^N is the identity. After N steps, every mask bit has been XORed '
          'into every position exactly once, so the accumulated mask is either all zeros or all ones, '
          'according to mask parity. Thus S(N)=S(0) XOR parity(M)*all_ones. Applying another N steps '
          'cancels that contribution and yields S(2N)=S(0). Actual periods can be proper divisors of 2N.\n\n'
          'For the decimal 1 GB configuration, N=8,000,000,000 bits, so the full-state period divides '
          '**16,000,000,000 steps**. This is an algebraic upper bound, not a measured period of the '
          'full 1 GB state. The large number of representable states does not imply this update rule '
          'explores them all. Sparse benchmark throughput must not be used to infer the runtime of '
          'materializing and evolving the full ring for this many steps.\n\n'
          '**Conclusion:** The current model is an exactly solvable affine rotation system with '
          'restricted dynamics. This is a useful negative result for the present recurrence, '
          'not a disproof of determinism or of every possible large-integer physical model.\n\n'
          'Raw results: [structure.csv](raw/structure.csv). The proof above is derived from the '
          'implemented recurrence; it is not attributed to an external physics source.\n')
    (OUT/'04-perturbation-and-period.md').write_text(s,encoding='utf-8')

    s = '''# 1GBInteger — Recorded case studies

Date: 2026-10-09 (Australia/Sydney).

**The current rule can look random under sparse sampling while remaining exactly predictable from neighboring history.**
All 32 predefined temporal runs and 40 small-ring structural cases completed.
The run suite processed 268,435,456 bits: 134,217,728 native observations and the
same number of control observations. The control ignores layout, so its two
layout runs per seed repeat the same data; they are not independent replicates.

1. [Temporal appearance and sampling geometry](01-temporal-statistics.md)
2. [Event waiting times and a discrete decay analogue](02-waiting-times.md)
3. [Exact prediction from neighboring history](03-structural-prediction.md)
4. [Perturbation propagation and a cycle bound](04-perturbation-and-period.md)

![Validation results](figures/case-studies.png)

The low-order distribution results are descriptive. The exact predictor and
cycle bound follow from the recurrence and supply stronger structural evidence.
No genuine quantum equivalence or physical radioactive-decay model is established.
As [NIST explains](https://csrc.nist.gov/pubs/sp/800/22/r1/upd1/final), statistical
testing cannot by itself certify a random generator.

## Reproduction and evidence

- [Protocol recorded before collection](PROTOCOL.md)
- [Every run's scalar measurements](summary.csv)
- [Detailed derived data](summary.json), including survival and hazard curves
- [Source/executable hashes, commands and output hashes](manifest.json)
- [Native raw counts and full-ring results](raw/)

Native executables compute all observations and diagnostic counts. Python only
orchestrates processes and analyzes the saved aggregate counters. Each worker
keeps two bounded observation windows, not the full integer or whole population
history. The source fixes the observed bit at zero for an explicit spatial
comparison; this is different from the throughput benchmark's j modulo 64 bit mapping.

All native counters were independently checked against a small fully evolved
state, and exact results were verified across 1, 2, 3, and 16 requested threads.
The existing temporal and lazy correctness suites remain separate checks.

```powershell
clang++ -O3 -std=c++17 -pthread temporal_experiments.cpp -o temporal_experiments.exe
clang++ -O3 -std=c++17 -pthread structural_experiments.cpp -o structural_experiments.exe
python run_experiments.py
python analyze_experiments.py
# Optional standalone charts: install matplotlib, then
python analyze_experiments.py --plots
python test_experiments.py
```

On Linux omit `.exe` from output names. The lab regression test also requires
the full and lazy executables described in the main README. Raw counts reproduce
exactly; runtime, platform metadata, and executable hashes may differ by compiler
and operating system. The manifest records the pre-change base commit plus exact
source hashes, because data collection preceded the commit containing these studies.

Native instrumented-run timings are saved for provenance, not presented as a
new throughput benchmark: each configuration was run once and the timings are short.
These runs include statistics collection and cannot be compared directly to the
aggregate-only performance benchmark.

## Implication for further work

Retain this recurrence as an exact baseline. A more expressive model would need
a separately specified transition and observation law with independently tested
predictions. Adding nonlinear mixing may remove these particular shortcuts but
could also invalidate sparse evaluation; it would still require a physical model
and evidence, rather than just better-looking random-number statistics.
'''
    (OUT/'README.md').write_text(s,encoding='utf-8')
    if plots:
        plot(results,structure)
    print('Verified provenance and wrote four case studies, index, summary CSV and JSON.')


def plot(results,structure):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    colors={'spaced':'#2077b4','adjacent':'#d04b28','control':'#555b66'}
    selected=[r for r in results if r['seed']=='20261009']
    # First predefined validation seed, not selected by its measured outcome.
    for layout in ('spaced','adjacent'):
        r=next(r for r in selected if r['source']=='native' and r['layout']==layout)
        axes[0,0].plot(r['lags'],r['correlations'],'o-',label=f'Native {layout}',color=colors[layout])
        axes[0,1].plot(r['survival'],label=f'Native {layout}',color=colors[layout])
    control=next(r for r in selected if r['source']=='control')
    axes[0,0].plot(control['lags'],control['correlations'],'o--',label='MT19937-64 control',color=colors['control'])
    axes[0,0].axhline(0,color='black',lw=.7);axes[0,0].set_xscale('log',base=2)
    axes[0,0].set(title='Temporal sign-product correlation',xlabel='Lag (evolution steps)',ylabel='Average sign product')
    axes[0,0].legend(fontsize=8)
    x=np.arange(1025);reference=(255/256)**x;band=1.96*np.sqrt(reference*(1-reference)/1024)
    axes[0,1].fill_between(x,np.maximum(0,reference-band),np.minimum(1,reference+band),color='#bbbbbb',alpha=.35,label='Pointwise 95% IID reference band')
    axes[0,1].plot(x,reference,'k--',label='Geometric reference')
    axes[0,1].plot(control['survival'],color=colors['control'],alpha=.7,label='MT19937-64 control')
    axes[0,1].set(title='First-event survival: eight ones per trial',xlabel='Nonoverlapping eight-step trials',ylabel='Fraction with no event yet',ylim=(0,1))
    axes[0,1].legend(fontsize=7)
    validation=[r for r in results if r['group']=='validation']
    labels=['Native\nadjacent','Native\nspaced','Control']
    series=[[r for r in validation if r['source']=='native' and r['layout']==l] for l in ('adjacent','spaced')]
    series.append([r for r in validation if r['source']=='control' and r['layout']=='spaced'])
    values=[sum(r['predicted'] for r in rs)/sum(r['predictions'] for r in rs) for rs in series]
    bars=axes[1,0].bar(labels,values,color=[colors['adjacent'],colors['spaced'],colors['control']])
    axes[1,0].bar_label(bars,labels=[f'{v:.3%}' for v in values],padding=4)
    axes[1,0].axhline(.5,color='black',ls='--',lw=.8)
    axes[1,0].set(title='Neighbor-history predictor: all validation seeds',ylabel='Correct next-bit fraction',ylim=(0,1.12))
    for parity,color in [(0,colors['spaced']),(1,colors['adjacent'])]:
        rs=[r for r in structure if r['mask_parity']==parity]
        axes[1,1].scatter([r['bits'] for r in rs],[r['first_return'] for r in rs],color=color,label=f'Mask parity {parity}',s=42)
    xx=np.array([64,1088]);axes[1,1].plot(xx,xx,'k:',label='N steps');axes[1,1].plot(xx,2*xx,'k--',label='2N upper bound')
    axes[1,1].set(title='Full-ring first return: all 40 cases',xlabel='Ring size N (bits)',ylabel='First return (steps)')
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('1GBInteger | Random-looking samples, exact structural predictability\nTop panels: predefined validation seed 20261009',fontsize=14)
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'case-studies.png',dpi=180)
    fig.savefig(dest/'case-studies.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plots',action='store_true')
    main(parser.parse_args().plots)

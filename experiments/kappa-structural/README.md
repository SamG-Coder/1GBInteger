# Conditional integer multiplication: structural search case study

On 2026-10-09, this experiment certified

`kappa = 236077489139 / 500000000000000 = 0.000472154978278`

under the inherited conditional framework for
`T(n) = O(n (log n)^(1-kappa))`.

This is a **parameter-only improvement over the published PR151 value**,
not a newly discovered circuit or a proof of linear-time multiplication.
The structural searches below found no additional improvement. They do not
establish global optimality of the circuit, schedule, or research programme.

## Comparison and provenance

| Construction / pricing | Conditional kappa |
|---|---:|
| PR150, published headline, atom 1/1000 | 0.000471809569 |
| PR150, published atom 1/2000 | 0.000472026274 |
| Previous local parameter refinement on PR150 | 0.000472038214820 |
| PR151 composition, atom 1/2000 | 0.000472143085 |
| PR151 published best, atom 0.000473 | 0.000472154791 |
| This experiment, same PR151 row, refined parameters | **0.000472154978278** |

The absolute increase over PR151 is `0.000000000187278`, or
**0.0000396645345%**. It is too small to suggest a practical multiplication
speedup. Increasing kappa reduces the logarithmic exponent; this experiment
does not benchmark multiplication throughput.

Pinned upstream references:

- [PR150, DaysSky](https://github.com/CrocSwap/integer-mult-bounds/pull/150),
  `40d4038760ebb6d6d3d702ce88fa3645de29f0c1`: recycled circuit, exact frame
  and identity checkers, pricing.
- [PR146, Th0rgal](https://github.com/CrocSwap/integer-mult-bounds/pull/146),
  `70355a3192028852598e96624ae29b389e664451`: additional gauge selection and
  telescoping interval min-cut idea.
- [PR151, SovereignSteak](https://github.com/CrocSwap/integer-mult-bounds/pull/151),
  `5a46ac37ae2e2f8f75637cb2cf6b9d0b36afc436`: same gauge/recycling composition,
  published kappa 0.000472154791.

The local search first combined PR146's gauges with PR150's handoffs. A live
upstream check then found PR151 already publishing that composition. The
reconstructed row equals PR151's row exactly as parsed JSON. We claim no
priority or novelty for it. The preceding framework and word credits include
icekylinx, hpst3r, jamesyc, an664, gupt1156, Swapnil Jain and Zhihao Chen.

## What was tried

1. **Greedy gauge removal.** Independently trace target visits, including
   repeated frames, and score the actual auxiliary/gauge/target cost of deleting
   one selected read. Neither PR150 nor the composed row has a profitable
   single removal at the search objective.
2. **Compose gauge selections.** Add the 187 PR146-selected gauges missing
   from PR150. All 3,338 reuse pairs survive, with 23,979 physical registers,
   W=27,521, m=69, deficit=2,024, and maximum child rank 60. This reproduces PR151.
3. **Joint gauge removal.** A rounded-capacity interval closure cut, segmented
   at mandatory recycled reads and redirected writes, has 8,786 nodes. It
   removes no gauges from the composed row.
4. **All-gauge cut search.** Enable all 6,678 eligible first-occupant gauges
   while keeping the handoffs fixed; the timetable and ledger remain legal.
   A 35,815-node cut removes 1,835, returning the same composed row. The cut
   uses floating-point scores rounded to integer capacities at scale 10^9;
   this is a search result, not an exact global-optimality certificate.
5. **Expanded reuse matching.** Rebuild donor/recipient compatibility for
   8,181 selected deferred slots. Screen 666,715 legal-time candidate edges
   by frame containment modulo 1,048,573 and maximize the PR150 moment-saving
   objective at four savings: 0.000472806534, 0.000473, 0.00048, 0.00046.
   Every result has 3,338 handoffs and exactly the same child histogram.
   Each proposal reconstructs its full ledger; because none improves the
   row, those alternative matchings were not separately certified over Q.
6. **Exact parameter refinement.** Reprice the composed row on a 10^-15 grid;
   choose the first grid atom strictly above `coarse/(1+coarse-old)` and
   eta=10^-14. This is the only gain beyond PR151.

The matching run took about 246 seconds on this machine. Its donor matrix
occupied 77,393,344 bytes (73.8 MiB), and its assignment matrix occupied
525,887,904 bytes (501.5 MiB). These are array sizes, not peak process memory.
No 12 GB BigInteger was allocated: this bounded search did not require one.
The existing rotation/XOR universe engine is not an arbitrary multiplication
algorithm and is not the source of this exponent improvement.

## Verification actually completed

`verify.py` rebuilds the literal ledger using the unchanged pinned upstream
modules and verifies the saved plan and row. It then checks:

- All **4,878 newly required frame transitions** are nested exactly over Q
  and their endpoints are nondegenerate: 3,338 handoffs, 605 target n-to-v
  moves, and 935 target sigma-to-v moves.
- The complete scalar identity holds over **F2 and Z** on **all 27,521 formal
  inputs**, including dirty scratch and targets. This uses symbolic forms,
  not only random vectors.
- Five additional scalar replays pass. Stale-read, premature-read and
  incorrect inverse-order corruptions all fail.
- All **47 strict assembly constraints and seven margins** are positive.
  The next 10^-15-grid kappa is rejected.

The new-move geometry and replay run took about 182 seconds. `--all`, which
also checks every inherited move, was not run here. Inherited frame audits
and the analytic, precision, weighted-bit, restored-row, tape and uniformity
contracts remain dependencies. This is not a new Lean theorem or an
unconditional integer multiplication theorem.

`check_results.py` is an independent offline arithmetic check: it reconstructs
the child histogram, computes rational upper bounds for the contaminated
moment, rejects the next coarse grid point, and invokes our previously frozen
independent assembly checker. It recomputes all 47 slacks and seven margins,
rejects three parameter corruptions, and verifies SHA-256 evidence bindings.
It reads the saved geometry/replay receipt; CI does not claim to rerun geometry.

The refined coarse saving is `46172513/97656250000`; atom is
`236300630177/500000000000000`; eta is `1/100000000000000`.
The next kappa `0.000472154978279` fails the assembly's strict absorption margin.
That rejection concerns this certificate and these parameters, not all possible
proofs. The existing assembly family also retains its previously documented
`kappa < 1/34` ceiling; approaching kappa=1 requires changing that framework.

## Reproduction

Use Python 3 with assertions enabled. The verification and offline checker use
the standard library. Matching needs NumPy/SciPy; the cut search needs NetworkX.
Measured versions: Python 3.14, NumPy 2.5.3, SciPy 1.18.1, NetworkX 3.7.

The ignored upstream checkout is `.local-llm/integer-mult-research`. Keep it
detached at PR150's exact pinned commit and unmodified; fetch PR146 and PR151
objects into that repository without changing HEAD. A fresh setup is:

```powershell
git clone https://github.com/CrocSwap/integer-mult-bounds .local-llm/integer-mult-research
git -C .local-llm/integer-mult-research fetch origin refs/pull/150/head
git -C .local-llm/integer-mult-research checkout --detach 40d4038760ebb6d6d3d702ce88fa3645de29f0c1
git -C .local-llm/integer-mult-research fetch origin refs/pull/146/head
git -C .local-llm/integer-mult-research fetch origin refs/pull/151/head
python -m pip install numpy scipy networkx
python experiments/kappa-structural/search.py
python experiments/kappa-structural/cut_search.py
python experiments/kappa-structural/cut_search.py --all-gauges
python experiments/kappa-structural/rematch.py
python experiments/kappa-structural/verify.py
```

Those commands regenerate evidence and elapsed times, so the archived manifest
will deliberately stop matching. To check the unchanged published archive:

```powershell
python experiments/kappa-structural/check_results.py
```

`freeze.py` creates a new manifest only after deliberate review of a new run.
Raw search results, alternative plans/rows, verification receipt, logs and
upstream source hashes are retained alongside this case study.

## Licence and next research boundary

New experiment drivers in this repository are MIT-licensed. Upstream code is
imported from an ignored, unmodified checkout and retains its Apache-2.0
notices. The recorded circuit plans and rows originate from the credited
upstream constructions; this repository does not claim to relicense them.

A further structural attempt must change something these searches fixed:
the operation/read order, the terminal-elimination choice, the circuit itself,
or the assembly. More memory alone changes none of those. Exact geometry and
complete dirty-register identity checks must accompany any changed word.

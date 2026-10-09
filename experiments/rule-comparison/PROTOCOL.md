# Fixed protocol: evolution-rule comparison

Recorded before candidate results, 2026-10-09. This is a local prospective
protocol, not an externally registered study. No rule constants or tests will
be tuned after observing results. The existing engine and studies remain intact.

## Equations

Limbs are little-endian 64-bit words; all limb additions wrap modulo 2^64.
R is a one-bit left rotation of the complete N-bit ring. `rot_w` rotates one
word, with the count reduced modulo w. M and S(0) use the existing seed hashes.

1. **xor** (baseline): `S' = R(S) XOR M`.
2. **add**: `S' = (R(S) + M) modulo 2^N`. Carry propagates from the least
   significant word to the most significant word; final overflow is discarded.
   This is whole-integer addition, not independent limb additions.
3. **local**: synchronous update from an unchanged source buffer:
   `S'_i = rot_64(S_i + rot_64(S_(i-1),17) + M_i,23) XOR S_(i+1)`.
   Neighbors wrap around the ring. Constants 17 and 23 are fixed design choices,
   not optimized against these tests. This rule is not assumed to be invertible.

The add rule is a permutation (rotation and addition by a constant are both
invertible). This does not imply long cycles or statistical independence.

## Fixed workloads

- Development seeds: 0, 1, 42, 12345.
- Validation seeds: 20261009, 314159265, 2718281828, 18446744073709551615.
- Temporal experiments: materialized 32,768-byte and 1,000,000-byte states;
  1,024 observed atoms, 8,192 consecutive observations, bit zero, both adjacent
  and evenly spaced limbs. All three rules use identical initial state and mask
  for a given seed and size. One native CPU thread for full-state evolution.
- Diagnostics: same lag list, bias, block and conditional entropy, completed
  internal runs, disjoint-pair cross correlation, original neighboring-history
  predictor, nonoverlapping eight-one events, first-event survival, discrete
  hazards, complete event gaps and right censoring as the first studies.
- Reuse the first studies' eight spaced MT19937-64 control files, one per seed.
  Controls are identical across geometry/size and must not be counted as new
  independent replicates. No claims of physical quantum reference data.
- These state sizes differ from the previous logical 1 GB sparse experiment.
  Compare rules within this suite, not against historical different-size runs.
  There is no burn-in; initial transient behavior is part of the specified test.
- Perturbations: materialize 1, 3, 17 and 256 limbs; flip bit zero of limb zero
  under the same mask; track every step through 4,096, Hamming distance and
  changed-word count, and first return to the initial state within that horizon.
- Exhaustive graphs: reduced-word analogues (w,limbs) = (4,1),(4,2),(4,3),
  (8,1),(8,2), for every seed/rule. Mask words are truncated to w bits; rotations
  are reduced modulo w. Enumerate all 2^(w*limbs) states, every edge, distinct
  images, maximum preimage count, cycle count/length range, cyclic-state count,
  and maximum transient depth. These are analogues, not exhaustive 64-bit tests.
- Matched performance: materialize 1 MB, 16 MB and 1 GB (decimal), seed 12345,
  three full steps per fresh-process run, three repetitions. Each run has a
  source, destination and cached mask buffer (3x logical bytes); no trace output.
  All rules use the same compiler flags, one CPU thread and full-update workload.
  Rotate rule order between repetitions. Record init, per-step and aggregate
  wall time, process CPU time during evolution, peak resident memory and checksum.
  Report medians and raw runs; short tests and active-desktop timing have noise.

## Correctness and interpretation

Before collection, check all three native kernels against independent Python
whole-integer/word reference formulas on edge cases and multiple steps. Check
all diagnostic counters from a small materialized trace, exhaustive graph
statistics against an independent graph traversal, and perturbation distances.
Use undefined-behavior sanitization for native small tests. Python is only the
test oracle, orchestration and report layer, not the native computation loop.

Publish all seeds and all candidate failures. Failure of the old predictor is
not proof of unpredictability. Hamming spreading is not a quantum phenomenon.
A collision in a reduced-word analogue is not a collision proof for 64-bit
words. No return within 4,096 steps is a censored observation, not proof of a
long period. Independent-event reference bands are descriptive for dependent
atoms; do not report uncorrected multi-test significance or call these tests a
physical decay/entanglement experiment.

# 1GBInteger temporal experiment protocol

Written before collecting this suite's results, 2026-10-09. This is a local
prospective protocol, not an independently registered study.

## Fixed design

- Recurrence: `S(t+1) = ROTL(S(t),1) XOR M`; unchanged native temporal kernel.
- Logical size: 1,000,000,000 bytes. Observe bit zero of each selected limb.
- 1,024 atoms; 8,192 consecutive observations starting at t=0 (last t=8,191).
- Layouts: evenly spaced `floor(j*limbs/atoms)` and adjacent limbs `j`.
- Development seeds: 0, 1, 42, 12345.
- Validation seeds, evaluated with the same settings: 20261009, 314159265,
  2718281828, 18446744073709551615.
- Reference: C++ `std::mt19937_64`, independently seeded per atom with
  `mix(seed XOR (j*C))`, extracting each word's bits low to high. This is a
  deterministic pseudorandom control, not measured physical quantum data.
- Eight worker threads. Collect each layout/seed for both native and control.
- No tuning of layouts, seeds, event rule, or tests after examining results.

## Case studies

1. Temporal appearance: pooled/per-atom ones, same-atom sign-product correlations
   at lags 1,2,4,8,16,32,63,64,65,128,256; disjoint atom-pair correlation;
   8-bit nonoverlapping block counts and empirical Shannon entropy;
   first-order conditional entropy; completed internal run lengths. First and
   last runs are boundary-censored and excluded from the run histogram.
2. Waiting times: one trial = eight nonoverlapping temporal observations. An
   event is `11111111`, giving p=1/256 under independent fair bits. Record first
   events, completed inter-event gaps, and right-censored tails. Compare first
   event survival to `(255/256)^k` at all k=0..1024; report maximum absolute
   deviation and discrete hazard with at-risk counts. Do not treat overlapping
   patterns as independent trials. No fitted half-life or physical time unit.
3. Structural prediction: for adjacent limbs, test
   `b_i(t+1) = b_i(t) XOR b_(i-1)(t-63) XOR b_(i-1)(t-64)` for t>=64.
   Derivation: consecutive-state XOR differences rotate by one bit per step.
   Report exact prediction accuracy and compare with the matched control.
4. Perturbation and recurrence: materialize rings of 1,2,3,7,17 limbs. Evolve
   two initial states differing by one bit under the same fixed mask. Measure
   Hamming distance through 2N steps, N=64*limbs. Check S(2N)=S(0), and check
   S(N) equals S(0) or its complement according to mask parity. Record first
   return time; this is measured on small rings, not by iterating a 1 GB ring.

## Interpretation and reporting

Publish all seeds, including unfavorable findings, raw integer counters,
source/executable hashes, commands, timings, and reproducible analysis.
Report seed-level ranges separately for development and validation. Pooled
sign-product correlation is an uncentered statistic (zero expected under IID
fair bits), not Pearson correlation. Entropies are finite-sample descriptive
estimates, not entropy-rate estimates or new physical randomness.

For first-event survival, show the pointwise normal IID reference band
`S +/- 1.96*sqrt(S*(1-S)/n)` clipped to [0,1]. It is a reference band, not a
simultaneous confidence band or valid coverage guarantee for dependent model
atoms. Do not declare significance from the largest of many uncorrected tests.
A diagnostic failure rejects the tested IID description; a diagnostic match
does not establish IID output, quantum equivalence, or radioactive decay.

Sources: [NIST geometric distribution](https://www.itl.nist.gov/div898/software/dataplot/refman2/ch8/geopdf.pdf)
uses failures before success; here waiting times count trials through success.
[NIST exponential survival](https://itl.nist.gov/div898/handbook/eda/section3/eda3667.htm)
provides the continuous comparison; our experiment is discrete.
[NIST statistical-testing limitations](https://csrc.nist.gov/pubs/sp/800/22/r1/upd1/final)
explain why a battery of statistical checks cannot certify a generator.

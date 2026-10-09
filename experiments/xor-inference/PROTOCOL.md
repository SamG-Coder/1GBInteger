# XOR inference protocol

Fixed before collection, 2026-10-09. Both exact hidden-state inference and
learned prediction are requested. The existing physics/randomness studies and
engine remain unchanged. This suite uses bounded synthetic problems to test
specific uses of XOR; it is not an LLM or real-world recognition benchmark.

## Seeds and separation

Development seeds: 0,1,42,12345. Validation seeds: 20261009,314159265,
2718281828,18446744073709551615. Every method/configuration uses both groups;
no configuration is selected based on validation performance. Simulation truth
and training labels are available to evaluation, never as hidden inputs to the
solver. Gaussian elimination is over GF(2), using packed C++ uint64_t rows.

## A. Exact state and mask inference

Dynamics: S(t+1)=ROTL(S(t),1) XOR M. Materialized rings of 64,256,1024 bits.
Initial state and mask are generated from independently salted deterministic
streams, with no assumption available to the solver that they share a 64-bit
seed. Unknowns are arbitrary initial bits, or arbitrary initial and mask bits.

Observed bit positions are floor(j*N/sensors). Observation times start at zero.
Known-mask cases (sensors,horizon): (1,N/2),(1,N),(4,N/4),(16,N/16),(N,1).
Unknown-mask cases: (1,N),(1,2N),(4,N),(N,2).

Record equation count, rank, contradictions, initial-bit and mask-bit uniquely
determined counts/accuracy, full-state forecast at t=horizon, and observed-site
forecasts over the next 32 times. A query is answered only when its value is
identical in every solution of the observation equations. Otherwise abstain;
never count an arbitrary zero-fill solution as recovered truth. Conflicting
observations invalidate exact inference rather than silently selecting a subset.

## B. Noise and redundancy

Known mask, 256 bits, one sensor, horizons 256 and 1024. Independently flip
each measured bit with p=0.01 or p=0.05. Compare a single reading against a
majority vote of five independently corrupted readings of the same bit. Record
actual raw/voted errors, contradictions and wrong uniquely inferred bits.
Repetition assumes independent measurement errors; no general error-correcting
decoder or noise-tolerant parity learner is claimed.

## C. Learned Boolean inference

Input is 16 bits. Shuffle all 65,536 inputs using a fixed per-seed Fisher-Yates
permutation; train on prefixes of 64,256,1024 inputs. Evaluate on the next
8,192 inputs starting at offset 1024, disjoint from every training prefix.
Training uses only features and labels; evaluation uses clean held-out labels.

Fixed tasks: parity of bits {0,2,5,7,11,15} XOR 1; AND(x0,x1);
majority(x0,x1,x2); mux(x0 selects x2 vs x1); bit 2 of the sum of two unsigned
3-bit numbers (x0..2 and x3..5); carry bit 3 of that same sum.

Models: (1) affine XOR: constant + 16 inputs; (2) affine XOR reservoir:
constant + 64 deterministic parity projections of the inputs, evolved 17 steps
under the existing 64-bit rotation/XOR rule with a fixed mask; (3) all monomials
up to degree 2 (137 features); (4) all monomials up to degree 3 (697 features).
Monomials use AND to form features, followed by a learned XOR readout. Models
3 and 4 are explicitly not XOR-only. The reservoir remains affine.

Fit coefficients by GF(2) elimination. Report rank, exact-fit consistency,
training accuracy and held-out accuracy. For consistent underdetermined fits,
report both accuracy of the particular solution (free coefficients zero) and
coverage/accuracy of predictions invariant across all fitting models. For
inconsistent systems, abstain rather than report an invalid model as learned.
Include the training-majority constant classifier as a baseline.

Additionally repeat the 1024-example affine parity fit with 1% and 5% label
flips and with 1 or 5 repeated readings. Never use clean labels to repair fits.
Store learned coefficient bitsets for reproducible inference.

## D. Inference cost and binary dot products

For clean, consistent 1024-example models, measure three repetitions of the
8,192-example held-out batch. Include feature encoding and learned readout,
exclude label generation, scoring and unique-prediction certification. Preserve
a checksum to prevent dead-code elimination. Report native wall time, not a
comparison with Python. No accuracy claim is inferred from timing.

Separately compare a scalar signed dot product with the exact packed identity
`dot(a,b)=N-2*popcount(a XOR b)` for bipolar {-1,+1} values, N=1024,
4096 input/weight pairs, three repetitions, packing outside the kernel timer
and reported separately. Require every output to match. This is an inference
primitive using random operands, not a trained binary neural network. Generic
x86-64 compiler flags only; do not assume hardware POPCNT or publish speedup
claims without matching measured outputs and workload.

## Limits and references

Known-mask observations can be much easier than joint state/mask identification.
Nonunique parameters may still permit unique forecasts at observed sites.
High training accuracy with excessive features may not generalize. XOR-only
composition is affine and cannot represent arbitrary Boolean functions. These
limits are part of the experiment, not failures to hide.

For context, [Blum, Kalai and Wasserman](https://arxiv.org/abs/cs/0010022)
study noise-tolerant parity learning; our exact elimination is not their
noise-tolerant algorithm. [XNOR-Net](https://arxiv.org/abs/1603.05279) demonstrates
binary operations in learned vision models; our dot-product microbenchmark
does not reproduce that network, dataset, training, accuracy or speedup.

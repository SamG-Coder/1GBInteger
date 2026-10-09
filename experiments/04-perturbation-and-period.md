# Case study 4 — Perturbation propagation and a cycle bound

**Question:** Does a large state produce spreading perturbations or a correspondingly enormous orbit?

We evolved complete rings of 1, 2, 3, 7, and 17 limbs for all eight fixed seeds. For each, a second initial state differs by one bit and evolves under the **same mask**. Across all 40 cases and every tested time through 2N steps, Hamming distance remains exactly one. There is no spreading of this perturbation. Changing the seed would also change the mask, and would be a different experiment.

| Ring bits N | Cases | Measured first-return times | Distance range | S(N) parity rule | S(2N)=S(0) |
|---:|---:|---|---|---|---|
| 64 | 8 | 64, 128 | 1–1 | all pass | all pass |
| 128 | 8 | 128, 256 | 1–1 | all pass | all pass |
| 192 | 8 | 192, 384 | 1–1 | all pass | all pass |
| 448 | 8 | 448, 896 | 1–1 | all pass | all pass |
| 1088 | 8 | 1088, 2176 | 1–1 | all pass | all pass |

**General proof:** R^N is the identity. After N steps, every mask bit has been XORed into every position exactly once, so the accumulated mask is either all zeros or all ones, according to mask parity. Thus S(N)=S(0) XOR parity(M)*all_ones. Applying another N steps cancels that contribution and yields S(2N)=S(0). Actual periods can be proper divisors of 2N.

For the decimal 1 GB configuration, N=8,000,000,000 bits, so the full-state period divides **16,000,000,000 steps**. This is an algebraic upper bound, not a measured period of the full 1 GB state. The large number of representable states does not imply this update rule explores them all. Sparse benchmark throughput must not be used to infer the runtime of materializing and evolving the full ring for this many steps.

**Conclusion:** The current model is an exactly solvable affine rotation system with restricted dynamics. This is a useful negative result for the present recurrence, not a disproof of determinism or of every possible large-integer physical model.

Raw results: [structure.csv](raw/structure.csv). The proof above is derived from the implemented recurrence; it is not attributed to an external physics source.

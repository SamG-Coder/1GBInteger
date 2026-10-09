# Case study 9 — Learning XOR readouts and testing unseen inputs

Each model is fitted from input/label pairs by GF(2) elimination. Training sets
contain 64, 256 or 1024 unique 16-bit inputs. The 8192 held-out inputs are
disjoint from every training prefix. Labels come from fixed synthetic tasks;
the predictor itself never calls the label oracle. Four development and four
validation seeds determine independent input permutations, with no model
selection or tuning on validation accuracy.

Models:

- **linear:** constant plus 16 input bits; 17 XOR coefficients.
- **reservoir:** constant plus 64 parity projections, followed by 17 steps of
  rotation/XOR evolution with a fixed mask; 65 coefficients but affine rank 17.
- **quadratic:** all 137 constant, input and pairwise-AND features.
- **cubic:** all 697 monomials through triple-AND features.

Quadratic/cubic models are **nonlinear features plus an XOR readout**, not
XOR-only networks. The target equations and monomial order are in the protocol
and native source. Coefficients are learned, not supplied from the task formula.

## 1024-example fits: fixed validation seeds

Each cell reports consistent fits out of four and held-out accuracy range when
a consistent fit exists. An inconsistent fit abstains; it is not scored as a
valid learned classifier.

| Task | Linear | XOR reservoir | Quadratic | Cubic |
|---|---|---|---|---|
| parity | 4/4; 100.00–100.00% | 4/4; 100.00–100.00% | 4/4; 100.00–100.00% | 4/4; 100.00–100.00% |
| AND | 0/4; no exact fit | 0/4; no exact fit | 4/4; 100.00–100.00% | 4/4; 100.00–100.00% |
| majority of 3 | 0/4; no exact fit | 0/4; no exact fit | 4/4; 100.00–100.00% | 4/4; 100.00–100.00% |
| selector | 0/4; no exact fit | 0/4; no exact fit | 4/4; 100.00–100.00% | 4/4; 100.00–100.00% |
| addition bit 2 | 0/4; no exact fit | 0/4; no exact fit | 0/4; no exact fit | 4/4; 100.00–100.00% |
| addition carry bit 3 | 0/4; no exact fit | 0/4; no exact fit | 0/4; no exact fit | 0/4; no exact fit |

The linear learner recovers parity perfectly. Expanding it into the affine
reservoir does not increase its rank or allow it to learn any additional task.
Composing XORs, rotations and constants cannot create nonlinear functions of
the input. A large deterministic state is not, by itself, extra model capacity.

AND features change the function class: quadratic features fit AND, three-bit
majority and the selector, while cubic features also fit bit 2 of addition.
The final carry bit requires degree four in this formulation and is a deliberate
negative control: none of these model classes fits its 1024 clean examples.

## Interpolation is not generalization

For each consistent underdetermined system, the evaluated model sets free
coefficients to zero in the fixed elimination order. It is one arbitrary
interpolant, not a regularized or sparsity-optimized solution. Independently, we
certify which held-out predictions are identical across all fitting models.

| Model/task | Training examples | Exact training fits / 4 | Held-out accuracy range | Certified held-out predictions / 8192 |
|---|---:|---:|---:|---:|
| linear / parity | 64 | 4 | 100.00–100.00 | 8192–8192 |
| linear / parity | 256 | 4 | 100.00–100.00 | 8192–8192 |
| linear / parity | 1024 | 4 | 100.00–100.00 | 8192–8192 |
| quadratic / majority of 3 | 64 | 4 | 49.41–50.16 | 0–0 |
| quadratic / majority of 3 | 256 | 4 | 100.00–100.00 | 8192–8192 |
| quadratic / majority of 3 | 1024 | 4 | 100.00–100.00 | 8192–8192 |
| cubic / addition bit 2 | 64 | 4 | 49.01–50.72 | 0–0 |
| cubic / addition bit 2 | 256 | 4 | 49.32–50.50 | 0–0 |
| cubic / addition bit 2 | 1024 | 4 | 100.00–100.00 | 8192–8192 |
| cubic / addition carry bit 3 | 64 | 4 | 50.96–52.37 | 0–0 |
| cubic / addition carry bit 3 | 256 | 4 | 49.67–51.14 | 0–0 |
| cubic / addition carry bit 3 | 1024 | 0 | — | — |

Large feature banks can fit 64 or 256 examples perfectly while performing near
chance on new inputs and certifying none of those predictions. At 1024 examples,
the tested representable tasks reach full feature rank and generalize exactly.
The carry-bit negative control moves from interpolation to detected model
mismatch as more constraints arrive. This shows why training fit alone is an
inadequate acceptance criterion.

The training-majority constant baseline is recorded for every fit. It matters
especially for imbalanced AND labels (about 75% negatives). No failed exact
fit is silently replaced with that baseline, and no inconsistency is hidden by
dropping training constraints.

## Noisy parity training

Linear parity, 1024 training examples, all eight seeds:

| Label flip probability | Repeated labels per example | Consistent fits / 8 | Residual voted label errors |
|---:|---:|---:|---:|
| 1% | 1 | 0 | 75 |
| 1% | 5 | 7 | 1 |
| 5% | 1 | 0 | 435 |
| 5% | 5 | 3 | 9 |

Exact elimination is fragile to label noise. The repeats are an explicitly
costed independent-measurement assumption, not a general noise-tolerant learning
algorithm. [Blum, Kalai and Wasserman](https://arxiv.org/abs/cs/0010022) study the
distinct problem of learning parity with noisy labels; this implementation
does not reproduce their algorithm.

All 608 fit records, including contradictions, ranks, baseline scores, weights,
certified coverage and timings: [learning.csv](raw/learning.csv).

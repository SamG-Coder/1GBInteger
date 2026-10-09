# Case study 3 — Exact prediction from neighboring history

**Question:** Can observed history reveal deterministic structure missed by low-order randomness tests?

Let R rotate the entire integer by one bit and define D(t)=S(t+1) XOR S(t). The fixed masks cancel, giving D(t+1)=R D(t). After 64 steps, limb i receives the old difference at limb i-1. For bit zero of those limbs, this yields:

```text
b_i(t+1) = b_i(t) XOR b_(i-1)(t-63) XOR b_(i-1)(t-64), t >= 64
```

Every input is available at or before time t. This is a direct algebraic predictor, with no training, fitted parameters, seed access, or mask access. We use 512 disjoint pairs per run and 8,127 predictions per pair.

| Group | Layout | Source | Correct / total predictions | Accuracy |
|---|---|---|---:|---:|
| development | spaced | native | 8,321,513 / 16,644,096 | 49.99678565% |
| development | spaced | control | 8,322,500 / 16,644,096 | 50.00271568% |
| development | adjacent | native | 16,644,096 / 16,644,096 | 100.00000000% |
| development | adjacent | control | 8,322,500 / 16,644,096 | 50.00271568% |
| validation | spaced | native | 8,319,572 / 16,644,096 | 49.98512385% |
| validation | spaced | control | 8,322,621 / 16,644,096 | 50.00344266% |
| validation | adjacent | native | 16,644,096 / 16,644,096 | 100.00000000% |
| validation | adjacent | control | 8,322,621 / 16,644,096 | 50.00344266% |

**Finding:** The predictor is exact on adjacent native limbs in both seed groups, while the control remains near chance. Applying the same formula to widely spaced pairs, where the neighbor assumption is false, also gives near-chance accuracy. This is a geometry-dependent structural failure of an IID interpretation. It does not contradict the high marginal entropy or almost balanced bits in case study 1.

The predictor requires neighboring histories; it does not establish that an observer restricted to a single isolated bit can always predict that bit. The proof applies to this recurrence, not to all deterministic models. Adjacent native predictions are mutually dependent, so the count is not treated as millions of independent statistical trials.


Design and limitations: [prospective protocol](PROTOCOL.md). All seeds and raw counters: [summary.csv](summary.csv), [manifest](manifest.json), and [raw directory](raw/). Development and validation use four seeds each. Range tables summarize seed-level results; the prediction table pools exact counts.

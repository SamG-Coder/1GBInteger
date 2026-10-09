# Case study 1 — Temporal appearance and sampling geometry

**Question:** Do individual temporal traces and paired atoms look like independent fair bits? We measured 8,388,608 bits per run, observing bit zero over 8,192 consecutive times for 1,024 atoms. The native runs use a logical 1 GB state. The control is deterministic MT19937-64, not physical randomness.

| Group | Layout | Source | Ones fraction | Maximum absolute lag statistic | 8-bit entropy | H(next bit given current bit) |
|---|---|---|---:|---:|---:|---:|
| development | spaced | native | 0.500109–0.500203 | 0.000360–0.000869 | 7.99979–7.99984 | 0.9999997–1.0000000 |
| development | spaced | control | 0.499807–0.500224 | 0.000581–0.000713 | 7.99981–7.99986 | 0.9999998–1.0000000 |
| development | adjacent | native | 0.499806–0.500120 | 0.003795–0.007818 | 7.98850–7.98982 | 0.9999963–0.9999998 |
| development | adjacent | control | 0.499807–0.500224 | 0.000581–0.000713 | 7.99981–7.99986 | 0.9999998–1.0000000 |
| validation | spaced | native | 0.499647–0.500181 | 0.000382–0.000588 | 7.99981–7.99984 | 0.9999996–0.9999999 |
| validation | spaced | control | 0.499856–0.500394 | 0.000466–0.000810 | 7.99980–7.99982 | 0.9999996–1.0000000 |
| validation | adjacent | native | 0.499952–0.500074 | 0.004714–0.008165 | 7.98613–7.98908 | 0.9999755–1.0000000 |
| validation | adjacent | control | 0.499856–0.500394 | 0.000466–0.000810 | 7.99980–7.99982 | 0.9999996–1.0000000 |

| Group | Layout | Source | Per-atom bias SD | Paired-atom sign correlation | Fraction of complete internal runs with length 1 |
|---|---|---|---:|---:|---:|
| development | spaced | native | 0.005521–0.005674 | -0.000687–0.000465 | 0.499620–0.500520 |
| development | spaced | control | 0.005279–0.005616 | -0.000200–-0.000060 | 0.499802–0.500185 |
| development | adjacent | native | 0.003615–0.006388 | -0.000997–0.001144 | 0.499243–0.501937 |
| development | adjacent | control | 0.005279–0.005616 | -0.000200–-0.000060 | 0.499802–0.500185 |
| validation | spaced | native | 0.005414–0.005688 | -0.000914–0.000350 | 0.499621–0.500204 |
| validation | spaced | control | 0.005315–0.005497 | -0.000670–0.000020 | 0.499756–0.500269 |
| validation | adjacent | native | 0.003678–0.006088 | -0.000129–0.000635 | 0.495718–0.502916 |
| validation | adjacent | control | 0.005315–0.005497 | -0.000670–0.000020 | 0.499756–0.500269 |

Under independent fair bits the per-atom bias standard deviation is `sqrt(0.25/8192) = 0.005524`, the expected sign correlation is zero, and the untruncated run-length probability P(L=1) is 1/2. The completed-run histogram is subject to finite-window boundary selection; the matched control uses the same selection. These are reference values, not confidence intervals.

**Finding:** Widely spaced native samples resemble the control on these low-order diagnostics. Adjacent native samples retain nearly balanced bits but exhibit larger temporal deviations and lower 8-bit block entropy. The same direction appears in the fixed validation seeds. This is a descriptive comparison, not a calibrated multiple-testing rejection threshold.

The lag statistic is the average product of signs (zero -> -1, one -> +1), not centered Pearson correlation. Entropies are plug-in histogram estimates. High first-order conditional entropy does not rule out predictability using more history or neighboring atoms; case study 3 demonstrates exactly that.

Adjacent traces reuse overlapping source regions, so their bits are not independent replicates. Spaced limbs are about 122,070 words apart, while an 8,192-step observation window accesses about 128 preceding words per limb. This difference in source overlap helps explain sensitivity to geometry; it is not evidence of a physical interaction law.

Per-atom bias ranges/standard deviations and paired-atom correlation appear for every run in `summary.csv`. Completed internal run-length histograms and all 256 block counts are preserved in raw JSON. The first and final run per trace are boundary-censored; run bin 128 means >=128.


Design and limitations: [prospective protocol](PROTOCOL.md). All seeds and raw counters: [summary.csv](summary.csv), [manifest](manifest.json), and [raw directory](raw/). Development and validation use four seeds each. Range tables summarize seed-level results; the prediction table pools exact counts.

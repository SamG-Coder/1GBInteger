# Case study 2 — Event waiting times and a discrete decay analogue

**Question:** Does a fixed rare-event rule yield independent-event waiting times? One trial consists of eight nonoverlapping temporal bits; `11111111` is an event. For independent fair bits, p=1/256, first-event survival is S(k)=(255/256)^k, and the hazard per trial is 1/256. Each atom has 1,024 trials. No time scale or event probability was fitted to the observations.

| Group | Layout | Source | Maximum absolute survival deviation | First-event censored atoms / 1024 | Event counts |
|---|---|---|---:|---:|---:|
| development | spaced | native | 0.02084–0.04457 | 12–21 | 4158–4247 |
| development | spaced | control | 0.01601–0.03359 | 15–24 | 4085–4160 |
| development | adjacent | native | 0.02453–0.06398 | 4–13 | 3768–4492 |
| development | adjacent | control | 0.01601–0.03359 | 15–24 | 4085–4160 |
| validation | spaced | native | 0.02128–0.03029 | 11–22 | 4057–4222 |
| validation | spaced | control | 0.02076–0.03916 | 12–20 | 4034–4215 |
| validation | adjacent | native | 0.01988–0.10466 | 0–12 | 3695–5364 |
| validation | adjacent | control | 0.02076–0.03916 | 12–20 | 4034–4215 |

**Finding:** Spaced native traces give first-event survival curves relatively close to the independent-event reference at this sample size. Adjacent native traces show seed-dependent deviations, including larger validation deviations than the matched controls. A balanced overall bit count therefore does not ensure the selected event behaves independently.

First-event bin zero stores right-censored atoms. Survival retains these atoms in the denominator; it never renormalizes to only the atoms that had events. `summary.json` contains every survival and hazard point. The restricted mean is the sum of S(k) for k=0..1023, estimating E[min(T,1024)] rather than an uncensored mean lifetime. Completed inter-event gaps and censored tails are recorded separately. A histogram of completed gaps alone is horizon-biased and is not used to estimate an uncensored mean.

The figure uses the first predefined validation seed, 20261009. Its shaded band is a pointwise normal IID reference band for 1,024 atoms, not a simultaneous band or a coverage guarantee for dependent native atoms. We do not derive formal significance from crossing it.

**Physical limit:** This is a mathematical event/first-passage experiment. No isotope, energy, coupling law, or experimental decay dataset has been specified. It demonstrates neither radioactive decay nor an explanation of a measured half-life. The independent-event reference is geometric in discrete trials; an exponential is a continuous analogue, not the distribution fitted here.

References: [NIST geometric distribution](https://www.itl.nist.gov/div898/software/dataplot/refman2/ch8/geopdf.pdf), [NIST exponential survival](https://itl.nist.gov/div898/handbook/eda/section3/eda3667.htm).


Design and limitations: [prospective protocol](PROTOCOL.md). All seeds and raw counters: [summary.csv](summary.csv), [manifest](manifest.json), and [raw directory](raw/). Development and validation use four seeds each. Range tables summarize seed-level results; the prediction table pools exact counts.

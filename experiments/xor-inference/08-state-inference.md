# Case study 8 — Exact hidden-state inference, observability and noise

The solver receives a known transition rule, observation positions/times, and
observed bit values. It does not receive the simulator seed or hidden initial
state. In known-mask cases it additionally receives M. Joint-inference cases
treat all N initial bits and all N mask bits as independent unknowns.

The exact closed form makes each observation a linear equation over GF(2).
Packed Gaussian elimination computes its rank and tests consistency. A queried
bit is returned only if its value is invariant over **every** solution of the
equations; otherwise the solver abstains. Rank alone is not a bit-recovery count.

## Clean observations

For N=1024, the following values are identical across all eight fixed seeds.
Every answered bit agrees with the independently simulated truth. All N=64 and
N=256 results are also recorded in [state.csv](raw/state.csv).

| Mask | Sensors | Consecutive observation times | Rank | Initial bits recovered / 1024 | Mask bits recovered / 1024 | Full forecast bits determined at first future time / 1024 | Observed-site future bits determined |
|---|---:|---:|---:|---:|---:|---:|---:|
| known | 1 | 512 | 512 | 512 | given | 512 | 0/32 |
| known | 1 | 1024 | 1024 | 1024 | given | 1024 | 32/32 |
| known | 4 | 256 | 1024 | 1024 | given | 1024 | 128/128 |
| known | 16 | 64 | 1024 | 1024 | given | 1024 | 512/512 |
| known | 1024 | 1 | 1024 | 1024 | given | 1024 | 32768/32768 |
| unknown | 1 | 1024 | 1024 | 1 | 0 | 0 | 0/32 |
| unknown | 1 | 2048 | 1025 | 1 | 0 | 1 | 32/32 |
| unknown | 4 | 1024 | 1028 | 4 | 0 | 4 | 128/128 |
| unknown | 1024 | 2 | 2048 | 1024 | 1024 | 1024 | 32768/32768 |

**Finding:** With known M, one bit observed for N consecutive times recovers
all N initial bits. Four appropriately spaced sensors cover the state in N/4
times; 16 do so in N/16. This is an exact inverse calculation, not a learned
statistical guess. Half a sweep leaves half the initial state unidentified.

With unknown M, one sensor over 2N times yields rank N+1 among 2N unknowns.
Only the directly observed initial bit is uniquely recovered, and no individual
mask bit is uniquely recovered. Nevertheless the next 32 values at that sensor
are all determined and correct. Forecastability is not full identifiability.
Observing every bit at two consecutive times recovers both S(0) and M exactly.

A structural ambiguity explains this: choose an arbitrary fixed vector q that
is zero at every observed site. Replace S(0) by S(0) XOR q and M by
M XOR q XOR R(q). The entire new trajectory is S(t) XOR q, so observations at
those sites are unchanged forever. Unobserved state bits cannot be uniquely
recovered from such observations without additional assumptions.

These are 64–1024-bit inverse problems. A dense joint solver for a 1 GB state
would be a very different memory/computation problem. The seed constraint of
the original generator is deliberately not used as a shortcut.

## Measurement noise

Known mask, N=256, one sensor. The table pools counts across all eight seeds.
Repeats=5 uses majority voting on independent corrupted readings of the same
measurement. Raw flips, voted errors, and equation conflicts are preserved.

| Times | Flip probability | Readings per measurement | Consistent systems / 8 | Voted bit errors | Wrong recovered initial bits across accepted systems |
|---:|---:|---:|---:|---:|---:|
| 256 | 1% | 1 | 8 | 19 | 19 |
| 256 | 1% | 5 | 8 | 0 | 0 |
| 256 | 5% | 1 | 8 | 101 | 101 |
| 256 | 5% | 5 | 8 | 5 | 5 |
| 1024 | 1% | 1 | 0 | 78 | 0 |
| 1024 | 1% | 5 | 8 | 0 | 0 |
| 1024 | 5% | 1 | 0 | 403 | 0 |
| 1024 | 5% | 5 | 0 | 14 | 0 |

At exactly N observations, there is enough freedom to fit corrupted measurements
without contradiction. A unique algebraic solution can still be **wrong**.
Longer observation windows expose contradictions through redundant constraints;
the solver then refuses exact inference. Five repeated readings help in the
tested independent-noise setting, but residual errors remain at 5% noise.

This is not a general noisy-parity decoder. Majority voting spends five times
the measurement budget and assumes independent errors. The experiment does not
address systematic bias, correlated noise, or error correction without repeats.
The counts are finite seeded experiments, not a guaranteed error probability.

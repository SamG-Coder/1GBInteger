# Case study 5 — Temporal comparison of three rules

All rules were evaluated on the same materialized initial state and mask for
each seed/size. Both layouts observe bit zero. There are 1,024 atoms and 8,192
consecutive observations per profile, without burn-in. Unlike the first suite,
these are 32,768-byte and 1,000,000-byte states, not a logical sparse 1 GB state.

The two new equations and all constants were fixed before collection in the
[protocol](PROTOCOL.md). The formulas were not tuned after seeing results.

The old neighbor-history predictor is:
`b_i(t+1) = b_i(t) XOR b_(i-1)(t-63) XOR b_(i-1)(t-64)`.
Its exact derivation applies only to the XOR baseline. Here it is also used
unchanged as a diagnostic attack against the candidate rules.

## Every seed group, size, rule and layout

Each cell is the range across the four fixed seeds in that group. Lag statistic
means the maximum absolute uncentered sign-product correlation over the fixed
11 lags. Survival deviation is the maximum absolute distance from
`(255/256)^k`, k=0..1024, for nonoverlapping eight-one events.

| Group | Bytes | Rule | Layout | Ones fraction | Max lag statistic | Block entropy (bits/8) | Predictor accuracy | Survival deviation |
|---|---:|---|---|---:|---:|---:|---:|---:|
| development | 32,768 | xor | spaced | 0.499791–0.500108 | 0.002589–0.004547 | 7.99605–7.99756 | 0.49842–0.50053 | 0.01445–0.08253 |
| development | 32,768 | xor | adjacent | 0.499830–0.500117 | 0.003829–0.006724 | 7.98731–7.98993 | 1.00000–1.00000 | 0.03776–0.06105 |
| development | 32,768 | add | spaced | 0.499689–0.500170 | 0.001386–0.004244 | 7.99903–7.99930 | 0.49942–0.50059 | 0.01427–0.05570 |
| development | 32,768 | add | adjacent | 0.499985–0.500278 | 0.003170–0.003762 | 7.99701–7.99788 | 0.65067–0.67416 | 0.01715–0.03593 |
| development | 32,768 | local | spaced | 0.499794–0.499929 | 0.000520–0.000780 | 7.99983–7.99984 | 0.49952–0.50034 | 0.01328–0.02552 |
| development | 32,768 | local | adjacent | 0.499765–0.500090 | 0.000541–0.001020 | 7.99982–7.99983 | 0.49957–0.50021 | 0.01523–0.03514 |
| development | 1,000,000 | xor | spaced | 0.499797–0.500370 | 0.000444–0.000921 | 7.99977–7.99984 | 0.49980–0.50066 | 0.01367–0.02579 |
| development | 1,000,000 | xor | adjacent | 0.499736–0.500102 | 0.004550–0.007143 | 7.98785–7.98984 | 1.00000–1.00000 | 0.03594–0.07463 |
| development | 1,000,000 | add | spaced | 0.499933–0.500335 | 0.000468–0.001083 | 7.99978–7.99984 | 0.49986–0.50030 | 0.01398–0.03339 |
| development | 1,000,000 | add | adjacent | 0.499983–0.500292 | 0.002907–0.004628 | 7.99718–7.99782 | 0.65059–0.67406 | 0.01762–0.03789 |
| development | 1,000,000 | local | spaced | 0.499639–0.500284 | 0.000626–0.000863 | 7.99981–7.99985 | 0.49972–0.50027 | 0.02023–0.03215 |
| development | 1,000,000 | local | adjacent | 0.499753–0.499967 | 0.000436–0.000907 | 7.99979–7.99984 | 0.49973–0.50013 | 0.02615–0.04064 |
| validation | 32,768 | xor | spaced | 0.499733–0.499924 | 0.001731–0.003455 | 7.99718–7.99726 | 0.49880–0.50140 | 0.02373–0.04682 |
| validation | 32,768 | xor | adjacent | 0.499800–0.500128 | 0.002743–0.009920 | 7.98700–7.98919 | 1.00000–1.00000 | 0.02349–0.11345 |
| validation | 32,768 | add | spaced | 0.499936–0.500213 | 0.001676–0.002805 | 7.99911–7.99939 | 0.49951–0.50045 | 0.01972–0.04125 |
| validation | 32,768 | add | adjacent | 0.499911–0.500042 | 0.002238–0.004292 | 7.99746–7.99774 | 0.66618–0.67368 | 0.03490–0.05148 |
| validation | 32,768 | local | spaced | 0.499991–0.500163 | 0.000537–0.000734 | 7.99981–7.99983 | 0.49973–0.50038 | 0.01722–0.04052 |
| validation | 32,768 | local | adjacent | 0.499873–0.500205 | 0.000603–0.000954 | 7.99982–7.99985 | 0.49964–0.50033 | 0.02458–0.04687 |
| validation | 1,000,000 | xor | spaced | 0.499674–0.500090 | 0.000463–0.000913 | 7.99981–7.99985 | 0.49980–0.50028 | 0.01987–0.03055 |
| validation | 1,000,000 | xor | adjacent | 0.499949–0.500106 | 0.003081–0.010153 | 7.98657–7.98968 | 1.00000–1.00000 | 0.02740–0.10368 |
| validation | 1,000,000 | add | spaced | 0.499636–0.500014 | 0.000554–0.001094 | 7.99980–7.99984 | 0.49986–0.50023 | 0.02145–0.03752 |
| validation | 1,000,000 | add | adjacent | 0.499904–0.500049 | 0.002942–0.003849 | 7.99726–7.99764 | 0.66606–0.67371 | 0.01928–0.05148 |
| validation | 1,000,000 | local | spaced | 0.499771–0.500272 | 0.000528–0.000848 | 7.99983–7.99984 | 0.49949–0.50018 | 0.02436–0.03445 |
| validation | 1,000,000 | local | adjacent | 0.499914–0.500236 | 0.000422–0.000769 | 7.99981–7.99984 | 0.49970–0.50013 | 0.01949–0.02677 |

| Control group | Ones fraction | Max lag statistic | Predictor accuracy | Survival deviation |
|---|---:|---:|---:|---:|
| development | 0.499807–0.500224 | 0.000581–0.000713 | 0.49966–0.50029 | 0.01601–0.03359 |
| validation | 0.499856–0.500394 | 0.000466–0.000810 | 0.49981–0.50021 | 0.02076–0.03916 |

## Findings

- **XOR:** The old predictor remains exact for adjacent limbs. Widely spaced
  observations look much closer to the control, particularly in the larger
  state. This replicates the geometry sensitivity in the first study.
- **Addition:** The old predictor is no longer exact, but its adjacent accuracy
  remains about 65–67%, well above chance in both seed groups. Adding carries
  therefore does not remove all of the tested predictability. Balanced bits
  and closer waiting-time curves do not repair that failure.
- **Local mixing:** The old predictor is near 50% and the measured low-order
  diagnostics resemble the control in both layouts. This is a failure of this
  particular attack, not proof that no other predictor exists. Case study 6
  finds a different weakness in the reduced-word analogues.

All conditional entropies, per-atom bias extrema/standard deviations, paired
correlations, run statistics, event counts and censored totals are in
[temporal-summary.csv](temporal-summary.csv). Raw JSON retains every lag count,
transition count, block/run histogram, first-event histogram and complete gap
histogram. [derived.json](derived.json) also includes all survival/hazard points.

First-event bin zero is censored, not discarded. Restricted mean waiting time
is E[min(T,1024)], not an uncensored mean. Completed gaps alone are biased by
the finite horizon. The first and last runs of each trace are excluded from
the completed internal-run histogram; bin 128 means >=128.

These are descriptive seed-level comparisons with dependent observations. No
uncorrected multiple-test p-values or confidence claims are made. The MT19937-64
control files are reused once per seed from the earlier experiment, not new
independent controls for every size/layout. The smaller ring's wider sampling
still shares substantial source history; observations cannot be treated as
independent solely because their initial limb indices differ.

There is no physical time calibration, isotope or quantum dataset. These
waiting-time experiments are mathematical event diagnostics, not radioactive
decay measurements or evidence of quantum equivalence.

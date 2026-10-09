# Conditional kappa parameter refinement

The research target is improving the exponent in
`T(n) = O(n (log n)^(1-kappa))`, ultimately kappa = 1 under a valid applicable
framework. Faster verification alone does not change this bound.

An exact parameter-only candidate on the **unchanged PR150 circuit row** is:

```
kappa = 23601910741 / 50000000000000
      = 0.000472038214820
```

This is 0.04846146% above PR150's headline `0.000471809569` and 0.00252969%
above its stronger reported atom-1/2000 variant `0.000472026274`. These are
relative changes in kappa, not runtime speedups. No global priority or current
world-record claim is made. The comparison is pinned to PR150 head
`40d4038760ebb6d6d3d702ce88fa3645de29f0c1`.

**Status:** locally checked conditional arithmetic candidate. It retains the
entire upstream construction and all its unproved/inherited interfaces. It has
not received external mathematical review and is not an unconditional
multiplication theorem. No upstream PR or message has been submitted.

## What was changed

No gate, register reuse, frame, role inventory or recursion child was changed.
The existing exact moment calculation is searched on a finer 1e-15 coarse grid.
Then the ordinary atom exponent is chosen just above the strict adapter limit,
and the assembly eta is set to 1e-14. The separate phase stopping parameter
remains 1e-6. These two stopping parameters must not be conflated.

Let `c` be the certified coarse bit saving and `o` the inherited ordinary leaf
saving. At atom exponent `b`, the stopped bit saving is

```
a(b) = (1-b)c + b o.
```

The retained strict toll condition requires `a(b) < b < 1-a(b)`. Since c > o,
the lower bound is exactly

```
b > c / (1+c-o).
```

The search uses the smallest 1e-15-grid atom strictly above that threshold,
`472484276089/10^15`. Equality is deliberately rejected. Reciprocal alternatives
including 1/2116 and the inadmissible 1/2117 are recorded in `candidate.json`.
The atom slack is extremely small; this may worsen eventual size thresholds.
No practical multiplication crossover or runtime claim follows.

For this **fixed certified coarse saving**, as the atom approaches its threshold
from above and eta approaches zero, the assembly has supremum

```
a_limit / (1+2 a_limit) ~= 0.00047203821482042017.
```

The candidate is below that supremum; the next 1e-15 kappa point fails the
actual positive-eta assembly. This ceiling is for these parameter choices and
the fixed coarse witness. It is not an upper bound on all circuits, all finer
coarse certificates, or integer multiplication. Progress toward kappa = 1 needs
more than further rounding of these parameters.

There is also a broader limitation of this particular assembly, even if its
circuit suppliers improve. It requires `0 < a < b < 1/32`, positive eta, and
`kappa < (1-eta)q / (1+(2+eta)q)` with `q = a(1-2eta) > 0`.
Therefore

```
kappa < a/(1+2a) < 1/34.
```

This is a ceiling of the currently used assembly assumptions, not a lower bound
on multiplication complexity. Reaching kappa = 1 would require changing the
proof machinery or algorithm family, not merely optimizing this certificate.

## Checks actually performed

- PR150's standard `verify.py` passed locally, in about 119 seconds. It reproduced
  the prior controls and ledger, checked all 4,878 new adjacent frame pairs over
  the rationals, and verified the complete scalar identity on 27,521 formal
  variables over F2 and Z. Its mutation controls and published pricing passed.
- The optional `--all` audit of every inherited frame move was **not** rerun.
  The standard check retains upstream's earlier inherited-frame evidence.
- `search.py` reproduces both published PR150 kappas before searching. It calls
  the original exact moment/contamination, complex certificate, bridge and
  assembly functions without changing their implementations.
- `check_candidate.py` independently recalculates the parameter equations and
  all 47 rational slacks plus seven margins, with no upstream imports. The
  next kappa, zero eta and atom-equality corruptions are rejected.
- The upstream working tree remains unchanged. Logs and source/input hashes
  are retained. Assertions were enabled throughout.

The baseline log is `upstream-verification.txt`. `candidate.json` contains the
full selected assembly, bridge, exact gaps, search cases and comparison data.
`independent-check.json` contains the independent arithmetic receipt.

## Reproduce

Fetch PR150 at the exact commit above into
`.local-llm/integer-mult-research`. Do not change the older pinned checkout used
by the separate circuit-replay study. Then run from the repository root:

```powershell
python .local-llm/integer-mult-research/research/bit-reuse-147/verify.py
python experiments/kappa-refinement/search.py
python experiments/kappa-refinement/check_candidate.py
python experiments/kappa-refinement/freeze.py
python experiments/kappa-refinement/check_results.py
```

Python 3.11+ and its standard library suffice. CI independently checks the saved
rational witness and provenance; it does not regenerate the upstream word.

All new harness source is MIT. The external repository and checker code retain
their Apache-2.0 licence and contributor notices. Construction credit belongs to
PR150/DaysSky, PR147/hpst3r, PR144 and the inherited lineage documented upstream.
Atom tightening was already explored by PR148, and assembly slack tightening by
PR138. This experiment combines a closer admissible atom with a finer coarse
grid on PR150's unchanged row; it does not claim those ideas or that circuit as
new. Relevant references:

- [PR137, the screenshot's witness](https://github.com/CrocSwap/integer-mult-bounds/pull/137)
- [PR138, earlier slack tightening](https://github.com/CrocSwap/integer-mult-bounds/pull/138)
- [PR150, the pinned comparison](https://github.com/CrocSwap/integer-mult-bounds/pull/150)
- [PR150 proof and retained atom condition](https://github.com/DaysSky/integer-mult-bounds/blob/40d4038760ebb6d6d3d702ce88fa3645de29f0c1/research/bit-reuse-147/PROOF.md)

# Case study 6 — Perturbations, collisions and cycles

## Full 64-bit word perturbations

For each rule, seed and ring size, two initial states differed only at bit zero
of limb zero and evolved synchronously under the **same mask**. Every step
through 4,096 is saved. A zero first-return field means no return was observed
within the finite horizon; it does not mean an infinite period.

Ranges below cover all eight fixed seeds. Complete seed/group data are in
[damage-summary.csv](damage-summary.csv).

| Rule | Limbs | Bits | Hamming distance at step 4096 | Changed limbs at step 4096 | Maximum distance during run | Returns within horizon / 8 |
|---|---:|---:|---:|---:|---:|---:|
| xor | 1 | 64 | 1–1 | 1–1 | 1–1 | 8 |
| xor | 3 | 192 | 1–1 | 1–1 | 1–1 | 8 |
| xor | 17 | 1,088 | 1–1 | 1–1 | 1–1 | 8 |
| xor | 256 | 16,384 | 1–1 | 1–1 | 1–1 | 0 |
| add | 1 | 64 | 1–43 | 1–1 | 1–49 | 1 |
| add | 3 | 192 | 5–26 | 1–1 | 8–65 | 1 |
| add | 17 | 1,088 | 1–8 | 1–1 | 14–65 | 0 |
| add | 256 | 16,384 | 1–3 | 1–1 | 11–65 | 0 |
| local | 1 | 64 | 25–37 | 1–1 | 45–47 | 0 |
| local | 3 | 192 | 84–105 | 3–3 | 117–124 | 0 |
| local | 17 | 1,088 | 510–576 | 17–17 | 598–614 | 0 |
| local | 256 | 16,384 | 8108–8238 | 256–256 | 8396–8434 | 0 |

**XOR** keeps one changed bit. **Addition** can produce transient carry-related
bursts of differing bits, but the final perturbation on the 256-word ring is
still confined to one word in these tests. It does not show the broad sustained
spread seen with **local mixing**, which changes all 256 words and about half
the ring's bits at the final time. The full trajectories, not just the last
point, are retained. One flipped location across eight seeded states is a
bounded experiment, not an exhaustive avalanche characterization.

The zero mask for seed 0 in a one-word ring makes both XOR and addition reduce
to pure rotation. This predetermined degenerate case is retained, not removed
to improve the candidate's appearance.

## Exhaustive reduced-word state graphs

We enumerated every state and transition for each (word width, limb count),
rule and seed: **120 graphs, 1,683,840 edges in total**. Word widths are 4 or 8
bits; rotation counts are reduced modulo that width and mask words truncated.
These are explicitly reduced analogues of the 64-bit rule.

Collision excess = number of input states minus number of distinct next states.
It is not the number of colliding pairs. A permutation has zero collision
excess, one preimage per state, and every state lies on a cycle. Local mixing
can have transient trees leading into cycles; maximum transient is the longest
distance to a cycle. Ranges cover eight masks/seeds per row.

| Rule | Word bits | Limbs | State count | Collision excess | Largest cycle | Cyclic states | Max transient |
|---|---:|---:|---:|---:|---:|---:|---:|
| xor | 4 | 1 | 16 | 0–0 | 4–8 | 16–16 | 0–0 |
| xor | 4 | 2 | 256 | 0–0 | 8–16 | 256–256 | 0–0 |
| xor | 4 | 3 | 4,096 | 0–0 | 12–24 | 4096–4096 | 0–0 |
| xor | 8 | 1 | 256 | 0–0 | 8–16 | 256–256 | 0–0 |
| xor | 8 | 2 | 65,536 | 0–0 | 16–32 | 65536–65536 | 0–0 |
| add | 4 | 1 | 16 | 0–0 | 4–15 | 16–16 | 0–0 |
| add | 4 | 2 | 256 | 0–0 | 33–152 | 256–256 | 0–0 |
| add | 4 | 3 | 4,096 | 0–0 | 614–3598 | 4096–4096 | 0–0 |
| add | 8 | 1 | 256 | 0–0 | 8–240 | 256–256 | 0–0 |
| add | 8 | 2 | 65,536 | 0–0 | 1663–57872 | 65536–65536 | 0–0 |
| local | 4 | 1 | 16 | 1–3 | 3–11 | 3–12 | 4–13 |
| local | 4 | 2 | 256 | 55–62 | 4–22 | 12–39 | 20–39 |
| local | 4 | 3 | 4,096 | 1243–1421 | 19–124 | 30–152 | 65–150 |
| local | 8 | 1 | 256 | 7–43 | 7–64 | 11–86 | 23–153 |
| local | 8 | 2 | 65,536 | 5227–6340 | 246–871 | 524–1185 | 696–1135 |

## Interpretation

- XOR and addition are permutations for any width, consistent with their
  algebraic definitions and zero collisions in every enumerated graph. Addition
  has substantially longer cycles than XOR in several small configurations,
  but that does not establish a long period at production width.
- Local mixing has collisions and short attracting cycles in the tested
  reduced-word configurations. In the 16-bit graphs, only a small fraction of
  states lie on cycles; other trajectories merge into those cycles. Improved
  marginal statistics therefore coexist with loss of state information in
  these analogues.
- The small-word collision counts do **not** prove non-injectivity of the
  64-bit-word implementation. Conversely, the absence of a return in the
  finite 64-bit tests does **not** prove reversibility or a long period.

No candidate is promoted to replace the baseline. A useful follow-up would be
an explicitly reversible local rule, which could test whether local spreading
can coexist with guaranteed information preservation. It would still need
independent structural and physical tests; reversibility alone is insufficient.

All graph metrics and raw paths: [graph-summary.csv](graph-summary.csv).
Small graphs with at most 256 states also store every edge. Independent Python
traversal verified every graph statistic for one seed at all five sizes and
all three rules, including the 65,536-state graphs.

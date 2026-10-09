# Case study 7 — Matched native full-state cost

Hardware: AMD Ryzen 7 9800X3D, 8 cores / 16 logical CPUs; Windows, Clang 22.1.8,
`-O3 -std=c++17`, generic x86-64 target, one computation thread. Windows links
Psapi only for process measurements. No explicit AVX2/AVX-512 kernel is added.

All rules materialize three equally sized buffers: source, destination and a
cached mask. Each run performs three synchronous full-state updates, then
computes a checksum outside the timed evolution. There are three fresh-process
replicates per rule/size; rule order rotates between repetitions. Initialization
is timed separately. No CSV/trace/statistics work occurs inside benchmark steps.

Median per-step time is the median of the three run means. CPU and peak memory
below come from that median run. CPU% is process CPU time during evolution,
divided by evolution wall time and 16 logical processors; a fully occupied
single logical CPU would be about 6.25%. CPU accounting is quantized: short
runs can round to zero or imply more than one busy core. CPU columns are therefore
suppressed for evolution durations below 0.1 seconds; raw readings remain in CSV.
Longer-run CPU figures are still approximate. Peak working set is the OS process high-water mark, not a memory-bandwidth
counter; it includes allocation/initialization and the checksum pass.

| Logical bytes | Rule | Median ms/step | Median-run init ms | Evolution CPU seconds (3 steps) | Machine CPU % | Peak working set MB |
|---:|---|---:|---:|---:|---:|---:|
| 1,000,000 | xor | 0.021667 | 0.429 | — | — | 7.41 |
| 1,000,000 | add | 0.121433 | 0.423 | — | — | 7.41 |
| 1,000,000 | local | 0.095733 | 0.434 | — | — | 7.41 |
| 16,000,000 | xor | 0.404700 | 5.835 | — | — | 52.42 |
| 16,000,000 | add | 1.941900 | 6.217 | — | — | 52.41 |
| 16,000,000 | local | 1.297133 | 5.732 | — | — | 52.42 |
| 1,000,000,000 | xor | 77.519300 | 352.847 | 0.234375 | 6.30 | 3004.41 |
| 1,000,000,000 | add | 124.126833 | 332.877 | 0.359375 | 6.03 | 3004.42 |
| 1,000,000,000 | local | 93.390500 | 379.301 | 0.265625 | 5.93 | 3004.42 |

## Limits of the comparison

This compares identical materialized workloads using the new common backend.
It is not a comparison against the fastest existing in-place multithreaded AVX2
baseline, and it is not comparable to billions of sparse observations/second.
The scalar carry dependency in whole-integer addition limits simple vectorization.
Local mixing reads neighboring words but can use independent destination words;
no thread-scaling optimization was attempted in this scientific comparison.

The 1 GB runs need about 3 GB of buffers, rather than the old sparse engine's
small worker-local state. All three requested sizes completed on the connected
machine. The largest tests are short, active-desktop measurements, not sustained
thermal or contention tests. No cache-miss or DRAM-bandwidth counters were
collected. Compiler/CPU/memory details and every replicate are retained in
[manifest.json](manifest.json) and [benchmark-summary.csv](benchmark-summary.csv).

Checksums match between all repeated runs for a rule/size; checksums are expected
to differ between different rules. Native transition correctness is separately
checked with an independent whole-integer/word oracle, including carries and
discarded overflow. Benchmark determinism alone is not that correctness proof.

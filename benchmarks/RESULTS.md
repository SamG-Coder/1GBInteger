# Native temporal performance, 2026-10-09

Windows, AMD Ryzen 7 9800X3D (8 cores / 16 logical CPUs), Clang 22.1.8 LLVM-MinGW, `-O3 -std=c++17 -pthread`. Generic x86-64 target, no AVX requirement added to the temporal engine. Baseline source: `3401542`; cached source: `883de4d`; prefix source: the commit containing this report.

Each cell is the median of three fresh-process runs, logical size 1,000,000,000 bytes, seed 12345, 1,000 steps, interval 10. This is 101 observations per atom, including step zero. Runs were sequential without explicit affinity or clock controls, on an active desktop. Small timings include significant startup/scheduling noise. These results measure sparse exact observations, not full-state evolution or physical atoms.

## Complete matrix

Runtime in milliseconds, **baseline / final exact-prefix version**.

| Atoms | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 7.695 / 0.245 | 4.121 / 0.744 | 3.357 / 0.778 | 2.242 / 0.820 | 3.566 / 1.636 |
| 10,000 | 87.940 / 2.493 | 45.662 / 1.879 | 24.040 / 1.718 | 15.433 / 2.035 | 12.718 / 2.122 |
| 100,000 | 726.220 / 28.495 | 383.506 / 17.529 | 216.279 / 11.004 | 139.245 / 6.869 | 111.352 / 6.488 |
| 1,000,000 | 7275.236 / 315.814 | 3844.385 / 170.050 | 2108.588 / 79.694 | 1354.238 / 59.890 | 962.128 / 53.734 |

## One million atoms, optimization increments

| Threads | Baseline seconds | Cached seconds | Prefix seconds | Final million observations/s | Baseline/final speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.275237 | 0.678906 | 0.315814 | 319.81 | 23.04x |
| 2 | 3.844385 | 0.379127 | 0.170050 | 593.94 | 22.61x |
| 4 | 2.108588 | 0.219690 | 0.079694 | 1267.36 | 26.46x |
| 8 | 1.354238 | 0.146378 | 0.059890 | 1686.41 | 22.61x |
| 16 | 0.962128 | 0.112998 | 0.053734 | 1879.64 | 17.91x |

## Longer run: one million atoms, 10,000 steps

1,001,000,000 observations per run. CPU and memory below belong to the median-runtime replicate. CPU utilization is total process CPU seconds divided by externally measured process wall time and 16 logical processors. 100% denotes the whole machine; a fully occupied single thread is approximately 6.25%. Process wall time includes process startup and monitoring delay; engine seconds include worker start/join and reduction, but exclude argument parsing and pre-timer setup. With CSV, engine seconds also include writing/flushing output.

Peak working set is sampled from the OS high-water mark while the process lives, every nominal 5 ms; it can miss the final peak and is not allocated heap size. Short-run CPU accounting is coarse (some rows round to zero). Prefer this longer run for resource interpretation. No hardware cache-miss or memory-bandwidth counters were collected.

| Version | Threads | Engine seconds | Million observations/s | Process CPU seconds | CPU % of machine | Sampled peak working set MB |
|---|---:|---:|---:|---:|---:|---:|
| baseline-long | 16 | 9.524207 | 105.10 | 108.031 | 70.66 | 28.93 |
| prefix-long | 1 | 2.632073 | 380.31 | 2.438 | 5.75 | 4.23 |
| prefix-long | 2 | 1.437879 | 696.16 | 2.594 | 11.04 | 4.62 |
| prefix-long | 4 | 0.926069 | 1080.91 | 3.547 | 23.20 | 4.66 |
| prefix-long | 8 | 0.527419 | 1897.92 | 3.906 | 43.54 | 4.72 |
| prefix-long | 16 | 0.423339 | 2364.54 | 4.141 | 56.90 | 4.88 |

## Changes and remaining costs

- Cache source words in 64-step blocks: eliminate repeated hashes and ring-index divisions from the inner observation loop. Keep accumulators in worker-local variables; remove the 24 MB million-atom array. The calling thread now participates, avoiding a separate worker for single-thread runs.
- Exact XOR prefixes: six shift/XOR stages calculate the XOR of every left shift of a word (and similarly right shifts). A truncated prefix follows by cancellation. Only observation times need individual evaluation; a whole mask block is accumulated in constant work. No approximation, reduced bit width, or changed recurrence.
- CSV values come from parallel computation and are emitted once. The original CSV path recomputed the entire mask history per row. At most one million 64-bit values are buffered (8 MB).
- The next optimization candidates are checksum hashing, prefix/observation arithmetic, and batching several independent atoms for SIMD. Worker startup dominates the smallest workloads; workers already persist across all steps, so a per-step pool would not help. Thread scaling is sublinear and the longer-run 16-thread CPU time exceeds single-thread CPU time. This establishes overhead/contention, not a particular hardware cause.
- Aggregate memory is now proportional to thread count, not atom count, and the per-atom loop uses register-sized state. This makes memory bandwidth a less plausible primary limit than arithmetic, but hardware counters are needed to quantify cache misses and bandwidth. No claim of a measured cache-hit rate is made.
- No handwritten assembly or architecture-specific kernel was added: the measured gains came from exact arithmetic restructuring. Any future AVX2/AVX-512 path needs runtime feature detection and matched-workload benchmarks.

## Correctness

All raw-file checksum/ones totals match across versions, repetitions, and thread counts for each workload. The 10,000-step signature is `6bcc9fb3a83b23dc`, with 500,521,033 ones. Checksums alone are not the correctness oracle: `test_temporal_parallel.py` compares every emitted limb against the original temporal engine over 27 seed/step/interval cases and five thread counts, including uneven partitions and threads greater than atoms. It also compares sampled limbs to full-buffer evolution, including a real 1 GB two-step run. Existing 18 temporal/full and 48 lazy/full cases pass.

`test_temporal_kernel.cpp` checks every selected observation against materialized small rings (1, 2, 3, 7, 17 limbs), multiple seeds, step boundaries, intervals 1 through UINT64_MAX, and repeated whole-ring wrap through 2,177 steps. It passes locally with undefined-behavior sanitization; CI runs the same sanitizer test. No quantum-physics inference follows from these correctness or throughput results.

## Reproduce on Windows

```powershell
clang++ -O3 -std=c++17 -pthread temporal_parallel.cpp -o temporal_parallel.exe
.\benchmark_temporal.ps1 -Output results.csv
.\benchmark_temporal.ps1 -Steps 10000 -Atoms 1000000 -Output long-results.csv
clang++ -O3 -mavx2 -std=c++17 -pthread main.cpp -o integer_universe.exe
clang++ -O3 -std=c++17 temporal.cpp -o integer_universe_temporal.exe
python test_temporal_parallel.py
clang++ -O2 -fsanitize=undefined -fno-sanitize-recover=all -std=c++17 test_temporal_kernel.cpp -o test_temporal_kernel.exe
.\test_temporal_kernel.exe
```

For baseline reproduction, build `temporal_parallel.cpp` from commit `3401542` into a separate executable using the same compiler/flags, then pass its path with `-Executable`. Raw CSV files in this directory preserve every replicate, engine time, external wall time, CPU time, sampled memory, and result signature. The intermediate cached version is retained in Git history and in `cached-results.csv`.

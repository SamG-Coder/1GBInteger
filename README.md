# 1GBInteger — Integer Universe Atomic Randomness Lab

Experimental deterministic **1,000,000,000-byte** integer (125 million 64-bit limbs) with native C++ evolution, AVX2 acceleration, persistent worker threads, and a browser dashboard for statistical tests. The seed initializes the state. Evolution is a bit rotation plus XOR with a fixed seed-dependent mask, **not** a physics simulation.

## Build

Requires a C++17 compiler with x86-64 AVX2 support and Python 3 (standard library only). AVX2 is compiled in this build; run on an AVX2-capable CPU.

Linux:
```bash
g++ -O3 -mavx2 -pthread -std=c++17 main.cpp -o integer_universe
```

Windows (MinGW-w64):
```powershell
g++ -O3 -mavx2 -pthread -std=c++17 main.cpp -o integer_universe.exe
```

## Performance experiments

```bash
./integer_universe --gb 1 --steps 100 --threads 8 --avx2
./integer_universe --gb 1 --steps 100 --threads 8 --avx2 --legacy-threads
./integer_universe --gb 1 --steps 100 --threads 8 --fast
./integer_universe --gb 1 --steps 100 --threads 8 --cache-mask
```

- `--avx2`: SIMD kernel with exact 64-bit arithmetic.
- `--fast`: threaded scalar evolution without expensive full-state statistics.
- `--legacy-threads`: create threads for every step rather than using the persistent worker pool.
- `--cache-mask`: precompute the XOR mask; requires an additional 1 GB of RAM.
- `--mb 16`: quicker 16 MB test instead of the default 1 GB.
- `--out sample.bin --sample-bytes 1048576`: sample evenly spaced limbs from the evolved integer.

Compare the `--avx2` and `--fast` sample file SHA-256 hashes for identical output after matching seeds/steps. Likewise compare worker pool vs legacy threads. Memory bandwidth limits performance for large buffers. For disassembly inspection:
```bash
objdump -d -C integer_universe > disassembly.txt
```
Search for AVX2 `vpsllq`, `vpsrlq`, `vpmuludq`, `vpxor`, and `vpor` instructions.

## Browser lab

```bash
python server.py
```
Open http://localhost:8765. The dashboard defaults to 1 GB. It displays bit balance, frequency z-score, run-count z-score, byte entropy, chi-square, bit-lag correlations, 256-bit block counts, reproducibility fingerprint, and native timing.

Command-line test:
```bash
python lab.py --gb 1 --seed 12345 --steps 1 --bits 65536 --out results.json
```

Tests compare against *ideal independent fair quantum spin measurement statistics*. This is not an actual quantum-physics reproduction or direct comparison against measured quantum-device data. Passing randomness tests does not establish quantum equivalence or prove determinism in physics. The original 64-bit seed limits the number of possible initialized states to at most 2^64 irrespective of buffer size.

## Notes

- 1 GB means decimal 1,000,000,000 bytes (not 1 GiB).
- Default source requires GCC/Clang-compatible `__builtin_popcountll`; MSVC needs a small portability fix.
- The main program can report aggregate per-step timing in the chosen runtime; performance results depend strongly on CPU and memory configuration.

## Fused AVX2 cached-mask experiment (October 2026)

The `--avx2-cache` option combines a precomputed 1 GB XOR mask with vectorized 64-bit rotate/carry and XOR operations. It uses approximately 2 GB for state + mask and should be compared with `--avx2`, `--fast` and `--cache-mask` rather than assumed faster. Reading the mask adds bandwidth traffic.

```bash
./integer_universe --gb 1 --steps 100 --threads 8 --avx2-cache
python benchmark.py --mb 16 --steps 5 --threads 4 --repeats 3
python benchmark.py --mb 1000 --steps 10 --threads 8 --repeats 3
```

The benchmark checks SHA-256 hashes of sampled evolved states across all four modes and reports median step times and speedup relative to scalar. It aborts on any output mismatch. The GitHub Actions workflow runs a 16 MB correctness test and uploads emitted assembly disassembly when Actions are enabled.

**AVX-512:** Not implemented yet. AVX-512DQ includes 64-bit integer multiplication, but the actual benefit depends on CPU instruction support, clocks, memory bandwidth and compiler output. An AVX-512 implementation must have runtime CPU-feature dispatch before being safe to distribute broadly.

**Performance reporting:** This commit has not yet been benchmarked on the user's machine. Do not treat benchmark results from other machines as equivalent.

## Lazy evolution: avoid scanning 1 GB for sparse samples

`lazy.cpp` calculates selected 64-bit limbs directly from the seed and the requested step without allocating the 1 GB integer. The current evolution is a fixed-mask rotation/XOR recurrence, which has the exact closed form `S(t) = ROTL(S(0),t) XOR XOR(ROTL(M,k), k=0..t-1)`. Arbitrary original limbs and mask limbs are reconstructed from the seed.

```bash
g++ -O3 -std=c++17 lazy.cpp -o integer_universe_lazy
./integer_universe_lazy --gb 1 --steps 100 --sample-bytes 8192 --out lazy-sample.bin
g++ -O3 -mavx2 -pthread -std=c++17 main.cpp -o integer_universe
python3 test_lazy.py
```

The lazy and full-state modes must produce identical sample files for matching seeds, steps, integer size and sample size. `test_lazy.py` checks 48 configurations. CI is configured to run those comparisons.

**Important limitations:** Lazy evolution is O(sampled limbs × steps), not O(1) in step count; it can become slower for large samples or very high steps. It is a mathematical shortcut specific to this fixed-mask rotation/XOR rule, not a faster way to update a general 1 GB arbitrary-precision integer. Sparse state evaluation is useful for current statistical tests but not for a model whose particles interact globally. It does not validate any quantum-mechanical hypothesis.

## Dashboard: lazy versus full state

Build both native engines before launching the browser lab:

```bash
g++ -O3 -mavx2 -pthread -std=c++17 main.cpp -o integer_universe
g++ -O3 -std=c++17 lazy.cpp -o integer_universe_lazy
python server.py
```

On Windows/MinGW-w64, use the same commands with `-o integer_universe.exe` and `-o integer_universe_lazy.exe`.

Open http://localhost:8765 and select **Lazy (sparse)** for tests that sample selected limbs without allocating the full state, or **Full 1 GB** to materialize and evolve every bit. The dashboard defaults to lazy mode. Both modes use the same seed, step count and sample-position mapping. The server runs the selected native executable, not a stand-in pseudorandom stream.

```bash
python lab.py --gb 1 --steps 100 --bits 65536 --mode lazy --out lazy.json
python lab.py --gb 1 --steps 100 --bits 65536 --mode full --out full.json
```

For equal parameters, the resulting `sha256` fields should match. Run `python test_lazy.py` for regression coverage. These are *randomness diagnostics*, not tests of actual atomic decay or entanglement.

## Temporal virtual-atom sampling

`temporal.cpp` observes the **same selected integer bits across time** without repeatedly reconstructing the full 1 GB state. For the fixed rotation/XOR evolution rule, it maintains the accumulated mask contribution for each selected limb, updating that contribution in O(1) work per simulated step and selected atom. It emits a CSV of step, atom, limb index, bit index, bit measurement, and exact 64-bit limb.

```bash
g++ -O3 -std=c++17 temporal.cpp -o integer_universe_temporal
./integer_universe_temporal --gb 1 --seed 12345 --steps 10000 --atoms 64 --interval 10 --out temporal.csv
python3 test_temporal.py
```

On Windows, use `-o integer_universe_temporal.exe`. The regression test compares 18 seed/step combinations, each with 8 atom observations, against the full native evolution. CI is configured to run the test and a 1 GB logical-state temporal benchmark.

**Interpretation:** These are deterministic bit observations, not a physical radioactive-decay process. This mode has O(atoms × steps) arithmetic even when observations are infrequent; CSV size is O(atoms × observed steps). The logical integer is not fully materialized. Benchmark timings for sparse observations must not be compared directly with full-buffer evolution timings.

## Parallel million-atom temporal scaling (experimental)

`temporal_parallel.cpp` distributes virtual atoms among persistent worker threads. Each thread maintains local accumulators and statistics, avoiding per-step synchronization and avoiding a full 1 GB allocation. Default aggregate-only mode avoids producing billions of CSV rows.

```bash
g++ -O3 -pthread -std=c++17 temporal_parallel.cpp -o temporal_parallel
./temporal_parallel --gb 1 --atoms 1000000 --steps 100 --interval 10 --threads 8
./temporal_parallel --gb 1 --atoms 1000000 --steps 1000 --interval 10 --threads 8
```

The command reports observations, ones, a deterministic XOR checksum, elapsed time and observations per second. Compare identical parameters with `--threads 1` and `--threads 8`: observations, ones and checksum must match. This is a scaling experiment, not a physical decay simulation. Runtime scales as O(atoms × steps), and the CSV option is limited to at most one million rows and is substantially slower.

GitHub Actions includes a 1-million-atom / 100-step smoke benchmark. **No measured speedup is claimed until the workflow completes successfully.**

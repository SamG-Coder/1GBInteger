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

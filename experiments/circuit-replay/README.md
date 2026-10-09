# Native exact circuit replay

[Measured comparison](CASE_STUDY.md) against two real circuits from
CrocSwap/integer-mult-bounds. This is a verifier benchmark, not a new bound on
integer multiplication or a benchmark of multiplying integers.

The implementation uses this project's packed `uint64_t` bit vectors. It checks
symbolic outputs, physical frame inclusion, scatter identity, rank counts and
the entire basis map, including scratch restoration, in both orientations.
`active` tracks nonzero limb intervals; `dense` is the full-row control. Both
execute every gate and verify every output coefficient.

## Reproduce on Windows

Requires Python 3.11+, LLVM-MinGW Clang, Git and the pinned nlohmann JSON header
already obtained by `experiments/stock-llama-integration/setup.py`. No model or
inference is involved in this benchmark; the header is the only llama checkout
file used. Commands run from the repository root:

```powershell
python experiments/circuit-replay/setup.py
python experiments/circuit-replay/run.py --runs 7
python experiments/circuit-replay/validate.py
python experiments/circuit-replay/report.py
python experiments/circuit-replay/freeze.py
python experiments/circuit-replay/check_results.py
```

The external repository is pinned to
`d1d6c070f5a8c684727ee7ec35d930f9ebfa9758` and left unchanged. Inputs are its
pair-assembly `frame-word-23.json.gz` and `frame-word-25.json.gz`. Both paths are
decompressed once to identical JSON bytes before measurements. The baseline
loads upstream's actual `replay` function and uses AST insertion of two clock
reads for the full-basis phase. Its computations and assertions are unchanged;
additional runs of the unmodified function validate result equivalence.

Both code paths are single-threaded. Fresh processes, rotating order, seven
repetitions, all samples retained. The summary distinguishes verifier time,
full-basis time, process wall time and peak process working set. Timings are not
hardware bandwidth measurements. Native validation additionally runs four
complete UBSan checks and tests rejection of seven corrupted fixtures.

All new source is MIT. The external Apache-2.0 repository, checker and fixtures
retain their licences and attribution. They are not copied into this source
release. The nlohmann header is MIT. The manifest binds all measured inputs,
external source, shared local headers, native executable and evidence hashes.
CI checks saved evidence without claiming to rerun these benchmarks.

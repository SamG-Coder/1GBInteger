# Decode output weight traffic

This study follows the stock integration by reducing bytes read for the Q8
vocabulary projection. It uses stock GGML quantizers and dot products. It adds
no XOR/popcount arithmetic optimization. See [measured results](CASE_STUDY.md).

Modes:

- `stock`: unmodified stock dispatch, including stock repacking.
- `output-q8`: byte-for-byte copy of the original output weights, using the same
  row dispatcher as the compressed modes. This is an exact control for dispatch
  and allocation changes.
- `output-q5`: requantize the output weights to stock Q5_0 once at load time.
- `output-q4`: requantize the output weights to stock Q4_0 once at load time.

Q5/Q4 are approximate. Original Q8 weights remain available for the tied input
embedding and prompt output projection. The hook only handles single-token
decode `result_output`, and uses llama.cpp's existing worker pool. Activation
quantization and output dot products are stock GGML operations. Added memory is
reported, as are conversion time and peak working set. No per-token weight
expansion or allocation is performed. Like the preceding experimental hook,
this uses process-global state for one model/context; it is not a concurrent
server integration. Stock remains the default.

## Reproduce on Windows

Use the LLVM-MinGW/Clang toolchain and prerequisites from the preceding
[integration](../stock-llama-integration/README.md). This small build script
reuses its pinned static libraries and response files. All commands below run
from the repository root. The model, binaries and large dumps are ignored.

```powershell
python experiments/output-weight-traffic/setup.py
.venv/Scripts/python.exe experiments/output-weight-traffic/distribution.py
python experiments/output-weight-traffic/run.py --part smoke
python experiments/output-weight-traffic/run.py --part all
.venv/Scripts/python.exe experiments/output-weight-traffic/analyze.py
python experiments/output-weight-traffic/report.py
python experiments/output-weight-traffic/freeze.py
python experiments/output-weight-traffic/check_results.py
```

NumPy and PyYAML are needed for GGUF inspection; NumPy is needed for analysis.
`setup.py` bootstraps the preceding build if absent and verifies the llama.cpp
revision. The executable is `.local-llm/traffic-build/integer-llm.exe` and accepts
the same CLI as the preceding runner, plus the modes above.

Timing uses a fixed stock-generated continuation for all four modes, five
process runs per workload/mode, and rotating execution order. The compressed
modes still compute every vocabulary logit and select an argmax, but feed back
the same teacher token to prevent output divergence changing the workload.
Instrumented accuracy runs and unrestricted greedy responses are separate.
The five quality prompts are diagnostics, not a general quality benchmark.

`distribution.py` measures the actual signed Q8 code histogram. The lossless
estimates include a 32-bit exception mask, original FP16 scale and packed high
bits, but omit indexing and alignment overhead. Failure of these schemes does
not prove all lossless compression is ineffective. Marginal code entropy is
not a lower bound on every possible structured encoding.

The committed manifest binds source, reused runner, binary hash and evidence.
`check_results.py` verifies saved evidence without the model; CI runs that check,
not a fresh inference benchmark. Native inference was tested on the documented
Windows machine. Repository code is MIT; the downloaded Qwen weights retain
their Apache-2.0 licence.

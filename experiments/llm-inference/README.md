# Real Qwen CPU inference experiments

MIT-licensed native integration and experimental kernels for 1GBInteger.
The baseline remains llama.cpp; experimental paths are opt-in. No model weights
or dependency binaries are distributed here.

Model: [bartowski/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF),
Q4_0, **352,972,352 bytes** (353 MB decimal / 337 MiB), pretrained Qwen2.5-0.5B-Instruct,
494,032,768 parameters. Despite the filename this is mixed quantization:
embedding Q8_0, FFN down Q4_1, most projections Q4_0. The Qwen model is Apache 2.0;
our repository's MIT licence does not relicense the weights. llama.cpp is MIT.
Pinned dependency and model revisions and SHA256 are in `setup.py`.

## Reproduce on Windows x64

Prerequisites: Python 3.11+, Git, CMake, LLVM-MinGW with clang/clang++ and
mingw32-make on PATH. Native build needs AVX2/FMA; AVX-512 VNNI is used when
the compiler detects it. Binaries are host-specific and are not shipped.

```powershell
python experiments/llm-inference/setup.py
python experiments/llm-inference/run_suite.py --part smoke
python experiments/llm-inference/run_suite.py --part timing
python experiments/llm-inference/run_suite.py --part accuracy
python experiments/llm-inference/run_suite.py --part tuning
python -m pip install numpy matplotlib
python experiments/llm-inference/analyze.py --dumps
python experiments/llm-inference/inspect_assembly.py
clang++ -O3 -std=c++17 -march=native experiments/llm-inference/memory_probe.cpp -o .local-llm/memory-probe.exe
.local-llm/memory-probe.exe > experiments/llm-inference/raw/memory-bandwidth.csv
python experiments/llm-inference/plot_results.py
python experiments/llm-inference/write_report.py
python experiments/llm-inference/freeze.py
python experiments/llm-inference/check_results.py
```

All downloaded weights, llama.cpp source, binaries, tensor dumps, and temporary
logs stay under ignored `.local-llm/`. Raw benchmark summaries go in this folder.
Run experiments sequentially on an idle machine. `setup.py` checks model SHA256
and dependency revision. An existing checkout at another revision is rejected.

## Methods

* `stock`: upstream kernels and runtime weight repacking, optional hook null.
* `control`: upstream kernels with runtime repacking disabled, hook null. This
  isolates operation changes from upstream repacking and GEMM choices.
* `packed`: exact integer Q4_0 x Q8_0 dot retaining original packed weights,
  shared activation zero-point correction and inline F16C scale conversion.
  `packed-call` retains the slower function-call scale conversion for comparison.
* `prepack`: exact integer Q4_0 x Q8_0 arithmetic, pre-expanded unsigned weights,
  shared activation zero-point correction, AVX2 or VNNI dots and FP32 block
  scale/FMA reduction. Used for single-token decode; upstream prompt GEMM stays.
  The final layout uses 36 bytes per block; `prepack-wide` retains the initial
  68-byte layout to measure compaction independently.
* `bitplane`: exact integer bit-plane dot in the first FFN up projection only;
  floating reduction order differs, and full-model numerical conformance fails
  despite the correct integer dots (see the case study). Four signed weight planes by eight signed
  activation planes require 32 AND/popcount intersections per 32-value block.
* `binary` / `ternary`: approximate first FFN up projection only. Weights are
  replaced by sign times block mean magnitude, or thresholded at half the block
  mean magnitude with mean retained magnitude as scale. Q8 activations and all
  other transformer operations remain conventional. No retraining/calibration.

Binary weighted-mask sums use the same counting principle as XOR/popcount
bipolar dots, generalized to eight activation bit planes. This is integer
arithmetic within dot products, not an integer-only transformer. FP scaling,
RMSNorm, RoPE, softmax, attention and KV cache remain upstream floating point.

The hook is a two-line dispatch addition plus a null function pointer in pinned
ggml CPU code. It is called before upstream dispatch; only eligible operations
are intercepted. Stock dispatch executes unchanged. `patch_llama.py` rejects
an unpinned revision and is idempotent.

Timed runs exclude load, one-time packing and warm-up (separately reported),
include native token selection and text assembly, and use fixed generation
length even after EOS. PP measures decode of the complete prompt; generation
measures N-1 subsequent decode/selection steps. TTFT includes prompt and first
greedy selection, excludes model load. CPU usage is process CPU time divided
by elapsed wall time and all 16 logical processors. Peak working set is the
process lifetime high-water mark, including load/packing/warm-up. No GPU layers.

Correctness runs dump logits at every step, replay stock generated tokens in
other engines, and dump last-position FFN/block outputs with scheduler callbacks.
Instrumentation changes graph scheduling and timing: those runs are **not**
performance evidence. Separate free-running responses test accumulated drift.
Five prompts cover arithmetic, fact, exact instruction, code and long-context
recall; this is a small diagnostic suite, not a general language benchmark.

Initial smoke results are exploratory. Repeated final measurements, numerical
comparisons and limitations are recorded in [CASE_STUDY.md](CASE_STUDY.md), with
the full table in [MEASUREMENTS.md](MEASUREMENTS.md).

## Run your own prompt

Save a UTF-8 file containing the Qwen chat template (see `chat()` in
`run_suite.py`), then run:

```powershell
.local-llm/experiment-build/integer-llm.exe --model .local-llm/Qwen2.5-0.5B-Instruct-Q4_0.gguf --prompt prompt.txt --output result.json --mode stock --threads 8 --context 2048 --generate 64 --repeat 3
```

The default mode is `stock`; mode selection is explicit for all experiments.
Results contain responses, token IDs, per-run timings, process CPU time and
peak working set. `--teacher tokens.json`, `--logits output.f32` and
`--trace prefix` enable numerical validation. F32 dumps use native little-endian
float32, step-major vocabulary logits; layer dumps contain the final position.
Multi-microbatch prefill overwrites each layer dump until the final prompt chunk.
Large prompts are split into consecutive batches; the context must fit the
complete prompt plus requested generation length. Fixed generation does not
stop at EOS; this runner is a benchmark, not an interactive chat application.

## Additional evidence

`raw/llama-bench-baseline.json` is the original native llama-bench baseline,
collected before the hook was added with `-ngl 0 -t 8 -p 32,128,512 -n 32,128
-r 3 -o json`. llama-bench uses its own synthetic token workloads, so these
numbers are recorded separately from real-prompt timings. To rebuild that tool,
configure pinned llama.cpp in a separate build directory using LLVM-MinGW,
Release, `GGML_NATIVE=ON`, `GGML_CUDA=OFF`, `BUILD_SHARED_LIBS=OFF`, and
`CMAKE_C_FLAGS`/`CMAKE_CXX_FLAGS` set to `-D_WIN32_WINNT=0x0A00`; build target
`llama-bench`. The hook is null in that tool and leaves dispatch unchanged.

The real-layer microbenchmark automatically runs after `--part accuracy` and
uses a captured stock activation. Direct invocation:

```powershell
.local-llm/experiment-build/bench-llm-kernels.exe .local-llm/Qwen2.5-0.5B-Instruct-Q4_0.gguf .local-llm/evaluation/quality-math-stock-activation.f32 experiments/llm-inference/raw/kernel-benchmark.json
```

The microbenchmark uses the 68-byte reference pack for bit-plane/binary/ternary
helpers and the compact 36-byte layout for expanded SIMD. The integrated hybrid
paths use compact 20-byte bit-plane or 12-byte binary/ternary blocks, so their
microkernel timings are not standalone timings of every integrated layout.
Weight packing and activation packing are measured outside its repeated dot
loop; model-level timings include activation packing and exclude one-time
weight conversion. Packed model bytes remain resident for fallback operations.

Linux CI builds/tests the portable native kernel targets and validates committed
evidence. The process-metric runner is Windows-specific. Model performance and
quality are measured locally, not inferred from CI, and no weights are downloaded
by CI. The source is MIT; upstream llama.cpp's MIT and Qwen's Apache 2.0 terms
remain separate. No CUDA implementation or GPU performance is claimed here.

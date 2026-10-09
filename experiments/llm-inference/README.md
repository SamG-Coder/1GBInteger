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
```

All downloaded weights, llama.cpp source, binaries, tensor dumps, and temporary
logs stay under ignored `.local-llm/`. Raw benchmark summaries go in this folder.
Run experiments sequentially on an idle machine. `setup.py` checks model SHA256
and dependency revision. An existing checkout at another revision is rejected.

## Methods

* `stock`: upstream kernels and runtime weight repacking, optional hook null.
* `control`: upstream kernels with runtime repacking disabled, hook null. This
  isolates operation changes from upstream repacking and GEMM choices.
* `prepack`: exact integer Q4_0 x Q8_0 arithmetic, pre-expanded unsigned weights,
  shared activation zero-point correction, AVX2 or VNNI dots and FP32 block
  scale/FMA reduction. Used for single-token decode; upstream prompt GEMM stays.
* `bitplane`: exact integer bit-plane dot in the first FFN up projection only;
  floating reduction order differs. Four signed weight planes by eight signed
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
comparisons and limitations will be recorded in the case study.

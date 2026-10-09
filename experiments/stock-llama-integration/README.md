# Stock llama.cpp plus targeted integer decode

This experiment keeps `use_extra_bufts=true` in **every mode**. The pinned
x86 backend stores this model's Q8 vocabulary output weights in ordinary CPU
storage while repacking its Q4 matrices. The custom path uses those existing
Q8 blocks directly; all repacked tensors remain exclusively owned by stock
llama.cpp. No expanded or duplicate model weights are allocated.

Modes:

* `stock` (default): null hook, ordinary optimized llama.cpp.
* `stock-plus-single`: replace only the vocabulary projection during decode,
  using shared activation preparation and one row per dot loop.
* `stock-plus`: same replacement with four rows evaluated together.

The runner explicitly marks prompt and generation phases. Checking the
matrix's column count alone is insufficient because the final vocabulary
projection can have one column even during prefill. A phase gate ensures
**zero custom calls during prefill**. Afterward, only a contiguous, ordinary
CPU Q8_0 `result_output` with supported dimensions is eligible. Other operations,
types, buffer layouts, batch sizes, and unsupported CPUs use stock dispatch.

## Arithmetic

For a signed Q8 weight byte `w`, `w XOR 0x80` is the unsigned value `w + 128`.
For each four-element accumulator lane:

```
sum(w * a) = VNNI_dot(w XOR 0x80, a) - 128 * sum(a)
```

The activation correction is calculated once per block by the existing ggml
workers, followed by their normal thread-pool barrier. Four output rows share
each activation load, scale and correction. Integer products and correction
are exact even for -128. Per-block FP32 FMA and eight-lane horizontal reduction
retain the upstream Q8 dot's order. This is an exact signed-to-unsigned dot
transformation, not a binary approximation of the model.

The VNNI routine uses explicit target attributes and a runtime ISA gate. These
are **native host builds**, not universal executable distributions. The optional
hook uses process-global state and supports the one-model, one-context runner;
concurrent model serving is outside this experiment's scope.

## Reproduce

Windows prerequisites match the preceding [LLM experiment](../llm-inference/README.md):
Python, CMake, Git and LLVM-MinGW on PATH. The model and dependency revisions
remain pinned there. Its raw evidence and sources are preserved unchanged.

```powershell
python experiments/stock-llama-integration/setup.py
python experiments/stock-llama-integration/run.py --part smoke
python experiments/stock-llama-integration/run.py --part all
```

The executable is `.local-llm/stock-build/integer-llm.exe`. It accepts the same
prompt-file and result-file interface as the previous runner, with modes above:

```powershell
.local-llm/stock-build/integer-llm.exe --model .local-llm/Qwen2.5-0.5B-Instruct-Q4_0.gguf --prompt prompt.txt --output result.json --mode stock-plus --threads 8 --context 2048 --generate 64 --repeat 3
```

Keep large downloads, compiled programs, logs and tensor dumps in ignored
`.local-llm/`. The new C++/Python code is MIT-licensed. llama.cpp and model
licences remain as described in the preceding experiment.

## Evaluation protocol

Four prompts (38, 326, 1,358 and 2,798 tokens) with fixed generation lengths,
five independent process runs per mode, and rotating execution order. Each
process loads and warms up before its measured run. Separate thread sweeps use
1, 4 and 16 threads. The main comparison uses eight threads. First-token latency
is warm latency; loading and initialization are separately recorded. Token
generation includes greedy selection and runs to its fixed length after EOS.

Correctness uses five prompts and 48 output positions, with stock token history
replayed for full-logit comparisons. Prefill and first-decode FFN/residual layer
outputs are byte-hashed. Free-running responses are compared separately.
Instrumented trace timings are excluded from performance results.

Unit tests initialize llama.cpp's FP16 lookup tables before calling its reference
dot routine. They check 32,000 random Q8 integer-oracle and scaled upstream-dot
cases over multiple block counts, including the -128 integer domain.

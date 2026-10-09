# Case study 12 — Combining integer decode with stock llama.cpp

**Decision: keep stock as the default.** The combination succeeds functionally and preserves stock outputs, repacking and prompt execution, but the repeated runs do not establish an end-to-end speedup. The initial smoke improvement did not survive the broader comparison. The four-row mode is approximately tied on the short workload and slower on the longer-context workloads.

The combined mode preserves stock repacking and prompt processing and replaces only the Q8 vocabulary projection during token generation. It uses the original compact weights, shared activation preparation and an exact XOR sign-bit transformation with VNNI dots. Four output rows share activation loads. All other operations stay on stock dispatch.

## Matched results

Qwen2.5-0.5B-Instruct Q4_0, 353 MB, Ryzen 7 9800X3D, native Windows CPU, eight threads for the context sweep. Five independent warmed processes per mode and workload; order rotates. These are medians, with paired speedup computed within each round. Loading is excluded from TTFT and throughput. Peak RAM includes loading and warm-up.

| Workload / prompt tokens | Mode | PP tok/s | Decode tok/s | Paired decode speedup | TTFT ms | Peak MiB |
|---|---|---:|---:|---:|---:|---:|
| short / 38 | stock | 1550.1 | 149.2 | 1.000x | 24.6 | 574.7 |
| short / 38 | stock-plus-single | 1641.5 | 152.3 | 1.021x | 23.3 | 574.7 |
| short / 38 | stock-plus | 1224.9 | 149.9 | 1.004x | 31.2 | 574.7 |
| medium / 326 | stock | 1624.3 | 141.7 | 1.000x | 200.8 | 601.6 |
| medium / 326 | stock-plus-single | 1541.0 | 126.6 | 0.955x | 211.7 | 601.5 |
| medium / 326 | stock-plus | 1625.5 | 137.9 | 0.938x | 200.7 | 601.6 |
| long / 1358 | stock | 1466.6 | 119.7 | 1.000x | 926.1 | 629.1 |
| long / 1358 | stock-plus-single | 1473.6 | 118.2 | 0.989x | 921.7 | 629.1 |
| long / 1358 | stock-plus | 1469.7 | 116.0 | 0.951x | 924.1 | 629.1 |
| extended / 2798 | stock | 1173.6 | 92.3 | 1.000x | 2384.2 | 655.1 |
| extended / 2798 | stock-plus-single | 1198.5 | 90.9 | 0.990x | 2334.7 | 655.1 |
| extended / 2798 | stock-plus | 1184.0 | 89.9 | 0.982x | 2363.3 | 655.1 |

Unlike the preceding experiment, every row enables upstream repacking. Captured logs confirm both the same 330.95 MiB CPU_Mapped allocation and 184.94 MiB CPU_REPACK allocation in stock and combined modes. There are no extra packed weights. Approximately 2 KiB of shared activation scratch is allocated once for the 896-wide projection.

The `stock-plus-single` comparison uses the same activation preparation and exact arithmetic with one output row per loop. `stock-plus` evaluates four rows together. Any improvement is specific to the measured workload; nearby process runs still experience desktop interference. Min/max observations and paired speedup ranges are in summary.json.

## Thread sweep

Three alternating process runs per configuration, same 47-token arithmetic prompt, 64 generated tokens.

| Threads | Mode | Decode tok/s | Paired speedup |
|---:|---|---:|---:|
| 1 | stock | 84.1 | 1.000x |
| 1 | stock-plus | 88.0 | 1.057x |
| 4 | stock | 160.7 | 1.000x |
| 4 | stock-plus | 163.1 | 1.007x |
| 16 | stock | 93.1 | 1.000x |
| 16 | stock-plus | 99.7 | 1.029x |

## Numerical and behavioral validation

Both combined variants produce **byte-identical full logits and traced layer outputs** to fully optimized stock on all five diagnostic prompts and 48 positions per prompt. Equal logit arrays imply equal token probabilities for the same softmax computation. Free-running token sequences also match stock. This is a stronger comparison than the earlier no-repack control.

The 151,936-entry vocabulary gives 36,464,640 compared logits per variant across the five prompts. Prefill and first-decode FFN/residual tensors are byte-hashed separately. Teacher-forced comparisons and free-running responses are both retained. Model quality remains 4/5 on this small diagnostic set, including the stock model's incorrect arithmetic answer; no new reasoning capability is claimed.

The integration asserts zero custom calls during prefill and exactly generation_length - 1 calls during each fixed-length generation run. The explicit phase gate is necessary: output projection input can have one column even during prompt processing.

Unit tests check 32,000 Q8 integer-oracle and upstream floating-point cases after initializing llama.cpp lookup tables. Runtime guards reject unsupported ISA, quantization, custom/repacked output buffers, noncontiguous tensors and unsupported batch dimensions, leaving stock dispatch responsible for them. Native builds target the build host; this is not a universal binary distribution.

## What changed in the kernel

For each four-value dot lane, signed weight bytes are transformed with XOR 0x80 into unsigned values w+128. VNNI computes their dot with signed activations; a shared 128*sum(activation) correction restores the exact signed result. This supports -128 and avoids saturating 16-bit intermediate products. Stock block scales, FP32 FMA order and final reduction order are retained. No model weight approximation, retraining or model-format conversion occurs.

The existing ggml worker pool prepares disjoint activation blocks once and synchronizes before processing output rows. There are no per-token allocations or separate custom worker threads. Native disassembly confirms XOR, VNNI dot and FMA instructions; the bounded excerpt is saved in assembly.txt.

Instrumented output-projection timings on the arithmetic prompt (diagnostic attribution only, excluded from the performance table):

| Mode | Median output projection ms |
|---|---:|
| stock | 2.2486 |
| stock-plus-single | 2.3478 |
| stock-plus | 2.0379 |

This projection contains 144,643,072 bytes of Q8 blocks. Dividing that logical weight volume by the instrumented stock projection time gives roughly 64 GB/s, near the preceding case study's approximately 68 GB/s eight-thread large-buffer read probe. This is consistent with a memory-bandwidth limit, but is not a measurement of hardware memory-controller traffic or a causal proof. Reducing arithmetic alone need not accelerate this operation once memory traffic dominates.

## Scope and default

Stock remains the default; `--mode stock-plus` enables the combined experiment. The public entry point is a native benchmark executable using llama.cpp, not yet a concurrent-serving backend. The hook and scratch are process-global and support one model/context at a time. Only CPU inference was tested. The Q8 target happens to use ordinary compact CPU storage on this pinned x86 backend; repacked layouts are explicitly rejected rather than decoded incorrectly.

See README.md for build/run commands. The preceding case study sources and results are preserved. Raw summaries, trace hashes, backend allocation evidence, source hashes and executable hash are committed; model weights, tensor dumps and binaries remain ignored.

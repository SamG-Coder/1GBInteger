# Case study 13: Reduce output weight traffic

Measured 2026-10-09 on an AMD Ryzen 7 9800X3D (8 cores / 16 threads), Windows 11, Clang 22.1.8 LLVM-MinGW, CPU only, eight inference threads. The pinned Qwen2.5-0.5B-Instruct Q4_0 GGUF is 352,972,352 bytes; its tied vocabulary projection is Q8_0 with 151,936 rows and 896 columns. llama.cpp is pinned to `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b`. See the manifest for model, executable, source and evidence hashes.

The experiment uses stock GGML quantizers and dot products. It changes the stored precision of the decode output projection, keeps stock prompt processing and repacking, and does not tune XOR/popcount arithmetic. Q8 is the exact control; Q5 and Q4 change model outputs and remain opt-in.

Q4 reduces projection reads by 47.1% and shows 1.139-1.155x median paired decode speedups across the three tested workloads. Q5 reduces reads by 35.3% and shows 1.071-1.150x median paired speedups. These are measured approximate-model trade-offs, not exact-output acceleration. Q4 has more logit drift; neither result justifies changing the default without broader quality evaluation.

## Bytes per decode token

These are logical weight bytes read for one full vocabulary projection, not hardware DRAM transaction measurements or total model traffic. Scales and quantization metadata are included. Original weights remain resident; each experimental mode adds a second projection copy.

| Format | Projection bytes/token | Reduction vs Q8 | Added weight storage |
|---|---:|---:|---:|
| output-q8 | 144,643,072 | 0.0% | 137.94 MiB |
| output-q5 | 93,592,576 | 35.3% | 89.26 MiB |
| output-q4 | 76,575,744 | 47.1% | 73.03 MiB |

The actual Q8 code entropy is 7.6649 bits/code. Tested low-bit-plus-exception-mask layouts estimate 1.010-1.060 times the original size even before row offsets and padding. These simple lossless schemes are rejected; this does not rule out other structured lossless encodings.

## Matched end-to-end performance

Five independent process runs per workload/mode, rotating order. All modes consume the same stock-generated teacher continuation, compute all logits and select an argmax. Warmed timings exclude model loading and one-time conversion. Generation throughput covers subsequent decode steps, excluding the first prompt-produced token. Instrumented numerical runs are excluded. Paired speedup is the median of within-repetition ratios; it need not equal the ratio of medians. All repetitions, including outliers, are retained.

| Workload | Mode | Prompt tokens | Decode tok/s median [min, max] | Paired speedup median [min, max] |
|---|---|---:|---:|---:|
| short | stock | 38 | 147.6 [139.6, 153.9] | 1.000x [1.000, 1.000] |
| short | output-q8 | 38 | 135.7 [124.6, 139.6] | 0.910x [0.844, 0.964] |
| short | output-q5 | 38 | 163.0 [162.0, 174.9] | 1.150x [1.056, 1.185] |
| short | output-q4 | 38 | 173.3 [160.1, 185.2] | 1.147x [1.130, 1.255] |
| medium | stock | 326 | 138.8 [134.0, 140.7] | 1.000x [1.000, 1.000] |
| medium | output-q8 | 326 | 139.5 [120.2, 147.0] | 1.000x [0.854, 1.097] |
| medium | output-q5 | 326 | 147.7 [130.0, 168.1] | 1.071x [0.936, 1.195] |
| medium | output-q4 | 326 | 157.5 [149.6, 185.1] | 1.139x [1.111, 1.316] |
| long | stock | 1358 | 112.6 [69.3, 117.1] | 1.000x [1.000, 1.000] |
| long | output-q8 | 1358 | 115.8 [110.3, 122.3] | 1.001x [0.979, 1.727] |
| long | output-q5 | 1358 | 117.2 [103.0, 127.4] | 1.097x [0.956, 1.487] |
| long | output-q4 | 1358 | 125.6 [117.5, 131.8] | 1.155x [1.044, 1.806] |

The copied Q8 control separates some dispatch/allocation effects from precision changes. Lower byte counts alone do not establish a speedup: stock Q4/Q5 unpacking, dot-product cost, cache behavior, other layers and system variation also contribute. No hardware counter measurement was made.

### Memory and startup (medium workload medians)

| Mode | Added conversion/copy ms | Peak working set MiB | Warm TTFT ms | Prompt tok/s |
|---|---:|---:|---:|---:|
| stock | 0.0 | 601.5 | 197.6 | 1651.2 |
| output-q8 | 46.8 | 739.6 | 208.5 | 1564.2 |
| output-q5 | 256.6 | 690.8 | 199.4 | 1636.1 |
| output-q4 | 173.5 | 674.5 | 208.5 | 1564.4 |

Prompt execution is unchanged, but its timing can vary with system state and the extra allocation. This prototype reduces decode reads; it does not shrink the GGUF or total resident model storage.

## Accuracy and free-running diagnostics

Five prompts, 48 generated positions each. Position zero is the stock prompt output and is checked separately. The following comparisons cover only the 235 decode positions per mode (35,704,960 full-vocabulary logits). All prompt logits and observed transformer layers remain byte-identical under the shared token history. KL is stock-to-candidate, in nats, averaged per decode position.

| Mode | Decode top-1 agreement | Mean KL | Maximum logit error | Largest probability difference | Diagnostic passes | Free sequences matching stock |
|---|---:|---:|---:|---:|---:|---:|
| stock | 100.00% | 0.000000 | 0.000000 | 0.000000 | 4/5 | 5/5 |
| output-q8 | 100.00% | 0.000000 | 0.000000 | 0.000000 | 4/5 | 5/5 |
| output-q5 | 97.45% | 0.006215 | 1.377495 | 0.112120 | 4/5 | 2/5 |
| output-q4 | 91.91% | 0.024412 | 2.720676 | 0.221999 | 4/5 | 0/5 |

The fixed-length suite continues past end-of-answer tokens. Full sequence equality above includes that continuation. To avoid confusing it with user-visible answer drift, the table below stops at the first stock end-of-answer token (GGUF EOS ID 151645), or the generation limit. The initial prompt-produced token remains excluded. Free response equality compares text before the first end marker.

| Mode | Answer decode positions | Top-1 matches | Mean KL on answer positions | Visible responses matching stock |
|---|---:|---:|---:|---:|
| stock | 81 | 81/81 (100.00%) | 0.000000 | 5/5 |
| output-q8 | 81 | 81/81 (100.00%) | 0.000000 | 5/5 |
| output-q5 | 81 | 79/81 (97.53%) | 0.004284 | 4/5 |
| output-q4 | 81 | 74/81 (91.36%) | 0.018030 | 4/5 |

The Q8 copy must match every compared logit exactly. Passing a small output diagnostic does not make Q5/Q4 exact or establish general quality preservation. The stock model already fails the arithmetic diagnostic; responses and scoring results are retained in `summary.json`. No held-out corpus perplexity or broad language benchmark was measured.

## Validation and limits

102 native runs: 4 smoke runs, 3 stock continuation seeds, 60 matched timing runs and 35 instrumented/free-running quality runs. Dispatch counts require zero prompt hook calls and exactly N-1 decode calls. Logs confirm stock CPU_REPACK remains active in every run. The saved evidence check validates these counts, shared timing histories, Q8 exactness, prompt/layer invariance, finite timings and source hashes. It does not rerun inference in CI.

Four additional smoke runs (all modes) pass with the runner and hook compiled under Clang undefined-behavior sanitization. Existing stock static libraries are not instrumented; see `sanitizer.json`.

The hook remains a single-model, single-context experimental integration with process-global state. Native execution is validated on this Windows machine only. Stock remains the default. Before choosing a compressed default, evaluate held-out perplexity and broader tasks, repeat performance on a controlled idle host, and compare reduced-memory loading that avoids retaining both projection copies.

Reproduction commands and format details: [README](README.md). All new code is MIT. Model weights are downloaded separately under their own licence.

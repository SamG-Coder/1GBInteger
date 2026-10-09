# Case study 14: Q2 output projection

Q2_K reduces logical output-projection reads to 51,050,496 bytes per decode token: 64.7% below Q8 and 33.3% below Q4. Median paired end-to-end decode speedups across the three tested workloads range from 1.173x to 1.354x versus stock. This is approximate inference; the accuracy and actual response changes are reported below. Stock remains the default.

Measured 2026-10-09 on AMD Ryzen 7 9800X3D, 8 cores / 16 threads, Windows 11, Clang 22.1.8 LLVM-MinGW. CPU-only inference with eight threads. Pinned llama.cpp: `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b`. Qwen2.5-0.5B-Instruct Q4_0 GGUF: 352,972,352 bytes, with a tied Q8_0 vocabulary matrix of shape 151,936 by 896. Model SHA-256 is recorded in the manifest.

## Format and implementation

The stock Q2_K block stores 256 weights in 84 bytes, including its scales and minima. The 896-column projection needs four blocks per row, padded to 1024 columns. Weight and activation padding are zero. Thus the effective storage is 3 bits per original weight for this shape. A literal 2-bit calculation would understate traffic. Q2 also uses stock Q8_K activation quantization rather than Q8_0, so its numerical changes include both quantization choices.

The hook uses stock quantizers and dot products, with the existing llama.cpp worker pool. Only single-token decode `result_output` is replaced. Prompt output, embeddings and other layers retain their original weights and stock repacking. No new XOR/popcount kernel was added. Original weights remain resident, so the compressed projection is additional storage. These byte counts are logical weight reads, not measured hardware DRAM transactions or total model traffic.

| Mode | Projection bytes/token | Reduction from Q8 | Additional weights MiB |
|---|---:|---:|---:|
| output-q8 | 144,643,072 | 0.0% | 137.94 |
| output-q5 | 93,592,576 | 35.3% | 89.26 |
| output-q4 | 76,575,744 | 47.1% | 73.03 |
| output-q2 | 51,050,496 | 64.7% | 48.69 |

## Matched performance

All five modes were remeasured in this executable; the Q4/Q5 numbers are not copied from the preceding study. Five separate process runs per workload/mode, rotating order. All modes feed back the same stock-generated teacher continuation to keep the work matched while still calculating every logit and selecting an argmax. Warmed decode throughput excludes model loading, conversion, and the first prompt-produced token. Instrumented runs are excluded. Paired speedup is the median of individual repetition ratios, not the ratio of median rates. All outliers are retained.

| Workload | Mode | Prompt tokens | Decode tok/s median [min, max] | Paired speedup median [min, max] |
|---|---|---:|---:|---:|
| short | stock | 38 | 158.8 [150.4, 178.1] | 1.000x [1.000, 1.000] |
| short | output-q8 | 38 | 154.8 [140.4, 166.9] | 1.000x [0.789, 1.110] |
| short | output-q5 | 38 | 184.6 [172.8, 200.3] | 1.124x [1.096, 1.228] |
| short | output-q4 | 38 | 184.1 [154.3, 197.9] | 1.146x [0.972, 1.316] |
| short | output-q2 | 38 | 212.3 [195.4, 224.4] | 1.354x [1.097, 1.492] |
| medium | stock | 326 | 141.5 [133.4, 147.8] | 1.000x [1.000, 1.000] |
| medium | output-q8 | 326 | 137.3 [123.8, 144.5] | 0.983x [0.881, 1.021] |
| medium | output-q5 | 326 | 146.7 [124.9, 159.6] | 1.002x [0.936, 1.135] |
| medium | output-q4 | 326 | 168.1 [157.4, 176.7] | 1.205x [1.065, 1.260] |
| medium | output-q2 | 326 | 185.5 [176.5, 229.3] | 1.343x [1.194, 1.620] |
| long | stock | 1358 | 116.4 [113.4, 124.1] | 1.000x [1.000, 1.000] |
| long | output-q8 | 1358 | 111.8 [103.7, 118.5] | 0.985x [0.836, 1.045] |
| long | output-q5 | 1358 | 127.8 [120.2, 135.5] | 1.080x [0.969, 1.193] |
| long | output-q4 | 1358 | 131.7 [123.8, 143.0] | 1.117x [1.062, 1.260] |
| long | output-q2 | 1358 | 141.1 [136.5, 146.0] | 1.173x [1.153, 1.286] |

Q2 versus Q4, paired within each repetition:

| Workload | Median Q2/Q4 speedup | Min | Max |
|---|---:|---:|---:|
| short | 1.134x | 1.109x | 1.346x |
| medium | 1.121x | 1.069x | 1.345x |
| long | 1.050x | 1.015x | 1.140x |

### Medium-workload startup and memory medians

| Mode | Conversion/copy ms | Peak working set MiB | Warm TTFT ms | Prompt tok/s |
|---|---:|---:|---:|---:|
| stock | 0.0 | 601.5 | 216.7 | 1505.2 |
| output-q8 | 47.1 | 739.5 | 226.6 | 1439.4 |
| output-q5 | 252.6 | 690.8 | 213.4 | 1528.8 |
| output-q4 | 161.4 | 674.5 | 222.5 | 1465.8 |
| output-q2 | 9031.9 | 650.3 | 199.9 | 1651.3 |

## Numerical differences

Five prompts with 48 fixed generation positions each. Prompt-produced logits are checked separately and stay byte-identical. The full decode comparison covers 235 positions per mode, including continuation after end-of-answer: 35,704,960 vocabulary logits. Observed transformer layers remain byte-identical under the shared token history. The copied Q8 control must match every compared logit exactly. KL is stock-to-candidate in nats.

| Mode | Decode top-1 agreement | Mean KL | Max logit error | Max probability change |
|---|---:|---:|---:|---:|
| stock | 100.00% | 0.000000 | 0.000000 | 0.000000 |
| output-q8 | 100.00% | 0.000000 | 0.000000 | 0.000000 |
| output-q5 | 97.45% | 0.006215 | 1.377495 | 0.112120 |
| output-q4 | 91.91% | 0.024412 | 2.720676 | 0.221999 |
| output-q2 | 80.00% | 0.272405 | 10.492155 | 0.643419 |

The following comparison stops at the first stock end-of-answer token (GGUF EOS 151645), or the generation limit; it excludes the prompt-produced first token. Free responses are compared before their first end marker.

| Mode | Answer top-1 matches | Answer mean KL | Diagnostic passes | Visible responses matching stock |
|---|---:|---:|---:|---:|
| stock | 81/81 (100.00%) | 0.000000 | 4/5 | 5/5 |
| output-q8 | 81/81 (100.00%) | 0.000000 | 4/5 | 5/5 |
| output-q5 | 79/81 (97.53%) | 0.004284 | 4/5 | 4/5 |
| output-q4 | 74/81 (91.36%) | 0.018030 | 4/5 | 4/5 |
| output-q2 | 68/81 (83.95%) | 0.179215 | 4/5 | 3/5 |

## Actual Q2 responses

### math: pass

````text
The calculation is as follows:

17 times 23 is:

\( 17 \times 23 = 391 \)

The answer is \( 391 \).
````

### fact: pass

````text
The capital of France is Paris.
````

### instruction: pass

````text
red, green, blue
````

### code: fail

````text
```python
def add(a, b):
    """
    This function takes two inputs, a and b, and returns their sum.

    Args:
        a (int): The first number to be added.
        b (int):
````

### conversation: pass

````text
ORBIT-731
````

The arithmetic test asks 17 times 23; the correct answer is 391. The stock model already fails this diagnostic. The case scores are small scripted diagnostics, not a quality benchmark. Fixed generation length can truncate longer responses. Full responses for all modes are saved in `summary.json`.

### Code completion with a larger generation limit

Q2 starts a longer docstring and its 48-token code response is truncated before a return statement. An additional 128-token run of the same prompt checks whether it can complete the function. This is reported separately and does not replace the original fixed-protocol failure.

stock: pass

````text
```python
def add(a, b):
    return a + b
```
````

output-q2: pass

````text
```python
def add(a, b):
    """
    This function takes two inputs, a and b, and returns their sum.

    Args:
        a (int): The first number to be added.
        b (int): The second number to be added.

    Returns:
        int: The sum of the two inputs.
    """
    return a + b
```
````

## Validation and limits

130 native runs: five smoke runs, three stock continuation seeds, 75 matched timing runs, 45 accuracy/free-running runs and two extended code-budget follow-ups. Hook counts require zero prefill calls and N-1 decode calls. Stock CPU_REPACK is confirmed in every log. Fresh copied-Q8 controls match stock logits and token sequences.

An additional 3,000 randomized native cases compare the stock Q2/Q8_K dot with an independently accumulated dequantized scalar reference, covering 768, 896 and 1024 source widths, all-zero inputs and exact zero padding. Five more inference smoke runs pass undefined-behavior sanitization. Sanitization covers the runner, hook and test harness; prebuilt stock libraries are not instrumented. Details are in `sanitizer.json`.

The manifest binds source, reused runner, executable and recorded evidence. CI checks saved evidence; it does not rerun model inference. Native execution is validated on this Windows machine only. The hook uses process-global state for a single model/context and is not a concurrent serving implementation.

Q2 results must be weighed against numerical drift and broader quality. No held-out corpus perplexity was measured. Timings can vary with other system activity; fewer bytes alone do not prove the entire speedup comes from memory bandwidth. Removing duplicate resident weights remains separate work.

[Reproduction instructions](README.md). Repository code is MIT; model weights retain their own licence.

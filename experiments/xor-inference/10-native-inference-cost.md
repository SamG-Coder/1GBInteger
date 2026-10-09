# Case study 10 — Native inference cost and binary dot products

The machine is the Ryzen 7 9800X3D on Windows, Clang 22.1.8, generic
`-O3 -std=c++17`, one thread. No explicit hardware POPCNT or AVX requirement is
introduced. The compiler may optimize both reference and packed loops.

## End-to-end learned readout

The timed batch includes feature construction and XOR readout for 8192 unseen
inputs. It excludes label generation, accuracy scoring and ambiguity
certification. For each fit, the native program times three repetitions and
records the median, keeping a result checksum. Below are medians across eight
clean 1024-example fits of the same parity task, which every model learns exactly.

| Model | Feature count | Median batch milliseconds | Million predictions/s |
|---|---:|---:|---:|
| linear | 17 | 0.6128 | 13.368 |
| reservoir | 65 | 0.6136 | 13.352 |
| quadratic | 137 | 2.0194 | 4.057 |
| cubic | 697 | 5.2414 | 1.563 |

The simpler sufficient feature bank costs less than unnecessary nonlinear
features. The current implementation builds dense feature rows; it does not
yet exploit sparse learned coefficients to skip unused monomials. These are
small synthetic Boolean tasks, not tokens/second or language-model inference.

## Exact packed bipolar dot products

For {-1,+1} vectors, matching bits contribute +1 and differing bits contribute
-1, so `dot = N - 2*popcount(a XOR b)`. This sums products; it is not the same
operation as a GF(2) parity readout. The experiment uses 4096 pairs of 1024-bit
vectors. Every packed result matches the signed reference sum exactly in all
eight seeds and all three repetitions. Operand generation is outside timing.

| Operation | Median milliseconds |
|---|---:|
| Signed dot-product batch | 0.4352 |
| Already-packed XOR/popcount batch | 0.0263 |
| Packing the two signed input arrays | 22.7452 |

The ratio of median kernel times is **16.51x** in favor of the
packed representation. However, packing plus one packed execution takes about
**22.772 ms**, versus **0.435 ms** for one signed
execution. The simple bit-by-bit packer is expensive: this experiment does
**not** show a one-use end-to-end speedup. Persistently packed weights/activations
or sufficient reuse are needed to amortize it, or the packer must be improved.
Using these medians, the arithmetic break-even estimate is about
**55.6 reuses** of the same packed operands; this is an
estimate, not a measured reused-network benchmark.

Packed operands occupy one bit rather than one signed byte per element here:
an 8x operand-storage reduction relative to this int8 reference. This is not a
32x claim against a float32 model, and no entire trained model is measured.

The published [XNOR-Net](https://arxiv.org/abs/1603.05279) demonstrates binary
operations in trained image models. It is relevant prior art, not an accuracy
or speed result reproduced by this microbenchmark. A practical binary network
would additionally need learned weights, activations/thresholds, scaling,
training and end-to-end task evaluation.

All raw timings/checksums: [binary-dot.csv](raw/binary-dot.csv). The packing
measurement is taken once per seed and repeated in its three timing rows;
those copies are not independent packing measurements. Active-desktop timing
and small batch lengths limit precision; no confidence interval is claimed.

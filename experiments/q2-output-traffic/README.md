# Q2 decode output projection

This follow-up measures stock GGML Q2_K for the vocabulary projection, with
fresh stock, copied-Q8, Q5 and Q4 controls. See [results](CASE_STUDY.md).
New code is MIT; the downloaded Qwen weights retain their Apache-2.0 licence.

Q2_K uses 256-element, 84-byte blocks. Each 896-column output row is padded to
1024 columns, giving four blocks and 336 bytes per row. For 151,936 vocabulary
rows this is **51,050,496 bytes/token**, including scales, minima and padding.
That is 64.7% fewer logical projection bytes than Q8, and 33.3% fewer than Q4.
The effective storage is 3 bits per original weight for this padded shape,
not a literal two bits per weight. Counts are not measured DRAM traffic.

`output-q2` uses the pinned stock Q2_K reference quantizer and native stock
Q2_K/Q8_K dot product. Activations use Q8_K rather than the Q8_0 used by the
other modes. Weights and activations are zero-padded; activation scratch is
preallocated and the partial block uses a fixed 256-float stack buffer.
Original Q8 weights remain resident for embeddings and prompt output. Q2 adds
48.69 MiB of weight storage; it does not shrink the original GGUF or total
resident weights. The hook is a single-context experiment, not a concurrent
server backend. Stock remains the default; Q2 is approximate and opt-in.

## Reproduce

Windows / LLVM-MinGW prerequisites are the same as the preceding
[traffic experiment](../output-weight-traffic/README.md). Run from the root:

```powershell
python experiments/q2-output-traffic/setup.py
.venv/Scripts/python.exe experiments/q2-output-traffic/distribution.py
python experiments/q2-output-traffic/run.py --part smoke
python experiments/q2-output-traffic/run.py --part all
.venv/Scripts/python.exe experiments/q2-output-traffic/analyze.py
python experiments/q2-output-traffic/validate_native.py
python experiments/q2-output-traffic/report.py
python experiments/q2-output-traffic/freeze.py
python experiments/q2-output-traffic/check_results.py
```

The executable is `.local-llm/q2-traffic-build/integer-llm.exe`; modes are
`stock`, `output-q8`, `output-q5`, `output-q4`, and `output-q2`.
NumPy/PyYAML are needed for distribution inspection, NumPy for analysis.
The pinned preceding libraries and runner are reused; old studies remain intact.

Five process runs per timing configuration use rotating order and the same
stock-generated teacher continuation. Instrumented full-logit comparisons and
unrestricted greedy responses are separate. Generation has fixed length and
continues after end-of-answer; the report separately measures positions through
the first stock end marker and visible response text. Five answer diagnostics
are not a broad quality benchmark or corpus perplexity measurement.

`validate_native.py` runs 3,000 stock Q2 dot/dequantized scalar oracle cases,
checks zero padding, and runs five UBSan inference smoke modes. Sanitization
covers the runner, hook and test harness; linked stock libraries are not
instrumented. CI checks committed evidence without downloading the model.

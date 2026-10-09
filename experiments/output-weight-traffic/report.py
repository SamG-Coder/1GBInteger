"""Generate the case study from the committed numeric summary."""
import json,pathlib,statistics
HERE=pathlib.Path(__file__).resolve().parent
d=json.loads((HERE/'summary.json').read_text());dist=json.loads((HERE/'distribution.json').read_text())
lines=['# Case study 13: Reduce output weight traffic', '',
'Measured 2026-10-09 on an AMD Ryzen 7 9800X3D (8 cores / 16 threads), Windows 11, Clang 22.1.8 LLVM-MinGW, CPU only, eight inference threads. The pinned Qwen2.5-0.5B-Instruct Q4_0 GGUF is 352,972,352 bytes; its tied vocabulary projection is Q8_0 with 151,936 rows and 896 columns. llama.cpp is pinned to `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b`. See the manifest for model, executable, source and evidence hashes.', '',
'The experiment uses stock GGML quantizers and dot products. It changes the stored precision of the decode output projection, keeps stock prompt processing and repacking, and does not tune XOR/popcount arithmetic. Q8 is the exact control; Q5 and Q4 change model outputs and remain opt-in.', '',
'Q4 reduces projection reads by 47.1% and shows 1.139-1.155x median paired decode speedups across the three tested workloads. Q5 reduces reads by 35.3% and shows 1.071-1.150x median paired speedups. These are measured approximate-model trade-offs, not exact-output acceleration. Q4 has more logit drift; neither result justifies changing the default without broader quality evaluation.', '',
'## Bytes per decode token', '',
'These are logical weight bytes read for one full vocabulary projection, not hardware DRAM transaction measurements or total model traffic. Scales and quantization metadata are included. Original weights remain resident; each experimental mode adds a second projection copy.', '',
'| Format | Projection bytes/token | Reduction vs Q8 | Added weight storage |',
'|---|---:|---:|---:|']
for mode,size in dist['projection_bytes'].items():lines.append(f'| {mode} | {size:,} | {100*(1-size/dist["q8_bytes"]):.1f}% | {size/2**20:.2f} MiB |')
lines+=['','The actual Q8 code entropy is %.4f bits/code. Tested low-bit-plus-exception-mask layouts estimate 1.010-1.060 times the original size even before row offsets and padding. These simple lossless schemes are rejected; this does not rule out other structured lossless encodings.'%dist['code_entropy_bits'],'',
'## Matched end-to-end performance','',
'Five independent process runs per workload/mode, rotating order. All modes consume the same stock-generated teacher continuation, compute all logits and select an argmax. Warmed timings exclude model loading and one-time conversion. Generation throughput covers subsequent decode steps, excluding the first prompt-produced token. Instrumented numerical runs are excluded. Paired speedup is the median of within-repetition ratios; it need not equal the ratio of medians. All repetitions, including outliers, are retained.','',
'| Workload | Mode | Prompt tokens | Decode tok/s median [min, max] | Paired speedup median [min, max] |',
'|---|---|---:|---:|---:|']
for r in d['timing']:
    lines.append(f'| {r["case"]} | {r["mode"]} | {r["prompt_tokens"]} | {r["generation_tps"]:.1f} [{r["generation_tps_min"]:.1f}, {r["generation_tps_max"]:.1f}] | {r["paired_decode_speedup_median"]:.3f}x [{r["paired_decode_speedup_min"]:.3f}, {r["paired_decode_speedup_max"]:.3f}] |')
lines+=['','The copied Q8 control separates some dispatch/allocation effects from precision changes. Lower byte counts alone do not establish a speedup: stock Q4/Q5 unpacking, dot-product cost, cache behavior, other layers and system variation also contribute. No hardware counter measurement was made.','',
'### Memory and startup (medium workload medians)','','| Mode | Added conversion/copy ms | Peak working set MiB | Warm TTFT ms | Prompt tok/s |','|---|---:|---:|---:|---:|']
for r in d['timing']:
    if r['case']=='medium':lines.append(f'| {r["mode"]} | {r["prepack_ms"]:.1f} | {r["peak_working_set_bytes"]/2**20:.1f} | {r["first_token_ms"]:.1f} | {r["pp_tps"]:.1f} |')
lines+=['','Prompt execution is unchanged, but its timing can vary with system state and the extra allocation. This prototype reduces decode reads; it does not shrink the GGUF or total resident model storage.','',
'## Accuracy and free-running diagnostics','',
'Five prompts, 48 generated positions each. Position zero is the stock prompt output and is checked separately. The following comparisons cover only the 235 decode positions per mode (35,704,960 full-vocabulary logits). All prompt logits and observed transformer layers remain byte-identical under the shared token history. KL is stock-to-candidate, in nats, averaged per decode position.','',
'| Mode | Decode top-1 agreement | Mean KL | Maximum logit error | Largest probability difference | Diagnostic passes | Free sequences matching stock |','|---|---:|---:|---:|---:|---:|---:|']
for mode in ['stock','output-q8','output-q5','output-q4']:
    rows=[r for r in d['accuracy'] if r['mode']==mode];qs=[r for r in d['quality'] if r['mode']==mode]
    lines.append(f'| {mode} | {100*statistics.mean(r["decode_top1_agreement"] for r in rows):.2f}% | {statistics.mean(r["mean_kl_nats"] for r in rows):.6f} | {max(r["max_abs_error"] for r in rows):.6f} | {max(r["max_probability_difference"] for r in rows):.6f} | {sum(r["diagnostic_pass"] for r in qs)}/5 | {sum(r["tokens_match_stock"] for r in qs)}/5 |')
lines+=['','The fixed-length suite continues past end-of-answer tokens. Full sequence equality above includes that continuation. To avoid confusing it with user-visible answer drift, the table below stops at the first stock end-of-answer token (GGUF EOS ID 151645), or the generation limit. The initial prompt-produced token remains excluded. Free response equality compares text before the first end marker.','','| Mode | Answer decode positions | Top-1 matches | Mean KL on answer positions | Visible responses matching stock |','|---|---:|---:|---:|---:|']
for mode in ['stock','output-q8','output-q5','output-q4']:
    rs=[r for r in d['accuracy'] if r['mode']==mode];qs=[r for r in d['quality'] if r['mode']==mode]
    n=sum(r['answer_decode_steps'] for r in rs);matches=sum(r['answer_top1_matches'] for r in rs);kl=sum(r['answer_mean_kl_nats']*r['answer_decode_steps'] for r in rs)/n
    lines.append(f'| {mode} | {n} | {matches}/{n} ({100*matches/n:.2f}%) | {kl:.6f} | {sum(r["response_matches_stock"] for r in qs)}/5 |')
lines+=['','The Q8 copy must match every compared logit exactly. Passing a small output diagnostic does not make Q5/Q4 exact or establish general quality preservation. The stock model already fails the arithmetic diagnostic; responses and scoring results are retained in `summary.json`. No held-out corpus perplexity or broad language benchmark was measured.','',
'## Validation and limits','',
'102 native runs: 4 smoke runs, 3 stock continuation seeds, 60 matched timing runs and 35 instrumented/free-running quality runs. Dispatch counts require zero prompt hook calls and exactly N-1 decode calls. Logs confirm stock CPU_REPACK remains active in every run. The saved evidence check validates these counts, shared timing histories, Q8 exactness, prompt/layer invariance, finite timings and source hashes. It does not rerun inference in CI.','',
'Four additional smoke runs (all modes) pass with the runner and hook compiled under Clang undefined-behavior sanitization. Existing stock static libraries are not instrumented; see `sanitizer.json`.','',
'The hook remains a single-model, single-context experimental integration with process-global state. Native execution is validated on this Windows machine only. Stock remains the default. Before choosing a compressed default, evaluate held-out perplexity and broader tasks, repeat performance on a controlled idle host, and compare reduced-memory loading that avoids retaining both projection copies.','',
'Reproduction commands and format details: [README](README.md). All new code is MIT. Model weights are downloaded separately under their own licence.','']
(HERE/'CASE_STUDY.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')

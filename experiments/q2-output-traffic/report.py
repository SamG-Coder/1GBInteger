"""Generate the Q2 case study from measured data."""
import json,pathlib,statistics
HERE=pathlib.Path(__file__).resolve().parent
d=json.loads((HERE/'summary.json').read_text());dist=json.loads((HERE/'distribution.json').read_text())
modes=['stock','output-q8','output-q5','output-q4','output-q2']
q2=[r for r in d['timing'] if r['mode']=='output-q2']
speed=[r['paired_decode_speedup_median'] for r in q2]
lines=['# Case study 14: Q2 output projection','',
f'Q2_K reduces logical output-projection reads to 51,050,496 bytes per decode token: 64.7% below Q8 and 33.3% below Q4. Median paired end-to-end decode speedups across the three tested workloads range from {min(speed):.3f}x to {max(speed):.3f}x versus stock. This is approximate inference; the accuracy and actual response changes are reported below. Stock remains the default.','',
'Measured 2026-10-09 on AMD Ryzen 7 9800X3D, 8 cores / 16 threads, Windows 11, Clang 22.1.8 LLVM-MinGW. CPU-only inference with eight threads. Pinned llama.cpp: `de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b`. Qwen2.5-0.5B-Instruct Q4_0 GGUF: 352,972,352 bytes, with a tied Q8_0 vocabulary matrix of shape 151,936 by 896. Model SHA-256 is recorded in the manifest.','',
'## Format and implementation','',
'The stock Q2_K block stores 256 weights in 84 bytes, including its scales and minima. The 896-column projection needs four blocks per row, padded to 1024 columns. Weight and activation padding are zero. Thus the effective storage is 3 bits per original weight for this shape. A literal 2-bit calculation would understate traffic. Q2 also uses stock Q8_K activation quantization rather than Q8_0, so its numerical changes include both quantization choices.','',
'The hook uses stock quantizers and dot products, with the existing llama.cpp worker pool. Only single-token decode `result_output` is replaced. Prompt output, embeddings and other layers retain their original weights and stock repacking. No new XOR/popcount kernel was added. Original weights remain resident, so the compressed projection is additional storage. These byte counts are logical weight reads, not measured hardware DRAM transactions or total model traffic.','',
'| Mode | Projection bytes/token | Reduction from Q8 | Additional weights MiB |','|---|---:|---:|---:|']
for mode,size in dist['projection_bytes'].items():lines.append(f'| {mode} | {size:,} | {100*(1-size/dist["q8_bytes"]):.1f}% | {size/2**20:.2f} |')
lines+=['','## Matched performance','',
'All five modes were remeasured in this executable; the Q4/Q5 numbers are not copied from the preceding study. Five separate process runs per workload/mode, rotating order. All modes feed back the same stock-generated teacher continuation to keep the work matched while still calculating every logit and selecting an argmax. Warmed decode throughput excludes model loading, conversion, and the first prompt-produced token. Instrumented runs are excluded. Paired speedup is the median of individual repetition ratios, not the ratio of median rates. All outliers are retained.','',
'| Workload | Mode | Prompt tokens | Decode tok/s median [min, max] | Paired speedup median [min, max] |','|---|---|---:|---:|---:|']
for r in d['timing']:lines.append(f'| {r["case"]} | {r["mode"]} | {r["prompt_tokens"]} | {r["generation_tps"]:.1f} [{r["generation_tps_min"]:.1f}, {r["generation_tps_max"]:.1f}] | {r["paired_decode_speedup_median"]:.3f}x [{r["paired_decode_speedup_min"]:.3f}, {r["paired_decode_speedup_max"]:.3f}] |')
lines+=['','Q2 versus Q4, paired within each repetition:','',
'| Workload | Median Q2/Q4 speedup | Min | Max |','|---|---:|---:|---:|']
for case in ['short','medium','long']:
    ratios=[]
    for rep in range(5):
        rates=[json.loads((HERE/'raw'/f'{case}-{rep}-{mode}.json').read_text())['samples'][0]['generation_tps'] for mode in ['output-q2','output-q4']]
        ratios.append(rates[0]/rates[1])
    lines.append(f'| {case} | {statistics.median(ratios):.3f}x | {min(ratios):.3f}x | {max(ratios):.3f}x |')
lines+=['','### Medium-workload startup and memory medians','','| Mode | Conversion/copy ms | Peak working set MiB | Warm TTFT ms | Prompt tok/s |','|---|---:|---:|---:|---:|']
for r in d['timing']:
    if r['case']=='medium':lines.append(f'| {r["mode"]} | {r["prepack_ms"]:.1f} | {r["peak_working_set_bytes"]/2**20:.1f} | {r["first_token_ms"]:.1f} | {r["pp_tps"]:.1f} |')
lines+=['','## Numerical differences','',
'Five prompts with 48 fixed generation positions each. Prompt-produced logits are checked separately and stay byte-identical. The full decode comparison covers 235 positions per mode, including continuation after end-of-answer: 35,704,960 vocabulary logits. Observed transformer layers remain byte-identical under the shared token history. The copied Q8 control must match every compared logit exactly. KL is stock-to-candidate in nats.','',
'| Mode | Decode top-1 agreement | Mean KL | Max logit error | Max probability change |','|---|---:|---:|---:|---:|']
for mode in modes:
    rs=[r for r in d['accuracy'] if r['mode']==mode]
    lines.append(f'| {mode} | {100*statistics.mean(r["decode_top1_agreement"] for r in rs):.2f}% | {statistics.mean(r["mean_kl_nats"] for r in rs):.6f} | {max(r["max_abs_error"] for r in rs):.6f} | {max(r["max_probability_difference"] for r in rs):.6f} |')
lines+=['','The following comparison stops at the first stock end-of-answer token (GGUF EOS 151645), or the generation limit; it excludes the prompt-produced first token. Free responses are compared before their first end marker.','',
'| Mode | Answer top-1 matches | Answer mean KL | Diagnostic passes | Visible responses matching stock |','|---|---:|---:|---:|---:|']
for mode in modes:
    rs=[r for r in d['accuracy'] if r['mode']==mode];qs=[r for r in d['quality'] if r['mode']==mode]
    n=sum(r['answer_decode_steps'] for r in rs);matches=sum(r['answer_top1_matches'] for r in rs);kl=sum(r['answer_mean_kl_nats']*r['answer_decode_steps'] for r in rs)/n
    lines.append(f'| {mode} | {matches}/{n} ({100*matches/n:.2f}%) | {kl:.6f} | {sum(r["diagnostic_pass"] for r in qs)}/5 | {sum(r["response_matches_stock"] for r in qs)}/5 |')
lines+=['','## Actual Q2 responses','']
for r in d['quality']:
    if r['mode']=='output-q2':
        lines.extend([f'### {r["case"]}: {"pass" if r["diagnostic_pass"] else "fail"}', '', '````text',r['response'],'````',''])
lines+=['The arithmetic test asks 17 times 23; the correct answer is 391. The stock model already fails this diagnostic. The case scores are small scripted diagnostics, not a quality benchmark. Fixed generation length can truncate longer responses. Full responses for all modes are saved in `summary.json`.','',
'### Code completion with a larger generation limit','',
'Q2 starts a longer docstring and its 48-token code response is truncated before a return statement. An additional 128-token run of the same prompt checks whether it can complete the function. This is reported separately and does not replace the original fixed-protocol failure.','']
for r in d['code_budget_followup']:
    lines.extend([f'{r["mode"]}: {"pass" if r["diagnostic_pass"] else "fail"}', '', '````text',r['response'],'````',''])
lines+=[
'## Validation and limits','',
'130 native runs: five smoke runs, three stock continuation seeds, 75 matched timing runs, 45 accuracy/free-running runs and two extended code-budget follow-ups. Hook counts require zero prefill calls and N-1 decode calls. Stock CPU_REPACK is confirmed in every log. Fresh copied-Q8 controls match stock logits and token sequences.','',
'An additional 3,000 randomized native cases compare the stock Q2/Q8_K dot with an independently accumulated dequantized scalar reference, covering 768, 896 and 1024 source widths, all-zero inputs and exact zero padding. Five more inference smoke runs pass undefined-behavior sanitization. Sanitization covers the runner, hook and test harness; prebuilt stock libraries are not instrumented. Details are in `sanitizer.json`.','',
'The manifest binds source, reused runner, executable and recorded evidence. CI checks saved evidence; it does not rerun model inference. Native execution is validated on this Windows machine only. The hook uses process-global state for a single model/context and is not a concurrent serving implementation.','',
'Q2 results must be weighed against numerical drift and broader quality. No held-out corpus perplexity was measured. Timings can vary with other system activity; fewer bytes alone do not prove the entire speedup comes from memory bandwidth. Removing duplicate resident weights remains separate work.','',
'[Reproduction instructions](README.md). Repository code is MIT; model weights retain their own licence.','']
rendered='\n'.join(line.rstrip() for line in '\n'.join(lines).splitlines())+'\n'
(HERE/'CASE_STUDY.md').write_text(rendered,encoding='utf-8',newline='\n')

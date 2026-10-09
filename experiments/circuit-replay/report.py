"""Summarize seven paired fresh-process runs on two real upstream circuits."""
import json,pathlib,statistics
HERE=pathlib.Path(__file__).resolve().parent
raw=json.loads((HERE/'runs.json').read_text());rows=[]
for h in [23,25]:
    for mode in ['python','dense','active']:
        ds=sorted([r for r in raw if r['h']==h and r['mode']==mode],key=lambda r:r['repetition'])
        bs=sorted([r for r in raw if r['h']==h and r['mode']=='python'],key=lambda r:r['repetition'])
        row=dict(h=h,mode=mode,samples=len(ds))
        for field in ['total_seconds','dirty_seconds','process_wall_seconds','peak_working_set_bytes']:
            values=[r[field] for r in ds];row[field]=statistics.median(values);row[field+'_min']=min(values);row[field+'_max']=max(values)
        for field in ['total_seconds','dirty_seconds','process_wall_seconds']:
            ratios=[b[field]/r[field] for r,b in zip(ds,bs)]
            row[field+'_paired_speedup']=statistics.median(ratios);row[field+'_speedup_min']=min(ratios);row[field+'_speedup_max']=max(ratios)
        rows.append(row)
(HERE/'summary.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8',newline='\n')
lines=['# Case study 15: Native exact circuit replay','',
'This experiment compares an independently written native checker using 1GBInteger packed 64-bit limbs with CrocSwap/integer-mult-bounds\' Python circuit replay checker. It tests a concrete verification workload; it does not improve the multiplication exponent, implement the whole multiplication algorithm, or validate its inherited mathematical assumptions.','',
'Upstream commit: `d1d6c070f5a8c684727ee7ec35d930f9ebfa9758`. Actual inputs: `research/pair-assembly/frame/frame-word-23.json.gz` and `frame-word-25.json.gz`. The external repository remains unchanged. Measured 2026-10-09 on Windows 11 / AMD Ryzen 7 9800X3D, CPython 3.14, Clang 22.1.8 LLVM-MinGW, `-O3 -march=native`. All checkers are single-threaded.','',
'## Result','',
'Seven fresh process runs per implementation and circuit, with rotating order and warm filesystem caches. Each complete verifier includes JSON parsing, source and frame checks, symbolic output checks, rank histogram, scatter identity, and full basis replay in both orientations. Gzip decompression is done once outside timing for every implementation. No correctness check is replaced by sampling or checksums.','',
'| Circuit | Implementation | Complete verifier median [min, max], seconds | Paired speedup vs Python | Peak working set median, MiB |','|---|---|---:|---:|---:|']
for r in rows:lines.append(f'| h={r["h"]} | {r["mode"]} | {r["total_seconds"]:.4f} [{r["total_seconds_min"]:.4f}, {r["total_seconds_max"]:.4f}] | {r["total_seconds_paired_speedup"]:.2f}x | {r["peak_working_set_bytes"]/2**20:.1f} |')
lines+=['','The separate full-basis phase initializes the exact identity map, replays every literal XOR operation in both orientations, and checks every resulting coefficient including restored scratch inputs. The Python wrapper inserts two clock reads through the AST while retaining every original operation and assertion. Separate runs of the untouched upstream function confirm matching results.','',
'| Circuit | Implementation | Full-basis phase median, seconds | Paired speedup vs Python | Full process wall median, seconds |','|---|---|---:|---:|---:|']
for r in rows:lines.append(f'| h={r["h"]} | {r["mode"]} | {r["dirty_seconds"]:.4f} | {r["dirty_seconds_paired_speedup"]:.2f}x | {r["process_wall_seconds"]:.4f} |')
lines+=['','Paired speedups use each repetition\'s Python time divided by its native time; they may differ from ratios of median times. Complete-verifier timings exclude process startup; process wall times above include it. All repetitions and source/input hashes are retained.','',
'## What changed','',
'The native verifier reuses `inference::Bits` from `xor_inference.h` for symbolic vectors. It constructs the complete basis map in a contiguous matrix of uint64 limbs. The `dense` control XORs every limb in a row. The `active` mode tracks each row\'s first and last nonzero limb, XORs only that source interval, and trims zero ends after cancellation. The stored matrix capacity is the same in both modes; active intervals reduce work and potential memory accesses, not matrix allocation. No hardware memory-traffic counter was measured.','',
'Frame incidence, symbolic output expectations, scalar rank counts, and every final basis coefficient are checked exactly. The native code parses the same decompressed JSON and preserves operation/source order. The native frame/output implementation also uses masks and indexed lookups, so total speedup includes more than optimized XOR. Comparing the full-basis phase and the native dense control distinguishes those effects.','',
'| Circuit | Full word XORs per orientation | Basis vectors | Scratch roles | Dense basis matrix bytes |','|---|---:|---:|---:|']
for h in [23,25]:
    r=next(r for r in raw if r['h']==h and r['mode']=='active')
    lines.append(f'| h={h} | {r["word_length"]:,} | {r["full_basis_vectors"]:,} | {r["roles"]:,} | {r["matrix_bytes"]:,} |')
lines+=['','A fixed 1 GB allocation is unnecessary for these inputs. The native matrix dimensions follow the actual circuit. The user\'s lazy rotation/XOR recurrence is not used: these are arbitrary published gate sequences, so each gate is replayed.','',
'## Validation','',
'All 42 timed runs pass. Both native variants match the upstream checker\'s h, role count, output count, elementary XOR count, full word length, basis dimension, rank mass and complete rank histogram. Two additional runs compare against the untouched upstream function.','',
'Seven corrupted fixtures are rejected by the original Python checker and both native variants: self-XOR, broken frame, wrong symbolic output, missing scatter operation, missing source, out-of-range role and invalid JSON. This gives 21 rejection checks; the missing-scatter fixture reaches the full-basis check. Four complete native replays pass Clang undefined-behavior sanitization (both modes and both circuits). Details and errors are retained in `validation.json`.','',
'These results establish a practical verifier improvement on two pinned published circuits and the tested Windows machine. They are not a proof that this independently written verifier accepts/rejects every possible input identically, a replacement for the rest of upstream\'s arithmetic/profile/formal verification, or a multiplication speed claim.','',
'## Reproduce and attribution','',
'See [README](README.md) for commands. Native code and this harness are MIT. The externally fetched upstream verifier and circuit artifacts retain their Apache-2.0 licence and contributor attribution; they are not relicensed or vendored here. The checker protocol and fixtures come from [CrocSwap/integer-mult-bounds](https://github.com/CrocSwap/integer-mult-bounds/tree/d1d6c070f5a8c684727ee7ec35d930f9ebfa9758), particularly `scripts/experiments/binary_frame_replay.py` and the pair-assembly frame proof. The nlohmann JSON header retains its MIT licence.','']
(HERE/'CASE_STUDY.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')

"""Collect the fixed native XOR inference suite and preserve provenance."""
import hashlib
import csv
import json
import os
import pathlib
import platform
import subprocess
import time

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'experiments'/'xor-inference'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    raw=OUT/'raw';raw.mkdir(parents=True,exist_ok=True)
    sources=['xor_inference.h','xor_inference.cpp','test_xor_inference.cpp','temporal_kernel.h',
             'run_xor_inference.py','experiments/xor-inference/PROTOCOL.md']
    exe=ROOT/('xor_inference.exe' if os.name=='nt' else 'xor_inference')
    source_hashes={p:sha(ROOT/p) for p in sources}
    args=[str(exe),str(raw.relative_to(ROOT))]
    start=time.perf_counter();subprocess.run(args,cwd=ROOT,check=True)
    elapsed=time.perf_counter()-start
    for path in raw.glob('*.csv'):
        with path.open(newline='') as f:
            records=csv.reader(f);columns=len(next(records))
            assert all(len(row)==columns for row in records),f'Malformed CSV: {path}'
    assert source_hashes=={p:sha(ROOT/p) for p in sources},'Source changed during collection'
    manifest={'date':'2026-10-09','timezone':'Australia/Sydney','platform':platform.platform(),
              'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
              'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
              'compiler':subprocess.check_output(['clang++','--version'],text=True),
              'flags':'-O3 -std=c++17','source_sha256':source_hashes,'executable_sha256':sha(exe),
              'arguments':['xor_inference',raw.relative_to(ROOT).as_posix()],
              'wall_seconds':elapsed,'outputs':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(raw.glob('*.csv'))}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Saved raw counters, learned weights and inference timings with source/output hashes.')


if __name__=='__main__':main()

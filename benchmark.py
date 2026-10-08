#!/usr/bin/env python3
"""Correctness gate and repeatable native throughput benchmark."""
import argparse,hashlib,pathlib,re,statistics,subprocess,tempfile
ROOT=pathlib.Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--exe',default=str(ROOT/('integer_universe.exe' if __import__('os').name=='nt' else 'integer_universe')))
p.add_argument('--mb',type=int,default=16)
p.add_argument('--steps',type=int,default=5)
p.add_argument('--threads',type=int,default=4)
p.add_argument('--repeats',type=int,default=3)
a=p.parse_args()
if min(a.mb,a.steps,a.threads,a.repeats)<1:raise SystemExit('Parameters must be positive')
modes={'scalar': ['--fast'],'avx2':['--avx2'],'cached-scalar':['--cache-mask'],'cached-avx2':['--avx2-cache']}
results={}
with tempfile.TemporaryDirectory() as d:
 for name,flags in modes.items():
  trials=[];hashes=[]
  for rep in range(a.repeats):
   dest=pathlib.Path(d)/f'{name}-{rep}.bin'
   cmd=[a.exe,'--mb',str(a.mb),'--steps',str(a.steps),'--threads',str(a.threads),'--seed','12345','--sample-bytes','1048576','--out',str(dest)]+flags
   result=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=240)
   times=[float(x) for x in re.findall(r'step=\d+ seconds=([0-9.eE+-]+)',result.stdout)]
   if len(times)!=a.steps:raise RuntimeError(f'Unexpected step count for {name}: {result.stdout}')
   trials.append(statistics.median(times))
   hashes.append(hashlib.sha256(dest.read_bytes()).hexdigest())
  if len(set(hashes))!=1:raise AssertionError(f'{name} not deterministic')
  results[name]=(statistics.median(trials),hashes[0])
 ref=results['scalar'][1]
 for name,(sec,digest) in results.items():
  if digest!=ref:raise AssertionError(f'Output mismatch: {name} vs scalar')
  print(f'{name:15s} median_step_ms={sec*1000:.3f} speedup={results["scalar"][0]/sec:.3f}x sha256={digest[:16]}')
 print('PASS: all modes produce identical evolved-state samples')

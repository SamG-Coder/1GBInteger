#!/usr/bin/env python3
"""Verify lazy samples are bit-identical to fully materialized integer evolution."""
import hashlib,pathlib,subprocess,tempfile,os
root=pathlib.Path(__file__).resolve().parent
ext='.exe' if os.name=='nt' else ''
full=root/('integer_universe'+ext)
lazy=root/('integer_universe_lazy'+ext)
with tempfile.TemporaryDirectory() as d:
 for mb in (1,16):
  for seed in (0,1,12345,987654321):
   for steps in (0,1,2,7,65,129):
    files=[pathlib.Path(d)/'full.bin',pathlib.Path(d)/'lazy.bin']
    base=['--mb',str(mb),'--seed',str(seed),'--steps',str(steps),'--sample-bytes','8192']
    subprocess.run([str(full),*base,'--fast','--threads','2','--out',str(files[0])],check=True,capture_output=True)
    subprocess.run([str(lazy),*base,'--out',str(files[1])],check=True,capture_output=True)
    if files[0].read_bytes()!=files[1].read_bytes():raise AssertionError(f'Mismatch mb={mb} seed={seed} steps={steps}')
 print('PASS: 48 native lazy/full comparisons produce identical sampled bytes')

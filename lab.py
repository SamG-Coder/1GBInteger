#!/usr/bin/env python3
"""Statistical checks on samples extracted from the ACTUAL evolved native integer.
These tests compare classical deterministic outputs with ideal iid fair quantum measurements.
Passing them does not establish quantum equivalence.
"""
import argparse, hashlib, json, math, os, pathlib, subprocess, tempfile
ROOT=pathlib.Path(__file__).resolve().parent

def splitmix_bits(seed,n):
    x=seed&((1<<64)-1); out=bytearray()
    while len(out)*8<n:
        x=(x+0x9e3779b97f4a7c15)&((1<<64)-1)
        z=x;z=((z^(z>>30))*0xbf58476d1ce4e5b9)&((1<<64)-1)
        z=((z^(z>>27))*0x94d049bb133111eb)&((1<<64)-1);z^=z>>31
        out.extend(z.to_bytes(8,'little'))
    return bytes(out[:(n+7)//8])

def analyze(raw,n):
    bits=[(b>>j)&1 for b in raw for j in range(8)][:n]
    ones=sum(bits); p=ones/n; runs=1+sum(a!=b for a,b in zip(bits,bits[1:]));
    def lag(k):return sum((1 if bits[i]==bits[i+k] else -1) for i in range(n-k))/(n-k)
    hist=[0]*256
    for b in raw:hist[b]+=1
    total=len(raw); chi=sum((c-total/256)**2/(total/256) for c in hist)
    entropy=-sum((c/total)*math.log2(c/total) for c in hist if c)
    runs_z=(runs-(n+1)/2)/math.sqrt((n-1)/4)
    blocks=[sum(bits[i:i+256]) for i in range(0,n-255,256)]
    return {'sample_bits':n,'ones_fraction':p,'ones_zscore':(ones-n/2)/math.sqrt(n/4),
      'runs':runs,'runs_zscore':runs_z,'adjacent_equal_fraction':(n-runs)/(n-1),
      'lag_correlations':{str(k):lag(k) for k in (1,2,4,8,16,32,64) if k<n},
      'byte_entropy_bits':entropy,'byte_chi_square':chi,'byte_chi_square_expected_mean':255,
      'block_256_ones':blocks[:256], 'block_256_expected_mean':128,
      'sample':bits[:512], 'sha256':hashlib.sha256(raw).hexdigest(),
      'reference':'Ideal independent quantum Z measurements of |+> give P(0)=P(1)=0.5; not a Bell or interference test.'}

def run(seed=12345,n=65536,gb=0,steps=1,mb=0):
    if not (0<=seed<2**64 and 4096<=n<=1000000 and 0<=steps<=1000 and 0<=gb<=100 and 0<=mb<=100000):raise ValueError('Invalid parameters')
    if gb or mb:
        binary=ROOT/('integer_universe.exe' if os.name=='nt' else 'integer_universe')
        if not binary.exists():raise RuntimeError('Compile main.cpp first (see README)')
        with tempfile.TemporaryDirectory() as tmp:
            dest=pathlib.Path(tmp)/'sample.bin'
            cmd=[str(binary),'--seed',str(seed),'--steps',str(steps),'--sample-bytes',str((n+7)//8*8),'--out',str(dest)]
            cmd+=['--mb',str(mb)] if mb else ['--gb',str(gb)]
            proc=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=180)
            raw=dest.read_bytes()
        origin='evolved_native_integer' if steps else 'initialized_native_integer'
        native=proc.stdout
    else:
        raw=splitmix_bits(seed,n);origin='seed_stream_only';native=''
    result=analyze(raw,n);result.update(seed=seed,origin=origin,integer_gb=gb,integer_mb=mb,steps=steps,native_output=native,
       notes=['Statistical resemblance to quantum randomness does not imply quantum physics.',
       'Native mode samples evenly spaced 64-bit limbs of the evolved integer; no independent random stream is substituted.',
       'The evolution is a linear rotation followed by a fixed XOR mask; it is not a physical model.',
       'Sampled limb order is spatial, not a sequence of measurements over time.'])
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=12345);p.add_argument('--bits',type=int,default=65536);p.add_argument('--gb',type=int,default=0);p.add_argument('--mb',type=int,default=0);p.add_argument('--steps',type=int,default=1);p.add_argument('--out',default='results.json')
    a=p.parse_args();r=run(a.seed,a.bits,a.gb,a.steps,a.mb);pathlib.Path(a.out).write_text(json.dumps(r,indent=2));print('origin:',r['origin'],'ones z:',round(r['ones_zscore'],3),'runs z:',round(r['runs_zscore'],3),'entropy:',round(r['byte_entropy_bits'],4),'sha256:',r['sha256'])

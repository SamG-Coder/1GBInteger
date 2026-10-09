"""Collect the fixed matched-workload suite using native computation only."""
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import time

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'experiments'/'rule-comparison'
EXT='.exe' if os.name=='nt' else ''
SEEDS={'development':[0,1,42,12345], 'validation':[20261009,314159265,2718281828,18446744073709551615]}
RULES=['xor','add','local']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw=OUT/'raw';raw.mkdir(parents=True,exist_ok=True)
    sources=['rule_kernels.h','rule_compare.cpp','rule_diagnostics.h','temporal_kernel.h',
             'test_rule_comparison.py','test_rule_edges.cpp','run_rule_comparison.py',
             'experiments/rule-comparison/PROTOCOL.md']
    manifest={'date':'2026-10-09','timezone':'Australia/Sydney','platform':platform.platform(),
              'cpu':platform.processor(),'logical_cpus':os.cpu_count(),
              'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'compiler':subprocess.check_output(['clang++','--version'],text=True),
              'flags':'-O3 -std=c++17 (Windows additionally -lpsapi)',
              'source_sha256':{s:sha(ROOT/s) for s in sources},
              'executable_sha256':sha(ROOT/('rule_compare'+EXT)), 'runs':[], 'controls':[]}
    def run(kind,rule,seed,group,filename,**options):
        dest=raw/filename
        args=['--mode',kind,'--rule',rule,'--seed',str(seed),'--out',str(dest.relative_to(ROOT))]
        for k,v in options.items():args.extend(['--'+k,str(v)])
        start=time.perf_counter()
        proc=subprocess.run([str(ROOT/('rule_compare'+EXT)),*args],cwd=ROOT,check=True,capture_output=True,text=True)
        manifest['runs'].append({'kind':kind,'rule':rule,'seed':str(seed),'group':group,
                                 'file':dest.relative_to(ROOT).as_posix(),'arguments':args,
                                 'sha256':sha(dest),'process_wall_seconds':time.perf_counter()-start})
        print(proc.stdout.strip(),flush=True)
    for group,seeds in SEEDS.items():
        for seed in seeds:
            control=ROOT/'experiments'/'raw'/f'{group}-{seed}-spaced-control.json'
            manifest['controls'].append({'group':group,'seed':str(seed),'file':control.relative_to(ROOT).as_posix(),'sha256':sha(control)})
            for rule in RULES:
                for size in (32768,1000000):
                    run('stats',rule,seed,group,f'stats-{rule}-{seed}-{size}.json',bytes=size,steps=8191,atoms=1024)
                run('damage',rule,seed,group,f'damage-{rule}-{seed}.json',steps=4096)
                run('graph',rule,seed,group,f'graph-{rule}-{seed}.json')
    for size in (1000000,16000000,1000000000):
        for repeat in range(3):
            for rule in RULES[repeat:]+RULES[:repeat]:
                run('bench',rule,12345,'benchmark',f'bench-{rule}-{size}-{repeat}.json',bytes=size,steps=3)
    for file,digest in manifest['source_sha256'].items():
        assert sha(ROOT/file)==digest, f'Source changed during collection: {file}'
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Saved {len(manifest["runs"])} native runs and eight control references.',flush=True)


if __name__=='__main__':main()

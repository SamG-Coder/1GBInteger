"""Run the fixed case-study protocol; Python only orchestrates native processes."""
import hashlib
import json
import os
import pathlib
import platform
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT/'experiments'/'raw'
EXT = '.exe' if os.name == 'nt' else ''
SEEDS = {'development': [0, 1, 42, 12345],
         'validation': [20261009, 314159265, 2718281828, 18446744073709551615]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tracked_sources = ['temporal_experiments.cpp', 'structural_experiments.cpp',
                       'temporal_kernel.h', 'experiments/PROTOCOL.md', 'run_experiments.py',
                       'test_experiments.py', 'analyze_experiments.py']
    manifest = {'date': '2026-10-09', 'timezone': 'Australia/Sydney',
                'platform': platform.platform(), 'cpu': platform.processor(),
                'logical_cpus': os.cpu_count(), 'base_commit': subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_sha256': {p: sha(ROOT/p) for p in tracked_sources},
                'compiler': subprocess.check_output(['clang++', '--version'], text=True),
                'build_flags': '-O3 -std=c++17 -pthread', 'runs': []}
    manifest['executable_sha256'] = {name: sha(ROOT/(name+EXT))
                                      for name in ('temporal_experiments', 'structural_experiments')}
    for group, seeds in SEEDS.items():
        for seed in seeds:
            for layout in ('spaced', 'adjacent'):
                for source in ('native', 'control'):
                    path = OUT/f'{group}-{seed}-{layout}-{source}.json'
                    args = ['--seed', str(seed), '--layout', layout, '--source', source,
                            '--atoms', '1024', '--observations', '8192', '--threads', '8',
                            '--out', str(path.relative_to(ROOT))]
                    p = subprocess.run([str(ROOT/('temporal_experiments'+EXT)), *args],
                                       cwd=ROOT, check=True, text=True, capture_output=True)
                    manifest['runs'].append({'group': group, 'file': str(path.relative_to(ROOT)).replace('\\','/'),
                                             'arguments': args, 'sha256': sha(path), 'stdout': p.stdout.strip()})
                    print(p.stdout.strip(), flush=True)
    structure = OUT/'structure.csv'
    with structure.open('w', newline='') as f:
        subprocess.run([str(ROOT/('structural_experiments'+EXT))], check=True, stdout=f, text=True)
    manifest['structure_sha256'] = sha(structure)
    (ROOT/'experiments'/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Saved all 32 runs, structural study, and provenance manifest.')


if __name__ == '__main__':
    main()

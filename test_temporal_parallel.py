"""Check parallel observations, aggregate checksums, and full-state samples."""
import csv
import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
EXT = '.exe' if os.name == 'nt' else ''
MASK = (1 << 64) - 1


def run(name, args):
    p = subprocess.run([str(ROOT / (name + EXT)), *map(str, args)],
                       check=True, capture_output=True, text=True)
    return dict(part.split('=', 1) for part in p.stdout.split() if '=' in part)


def mix(x):
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK
    return x ^ (x >> 31)


def rows(path):
    with path.open(newline='') as f:
        return [tuple(int(v, 16 if k == 'limb_hex' else 10)
                      for k, v in row.items()) for row in csv.DictReader(f)]


with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    cases = 0
    for seed in (0, 12345, MASK):
        for steps, interval in ((0, 1), (1, 1), (63, 1), (64, 1), (65, 1),
                                (129, 7), (130, 65), (257, 10), (3, 100)):
            common = ['--mb', 1, '--seed', seed, '--atoms', 8,
                      '--steps', steps, '--interval', interval]
            ref = tmp / 'reference.csv'
            run('integer_universe_temporal', [*common, '--out', ref])
            expected = rows(ref)
            checksum = 0
            for t, j, _, _, _, value in expected:
                checksum ^= mix(value ^ (j << 32) ^ t)
            for threads in (1, 2, 4, 8, 16):
                out = tmp / 'parallel.csv'
                stats = run('temporal_parallel', [*common, '--threads', threads, '--out', out])
                assert rows(out) == expected, (seed, steps, interval, threads)
                assert int(stats['checksum'], 16) == checksum
                assert int(stats['ones']) == sum(r[4] for r in expected)
                assert int(stats['observations']) == len(expected)
            final_t = steps // interval * interval
            sample = tmp / 'full.bin'
            run('integer_universe', ['--mb', 1, '--seed', seed, '--steps', final_t,
                                    '--fast', '--threads', 2, '--sample-bytes', 64, '--out', sample])
            raw = sample.read_bytes()
            for j, row in enumerate(r for r in expected if r[0] == final_t):
                assert row[-1] == int.from_bytes(raw[j*8:j*8+8], 'little')
            cases += 1
    # A true 1 GB full-state cross-check, separate from sparse performance runs.
    common = ['--gb', 1, '--seed', 12345, '--steps', 2]
    out = tmp / 'parallel.csv'
    run('temporal_parallel', [*common, '--atoms', 8, '--interval', 1, '--threads', 4, '--out', out])
    sample = tmp / 'full.bin'
    run('integer_universe', [*common, '--fast', '--threads', 8, '--sample-bytes', 64, '--out', sample])
    raw = sample.read_bytes()
    assert [r[-1] for r in rows(out) if r[0] == 2] == [int.from_bytes(raw[j:j+8], 'little') for j in range(0, 64, 8)]
    print(f'PASS: {cases} temporal/full cases x 5 thread counts; true 1 GB full-state check')

"""Independent full-buffer oracle for every native diagnostic counter, plus lab API/CLI."""
import collections
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import urllib.request
from http.server import ThreadingHTTPServer
import lab
from server import Handler

ROOT = pathlib.Path(__file__).resolve().parent
EXT = '.exe' if os.name == 'nt' else ''
MASK = (1 << 64) - 1
C = 0x9e3779b97f4a7c15


def mix(x):
    x &= MASK
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK
    return x ^ (x >> 31)


def check(data, seed, layout):
    atoms, n, limbs = 8, 512, 17
    state = [mix(seed + (i+1)*C) for i in range(limbs)]
    mask = [mix(seed ^ ((i*C) & MASK)) for i in range(limbs)]
    traces = [[] for _ in range(atoms)]
    for _ in range(n):
        for j in range(atoms):
            traces[j].append(state[j*limbs//atoms if layout == 'spaced' else j] & 1)
        state = [((state[i] << 1) | (state[i-1] >> 63)) & MASK ^ mask[i] for i in range(limbs)]
    assert data['atom_ones'] == list(map(sum, traces))
    assert data['ones'] == sum(map(sum, traces))
    for k, lag in enumerate(data['lags']):
        assert data['lag_n'][k] == atoms*(n-lag)
        assert data['lag_equal'][k] == sum(b[t] == b[t-lag] for b in traces for t in range(lag, n))
    expected = {k: [0]*len(data[k]) for k in ('transitions', 'blocks', 'runs', 'first', 'gaps')}
    left = tails = 0
    for bits in traces:
        runs = []
        for b in bits:
            if not runs or runs[-1][0] != b:
                runs.append([b, 1])
            else:
                runs[-1][1] += 1
        for _, length in runs[1:-1]:
            expected['runs'][min(length, 128)] += 1
        left += 1
        for a, b in zip(bits, bits[1:]):
            expected['transitions'][a*2+b] += 1
        events = []
        for trial, pos in enumerate(range(0, n, 8), 1):
            value = sum(bits[pos+k] << k for k in range(8))
            expected['blocks'][value] += 1
            if value == 255:
                events.append(trial)
        expected['first'][events[0] if events else 0] += 1
        for a, b in zip(events, events[1:]):
            expected['gaps'][b-a] += 1
        tails += not events or events[-1] < n//8
    for key, value in expected.items():
        assert data[key] == value, key
    assert data['run_left_censored'] == left
    assert data['run_right_censored'] == atoms
    assert data['tail_censored'] == tails
    assert data['cross_equal'] == sum(traces[j][t] == traces[j+1][t] for j in range(0, atoms, 2) for t in range(n))
    assert data['cross_n'] == atoms//2*n
    assert data['predictions'] == atoms//2*(n-65)
    assert data['predicted'] == sum(traces[j+1][t+1] == (traces[j+1][t]^traces[j][t-63]^traces[j][t-64])
                                    for j in range(0, atoms, 2) for t in range(64, n-1))


with tempfile.TemporaryDirectory() as td:
    out = pathlib.Path(td)/'result.json'
    for source in ('native', 'control'):
        for seed in (0, 12345, MASK):
            for layout in ('adjacent', 'spaced'):
                reference = None
                for threads in (1, 2, 3, 16):
                    subprocess.run([str(ROOT/('temporal_experiments'+EXT)), '--seed', str(seed),
                                    '--source', source, '--layout', layout, '--atoms', '8', '--observations', '512',
                                    '--limbs', '17', '--threads', str(threads), '--out', str(out)], check=True, capture_output=True)
                    data = json.loads(out.read_text())
                    if source == 'native':
                        check(data, seed, layout)
                    data.pop('seconds'); data.pop('threads')
                    if reference is not None:
                        assert reference == data
                    reference = data
    for source in ('native', 'control'):
        bad = subprocess.run([str(ROOT/('temporal_experiments'+EXT)), '--source', source,
                              '--observations', '0', '--out', str(out)], capture_output=True)
        assert bad.returncode != 0
    subprocess.run([str(ROOT/('structural_experiments'+EXT))], check=True, capture_output=True)
    subprocess.run([sys.executable, str(ROOT/'lab.py'), '--mb', '1', '--bits', '4096', '--mode', 'lazy',
                    '--out', str(out)], check=True, capture_output=True)
    assert json.loads(out.read_text())['sha256'] == lab.run(n=4096, mb=1, mode='full')['sha256']
    http = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True); thread.start()
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{http.server_port}/api/run?mb=1&bits=4096&mode=lazy') as response:
            assert json.load(response)['origin'] == 'lazy_evolved_integer'
    finally:
        http.shutdown(); http.server_close(); thread.join()
print('PASS: all native counters against full-state oracle, controls/thread invariance, structure, lab CLI and HTTP API')

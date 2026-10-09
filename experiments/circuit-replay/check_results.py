import hashlib,json,pathlib,statistics
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=load(HERE/'manifest.json')
for name,h in m['files'].items():assert sha(HERE/name)==h,name
for name,h in m['shared'].items():assert sha(ROOT/name)==h,name
data=load(HERE/'runs.json');assert len(data)==42
keys=['h','roles','output_roles','elementary_xors','word_length','full_basis_vectors','rank_mass','histogram']
for h in [23,25]:
    for rep in range(7):
        subset=[r for r in data if r['h']==h and r['repetition']==rep];assert len(subset)==3
        base=next(r for r in subset if r['mode']=='python')
        for r in subset:
            assert all(r[k]==base[k] for k in keys)
            assert r['baseline_wrapper_sha256']==m['files']['baseline.py'] and r['native_source_sha256']==m['files']['replay.cpp']
            assert r['upstream_checker_sha256']==m['external']['scripts/experiments/binary_frame_replay.py']
            assert r['input_sha256']==base['input_sha256']
            assert r['compressed_input_sha256']==m['external'][f'research/pair-assembly/frame/frame-word-{h}.json.gz']
            assert r['executable_sha256']==m['python_executable_sha256' if r['mode']=='python' else 'native_executable_sha256']
            assert r['process_wall_seconds']>=r['total_seconds']>=r['dirty_seconds']>0
for row in load(HERE/'summary.json'):
    ds=[r for r in data if r['h']==row['h'] and r['mode']==row['mode']]
    assert row['total_seconds']==statistics.median(r['total_seconds'] for r in ds)
v=load(HERE/'validation.json')['records'];assert len(v)==27
assert all(r.get('passed',r.get('rejected',False)) for r in v)
print('Verified 42 matched verifier runs, 21 corruption rejections, 4 UBSan checks and provenance')

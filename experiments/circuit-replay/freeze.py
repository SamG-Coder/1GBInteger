import hashlib,json,pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json']
up=ROOT/'.local-llm/integer-mult-bounds'
external=['scripts/experiments/binary_frame_replay.py']+[f'research/pair-assembly/frame/frame-word-{h}.json.gz' for h in [23,25]]
record=dict(upstream_commit='d1d6c070f5a8c684727ee7ec35d930f9ebfa9758',
    files={p.relative_to(HERE).as_posix():sha(p) for p in files},
    shared={name:sha(ROOT/name) for name in ['xor_inference.h','temporal_kernel.h']},
    external={name:sha(up/name) for name in external},
    json_header_sha256=sha(ROOT/'.local-llm/llama.cpp/vendor/nlohmann/json.hpp'),
    json_header_llama_commit='de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b',
    native_executable_sha256=sha(ROOT/'.local-llm/circuit-replay/replay.exe'),python_executable_sha256=sha(pathlib.Path(sys.executable)))
(HERE/'manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')

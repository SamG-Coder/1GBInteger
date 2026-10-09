"""Bind the experiment and reused runner to the measured executable."""
import hashlib,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources=[p for p in HERE.iterdir() if p.suffix in ('.cpp','.h','.py')]
evidence=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p not in sources and p.name!='manifest.json']
shared=['experiments/stock-llama-integration/'+p for p in ['runner.cpp','kernels.h','setup.py','CMakeLists.txt']]+['experiments/llm-inference/'+p for p in ['patch_llama.py','setup.py','run_suite.py','analyze.py']]
record=dict(llama_commit='de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b',model_sha256='c8cd5f37dd1235fb010c45316d4ff8af875e1a4e0ff368b4bf6cacb9053d4919',
    executable_sha256=sha(ROOT/'.local-llm/q2-traffic-build/integer-llm.exe'),source_sha256={p.name:sha(p) for p in sources},
    evidence_sha256={p.relative_to(HERE).as_posix():sha(p) for p in evidence},shared_source_sha256={p:sha(ROOT/p) for p in shared})
(HERE/'manifest.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')

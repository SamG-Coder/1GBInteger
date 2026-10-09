"""Bind finished experiment evidence to exact source and executable bytes."""
import json,pathlib,subprocess,time
from setup import LOCAL,MODEL,MODEL_SHA,MODEL_REV,LLAMA,sha
HERE=pathlib.Path(__file__).resolve().parent
def files_hash(paths):return {p.relative_to(HERE).as_posix():sha(p) for p in sorted(paths)}
assert sha(LOCAL/MODEL)==MODEL_SHA
source=[p for p in HERE.iterdir() if p.suffix in ('.cpp','.h','.py') or p.name=='CMakeLists.txt']
evidence=[p for p in HERE.iterdir() if p.suffix in ('.json','.md','.txt') and p.name!='manifest.json']+list((HERE/'figures').glob('*'))
manifest={'llama_commit':LLAMA,'model_revision':MODEL_REV,'model_file':MODEL,'model_bytes':(LOCAL/MODEL).stat().st_size,'model_sha256':MODEL_SHA,'compiler':subprocess.check_output(['clang++','--version'],text=True),'executable_sha256':sha(LOCAL/'experiment-build/integer-llm.exe'),'source_sha256':files_hash(source),'raw_sha256':{p.name:sha(p) for p in sorted((HERE/'raw').iterdir()) if p.is_file()},'evidence_sha256':files_hash(evidence),'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Bound',len(source),'sources,',len(manifest['raw_sha256']),'raw files,',len(evidence),'derived artifacts')

"""Native padding/oracle and sanitizer evidence. Run after timing finishes."""
import hashlib,json,pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2];HERE=pathlib.Path(__file__).resolve().parent
LOCAL=ROOT/'.local-llm';BUILD=LOCAL/'stock-build';OUT=LOCAL/'q2-traffic-build';SCRATCH=LOCAL/'q2-traffic-evaluation'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
flags=['-O2','-std=c++17','-march=native','-D_WIN32_WINNT=0x0A00','-fsanitize=undefined','-fno-sanitize-recover=all']
def build(exe,sources):
    subprocess.run(['clang++',*flags,'@CMakeFiles/integer-llm.dir/includes_CXX.rsp',*map(str,sources),'@CMakeFiles/integer-llm.dir/linkLibs.rsp','-o',str(exe)],cwd=BUILD,check=True)
test=OUT/'test-padding-ubsan.exe';build(test,[HERE/'test_padding.cpp'])
oracle=json.loads(subprocess.check_output([str(test)],text=True));assert oracle['cases']==3000 and oracle['padding_zero']
exe=OUT/'integer-llm-ubsan.exe';build(exe,[ROOT/'experiments/stock-llama-integration/runner.cpp',HERE/'kernels.cpp'])
runs=[]
for mode in ['stock','output-q8','output-q5','output-q4','output-q2']:
    log=SCRATCH/f'ubsan-{mode}.txt'
    cmd=[str(exe),'--model',str(LOCAL/'Qwen2.5-0.5B-Instruct-Q4_0.gguf'),'--mode',mode,'--prompt',str(SCRATCH/'smoke.txt'),'--output',str(SCRATCH/f'ubsan-{mode}.json'),'--generate','16','--context','512','--threads','8','--repeat','1']
    with log.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=f)
    assert r.returncode==0 and 'runtime error:' not in log.read_text()
    runs.append(dict(mode=mode,exit_code=r.returncode,log_sha256=sha(log)))
(HERE/'sanitizer.json').write_text(json.dumps(dict(scope='Runner, hook and padding test instrumented; existing stock libraries are not.',flags=flags,padding_oracle=oracle,padding_test_sha256=sha(test),executable_sha256=sha(exe),runs=runs),indent=2)+'\n',encoding='utf-8',newline='\n')
print('3000 padding/oracle cases and five sanitized inference smoke modes passed')

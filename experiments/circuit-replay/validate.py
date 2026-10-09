"""Check original upstream equivalence, negative fixtures and native UBSan."""
import copy,json,pathlib,subprocess,sys,hashlib
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1];LOCAL=ROOT/'.local-llm/circuit-replay'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
fields=['h','roles','output_roles','elementary_xors','word_length','full_basis_vectors','rank_mass','histogram']
records=[]
for h in [23,25]:
    path=LOCAL/f'word-{h}.json'
    original=json.loads(subprocess.check_output([sys.executable,str(HERE/'baseline.py'),str(path),'--unmodified'],text=True))
    timed=json.loads((HERE/'raw'/f'h{h}-0-python.json').read_text())
    assert all(original[k]==timed[k] for k in fields)
    records.append(dict(test='unmodified-upstream',h=h,passed=True))
source=json.loads((LOCAL/'word-23.json').read_text())
def self_xor(d):d['ops'][0][0]=d['ops'][0][1]
def frame(d):d['frames'][0][1]=0
def output(d):d['outputs'][0][2]=(d['outputs'][0][2]+1)%d['h']
def scatter(d):d['scatter'].pop(0)
def source_missing(d):d['sources'].pop(next(iter(d['sources'])))
def bounds(d):d['ops'][0][0]=d['R']
for name,mutate in [('self-xor',self_xor),('broken-frame',frame),('wrong-output',output),('missing-scatter',scatter),('missing-source',source_missing),('out-of-bounds',bounds),('invalid-json',None)]:
    data=copy.deepcopy(source)
    if mutate:mutate(data)
    path=LOCAL/(name+'.json');path.write_text(json.dumps(data) if mutate else '{',encoding='utf-8')
    for mode in ['python','active','dense']:
        cmd=[sys.executable,str(HERE/'baseline.py'),str(path),'--unmodified'] if mode=='python' else [str(LOCAL/'replay.exe'),str(path),mode]
        p=subprocess.run(cmd,capture_output=True,text=True)
        assert p.returncode!=0,(name,mode,'accepted corruption')
        records.append(dict(test=name,mode=mode,rejected=True,error=p.stderr[-300:]))
exe=LOCAL/'replay-ubsan.exe'
subprocess.run(['clang++','-O2','-std=c++17','-march=native','-fsanitize=undefined','-fno-sanitize-recover=all','-I'+str(ROOT/'.local-llm/llama.cpp/vendor'),str(HERE/'replay.cpp'),'-lpsapi','-o',str(exe)],check=True)
for h in [23,25]:
    for mode in ['active','dense']:
        p=subprocess.run([str(exe),str(LOCAL/f'word-{h}.json'),mode],capture_output=True,text=True,check=True)
        assert 'runtime error:' not in p.stderr
        result=json.loads(p.stdout);ref=json.loads((HERE/'raw'/f'h{h}-0-python.json').read_text())
        assert all(result[k]==ref[k] for k in fields)
        records.append(dict(test='ubsan',h=h,mode=mode,passed=True))
(HERE/'validation.json').write_text(json.dumps(dict(records=records,sanitizer_executable_sha256=sha(exe)),indent=2)+'\n',encoding='utf-8',newline='\n')
print('Unmodified upstream equivalence, 21 corruption rejections and four full UBSan replays passed')

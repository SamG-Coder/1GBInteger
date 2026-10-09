"""Real-model evaluation: separate uninstrumented timings and teacher-forced traces."""
import argparse, hashlib, json, pathlib, statistics, subprocess, time
from setup import ROOT,LOCAL,MODEL,MODEL_SHA,MODEL_REV,LLAMA,sha
HERE=pathlib.Path(__file__).resolve().parent
RAW=HERE/'raw'
SCRATCH=LOCAL/'evaluation'
MODES=['stock','control','prepack','bitplane','binary','ternary']
def write(p,s):p.write_text(s,encoding='utf-8',newline='\n')
def chat(user):return '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+user+'<|im_end|>\n<|im_start|>assistant\n'
QUESTIONS={
 'math':'What is 17 times 23? Give the answer and a short calculation.',
 'fact':'What is the capital of France? Answer in one sentence.',
 'instruction':'Write exactly these three words in this order, separated by commas: red, green, blue. Do not write anything else.',
 'code':'Write a Python function named add that takes a and b and returns their sum. Output only the code.',
 'conversation':'Remember this: my project code is ORBIT-731.\n'+('We discussed packing books, planning a garden, and making tea.\n'*90)+'What is my project code? Reply with the code only.'}
def run(mode,case,prompt,gen=64,ctx=2048,repeat=3,trace=False,teacher=None,threads=8):
    tag=f'{case}-{mode}'
    pp=SCRATCH/(case+'.txt');write(pp,prompt)
    out=RAW/(tag+'.json')
    cmd=[str(LOCAL/'experiment-build/integer-llm.exe'),'--model',str(LOCAL/MODEL),'--mode',mode,'--prompt',str(pp),'--output',str(out),'--generate',str(gen),'--context',str(ctx),'--threads',str(threads),'--repeat',str(repeat)]
    if trace:cmd+=['--trace',str(SCRATCH/tag),'--logits',str(SCRATCH/(tag+'.f32'))]
    if teacher:cmd+=['--teacher',str(teacher)]
    with (SCRATCH/(tag+'.log')).open('w') as log:subprocess.run(cmd,check=True,cwd=ROOT,stdout=log,stderr=log)
    d=json.loads(out.read_text());print(tag,'pp',round(statistics.median(s['pp_tps'] for s in d['samples']),1),'tg',round(statistics.median(s['generation_tps'] for s in d['samples']),1),'hooks',d['samples'][0]['hook_calls'],flush=True)
    if mode not in ('stock','control'):assert d['samples'][0]['hook_calls']>0
    return d
def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['smoke','timing','accuracy','all'],default='all');a=p.parse_args()
    RAW.mkdir(exist_ok=True);SCRATCH.mkdir(exist_ok=True)
    assert sha(LOCAL/MODEL)==MODEL_SHA
    if a.part=='smoke':
        for m in MODES:run(m,'smoke',chat(QUESTIONS['math']),gen=16,repeat=1)
    if a.part in ('timing','all'):
        for case,n,gen,ctx in [('short',0,32,512),('medium',24,64,1024),('long',110,128,2048),('extended',230,128,4096)]:
            prompt=chat(('A small river flows past the trees and into the lake.\n'*n)+'Summarize the scene in three sentences.')
            # Rotate mode order across workloads to reduce fixed-order drift.
            offset=['short','medium','long','extended'].index(case)
            for m in MODES[offset:]+MODES[:offset]:run(m,case,prompt,gen,ctx)
        for t in [1,4,16]:
            for m in ['stock','prepack']:run(m,f'threads-{t}',chat(QUESTIONS['math']),gen=64,repeat=3,threads=t)
    if a.part in ('accuracy','all'):
        for case,q in QUESTIONS.items():
            base=run('stock','quality-'+case,chat(q),gen=48,ctx=4096,repeat=1,trace=True)
            teacher=SCRATCH/(case+'-teacher.json');write(teacher,json.dumps(base['samples'][0]['tokens']))
            for m in MODES[1:]:
                run(m,'quality-'+case,chat(q),gen=48,ctx=4096,repeat=1,trace=True,teacher=teacher)
                # Independent free-running response measures divergence beyond teacher forcing.
                if m in ('prepack','binary','ternary'):run(m,'free-'+case,chat(q),gen=48,ctx=4096,repeat=1)
    manifest={'llama_commit':LLAMA,'model_revision':MODEL_REV,'model_file':MODEL,'model_bytes':(LOCAL/MODEL).stat().st_size,'model_sha256':MODEL_SHA,'compiler':subprocess.check_output(['clang++','--version'],text=True),'executable_sha256':sha(LOCAL/'experiment-build/integer-llm.exe'),'source_sha256':{p.name:sha(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.cpp','.h','.py') or p.name=='CMakeLists.txt'},'raw_sha256':{p.name:sha(p) for p in sorted(RAW.glob('*.json'))},'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write(HERE/'manifest.json',json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()

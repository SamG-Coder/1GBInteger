"""Matched stock / stock-plus runs. Large dumps remain ignored."""
import argparse,hashlib,json,pathlib,statistics,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/llm-inference'))
from setup import MODEL,MODEL_SHA,LLAMA,sha
from run_suite import chat,QUESTIONS
LOCAL=ROOT/'.local-llm';SCRATCH=LOCAL/'stock-evaluation';RAW=HERE/'raw'
EXE=LOCAL/'stock-build/integer-llm.exe'
MODES=['stock','stock-plus-single','stock-plus']
def write(p,s):p.write_text(s,encoding='utf-8',newline='\n')
def run(mode,case,prompt,gen=64,ctx=2048,repeat=1,threads=8,trace=False,teacher=None):
    SCRATCH.mkdir(exist_ok=True);RAW.mkdir(exist_ok=True)
    tag=f'{case}-{mode}';pp=SCRATCH/(case+'.txt');write(pp,prompt);out=RAW/(tag+'.json')
    cmd=[str(EXE),'--model',str(LOCAL/MODEL),'--mode',mode,'--prompt',str(pp),'--output',str(out),'--generate',str(gen),'--context',str(ctx),'--threads',str(threads),'--repeat',str(repeat)]
    if trace:cmd+=['--trace',str(SCRATCH/tag),'--logits',str(SCRATCH/(tag+'.f32'))]
    if teacher:cmd+=['--teacher',str(teacher)]
    with (SCRATCH/(tag+'.log')).open('w') as log:subprocess.run(cmd,check=True,cwd=ROOT,stdout=log,stderr=log)
    d=json.loads(out.read_text());d['executable_sha256']=sha(EXE);d['stock_repacking_enabled']=True
    write(out,json.dumps(d,indent=2)+'\n')
    for s in d['samples']:
        assert s['prefill_hook_calls']==0
        assert s['hook_calls']==(0 if mode=='stock' else gen-1),(tag,s['hook_calls'])
        assert d['extra_packed_bytes']==0
    print(tag,'PP',round(statistics.median(s['pp_tps'] for s in d['samples']),1),'TG',round(statistics.median(s['generation_tps'] for s in d['samples']),1),flush=True)
    return d
def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['smoke','timing','accuracy','all'],default='all');args=p.parse_args()
    assert sha(LOCAL/MODEL)==MODEL_SHA
    if args.part=='smoke':
        for mode in MODES:run(mode,'smoke',chat(QUESTIONS['math']),gen=16,repeat=2)
    if args.part in ['timing','all']:
        for case,n,gen,ctx in [('short',0,64,512),('medium',24,96,1024),('long',110,128,2048),('extended',230,128,4096)]:
            prompt=chat('A small river flows past the trees and into the lake.\n'*n+'Summarize the scene in three sentences.')
            for rep in range(5):
                order=MODES[rep%3:]+MODES[:rep%3]
                for mode in order:run(mode,f'{case}-{rep}',prompt,gen,ctx)
        for threads in [1,4,16]:
            for rep in range(3):
                for mode in (['stock','stock-plus'] if rep%2==0 else ['stock-plus','stock']):run(mode,f'threads-{threads}-{rep}',chat(QUESTIONS['math']),threads=threads)
    if args.part in ['accuracy','all']:
        for case,q in QUESTIONS.items():
            base=run('stock','quality-'+case,chat(q),gen=48,ctx=4096,trace=True)
            teacher=SCRATCH/(case+'-teacher.json');write(teacher,json.dumps(base['samples'][0]['tokens']))
            for mode in MODES[1:]:
                run(mode,'quality-'+case,chat(q),gen=48,ctx=4096,trace=True,teacher=teacher)
                run(mode,'free-'+case,chat(q),gen=48,ctx=4096)
if __name__=='__main__':main()

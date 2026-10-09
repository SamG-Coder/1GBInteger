"""Matched decode traffic experiment. Fixed stock continuation for all timing."""
import argparse,hashlib,json,pathlib,statistics,subprocess,sys
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/llm-inference'))
from setup import MODEL,MODEL_SHA,sha
from run_suite import chat,QUESTIONS
LOCAL=ROOT/'.local-llm';SCRATCH=LOCAL/'traffic-evaluation';RAW=HERE/'raw'
EXE=LOCAL/'traffic-build/integer-llm.exe'
MODES=['stock','output-q8','output-q5','output-q4']
def write(p,s):p.write_text(s,encoding='utf-8',newline='\n')
def run(mode,case,prompt,gen=64,ctx=2048,trace=False,teacher=None):
    SCRATCH.mkdir(exist_ok=True);RAW.mkdir(exist_ok=True)
    tag=f'{case}-{mode}';pp=SCRATCH/(case+'.txt');write(pp,prompt);out=RAW/(tag+'.json')
    cmd=[str(EXE),'--model',str(LOCAL/MODEL),'--mode',mode,'--prompt',str(pp),'--output',str(out),'--generate',str(gen),'--context',str(ctx),'--threads','8','--repeat','1']
    if trace:cmd+=['--trace',str(SCRATCH/tag),'--logits',str(SCRATCH/(tag+'.f32'))]
    if teacher:cmd+=['--teacher',str(teacher)]
    with (SCRATCH/(tag+'.log')).open('w',encoding='utf-8') as log:subprocess.run(cmd,check=True,cwd=ROOT,stdout=log,stderr=log)
    d=json.loads(out.read_text());d['executable_sha256']=sha(EXE);d['stock_repacking_enabled']=True
    d['teacher_tokens_sha256']=sha(teacher) if teacher else None
    expected={'stock':0,'output-q8':144643072,'output-q5':93592576,'output-q4':76575744}[mode]
    assert d['extra_packed_bytes']==expected
    for s in d['samples']:
        assert s['prefill_hook_calls']==0
        assert s['hook_calls']==(0 if mode=='stock' else gen-1)
    assert 'CPU_REPACK' in (SCRATCH/(tag+'.log')).read_text(encoding='utf-8')
    write(out,json.dumps(d,indent=2)+'\n')
    print(tag,'PP',round(statistics.median(s['pp_tps'] for s in d['samples']),1),'TG',round(statistics.median(s['generation_tps'] for s in d['samples']),1),flush=True)
    return d
def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['smoke','timing','accuracy','all'],default='all');args=p.parse_args()
    assert sha(LOCAL/MODEL)==MODEL_SHA
    if args.part=='smoke':
        for mode in MODES:run(mode,'smoke',chat(QUESTIONS['math']),gen=16)
    if args.part in ['timing','all']:
        for case,n,gen,ctx in [('short',0,64,512),('medium',24,96,1024),('long',110,128,2048)]:
            prompt=chat('A small river flows past the trees and into the lake.\n'*n+'Summarize the scene in three sentences.')
            base=run('stock',case+'-seed',prompt,gen,ctx)
            teacher=SCRATCH/(case+'-teacher.json');write(teacher,json.dumps(base['samples'][0]['tokens']))
            for rep in range(5):
                order=MODES[rep%4:]+MODES[:rep%4]
                for mode in order:run(mode,f'{case}-{rep}',prompt,gen,ctx,teacher=teacher)
    if args.part in ['accuracy','all']:
        for case,q in QUESTIONS.items():
            base=run('stock','quality-'+case,chat(q),gen=48,ctx=4096,trace=True)
            teacher=SCRATCH/(case+'-teacher.json');write(teacher,json.dumps(base['samples'][0]['tokens']))
            for mode in MODES[1:]:
                run(mode,'quality-'+case,chat(q),gen=48,ctx=4096,trace=True,teacher=teacher)
                run(mode,'free-'+case,chat(q),gen=48,ctx=4096)
if __name__=='__main__':main()

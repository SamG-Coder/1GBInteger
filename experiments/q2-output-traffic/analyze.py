"""Separate exact control, approximate decode error, and fixed-work timing."""
import hashlib,json,pathlib,statistics,sys
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
SCRATCH=ROOT/'.local-llm/q2-traffic-evaluation';RAW=HERE/'raw'
sys.path.insert(0,str(ROOT/'experiments/llm-inference'))
from analyze import score,response
MODES=['stock','output-q8','output-q5','output-q4','output-q2']
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')
def main():
    accuracy=[];quality=[];timing=[];traces={};profiles=[]
    for case in ['math','fact','instruction','code','conversation']:
        base=load(RAW/f'quality-{case}-stock.json');nv=base['vocab_size'];steps=base['forced_length']
        ref=np.fromfile(SCRATCH/f'quality-{case}-stock.f32',dtype='<f4').reshape(steps,nv)
        # Pinned GGUF tokenizer.ggml.eos_token_id = 151645 (<|im_end|>).
        bt=base['samples'][0]['tokens'];answer_end=bt.index(151645)+1 if 151645 in bt else steps
        nodes=load(SCRATCH/f'quality-{case}-stock-nodes.json')
        layers={(n['step'],n['name']):sha(pathlib.Path(n['file'])) for n in nodes if 'file' in n}
        for mode in MODES:
            tag=f'quality-{case}-{mode}';path=SCRATCH/f'{tag}.f32'
            other=np.fromfile(path,dtype='<f4').reshape(steps,nv);assert np.isfinite(other).all()
            traces[path.name]=sha(path);layer_count=0;layer_equal=True
            ns=load(SCRATCH/f'{tag}-nodes.json')
            for n in ns:
                if 'file' not in n:continue
                path=pathlib.Path(n['file']);h=sha(path);traces[path.name]=h
                layer_equal &= h==layers[(n['step'],n['name'])];layer_count+=1
            # Step zero uses original stock output; report it separately.
            e=other[1:].astype(np.float64)-ref[1:].astype(np.float64)
            kl=[];maxprob=0
            for x,y in zip(ref[1:],other[1:]):
                x=x.astype(np.float64);y=y.astype(np.float64)
                x-=x.max();y-=y.max();lp=x-np.log(np.exp(x).sum());lq=y-np.log(np.exp(y).sum())
                p=np.exp(lp);q=np.exp(lq);kl.append(float(np.sum(p*(lp-lq))));maxprob=max(maxprob,float(np.max(np.abs(p-q))))
            row=dict(case=case,mode=mode,decode_steps=steps-1,decode_logits_compared=int(e.size),prefill_logits_byte_equal=bool(np.array_equal(ref[0].view(np.uint32),other[0].view(np.uint32))),
                decode_logits_byte_equal=bool(np.array_equal(ref[1:].view(np.uint32),other[1:].view(np.uint32))),max_abs_error=float(np.abs(e).max()),rmse=float(np.sqrt(np.mean(e*e))),
                decode_top1_agreement=float(np.mean(np.argmax(ref[1:],axis=1)==np.argmax(other[1:],axis=1))),mean_kl_nats=statistics.mean(kl),max_probability_difference=maxprob,
                layers_checked=layer_count,layers_byte_equal=layer_equal)
            accuracy.append(row)
            row.update(answer_decode_steps=answer_end-1,answer_top1_matches=int(np.sum(np.argmax(ref[1:answer_end],axis=1)==np.argmax(other[1:answer_end],axis=1))),answer_mean_kl_nats=statistics.mean(kl[:answer_end-1]))
            assert row['prefill_logits_byte_equal'] and layer_equal
            if mode=='output-q8':assert row['decode_logits_byte_equal']
            free=base if mode=='stock' else load(RAW/f'free-{case}-{mode}.json')
            text=response(free);quality.append(dict(case=case,mode=mode,response=text,diagnostic_pass=score(case,text),response_matches_stock=text==response(base),tokens_match_stock=free['samples'][0]['tokens']==base['samples'][0]['tokens']))
            profiles.append(dict(case=case,mode=mode,output_decode_median_ms=statistics.median(n['ms'] for n in ns if n['name']=='result_output' and n['step']>0)))
    for case in ['short','medium','long']:
        for mode in MODES:
            ds=[load(RAW/f'{case}-{rep}-{mode}.json') for rep in range(5)]
            bases=[load(RAW/f'{case}-{rep}-stock.json') for rep in range(5)]
            row=dict(case=case,mode=mode,samples=5,prompt_tokens=len(ds[0]['prompt_tokens']),generated_tokens=ds[0]['forced_length'],threads=8)
            for d,b in zip(ds,bases):
                assert d['teacher_forced'] and d['teacher_tokens_sha256']==b['teacher_tokens_sha256'] and d['prompt_tokens']==b['prompt_tokens']
                if mode=='output-q8':assert d['samples'][0]['tokens']==b['samples'][0]['tokens']
            for field in ['pp_tps','generation_tps','first_token_ms','peak_working_set_bytes','cpu_machine_percent']:
                vs=[d['samples'][0][field] for d in ds];row[field]=statistics.median(vs);row[field+'_min']=min(vs);row[field+'_max']=max(vs)
            for field in ['load_ms','prepack_ms','extra_packed_bytes']:row[field]=statistics.median(d[field] for d in ds)
            ratios=[d['samples'][0]['generation_tps']/b['samples'][0]['generation_tps'] for d,b in zip(ds,bases)]
            row.update(paired_decode_speedup_median=statistics.median(ratios),paired_decode_speedup_min=min(ratios),paired_decode_speedup_max=max(ratios));timing.append(row)
    followup=[]
    for mode in ['stock','output-q2']:
        text=response(load(RAW/f'code-budget-128-{mode}.json'))
        followup.append(dict(case='code-budget-128',mode=mode,response=text,diagnostic_pass=score('code',text)))
    save(HERE/'summary.json',dict(timing=timing,accuracy=accuracy,quality=quality,code_budget_followup=followup,instrumented_profiles=profiles,trace_sha256=traces))
    save(HERE/'backend-evidence.json',{'model_buffers':{p.name:[l for l in p.read_text(encoding='utf-8').splitlines() if 'model buffer size' in l] for p in SCRATCH.glob('*.log')}})
    for mode in MODES:
        rows=[r for r in accuracy if r['mode']==mode]
        print(mode,'top1',statistics.mean(r['decode_top1_agreement'] for r in rows),'KL',statistics.mean(r['mean_kl_nats'] for r in rows),'quality',sum(r['diagnostic_pass'] for r in quality if r['mode']==mode))
if __name__=='__main__':main()

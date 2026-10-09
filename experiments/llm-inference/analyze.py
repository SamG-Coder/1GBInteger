"""Reduce real-model timings and numerical dumps; needs NumPy only for --dumps."""
import argparse, ast, collections, json, math, pathlib, statistics
from setup import ROOT,LOCAL,sha
HERE=pathlib.Path(__file__).resolve().parent
RAW=HERE/'raw'
MODES=['stock','control','packed','prepack','bitplane','binary','ternary']
CASES=['math','fact','instruction','code','conversation']
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')
def response(d):return d['samples'][0]['response'].split('<|im_end|>')[0].split('<|endoftext|>')[0].strip()
def score(case,s):
    if case=='math':return '391' in s
    if case=='fact':return 'paris' in s.lower()
    if case=='instruction':return s.strip()== 'red, green, blue'
    if case=='code':
        # Parse only; never execute arbitrary generated code. The requested
        # function is simple enough to check its complete arithmetic AST.
        code=s.removeprefix('```python').removeprefix('```').removesuffix('```').strip()
        try:tree=ast.parse(code)
        except SyntaxError:return False
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):return False
        f=tree.body[0]
        returns=[n for n in f.body if isinstance(n,ast.Return)]
        return f.name=='add' and [x.arg for x in f.args.args]==['a','b'] and len(returns)==1 and ast.dump(returns[0].value)==ast.dump(ast.parse('a+b',mode='eval').body)
    if case=='conversation':return s.strip()=='ORBIT-731'
    raise ValueError(case)
def difference(a,b):
    import numpy as np
    a=a.astype(np.float64);b=b.astype(np.float64);e=b-a
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return {'max_abs':float(np.max(np.abs(e))),'rmse':float(np.sqrt(np.mean(e*e))),'bitwise_equal':bool(np.array_equal(a,b))}
def dumps():
    import numpy as np
    scratch=LOCAL/'evaluation';results=[];profiles={};layers=[];trace_hashes={}
    for case in CASES:
        stock=load(RAW/f'quality-{case}-stock.json');nv=stock['vocab_size'];steps=stock['forced_length']
        refs={m:np.fromfile(scratch/f'quality-{case}-{m}.f32',dtype='<f4').reshape(steps,nv) for m in ['stock','control']}
        reference_nodes={m:load(scratch/f'quality-{case}-{m}-nodes.json') for m in ['stock','control']}
        reference_tensors={m:{(n['step'],n['name']):np.fromfile(n['file'],dtype='<f4') for n in ns if 'file' in n} for m,ns in reference_nodes.items()}
        for mode in MODES:
            prefix=f'quality-{case}-{mode}';data=load(RAW/f'{prefix}.json')
            path=scratch/f'{prefix}.f32';trace_hashes[path.name]=sha(path)
            logits=np.fromfile(path,dtype='<f4').reshape(steps,nv)
            for ref,a in refs.items():
                d=difference(a,logits);maxprob=0;kl=[];tops=[]
                for x,y in zip(a,logits):
                    x=x.astype(np.float64);y=y.astype(np.float64)
                    xp=x-np.max(x);yp=y-np.max(y)
                    lp=xp-np.log(np.exp(xp).sum());lq=yp-np.log(np.exp(yp).sum());p=np.exp(lp);q=np.exp(lq)
                    maxprob=max(maxprob,float(np.max(np.abs(p-q))));kl.append(float(np.sum(p*(lp-lq))));tops.append(int(np.argmax(x)==np.argmax(y)))
                d.update(case=case,mode=mode,reference=ref,top1_agreement=sum(tops)/len(tops),max_probability_difference=maxprob,mean_kl_nats=statistics.mean(kl),steps=steps)
                results.append(d)
            nodes=load(scratch/f'{prefix}-nodes.json');ops=collections.defaultdict(float);names=collections.defaultdict(float)
            for n in nodes:
                ops[('prompt:' if n['step']==0 else 'decode:')+n['op']]+=n['ms'];names[n['name']]+=n['ms']
                if 'input_file' in n:
                    path=pathlib.Path(n['input_file']);trace_hashes[path.name]=sha(path)
                if 'file' in n:
                    path=pathlib.Path(n['file']);trace_hashes[path.name]=sha(path);x=np.fromfile(path,dtype='<f4')
                    for ref,ts in reference_tensors.items():
                        key=(n['step'],n['name']);assert key in ts
                        layers.append(dict(case=case,mode=mode,reference=ref,step=n['step'],tensor=n['name'],**difference(ts[key],x)))
            profiles[prefix]={'operation_ms':dict(ops),'top_nodes_ms':sorted(names.items(),key=lambda x:-x[1])[:20]}
    save(HERE/'accuracy.json',{'logits':results,'layers':layers,'trace_sha256':trace_hashes})
    save(HERE/'profiles.json',profiles)
def main():
    p=argparse.ArgumentParser();p.add_argument('--dumps',action='store_true');a=p.parse_args()
    if a.dumps:dumps()
    timing=[];quality=[]
    for case in ['short','medium','long','extended','threads-1','threads-4','threads-8','threads-16']:
        for mode in MODES:
            path=RAW/f'{case}-{mode}.json'
            if not path.exists():continue
            d=load(path);assert not d['instrumented'];s=d['samples'];row={'case':case,'mode':mode,'prompt_tokens':len(d['prompt_tokens']),'generated_tokens':d['forced_length'],'context':d['context'],'threads':d['threads'],'samples':len(s),'load_ms':d['load_ms'],'prepack_ms':d['prepack_ms'],'extra_packed_bytes':d['extra_packed_bytes']}
            for field in ['pp_tps','generation_tps','first_token_ms','peak_working_set_bytes','cpu_machine_percent']:
                row[field]=statistics.median(x[field] for x in s)
                row[field+'_min']=min(x[field] for x in s);row[field+'_max']=max(x[field] for x in s)
            timing.append(row)
    for case in CASES:
        for mode in MODES:
            path=RAW/f'free-{case}-{mode}.json'
            if not path.exists():path=RAW/f'quality-{case}-{mode}.json'
            if not path.exists():continue
            # Teacher-forced responses are not a free-running quality score.
            if path.name.startswith('quality') and mode!='stock':continue
            d=load(path);s=response(d);quality.append({'case':case,'mode':mode,'response':s,'diagnostic_pass':score(case,s)})
    save(HERE/'summary.json',{'timing':timing,'quality':quality})
    lines=['# Repeated CPU measurements','','Medians of three warm repetitions; min/max remain in summary.json.','', '| Workload | Mode | Prompt tokens | PP tok/s | Decode tok/s | TTFT ms | Peak MiB | CPU % of machine |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in timing:lines.append(f"| {r['case']} | {r['mode']} | {r['prompt_tokens']} | {r['pp_tps']:.1f} | {r['generation_tps']:.1f} | {r['first_token_ms']:.1f} | {r['peak_working_set_bytes']/2**20:.1f} | {r['cpu_machine_percent']:.1f} |")
    (HERE/'MEASUREMENTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('Reduced',len(timing),'timing configurations and',len(quality),'free-running responses')
if __name__=='__main__':main()

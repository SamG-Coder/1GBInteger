"""Byte-exact stock comparisons plus paired end-to-end measurements."""
import collections,hashlib,json,pathlib,statistics,sys
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1];SCRATCH=ROOT/'.local-llm/stock-evaluation';RAW=HERE/'raw'
sys.path.insert(0,str(ROOT/'experiments/llm-inference'))
from analyze import score,response
MODES=['stock','stock-plus-single','stock-plus']
def load(p):return json.loads(p.read_text())
def save(p,data):p.write_text(json.dumps(data,indent=2)+'\n',newline='\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    accuracy=[];quality=[];timing=[];traces={}
    for case in ['math','fact','instruction','code','conversation']:
        base=load(RAW/f'quality-{case}-stock.json');nv=base['vocab_size'];steps=base['forced_length']
        prefix=f'quality-{case}-stock';ref=np.fromfile(SCRATCH/f'{prefix}.f32',dtype='<f4').reshape(steps,nv)
        nodes=load(SCRATCH/f'{prefix}-nodes.json');layers={(n['step'],n['name']):pathlib.Path(n['file']) for n in nodes if 'file' in n}
        for mode in MODES:
            tag=f'quality-{case}-{mode}';d=load(RAW/f'{tag}.json');path=SCRATCH/f'{tag}.f32';other=np.fromfile(path,dtype='<f4').reshape(steps,nv)
            assert np.isfinite(other).all();traces[path.name]=sha(path)
            error=np.abs(other.astype(np.float64)-ref.astype(np.float64))
            same=bool(np.array_equal(ref.view(np.uint32),other.view(np.uint32)))
            layer_count=0;layer_equal=True
            for n in load(SCRATCH/f'{tag}-nodes.json'):
                if 'file' not in n:continue
                path=pathlib.Path(n['file']);h=sha(path);traces[path.name]=h
                layer_equal &= h==sha(layers[(n['step'],n['name'])]);layer_count+=1
            accuracy.append({'case':case,'mode':mode,'logits_byte_equal':same,'max_abs_error':float(error.max()),'top1_agreement':float(np.mean(np.argmax(ref,axis=1)==np.argmax(other,axis=1))),'layers_checked':layer_count,'layers_byte_equal':layer_equal,'probabilities_equal':same})
            free=base if mode=='stock' else load(RAW/f'free-{case}-{mode}.json')
            text=response(free);quality.append({'case':case,'mode':mode,'response':text,'diagnostic_pass':score(case,text),'tokens_match_stock':free['samples'][0]['tokens']==base['samples'][0]['tokens']})
    for case in ['short','medium','long','extended','threads-1','threads-4','threads-16']:
        for mode in MODES:
            paths=sorted(RAW.glob(f'{case}-*-{mode}.json'))
            if not paths:continue
            ds=[load(p) for p in paths];ss=[s for d in ds for s in d['samples']]
            row={'case':case,'mode':mode,'samples':len(ss),'prompt_tokens':len(ds[0]['prompt_tokens']),'generated_tokens':ds[0]['forced_length'],'context':ds[0]['context'],'threads':ds[0]['threads']}
            for field in ['pp_tps','generation_tps','first_token_ms','peak_working_set_bytes','cpu_machine_percent']:
                vals=[s[field] for s in ss];row[field]=statistics.median(vals);row[field+'_min']=min(vals);row[field+'_max']=max(vals)
            ratios=[]
            for path,d in zip(paths,ds):
                basepath=path.with_name(path.name[:-len(mode+'.json')]+'stock.json');bd=load(basepath)
                assert d['prompt_tokens']==bd['prompt_tokens']
                ratios.append(d['samples'][0]['generation_tps']/bd['samples'][0]['generation_tps'])
            row['paired_decode_speedup_median']=statistics.median(ratios);row['paired_decode_speedup_min']=min(ratios);row['paired_decode_speedup_max']=max(ratios);timing.append(row)
    result={'timing':timing,'accuracy':accuracy,'quality':quality,'trace_sha256':traces}
    save(HERE/'summary.json',result)
    assert all(r['logits_byte_equal'] and r['layers_byte_equal'] for r in accuracy), 'Stock numerical conformance failed'
    assert all(r['tokens_match_stock'] for r in quality),'Free-running response drift'
    print('Stock byte-exact logits/layers and free-running token sequences:',len(accuracy),'cases; timing configurations:',len(timing))
if __name__=='__main__':main()

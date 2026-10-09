"""Verify saved evidence without downloading model weights or rerunning timings."""
import json, math, pathlib
from setup import sha
HERE=pathlib.Path(__file__).resolve().parent
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def check():
    manifest=load(HERE/'manifest.json')
    for name,h in manifest['source_sha256'].items():assert sha(HERE/name)==h, f'source changed: {name}'
    for name,h in manifest['raw_sha256'].items():assert sha(HERE/'raw'/name)==h, f'raw evidence changed: {name}'
    for name,h in manifest.get('evidence_sha256',{}).items():assert sha(HERE/name)==h, f'derived evidence changed: {name}'
    modes=['stock','control','packed','prepack','bitplane','binary','ternary']
    samples=0
    for case in ['short','medium','long','extended']:
        reference=None
        for mode in modes:
            d=load(HERE/'raw'/f'{case}-{mode}.json');assert not d['instrumented'] and not d['teacher_forced']
            assert len(d['samples'])==3 and d['mode']==mode and d['threads']==8
            assert len(d['prompt_tokens'])+d['forced_length']<=d['context']
            if reference is None:reference=d['prompt_tokens']
            assert d['prompt_tokens']==reference
            for s in d['samples']:
                for k in ['pp_tps','generation_tps','first_token_ms','peak_working_set_bytes','elapsed_ms']:
                    assert math.isfinite(s[k]) and s[k]>0, (case,mode,k)
                assert len(s['tokens'])==d['forced_length']
                assert s['hook_calls']==0 if mode in ('stock','control') else s['hook_calls']>0
                samples+=1
    for case in ['math','fact','instruction','code','conversation']:
        base=load(HERE/'raw'/f'quality-{case}-stock.json')
        for mode in modes:
            d=load(HERE/'raw'/f'quality-{case}-{mode}.json')
            assert d['instrumented'] and d['prompt_tokens']==base['prompt_tokens']
            assert d['teacher_forced']==(mode!='stock')
    accuracy=load(HERE/'accuracy.json')
    assert len(accuracy['logits'])==70 and len(accuracy['layers'])>5000
    for r in accuracy['logits']:
        assert math.isfinite(r['rmse']) and 0<=r['top1_agreement']<=1
        if r['reference']=='control' and r['mode'] in ('packed','prepack'):
            assert r['max_abs']<0.001, r
    kernels=load(HERE/'raw/kernel-benchmark.json')
    assert len(kernels['runs'])==49
    for r in kernels['runs']:
        assert r['batch_ms']>0
        if r['mode'] in ('packed','expanded','packed-call'):assert r['max_abs_error']<0.001
    print(f'Checked pinned source/raw hashes, {samples} matched timing samples, 70 logit comparisons and layer evidence')
if __name__=='__main__':check()

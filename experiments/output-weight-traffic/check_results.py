"""Offline consistency checks; no model download or inference claims in CI."""
import hashlib,json,math,pathlib,statistics
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=load(HERE/'manifest.json')
for group in ['source_sha256','evidence_sha256']:
    for name,h in m[group].items():assert sha(HERE/name)==h,name
for name,h in m['shared_source_sha256'].items():assert sha(ROOT/name)==h,name
d=load(HERE/'summary.json');dist=load(HERE/'distribution.json')
assert dist['model_sha256']==m['model_sha256']
assert dist['shape']==[896,151936] and dist['q8_bytes']==144643072
assert sum(dist['histogram_unsigned_code'])==896*151936
assert len(d['accuracy'])==20 and len(d['quality'])==20 and len(d['timing'])==12
assert all(r['prefill_logits_byte_equal'] and r['layers_byte_equal'] for r in d['accuracy'])
assert all(r['decode_logits_byte_equal'] and r['decode_top1_agreement']==1 for r in d['accuracy'] if r['mode'] in ['stock','output-q8'])
assert all(r['tokens_match_stock'] for r in d['quality'] if r['mode']=='output-q8')
assert all(0<=r['answer_top1_matches']<=r['answer_decode_steps'] for r in d['accuracy'])
san=load(HERE/'sanitizer.json')
assert {r['mode'] for r in san['runs']}=={'stock','output-q8','output-q5','output-q4'}
assert all(r['exit_code']==0 for r in san['runs'])
sizes={'stock':0,**dist['projection_bytes']};count=0
for p in (HERE/'raw').glob('*.json'):
    r=load(p);assert r['executable_sha256']==m['executable_sha256'] and r['stock_repacking_enabled']
    assert r['extra_packed_bytes']==sizes[r['mode']]
    assert len(r['samples'])==1
    s=r['samples'][0];assert s['prefill_hook_calls']==0
    assert s['hook_calls']==(0 if r['mode']=='stock' else r['forced_length']-1)
    assert len(s['tokens'])==r['forced_length']
    assert all(math.isfinite(s[k]) and s[k]>0 for k in ['generation_tps','pp_tps','first_token_ms'])
    count+=1
assert count==102,count # 4 smoke + 3 seeds + 60 timings + 35 accuracy/free runs
for row in d['timing']:
    ds=[load(HERE/'raw'/f"{row['case']}-{rep}-{row['mode']}.json") for rep in range(5)]
    bs=[load(HERE/'raw'/f"{row['case']}-{rep}-stock.json") for rep in range(5)]
    for r,b in zip(ds,bs):assert r['teacher_forced'] and r['teacher_tokens_sha256']==b['teacher_tokens_sha256'] and r['prompt_tokens']==b['prompt_tokens']
    assert row['generation_tps']==statistics.median(r['samples'][0]['generation_tps'] for r in ds)
    assert row['paired_decode_speedup_median']==statistics.median(r['samples'][0]['generation_tps']/b['samples'][0]['generation_tps'] for r,b in zip(ds,bs))
buffers=load(HERE/'backend-evidence.json')['model_buffers'];assert len(buffers)==count
assert all(any('CPU_REPACK' in l for l in lines) for lines in buffers.values())
print('Validated',count,'runs, 20 logit comparisons, 12 matched timing configurations, unchanged prefill/layers, stock repacking and provenance')

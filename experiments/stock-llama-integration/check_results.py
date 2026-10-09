"""Validate committed experiment without downloading the model."""
import hashlib,json,pathlib,math
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=load(HERE/'manifest.json')
for k in ['source_sha256','evidence_sha256']:
    for name,h in m[k].items():assert sha(HERE/name)==h,name
for name,h in m['shared_source_sha256'].items():assert sha(ROOT/name)==h,name
d=load(HERE/'summary.json')
assert len(d['timing'])==18 and len(d['accuracy'])==15
assert all(r['logits_byte_equal'] and r['layers_byte_equal'] and r['top1_agreement']==1 for r in d['accuracy'])
assert all(r['tokens_match_stock'] for r in d['quality'])
count=0
for p in (HERE/'raw').glob('*.json'):
    r=load(p)
    if 'samples' not in r:continue
    assert r['executable_sha256']==m['executable_sha256'] and r['stock_repacking_enabled'] and r['extra_packed_bytes']==0
    for s in r['samples']:
        assert s['prefill_hook_calls']==0
        assert s['hook_calls']==(0 if r['mode']=='stock' else r['forced_length']-1)
        assert len(s['tokens'])==r['forced_length']
        assert all(math.isfinite(s[k]) and s[k]>0 for k in ['pp_tps','generation_tps','first_token_ms'])
        count+=1
buffers=load(HERE/'backend-evidence.json')['model_buffers']
assert len(buffers)>75
for name,lines in buffers.items():assert any('CPU_REPACK' in l for l in lines),name
print('Validated',count,'runs, 15 stock numerical comparisons, dispatch counts, repacking and provenance')

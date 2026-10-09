"""Measure actual Q8 codes; optimistic lossless exception-packing estimates."""
import hashlib,json,pathlib,sys
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'.local-llm/llama.cpp/gguf-py'))
from gguf import GGUFReader
model=ROOT/'.local-llm/Qwen2.5-0.5B-Instruct-Q4_0.gguf'
t=next(t for t in GGUFReader(model).tensors if t.name=='token_embd.weight')
blocks=t.data.reshape(-1,34);hist=np.zeros(256,np.int64)
for chunk in np.array_split(blocks,128):hist+=np.bincount(chunk[:,2:].reshape(-1),minlength=256)
estimates=[]
for bits in [4,5,6,7]:
    exceptions=sum(int(hist[x%256]) for x in range(-128,128) if not -(1<<(bits-1))<=x<(1<<(bits-1)))
    size=len(blocks)*(2+4+bits*4)+(8-bits)*exceptions/8
    estimates.append(dict(low_bits=bits,exception_fraction=exceptions/int(hist.sum()),ideal_bytes=size,ratio=size/blocks.size))
p=hist[hist>0]/hist.sum()
result=dict(model_sha256=hashlib.file_digest(model.open('rb'),'sha256').hexdigest(),shape=t.shape.tolist(),q8_bytes=blocks.size,
    code_entropy_bits=float(-sum(p*np.log2(p))),histogram_unsigned_code=hist.tolist(),lossless_estimates=estimates,
    estimate_definition='Per 32 weights: original 2-byte scale, low bits, 4-byte exception mask, remaining high bits for exceptions. Excludes padding, row offsets and headers.',
    projection_bytes={name:int(len(blocks)*size) for name,size in [('output-q8',34),('output-q5',22),('output-q4',18)]})
result['projection_bytes']['output-q2']=151936*4*84
result['q2_layout']={'original_width':896,'padded_width':1024,'block_elements':256,'block_bytes':84,'blocks_per_row':4,'activation_format':'Q8_K'}
(pathlib.Path(__file__).parent/'distribution.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({k:v for k,v in result.items() if k!='histogram_unsigned_code'},indent=2))

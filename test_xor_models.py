"""Exhaustively verify exported learned mappings and exercise native inference."""
import hashlib
import itertools
import json
import os
import pathlib
import subprocess

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'experiments'/'xor-inference'
EXE=ROOT/('xor_predict.exe' if os.name=='nt' else 'xor_predict')


def target(x,task):
    if task==0:return (sum((x>>i)&1 for i in (0,2,5,7,11,15))+1)%2
    if task==2:return int(sum((x>>i)&1 for i in (0,1,2))>=2)
    if task==4:return (((x%8)+((x//8)%8))//4)%2
    raise AssertionError('Unexpected exported task')


for entry in json.loads((OUT/'models'/'manifest.json').read_text()):
    path=OUT/entry['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
    words=path.read_text().split();magic,bank,size=words[:3];weights=[int(w,16) for w in words[3:]]
    assert magic=='XOR_MODEL_V1';degree={'linear':1,'quadratic':2,'cubic':3}[bank]
    masks=[0]
    for d in range(1,degree+1):masks.extend(sum(1<<i for i in indices) for indices in itertools.combinations(range(16),d))
    assert int(size)==len(masks)==entry['feature_count']
    selected=[m for i,m in enumerate(masks) if (weights[i//64]>>(i%64))&1]
    # Independent sparse polynomial evaluation of every possible input.
    for x in range(65536):
        assert sum((x&m)==m for m in selected)%2==target(x,entry['task_id']),(entry['file'],x)
    inputs=list(range(64))+[1<<i for i in range(16)]+[65535-i for i in range(64)]
    output=subprocess.check_output([str(EXE),str(path),*map(str,inputs)],text=True)
    expected=[f'{x},{target(x,entry["task_id"])}' for x in inputs]
    assert output.splitlines()==expected
    for value in ('-1','65536','nonsense'):
        assert subprocess.run([str(EXE),str(path),value],capture_output=True).returncode!=0
print('PASS: three learned models over all 65,536 inputs each, native model loading/prediction, invalid-input rejection')

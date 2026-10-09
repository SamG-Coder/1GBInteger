"""Save bounded disassembly evidence; entire objdump remains ignored."""
import collections, json, pathlib, re, subprocess
from setup import LOCAL,sha
HERE=pathlib.Path(__file__).resolve().parent
exe=LOCAL/'experiment-build/integer-llm.exe'
assembly=subprocess.check_output(['llvm-objdump','-d','--demangle',str(exe)],text=True)
(LOCAL/'assembly.txt').write_text(assembly,newline='\n')
names=['experiment::packed_dot(','experiment::packed_dot_call(','experiment::simd_dot(experiment::Expanded','experiment::bit_dot(','ggml_vec_dot_q4_0_q8_0']
blocks={};current=None
for line in assembly.splitlines():
    if re.match(r'^[0-9a-f]+ <.*>:$',line):
        current=next((n for n in names if n in line),None)
        if current:blocks[current]=[line]
    elif current:blocks[current].append(line)
report={'executable_sha256':sha(exe),'functions':{}}
for name,lines in blocks.items():
    instructions=collections.Counter()
    for l in lines:
        fields=l.split('\t')
        if len(fields)>1:instructions[fields[1].strip()]+=1
    report['functions'][name]={'instructions':dict(instructions),'calls':[l.strip() for l in lines if '\tcall' in l]}
assert len(blocks)==len(names)
assert not report['functions']['experiment::packed_dot(']['calls']
assert any('ggml_fp16_to_fp32' in l for l in report['functions']['experiment::packed_dot_call(']['calls'])
(HERE/'assembly.json').write_text(json.dumps(report,indent=2)+'\n',newline='\n')
excerpt=[]
for name in names[:2]:excerpt.extend(blocks[name]);excerpt.append('')
(HERE/'assembly-excerpts.txt').write_text('\n'.join(excerpt),newline='\n')
print('Verified inline packed loop has no calls; retained comparison has fp16 conversion calls')

"""Independent integer oracle, counters, graph traversal and perturbation checks."""
import collections
import json
import os
import pathlib
import subprocess
import tempfile

ROOT=pathlib.Path(__file__).resolve().parent
EXT='.exe' if os.name=='nt' else ''
MASK=(1<<64)-1
C=0x9e3779b97f4a7c15


def mix(x):
    x&=MASK;x=((x^(x>>30))*0xbf58476d1ce4e5b9)&MASK
    x=((x^(x>>27))*0x94d049bb133111eb)&MASK
    return x^(x>>31)


def rot(x,n,width):
    n%=width
    return ((x<<n)|(x>>((width-n)%width)))&((1<<width)-1)


def step(name, state, mask, width=64):
    word=(1<<width)-1
    if name=='local':
        return [rot((x+rot(state[i-1],17,width)+mask[i])&word,23,width)^state[(i+1)%len(state)]
                for i,x in enumerate(state)]
    size=len(state)*width
    big=sum(x<<(width*i) for i,x in enumerate(state))
    m=sum(x<<(width*i) for i,x in enumerate(mask))
    result=(rot(big,1,size)^m) if name=='xor' else (rot(big,1,size)+m)&((1<<size)-1)
    return [(result>>(width*i))&word for i in range(len(state))]


def initial(seed,limbs):
    return ([mix(seed+(i+1)*C) for i in range(limbs)],
            [mix(seed^((i*C)&MASK)) for i in range(limbs)])


def check_counts(d,traces):
    atoms,n=len(traces),len(traces[0])
    assert d['ones']==sum(map(sum,traces))
    assert d['atom_ones']==list(map(sum,traces))
    for k,lag in enumerate(d['lags']):
        assert d['lag_n'][k]==atoms*(n-lag)
        assert d['lag_equal'][k]==sum(b[t]==b[t-lag] for b in traces for t in range(lag,n))
    counts={k:[0]*len(d[k]) for k in ('blocks','runs','transitions','first','gaps')}
    tails=0
    for b in traces:
        runs=[]
        for t,x in enumerate(b):
            if not t or x!=b[t-1]:runs.append(1)
            else:runs[-1]+=1
        for r in runs[1:-1]:counts['runs'][min(128,r)]+=1
        for x,y in zip(b,b[1:]):counts['transitions'][2*x+y]+=1
        events=[]
        for i,t in enumerate(range(0,n,8),1):
            x=sum(b[t+k]<<k for k in range(8));counts['blocks'][x]+=1
            if x==255:events.append(i)
        counts['first'][events[0] if events else 0]+=1
        for x,y in zip(events,events[1:]):counts['gaps'][y-x]+=1
        tails+=not events or events[-1]<n//8
    for k,v in counts.items():assert d[k]==v,k
    assert d['run_left_censored']==d['run_right_censored']==atoms
    assert d['tail_censored']==tails
    assert d['cross_n']==atoms//2*n
    assert d['cross_equal']==sum(traces[j][t]==traces[j+1][t] for j in range(0,atoms,2) for t in range(n))
    assert d['predictions']==atoms//2*(n-65)
    assert d['predicted']==sum(traces[j+1][t+1]==(traces[j+1][t]^traces[j][t-63]^traces[j][t-64])
                               for j in range(0,atoms,2) for t in range(64,n-1))


def graph_reference(name,seed,w,limbs):
    size=1<<(w*limbs);word=(1<<w)-1
    mask=[mix(seed^((i*C)&MASK))&word for i in range(limbs)]
    edges=[]
    for x in range(size):
        state=[(x>>(i*w))&word for i in range(limbs)]
        edges.append(sum(v<<(i*w) for i,v in enumerate(step(name,state,mask,w))))
    pre=collections.Counter(edges);lengths=[];depth={}
    for start in range(size):
        if start in depth:continue
        path=[];positions={};v=start
        while v not in depth and v not in positions:
            positions[v]=len(path);path.append(v);v=edges[v]
        if v in positions:
            offset=positions[v];lengths.append(len(path)-offset)
            for node in path[offset:]:depth[node]=0
            path=path[:offset]
        for node in reversed(path):depth[node]=depth[edges[node]]+1
    return dict(states=size,distinct_images=len(pre),collision_excess=size-len(pre),maximum_preimages=max(pre.values()),
                cycles=len(lengths),cyclic_states=sum(lengths),min_cycle=min(lengths),max_cycle=max(lengths),
                max_transient=max(depth.values()),edges=edges)


def main():
    edge_output=subprocess.check_output([str(ROOT/('test_rule_edges'+EXT))],text=True)
    for line in edge_output.splitlines():
        name,*numbers=line.split(',');a,b,*actual=map(int,numbers)
        assert actual==step(name,[a,b,a],[b,a,b])
    with tempfile.TemporaryDirectory() as td:
        path=pathlib.Path(td)/'result.json'
        def run(name,mode,seed,**options):
            args=['--mode',mode,'--rule',name,'--seed',str(seed),'--out',str(path)]
            for k,v in options.items():args.extend(['--'+k,str(v)])
            subprocess.run([str(ROOT/('rule_compare'+EXT)),*args],check=True,capture_output=True)
            return json.loads(path.read_text())
        for name in ('xor','add','local'):
            for seed in (0,12345,MASK):
                for limbs in (1,2,3,17):
                    d=run(name,'trace',seed,bytes=limbs*8,steps=127)
                    state,mask=initial(seed,limbs)
                    for actual in d['states']:
                        assert state==actual,(name,seed,limbs)
                        state=step(name,state,mask)
                d=run(name,'stats',seed,bytes=17*8,steps=511,atoms=8)
                state,mask=initial(seed,17)
                traces={layout:[[] for _ in range(8)] for layout in ('spaced','adjacent')}
                for _ in range(512):
                    for j in range(8):
                        traces['spaced'][j].append(state[j*17//8]&1)
                        traces['adjacent'][j].append(state[j]&1)
                    state=step(name,state,mask)
                for layout in d['layouts']:check_counts(layout,traces[layout['layout']])
            d=run(name,'graph',12345)
            for case in d['graphs']:
                ref=graph_reference(name,12345,case['word_bits'],case['limbs'])
                for k,v in ref.items():
                    if k!='edges' or k in case:assert case[k]==v,(name,k)
            d=run(name,'damage',12345,steps=128)
            for case in d['damage']:
                a,m=initial(12345,case['limbs']);b=a.copy();b[0]^=1;start=a.copy();first=0
                for t in range(129):
                    assert case['hamming'][t]==sum((x^y).bit_count() for x,y in zip(a,b))
                    assert case['changed_limbs'][t]==sum(x!=y for x,y in zip(a,b))
                    if t and a==start and not first:first=t
                    a=step(name,a,m);b=step(name,b,m)
                assert case['first_return']==first
        assert run('xor','bench',12345,bytes=1000000,steps=3)['checksum']==run('xor','bench',12345,bytes=1000000,steps=3)['checksum']
    print('PASS: carry/overflow edge cases, 36 full traces, every diagnostic counter, all 15 graph oracles and damage trajectories')


if __name__=='__main__':main()

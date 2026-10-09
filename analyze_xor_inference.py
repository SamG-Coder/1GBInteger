"""Verify native XOR inference evidence, publish studies, and export learned models."""
import argparse
import csv
import hashlib
import json
import pathlib
import statistics

ROOT=pathlib.Path(__file__).resolve().parent
OUT=ROOT/'experiments'/'xor-inference'
TASKS=['parity','AND','majority of 3','selector','addition bit 2','addition carry bit 3']
MODELS=['linear','reservoir','quadratic','cubic']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def load(file):
    with (OUT/'raw'/file).open(newline='') as f:rows=list(csv.DictReader(f))
    for r in rows:
        assert None not in r,'Malformed CSV row'
        for k,v in list(r.items()):
            if k in ('group','model','weights_hex'):continue
            r[k]=None if v=='' else float(v) if k.endswith('seconds') else int(v)
    return rows


def range_value(rs,key,scale=1,digits=3):
    v=[r[key]*scale for r in rs if r[key] is not None]
    return '—' if not v else f'{min(v):.{digits}f}–{max(v):.{digits}f}'


def main(plots=False):
    manifest=json.loads((OUT/'manifest.json').read_text())
    for path,digest in {**manifest['source_sha256'],**manifest['outputs']}.items():assert sha(ROOT/path)==digest,path
    state,learn,dot=load('state.csv'),load('learning.csv'),load('binary-dot.csv')
    assert (len(state),len(learn),len(dot))==(280,608,24)
    for r in state:
        assert r['rank']<=r['bits']*(2 if r['unknown_mask'] else 1)
        assert r['equations']==r['sensors']*r['horizon']
        for kind in ('initial','mask','full','sensor'):
            assert 0<=r[kind+'_correct']<=r[kind+'_answered']<=r[kind+'_total']
            if r['conflicts']:assert r[kind+'_answered']==0
            if r['noise_bp']==0:assert r[kind+'_correct']==r[kind+'_answered']
    for r in learn:
        if r['conflicts']:assert r['test_correct'] is None and r['weights_hex']==''
        else:
            assert r['train_correct']==r['train_count']
            assert 0<=r['certified_correct']<=r['certified_answered']<=r['test_count']
            assert len(r['weights_hex'].split(';'))==(r['features']+63)//64
            r['test_accuracy']=r['test_correct']/r['test_count']
    for seed in {r['seed'] for r in dot}:
        rs=[r for r in dot if r['seed']==seed];assert len(rs)==3 and len({r['checksum'] for r in rs})==1
    report_state(state)
    report_learning(learn)
    report_cost(learn,dot)
    export_models(learn)
    index()
    (OUT/'summary.json').write_text(json.dumps({'state':state,'learning':learn,'binary_dot':dot},separators=(',',':'))+'\n')
    (OUT/'analysis-provenance.json').write_text(json.dumps({'manifest_sha256':sha(OUT/'manifest.json'),
        'sources':{f:sha(ROOT/f) for f in ['analyze_xor_inference.py','xor_predict.cpp']}},indent=2)+'\n')
    if plots:plot(state,learn,dot)
    print('PASS: verified 280 state cases, 608 learning fits and 24 dot-product timings; exported three learned models.')


def report_state(rows):
    s='''# Case study 8 — Exact hidden-state inference, observability and noise

The solver receives a known transition rule, observation positions/times, and
observed bit values. It does not receive the simulator seed or hidden initial
state. In known-mask cases it additionally receives M. Joint-inference cases
treat all N initial bits and all N mask bits as independent unknowns.

The exact closed form makes each observation a linear equation over GF(2).
Packed Gaussian elimination computes its rank and tests consistency. A queried
bit is returned only if its value is invariant over **every** solution of the
equations; otherwise the solver abstains. Rank alone is not a bit-recovery count.

## Clean observations

For N=1024, the following values are identical across all eight fixed seeds.
Every answered bit agrees with the independently simulated truth. All N=64 and
N=256 results are also recorded in [state.csv](raw/state.csv).

| Mask | Sensors | Consecutive observation times | Rank | Initial bits recovered / 1024 | Mask bits recovered / 1024 | Full forecast bits determined at first future time / 1024 | Observed-site future bits determined |
|---|---:|---:|---:|---:|---:|---:|---:|
'''
    for r in rows:
        if r['seed']!=20261009 or r['bits']!=1024 or r['noise_bp']:continue
        mask='unknown' if r['unknown_mask'] else 'known'
        recovered_mask=str(r['mask_answered']) if r['unknown_mask'] else 'given'
        s+=f'| {mask} | {r["sensors"]} | {r["horizon"]} | {r["rank"]} | {r["initial_answered"]} | {recovered_mask} | {r["full_answered"]} | {r["sensor_answered"]}/{r["sensor_total"]} |\n'
    s+='''
**Finding:** With known M, one bit observed for N consecutive times recovers
all N initial bits. Four appropriately spaced sensors cover the state in N/4
times; 16 do so in N/16. This is an exact inverse calculation, not a learned
statistical guess. Half a sweep leaves half the initial state unidentified.

With unknown M, one sensor over 2N times yields rank N+1 among 2N unknowns.
Only the directly observed initial bit is uniquely recovered, and no individual
mask bit is uniquely recovered. Nevertheless the next 32 values at that sensor
are all determined and correct. Forecastability is not full identifiability.
Observing every bit at two consecutive times recovers both S(0) and M exactly.

A structural ambiguity explains this: choose an arbitrary fixed vector q that
is zero at every observed site. Replace S(0) by S(0) XOR q and M by
M XOR q XOR R(q). The entire new trajectory is S(t) XOR q, so observations at
those sites are unchanged forever. Unobserved state bits cannot be uniquely
recovered from such observations without additional assumptions.

These are 64–1024-bit inverse problems. A dense joint solver for a 1 GB state
would be a very different memory/computation problem. The seed constraint of
the original generator is deliberately not used as a shortcut.

## Measurement noise

Known mask, N=256, one sensor. The table pools counts across all eight seeds.
Repeats=5 uses majority voting on independent corrupted readings of the same
measurement. Raw flips, voted errors, and equation conflicts are preserved.

| Times | Flip probability | Readings per measurement | Consistent systems / 8 | Voted bit errors | Wrong recovered initial bits across accepted systems |
|---:|---:|---:|---:|---:|---:|
'''
    for h in (256,1024):
        for noise in (100,500):
            for repeats in (1,5):
                rs=[r for r in rows if r['horizon']==h and r['noise_bp']==noise and r['repeats']==repeats]
                s+=f'| {h} | {noise/100:.0f}% | {repeats} | {sum(r["conflicts"]==0 for r in rs)} | {sum(r["voted_errors"] for r in rs)} | {sum(r["initial_answered"]-r["initial_correct"] for r in rs)} |\n'
    s+='''
At exactly N observations, there is enough freedom to fit corrupted measurements
without contradiction. A unique algebraic solution can still be **wrong**.
Longer observation windows expose contradictions through redundant constraints;
the solver then refuses exact inference. Five repeated readings help in the
tested independent-noise setting, but residual errors remain at 5% noise.

This is not a general noisy-parity decoder. Majority voting spends five times
the measurement budget and assumes independent errors. The experiment does not
address systematic bias, correlated noise, or error correction without repeats.
The counts are finite seeded experiments, not a guaranteed error probability.
'''
    (OUT/'08-state-inference.md').write_text(s,encoding='utf-8')


def report_learning(rows):
    s='''# Case study 9 — Learning XOR readouts and testing unseen inputs

Each model is fitted from input/label pairs by GF(2) elimination. Training sets
contain 64, 256 or 1024 unique 16-bit inputs. The 8192 held-out inputs are
disjoint from every training prefix. Labels come from fixed synthetic tasks;
the predictor itself never calls the label oracle. Four development and four
validation seeds determine independent input permutations, with no model
selection or tuning on validation accuracy.

Models:

- **linear:** constant plus 16 input bits; 17 XOR coefficients.
- **reservoir:** constant plus 64 parity projections, followed by 17 steps of
  rotation/XOR evolution with a fixed mask; 65 coefficients but affine rank 17.
- **quadratic:** all 137 constant, input and pairwise-AND features.
- **cubic:** all 697 monomials through triple-AND features.

Quadratic/cubic models are **nonlinear features plus an XOR readout**, not
XOR-only networks. The target equations and monomial order are in the protocol
and native source. Coefficients are learned, not supplied from the task formula.

## 1024-example fits: fixed validation seeds

Each cell reports consistent fits out of four and held-out accuracy range when
a consistent fit exists. An inconsistent fit abstains; it is not scored as a
valid learned classifier.

| Task | Linear | XOR reservoir | Quadratic | Cubic |
|---|---|---|---|---|
'''
    for task,name in enumerate(TASKS):
        cells=[]
        for model in MODELS:
            rs=[r for r in rows if r['group']=='validation' and r['task']==task and r['model']==model and r['train_count']==1024 and not r['noise_bp']]
            valid=[r for r in rs if not r['conflicts']]
            cells.append(f'{len(valid)}/4; '+(range_value(valid,'test_accuracy',100,2)+'%' if valid else 'no exact fit'))
        s+='| '+name+' | '+' | '.join(cells)+' |\n'
    s+='''
The linear learner recovers parity perfectly. Expanding it into the affine
reservoir does not increase its rank or allow it to learn any additional task.
Composing XORs, rotations and constants cannot create nonlinear functions of
the input. A large deterministic state is not, by itself, extra model capacity.

AND features change the function class: quadratic features fit AND, three-bit
majority and the selector, while cubic features also fit bit 2 of addition.
The final carry bit requires degree four in this formulation and is a deliberate
negative control: none of these model classes fits its 1024 clean examples.

## Interpolation is not generalization

For each consistent underdetermined system, the evaluated model sets free
coefficients to zero in the fixed elimination order. It is one arbitrary
interpolant, not a regularized or sparsity-optimized solution. Independently, we
certify which held-out predictions are identical across all fitting models.

| Model/task | Training examples | Exact training fits / 4 | Held-out accuracy range | Certified held-out predictions / 8192 |
|---|---:|---:|---:|---:|
'''
    for model,task in [('linear',0),('quadratic',2),('cubic',4),('cubic',5)]:
        for n in (64,256,1024):
            rs=[r for r in rows if r['group']=='validation' and r['model']==model and r['task']==task and r['train_count']==n and not r['noise_bp']]
            valid=[r for r in rs if not r['conflicts']]
            s+=f'| {model} / {TASKS[task]} | {n} | {len(valid)} | {range_value(valid,"test_accuracy",100,2)} | {range_value(valid,"certified_answered",1,0)} |\n'
    s+='''
Large feature banks can fit 64 or 256 examples perfectly while performing near
chance on new inputs and certifying none of those predictions. At 1024 examples,
the tested representable tasks reach full feature rank and generalize exactly.
The carry-bit negative control moves from interpolation to detected model
mismatch as more constraints arrive. This shows why training fit alone is an
inadequate acceptance criterion.

The training-majority constant baseline is recorded for every fit. It matters
especially for imbalanced AND labels (about 75% negatives). No failed exact
fit is silently replaced with that baseline, and no inconsistency is hidden by
dropping training constraints.

## Noisy parity training

Linear parity, 1024 training examples, all eight seeds:

| Label flip probability | Repeated labels per example | Consistent fits / 8 | Residual voted label errors |
|---:|---:|---:|---:|
'''
    for noise in (100,500):
        for repeats in (1,5):
            rs=[r for r in rows if r['noise_bp']==noise and r['repeats']==repeats]
            s+=f'| {noise/100:.0f}% | {repeats} | {sum(r["conflicts"]==0 for r in rs)} | {sum(r["voted_errors"] for r in rs)} |\n'
    s+='''
Exact elimination is fragile to label noise. The repeats are an explicitly
costed independent-measurement assumption, not a general noise-tolerant learning
algorithm. [Blum, Kalai and Wasserman](https://arxiv.org/abs/cs/0010022) study the
distinct problem of learning parity with noisy labels; this implementation
does not reproduce their algorithm.

All 608 fit records, including contradictions, ranks, baseline scores, weights,
certified coverage and timings: [learning.csv](raw/learning.csv).
'''
    (OUT/'09-learned-inference.md').write_text(s,encoding='utf-8')


def report_cost(rows,dot):
    s='''# Case study 10 — Native inference cost and binary dot products

The machine is the Ryzen 7 9800X3D on Windows, Clang 22.1.8, generic
`-O3 -std=c++17`, one thread. No explicit hardware POPCNT or AVX requirement is
introduced. The compiler may optimize both reference and packed loops.

## End-to-end learned readout

The timed batch includes feature construction and XOR readout for 8192 unseen
inputs. It excludes label generation, accuracy scoring and ambiguity
certification. For each fit, the native program times three repetitions and
records the median, keeping a result checksum. Below are medians across eight
clean 1024-example fits of the same parity task, which every model learns exactly.

| Model | Feature count | Median batch milliseconds | Million predictions/s |
|---|---:|---:|---:|
'''
    for model in MODELS:
        rs=[r for r in rows if r['model']==model and r['task']==0 and r['train_count']==1024 and not r['noise_bp']]
        sec=statistics.median(r['inference_seconds'] for r in rs)
        s+=f'| {model} | {rs[0]["features"]} | {1000*sec:.4f} | {8192/sec/1e6:.3f} |\n'
    scalar=statistics.median(r['scalar_seconds'] for r in dot)
    packed=statistics.median(r['packed_seconds'] for r in dot)
    packing=statistics.median(r['packing_seconds'] for r in dot)
    s+='''
The simpler sufficient feature bank costs less than unnecessary nonlinear
features. The current implementation builds dense feature rows; it does not
yet exploit sparse learned coefficients to skip unused monomials. These are
small synthetic Boolean tasks, not tokens/second or language-model inference.

## Exact packed bipolar dot products

For {-1,+1} vectors, matching bits contribute +1 and differing bits contribute
-1, so `dot = N - 2*popcount(a XOR b)`. This sums products; it is not the same
operation as a GF(2) parity readout. The experiment uses 4096 pairs of 1024-bit
vectors. Every packed result matches the signed reference sum exactly in all
eight seeds and all three repetitions. Operand generation is outside timing.

| Operation | Median milliseconds |
|---|---:|
'''
    s+=f'| Signed dot-product batch | {scalar*1000:.4f} |\n| Already-packed XOR/popcount batch | {packed*1000:.4f} |\n| Packing the two signed input arrays | {packing*1000:.4f} |\n'
    s+=f'''
The ratio of median kernel times is **{scalar/packed:.2f}x** in favor of the
packed representation. However, packing plus one packed execution takes about
**{(packing+packed)*1000:.3f} ms**, versus **{scalar*1000:.3f} ms** for one signed
execution. The simple bit-by-bit packer is expensive: this experiment does
**not** show a one-use end-to-end speedup. Persistently packed weights/activations
or sufficient reuse are needed to amortize it, or the packer must be improved.
Using these medians, the arithmetic break-even estimate is about
**{packing/(scalar-packed):.1f} reuses** of the same packed operands; this is an
estimate, not a measured reused-network benchmark.

Packed operands occupy one bit rather than one signed byte per element here:
an 8x operand-storage reduction relative to this int8 reference. This is not a
32x claim against a float32 model, and no entire trained model is measured.

The published [XNOR-Net](https://arxiv.org/abs/1603.05279) demonstrates binary
operations in trained image models. It is relevant prior art, not an accuracy
or speed result reproduced by this microbenchmark. A practical binary network
would additionally need learned weights, activations/thresholds, scaling,
training and end-to-end task evaluation.

All raw timings/checksums: [binary-dot.csv](raw/binary-dot.csv). The packing
measurement is taken once per seed and repeated in its three timing rows;
those copies are not independent packing measurements. Active-desktop timing
and small batch lengths limit precision; no confidence interval is claimed.
'''
    (OUT/'10-native-inference-cost.md').write_text(s,encoding='utf-8')


def export_models(rows):
    folder=OUT/'models';folder.mkdir(exist_ok=True)
    entries=[]
    for model,task,name in [('linear',0,'parity'),('quadratic',2,'majority'),('cubic',4,'addition-bit2')]:
        r=next(r for r in rows if r['seed']==0 and r['model']==model and r['task']==task and r['train_count']==1024 and not r['noise_bp'])
        assert not r['conflicts'] and r['test_correct']==8192
        path=folder/(name+'.model')
        path.write_text(f'XOR_MODEL_V1\n{model}\n{r["features"]}\n'+r['weights_hex'].replace(';',' ')+'\n',newline='\n')
        entries.append({'file':path.relative_to(OUT).as_posix(),'sha256':sha(path),'training_seed':'0',
                        'training_count':1024,'held_out_count':8192,'held_out_correct':8192,'task_id':task,
                        'model':model,'feature_count':r['features']})
    (folder/'manifest.json').write_text(json.dumps(entries,indent=2)+'\n')


def index():
    s='''# 1GBInteger — XOR for inference

Recorded 2026-10-09. MIT Licence. This suite compares **exact hidden-state
inference**, **learned XOR readouts**, and a **packed binary inference primitive**.
It is a bounded synthetic experiment, not an LLM or physical inference claim.

## Case studies

8. [State recovery, identifiability, forecasting and noise](08-state-inference.md)
9. [Learning Boolean tasks and testing unseen inputs](09-learned-inference.md)
10. [Native learned inference and packed dot-product cost](10-native-inference-cost.md)

![XOR inference results](figures/xor-inference.png)

The native suite contains 280 state-recovery configurations, 608 learned fits
and 24 dot-product timing records. Four development and four fixed validation
seeds use the same predefined methods. No settings were tuned on validation.
The original integer engine and all previous experiment records are unchanged.

## What can be used now

- **Exact inverse/forecast engine:** solve XOR constraints, recover uniquely
  identifiable bits, and abstain on ambiguity or inconsistency. Sensor placement
  and mask knowledge are explicit. Zero noise is an assumption, not a guarantee.
- **Learned Boolean predictors:** three learned coefficient files are exported
  from development seed 0. `xor_predict` loads a model and performs inference
  without consulting the task-label generator. Nonlinear models explicitly
  compute AND features before the XOR readout.
- **Binary arithmetic primitive:** XOR/popcount exactly computes bipolar dot
  products on already-packed operands. Packing and reuse determine whether
  there is an end-to-end benefit.

```powershell
clang++ -O3 -std=c++17 xor_predict.cpp -o xor_predict.exe
.\\xor_predict.exe experiments/xor-inference/models/majority.model 0 1 3 7
# Expected input,prediction rows: 0,0  1,0  3,1  7,1
.\\xor_predict.exe experiments/xor-inference/models/addition-bit2.model 0 4 32 36
# Expected rows: 0,0  4,1  32,1  36,0
```

Inputs are 16-bit integers. `majority` uses bits 0..2. `addition-bit2` predicts
bit 2 of `(input & 7) + ((input >> 3) & 7)`. `parity` predicts the XOR of bits
0,2,5,7,11,15 and a constant one. These are learned synthetic mappings, not a
general-purpose neural network. The model format lists its feature bank/count
and packed hexadecimal coefficient words in the fixed native feature order.

## Reproduction

```powershell
clang++ -O3 -std=c++17 xor_inference.cpp -o xor_inference.exe
clang++ -O2 -std=c++17 -fsanitize=undefined -fno-sanitize-recover=all test_xor_inference.cpp -o test_xor_inference.exe
.\\test_xor_inference.exe
python run_xor_inference.py
python analyze_xor_inference.py
# Optional charts (matplotlib required only here):
python analyze_xor_inference.py --plots
python test_xor_models.py
```

On Linux omit `.exe`. [PROTOCOL.md](PROTOCOL.md) fixes every configuration and
defines abstention, noise, train/test separation and timing boundaries.
[manifest.json](manifest.json) records exact source/output hashes, compiler and
the collection command. [analysis-provenance.json](analysis-provenance.json)
records the reporting sources. Raw CSVs include all failures, ranks, noise
counts, exported coefficient bitsets and scores. [summary.json](summary.json)
is a machine-readable analysis, and [models/manifest.json](models/manifest.json)
identifies each deployable model's training origin.

The GF(2) solver is tested against exhaustive enumeration of all assignments
and queries for small systems, including inconsistent and underdetermined
systems. Additional tests cover 2048-variable systems, observation equations
against materialized evolution, feature definitions, task labels and exported
model truth tables. Native loops perform training, state inference and timing;
Python orchestrates collection and validates/reports results.

## Conclusion and next boundary

XOR is useful for exact linear inference and efficient binary computation.
Increasing an affine reservoir's state size does not supply nonlinear task
capacity. Nonlinear features plus an XOR readout learn a larger class, but need
enough independent data and are not automatically robust to noise.

The next meaningful AI experiment would be a noisy real dataset with a trained
binary model and an ordinary baseline, measuring end-to-end accuracy, memory,
packing and inference cost. These synthetic results do not yet justify replacing
a conventional model or claiming language, image or general reasoning ability.
'''
    (OUT/'README.md').write_text(s,encoding='utf-8')


def plot(state,learn,dot):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    known=next(r for r in state if r['seed']==20261009 and r['bits']==1024 and r['sensors']==1 and r['horizon']==1024 and not r['unknown_mask'] and not r['noise_bp'])
    unknown=next(r for r in state if r['seed']==20261009 and r['bits']==1024 and r['sensors']==1 and r['horizon']==2048 and r['unknown_mask'] and not r['noise_bp'])
    x=np.arange(2)
    for off,key,total,label,color in [(-.18,'initial_answered','initial_total','Initial-state bits uniquely known','#4477aa'),(.18,'sensor_answered','sensor_total','Observed-site forecast coverage','#228855')]:
        vals=[r[key]/r[total] for r in (known,unknown)]
        bars=ax[0,0].bar(x+off,vals,.36,label=label,color=color)
        ax[0,0].bar_label(bars,labels=[f'{v:.2%}' for v in vals],padding=3,fontsize=8)
    ax[0,0].set(xticks=x,xticklabels=['Known mask\n1024 observations','Unknown mask\n2048 observations'],ylim=(0,1.25),ylabel='Fraction',title='One sensor: recovery is not forecasting')
    ax[0,0].legend(fontsize=7,loc='upper right')
    matrix=np.zeros((6,4))
    for task in range(6):
        for col,model in enumerate(MODELS):
            rs=[r for r in learn if r['group']=='validation' and r['task']==task and r['model']==model and r['train_count']==1024 and not r['noise_bp']]
            matrix[task,col]=sum(r['conflicts']==0 for r in rs)/4
    ax[0,1].imshow(matrix,vmin=0,vmax=1,cmap='YlGn',aspect='auto')
    for task in range(6):
        for col in range(4):ax[0,1].text(col,task,'100%\nheld-out' if matrix[task,col]==1 else 'No fit',ha='center',va='center',fontsize=8,color='white' if matrix[task,col]==1 else 'black')
    ax[0,1].set(xticks=range(4),xticklabels=['linear','XOR\nreservoir','quadratic','cubic'],yticks=range(6),yticklabels=TASKS,title='1024 examples: all four validation seeds')
    for name,task,color in [('quadratic',2,'#4477aa'),('cubic',4,'#dd9933')]:
        xs=[64,256,1024];ys=[]
        for n in xs:
            rs=[r for r in learn if r['group']=='validation' and r['model']==name and r['task']==task and r['train_count']==n and not r['noise_bp']]
            ys.append(statistics.mean(r['test_correct']/8192 for r in rs))
        ax[1,0].plot(xs,ys,'o-',label=f'{name}: {TASKS[task]}',color=color)
    ax[1,0].axhline(1,color='black',ls='--',lw=.8,label='Training accuracy (both)')
    ax[1,0].set(xscale='log',xticks=[64,256,1024],xticklabels=['64','256','1024'],ylim=(.4,1.07),xlabel='Training examples',ylabel='Accuracy',title='Perfect fit can generalize near chance')
    ax[1,0].legend(fontsize=8)
    vals=[statistics.median(r[k] for r in dot)*1000 for k in ('scalar_seconds','packed_seconds','packing_seconds')]
    bars=ax[1,1].bar(['Signed dot\nreference','Already packed\nXOR/popcount','Packing\ninputs'],vals,color=['#777777','#228855','#dd9933'])
    ax[1,1].bar_label(bars,labels=[f'{v:.3f} ms' for v in vals],padding=3)
    ax[1,1].set(yscale='log',ylim=(.01,100),ylabel='Milliseconds (log scale)',title='4096 × 1024-bit dot products: packing matters')
    for a in (ax[0,0],ax[1,0],ax[1,1]):a.grid(alpha=.18);a.set_axisbelow(True)
    fig.suptitle('1GBInteger | XOR inference: exact constraints, learned readouts, binary kernels',fontsize=14)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'xor-inference.png',dpi=180);fig.savefig(folder/'xor-inference.pdf');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plots',action='store_true');main(parser.parse_args().plots)

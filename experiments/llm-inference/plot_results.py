"""Static, exportable measured-result figures (matplotlib)."""
import json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=pathlib.Path(__file__).resolve().parent
d=json.loads((HERE/'summary.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
modes=['stock','control','packed','prepack','bitplane','binary','ternary']
labels=['llama.cpp stock','No-repack control','Packed exact','Expanded exact','Bit-plane integer','Binary hybrid','Ternary hybrid']
for mode,label in zip(modes,labels):
    rows=[r for r in d['timing'] if r['mode']==mode and r['case'] in ['short','medium','long','extended']]
    rows.sort(key=lambda r:r['prompt_tokens'])
    for ax,key,title in [(axes[0,0],'generation_tps','Generation (tokens/s)'),(axes[0,1],'pp_tps','Prompt processing (tokens/s)')]:
        x=[r['prompt_tokens'] for r in rows];y=[r[key] for r in rows]
        ax.plot(x,y,'o-',label=label);ax.set_xscale('log');ax.set_xlabel('Prompt tokens');ax.set_ylabel(title);ax.grid(alpha=.25)
axes[0,0].legend(fontsize=8)
rows={r['mode']:r for r in d['timing'] if r['case']=='extended'}
axes[1,0].barh(labels,[rows[m]['peak_working_set_bytes']/2**20 for m in modes]);axes[1,0].set_xlabel('Process peak working set (MiB), extended context')
qm=['stock','packed','prepack','binary','ternary'];ql=['Stock','Packed exact','Expanded exact','Binary hybrid','Ternary hybrid']
axes[1,1].bar(ql,[sum(x['diagnostic_pass'] for x in d['quality'] if x['mode']==m) for m in qm]);axes[1,1].set_ylim(0,5.5);axes[1,1].set_ylabel('Diagnostic prompts passed / 5');axes[1,1].tick_params(axis='x',rotation=20)
fig.suptitle('1GBInteger: measured Qwen2.5-0.5B CPU inference\nRyzen 7 9800X3D, 8 threads, 353 MB mixed Q4_0 GGUF',fontsize=15)
out=HERE/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'inference-results.png',dpi=160);fig.savefig(out/'inference-results.pdf')

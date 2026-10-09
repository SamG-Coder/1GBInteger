"""Fresh-process matched runs of the entire pinned frame replay verifier."""
import argparse,gzip,hashlib,json,pathlib,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
UPSTREAM=ROOT/'.local-llm/integer-mult-bounds';LOCAL=ROOT/'.local-llm/circuit-replay'
PIN='d1d6c070f5a8c684727ee7ec35d930f9ebfa9758'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',type=int,default=5);args=p.parse_args()
    assert subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()==PIN
    LOCAL.mkdir(exist_ok=True);raw=HERE/'raw';raw.mkdir(exist_ok=True)
    executable=LOCAL/'replay.exe';records=[]
    for h in [23,25]:
        source=UPSTREAM/f'research/pair-assembly/frame/frame-word-{h}.json.gz'
        path=LOCAL/f'word-{h}.json';path.write_bytes(gzip.decompress(source.read_bytes()))
        for rep in range(args.runs):
            modes=['python','active','dense'];modes=modes[rep%3:]+modes[:rep%3]
            for mode in modes:
                cmd=[sys.executable,str(HERE/'baseline.py'),str(path)] if mode=='python' else [str(executable),str(path),mode]
                start=time.perf_counter();proc=subprocess.run(cmd,check=True,capture_output=True,text=True);wall=time.perf_counter()-start
                result=json.loads(proc.stdout);result.update(repetition=rep,process_wall_seconds=wall,input_sha256=sha(path),compressed_input_sha256=sha(source),executable_sha256=sha(executable) if mode!='python' else sha(pathlib.Path(sys.executable)))
                result['baseline_wrapper_sha256']=sha(HERE/'baseline.py')
                result['native_source_sha256']=sha(HERE/'replay.cpp')
                result['upstream_checker_sha256']=sha(UPSTREAM/'scripts/experiments/binary_frame_replay.py')
                save(raw/f'h{h}-{rep}-{mode}.json',result);records.append(result)
                print(h,rep,mode,round(result['total_seconds'],3),'seconds',flush=True)
            reference=json.loads((raw/f'h{h}-{rep}-python.json').read_text())
            for mode in ['active','dense']:
                native=json.loads((raw/f'h{h}-{rep}-{mode}.json').read_text())
                for key in ['h','roles','output_roles','elementary_xors','word_length','full_basis_vectors','rank_mass','histogram']:
                    assert native[key]==reference[key],(h,mode,key)
    save(HERE/'runs.json',records)
if __name__=='__main__':main()

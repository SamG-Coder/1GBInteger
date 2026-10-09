"""Fetch pinned external Apache-2.0 checker and MIT JSON header; build native."""
import pathlib,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[2];LOCAL=ROOT/'.local-llm/circuit-replay';UP=ROOT/'.local-llm/integer-mult-bounds'
PIN='d1d6c070f5a8c684727ee7ec35d930f9ebfa9758'
def run(*a):subprocess.run(list(map(str,a)),check=True,cwd=ROOT)
if not UP.exists():
    run('git','init',UP);run('git','-C',UP,'remote','add','origin','https://github.com/CrocSwap/integer-mult-bounds.git')
    run('git','-C',UP,'fetch','--depth','1','origin',PIN);run('git','-C',UP,'checkout','--detach','FETCH_HEAD')
assert subprocess.check_output(['git','-C',str(UP),'rev-parse','HEAD'],text=True).strip()==PIN
LOCAL.mkdir(exist_ok=True)
include=ROOT/'.local-llm/llama.cpp/vendor'
if not (include/'nlohmann/json.hpp').exists():
    raise RuntimeError('Run experiments/stock-llama-integration/setup.py to obtain the pinned MIT nlohmann header first')
run('clang++','-O3','-std=c++17','-march=native','-I'+str(include),ROOT/'experiments/circuit-replay/replay.cpp','-lpsapi','-o',LOCAL/'replay.exe')

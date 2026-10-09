"""Build against the pinned stock-integration libraries; no upstream changes."""
import pathlib, subprocess, sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
BUILD=ROOT/'.local-llm/stock-build'
if not (BUILD/'CMakeFiles/integer-llm.dir/linkLibs.rsp').exists():
    subprocess.run([sys.executable,str(ROOT/'experiments/stock-llama-integration/setup.py')],check=True,cwd=ROOT)
revision=subprocess.check_output(['git','-C',str(ROOT/'.local-llm/llama.cpp'),'rev-parse','HEAD'],text=True).strip()
if revision!='de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b':raise RuntimeError('wrong llama.cpp revision')
OUT=ROOT/'.local-llm/q2-traffic-build';OUT.mkdir(exist_ok=True)
subprocess.run(['clang++','-O3','-DNDEBUG','-std=c++17','-march=native','-D_WIN32_WINNT=0x0A00',
    '@CMakeFiles/integer-llm.dir/includes_CXX.rsp',
    str(ROOT/'experiments/stock-llama-integration/runner.cpp'),
    str(ROOT/'experiments/q2-output-traffic/kernels.cpp'),
    '@CMakeFiles/integer-llm.dir/linkLibs.rsp','-o',str(OUT/'integer-llm.exe')],cwd=BUILD,check=True)

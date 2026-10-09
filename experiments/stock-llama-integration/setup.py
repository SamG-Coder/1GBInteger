"""Reuse the pinned model and null dispatch hook from case study 11."""
import pathlib,subprocess,sys
ROOT=pathlib.Path(__file__).resolve().parents[2]
def run(*args):subprocess.run(list(map(str,args)),cwd=ROOT,check=True)
run(sys.executable,ROOT/'experiments/llm-inference/setup.py')
run('cmake','-S',ROOT/'experiments/stock-llama-integration','-B',ROOT/'.local-llm/stock-build','-G','MinGW Makefiles',
    f'-DLLAMA_SOURCE={(ROOT/".local-llm/llama.cpp").as_posix()}','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang++','-DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00','-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00','-DGGML_NATIVE=ON')
run('cmake','--build',ROOT/'.local-llm/stock-build','-j','8')
run('ctest','--test-dir',ROOT/'.local-llm/stock-build','--output-on-failure')

"""Pinned dependency/model setup. Downloads live only in ignored .local-llm."""
import hashlib, json, pathlib, subprocess, urllib.request
ROOT=pathlib.Path(__file__).resolve().parents[2]
LOCAL=ROOT/'.local-llm'
LLAMA='de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b'
MODEL_REV='41ba88dbac95fed2528c92514c131d73eb5a174b'
MODEL='Qwen2.5-0.5B-Instruct-Q4_0.gguf'
MODEL_SHA='c8cd5f37dd1235fb010c45316d4ff8af875e1a4e0ff368b4bf6cacb9053d4919'
MODEL_URL=f'https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/{MODEL_REV}/{MODEL}'
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(*args):subprocess.run(list(map(str,args)),cwd=ROOT,check=True)
if __name__=='__main__':
    LOCAL.mkdir(exist_ok=True)
    src=LOCAL/'llama.cpp'
    if not src.exists():
        run('git','clone','https://github.com/ggml-org/llama.cpp',src)
        run('git','-C',src,'checkout',LLAMA)
    assert subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip()==LLAMA
    model=LOCAL/MODEL
    if not model.exists():urllib.request.urlretrieve(MODEL_URL,model)
    assert sha(model)==MODEL_SHA, 'Model SHA256 mismatch'
    print('Verified model:',model.stat().st_size,'bytes',MODEL_SHA)
    run('python',ROOT/'experiments/llm-inference/patch_llama.py',src)
    run('cmake','-S',ROOT/'experiments/llm-inference','-B',LOCAL/'experiment-build','-G','MinGW Makefiles',
        f'-DLLAMA_SOURCE={src.as_posix()}','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER=clang','-DCMAKE_CXX_COMPILER=clang++',
        '-DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00','-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00','-DGGML_NATIVE=ON')
    run('cmake','--build',LOCAL/'experiment-build','-j','8')
    run(LOCAL/'experiment-build/test-llm-kernels.exe')

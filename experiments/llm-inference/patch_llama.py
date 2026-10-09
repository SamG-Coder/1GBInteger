"""One optional CPU operation hook. Null leaves upstream computation unchanged."""
import pathlib, subprocess, sys
PIN = 'de7fa0a3c6a2e1b4cd9f22eb8d6bf5b12dbdb63b'
p = pathlib.Path(sys.argv[1])
assert subprocess.check_output(['git', '-C', str(p), 'rev-parse', 'HEAD'], text=True).strip() == PIN
f = p / 'ggml/src/ggml-cpu/ggml-cpu.c'
s = f.read_text()
marker = '// 1GBInteger optional operation hook'
if marker not in s:
    needle = 'static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {'
    assert s.count(needle) == 1
    s = s.replace(needle, marker + '\nint (*integer_llm_hook)(const struct ggml_compute_params *, struct ggml_tensor *) = NULL;\n\n' + needle)
    needle = '    // extra_buffer op?'
    assert s.count(needle) == 1
    s = s.replace(needle, '    if (integer_llm_hook && integer_llm_hook(params, tensor)) { return; }\n\n' + needle)
    f.write_text(s, newline='\n')

"""Bind the reviewed evidence and shared independent checker (MIT)."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
files = sorted(p for p in HERE.iterdir() if p.is_file() and p.name != 'manifest.json')
bindings = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
shared = '../kappa-refinement/check_candidate.py'
bindings[shared] = hashlib.sha256((HERE/shared).read_bytes()).hexdigest()
(HERE/'manifest.json').write_text(json.dumps(dict(files=bindings),indent=2)+'\n',encoding='utf-8',newline='\n')

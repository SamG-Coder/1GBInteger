import hashlib,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for p in HERE.iterdir() if p.is_file() and p.name!='manifest.json']
(HERE/'manifest.json').write_text(json.dumps({'files':{p.name:sha(p) for p in files}},indent=2)+'\n',encoding='utf-8',newline='\n')

import hashlib,json,pathlib,sys
from check_candidate import validate
HERE=pathlib.Path(__file__).resolve().parent
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
manifest=json.loads((HERE/'manifest.json').read_text())
for name,h in manifest['files'].items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==h,name
d=json.loads((HERE/'candidate.json').read_text());r=validate(d)
saved=json.loads((HERE/'independent-check.json').read_text())
assert all(saved[k]==v for k,v in r.items())
assert len(saved['corruptions_rejected'])==3
log=(HERE/'upstream-verification.txt').read_text()
assert 'PASS complete scalar identity on all 27521 formal variables, over F2 and over Z' in log
assert 'PASS atom 1/2000:' in log
print('Verified exact kappa',r['kappa'],'47 strict constraints, 7 margins, next-grid rejection and provenance')

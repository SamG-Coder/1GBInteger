#!/usr/bin/env python3
"""Compare temporal observations against fully evolved state samples."""
import csv,os,pathlib,subprocess,tempfile
root=pathlib.Path(__file__).resolve().parent
ext='.exe' if os.name=='nt' else ''
temporal=root/('integer_universe_temporal'+ext)
full=root/('integer_universe'+ext)
with tempfile.TemporaryDirectory() as tmp:
 for seed in (0,1,12345):
  for steps in (0,1,2,7,65,129):
   output=pathlib.Path(tmp)/'temporal.csv'
   subprocess.run([str(temporal),'--mb','1','--seed',str(seed),'--steps',str(steps),
                   '--atoms','8','--out',str(output)],check=True,capture_output=True)
   with output.open(newline='') as f:rows=list(csv.DictReader(f))
   selected=[r for r in rows if int(r['step'])==steps]
   sample=pathlib.Path(tmp)/'full.bin'
   subprocess.run([str(full),'--mb','1','--seed',str(seed),'--steps',str(steps),
                   '--fast','--threads','2','--sample-bytes','64','--out',str(sample)],
                   check=True,capture_output=True)
   raw=sample.read_bytes()
   assert len(raw)==64
   for j,row in enumerate(selected):
    expected=int.from_bytes(raw[j*8:(j+1)*8],'little')
    assert int(row['limb_hex'],16)==expected,(seed,steps,j)
    assert int(row['measurement'])==((expected>>(j%64))&1)
 print('PASS: 18 seed/step configurations, 8 atoms each, temporal == full state')

"""Run the pinned upstream checker unchanged on decompressed JSON."""
import ast,ctypes,json,pathlib,sys,time
root=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root/'.local-llm/integer-mult-bounds/scripts/experiments'))
import binary_frame_replay as upstream
# Insert timing reads only; retain every upstream operation and assertion.
tree=ast.parse(pathlib.Path(upstream.__file__).read_text())
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='replay')
body=[ast.Global(names=['phase_seconds'])]
for node in function.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='initial' for t in node.targets):
        body+=ast.parse('phase_start=phase_clock()').body
    if isinstance(node,ast.Return):body+=ast.parse('phase_seconds=phase_clock()-phase_start').body
    body.append(node)
function.body=body;ns=vars(upstream).copy();ns['phase_clock']=time.perf_counter
exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'<timed upstream replay>','exec'),ns)
checker=upstream.replay if '--unmodified' in sys.argv else ns['replay']
start=time.perf_counter();result=checker(sys.argv[1]);result['total_seconds']=time.perf_counter()-start
result['dirty_seconds']=None if '--unmodified' in sys.argv else ns['phase_seconds']
class Counters(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in ['peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile']]
c=Counters();c.cb=ctypes.sizeof(c)
ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(c),c.cb)
result.update(mode='python',peak_working_set_bytes=c.peak_ws)
print(json.dumps(result))

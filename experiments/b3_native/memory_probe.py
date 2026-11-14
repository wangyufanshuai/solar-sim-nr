"""Bounded memory isolation: monitor calls, fixed-state native RHS, diagnostics."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from experiments.b1_gauge_wave import run_b1
from experiments.b3_native.gauge_growth_modes import inspect_state
from experiments.b3_native.gauge_spatial_refinement import ko6_x
from experiments.b3_native.long_diagnostics import peak_rss
from kernel.native_evolve import make_rhs, state_from_slots
from science.numerical_relativity import BSSNState3D


# Independent fixed-type observer; it never calls the peak_rss being profiled.
class _MemoryCounters(ctypes.Structure):
    _fields_ = [("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)] + [
        (name,ctypes.c_size_t) for name in ("peak","rss","pool_peak","pool",
        "nonpaged_peak","nonpaged","pagefile","pagefile_peak")]


if os.name == "nt":
    _kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    _kernel.GetCurrentProcess.restype=wintypes.HANDLE
    _query=ctypes.WinDLL("psapi",use_last_error=True).GetProcessMemoryInfo
    _query.argtypes=(wintypes.HANDLE,ctypes.POINTER(_MemoryCounters),wintypes.DWORD)
    _query.restype=wintypes.BOOL


def memory():
    if os.name != "nt":
        import resource
        value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        value=int(value*(1 if __import__("sys").platform == "darwin" else 1024))
        return dict(peak_bytes=value,rss_bytes=None)
    counters=_MemoryCounters(); counters.cb=ctypes.sizeof(counters)
    if not _query(_kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(peak_bytes=counters.peak,rss_bytes=counters.rss)


def run(out,mode):
    if mode not in ("monitor","rhs","diagnostics"):
        raise ValueError("unknown registered probe")
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    start=time.perf_counter(); root=Path(__file__).resolve().parents[2]
    checkpoints={"monitor":(0,1024,2048,4096,8192),"rhs":(0,64,128,256),
                 "diagnostics":(0,8,16,32)}[mode]
    q=None; current=14.; original=None; first=None
    if mode != "monitor":
        source=root/"docs/gauge-stage/dissipation-timestep/nx32-ko0p1-dt4-step7168-sample.npz"
        with np.load(source,allow_pickle=False) as saved:q=saved["q"].copy()
        original=q.tobytes(); grid=run_b1.grid(32,4)
        native=make_rhs(spacing=grid.spacing,order=4,gauge=run_b1.GAUGE,
                        config=run_b1.CONFIG,harmonic=True,threads=1)
    result=dict(mode=mode,source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
                scientific_evolution=False,status="RUNNING",samples=[],deterministic_outputs=True,
                max_seconds=600,max_working_set_bytes=512*2**20,max_output_bytes=64*2**20)
    def observe(calls):
        row=dict(calls=calls,seconds=time.perf_counter()-start,**memory())
        # Read CPython's cache only; do not clear or mutate runtime pointer types.
        row["pointer_type_cache_entries"]=len(ctypes._pointer_type_cache) if hasattr(ctypes,"_pointer_type_cache") else None
        result["samples"].append(row)
        print(json.dumps(row),flush=True)
    def check():
        if time.perf_counter()-start>=600 or memory()["peak_bytes"]>512*2**20:
            raise TimeoutError("600 s / 512 MiB probe ceiling")
    try:
        observe(0)
        for count in range(1,checkpoints[-1]+1):
            if mode == "monitor":peak_rss()
            elif mode == "rhs":
                raw=native(q)+ko6_x(q,grid.spacing[0],.1)
                digest=hashlib.sha256(raw.tobytes()).hexdigest()
            else:
                details,arrays=inspect_state(state_from_slots(q,grid),
                    BSSNState3D.gauge_wave(grid,amplitude=.1,time=current))
                hasher=hashlib.sha256(json.dumps(details,sort_keys=True,allow_nan=False).encode())
                for name,array in sorted(arrays.items()):hasher.update(name.encode()); hasher.update(array.tobytes())
                digest=hasher.hexdigest()
            if mode != "monitor":
                first=digest if first is None else first
                result["deterministic_outputs"] &= digest==first
            check()
            if count in checkpoints:observe(count)
        result["status"]="COMPLETED"
    except TimeoutError as exc:result.update(status="RESOURCE_STOPPED",reason=str(exc))
    result["collected_after"]=gc.collect(); observe(count)
    result.update(input_unchanged=q is None or q.tobytes()==original,
                  output_sha256=first,seconds=time.perf_counter()-start,**memory())
    payload=(json.dumps(result,indent=2,allow_nan=False)+"\n").encode()
    if len(payload)>64*2**20:raise RuntimeError("64 MiB output limit")
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("xb") as f:f.write(payload)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("monitor","rhs","diagnostics"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); run(args.out,args.mode)

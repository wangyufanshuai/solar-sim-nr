"""Registered S3 native space/time refinement from analytic gauge-wave data."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.b1_gauge_wave import run_b1
from experiments.b3_native.gauge_growth_modes import inspect_state
from experiments.b3_native.gauge_restart import CheckedRHS, ResourceStopped
from experiments.b3_native.long_diagnostics import ScientificFailure, peak_rss, validate_slots
from kernel.native_evolve import make_rhs, rk4, slots, state_from_slots
from science.numerical_relativity import BSSNState3D, FOCCZ4State3D


def replay_control(q, path, current, dt):
    """Compare the same-grid/time baseline; retain mismatch without changing q."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as saved:
        reference = saved["q"]
        if float(saved["time"]) != current or float(saved["dt"]) != dt or reference.shape != q.shape:
            raise ValueError("replay reference must match grid, time and timestep")
        delta = q-reference
    return dict(path=path.as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bitwise_equal=q.tobytes() == reference.tobytes(),
                max_slot_difference=float(np.max(np.abs(delta))))


def run(out, *, points=(32,16,64), target_time=24., max_seconds=600):
    if not points or len(set(points)) != len(points) or any(n not in (16,32,64) for n in points):
        raise ValueError("unique registered nx values (16,32,64) required")
    if not 0 < target_time <= 24 or not 0 < max_seconds <= 600:
        raise ValueError("registered target <=24 and deadline <=600 required")
    if any(not float(target_time*4*n).is_integer() for n in points):
        raise ValueError("target time must align with every timestep")
    out = Path(out)
    out.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    def check():
        if time.perf_counter()-start >= max_seconds or peak_rss() > 512*1024**2:
            raise ResourceStopped("600 s/requested deadline or 512 MiB process ceiling")
        if sum(p.stat().st_size for p in out.iterdir()) > 60*1024**2:
            raise ResourceStopped("64 MiB output ceiling; final report reserve")
    def write(name,data):
        if sum(p.stat().st_size for p in out.iterdir())+len(data) > 64*1024**2:
            raise ResourceStopped("64 MiB output ceiling")
        (out/name).write_bytes(data)
    def save(name,**arrays):
        payload=io.BytesIO()
        np.savez_compressed(payload,**arrays)
        write(name,payload.getvalue())
    result=dict(source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True,timeout=5).strip(),
                source_tree_clean=not subprocess.check_output(["git","status","--porcelain"],cwd=root,text=True,timeout=5).strip(),
                status="RUNNING",scientific_acceptance_established=False,external_independent_reproduction=False,
                settings=dict(points=list(points),target_time=target_time,amplitude=.1,spatial_order=4,
                              cfl=.25,integrator="additive canonical-slot RK4",threads=1,
                              max_seconds=max_seconds,max_working_set_bytes=512*1024**2,max_output_bytes=64*1024**2),
                trajectories=[])
    resource_stopped=False
    for nx in points:
        if resource_stopped:
            result["trajectories"].append(dict(nx=nx,status="NOT_STARTED_RESOURCE_STOP"))
            continue
        grid=run_b1.grid(nx,4)
        dt=1/(4*nx)
        q=slots(FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(grid,amplitude=.1)))
        validate_slots(q)
        count=0
        row=dict(nx=nx,shape=list(grid.shape),dt=dt,status="RUNNING",samples=[])
        result["trajectories"].append(row)
        rhs=CheckedRHS(make_rhs(spacing=grid.spacing,order=4,gauge=run_b1.GAUGE,
                               config=run_b1.CONFIG,harmonic=True,threads=1),check)
        def sample():
            current=count*dt
            details,raw=inspect_state(state_from_slots(q,grid),
                BSSNState3D.gauge_wave(grid,amplitude=.1,time=current))
            filename=f"nx{nx}-step{count}-sample.npz"
            save(filename,q=q,time=current,dt=dt,**raw)
            sample_row=dict(time=current,step=count,arrays_path=filename,**details)
            if nx == 32 and current in (0.,15.,18.,22.):
                sample_row["baseline_replay"]=replay_control(q,root/f"docs/gauge-stage/baseline-t{int(current)}.npz",current,dt)
            row["samples"].append(sample_row)
            print(json.dumps(dict(nx=nx,time=current,max_field_error_over_amplitude=
                  max(v["l_inf_over_amplitude"] for v in details["fields"].values()))),flush=True)
        candidate=None
        try:
            check()
            sample()
            for step in range(1,int(target_time/dt)+1):
                candidate=None
                try:
                    with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):
                        candidate=rk4(q,dt,1,rhs)
                        validate_slots(candidate)
                except (ScientificFailure,FloatingPointError,np.linalg.LinAlgError) as exc:
                    phase="rk_stage" if rhs.failed_input is not None else "completed_step_reconstruction"
                    row.update(status="SCIENTIFIC_FAILURE",failure=dict(message=str(exc),phase=phase,
                        stage=rhs.failed_stage,failed_interval=[count*dt,(count+1)*dt]))
                    if rhs.failed_input is not None:
                        stage_time=count*dt+(0.,.5,.5,1.)[rhs.failed_stage-1]*dt
                        row["failure"]["failed_stage_time"]=stage_time
                        save(f"nx{nx}-failed-input.npz",q=rhs.failed_input,time=stage_time,
                             last_valid_time=count*dt,dt=dt,stage=rhs.failed_stage)
                    elif candidate is not None:
                        save(f"nx{nx}-failed-candidate.npz",q=candidate,time=(count+1)*dt,dt=dt)
                    break
                q,count=candidate,step
                if step % (4*nx) == 0 or step == int(target_time/dt):
                    with np.errstate(over="raise",invalid="raise",divide="raise"):
                        sample()
            else:
                row["status"]="REACHED_TARGET"
        except ResourceStopped as exc:
            row.update(status="RESOURCE_STOPPED",resource_stop=str(exc))
            resource_stopped=True
        except (ScientificFailure,FloatingPointError,np.linalg.LinAlgError) as exc:
            row.update(status="DIAGNOSTIC_FAILURE",failure=dict(message=str(exc),phase="sampled_diagnostics"))
        row.update(completed_steps=count,last_valid_time=count*dt)
        filename=f"nx{nx}-last-valid.npz"
        save(filename,q=q,time=count*dt,dt=dt)
        row["last_valid_arrays"]=filename
    result["all_trajectories_reached_target"]=all(t["status"] == "REACHED_TARGET" for t in result["trajectories"])
    result["status"]=("RESOURCE_STOPPED" if resource_stopped else
                       "COMPLETED" if result["all_trajectories_reached_target"] else "SCIENTIFIC_FAILURE")
    result.update(seconds=time.perf_counter()-start,peak_rss_bytes=peak_rss())
    write("result.json",(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--target-time",type=float,default=24.)
    parser.add_argument("--max-seconds",type=float,default=600.)
    args=parser.parse_args()
    result=run(args.out,target_time=args.target_time,max_seconds=args.max_seconds)
    print(json.dumps({k:result[k] for k in ("status","source_commit","seconds","peak_rss_bytes")}))

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


def ko6_x(q, dx, sigma):
    """Experimental x-only KO p=5 term, Q=sigma delta2_x^3/(64 dx).

    Symbol -sigma*sin(pi*k/nx)^6/dx. Each second difference is written
    as two differences from the centre so constants are annihilated exactly.
    This dissipates all 58 canonical slots, including logarithmic gauge fields.
    """
    q=np.asarray(q)
    if q.ndim != 4 or q.shape[0] != 58 or q.shape[1] < 8 or q.dtype != np.float64:
        raise ValueError("expected binary64 canonical-slot array with nx>=8")
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(sigma) or not 0 <= sigma <= 1/3:
        raise ValueError("positive finite dx and 0<=sigma<=1/3 required")
    if sigma == 0:
        return np.zeros_like(q)
    delta=q
    for _ in range(3):
        delta=(np.roll(delta,1,axis=1)-delta)+(np.roll(delta,-1,axis=1)-delta)
    return sigma/(64*dx)*delta


def convergence_summary(rows):
    """Observed analytic-error ratios at common integer times; no acceptance gate."""
    grids={row["nx"]: {sample["time"]: sample for sample in row.get("samples",[])} for row in rows}
    if set(grids) != {16,32,64}:
        return []
    common=sorted((set(grids[16]) & set(grids[32]) & set(grids[64])) - {0.})
    def errors(sample):
        return {**{name: value["l_inf_over_amplitude"] for name,value in sample["fields"].items()},
                "hamiltonian_difference": sample["constraints"]["hamiltonian"]["difference"]["l_inf"],
                "metric_gradient_difference": sample["auxiliary_consistency"]["metric_gradient"]["difference"]["l_inf"]}
    result=[]
    for current in common:
        by_grid={nx: errors(grids[nx][current]) for nx in grids}
        metrics={}
        for name in by_grid[16]:
            values=[by_grid[nx][name] for nx in (16,32,64)]
            orders=[float(np.log2(a)-np.log2(b)) if a>0 and b>0 else None
                    for a,b in zip(values,values[1:])]
            metrics[name]=dict(errors=values,orders=orders)
        result.append(dict(time=current,metrics=metrics))
    return result


def run(out, *, points=(32,16,64), target_time=24., max_seconds=600, sigmas=None, fixed_sigma=None):
    if not points or len(set(points)) != len(points) or any(n not in (16,32,64) for n in points):
        raise ValueError("unique registered nx values (16,32,64) required")
    if sigmas is not None and (tuple(points) != (32,) or tuple(sigmas) != (0.,.02,.1)):
        raise ValueError("registered dissipation comparison requires nx32 and sigmas=(0,.02,.1)")
    if fixed_sigma is not None and (tuple(points) != (16,32,64) or fixed_sigma != .1 or sigmas is not None):
        raise ValueError("registered fixed-sigma refinement requires nx=(16,32,64), sigma=.1, no sigma suite")
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
                settings=dict(points=list(points),dissipation_sigmas=None if sigmas is None else list(sigmas),
                              fixed_dissipation_sigma=fixed_sigma,
                              target_time=target_time,amplitude=.1,spatial_order=4,
                              cfl=.25,integrator="additive canonical-slot RK4",threads=1,
                              max_seconds=max_seconds,max_working_set_bytes=512*1024**2,max_output_bytes=64*1024**2),
                trajectories=[])
    resource_stopped=False
    cases=[(nx,0. if fixed_sigma is None else fixed_sigma) for nx in points] if sigmas is None else [(32,sigma) for sigma in sigmas]
    for nx,sigma in cases:
        name=f"nx{nx}" if sigma == 0 else f"nx{nx}-ko{sigma:g}".replace(".","p")
        if resource_stopped:
            result["trajectories"].append(dict(nx=nx,dissipation_sigma=sigma,name=name,status="NOT_STARTED_RESOURCE_STOP"))
            continue
        grid=run_b1.grid(nx,4)
        dt=1/(4*nx)
        q=slots(FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(grid,amplitude=.1)))
        validate_slots(q)
        count=0
        row=dict(nx=nx,dissipation_sigma=sigma,name=name,shape=list(grid.shape),dt=dt,status="RUNNING",samples=[])
        result["trajectories"].append(row)
        native=make_rhs(spacing=grid.spacing,order=4,gauge=run_b1.GAUGE,
                        config=run_b1.CONFIG,harmonic=True,threads=1)
        # Baseline calls the native RHS directly, avoiding even an addition of zero.
        rhs=CheckedRHS(native if sigma == 0 else lambda q:native(q)+ko6_x(q,grid.spacing[0],sigma),check)
        def sample():
            current=count*dt
            details,raw=inspect_state(state_from_slots(q,grid),
                BSSNState3D.gauge_wave(grid,amplitude=.1,time=current))
            filename=f"{name}-step{count}-sample.npz"
            save(filename,q=q,time=current,dt=dt,**raw)
            sample_row=dict(time=current,step=count,arrays_path=filename,**details)
            if nx == 32 and sigma == 0 and current in (0.,15.,18.,22.):
                sample_row["baseline_replay"]=replay_control(q,root/f"docs/gauge-stage/baseline-t{int(current)}.npz",current,dt)
            if fixed_sigma is not None and nx == 32 and float(current).is_integer():
                sample_row["fixed_sigma_replay"]=replay_control(q,
                    root/f"docs/gauge-stage/dissipation/nx32-ko0p1-step{count}-sample.npz",current,dt)
            row["samples"].append(sample_row)
            print(json.dumps(dict(nx=nx,dissipation_sigma=sigma,time=current,max_field_error_over_amplitude=
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
                        save(f"{name}-failed-input.npz",q=rhs.failed_input,time=stage_time,
                             last_valid_time=count*dt,dt=dt,stage=rhs.failed_stage)
                    elif candidate is not None:
                        save(f"{name}-failed-candidate.npz",q=candidate,time=(count+1)*dt,dt=dt)
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
        filename=f"{name}-last-valid.npz"
        save(filename,q=q,time=count*dt,dt=dt)
        row["last_valid_arrays"]=filename
    result["all_trajectories_reached_target"]=all(t["status"] == "REACHED_TARGET" for t in result["trajectories"])
    result["status"]=("RESOURCE_STOPPED" if resource_stopped else
                       "COMPLETED" if result["all_trajectories_reached_target"] else "SCIENTIFIC_FAILURE")
    if fixed_sigma is not None:
        result["analytic_error_convergence"]=convergence_summary(result["trajectories"])
    result.update(seconds=time.perf_counter()-start,peak_rss_bytes=peak_rss())
    write("result.json",(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--target-time",type=float,default=24.)
    parser.add_argument("--max-seconds",type=float,default=600.)
    study=parser.add_mutually_exclusive_group()
    study.add_argument("--dissipation",action="store_true",help="registered nx32 baseline/sigma=.02/.1 comparison")
    study.add_argument("--dissipation-refinement",action="store_true",help="registered nx16/32/64, fixed sigma=.1 comparison")
    args=parser.parse_args()
    options=dict(points=(32,),sigmas=(0.,.02,.1)) if args.dissipation else {}
    if args.dissipation_refinement:
        options=dict(points=(16,32,64),fixed_sigma=.1)
    result=run(args.out,target_time=args.target_time,max_seconds=args.max_seconds,**options)
    print(json.dumps({k:result[k] for k in ("status","source_commit","seconds","peak_rss_bytes")}))

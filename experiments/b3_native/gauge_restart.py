"""S3 bounded same-snapshot restarts with a common additive-slot RK4."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from experiments.b1_gauge_wave import run_b1
from experiments.b3_native.gauge_stage_diagnostic import analytic_error, python_rhs, pack_rhs
from experiments.b3_native.geometry_terms import arithmetic_ablation_rhs
from experiments.b3_native.long_diagnostics import ScientificFailure, peak_rss, validate_slots
from kernel.native_evolve import make_rhs, rk4, state_from_slots, base_fields


class ResourceStopped(RuntimeError):
    pass


class CheckedRHS:
    """Retain the actual failed stage input; count stage calls within each RK4 step."""
    def __init__(self, rhs, resource_check):
        self.rhs, self.resource_check = rhs, resource_check
        self.calls, self.failed_input, self.failed_stage = 0, None, None

    def __call__(self, q):
        self.resource_check()
        self.calls += 1
        try:
            validate_slots(q)
            out = self.rhs(q)
            if not np.all(np.isfinite(out)):
                raise ScientificFailure("nonfinite evaluated RHS")
            return out
        except (ScientificFailure, FloatingPointError, np.linalg.LinAlgError) as exc:
            self.failed_input, self.failed_stage = q.copy(), 1+(self.calls-1) % 4
            raise ScientificFailure(f"RK4 stage {self.failed_stage}: {exc}") from exc


def difference(q, reference):
    fields, ref_fields = base_fields(q), base_fields(reference)
    normalized = max(float(np.max(np.abs(fields[f]-ref_fields[f]))) for f in run_b1.FIELDS)/.1
    return dict(max_slot_difference=float(np.max(np.abs(q-reference))),
                base_field_difference_over_amplitude=normalized,
                passes_1_percent=normalized <= .01)


def terminal_states(trajectories, counts, grid, base_time, dt):
    """Compare only states at the same last-valid time, including a failed-step endpoint."""
    result = {}
    for name,q in trajectories.items():
        current = base_time+counts[name]*dt
        row = dict(last_valid_time=current,
                   analytic_error_over_amplitude=analytic_error(q,grid,current))
        if counts[name] == counts["native"]:
            row.update(difference(q,trajectories["native"]))
        result[name] = row
    return result


def run(out, snapshot, *, steps=256, max_seconds=600):
    if not 1 <= steps <= 256 or not 0 < max_seconds <= 600:
        raise ValueError("registered bounds: 1..256 steps and 0..600 seconds")
    out, snapshot = Path(out), Path(snapshot)
    start = time.perf_counter()
    def resource_check():
        if time.perf_counter()-start >= max_seconds or peak_rss() > 512*1024**2:
            raise ResourceStopped("600 s/requested deadline or 512 MiB process ceiling")
        if out.exists() and sum(p.stat().st_size for p in out.iterdir()) > 63*1024**2:
            raise ResourceStopped("64 MiB output ceiling; report space reserved")
    with np.load(snapshot, allow_pickle=False) as saved:
        q0, base_time, dt = saved["q"].copy(), float(saved["time"]), float(saved["dt"])
    grid = run_b1.grid(32, 4)
    if q0.dtype != np.float64 or q0.shape != (58, *grid.shape) or base_time != 22 or dt != 1/128:
        raise ValueError("expected registered binary64 T22 snapshot and dt=1/128")
    validate_slots(q0)
    out.mkdir(parents=True, exist_ok=False)
    native = make_rhs(spacing=grid.spacing, order=4, gauge=run_b1.GAUGE,
                      config=run_b1.CONFIG, harmonic=True, threads=1)
    ablation = arithmetic_ablation_rhs()
    functions = dict(native=native, native_one_ulp=native, python=lambda q: python_rhs(q, grid),
                     arithmetic=lambda q: pack_rhs(ablation(state_from_slots(q, grid),
                                                            gauge=run_b1.GAUGE, config=run_b1.CONFIG)))
    trajectories = {name: q0.copy() for name in functions}
    index = (0,8,0,0)
    trajectories["native_one_ulp"][index] = np.nextafter(q0[index], np.inf)
    seed = float(trajectories["native_one_ulp"][index]-q0[index])
    rhs = {name: CheckedRHS(f, resource_check) for name, f in functions.items()}
    counts, failures = {name: 0 for name in functions}, {}
    result = dict(source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,timeout=5).strip(),
                  input_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                  settings=dict(base_time=base_time,dt=dt,target_time=base_time+steps*dt,steps=steps,
                                shape=list(grid.shape),amplitude=.1,threads=1,max_seconds=max_seconds,
                                max_working_set_bytes=512*1024**2,max_output_bytes=64*1024**2),
                  integrator="common additive canonical-slot RK4", external_independent_reproduction=False,
                  perturbation=dict(index=list(index),delta=seed), samples=[],failures=failures)
    def sample(step):
        current = base_time+step*dt
        live = [name for name in trajectories if counts[name] == step and name not in failures]
        row = dict(time=current,trajectories={})
        for name in live:
            entry = dict(analytic_error_over_amplitude=analytic_error(trajectories[name],grid,current))
            if "native" in live:
                entry.update(difference(trajectories[name],trajectories["native"]))
            row["trajectories"][name] = entry
        result["samples"].append(row)
        print(json.dumps(dict(time=current,live=live,failures=list(failures))),flush=True)
    result["status"] = "RUNNING"
    try:
        sample(0)
        for step in range(1,steps+1):
            new_failure = False
            for name, q in trajectories.items():
                if name in failures:
                    continue
                resource_check()
                try:
                    with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):
                        candidate = rk4(q,dt,1,rhs[name])
                        validate_slots(candidate)
                    trajectories[name], counts[name] = candidate, step
                except (ScientificFailure,FloatingPointError,np.linalg.LinAlgError) as exc:
                    last = base_time+counts[name]*dt
                    failures[name] = dict(message=str(exc),last_valid_time=last,
                                          failed_interval=[last,last+dt],stage=rhs[name].failed_stage)
                    if rhs[name].failed_input is not None:
                        np.savez_compressed(out/f"{name}-failed-input.npz",q=rhs[name].failed_input,
                                            step_start_time=last,stage=rhs[name].failed_stage,dt=dt)
                    new_failure = True
            if step % 16 == 0 or new_failure or step == steps:
                sample(step)
            if len(failures) == len(trajectories):
                break
        result["status"] = "SCIENTIFIC_FAILURE" if failures else "COMPLETED"
    except ResourceStopped as exc:
        result.update(status="RESOURCE_STOPPED",resource_reason=str(exc))
    for name,q in trajectories.items():
        np.savez_compressed(out/f"{name}-last-valid.npz",q=q,time=base_time+counts[name]*dt,dt=dt)
    result["terminal_states"] = terminal_states(trajectories,counts,grid,base_time,dt)
    result.update(steps=counts,seconds=time.perf_counter()-start,peak_rss_bytes=peak_rss(),
                  all_trajectories_reached_target=all(n == steps for n in counts.values()))
    if result["seconds"] > max_seconds or result["peak_rss_bytes"] > 512*1024**2:
        result["status"] = "RESOURCE_STOPPED"
    text = json.dumps(result,indent=2,allow_nan=False)+"\n"
    if sum(p.stat().st_size for p in out.iterdir())+len(text.encode()) > 64*1024**2:
        raise ResourceStopped("64 MiB output limit")
    (out/"result.json").write_text(text,encoding="utf-8")
    return result


def run_refinement(out, snapshot, *, coarse_steps=256):
    """Timestep sensitivity of native restarts; the T22 initial state is already inaccurate."""
    if not 1 <= coarse_steps <= 256:
        raise ValueError("registered bound: 1..256 coarse steps")
    out, snapshot = Path(out), Path(snapshot)
    start = time.perf_counter()
    with np.load(snapshot, allow_pickle=False) as saved:
        q0, base_time, coarse_dt = saved["q"].copy(), float(saved["time"]), float(saved["dt"])
    grid = run_b1.grid(32, 4)
    if q0.dtype != np.float64 or q0.shape != (58, *grid.shape) or base_time != 22 or coarse_dt != 1/128:
        raise ValueError("expected registered binary64 T22 snapshot and dt=1/128")
    validate_slots(q0)
    out.mkdir(parents=True, exist_ok=False)
    def resource_check():
        if time.perf_counter()-start >= 600 or peak_rss() > 512*1024**2:
            raise ResourceStopped("600 s or 512 MiB process ceiling")
        if sum(p.stat().st_size for p in out.iterdir()) > 63*1024**2:
            raise ResourceStopped("64 MiB output ceiling; report space reserved")
    native = make_rhs(spacing=grid.spacing,order=4,gauge=run_b1.GAUGE,
                      config=run_b1.CONFIG,harmonic=True,threads=1)
    factors = dict(native=1,half=2,quarter=4)
    q = {name:q0.copy() for name in factors}
    counts = {name:0 for name in factors}
    rhs = {name:CheckedRHS(native,resource_check) for name in factors}
    failures = {}
    result = dict(source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,timeout=5).strip(),
                  input_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                  settings=dict(base_time=22,coarse_dt=coarse_dt,target_time=22+coarse_steps*coarse_dt,
                                timestep_by_trajectory={n:coarse_dt/f for n,f in factors.items()},
                                max_seconds=600,max_working_set_bytes=512*1024**2,max_output_bytes=64*1024**2,
                                shape=list(grid.shape),amplitude=.1,threads=1),
                  status="RUNNING",external_independent_reproduction=False,failures=failures,samples=[])
    def sample(step):
        current = base_time+step*coarse_dt
        live = [n for n,f in factors.items() if n not in failures and counts[n] == step*f]
        row = dict(time=current,trajectories={})
        for name in live:
            entry = dict(analytic_error_over_amplitude=analytic_error(q[name],grid,current))
            if "quarter" in live:
                entry["difference_from_finest"] = difference(q[name],q["quarter"])
            row["trajectories"][name] = entry
        result["samples"].append(row)
        print(json.dumps(dict(time=current,live=live,failures=list(failures))),flush=True)
    try:
        sample(0)
        for step in range(1,coarse_steps+1):
            new_failure = False
            for name,factor in factors.items():
                if name in failures:
                    continue
                dt = coarse_dt/factor
                for _ in range(factor):
                    try:
                        resource_check()
                        with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):
                            candidate = rk4(q[name],dt,1,rhs[name])
                            validate_slots(candidate)
                        q[name], counts[name] = candidate, counts[name]+1
                    except (ScientificFailure,FloatingPointError,np.linalg.LinAlgError) as exc:
                        last = base_time+counts[name]*dt
                        failures[name] = dict(message=str(exc),last_valid_time=last,
                                              failed_interval=[last,last+dt],stage=rhs[name].failed_stage)
                        if rhs[name].failed_input is not None:
                            np.savez_compressed(out/f"{name}-failed-input.npz",q=rhs[name].failed_input,
                                                step_start_time=last,stage=rhs[name].failed_stage,dt=dt)
                        new_failure = True
                        break
            if step % 16 == 0 or new_failure or step == coarse_steps:
                sample(step)
            if len(failures) == len(factors):
                break
        result["status"] = "SCIENTIFIC_FAILURE" if failures else "COMPLETED"
    except ResourceStopped as exc:
        result.update(status="RESOURCE_STOPPED",resource_reason=str(exc))
    for name,factor in factors.items():
        np.savez_compressed(out/f"{name}-last-valid.npz",q=q[name],
                            time=base_time+counts[name]*coarse_dt/factor,dt=coarse_dt/factor)
    result["terminal_states"] = terminal_states(q,{n:c/factors[n] for n,c in counts.items()},
                                                grid,base_time,coarse_dt)
    result.update(steps=counts,seconds=time.perf_counter()-start,peak_rss_bytes=peak_rss(),
                  all_trajectories_reached_target=all(counts[n] == coarse_steps*f for n,f in factors.items()))
    if result["seconds"] > 600 or result["peak_rss_bytes"] > 512*1024**2:
        result["status"] = "RESOURCE_STOPPED"
    text = json.dumps(result,indent=2,allow_nan=False)+"\n"
    if sum(p.stat().st_size for p in out.iterdir())+len(text.encode()) > 64*1024**2:
        raise ResourceStopped("64 MiB output limit")
    (out/"result.json").write_text(text,encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--refine",action="store_true",help="native dt, dt/2, dt/4 restart diagnostic")
    args = parser.parse_args()
    result = (run_refinement if args.refine else run)(args.out,args.snapshot)
    print(json.dumps({k:v for k,v in result.items() if k != "samples"}),flush=True)

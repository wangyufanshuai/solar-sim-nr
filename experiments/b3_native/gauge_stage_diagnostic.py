"""S3 same-state stage comparison and one-ULP initial perturbation sensitivity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from science.numerical_relativity import BSSNState3D, FOCCZ4State3D
from science.numerical_relativity.fo_ccz4 import FOCCZ4_VARIABLE_ORDER
from science.numerical_relativity.fo_ccz4_checkpoint import SYMMETRIC
from experiments.b1_gauge_wave import run_b1
from experiments.b1_gauge_wave.harmonic import harmonic_fo_ccz4_rhs
from kernel.native_evolve import slots, state_from_slots, base_fields, make_rhs, rk4
from experiments.b3_native.long_diagnostics import checked_rhs, validate_slots, ScientificFailure, peak_rss

SAMPLE_TIMES=(0,1,10,15,17,18,20,22)


def pack_rhs(r):
    """Canonical independent symmetric components and the evolved auxiliary RHS."""
    return np.stack([*[r.h[...,i,j] for i,j in SYMMETRIC],r.ln_lapse,
        *[r.shift[...,i] for i in range(3)],r.ln_phi,*[r.a[...,i,j] for i,j in SYMMETRIC],r.trace_k,r.theta,
        *[r.conformal_gamma[...,i] for i in range(3)],*[r.b[...,i] for i in range(3)],
        *[r.lapse_gradient[...,i] for i in range(3)],*[r.shift_gradient[...,k,i] for k in range(3) for i in range(3)],
        *[r.metric_gradient[...,k,i,j] for k in range(3) for i in range(3) for j in range(i,3)],
        *[r.conformal_factor_gradient[...,i] for i in range(3)]])


def python_rhs(q,g):
    return pack_rhs(harmonic_fo_ccz4_rhs(state_from_slots(q,g),gauge=run_b1.GAUGE,config=run_b1.CONFIG))


def compare_arrays(reference,native):
    if reference.shape!=native.shape or reference.shape[0]!=58:
        raise ValueError("expected matching 58-slot arrays")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(native)):
        raise ValueError("comparison requires finite arrays")
    magnitude=np.max(np.abs(reference),axis=(1,2,3))
    differences=np.max(np.abs(native-reference),axis=(1,2,3))
    scale=float(np.max(magnitude))
    rows=[]
    for name,size,diff in zip(FOCCZ4_VARIABLE_ORDER,magnitude,differences):
        active=bool(size>1e-10*scale)
        rows.append(dict(slot=name,reference_max=float(size),absolute_difference=float(diff),
                         relative_difference=float(diff/size) if size else None,active=active))
    ratios=[r["relative_difference"] for r in rows if r["active"]]
    return dict(global_max_absolute_difference=float(np.max(differences)),reference_scale=scale,
                worst_active_relative_difference=max(ratios) if ratios else 0.,
                active_slots_pass_relative_1e_minus12=all(v<=1e-12 for v in ratios),slots=rows)


def compare_stages(q,g,dt,native_rhs):
    result=[]
    k1=native_rhs(q)
    inputs=[q,q+.5*dt*k1]
    k2=native_rhs(inputs[1]); inputs.append(q+.5*dt*k2)
    k3=native_rhs(inputs[2]); inputs.append(q+dt*k3)
    for stage,state in enumerate(inputs,1):
        result.append(dict(stage=stage,**compare_arrays(python_rhs(state,g),native_rhs(state))))
    return result


def analytic_error(q,g,t):
    fields=base_fields(q)
    exact=BSSNState3D.gauge_wave(g,amplitude=.1,time=float(t))
    return max(float(np.max(np.abs(fields[f]-getattr(exact,f)))) for f in run_b1.FIELDS)/.1


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter(); deadline=start+600
    g=run_b1.grid(32,4); dt=1/128
    q0=slots(FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(g,amplitude=.1)))
    trajectories={"baseline":q0.copy(),"one_ulp":q0.copy()}
    index=(0,8,0,0); original=float(trajectories["one_ulp"][index])
    trajectories["one_ulp"][index]=np.nextafter(original,np.inf)
    seed_delta=float(trajectories["one_ulp"][index]-original)
    raw_rhs=make_rhs(spacing=g.spacing,order=4,gauge=run_b1.GAUGE,config=run_b1.CONFIG,harmonic=True,threads=1)
    rhs={name:checked_rhs(raw_rhs) for name in trajectories}
    alive={name:True for name in trajectories}; steps={name:0 for name in trajectories}
    data=dict(status="RUNNING",settings=dict(shape=list(g.shape),dt=dt,amplitude=.1,target=22,threads=1),
              perturbation=dict(index=list(index),original=original,delta=seed_delta),samples=[],stage_comparisons=[],failures={},external_independent_reproduction=False)

    def save():
        data.update(seconds=time.perf_counter()-start,peak_rss_bytes=peak_rss(),steps=dict(steps))
        encoded=json.dumps(data,indent=1,allow_nan=False)+'\n'
        size=sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())
        if size+len(encoded.encode())>64*1024**2:
            raise RuntimeError("64 MiB output limit")
        (args.out/'result.json').write_text(encoded,encoding='utf-8')

    def sample(unit):
        row=dict(time=unit)
        for name,q in trajectories.items():
            if alive[name]: row[name]=dict(error=analytic_error(q,g,unit))
        if all(alive.values()):
            diff=np.max(np.abs(trajectories["one_ulp"]-trajectories["baseline"]),axis=(1,2,3))
            row["max_slot_difference"]=float(np.max(diff))
            row["slot_differences"]={key:float(v) for key,v in zip(FOCCZ4_VARIABLE_ORDER,diff)}
            row["amplification_vs_initial_max"]=float(np.max(diff)/seed_delta)
        data["samples"].append(row)
        if unit in SAMPLE_TIMES and alive["baseline"]:
            data["stage_comparisons"].append(dict(time=unit,stages=compare_stages(trajectories["baseline"],g,dt,raw_rhs)))
        if unit in (0,15,18,22) and alive["baseline"]:
            np.savez_compressed(args.out/f'baseline-t{unit}.npz',q=trajectories["baseline"],time=unit,dt=dt)
        save()
        print(json.dumps(dict(time=unit,seconds=round(data["seconds"],2),max_slot_difference=row.get("max_slot_difference"))),flush=True)

    try:
        sample(0)
        for unit in range(1,23):
            for _ in range(128):
                if time.perf_counter()>=deadline: raise TimeoutError("600 second runtime limit")
                for name,q in trajectories.items():
                    if not alive[name]: continue
                    try:
                        with np.errstate(over="raise",invalid="raise",divide="raise",under="ignore"):
                            candidate=rk4(q,dt,1,rhs[name]); validate_slots(candidate)
                        trajectories[name]=candidate; steps[name]+=1
                    except (ScientificFailure,FloatingPointError,np.linalg.LinAlgError) as exc:
                        alive[name]=False
                        data["failures"][name]=dict(message=str(exc),last_time=steps[name]*dt,failed_interval=[steps[name]*dt,(steps[name]+1)*dt])
            sample(unit)
            if data["peak_rss_bytes"]>512*1024**2: raise RuntimeError("512 MiB process limit")
            if not any(alive.values()): break
        data["status"]="COMPLETED" if all(alive.values()) else "SCIENTIFIC_FAILURE"
    except (TimeoutError,RuntimeError) as exc:
        data.update(status="RESOURCE_STOPPED",failure=str(exc))
    data["complete_sampled_stage_set"]=len(data["stage_comparisons"])==len(SAMPLE_TIMES)
    data["all_sampled_stages_pass_rhs_target"]=bool(data["stage_comparisons"]) and all(s["active_slots_pass_relative_1e_minus12"] for record in data["stage_comparisons"] for s in record["stages"])
    save()
    print(json.dumps({k:v for k,v in data.items() if k not in ('samples','stage_comparisons')}),flush=True)


if __name__=='__main__': main()

"""Explain S2-c secular lapse growth with centred-derivative parity null modes."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from science.numerical_relativity import BSSNState3D, FOCCZ4State3D, FOCCZ4Config, GaugeConfig, Grid3D
from science.numerical_relativity.bssn3d import _d1
from kernel.native_evolve import slots, make_rhs, rk4
from experiments.s2_periodic_evolution.robust_diagnostic import noisy_state
from experiments.b3_native.long_diagnostics import checked_rhs, validate_slots, peak_rss


def parity_modes(shape):
    if len(shape)!=3 or any(n%2 or n<2 for n in shape):
        raise ValueError("three even grid sizes required")
    index=np.indices(shape)
    for bits in itertools.product((0,1),repeat=3):
        yield bits,(-1.)**sum(b*index[axis] for axis,b in enumerate(bits))


def predict_lapse_slope(state):
    source=state.base.trace_k-2*state.base.theta
    coefficients=[dict(parity=list(bits),coefficient=float(np.mean(source*parity))) for bits,parity in parity_modes(source.shape)]
    return dict(coefficients=coefficients,predicted_rms_slope=2*float(np.sqrt(sum(row["coefficient"]**2 for row in coefficients))),
                constant_only_slope=2*abs(coefficients[0]["coefficient"]))


def null_derivative_checks():
    rows=[]
    for order in (2,4):
        grid=Grid3D((8,)*3,(0.,)*3,(1.,)*3,boundary="periodic",spatial_order=order)
        state=BSSNState3D.minkowski(grid,formulation="ccz4")
        for bits,p in parity_modes(grid.shape):
            error=max(float(np.max(np.abs(_d1(p,state,axis)))) for axis in range(3))
            rows.append(dict(order=order,parity=list(bits),max_derivative=error))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args(); args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream: stream.write('{}\n')
    start=time.perf_counter()
    prediction=predict_lapse_slope(noisy_state(8,4))
    saved=json.loads(Path('docs/long-diagnostics/summary.json').read_text())
    observed=saved['s2']['lapse_rms_linear_slope_t100_to_last']
    prediction.update(observed_rms_slope=observed,relative_difference=abs(prediction['predicted_rms_slope']-observed)/observed,
                      fit_to_initial_data_only=True,explanation_posthoc=True)
    g=Grid3D((8,)*3,(0.,)*3,(1.,)*3,boundary="periodic",spatial_order=4)
    base=BSSNState3D.minkowski(g,formulation="ccz4")
    p=dict(parity_modes(g.shape))[(1,1,1)]
    amplitude=1e-8; base.trace_k[:]=amplitude*p
    initial=FOCCZ4State3D.from_bssn_state(base); q=slots(initial)
    rhs=checked_rhs(make_rhs(spacing=g.spacing,order=4,gauge=GaugeConfig(theta_coupling=2.),config=FOCCZ4Config(kappa1=0.),threads=1))
    dt=.0125; completed=0
    result=dict(status="RUNNING",seed42_prediction=prediction,null_derivative_checks=null_derivative_checks(),checkerboard_amplitude=amplitude,dt=dt,rows=[],boundedness_proven=False)
    try:
        for unit in range(1,11):
            for _ in range(80):
                if time.perf_counter()-start>=600: raise TimeoutError('600 second limit')
                with np.errstate(over='raise',invalid='raise',divide='raise',under='ignore'):
                    candidate=rk4(q,dt,1,rhs); validate_slots(candidate)
                q=candidate; completed+=1
            t=completed*dt
            result['rows'].append(dict(time=t,
                max_lnalpha_error=float(np.max(np.abs(q[6]-(-2*amplitude*p*t)))),
                max_lnphi_error=float(np.max(np.abs(q[10]-amplitude*p*t/3))),
                measured_lnalpha_rms=float(np.sqrt(np.mean(q[6]**2))),predicted_lnalpha_rms=2*amplitude*t))
            print(json.dumps(result['rows'][-1]),flush=True)
        result['status']='COMPLETED'
    except TimeoutError as exc:
        result.update(status='RESOURCE_STOPPED',failure=str(exc))
    result.update(seconds=time.perf_counter()-start,peak_rss_bytes=peak_rss(),last_time=completed*dt)
    result['analytic_checkerboard_passed']=result['status']=='COMPLETED' and all(max(row['max_lnalpha_error'],row['max_lnphi_error'])<=1e-11 for row in result['rows'])
    result['all_parities_zero_derivative']=all(row['max_derivative']==0 for row in result['null_derivative_checks'])
    encoded=json.dumps(result,indent=1,allow_nan=False)+'\n'
    if len(encoded.encode())>64*1024**2: raise RuntimeError('64 MiB output limit')
    args.out.write_text(encoded,encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','null_derivative_checks')}),flush=True)


if __name__=='__main__': main()

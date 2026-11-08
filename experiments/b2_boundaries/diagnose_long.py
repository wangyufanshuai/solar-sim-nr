"""Post-hoc diagnostic for the failed B2-c run 01 (not a registered test; counts toward no gate).

Repeats the B2-c setup (sine TT wave, exact incoming data on all faces, Nx = 17) with the same
cube, dt and boundary data as run_b2.evolve, but appends one JSON line about every 20 steps:
error ÷ A per slot group, the slot and grid point of the largest error, min lapse / chi, the
largest SAT term and basis condition number. On an exception it records whether the state was
still finite, so a blow-up can be told apart from a linear-algebra failure on finite data.

    python -B -m experiments.b2_boundaries.diagnose_long --out outputs/b2c-diag --until 3
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from .linear_wave_slots import wave_slots
from .run_b2 import AMPLITUDE, CFL_CAP, LONG, TRANSVERSE, PooledB2Cube, make_grid

GROUPS = dict(h=(0, 6), ln_alpha=(6, 7), beta=(7, 10), ln_phi=(10, 11), a_tilde=(11, 17), trace_k=(17, 18),
              theta=(18, 19), gamma_hat=(19, 22), b=(22, 25), a=(25, 28), b_shift=(28, 37), d=(37, 55), p=(55, 58))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--until", type=float, default=LONG["final_time"])
    parser.add_argument("--every", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "trace.jsonl").open("a", encoding="utf-8")

    nx = LONG["points"]
    grid = make_grid(nx)
    cube = PooledB2Cube(grid)
    x = grid.coordinates[0][:, 0, 0]
    flat = wave_slots(x, amplitude=0.0, time=0.0, transverse_shape=(TRANSVERSE, TRANSVERSE))
    rate = cube.cfl(cube.state(flat), 1e-9) / 1e-9
    steps_per_unit = math.ceil(rate / (0.95 * CFL_CAP))
    dt = 1.0 / steps_per_unit

    def exact(t):
        return wave_slots(x, amplitude=AMPLITUDE, time=t, transverse_shape=(TRANSVERSE, TRANSVERSE))

    def boundary_for(_step):
        return lambda half_step: cube.boundary_targets(exact(0.5 * dt * half_step))

    def emit(row):
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)

    q = exact(0.0)
    emit(dict(event="start", nx=nx, dt=dt, steps_per_unit=steps_per_unit))
    start = time.perf_counter()
    for step in range(int(round(args.until * steps_per_unit))):
        try:
            q_next, stages = cube.step(q, step, dt, boundary_for(step))
        except Exception as error:  # record and stop; the diagnostic is about this failure
            emit(dict(event="error", step=step, t=step * dt, finite=bool(np.all(np.isfinite(q))),
                      max_abs=float(np.nanmax(np.abs(q))), message=repr(error),
                      where=traceback.format_exc().splitlines()[-3:]))
            np.save(args.out / "state-before-error.npy", q)
            return
        q = q_next
        if (step + 1) % args.every == 0:
            t = (step + 1) * dt
            diff = np.abs(q - exact(t)) / AMPLITUDE
            worst = np.unravel_index(int(np.nanargmax(diff)), diff.shape)
            emit(dict(event="sample", step=step + 1, t=t, seconds=round(time.perf_counter() - start),
                      error={k: float(np.max(diff[a:b])) for k, (a, b) in GROUPS.items()},
                      worst_slot=int(worst[0]), worst_point=[int(i) for i in worst[1:]],
                      min_lapse=stages[-1]["min_lapse"], min_chi=stages[-1]["min_chi"],
                      sat_max_abs=max(r["sat_max_abs"] for r in stages),
                      max_basis_condition=max(r["max_basis_condition"] for r in stages)))
    emit(dict(event="done", t=args.until))


if __name__ == "__main__":
    main()

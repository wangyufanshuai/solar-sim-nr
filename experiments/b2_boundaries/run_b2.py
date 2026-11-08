"""B2: SBP(4,2) + characteristic SAT outer boundaries on the exact linearised TT wave.

Uses the existing nonlinear SBP volume equations and the stage-local characteristic SAT of
science/numerical_relativity/fo_ccz4_sbp_nonlinear.py (incoming characteristic data supplied on
all six faces, profile gauge, tau_inverse = 1). B2Cube only removes that driver's 9..17-point
reference cap; the equations, SAT and RK4 stage times are unchanged.

    python -B -m experiments.b2_boundaries.run_b2 converge --out outputs/b2
    python -B -m experiments.b2_boundaries.run_b2 reflect --nx 33 --out outputs/b2
    python -B -m experiments.b2_boundaries.run_b2 long --out outputs/b2
    python -B -m experiments.b2_boundaries.run_b2 summarize --out outputs/b2

ACCEPTANCE and docs/B2_SAT_BOUNDARIES.md were committed before the first run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from science.numerical_relativity import Grid3D
from science.numerical_relativity.fo_ccz4_sbp_nonlinear import CONFIG, GAUGE, PROFILE, NonlinearSBPCube
from science.numerical_relativity.fo_ccz4_sbp_volume import SBPVolumeOperator

from .linear_wave_slots import wave_slots
from .parallel_sat import ParallelSATMixin

AMPLITUDE = 1.0e-8
TRANSVERSE = 9
CFL_CAP = 0.25  # the driver's own cap on dt * sum(max char. speed / dx); we use <= 0.95 of it
CONVERGE = dict(points=(17, 33, 65), final_time=1.0)
REFLECT = dict(points=(17, 33, 65), final_time=1.0)
LONG = dict(points=17, final_time=10.0)
ACCEPTANCE = dict(
    converge_min_order=2.6,
    converge_finest_max_error=1.0e-2,
    reflect_min_order=2.0,
    reflect_finest_max=1.0e-3,
    long_max_late_log_slope=0.02,
)


class B2Cube(NonlinearSBPCube):
    """NonlinearSBPCube without the 9..17-point reference-grid cap (nothing else changes)."""

    def __init__(self, grid, *, gauge=GAUGE, config=CONFIG, profile=PROFILE):
        self.volume = SBPVolumeOperator(grid)
        profile.validate_formulation(gauge, config)
        self.gauge, self.config, self.profile = gauge, config, profile


class PooledB2Cube(ParallelSATMixin, B2Cube):
    """B2Cube with the process-parallel SAT penalties (bitwise-identical RHS)."""


def make_grid(nx):
    width = (TRANSVERSE - 1) / (nx - 1)  # same spacing in y and z as in x
    return Grid3D((nx, TRANSVERSE, TRANSVERSE), (0.0, 0.0, 0.0), (1.0, width, width),
                  boundary="sbp-volume", spatial_order=4)


def evolve(nx, final_time, *, profile, outgoing_free, samples=None):
    """RK4 with SAT targets from the exact solution at each stage time.

    outgoing_free: the x = 1 face receives the flat background as incoming data (radiative
    test) instead of the exact solution.
    """
    grid = make_grid(nx)
    cube = PooledB2Cube(grid)
    x = grid.coordinates[0][:, 0, 0]
    flat = wave_slots(x, amplitude=0.0, time=0.0, transverse_shape=(TRANSVERSE, TRANSVERSE))
    # Largest characteristic speed per unit dt from the driver's own CFL formula (profile gauge
    # speeds exceed 1); steps per time unit is an integer so samples land on whole times.
    rate = cube.cfl(cube.state(flat), 1e-9) / 1e-9
    steps_per_unit = math.ceil(rate / (0.95 * CFL_CAP))
    dt = 1.0 / steps_per_unit

    def exact(t):
        return wave_slots(x, amplitude=AMPLITUDE, time=t, transverse_shape=(TRANSVERSE, TRANSVERSE), profile=profile)

    def boundary_for(step_index):
        def targets(half_step):
            ref = exact(0.5 * dt * half_step)
            faces = list(cube.boundary_targets(ref))
            if outgoing_free:
                faces[1] = cube.boundary_targets(flat)[1]  # axis 0, side 1 (x = 1)
            return tuple(faces)
        return targets

    q = exact(0.0)
    total = int(round(final_time * steps_per_unit))
    trace = []
    start = time.perf_counter()
    for step in range(total):
        q, _ = cube.step(q, step, dt, boundary_for(step))
        t = (step + 1) * dt
        if samples and abs(t - round(t)) < 1e-12 and int(round(t)) in samples:
            trace.append([float(round(t)), float(np.max(np.abs(q - exact(t)))) / AMPLITUDE])
    return dict(grid=grid, cube=cube, q=q, exact=exact, flat=flat, dt=dt, steps=total,
                seconds=time.perf_counter() - start, trace=trace)


def run_converge(nx, out):
    r = evolve(nx, CONVERGE["final_time"], profile="sine", outgoing_free=False)
    err = float(np.max(np.abs(r["q"] - r["exact"](CONVERGE["final_time"])))) / AMPLITUDE
    write(out / f"converge-n{nx}.json", dict(nx=nx, dt=r["dt"], steps=r["steps"], error=err,
                                              finite=math.isfinite(err), seconds=r["seconds"]))
    print(json.dumps(dict(nx=nx, error=err, seconds=round(r["seconds"]))), flush=True)


def run_reflect(nx, out):
    r = evolve(nx, REFLECT["final_time"], profile="pulse", outgoing_free=True)
    # At t = 1 the packet centre is at x = 1.5, so the exact field in [0, 1] is flat to
    # exp(-(0.5/0.12)^2) ~ 3e-8 of A; anything left is reflected or spurious.
    residual = float(np.max(np.abs(r["q"] - r["flat"]))) / AMPLITUDE
    write(out / f"reflect-n{nx}.json", dict(nx=nx, dt=r["dt"], steps=r["steps"], reflection=residual,
                                            finite=math.isfinite(residual), seconds=r["seconds"]))
    print(json.dumps(dict(nx=nx, reflection=residual, seconds=round(r["seconds"]))), flush=True)


def run_long(out):
    units = list(range(1, int(LONG["final_time"]) + 1))
    r = evolve(LONG["points"], LONG["final_time"], profile="sine", outgoing_free=False, samples=units)
    write(out / "long.json", dict(nx=LONG["points"], dt=r["dt"], samples=r["trace"], seconds=r["seconds"]))
    print(json.dumps(dict(final=r["trace"][-1], seconds=round(r["seconds"]))), flush=True)


def orders(values, ns):
    return [math.log(a / b) / math.log((n2 - 1) / (n1 - 1)) for a, b, n1, n2 in zip(values, values[1:], ns, ns[1:])]


def summarize(out):
    a = ACCEPTANCE
    ns = CONVERGE["points"]
    conv = [json.loads((out / f"converge-n{n}.json").read_text()) for n in ns]
    errs = [c["error"] for c in conv]
    p = orders(errs, ns)
    refl = [json.loads((out / f"reflect-n{n}.json").read_text()) for n in REFLECT["points"]]
    rs = [c["reflection"] for c in refl]
    pr = orders(rs, REFLECT["points"])
    long = json.loads((out / "long.json").read_text())
    t = np.array([s[0] for s in long["samples"]]); e = np.array([s[1] for s in long["samples"]])
    complete = bool(np.all(np.isfinite(e))) and t[-1] >= LONG["final_time"]
    late = t >= LONG["final_time"] / 2
    slope = float(np.polyfit(t[late], np.log(e[late]), 1)[0]) if complete else math.inf
    checks = dict(
        converge=dict(errors=errs, observed_orders=p,
                      passed=all(c["finite"] for c in conv) and all(x >= a["converge_min_order"] for x in p)
                      and errs[-1] <= a["converge_finest_max_error"]),
        reflect=dict(reflections=rs, observed_orders=pr,
                     passed=all(c["finite"] for c in refl) and all(x >= a["reflect_min_order"] for x in pr)
                     and rs[-1] <= a["reflect_finest_max"]),
        long=dict(complete=complete, late_log_slope=slope, final_error=float(e[-1]),
                  passed=complete and slope <= a["long_max_late_log_slope"]),
    )
    result = dict(acceptance=a, checks=checks, passed=all(c["passed"] for c in checks.values()))
    write(out / "summary.json", result)
    print(json.dumps(result, indent=1))


def write(path, payload):
    payload = {k: v for k, v in payload.items() if k not in ("grid", "cube", "q", "exact", "flat")}
    payload.update(python=platform.python_version(), numpy=np.__version__, platform=platform.platform())
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("converge", "reflect", "long", "summarize"))
    parser.add_argument("--nx", type=int, default=33)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.part == "converge":
        run_converge(args.nx, args.out)
    elif args.part == "reflect":
        run_reflect(args.nx, args.out)
    elif args.part == "long":
        run_long(args.out)
    else:
        summarize(args.out)


if __name__ == "__main__":
    main()

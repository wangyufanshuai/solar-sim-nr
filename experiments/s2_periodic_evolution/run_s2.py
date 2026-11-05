"""S2 runner: periodic FO-CCZ4 evolution, convergence against an exact solution, robustness.

Usage (repository root):
    python -B -m experiments.s2_periodic_evolution.run_s2 spatial --order 4 --out outputs/s2
    python -B -m experiments.s2_periodic_evolution.run_s2 time --out outputs/s2
    python -B -m experiments.s2_periodic_evolution.run_s2 robust --order 2 --out outputs/s2
    python -B -m experiments.s2_periodic_evolution.run_s2 summarize --out outputs/s2

Acceptance rules are fixed in ACCEPTANCE and docs/S2_PERIODIC_EVOLUTION.md; they were
committed before the first run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from science.numerical_relativity import FOCCZ4Config, FOCCZ4State3D, GaugeConfig
from science.numerical_relativity.constraints3d import constraint_report_3d
from science.numerical_relativity.fo_ccz4_rhs import (
    advance_fo_ccz4_rk4_3d,
    fo_ccz4_robust_stability_campaign_3d,
)

from .linear_wave import field_errors, linear_wave_state, periodic_grid

GAUGE = GaugeConfig(theta_coupling=2.0)
CONFIG = FOCCZ4Config()
FINAL_TIME = 1.0
SPATIAL = dict(points=(16, 32, 64), cfl=0.1, amplitude=1.0e-8)
TEMPORAL = dict(points=32, order=4, cfls=(0.8, 0.4, 0.2), reference_cfl=0.1, amplitude=1.0e-6)
ROBUST = dict(points=8, cfl=0.1, final_time=10.0, every=80, amplitude=1.0e-10, seed=42, growth_limit=10.0)
ACCEPTANCE = dict(
    spatial_order_window={2: (1.7, 2.3), 4: (3.7, 4.3)},
    fd4_finest_max_error=1.0e-3,
    time_order_window=(3.7, 4.3),
    robust_growth_limit=10.0,
)
TT_FIELDS = (("h", (1, 1)), ("a", (1, 1)))


def evolve(nx, order, cfl, amplitude):
    grid = periodic_grid(nx, order=order)
    steps = int(round(FINAL_TIME / (cfl / nx)))
    dt = FINAL_TIME / steps
    state = FOCCZ4State3D.from_bssn_state(linear_wave_state(grid, amplitude=amplitude))
    start = time.perf_counter()
    state = advance_fo_ccz4_rk4_3d(state, dt, steps=steps, gauge=GAUGE, config=CONFIG)
    return grid, state, dict(nx=nx, order=order, cfl=cfl, dt=dt, steps=steps,
                             seconds=time.perf_counter() - start)


def tt_error(state, other, amplitude):
    return max(
        float(np.max(np.abs(getattr(state, f)[..., i, j] - getattr(other, f)[..., i, j])))
        for f, (i, j) in TT_FIELDS
    ) / amplitude


def run_spatial(order, out):
    rows = []
    for nx in SPATIAL["points"]:
        grid, state, meta = evolve(nx, order, SPATIAL["cfl"], SPATIAL["amplitude"])
        exact = linear_wave_state(grid, amplitude=SPATIAL["amplitude"], time=FINAL_TIME)
        constraints = constraint_report_3d(state.base)
        rows.append(dict(meta, tt_error=tt_error(state.base, exact, SPATIAL["amplitude"]),
                         field_errors=field_errors(state.base, exact, SPATIAL["amplitude"]),
                         constraint_norms={k: float(v) for k, v in constraints.norms.items()},
                         finite=bool(all(np.all(np.isfinite(getattr(state.base, f)))
                                         for f in ("chi", "h", "a", "trace_k", "lapse")))))
        print(json.dumps({k: rows[-1][k] for k in ("nx", "order", "steps", "tt_error", "seconds")}), flush=True)
    write(out / f"spatial-fd{order}.json", dict(part="spatial", order=order, settings=SPATIAL, rows=rows))


def run_time(out):
    p = TEMPORAL
    _, reference, ref_meta = evolve(p["points"], p["order"], p["reference_cfl"], p["amplitude"])
    rows = []
    for cfl in p["cfls"]:
        _, state, meta = evolve(p["points"], p["order"], cfl, p["amplitude"])
        rows.append(dict(meta, tt_difference=tt_error(state.base, reference.base, p["amplitude"])))
        print(json.dumps({k: rows[-1][k] for k in ("cfl", "steps", "tt_difference", "seconds")}), flush=True)
    write(out / "time-fd4.json", dict(part="time", settings=p, reference=ref_meta, rows=rows))


def run_robust(order, out):
    p = ROBUST
    dt = p["cfl"] / p["points"]
    total = int(round(p["final_time"] / dt))
    start = time.perf_counter()
    report = fo_ccz4_robust_stability_campaign_3d(
        points=p["points"], dt=dt, checkpoints=tuple(range(p["every"], total + 1, p["every"])),
        amplitude=p["amplitude"], seed=p["seed"], spatial_order=order,
        growth_limit=p["growth_limit"], gauge=GAUGE)
    samples = [dict(step=s.step, time=s.time, growth_ratio=s.growth_ratio,
                    hamiltonian_l2=s.hamiltonian_l2, momentum_l2=s.momentum_l2,
                    finite_positive_state=s.finite_positive_state) for s in report.samples]
    write(out / f"robust-fd{order}.json", dict(part="robust", order=order, settings=p, dt=dt,
                                                passed=report.passed, samples=samples,
                                                seconds=time.perf_counter() - start))
    print(json.dumps(dict(order=order, passed=report.passed,
                          max_growth=max(s["growth_ratio"] for s in samples))), flush=True)


def orders(errors, factor=2.0):
    return [math.log(a / b, factor) for a, b in zip(errors, errors[1:])]


def summarize(out):
    a = ACCEPTANCE
    result = dict(acceptance=a, checks={})
    for order in (2, 4):
        data = json.loads((out / f"spatial-fd{order}.json").read_text())
        errs = [r["tt_error"] for r in data["rows"]]
        p = orders(errs)
        lo, hi = a["spatial_order_window"][order]
        ok = all(lo <= x <= hi for x in p) and all(r["finite"] for r in data["rows"])
        if order == 4:
            ok = ok and errs[-1] <= a["fd4_finest_max_error"]
        result["checks"][f"spatial_fd{order}"] = dict(errors=errs, observed_orders=p, passed=ok)
    data = json.loads((out / "time-fd4.json").read_text())
    diffs = [r["tt_difference"] for r in data["rows"]]
    p = orders(diffs)
    lo, hi = a["time_order_window"]
    result["checks"]["time_fd4"] = dict(differences=diffs, observed_orders=p,
                                        passed=all(lo <= x <= hi for x in p))
    for order in (2, 4):
        data = json.loads((out / f"robust-fd{order}.json").read_text())
        result["checks"][f"robust_fd{order}"] = dict(
            max_growth=max(s["growth_ratio"] for s in data["samples"]),
            final_growth=data["samples"][-1]["growth_ratio"], passed=data["passed"])
    result["passed"] = all(c["passed"] for c in result["checks"].values())
    write(out / "summary.json", result)
    print(json.dumps(result, indent=1))


def write(path, payload):
    payload = dict(payload, python=platform.python_version(), numpy=np.__version__,
                   platform=platform.platform())
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("spatial", "time", "robust", "summarize"))
    parser.add_argument("--order", type=int, choices=(2, 4), default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.part == "spatial":
        run_spatial(args.order, args.out)
    elif args.part == "time":
        run_time(args.out)
    elif args.part == "robust":
        run_robust(args.order, args.out)
    else:
        summarize(args.out)


if __name__ == "__main__":
    main()

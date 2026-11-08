"""B1: FO-CCZ4 with harmonic slicing on the Apples-with-Apples nonlinear gauge wave.

Exact solution: ds^2 = H(-dt^2 + dx^2) + dy^2 + dz^2,  H = 1 - A sin(2 pi (x - t)),
zero shift, harmonic slicing d_t alpha = -alpha^2 K (experiments/b1_gauge_wave/harmonic.py, f = 1);
BSSNState3D.gauge_wave(time=t) gives the exact slice. Shift evolution is off
(shift_strength = 0) and kappa1 = 0, so the test exercises the undamped system.

    python -B -m experiments.b1_gauge_wave.run_b1 converge --amplitude 0.01 --order 4 --out outputs/b1
    python -B -m experiments.b1_gauge_wave.run_b1 long --amplitude 0.01 --out outputs/b1
    python -B -m experiments.b1_gauge_wave.run_b1 summarize --out outputs/b1

ACCEPTANCE and docs/B1_GAUGE_WAVE.md were committed before the first run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from science.numerical_relativity import BSSNState3D, FOCCZ4Config, FOCCZ4State3D, GaugeConfig, Grid3D

from .harmonic import advance_harmonic_rk4

GAUGE = GaugeConfig(theta_coupling=2.0, lapse_coefficient=1.0)
CONFIG = FOCCZ4Config(kappa1=0.0, shift_strength=0.0)
CONVERGE = dict(points=(16, 32, 64), cfl=0.1, final_time=1.0)
LONG = dict(points=32, order=4, cfl=0.25, final_time=100.0)
FIELDS = ("chi", "h", "trace_k", "a", "theta", "conformal_gamma", "lapse")
ACCEPTANCE = dict(
    spatial_order_window={2: (1.7, 2.3), 4: (3.7, 4.3)},
    fd4_finest_max_error=1.0e-3,
    long_gated_amplitude=0.01,
    long_max_late_log_slope=0.02,
    long_max_final_error=1.0,
)


def grid(nx, order):
    return Grid3D((nx, 5, 5), (0.0, 0.0, 0.0), (1.0, 5 / nx, 5 / nx),
                  boundary="periodic", spatial_order=order)


def error(state, exact, amplitude):
    """Max-norm error over the evolved base fields, divided by the wave amplitude."""
    return max(float(np.max(np.abs(getattr(state, f) - getattr(exact, f)))) for f in FIELDS) / amplitude


def advance(state, dt, steps):
    return advance_harmonic_rk4(state, dt, steps=steps, gauge=GAUGE, config=CONFIG)


def run_converge(amplitude, order, out):
    rows = []
    for nx in CONVERGE["points"]:
        g = grid(nx, order)
        steps = int(round(CONVERGE["final_time"] * nx / CONVERGE["cfl"]))
        dt = CONVERGE["final_time"] / steps
        start = time.perf_counter()
        state = advance(FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(g, amplitude=amplitude)), dt, steps)
        exact = BSSNState3D.gauge_wave(g, amplitude=amplitude, time=CONVERGE["final_time"])
        e = error(state.base, exact, amplitude)
        rows.append(dict(nx=nx, steps=steps, dt=dt, error=e, finite=math.isfinite(e),
                         seconds=time.perf_counter() - start))
        print(json.dumps(rows[-1]), flush=True)
    write(out / f"converge-a{amplitude}-fd{order}.json", dict(amplitude=amplitude, order=order, settings=CONVERGE, rows=rows))


def run_long(amplitude, out):
    p = LONG
    g = grid(p["points"], p["order"])
    per_unit = int(round(p["points"] / p["cfl"]))
    dt = 1.0 / per_unit
    state = FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(g, amplitude=amplitude))
    samples, start = [], time.perf_counter()
    for unit in range(1, int(p["final_time"]) + 1):
        state = advance(state, dt, per_unit)
        e = error(state.base, BSSNState3D.gauge_wave(g, amplitude=amplitude, time=float(unit)), amplitude)
        samples.append([float(unit), e])
        write(out / f"long-a{amplitude}.json", dict(amplitude=amplitude, settings=p, dt=dt, samples=samples,
                                                    seconds=time.perf_counter() - start))
        if not math.isfinite(e):
            break
    print(json.dumps(dict(amplitude=amplitude, final=samples[-1])), flush=True)


def orders(errors):
    return [math.log2(a / b) for a, b in zip(errors, errors[1:])]


def summarize(out):
    a = ACCEPTANCE
    checks = {}
    for amplitude in (0.01, 0.1):
        for order in (2, 4):
            d = json.loads((out / f"converge-a{amplitude}-fd{order}.json").read_text())
            errs = [r["error"] for r in d["rows"]]
            p = orders(errs)
            lo, hi = a["spatial_order_window"][order]
            ok = all(r["finite"] for r in d["rows"]) and all(lo <= x <= hi for x in p)
            if order == 4:
                ok = ok and errs[-1] <= a["fd4_finest_max_error"]
            checks[f"converge_a{amplitude}_fd{order}"] = dict(errors=errs, observed_orders=p, passed=ok)
    for amplitude in (0.01, 0.1):
        d = json.loads((out / f"long-a{amplitude}.json").read_text())
        t = np.array([s[0] for s in d["samples"]])
        e = np.array([s[1] for s in d["samples"]])
        complete = bool(np.all(np.isfinite(e)) and t[-1] >= LONG["final_time"])
        late = t >= LONG["final_time"] / 2
        slope = float(np.polyfit(t[late], np.log(e[late]), 1)[0]) if complete else math.inf
        entry = dict(complete=complete, late_log_slope=slope, final_error=float(e[-1]),
                     error_at={str(int(x)): float(y) for x, y in zip(t, e) if int(x) in (1, 10, 50, 100)})
        if amplitude == a["long_gated_amplitude"]:
            entry["passed"] = bool(complete and slope <= a["long_max_late_log_slope"]
                                   and e[-1] <= a["long_max_final_error"])  # numpy bool is not JSON
        else:
            entry["reported_only"] = True
        checks[f"long_a{amplitude}"] = entry
    result = dict(acceptance=a, checks=checks,
                  passed=all(c["passed"] for c in checks.values() if "passed" in c))
    write(out / "summary.json", result)
    print(json.dumps(result, indent=1))


def write(path, payload):
    payload = dict(payload, python=platform.python_version(), numpy=np.__version__, platform=platform.platform())
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("converge", "long", "summarize"))
    parser.add_argument("--amplitude", type=float, default=0.01)
    parser.add_argument("--order", type=int, choices=(2, 4), default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.part == "converge":
        run_converge(args.amplitude, args.order, args.out)
    elif args.part == "long":
        run_long(args.amplitude, args.out)
    else:
        summarize(args.out)


if __name__ == "__main__":
    main()

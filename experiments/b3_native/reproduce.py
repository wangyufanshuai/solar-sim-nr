"""B3 step 2: rerun recorded S2 and B1 cases with the native RK4 and compare with the records.

    python -B -m experiments.b3_native.reproduce --out outputs/b3-repro
Reference files: outputs/s2-20261105-01/*.json and outputs/b1-20261108-01/*.json (Python runs).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from science.numerical_relativity import BSSNState3D, FOCCZ4State3D
from experiments.s2_periodic_evolution import run_s2
from experiments.s2_periodic_evolution.linear_wave import linear_wave_state, periodic_grid
from experiments.b1_gauge_wave import run_b1
from kernel.native_evolve import base_fields, make_rhs, rk4, slots

S2_REF = Path("outputs/s2-20261105-01")
B1_REF = Path("outputs/b1-20261108-01")


def tt_error(fields, exact, amplitude):
    return max(float(np.max(np.abs(fields[f][..., i, j] - getattr(exact, f)[..., i, j])))
               for f, (i, j) in run_s2.TT_FIELDS) / amplitude


def s2_evolve(nx, order, cfl, amplitude):
    grid = periodic_grid(nx, order=order)
    steps = int(round(run_s2.FINAL_TIME / (cfl / nx)))
    dt = run_s2.FINAL_TIME / steps
    q = slots(FOCCZ4State3D.from_bssn_state(linear_wave_state(grid, amplitude=amplitude)))
    start = time.perf_counter()
    q = rk4(q, dt, steps, make_rhs(spacing=grid.spacing, order=order, gauge=run_s2.GAUGE, config=run_s2.CONFIG))
    return grid, base_fields(q), time.perf_counter() - start


def s2(out):
    rows = []
    for order in (2, 4):
        ref = json.loads((S2_REF / f"spatial-fd{order}.json").read_text())["rows"]
        for r in ref:
            grid, fields, seconds = s2_evolve(r["nx"], order, run_s2.SPATIAL["cfl"], run_s2.SPATIAL["amplitude"])
            exact = linear_wave_state(grid, amplitude=run_s2.SPATIAL["amplitude"], time=run_s2.FINAL_TIME)
            e = tt_error(fields, exact, run_s2.SPATIAL["amplitude"])
            rows.append(dict(case="S2-a", order=order, nx=r["nx"], native=e, python=r["tt_error"],
                             native_s=seconds, python_s=r["seconds"]))
            print(json.dumps(rows[-1]), flush=True)
    p = run_s2.TEMPORAL
    ref = json.loads((S2_REF / "time-fd4.json").read_text())
    _, reference, _ = s2_evolve(p["points"], p["order"], p["reference_cfl"], p["amplitude"])
    for r in ref["rows"]:
        _, fields, seconds = s2_evolve(p["points"], p["order"], r["cfl"], p["amplitude"])
        diff = max(float(np.max(np.abs(fields[f][..., i, j] - reference[f][..., i, j])))
                   for f, (i, j) in run_s2.TT_FIELDS) / p["amplitude"]
        rows.append(dict(case="S2-b", order=4, cfl=r["cfl"], native=diff, python=r["tt_difference"],
                         native_s=seconds, python_s=r["seconds"]))
        print(json.dumps(rows[-1]), flush=True)
    return rows


def b1_rhs(grid, order):
    return make_rhs(spacing=grid.spacing, order=order, gauge=run_b1.GAUGE, config=run_b1.CONFIG, harmonic=True)


def b1_error(fields, exact, amplitude):
    return max(float(np.max(np.abs(fields[f] - getattr(exact, f)))) for f in run_b1.FIELDS) / amplitude


def b1(out, long=True):
    rows = []
    c = run_b1.CONVERGE
    for amplitude in (0.01, 0.1):
        for order in (2, 4):
            ref = json.loads((B1_REF / f"converge-a{amplitude}-fd{order}.json").read_text())["rows"]
            for r in ref:
                g = run_b1.grid(r["nx"], order)
                q = slots(FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(g, amplitude=amplitude)))
                start = time.perf_counter()
                q = rk4(q, r["dt"], r["steps"], b1_rhs(g, order))
                seconds = time.perf_counter() - start
                e = b1_error(base_fields(q), BSSNState3D.gauge_wave(g, amplitude=amplitude, time=c["final_time"]), amplitude)
                rows.append(dict(case="B1-a", amplitude=amplitude, order=order, nx=r["nx"], native=e,
                                 python=r["error"], native_s=seconds, python_s=r["seconds"]))
                print(json.dumps(rows[-1]), flush=True)
    if long:
        p = run_b1.LONG
        ref = json.loads((B1_REF / "long-a0.01.json").read_text())
        g = run_b1.grid(p["points"], p["order"])
        per_unit = int(round(p["points"] / p["cfl"]))
        q = slots(FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(g, amplitude=0.01)))
        rhs, start, samples = b1_rhs(g, p["order"]), time.perf_counter(), []
        for unit in range(1, int(p["final_time"]) + 1):
            q = rk4(q, 1.0 / per_unit, per_unit, rhs)
            samples.append([float(unit), b1_error(base_fields(q), BSSNState3D.gauge_wave(g, amplitude=0.01, time=float(unit)), 0.01)])
        seconds = time.perf_counter() - start
        for (t, e), (_, e_ref) in zip(samples, ref["samples"]):
            if t in (1, 10, 50, 100):
                rows.append(dict(case="B1-b", amplitude=0.01, t=t, native=e, python=e_ref))
                print(json.dumps(rows[-1]), flush=True)
        t = np.array([s[0] for s in samples]); e = np.array([s[1] for s in samples])
        late = t >= p["final_time"] / 2
        rows.append(dict(case="B1-b slope", native=float(np.polyfit(t[late], np.log(e[late]), 1)[0]),
                         python=0.014575671961370492, native_s=seconds, python_s=ref["seconds"]))
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parts", nargs="+", default=["s2", "b1"])
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    if "s2" in a.parts: rows += s2(a.out)
    if "b1" in a.parts: rows += b1(a.out)
    for r in rows:
        r["relative_difference"] = abs(r["native"] - r["python"]) / abs(r["python"])
    worst = max(r["relative_difference"] for r in rows)
    summary = dict(rows=rows, worst_relative_difference=worst, passed=worst <= 0.01,
                   python=platform.python_version(), numpy=np.__version__, platform=platform.platform())
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print("worst relative difference", worst, "passed", worst <= 0.01)


if __name__ == "__main__":
    main()

"""Diagnostic (not an acceptance gate): per-field growth of the S2 robust-stability noise.

Rebuilds exactly the seeded noise of fo_ccz4_robust_stability_campaign_3d, evolves it with
the same RK4 path, and records the RMS of every base and auxiliary field once per crossing
time, so that bounded, linear and exponential growth can be told apart.

    python -B -m experiments.s2_periodic_evolution.robust_diagnostic --order 4 --points 8 \
        --final-time 40 --out outputs/s2-diag/fd4-n8.json [--kappa1 0.1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from science.numerical_relativity import BSSNState3D, FOCCZ4Config, FOCCZ4State3D, GaugeConfig, Grid3D
from science.numerical_relativity.constraints3d import constraint_report_3d
from science.numerical_relativity.fo_ccz4_rhs import advance_fo_ccz4_rk4_3d

BASE = ("chi", "h", "trace_k", "a", "theta", "conformal_gamma", "lapse", "shift", "b")
AUX = ("lapse_gradient", "shift_gradient", "metric_gradient", "conformal_factor_gradient")


def noisy_state(points, order, amplitude=1.0e-10, seed=42):
    grid = Grid3D((points,) * 3, (0.0,) * 3, (1.0,) * 3, boundary="periodic", spatial_order=order)
    state = BSSNState3D.minkowski(grid, formulation="ccz4")
    rng = np.random.default_rng(seed)
    state.chi += amplitude * rng.normal(size=grid.shape)
    state.lapse += amplitude * rng.normal(size=grid.shape)
    state.trace_k += amplitude * rng.normal(size=grid.shape)
    state.theta += 0.1 * amplitude * rng.normal(size=grid.shape)
    state.conformal_gamma += amplitude * rng.normal(size=(*grid.shape, 3))
    state.shift += amplitude * rng.normal(size=(*grid.shape, 3))
    state.b += amplitude * rng.normal(size=(*grid.shape, 3))
    for diagonal in range(3):
        state.h[..., diagonal, diagonal] += amplitude * rng.normal(size=grid.shape)
    return FOCCZ4State3D.from_bssn_state(state)


def rms(state):
    out = {}
    identity = np.eye(3)
    for name in BASE:
        value = getattr(state.base, name)
        if name == "h":
            value = value - identity
        elif name in ("chi", "lapse"):
            value = value - 1.0
        out[name] = float(np.sqrt(np.mean(value * value)))
    for name in AUX:
        value = getattr(state.auxiliary, name)
        out["aux_" + name] = float(np.sqrt(np.mean(value * value)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, choices=(2, 4), required=True)
    parser.add_argument("--points", type=int, default=8)
    parser.add_argument("--final-time", type=float, default=40.0)
    parser.add_argument("--kappa1", type=float, default=0.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dt = 0.1 / args.points
    per_unit = int(round(1.0 / dt))
    state = noisy_state(args.points, args.order)
    gauge, config = GaugeConfig(theta_coupling=2.0), FOCCZ4Config(kappa1=args.kappa1)
    rows = [dict(time=0.0, rms=rms(state))]
    start = time.perf_counter()
    for unit in range(1, int(round(args.final_time)) + 1):
        state = advance_fo_ccz4_rk4_3d(state, dt, steps=per_unit, gauge=gauge, config=config)
        norms = constraint_report_3d(state.base).norms
        rows.append(dict(time=float(unit), rms=rms(state),
                         constraints={k: float(v) for k, v in norms.items()}))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(dict(vars(args), out=str(args.out), dt=dt, rows=rows,
                                            seconds=time.perf_counter() - start), indent=1))
        if not all(np.isfinite(v) for v in rows[-1]["rms"].values()):
            break


if __name__ == "__main__":
    main()

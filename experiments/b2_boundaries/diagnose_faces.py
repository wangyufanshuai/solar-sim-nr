"""Post-hoc diagnostic for B2-a (not a registered test; counts toward no gate).

Question: is the observed order ~2 (instead of ~3) caused by zero-speed characteristic fields,
whose second-order boundary truncation errors are not carried out of the domain?

Method: evolve exactly the B2-a problem (run_b2.evolve: sine TT wave, exact incoming data on all
faces, T = 0.25) on Nx = 13 and 25, take the final error e = (q - exact) / A, and project it on
the x-direction characteristic fields of the flat background, w = basis^-1 e (basis and speeds
from the same local_split the SAT uses; A = 1e-8, so the flat basis is exact to ~1e-8). Fields
are grouped by speed: the 11 ODE slots, the zero-speed PDE fields, and the moving fields
(|speed| > 0). For each group the maximum |w| is reported on the x-face nodes, on the next three
nodes, and on the interior, plus the observed order between the two grids.

    python -B -m experiments.b2_boundaries.diagnose_faces --out outputs/b2a-faces --nx 13 25
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .linear_wave_slots import wave_slots
from .parallel_sat import robust_local_split
from .run_b2 import AMPLITUDE, CONVERGE, TRANSVERSE, B2Cube, evolve, make_grid


def flat_split(nx):
    grid = make_grid(nx)
    cube = B2Cube(grid)
    flat = wave_slots(grid.coordinates[0][:, 0, 0], amplitude=0.0, time=0.0, transverse_shape=(TRANSVERSE, TRANSVERSE))
    base = cube.state(flat).base
    split, _ = robust_local_split(base.h[0, 0, 0], base.chi[0, 0, 0], base.lapse[0, 0, 0], base.shift[0, 0, 0], 0,
                                  gauge=cube.gauge, config=cube.config, profile=cube.profile)
    return split


def families(roots):
    roots = np.asarray(roots)
    ode = np.zeros(58, bool)
    ode[:11] = True
    zero = ~ode & (abs(roots) < 1e-12)
    groups = dict(ode=ode, pde_zero_speed=zero, moving=~ode & ~zero)
    for value in sorted({round(abs(r), 9) for r in roots[~ode & ~zero]}):
        groups[f"speed_{value:g}"] = ~ode & (abs(abs(roots) - value) < 1e-9)
    return groups


def analyse(nx, out):
    r = evolve(nx, CONVERGE["final_time"], profile="sine", outgoing_free=False)
    error = (r["q"] - r["exact"](CONVERGE["final_time"])) / AMPLITUDE  # 58 x Nx x 9 x 9
    split = flat_split(nx)
    w = np.einsum("ab,bxyz->axyz", np.linalg.inv(split.basis), error)
    absw = np.max(np.abs(w), axis=(2, 3))  # 58 x Nx, max over the transverse plane
    regions = dict(face=[0, nx - 1], near=[1, 2, 3, nx - 4, nx - 3, nx - 2], interior=list(range(4, nx - 4)))
    table = {name: {region: float(np.max(absw[mask][:, idx])) for region, idx in regions.items()}
             for name, mask in families(split.roots).items()}
    profile = {name: [float(v) for v in np.max(absw[mask], axis=0)] for name, mask in families(split.roots).items()}
    result = dict(nx=nx, steps=r["steps"], seconds=r["seconds"], max_error=float(np.max(np.abs(error))),
                  roots=[float(x) for x in split.roots], families=table, x_profile=profile)
    (out / f"faces-n{nx}.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(dict(nx=nx, max_error=result["max_error"], families=table)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nx", type=int, nargs="+", default=[13, 25])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    results = [analyse(n, args.out) for n in args.nx]
    if len(results) >= 2:
        a, b = results[-2], results[-1]
        ratio = math.log((b["nx"] - 1) / (a["nx"] - 1))
        orders = {name: {region: (math.log(a["families"][name][region] / b["families"][name][region]) / ratio
                                  if b["families"][name][region] > 0 and a["families"][name][region] > 0 else None)
                         for region in a["families"][name]} for name in a["families"]}
        (args.out / "orders.json").write_text(json.dumps(orders, indent=1) + "\n", encoding="utf-8")
        print(json.dumps(dict(orders=orders), indent=1), flush=True)


if __name__ == "__main__":
    main()

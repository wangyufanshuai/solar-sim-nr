"""B3 step 1 benchmark: Python vs native FO-CCZ4 RHS on an Nx^3 periodic FD4 state.

    python -m kernel.bench_rhs --points 32 48 --threads 1 8 24 --out outputs/b3-bench.json
Each timing is the best of `repeat` calls after one warm-up call.
"""
import argparse, json, os, platform, time
from pathlib import Path
import numpy as np
from science.numerical_relativity import Grid3D
from science.numerical_relativity.nr3d import BSSNState3D
from science.numerical_relativity.ccz4_reference import GaugeConfig
from science.numerical_relativity.fo_ccz4_rhs import FOCCZ4State3D, FOCCZ4Config, fo_ccz4_rhs_3d
from science.numerical_relativity.fo_ccz4_checkpoint import state_slots
from kernel.native_rhs import native_rhs_slots

GAUGE = GaugeConfig(theta_coupling=2.0)
CONFIG = FOCCZ4Config()


def state(n):
    g = Grid3D((n,) * 3, (0.,) * 3, (1.,) * 3, boundary="periodic", spatial_order=4)
    s = BSSNState3D.minkowski(g, formulation="ccz4")
    x, y, z = g.coordinates
    s.chi[:] = 1 + 1e-3 * np.sin(2 * np.pi * (x + y + z))
    s.lapse[:] = 1 + 1e-3 * np.cos(2 * np.pi * (x - z))
    s.shift[..., 0] = 1e-3 * np.sin(2 * np.pi * y)
    return FOCCZ4State3D.from_bssn_state(s), g


def best(f, repeat):
    f()
    times = []
    for _ in range(repeat):
        t = time.perf_counter(); f(); times.append(time.perf_counter() - t)
    return min(times)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--points", type=int, nargs="+", default=[32])
    p.add_argument("--threads", type=int, nargs="+", default=[1, 8])
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    rows = []
    for n in a.points:
        st, g = state(n)
        q = np.stack(state_slots(st))
        py = best(lambda: fo_ccz4_rhs_3d(st, gauge=GAUGE, config=CONFIG), max(1, a.repeat - 2))
        for t in a.threads:
            nat = best(lambda: native_rhs_slots(q, spacing=g.spacing, order=4, gauge=GAUGE, config=CONFIG, threads=t), a.repeat)
            rows.append(dict(points=n, threads=t, python_s=py, native_s=nat, speedup=py / nat))
            print(json.dumps(rows[-1]), flush=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(rows=rows, cpu_count=os.cpu_count(), platform=platform.platform(),
                                         numpy=np.__version__), indent=1) + "\n")


if __name__ == "__main__":
    main()

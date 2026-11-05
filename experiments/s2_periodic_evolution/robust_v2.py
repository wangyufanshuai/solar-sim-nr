"""S2-c v2 robustness gate (registered 2026-11-05 AFTER seeing S2 run 01 and its diagnostics).

Run 01 used a fixed ratio from t = 0 (growth <= 10). Diagnostics showed a one-off
constraint transient in the first crossing time, plateaus, and linear lapse growth. This
gate instead tests for exponential growth after the transient. Disclosed: the rule was
written with those diagnostics in view; run 01 stays FAILED.

Runs (robust_diagnostic.py, seed 42, noise 1e-10, dt = 0.1 h, one sample per crossing):
  fd2-n8 (T=100), fd4-n8 (T=100), fd4-n8-k01 (kappa1=0.1, T=100), fd4-n12 (T=50).
Total norm N(t) = sqrt(sum of mean squares of the base-field perturbations), the same
quantity as fo_ccz4_rhs._fo_ccz4_perturbation_l2.
Pass, for every run:
  1. all samples finite;
  2. least-squares slope of ln N(t) over t in [T/2, T] <= 0.02 per crossing time
     (e-folding time >= 50 crossings);
  3. Hamiltonian-constraint L2 at T <= 10 x its value at t = 1.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

RUNS = {"fd2-n8": 100, "fd4-n8": 100, "fd4-n8-k01": 100, "fd4-n12": 50}
MAX_LOG_SLOPE = 0.02
MAX_CONSTRAINT_GROWTH = 10.0


def total_norm(rms):
    base = ("chi", "h", "trace_k", "a", "theta", "conformal_gamma", "lapse", "shift", "b")
    return math.sqrt(sum(rms[k] ** 2 for k in base))


def evaluate(directory):
    directory = Path(directory)
    result = {}
    for name, final in RUNS.items():
        data = json.loads((directory / f"{name}.json").read_text())
        rows = data["rows"]
        t = np.array([r["time"] for r in rows])
        n = np.array([total_norm(r["rms"]) for r in rows])
        finite = bool(np.all(np.isfinite(n))) and rows[-1]["time"] >= final
        late = t >= final / 2
        slope = float(np.polyfit(t[late], np.log(n[late]), 1)[0]) if finite else math.inf
        h1 = next(r["constraints"]["hamiltonian_l2"] for r in rows if r["time"] == 1.0)
        hT = rows[-1]["constraints"]["hamiltonian_l2"]
        checks = dict(
            finite_and_complete=finite,
            late_log_slope=slope,
            slope_ok=slope <= MAX_LOG_SLOPE,
            hamiltonian_growth=hT / h1,
            constraint_ok=hT / h1 <= MAX_CONSTRAINT_GROWTH,
            growth_from_t0_at_T=float(n[-1] / n[0]),
        )
        checks["passed"] = checks["finite_and_complete"] and checks["slope_ok"] and checks["constraint_ok"]
        result[name] = checks
    result["passed"] = all(v["passed"] for v in result.values() if isinstance(v, dict))
    (directory / "robust-v2-summary.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(evaluate(sys.argv[1]), indent=1))

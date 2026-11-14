"""ctypes wrapper for the native periodic FO-CCZ4 volume RHS (B3 step 1, kernel/fo_ccz4_rhs.cpp).

Works on the canonical 58-slot layout. Only the case the native code implements is accepted:
periodic grid, FD2/FD4, gauge_profile=None (constant f, mu). Everything else stays in Python.

    python -m kernel.native_rhs build      # compiles kernel/build/fo_ccz4_rhs.dll with g++
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
LIBRARY = HERE / "build" / ("fo_ccz4_rhs.dll" if os.name == "nt" else "libfo_ccz4_rhs.so")
FLAGS = ["-O3", "-std=c++17", "-fopenmp", "-ffp-contract=off", "-fno-fast-math", "-shared"]
THETA_TERM_NAMES = (
    "advection", "curvature", "extrinsic", "theta_k", "z_lapse", "damping",
    "ricci_z_scalar", "a_squared", "z_dot_lapse_gradient", "lapse", "chi", "theta_rhs",
)
_lib = None


def build(compiler: str | None = None) -> Path:
    compiler = compiler or shutil.which("g++") or "g++"
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    extra = ["-static"] if os.name == "nt" else ["-fPIC"]
    subprocess.run([compiler, *FLAGS, *extra, "-o", str(LIBRARY), str(HERE / "fo_ccz4_rhs.cpp")], check=True)
    return LIBRARY


def _load():
    global _lib
    if _lib is None:
        if not LIBRARY.exists():
            raise FileNotFoundError(f"{LIBRARY} not built; run `python -m kernel.native_rhs build`")
        lib = ctypes.CDLL(str(LIBRARY))
        f = lib.fo_ccz4_rhs
        p = ctypes.POINTER(ctypes.c_double)
        f.argtypes = [p, p, ctypes.c_int, ctypes.c_int, ctypes.c_int, p, ctypes.c_int,
                      ctypes.POINTER(ctypes.c_int), p, ctypes.c_int]
        f.restype = ctypes.c_int
        _lib = lib
    return _lib


def native_rhs_slots(q, *, spacing, order, gauge, config, threads=None, sbp_axes=(0, 0, 0), profile=None, return_theta_terms=False):
    """RHS of the 58 canonical slots q[58, nx, ny, nz].

    sbp_axes: per axis 1 = SBP(4,2) (non-periodic, order 4 only), 0 = periodic.
    return_theta_terms: opt-in (RHS, 12-term diagnostic array); ordinary calls return RHS.
    profile: an FOCCZ4GaugeProfile (field-dependent f, mu) or None (constant coefficients).
    """
    q = np.ascontiguousarray(q, dtype=np.float64)
    if q.ndim != 4 or q.shape[0] != 58:
        raise ValueError("q must have shape (58, nx, ny, nz)")
    if order not in (2, 4):
        raise ValueError("order must be 2 or 4")
    if not gauge.advective or gauge.theta_coupling != 2.0 * config.c:
        raise ValueError("FO-CCZ4 requires advective gauge with theta_coupling=2*c")
    params = np.array([gauge.lapse_coefficient, gauge.shift_coefficient, gauge.shift_damping,
                       config.kappa1, config.kappa2, config.kappa3, config.cleaning_speed,
                       config.ordering_mu, config.shift_strength, config.tau_inverse, config.k0, config.c,
                       1.0 if profile is not None else 0.0,
                       profile.shift_scale if profile is not None else 0.0,
                       profile.ordering_scale if profile is not None else 0.0],
                      dtype=np.float64)
    h = np.ascontiguousarray(spacing, dtype=np.float64)
    out = np.empty_like(q)
    dp = ctypes.POINTER(ctypes.c_double)
    threads = int(threads or os.environ.get("NR_THREADS", os.cpu_count() or 1))
    axes = (ctypes.c_int * 3)(*[int(bool(a)) for a in sbp_axes])
    arguments = (*q.shape[1:], h.ctypes.data_as(dp), order, axes, params.ctypes.data_as(dp), threads)
    if return_theta_terms:
        terms = np.empty((len(THETA_TERM_NAMES), *q.shape[1:]), dtype=np.float64)
        diagnostic = _load().fo_ccz4_rhs_theta_terms
        diagnostic.argtypes = [dp, dp, dp, ctypes.c_int, ctypes.c_int, ctypes.c_int, dp,
                              ctypes.c_int, ctypes.POINTER(ctypes.c_int), dp, ctypes.c_int]
        diagnostic.restype = ctypes.c_int
        status = diagnostic(q.ctypes.data_as(dp), out.ctypes.data_as(dp), terms.ctypes.data_as(dp), *arguments)
    else:
        status = _load().fo_ccz4_rhs(q.ctypes.data_as(dp), out.ctypes.data_as(dp), *arguments)
    if status:
        raise ValueError(f"native RHS rejected its input (status {status})")
    return (out, terms) if return_theta_terms else out


if __name__ == "__main__" and sys.argv[1:] == ["build"]:
    print(build())

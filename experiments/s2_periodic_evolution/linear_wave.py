"""S2: Apples-with-Apples linearised TT wave on a periodic grid (exact linear solution).

Physical metric, lapse 1, zero shift:
    gamma_xx = 1, gamma_yy = 1 + b, gamma_zz = 1 - b,  b = A sin(2 pi (x - t)),
    K_ij = -d_t gamma_ij / 2.
This solves the linearised Einstein equations; with the 1+log lapse and the Gamma-driver
shift used by fo_ccz4_rhs, K, Theta and GammaHat are O(A^2), so lapse and shift stay at
their initial values to O(A^2). The full nonlinear solution differs at O(A^2), i.e. by a
relative 1e-8 at the registered A = 1e-8. Conversion to conformal variables follows
BSSNState3D.gauge_wave: chi = det(gamma)^(-1/3), h = chi gamma,
Atilde = chi (K_ij - gamma_ij K / 3), GammaHat^i = -d_j h^ij (analytic).
"""

from __future__ import annotations

import math

import numpy as np

from science.numerical_relativity import BSSNState3D, Grid3D

AMPLITUDE = 1.0e-8


def periodic_grid(nx, *, order, transverse=5):
    return Grid3D(
        shape=(nx, transverse, transverse),
        lower=(0.0, 0.0, 0.0),
        upper=(1.0, transverse / nx, transverse / nx),
        boundary="periodic",
        spatial_order=order,
    )


def linear_wave_state(grid, *, amplitude=AMPLITUDE, time=0.0):
    """Exact linear TT wave on the slice t = time (wavelength 1 along x)."""
    if grid.boundary != "periodic":
        raise ValueError("periodic grid required")
    amplitude, time = float(amplitude), float(time)
    if not (math.isfinite(amplitude) and math.isfinite(time) and 0.0 < amplitude < 1e-2):
        raise ValueError("small finite positive amplitude and finite time required")
    k = 2.0 * np.pi
    x, _, _ = grid.coordinates
    b = amplitude * np.sin(k * (x - time))
    b_t = -k * amplitude * np.cos(k * (x - time))
    b_x = -b_t
    det = 1.0 - b * b
    chi = np.power(det, -1.0 / 3.0)
    gyy, gzz = 1.0 + b, 1.0 - b
    kyy, kzz = -0.5 * b_t, 0.5 * b_t
    trace_k = kyy / gyy + kzz / gzz
    state = BSSNState3D.minkowski(grid)
    state.chi[:] = chi
    state.h[..., 0, 0] = chi
    state.h[..., 1, 1] = chi * gyy
    state.h[..., 2, 2] = chi * gzz
    state.trace_k[:] = trace_k
    state.a[..., 0, 0] = chi * (-trace_k / 3.0)
    state.a[..., 1, 1] = chi * (kyy - gyy * trace_k / 3.0)
    state.a[..., 2, 2] = chi * (kzz - gzz * trace_k / 3.0)
    state.conformal_gamma[..., 0] = (2.0 / 3.0) * b * b_x * np.power(det, -2.0 / 3.0)
    state.metadata = {
        "initial_data": "awa-linear-tt-wave",
        "amplitude": f"{amplitude:.17g}",
        "time": f"{time:.17g}",
    }
    return state


BASE_FIELDS = ("chi", "h", "trace_k", "a", "theta", "conformal_gamma", "lapse", "shift", "b")


def field_errors(state, exact, amplitude):
    """Max-norm error of every base field, divided by the wave amplitude."""
    return {
        name: float(np.max(np.abs(getattr(state, name) - getattr(exact, name)))) / amplitude
        for name in BASE_FIELDS
    }

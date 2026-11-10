"""B3 step 2: classical RK4 on the 58 canonical slots with the native RHS.

The Python driver (advance_fo_ccz4_rk4_3d) updates chi and lapse multiplicatively,
chi * exp(2 s r_lnphi) and lapse * exp(s r_lnalpha); on the slots, which store ln(alpha) and
ln(phi) = ln(chi)/2, that is exactly the additive update used here, so both drivers take the
same RK4 steps (stage times t, t+dt/2, t+dt/2, t+dt; weights 1, 2, 2, 1) up to rounding.

`harmonic_correction` reproduces experiments/b1_gauge_wave/harmonic.py on the slots (harmonic
slicing G = f alpha on top of the constant-G RHS; requires shift_strength = 0).
"""

from __future__ import annotations

import numpy as np

from science.numerical_relativity.fo_ccz4_checkpoint import SYMMETRIC, state_slots

from .native_rhs import native_rhs_slots


def slots(state) -> np.ndarray:
    return np.ascontiguousarray(np.stack(state_slots(state)), dtype=np.float64)


def base_fields(q: np.ndarray) -> dict[str, np.ndarray]:
    """Base fields named as on BSSNState3D, rebuilt from the slots."""
    shape = q.shape[1:]
    h, a = np.empty((*shape, 3, 3)), np.empty((*shape, 3, 3))
    for offset, (i, j) in enumerate(SYMMETRIC):
        h[..., i, j] = h[..., j, i] = q[offset]
        a[..., i, j] = a[..., j, i] = q[11 + offset]
    return dict(h=h, a=a, lapse=np.exp(q[6]), chi=np.exp(2.0 * q[10]), shift=np.moveaxis(q[7:10], 0, -1),
                trace_k=q[17], theta=q[18], conformal_gamma=np.moveaxis(q[19:22], 0, -1),
                b=np.moveaxis(q[22:25], 0, -1))


def periodic_d1(values: np.ndarray, spacing: float, axis: int, order: int) -> np.ndarray:
    """Same centred differences as bssn3d._d1 on a periodic grid."""
    if order == 4:
        return (8.0 * (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis))
                - (np.roll(values, -2, axis=axis) - np.roll(values, 2, axis=axis))) / (12.0 * spacing)
    return (np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)) / (2.0 * spacing)


def harmonic_correction(q, rhs, *, spacing, order, gauge, config):
    if config.shift_strength != 0.0:
        raise ValueError("harmonic correction requires shift_strength = 0")
    f = gauge.lapse_coefficient
    g = f * np.exp(q[6])
    source = q[17] - config.k0 - 2.0 * config.c * q[18]
    rhs[6] += (f - g) * source
    for axis in range(3):
        d_k = periodic_d1(q[17], spacing[axis], axis, order) - 2.0 * config.c * periodic_d1(q[18], spacing[axis], axis, order)
        rhs[25 + axis] += (f - g) * d_k - g * q[25 + axis] * source
    return rhs


def make_rhs(*, spacing, order, gauge, config, harmonic=False, threads=None):
    def rhs(q):
        r = native_rhs_slots(q, spacing=spacing, order=order, gauge=gauge, config=config, threads=threads)
        return harmonic_correction(q, r, spacing=spacing, order=order, gauge=gauge, config=config) if harmonic else r
    return rhs


def rk4(q: np.ndarray, dt: float, steps: int, rhs) -> np.ndarray:
    if not dt > 0 or steps < 1:
        raise ValueError("dt must be positive and steps >= 1")
    for _ in range(steps):
        k1 = rhs(q)
        k2 = rhs(q + 0.5 * dt * k1)
        k3 = rhs(q + 0.5 * dt * k2)
        k4 = rhs(q + dt * k3)
        q = q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return q

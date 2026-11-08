"""Harmonic slicing for the FO-CCZ4 RHS, as a correction on top of the constant-coefficient law.

fo_ccz4_rhs_3d evolves  d_t ln(alpha) = beta^k A_k − G (K − K0 − 2c Theta)  with constant
G = f (lapse_coefficient; f = 2 is 1+log). Harmonic slicing is G = f·alpha. Writing
S = K − K0 − 2c Theta, the harmonic RHS differs from the constant one by

    ln(alpha):  + (f − f alpha) S
    A_k:        + (f − f alpha) d_k(K − 2c Theta) − (f alpha) A_k S     [the last term is −(d_k G) S]

with d_k the same centred derivative (_d1) the RHS uses, so the result is exactly the RHS one would
get by evaluating the equations with G = f alpha. The constant-law RHS also scales two terms of the
A_k equation by shift_strength·G; this wrapper therefore requires shift_strength = 0 (shift
evolution off), which is what the zero-shift gauge-wave tests need.

Kept outside science/ because science/numerical_relativity/fo_ccz4_rhs.py is byte-bound by the
frozen S1 registration and the constraint-bridge export manifest.
"""

from __future__ import annotations

from dataclasses import replace

from science.numerical_relativity import FOCCZ4Config, FOCCZ4State3D, GaugeConfig
from science.numerical_relativity.bssn3d import _d1
from science.numerical_relativity.fo_ccz4_rhs import (
    FOCCZ4RHS3D,
    _state_add,
    _state_rk4_update,
    fo_ccz4_rhs_3d,
)


def harmonic_fo_ccz4_rhs(state: FOCCZ4State3D, *, gauge: GaugeConfig, config: FOCCZ4Config) -> FOCCZ4RHS3D:
    if config.shift_strength != 0.0:
        raise ValueError("harmonic wrapper requires shift_strength = 0")
    if state.base.grid.boundary != "periodic":
        raise ValueError("harmonic wrapper is validated on periodic grids only")
    rhs = fo_ccz4_rhs_3d(state, gauge=gauge, config=config, boundary_mode="periodic")
    base, aux = state.base, state.auxiliary
    f = gauge.lapse_coefficient
    g_harmonic = f * base.lapse
    source = base.trace_k - config.k0 - 2.0 * config.c * base.theta
    ln_lapse = rhs.ln_lapse + (f - g_harmonic) * source
    lapse_gradient = rhs.lapse_gradient.copy()
    for axis in range(3):
        d_k = _d1(base.trace_k, base, axis) - 2.0 * config.c * _d1(base.theta, base, axis)
        lapse_gradient[..., axis] += (f - g_harmonic) * d_k - g_harmonic * aux.lapse_gradient[..., axis] * source
    return replace(rhs, ln_lapse=ln_lapse, lapse_gradient=lapse_gradient)


def advance_harmonic_rk4(state: FOCCZ4State3D, dt: float, *, steps: int, gauge: GaugeConfig,
                         config: FOCCZ4Config) -> FOCCZ4State3D:
    """Classical RK4, same stage combination as advance_fo_ccz4_rk4_3d."""
    if not dt > 0 or steps < 1:
        raise ValueError("dt must be positive and steps >= 1")
    for _ in range(steps):
        k1 = harmonic_fo_ccz4_rhs(state, gauge=gauge, config=config)
        k2 = harmonic_fo_ccz4_rhs(_state_add(state, k1, 0.5 * dt), gauge=gauge, config=config)
        k3 = harmonic_fo_ccz4_rhs(_state_add(state, k2, 0.5 * dt), gauge=gauge, config=config)
        k4 = harmonic_fo_ccz4_rhs(_state_add(state, k3, dt), gauge=gauge, config=config)
        state = _state_rk4_update(state, dt, k1, k2, k3, k4)
    return state

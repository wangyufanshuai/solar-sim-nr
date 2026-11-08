"""B2 tooling: the exact linearised TT wave as 58 canonical FO-CCZ4 slots on a non-periodic grid.

Fields as functions of b = A sin(2 pi (x - t)) (lapse 1, zero shift; same conversion as
experiments/s2_periodic_evolution/linear_wave.py):
    gamma = diag(1, 1 + b, 1 - b), K_yy = -b_t/2, K_zz = +b_t/2, chi = det(gamma)^(-1/3),
    h = chi gamma, Atilde = chi (K - gamma K/3), GammaHat^x = (2/3) b b_x det^(-2/3).
Auxiliary slots are exact derivatives, not finite differences: D_xij = (1/2) d_x h_ij and
P_x = d_x (1/2 ln chi) are dF/db * b_x with dF/db from a complex step (no cancellation even at
A = 1e-8). A, B and the y, z components vanish.

Slot layout (fo_ccz4_sbp_volume.state_from_slots): 0-5 h, 6 ln alpha, 7-9 beta, 10 ln phi,
11-16 Atilde, 17 K, 18 Theta, 19-21 GammaHat, 22-24 b, 25-27 A, 28-36 B, 37-54 D, 55-57 P.
"""

from __future__ import annotations

import numpy as np

SYMMETRIC = tuple((i, j) for i in range(3) for j in range(i, 3))
K_WAVE = 2.0 * np.pi
COMPLEX_STEP = 1e-30


def _metric_parts(b):
    """chi and h_ij for amplitude field b (real or complex)."""
    det = 1.0 - b * b
    chi = det ** (-1.0 / 3.0)
    h = {(0, 0): chi, (1, 1): chi * (1.0 + b), (2, 2): chi * (1.0 - b)}
    return chi, h


PULSE_CENTRE, PULSE_WIDTH = 0.5, 0.08


def wave_slots(x, *, amplitude, time, transverse_shape, profile="sine"):
    """58 x Nx x Ny x Nz canonical slots of the exact linear TT wave at the given time.

    profile "sine": b = A sin(2 pi (x - t)); "pulse": b = A exp(-((x - t - 0.5)/0.08)^2),
    a right-moving packet (any b(x - t) is an exact linear solution).
    """
    x = np.asarray(x, dtype=np.float64)
    if profile == "sine":
        phase = K_WAVE * (x - time)
        b = amplitude * np.sin(phase)
        b_x = K_WAVE * amplitude * np.cos(phase)
    elif profile == "pulse":
        u = (x - time - PULSE_CENTRE) / PULSE_WIDTH
        b = amplitude * np.exp(-u * u)
        b_x = -2.0 * u / PULSE_WIDTH * b
    else:
        raise ValueError("profile must be 'sine' or 'pulse'")
    b_t = -b_x
    chi, h = _metric_parts(b)
    gamma_yy, gamma_zz = 1.0 + b, 1.0 - b
    k_yy, k_zz = -0.5 * b_t, 0.5 * b_t
    trace_k = k_yy / gamma_yy + k_zz / gamma_zz
    det = 1.0 - b * b
    # dF/db by complex step, then the chain rule for d/dx.
    chi_c, h_c = _metric_parts(b + 1j * COMPLEX_STEP)
    dchi_db = chi_c.imag / COMPLEX_STEP
    dh_db = {key: value.imag / COMPLEX_STEP for key, value in h_c.items()}

    nx = x.size
    q = np.zeros((58, nx))
    for offset, (i, j) in enumerate(SYMMETRIC):
        if (i, j) in h:
            q[offset] = h[(i, j)]
            q[37 + offset] = 0.5 * dh_db[(i, j)] * b_x  # D_x,ij (axis 0 block)
    q[10] = 0.5 * np.log(chi)
    a_tilde = {(0, 0): chi * (-trace_k / 3.0),
               (1, 1): chi * (k_yy - gamma_yy * trace_k / 3.0),
               (2, 2): chi * (k_zz - gamma_zz * trace_k / 3.0)}
    for offset, (i, j) in enumerate(SYMMETRIC):
        if (i, j) in a_tilde:
            q[11 + offset] = a_tilde[(i, j)]
    q[17] = trace_k
    q[19] = (2.0 / 3.0) * b * b_x * det ** (-2.0 / 3.0)
    q[55] = 0.5 * (dchi_db / chi) * b_x  # P_x = d_x (1/2 ln chi)
    return np.broadcast_to(q[:, :, None, None], (58, nx, *transverse_shape)).copy()

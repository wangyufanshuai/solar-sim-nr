"""v2 slot gate (registered 2026-11-03, before the v2 run): Richardson elimination.

Every coefficient is a symmetric +/-eps difference, so for each slot
    value(eps) = reference + c2*eps**2 + c4*eps**4 + ...
With the two registered amplitudes eps_L = eps_H/4,
    R = (16*value_L - value_H) / 15
removes the c2 term exactly and leaves R - reference = -c4*eps_H**4/16 + O(eps_H**6).
v1 bounded |c2| by 64*w; v2 bounds |c4| by 64**2*w (same geometric growth), giving
    |R - reference| <= 256*eps_H**4*w + 8*nu_R,  nu_R = (16*nu_L + nu_H)/15,
where nu_* are the v1 per-amplitude resolution indicators (empirical, not rigorous).
An O(1) coefficient fault is not removed by R and is still rejected. All other v1
checks, amplitudes, grid, modes and negative-control gates are unchanged.

Motivation, disclosed: v1 run 04 failed one slot whose error scaled exactly as
eps**2 with |c2| ~= 64.3 w. v2 was written after seeing that result.
"""

from __future__ import annotations

import importlib

EPS_H, EPS_L = 2.0**-10, 2.0**-12
C4_BOUND = 64.0**2


def richardson_slots(values, reference, resolution, weights):
    """values/resolution: leading axis = (eps_H, eps_L); returns a JSON-safe summary."""
    np = importlib.import_module("numpy")
    values = np.asarray(values)
    resolution = np.asarray(resolution, dtype=np.float64)
    if values.shape[0] != 2 or resolution.shape != values.shape:
        raise ValueError("two registered amplitudes required")
    if not (np.all(np.isfinite(values)) and np.all(resolution > 0)):
        raise ValueError("finite values and positive resolution required")
    combined = (16.0 * values[1] - values[0]) / 15.0
    nu = (16.0 * resolution[1] + resolution[0]) / 15.0
    error = np.abs(combined - reference)
    limit = C4_BOUND * EPS_H**4 / 16.0 * np.broadcast_to(weights, error.shape) + 8.0 * nu
    ratio = error / limit
    return dict(
        all_slots=bool(np.all(error <= limit)),
        failed_slots=int(np.count_nonzero(error > limit)),
        worst_ratio=float(np.max(ratio)),
    )

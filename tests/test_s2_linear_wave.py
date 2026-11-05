"""Fast controls for the S2 linearised-wave setup (no long evolution)."""

import numpy as np

from experiments.s2_periodic_evolution.linear_wave import (
    AMPLITUDE,
    linear_wave_state,
    periodic_grid,
)
from experiments.s2_periodic_evolution.run_s2 import CONFIG, GAUGE, orders
from science.numerical_relativity import FOCCZ4State3D
from science.numerical_relativity.fo_ccz4_rhs import fo_ccz4_rhs_3d


def test_conformal_metric_has_unit_determinant():
    state = linear_wave_state(periodic_grid(16, order=4), amplitude=1e-3)
    assert np.max(np.abs(np.linalg.det(state.h) - 1.0)) < 1e-14


def _rhs_error(nx, order):
    grid = periodic_grid(nx, order=order)
    rhs = fo_ccz4_rhs_3d(
        FOCCZ4State3D.from_bssn_state(linear_wave_state(grid)),
        gauge=GAUGE, config=CONFIG, boundary_mode="periodic",
    )
    d = 1e-3
    a_t = (linear_wave_state(grid, time=d).a - linear_wave_state(grid, time=-d).a) / (2 * d)
    return np.max(np.abs(rhs.a[..., 1, 1] - a_t[..., 1, 1])) / AMPLITUDE


def test_initial_rhs_matches_exact_time_derivative_at_design_order():
    for order, low in ((2, 1.8), (4, 3.7)):
        p = orders([_rhs_error(16, order), _rhs_error(32, order)])[0]
        assert p > low


def test_orders_helper():
    assert orders([16.0, 4.0, 1.0]) == [2.0, 2.0]

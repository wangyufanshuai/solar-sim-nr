"""Fast controls for the harmonic lapse law (no long evolution)."""

import numpy as np
import pytest

from science.numerical_relativity import BSSNState3D, FOCCZ4Config, FOCCZ4State3D, GaugeConfig, Grid3D
from experiments.b1_gauge_wave.harmonic import harmonic_fo_ccz4_rhs
from science.numerical_relativity.fo_ccz4_rhs import fo_ccz4_rhs_3d

GAUGE = GaugeConfig(theta_coupling=2.0, lapse_coefficient=1.0)
CONFIG = FOCCZ4Config(kappa1=0.0, shift_strength=0.0)


def _grid(nx, order):
    return Grid3D((nx, 5, 5), (0.0, 0.0, 0.0), (1.0, 5 / nx, 5 / nx), boundary="periodic", spatial_order=order)


def _rhs_error(nx, order, amplitude):
    g = _grid(nx, order)
    rhs = harmonic_fo_ccz4_rhs(FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(g, amplitude=amplitude)),
                               gauge=GAUGE, config=CONFIG)
    d = 1e-4
    plus = BSSNState3D.gauge_wave(g, amplitude=amplitude, time=d)
    minus = BSSNState3D.gauge_wave(g, amplitude=amplitude, time=-d)
    ln_lapse_t = (np.log(plus.lapse) - np.log(minus.lapse)) / (2 * d)
    k_t = (plus.trace_k - minus.trace_k) / (2 * d)
    return max(np.max(np.abs(rhs.ln_lapse - ln_lapse_t)), np.max(np.abs(rhs.trace_k - k_t))) / amplitude


@pytest.mark.parametrize("order,low", [(2, 1.8), (4, 3.7)])
def test_harmonic_rhs_matches_exact_gauge_wave_at_design_order(order, low):
    e16, e32 = _rhs_error(16, order, 0.1), _rhs_error(32, order, 0.1)
    assert np.log2(e16 / e32) > low


def test_harmonic_wrapper_differs_from_constant_law_only_in_lapse_fields():
    g = _grid(16, 4)
    state = FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(g, amplitude=0.01))
    constant = fo_ccz4_rhs_3d(state, gauge=GAUGE, config=CONFIG)
    harmonic = harmonic_fo_ccz4_rhs(state, gauge=GAUGE, config=CONFIG)
    assert not np.array_equal(constant.ln_lapse, harmonic.ln_lapse)
    for name in ("h", "a", "trace_k", "theta", "conformal_gamma", "metric_gradient"):
        assert np.array_equal(getattr(constant, name), getattr(harmonic, name))


def test_harmonic_wrapper_requires_frozen_shift():
    g = _grid(16, 4)
    state = FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(g, amplitude=0.01))
    with pytest.raises(ValueError):
        harmonic_fo_ccz4_rhs(state, gauge=GAUGE, config=FOCCZ4Config(kappa1=0.0, shift_strength=1.0))

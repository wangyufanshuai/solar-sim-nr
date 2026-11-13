"""Native diagnostics preserve evolved auxiliaries and detect invalid trajectories."""
import numpy as np
import pytest
from experiments.b3_native.long_diagnostics import validate_slots, ScientificFailure, windows
from experiments.s2_periodic_evolution.robust_diagnostic import noisy_state
from kernel.native_evolve import state_from_slots, slots, rk4, make_rhs
from science.numerical_relativity import GaugeConfig, FOCCZ4Config
from science.numerical_relativity.fo_ccz4_rhs import advance_fo_ccz4_rk4_3d


def test_restore_preserves_evolved_auxiliary_not_reconstructed_derivative():
    state = noisy_state(8, 4)
    q = slots(state)
    q[25:] += np.arange(33)[:, None, None, None] * 1e-7
    restored = slots(state_from_slots(q, state.grid))
    np.testing.assert_allclose(restored, q, rtol=0, atol=2e-16)
    assert np.array_equal(restored[25:], q[25:])


def test_s2_one_crossing_matches_python_all_slots():
    state = noisy_state(8, 4)
    gauge, config = GaugeConfig(theta_coupling=2.), FOCCZ4Config(kappa1=0.)
    ref = slots(advance_fo_ccz4_rk4_3d(state, .0125, steps=80, gauge=gauge, config=config))
    native = rk4(slots(state), .0125, 80, make_rhs(spacing=state.grid.spacing, order=4, gauge=gauge, config=config, threads=1))
    diff = np.max(np.abs(native-ref), axis=(1,2,3))
    scale = np.max(np.abs(ref-slots(state)), axis=(1,2,3))
    assert np.all(diff <= 1e-6*scale + 5e-15)


def test_checked_slots_reject_nonfinite_and_underflow():
    q = slots(noisy_state(8, 4))
    q[6,0,0,0] = -1000
    with pytest.raises(ScientificFailure, match="positivity"):
        validate_slots(q)
    q[6,0,0,0] = np.nan
    with pytest.raises(ScientificFailure, match="nonfinite"):
        validate_slots(q)


def test_window_reports_positive_growth_even_below_old_gate():
    rows = [dict(time=t, rms={k: np.exp(.001*t) for k in ("chi", "h", "trace_k", "a", "theta", "conformal_gamma", "lapse", "shift", "b")}) for t in range(501)]
    result = windows(rows)
    assert len(result) == 3
    assert all(abs(w["log_slope"]-.001) < 1e-12 for w in result)
    assert all(w["norm_ratio"] > 1 for w in result)

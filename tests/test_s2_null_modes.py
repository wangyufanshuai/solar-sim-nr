import numpy as np
import pytest
from experiments.s2_periodic_evolution.null_mode_diagnostic import parity_modes, predict_lapse_slope, null_derivative_checks
from experiments.s2_periodic_evolution.robust_diagnostic import noisy_state


def test_eight_parities_are_orthonormal_on_even_lattice():
    basis=np.array([p.ravel() for _,p in parity_modes((8,6,10))])
    np.testing.assert_array_equal(basis@basis.T/basis.shape[1],np.eye(8))
    with pytest.raises(ValueError): list(parity_modes((7,8,8)))


def test_frozen_fd2_fd4_derivatives_annihilate_all_parities():
    rows=null_derivative_checks()
    assert len(rows)==16
    assert all(row['max_derivative']==0 for row in rows)


def test_prediction_for_known_mixed_constant_and_checkerboard_source():
    state=noisy_state(8,4)
    p=dict(parity_modes((8,)*3))[(1,1,1)]
    state.base.trace_k[:]=3e-10+4e-10*p
    state.base.theta[:]=0
    prediction=predict_lapse_slope(state)
    assert abs(prediction['predicted_rms_slope']-1e-9)<1e-24
    assert abs(prediction['constant_only_slope']-6e-10)<1e-24

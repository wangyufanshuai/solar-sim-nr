"""Controls for the v2 Richardson slot gate (synthetic data, no physics)."""

import numpy as np
import pytest

from experiments.case12_binary64_v1.richardson import EPS_H, EPS_L, richardson_slots

REF = np.array([[0.0, 1.5], [-2.0, 0.25]])
W = np.maximum(1.0, np.sum(np.abs(REF), axis=-1))[:, None]
NU = np.full((2, 2, 2), 1e-12)


def measured(c2, c4=0.0, c1=0.0, offset=0.0):
    return np.stack(
        [REF + c1 * e + c2 * e**2 + c4 * e**4 + offset for e in (EPS_H, EPS_L)]
    )


def test_pure_eps2_term_is_removed_even_above_v1_constant():
    assert richardson_slots(measured(64.3 * W), REF, NU, W)["all_slots"]
    assert richardson_slots(measured(-500.0 * W), REF, NU, W)["all_slots"]


def test_eps4_term_within_registered_bound_passes():
    assert richardson_slots(measured(10.0 * W, c4=4000.0 * W), REF, NU, W)["all_slots"]


def test_eps4_term_beyond_bound_fails():
    result = richardson_slots(measured(0.0, c4=5000.0 * W), REF, NU, W)
    assert not result["all_slots"] and result["failed_slots"] == REF.size


@pytest.mark.parametrize("offset", [1e-6, -3e-7])
def test_constant_coefficient_fault_is_rejected(offset):
    assert not richardson_slots(measured(10.0 * W, offset=offset), REF, NU, W)["all_slots"]


def test_linear_in_eps_fault_is_rejected():
    assert not richardson_slots(measured(0.0, c1=1e-3), REF, NU, W)["all_slots"]


def test_complex_modal_errors_and_zero_reference():
    errors = measured(64.3 * W) - REF
    errors = errors * (1 + 1j) / np.sqrt(2)
    assert richardson_slots(errors, 0.0, NU, W)["all_slots"]


def test_shape_and_resolution_guards():
    with pytest.raises(ValueError):
        richardson_slots(measured(0.0)[:1], REF, NU[:1], W)
    with pytest.raises(ValueError):
        richardson_slots(measured(0.0), REF, np.zeros_like(NU), W)

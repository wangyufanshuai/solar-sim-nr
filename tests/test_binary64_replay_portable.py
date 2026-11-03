"""Platform-portable copy of the bound replay hand-recovery check.

tests/test_case12_binary64_replay.py is byte-bound by the v1 manifest. Two of its
assertions use absolute tolerances of about one ULP, which Linux OpenBLAS/libm builds
miss by a few ULP (5.3e-15 vs 5e-15, 2.1e-17 vs 1e-17). This copy checks the same
values with a 64-ULP tolerance scaled to each magnitude; the bound originals still run
unchanged on Windows.
"""

import numpy as np
import pytest

from experiments.case12_binary64_v1 import replay as R

ULP = np.finfo(np.float64).eps


@pytest.mark.parametrize("order", [2, 4])
def test_hand_recovery_mixed_coefficient_portable(order):
    k = 4.0 * np.sqrt(2.0) if order == 2 else (16.0 * np.sqrt(2.0) - 4.0) / 3.0
    values = np.zeros((1, 13), dtype=np.complex128)
    values[0, 0] = 3.0
    values[0, 1:7] = [
        3.0 - 4.0 * k * k + 2.0j * k,
        3.0 - 2.0 * k * k - 1.0j * k,
        3.0 + k * k + 5.0j * k,
        3.0 - 13.0 * k * k + 1.0j * k,
        3.0 + 0.0 * k * k + 7.0j * k,
        3.0 - 7.0 * k * k + 4.0j * k,
    ]
    values[0, 7:] = values[0, 1:7].conjugate()
    answer, resolution = R.recover_coefficients(
        values, np.full((1, 13), 0.25), order, second_order=True
    )
    expected = np.array([3.0, 2.0, -1.0, 5.0, 4.0, 7.0, -3.0, 2.0, 6.0, -1.0])
    scale = 13.0 * k * k  # largest input magnitude entering the recovery
    np.testing.assert_allclose(answer[:, 0], expected, rtol=0.0, atol=64 * ULP * scale)
    assert abs(answer[5, 0] - 14.0) > 6.0
    target = 1.5 / (k * k)
    np.testing.assert_allclose(resolution[5, 0], target, rtol=64 * ULP, atol=0.0)

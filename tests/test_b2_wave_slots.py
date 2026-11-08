"""Auxiliary slots of the B2 exact wave must be the derivatives of its metric slots."""

import numpy as np
import pytest

from experiments.b2_boundaries.linear_wave_slots import wave_slots


@pytest.mark.parametrize("profile, extra", [("sine", {}), ("pulse", dict(centre=0.8, width=0.1)),
                                            ("pulse", dict(centre=0.5, width=0.12))])
def test_d_and_p_match_fourth_order_differences(profile, extra):
    x = np.linspace(0.0, 1.0, 4001)
    dx = x[1] - x[0]
    q = wave_slots(x, amplitude=1e-3, time=0.05, transverse_shape=(1, 1), profile=profile, **extra)[:, :, 0, 0]

    def d4(f):
        return (f[:-4] - 8 * f[1:-3] + 8 * f[3:-1] - f[4:]) / (12 * dx)

    for slot, aux in ((0, 37), (3, 40), (5, 42)):  # h_xx, h_yy, h_zz -> D_x,ij
        scale = np.max(abs(q[aux])) or 1.0
        assert np.max(abs(q[aux][2:-2] - 0.5 * d4(q[slot]))) <= 1e-6 * scale
    assert np.max(abs(q[55][2:-2] - d4(q[10]))) <= 1e-6 * np.max(abs(q[55]))  # P_x = d_x (1/2 ln chi)

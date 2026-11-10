"""B3 step 3: native SBP(4,2) volume RHS with the B2 gauge profile equals the Python SBP volume RHS."""

import numpy as np
import pytest

from science.numerical_relativity import Grid3D
from science.numerical_relativity.fo_ccz4_checkpoint import state_slots
from science.numerical_relativity.fo_ccz4_principal_symbol import _rhs_components
from experiments.b2_boundaries.run_b2 import B2Cube
from kernel.native_rhs import LIBRARY, build, native_rhs_slots
from tests.test_native_rhs import state

if not LIBRARY.exists():
    build()


@pytest.mark.parametrize("amp", [1e-3, 0.2])
def test_native_sbp_volume_with_profile_matches_python(amp):
    st, periodic = state(10, 4, amp)
    q = np.stack(state_slots(st))
    grid = Grid3D(periodic.shape, (0.0,) * 3, (1.0, 1.3, 0.8), boundary="sbp-volume", spatial_order=4)
    cube = B2Cube(grid)
    ref = np.array(_rhs_components(cube.volume.rhs(cube.state(q), gauge=cube.gauge, config=cube.config,
                                                   gauge_profile=cube.profile)), dtype=np.float64)
    nat = native_rhs_slots(q, spacing=grid.spacing, order=4, gauge=cube.gauge, config=cube.config,
                           sbp_axes=(1, 1, 1), profile=cube.profile)
    scale = np.max(np.abs(ref), axis=(1, 2, 3))
    assert np.all(scale > 1e-10 * scale.max())
    assert (np.max(np.abs(nat - ref), axis=(1, 2, 3)) / scale).max() <= 1e-12

"""B3 step 2: native RK4 on the slots takes the same steps as advance_fo_ccz4_rk4_3d."""

import numpy as np

from science.numerical_relativity.fo_ccz4_rhs import advance_fo_ccz4_rk4_3d
from kernel.native_evolve import make_rhs, rk4, slots
from kernel.native_rhs import LIBRARY, build
from tests.test_native_rhs import CONFIG, GAUGE, state

if not LIBRARY.exists():
    build()


def test_native_rk4_matches_python_driver():
    st, g = state(12, 4, 0.05)
    dt, steps = 0.004, 3
    ref = slots(advance_fo_ccz4_rk4_3d(st, dt, steps=steps, gauge=GAUGE, config=CONFIG))
    nat = rk4(slots(st), dt, steps, make_rhs(spacing=g.spacing, order=4, gauge=GAUGE, config=CONFIG))
    change = np.max(np.abs(ref - slots(st)), axis=(1, 2, 3))
    diff = np.max(np.abs(nat - ref), axis=(1, 2, 3))
    assert np.all(diff <= 1e-11 * np.maximum(change, 1e-300) + 1e-15)

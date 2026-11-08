"""The pooled SAT RHS must equal NonlinearSBPCube.rhs bit for bit."""

import numpy as np

from experiments.b2_boundaries.linear_wave_slots import wave_slots
from experiments.b2_boundaries.parallel_sat import ParallelSATMixin
from experiments.b2_boundaries.run_b2 import B2Cube, make_grid


class _PooledCube(ParallelSATMixin, B2Cube):
    workers = 2


def test_pooled_rhs_is_bitwise_identical_on_a_nonuniform_state():
    grid = make_grid(9)
    x = grid.coordinates[0][:, 0, 0]
    q = wave_slots(x, amplitude=1e-3, time=0.1, transverse_shape=(9, 9))
    rng = np.random.default_rng(3)
    q[6:11] += 1e-6 * rng.standard_normal(q[6:11].shape)  # distinct face points; keeps det(h) = 1
    targets = B2Cube(grid).boundary_targets(wave_slots(x, amplitude=1e-3, time=0.12, transverse_shape=(9, 9)))
    reference, ref_report = B2Cube(grid).rhs(q, targets, 1e-4)
    pooled, report = _PooledCube(grid).rhs(q, targets, 1e-4)
    assert np.array_equal(reference, pooled)
    assert report["basis_builds"] == ref_report["basis_builds"]
    assert report["max_basis_condition"] == ref_report["max_basis_condition"]

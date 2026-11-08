"""The pooled SAT RHS must equal NonlinearSBPCube.rhs bit for bit."""

import numpy as np

from experiments.b2_boundaries.linear_wave_slots import wave_slots
from experiments.b2_boundaries.parallel_sat import ParallelSATMixin, robust_local_split
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


# Face point (axis 0, x = 1) where NumPy 2.2.6's SVD did not converge in B2-c run 01, step 902.
_H = np.array([[float.fromhex(v) for v in row] for row in (
    ("0x1.0000000000000p+0", "-0x1.73a3c5cc7dc56p-37", "-0x1.73a3b1928e31fp-37"),
    ("-0x1.73a3c5cc7dc56p-37", "0x1.ffffffaaa8b80p-1", "0x1.6410c52cfb650p-57"),
    ("-0x1.73a3b1928e31fp-37", "0x1.6410c52cfb650p-57", "0x1.0000002aaba43p+0"))])
_CHI, _ALPHA = float.fromhex("0x1.ffffffffffffep-1"), float.fromhex("0x1.ffffffffffffdp-1")
_BETA = np.array([float.fromhex(v) for v in ("-0x1.5e16e4f9447e8p-51", "-0x1.94c2528d8b1d2p-37", "-0x1.94c291a5992c0p-37")])


def test_robust_split_handles_the_b2c_svd_failure_point():
    split, _fell_back = robust_local_split(_H, _CHI, _ALPHA, _BETA, 0)  # fallback depends on the BLAS build
    assert np.isfinite(split.condition) and split.condition < 1e3
    assert max(split.residual, split.symmetry_residual) < 1e-10

"""Process-parallel SAT penalties for the nonlinear SBP cube (same numbers, faster).

NonlinearSBPCube.rhs computes one characteristic split (local_split) per distinct face point and
caches it only within that RHS call; during an evolution almost every face point is distinct, so
the splits dominate the cost (≈ 6.7 ms each). Here the main process groups face points by the same
cache key, a persistent process pool runs local_split and split.penalty for each group, and only
the 58-component penalties come back. Every number is produced by the same functions in the same
order, so the RHS is bitwise identical to NonlinearSBPCube.rhs (checked in
tests/test_b2_parallel_sat.py).

SVD fallback: NumPy 2.2.6's bundled LAPACK SVD failed to converge on a finite, well-scaled 47 x 47
matrix in B2-c run 01 (face point in tests/test_b2_parallel_sat.py). When local_split raises
LinAlgError, that one call is repeated with np.linalg.svd temporarily routed to SciPy's gesvd
driver (same full (u, s, vh) contract); all other calls are unchanged, so the pooled RHS is still
bitwise identical whenever no fallback happens. The number of fallbacks is reported.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import os

import numpy as np

from science.numerical_relativity.fo_ccz4_principal_symbol import _rhs_components
from science.numerical_relativity.fo_ccz4_sbp_nonlinear import face, local_split, require


def robust_local_split(*args, **kwargs):
    """local_split; on LinAlgError, the same call with SciPy's gesvd SVD. Returns (split, fell_back)."""
    try:
        return local_split(*args, **kwargs), False
    except np.linalg.LinAlgError:
        import scipy.linalg
        numpy_svd = np.linalg.svd
        np.linalg.svd = lambda a: scipy.linalg.svd(a, full_matrices=True, lapack_driver="gesvd")
        try:
            return local_split(*args, **kwargs), True
        finally:
            np.linalg.svd = numpy_svd


def _penalties(job):
    """Worker: one split per key, then the penalty for each (jump, side, weight) of that key."""
    (h, chi, alpha, beta, axis, gauge, config, profile), items = job
    split, fell_back = robust_local_split(h, chi, alpha, beta, axis, gauge=gauge, config=config, profile=profile)
    return ([split.penalty(jump, side, weight) for jump, side, weight in items],
            split.condition, max(split.residual, split.symmetry_residual), fell_back)


class ParallelSATMixin:
    """Mix into a NonlinearSBPCube subclass to replace its rhs with the pooled version."""

    workers = int(os.environ.get("B2_WORKERS", max(1, min(20, (os.cpu_count() or 2) - 4))))
    _pool = None

    def _executor(self):
        if ParallelSATMixin._pool is None:
            ParallelSATMixin._pool = ProcessPoolExecutor(max_workers=self.workers)
        return ParallelSATMixin._pool

    def rhs(self, values, data, dt):
        q, targets = self.field(values), self.targets(data)
        state = self.state(q)
        cfl = self.cfl(state, dt)
        bulk = np.array(_rhs_components(self.volume.rhs(
            state, gauge=self.gauge, config=self.config, gauge_profile=self.profile)), dtype=np.float64)
        result = bulk.copy()
        groups, order = {}, []
        for axis in range(3):
            tangent = [d for d in range(3) if d != axis]
            for side in range(2):
                target = targets[2 * axis + side]
                weight = self.volume.axes[axis].norm_weights[0 if side == 0 else -1]
                for ij in np.ndindex(target.shape[1:]):
                    point = [0, 0, 0]
                    point[axis] = 0 if side == 0 else self.volume.grid.shape[axis] - 1
                    for d, index in zip(tangent, ij, strict=True):
                        point[d] = index
                    point = tuple(point)
                    cell = q[(slice(None), *point)]
                    key = (axis, cell[:11].tobytes())
                    if key not in groups:
                        coefficients = (state.base.h[point], state.base.chi[point], state.base.lapse[point],
                                        state.base.shift[point], axis, self.gauge, self.config, self.profile)
                        groups[key] = (coefficients, [])
                    groups[key][1].append((cell - target[(slice(None), *ij)], side, weight))
                    order.append((axis, side, ij, key, len(groups[key][1]) - 1))
        keys = list(groups)
        outputs = dict(zip(keys, self._executor().map(_penalties, [groups[k] for k in keys], chunksize=8)))
        worst_condition = max(o[1] for o in outputs.values())
        worst_residual = max(o[2] for o in outputs.values())
        svd_fallbacks = sum(o[3] for o in outputs.values())
        face_rows = []
        for axis in range(3):
            for side in range(2):
                target = targets[2 * axis + side]
                addition = np.zeros_like(target)
                for a, s, ij, key, index in order:
                    if a == axis and s == side:
                        addition[(slice(None), *ij)] = outputs[key][0][index]
                face(result, axis, side)[...] += addition
                face_rows.append(dict(axis=axis, side=side, sat_max_abs=float(np.max(abs(addition)))))
        require(np.all(np.isfinite(result)), "nonfinite coupled nonlinear RHS")
        require(np.array_equal(result[:11], bulk[:11]), "SAT modified ODE equations")
        report = dict(cfl=cfl, coefficients_sha256=hashlib.sha256(q[:11].tobytes()).hexdigest(),
                      basis_builds=len(groups), svd_fallbacks=svd_fallbacks, max_basis_condition=worst_condition,
                      max_characteristic_residual=worst_residual, faces=face_rows,
                      sat_max_abs=max(row["sat_max_abs"] for row in face_rows),
                      min_lapse=float(np.min(state.base.lapse)), min_chi=float(np.min(state.base.chi)),
                      det_h_max_error=float(np.max(abs(np.linalg.det(state.base.h) - 1.))))
        return result, report

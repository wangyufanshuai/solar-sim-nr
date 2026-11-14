import sys
import numpy as np
import pytest

from experiments.b1_gauge_wave import run_b1
from experiments.b3_native.geometry_terms import (
    python_geometry, scalar_contraction, tensor_report,
)
from kernel.native_evolve import slots, state_from_slots
from kernel.native_rhs import GEOMETRY_FIELDS, GEOMETRY_SLOT_COUNT, native_rhs_slots
from science.numerical_relativity import BSSNState3D, FOCCZ4State3D


def data():
    grid = run_b1.grid(8, 4)
    base = BSSNState3D.gauge_wave(grid, amplitude=.1)
    x = grid.coordinates[0]
    base.h[..., 0, 1] = base.h[..., 1, 0] = .003*np.sin(2*np.pi*x)
    base.conformal_gamma[..., 1] += .002*np.cos(2*np.pi*x)
    return grid, slots(FOCCZ4State3D.from_bssn_state(base))


def ricci_from_christoffels(fields, shape):
    pc, dpc = fields["physical_christoffel"], fields["d_physical_christoffel"]
    result = np.zeros((*shape, 3, 3))
    for first in range(3):
        for second in range(3):
            value = np.zeros(shape)
            for upper in range(3):
                value += dpc[..., upper, upper, first, second]-dpc[..., second, upper, first, upper]
                for ell in range(3):
                    value += (pc[..., ell, first, second]*pc[..., upper, ell, upper]
                              - pc[..., ell, first, upper]*pc[..., upper, ell, second])
            result[..., first, second] = value
    return result


def test_geometry_exports_reconstruct_actual_tensors_and_preserve_rhs():
    grid, q = data()
    before = q.tobytes()
    kw = dict(spacing=grid.spacing, order=4, gauge=run_b1.GAUGE, config=run_b1.CONFIG, threads=1)
    normal = native_rhs_slots(q, **kw)
    observed, native = native_rhs_slots(q, return_geometry=True, **kw)
    reference = python_geometry(q, grid)
    assert observed.tobytes() == normal.tobytes()
    assert q.tobytes() == before
    assert GEOMETRY_SLOT_COUNT == 298
    for name, shape in GEOMETRY_FIELDS:
        assert native[name].shape == (*grid.shape, *shape)
        assert np.all(np.isfinite(native[name]))
    for fields in (reference, native):
        np.testing.assert_array_equal(ricci_from_christoffels(fields, grid.shape), fields["ricci"])
        rz = fields["ricci"]+fields["covd_z"]+np.swapaxes(fields["covd_z"], -1, -2)
        np.testing.assert_array_equal(rz, fields["ricci_z"])
        assert np.max(np.abs(fields["covd_z"])) > 0
    chi = state_from_slots(q, grid).base.chi
    np.testing.assert_array_equal(scalar_contraction(native, chi, sequential=True), native["ricci_z_scalar"])
    np.testing.assert_array_equal(scalar_contraction(reference, chi, sequential=False), reference["ricci_z_scalar"])
    with pytest.raises(ValueError, match="either theta terms"):
        native_rhs_slots(q, return_geometry=True, return_theta_terms=True, **kw)


def test_geometry_ablation_does_not_replace_production_globals():
    import science.numerical_relativity.fo_ccz4_rhs as production
    inverse, geometry = np.linalg.inv, production._geometry
    grid, q = data()
    before = python_geometry(q, grid)
    ablated = python_geometry(q, grid, inverse="adjugate")
    after = python_geometry(q, grid)
    assert np.linalg.inv is inverse and production._geometry is geometry
    assert all(before[n].tobytes() == after[n].tobytes() for n, _ in GEOMETRY_FIELDS)
    assert all(np.all(np.isfinite(v)) for v in ablated.values())
    assert sys.getprofile() is None


def test_geometry_observer_restores_profiler_on_failure_and_refuses_existing_profiler():
    with pytest.raises((TypeError, AttributeError)):
        python_geometry(None, None)
    assert sys.getprofile() is None
    profiler = lambda *args: None
    sys.setprofile(profiler)
    try:
        with pytest.raises(RuntimeError, match="without a profiler"):
            python_geometry(None, None)
        assert sys.getprofile() is profiler
    finally:
        sys.setprofile(None)


def test_tensor_report_keeps_location_scale_and_signed_values():
    grid, q = data()
    ref = python_geometry(q, grid)
    actual = {name: value.copy() for name, value in ref.items()}
    actual["ricci"][2, 1, 3, 0, 1] += .25
    row = next(r for r in tensor_report(ref, actual) if r["field"] == "ricci")
    assert row["worst_index"] == [2, 1, 3, 0, 1]
    assert row["changed_elements"] == 1
    assert row["absolute_difference"] == pytest.approx(.25)
    assert row["native_at_worst"]-row["reference_at_worst"] == pytest.approx(.25)
    actual["ricci"][0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="matching finite"):
        tensor_report(ref, actual)


def test_registered_four_stage_run_compares_raw_rhs_before_inplace_gauge_correction(tmp_path):
    from pathlib import Path
    from experiments.b3_native.geometry_terms import run
    baseline = Path(__file__).resolve().parents[1]/"docs/gauge-stage/theta-terms/before.npz"
    result = run(tmp_path/"run", baseline)
    assert result["instrumentation_pass"]
    assert len(result["stages"]) == 4
    for stage in result["stages"]:
        assert stage["all_58_outputs_bitwise_equal_before"]
        assert stage["contraction"]["python_numpy_reconstruction_error"] == 0
        assert stage["contraction"]["native_sequential_reconstruction_error"] == 0

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
    assert GEOMETRY_SLOT_COUNT == 397
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
    ablated = python_geometry(q, grid, inverse="adjugate", derivative="reciprocal")
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
    assert result["historical_replay_relative_pass"], result["stages"]
    for stage in result["stages"]:
        assert stage["all_58_outputs_bitwise_equal_fresh_control"]
        assert stage["all_58_outputs_bitwise_equal_before"] == stage["all_58_outputs_bitwise_equal_control"]
        assert stage["historical_rhs_comparison"]["active_slots_pass_relative_1e_minus12"]
        assert stage["contraction"]["python_numpy_reconstruction_error"] == 0
        assert stage["contraction"]["native_sequential_reconstruction_error"] == 0


def test_reciprocal_derivative_retains_pairing_and_fourth_order_accuracy():
    from experiments.b3_native.geometry_terms import reciprocal_derivative
    from science.numerical_relativity.bssn3d import _d1
    errors = []
    for nx in (16, 32, 64):
        grid = run_b1.grid(nx, 4)
        state = BSSNState3D.gauge_wave(grid, amplitude=.1)
        x = grid.coordinates[0]
        values = np.sin(2*np.pi*x)
        d = reciprocal_derivative(values, state, 0)
        errors.append(np.max(np.abs(d-2*np.pi*np.cos(2*np.pi*x))))
        np.testing.assert_allclose(d, _d1(values, state, 0), atol=2e-15, rtol=2e-15)
        np.testing.assert_array_equal(reciprocal_derivative(values, state, 1), np.zeros(grid.shape))
        np.testing.assert_array_equal(reciprocal_derivative(np.ones(grid.shape), state, 0), np.zeros(grid.shape))
    assert all(15 < errors[i]/errors[i+1] < 17 for i in (0, 1))


def test_complete_rhs_ablation_keeps_kwdefaults_and_harmonic_derivative_unchanged():
    from experiments.b3_native.geometry_terms import arithmetic_ablation_rhs
    from experiments.b1_gauge_wave.harmonic import harmonic_fo_ccz4_rhs
    from experiments.b3_native.gauge_stage_diagnostic import pack_rhs
    from science.numerical_relativity.fo_ccz4_rhs import _fo_ccz4_volume_rhs_3d
    original_defaults = _fo_ccz4_volume_rhs_3d.__kwdefaults__.copy()
    derivative = harmonic_fo_ccz4_rhs.__globals__["_d1"]
    grid, q = data()
    state = state_from_slots(q, grid)
    before = pack_rhs(harmonic_fo_ccz4_rhs(state, gauge=run_b1.GAUGE, config=run_b1.CONFIG))
    ablated = pack_rhs(arithmetic_ablation_rhs()(state, gauge=run_b1.GAUGE, config=run_b1.CONFIG))
    after = pack_rhs(harmonic_fo_ccz4_rhs(state, gauge=run_b1.GAUGE, config=run_b1.CONFIG))
    assert np.all(np.isfinite(ablated))
    assert _fo_ccz4_volume_rhs_3d.__kwdefaults__ == original_defaults
    assert harmonic_fo_ccz4_rhs.__globals__["_d1"] is derivative
    assert before.tobytes() == after.tobytes()


@pytest.mark.parametrize("time", [15, 18, 22])
def test_late_snapshot_inputs_use_native_stages_and_keep_source_unchanged(tmp_path, time):
    from pathlib import Path
    from experiments.b3_native.geometry_terms import snapshot_stage_inputs
    from kernel.native_evolve import make_rhs
    source = Path(__file__).resolve().parents[1]/f"docs/gauge-stage/baseline-t{time}.npz"
    before = source.read_bytes()
    grid = run_b1.grid(32, 4)
    states, controls, actual_time, dt = snapshot_stage_inputs(source, grid)
    ordinary = make_rhs(spacing=grid.spacing, order=4, gauge=run_b1.GAUGE,
                        config=run_b1.CONFIG, harmonic=True, threads=1)
    assert actual_time == time and dt == 1/128
    assert source.read_bytes() == before
    with np.load(source, allow_pickle=False) as saved:
        np.testing.assert_array_equal(states[0], saved["q"])
    for stage in range(4):
        assert controls[stage].tobytes() == ordinary(states[stage]).tobytes()
        if stage < 3:
            factor = (.5, .5, 1.)[stage]
            np.testing.assert_array_equal(states[stage+1], states[0]+factor*dt*controls[stage])


def test_snapshot_validation_and_time_metadata(tmp_path):
    from pathlib import Path
    from experiments.b3_native.geometry_terms import snapshot_stage_inputs, run
    grid, q = data()
    invalid = tmp_path/"invalid.npz"
    np.savez(invalid, q=q, time=18., dt=1/128)
    with pytest.raises(ValueError, match="registered grid"):
        snapshot_stage_inputs(invalid, run_b1.grid(32, 4))
    source = Path(__file__).resolve().parents[1]/"docs/gauge-stage/baseline-t18.npz"
    with np.load(source, allow_pickle=False) as saved:
        q = saved["q"].copy()
    np.savez(invalid, q=q, time=float("nan"), dt=1/128)
    with pytest.raises(ValueError, match="nonnegative time"):
        snapshot_stage_inputs(invalid, run_b1.grid(32, 4))
    np.savez(invalid, q=q, time=18., dt=1/64)
    with pytest.raises(ValueError, match="registered dt"):
        snapshot_stage_inputs(invalid, run_b1.grid(32, 4))
    result = run(tmp_path/"late", snapshot=source)
    assert result["instrumentation_pass"]
    assert result["base_time"] == 18 and result["dt"] == 1/128
    assert result["control_kind"] == "fresh_ordinary_native_rhs"
    assert "before_sha256" not in result
    for stage in result["stages"]:
        assert "all_58_outputs_bitwise_equal_before" not in stage
        assert stage["base_time"] == 18
        assert 18 <= stage["nominal_stage_time"] <= 18+1/128


def test_historical_drift_is_reported_separately_from_observer_preservation(tmp_path):
    from pathlib import Path
    from experiments.b3_native.geometry_terms import run
    source = Path(__file__).resolve().parents[1]/"docs/gauge-stage/theta-terms/before.npz"
    with np.load(source, allow_pickle=False) as saved:
        states, rhs = saved["states"].copy(), saved["rhs"].copy()
    rhs[0, 0, 0, 0, 0] += .01
    altered = tmp_path/"altered.npz"
    np.savez_compressed(altered, states=states, rhs=rhs)
    result = run(tmp_path/"observed", altered)
    assert result["instrumentation_pass"]
    assert not result["historical_replay_bitwise_pass"]
    assert not result["historical_replay_relative_pass"]
    assert not result["stages"][0]["all_58_outputs_bitwise_equal_before"]
    assert result["stages"][0]["historical_rhs_comparison"]["global_max_absolute_difference"] > .009

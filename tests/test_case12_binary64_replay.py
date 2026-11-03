"""Small independent-replay controls; never run a complete actual RHS case."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.case12_binary64_v1 import replay as R


class StopControl(RuntimeError):
    pass


def stop():
    raise StopControl("owned control stop")


def test_replay_registered_schedule_has_964_total_with_separate_paths():
    plan = R.evaluation_plan()
    assert len(plan) == len(set(plan)) == 482
    assert sum(row[0] == "main" for row in plan) == 464
    assert sum(row[0] == "background" for row in plan) == 2
    assert sum(row[0] == "negative" for row in plan) == 16
    assert 2 * len(plan) == 964
    assert plan[0] == ("background", 2, None, None, None, "baseline")
    assert plan[465] == ("main", 4, 2.0**-12, 57, 1, "baseline")
    assert plan[-1] == ("negative", 4, 2.0**-12, 11, 1, "stored_trace_diagnostic")
    assert {row[3] for row in plan if row[5] == "kappa1_doubled"} == {18}
    assert {row[3] for row in plan if row[5] == "stored_trace_diagnostic"} == {11}


@pytest.mark.parametrize("mode_index", [1, 2, 3, 4, 5, 6])
def test_replay_projection_preserves_all_signed_modes(mode_index):
    xyz = np.indices((8, 8, 8), dtype=np.float64) / 8.0
    mode = R.MODES[mode_index]
    phase = 2.0 * np.pi * sum(mode[a] * xyz[a] for a in range(3))
    field = 3.0 + 2.0 * np.cos(phase) + 4.0 * np.sin(phase)
    before = field.copy()
    projected = R.project_explicit(field)
    expected = np.zeros(13, dtype=np.complex128)
    expected[0] = 3.0
    expected[mode_index] = 1.0 - 2.0j
    expected[mode_index + 6] = 1.0 + 2.0j
    np.testing.assert_allclose(projected, expected, rtol=0.0, atol=3.0e-15)
    np.testing.assert_allclose(
        R.reconstruct_explicit(projected), field, rtol=0.0, atol=2.0e-14
    )
    np.testing.assert_array_equal(field, before)


def test_replay_unregistered_high_mode_is_visible_in_full_grid():
    x = np.indices((8, 8, 8), dtype=np.float64)[0] / 8.0
    field = np.cos(4.0 * np.pi * x)
    projections = R.project_explicit(field)
    assert np.max(np.abs(projections)) < 1.0e-15
    residual = field - R.reconstruct_explicit(projections)
    assert np.max(np.abs(residual)) > 0.99


@pytest.mark.parametrize(
    "order, expected", [(2, 4.0 * np.sqrt(2.0)), (4, (16.0 * np.sqrt(2.0) - 4.0) / 3.0)]
)
def test_replay_modified_wave_number_is_d1_not_continuum_or_d2(order, expected):
    wave = R.modified_wave_numbers(order)
    np.testing.assert_allclose(wave[1], [expected, 0.0, 0.0], rtol=0.0, atol=2.0e-15)
    np.testing.assert_allclose(
        wave[4], [expected, expected, 0.0], rtol=0.0, atol=2.0e-15
    )
    np.testing.assert_array_equal(wave[7:], -wave[1:7])
    assert abs(expected - 2.0 * np.pi) > 0.05
    standard_d2_magnitude = 4.0 * 8.0**2 * np.sin(np.pi / 8.0) ** 2
    assert abs(expected**2 - standard_d2_magnitude) > 1.0


@pytest.mark.parametrize("order", [2, 4])
def test_replay_hand_recovery_keeps_mixed_coefficient_without_factor_two(order):
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
    np.testing.assert_allclose(
        answer[:, 0],
        [3.0, 2.0, -1.0, 5.0, 4.0, 7.0, -3.0, 2.0, 6.0, -1.0],
        rtol=0.0,
        atol=5.0e-15,
    )
    assert abs(answer[5, 0] - 14.0) > 6.0
    np.testing.assert_allclose(resolution[5, 0], 1.5 / (k * k), rtol=0.0, atol=1.0e-17)


def test_replay_symbols_use_original_shifted_generator_and_real_zero_order():
    first = np.zeros((4, 2, 2), dtype=np.float64)
    first[0] = [[2.0, 1.0], [0.0, -3.0]]
    first[1] = [[0.75, 2.0], [0.0, 0.75]]
    symbol = R.assemble_symbols(first, 2)
    k = 4.0 * np.sqrt(2.0)
    np.testing.assert_allclose(
        symbol[:, :, 1], first[0] + 1.0j * k * first[1], rtol=0.0, atol=2.0e-15
    )
    assert symbol[0, 0, 1].imag != 0.0
    np.testing.assert_array_equal(symbol[:, :, 0], first[0])


@pytest.mark.parametrize(
    "block, mode, expected_multiplier",
    [
        (4, 1, -32.0),
        (5, 4, -32.0),
        (6, 5, -32.0),
        (7, 2, -32.0),
        (8, 6, -32.0),
        (9, 3, -32.0),
    ],
)
def test_replay_each_hessian_block_uses_full_monomial(block, mode, expected_multiplier):
    matrix = np.zeros((10, 1, 1), dtype=np.float64)
    matrix[block, 0, 0] = 3.0
    values = R.assemble_symbols(matrix, 2)
    np.testing.assert_allclose(
        values[0, 0, mode], 3.0 * expected_multiplier, rtol=0.0, atol=3.0e-14
    )
    assert values[0, 0, 0] == 0.0


def test_replay_shape_derivatives_follow_composed_first_derivative():
    projected = np.zeros(13, dtype=np.complex128)
    projected[0] = 0.125
    projected[4] = projected[10] = 0.0625
    basis = R.derivative_shapes(projected, 2)
    xyz = np.indices((8, 8, 8), dtype=np.float64) / 8.0
    phase = 2.0 * np.pi * (xyz[0] + xyz[1])
    np.testing.assert_allclose(basis[5], -4.0 * np.cos(phase), rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(basis[4], basis[5], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(basis[6], 0.0, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("operation", ["projection", "reconstruction"])
def test_replay_projection_deadline_propagates(operation):
    with pytest.raises(StopControl):
        if operation == "projection":
            R.project_explicit(np.zeros((8, 8, 8), dtype=np.float64), check=stop)
        else:
            R.reconstruct_explicit(np.zeros(13, dtype=np.complex128), check=stop)


@pytest.mark.parametrize(
    "invalid", ["float32", "nan", "wrong_shape", "list", "complex"]
)
def test_replay_projection_rejects_noncanonical_fields(invalid):
    value = np.zeros((8, 8, 8), dtype=np.float64)
    if invalid == "float32":
        value = value.astype(np.float32)
    elif invalid == "nan":
        value[7, 7, 7] = np.nan
    elif invalid == "wrong_shape":
        value = value[:7]
    elif invalid == "list":
        value = value.tolist()
    else:
        value = value.astype(np.complex128)
    with pytest.raises(ValueError):
        R.project_explicit(value)


@pytest.mark.parametrize(
    "case", ["zero_absolute", "nonzero_unresolved", "relative_miss", "pass"]
)
def test_replay_coefficient_gate_separates_zero_signal_and_resolution(case):
    ref = np.zeros((4, 1, 1), dtype=np.float64)
    ref[0, 0, 0] = 1.0
    actual = ref.copy()
    resolution = np.full(ref.shape, 1.0e-12)
    if case == "zero_absolute":
        actual[1, 0, 0] = 0.01
    elif case == "nonzero_unresolved":
        resolution[0, 0, 0] = 0.1
    elif case == "relative_miss":
        actual[0, 0, 0] += 0.002
        resolution[0, 0, 0] = 0.001
    result = R.coefficient_gates(actual, resolution, ref, 2.0**-12, fine=True)
    assert result["accepted"] is (case == "pass")
    if case == "relative_miss":
        assert result["absolute"] is True
        assert result["relative_nonzero"] is False
    if case == "nonzero_unresolved":
        assert result["resolved_nonzero"] is False


@pytest.mark.parametrize(
    "fine_error, expected", [(0.1, True), (0.2, False), (0.0, True)]
)
def test_replay_convergence_requires_decrease_only_above_resolution(
    fine_error, expected
):
    errors = np.zeros((2, 2, 4, 1, 1), dtype=np.float64)
    errors[:, 0, 0, 0, 0] = 1.0
    errors[:, 1, 0, 0, 0] = fine_error
    nu = np.full(errors.shape, 1.0e-12)
    result = R.convergence_gate(errors, nu)
    assert result["accepted"] is expected
    assert result["tested_rows"] == 2
    assert result["roundoff_limited_rows"] == 6


def test_replay_full_grid_gate_detects_last_node_and_raw_even_response():
    plus = np.zeros((101, 8, 8, 8), dtype=np.float64)
    minus = plus.copy()
    base = plus.copy()
    plus[100, 7, 7, 7] = 1.0
    odd_projection = R.project_explicit(plus / (2.0 * 2.0**-12))
    result = R._field_statistics(
        plus, minus, base, odd_projection, base, 2.0**-12, 1.0, np.ones(101)
    )
    assert result[100, 2] == 511.0
    assert result[100, 0] == 2048.0
    assert result[100, 5] == 1.0
    assert result[100, 10] == 0.0
    assert result[0, 10] == 1.0


@pytest.mark.parametrize(
    "case", ["valid", "different_slot", "unresolved", "outside_target"]
)
def test_replay_negative_requires_causal_target_and_same_slot(case):
    change = np.zeros((101, 13), dtype=np.complex128)
    error = change.copy()
    nu = np.full((101, 13), 1.0e-8)
    limits = np.ones((101, 13), dtype=np.float64)
    change[18, 1] = error[18, 1] = 2.0
    if case == "different_slot":
        error[18, 1] = 0.0
        error[17, 2] = 2.0
    elif case == "unresolved":
        nu[18, 1] = 1.0
    elif case == "outside_target":
        change[:] = error[:] = 0.0
        change[19, 1] = error[19, 1] = 2.0
    result = R.negative_gate(change, error, nu, limits, (17, 18), fine=True)
    assert result["rejected"] is (case == "valid")


def test_replay_roundtrip_gate_covers_every_slot_and_node():
    q = np.zeros((58, 8, 8, 8), dtype=np.float64)
    q[6] = -0.2
    recovered = q.copy()
    recovered[57, 7, 7, 7] += 256.0 * R.U
    stats = R._roundtrip_statistics(q, recovered)
    assert stats.shape == (58, 3)
    assert stats[57, 2] == 2.0
    assert np.count_nonzero(stats[:, 0]) == 1


@pytest.mark.parametrize("failure", ["guard", "sink"])
def test_replay_early_abort_never_calls_actual_scientific_evaluator(
    monkeypatch, failure
):
    from experiments.case12_binary64_v1 import kernel

    q0 = np.zeros(58, dtype=np.float64)
    xyz = np.indices((8, 8, 8), dtype=np.float64) / 8.0
    f = np.ones((8, 8, 8), dtype=np.float64)
    for mode in R.WAVES:
        f += np.cos(2.0 * np.pi * sum(mode[a] * xyz[a] for a in range(3)))
    f /= 8.0
    context = SimpleNamespace(q0=q0, scales=np.ones(58), f=f, modes=R.MODES, W=R.WAVES)
    monkeypatch.setattr(kernel, "build_context", lambda parameters, check: context)
    monkeypatch.setattr(
        R, "independent_preflight", lambda *args, **kwargs: {"accepted": True}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("a unit control must not evaluate the full RHS")

    monkeypatch.setattr(kernel, "evaluate_state", forbidden)
    saved = {}

    def sink(name, array):
        assert name == "q0"
        raise StopControl("first output rejected")

    with pytest.raises(StopControl):
        R.run(
            {
                "parameters": {},
                "rhs": np.zeros((4, 58, 58)),
                "constraints": np.zeros((10, 43, 58)),
            },
            guard=stop if failure == "guard" else lambda: None,
            check=lambda: None,
            array_sink=sink,
            json_sink=lambda name, record: saved.update({name: record}),
        )
    if failure == "sink":
        assert saved["partial_calls"]["records"] == []
        assert saved["acceptance"]["completed"] is False
        assert saved["acceptance"]["accepted"] is False
        assert saved["acceptance"]["started_evaluations"] == 0


def test_replay_has_separate_projection_recovery_and_acceptance_source():
    source = Path(R.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "kernel"
    }
    assert kernel_attributes == {"build_context", "evaluate_state"}
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"fft", "fftn", "ifft", "ifftn"}
        for node in ast.walk(tree)
    )
    manifest = R.replay_manifest()
    assert manifest["independent_physical_formulation"] is False
    assert manifest["rigorous_roundoff_certificate"] is False
    assert manifest["planned_state_evaluations"] == 482


def test_replay_failed_first_evaluation_preserves_started_not_completed(monkeypatch):
    from experiments.case12_binary64_v1 import kernel

    xyz = np.indices((8, 8, 8), dtype=np.float64) / 8.0
    shape = np.ones((8, 8, 8), dtype=np.float64)
    for mode in R.WAVES:
        shape += np.cos(2.0 * np.pi * sum(mode[a] * xyz[a] for a in range(3)))
    shape /= 8.0
    context = SimpleNamespace(
        q0=np.zeros(58), scales=np.ones(58), f=shape, modes=R.MODES, W=R.WAVES
    )
    monkeypatch.setattr(kernel, "build_context", lambda parameters, check: context)
    monkeypatch.setattr(
        R, "independent_preflight", lambda *args, **kwargs: {"accepted": True}
    )

    def fail(*args, **kwargs):
        raise StopControl(
            "injected evaluator interruption, no scientific function called"
        )

    monkeypatch.setattr(kernel, "evaluate_state", fail)
    saved = {}
    with pytest.raises(StopControl):
        R.run(
            {
                "parameters": {},
                "rhs": np.zeros((4, 58, 58)),
                "constraints": np.zeros((10, 43, 58)),
            },
            guard=lambda: None,
            check=lambda: None,
            array_sink=lambda name, array: None,
            json_sink=lambda name, record: saved.update({name: record}),
        )
    assert saved["progress"]["started"] == 1
    assert saved["progress"]["completed"] == 0
    assert saved["partial_calls"]["completed"] == 0
    assert saved["partial_calls"]["records"][0]["completed"] is False
    assert saved["acceptance"]["completed_evaluations"] == 0
    assert saved["acceptance"]["accepted"] is False


def test_replay_array_registration_preserves_all_zero_and_last_slots():
    assert R.ARRAY_SPECS["main_projections"] == (2, 2, 58, 2, 101, 13, 2)
    assert R.ARRAY_SPECS["constraint_coefficients"] == (2, 2, 10, 43, 58)
    assert R.ARRAY_SPECS["roundtrip_stats"] == (482, 58, 3)
    assert R.ARRAY_SPECS["field_stats"] == (2, 2, 58, 101, 11)
    assert all(
        type(value) is tuple and all(type(n) is int and n > 0 for n in value)
        for value in R.ARRAY_SPECS.values()
    )


@pytest.mark.parametrize("order", [2, 4])
def test_replay_weighted_full_design_has_no_missing_mixed_column(order):
    fhat = np.full(13, 0.0625, dtype=np.complex128)
    fhat[0] = 0.125
    record = R.design_diagnostics(fhat, order)
    assert [row["rank"] for row in record["rows"]] == [4, 10]
    assert record["accepted"] is True
    fhat[4] = fhat[10] = 0.0
    missing = R.design_diagnostics(fhat, order)
    assert missing["rows"][1]["rank"] == 9
    assert missing["accepted"] is False


@pytest.mark.parametrize("fault", [False, True])
def test_replay_actual_scalar_stencils_are_checked_without_full_rhs(fault):
    from science.numerical_relativity.bssn3d import _d1
    from science.numerical_relativity.fo_ccz4_relaxation_convergence import derivative
    from science.numerical_relativity.nr3d import Grid3D

    xyz = np.indices((8, 8, 8), dtype=np.float64) / 8.0
    field = np.ones((8, 8, 8), dtype=np.float64)
    for mode in R.WAVES:
        field += np.cos(2.0 * np.pi * sum(mode[a] * xyz[a] for a in range(3)))
    field /= 8.0

    def controlled_derivative(values, axis, order):
        return derivative(values, axis, order) + (0.01 if fault else 0.0)

    context = SimpleNamespace(
        f=field,
        rhs=SimpleNamespace(_d1=_d1),
        diagnostic=SimpleNamespace(derivative=controlled_derivative),
        checkpoint=SimpleNamespace(Grid3D=Grid3D),
    )
    record = R.independent_preflight(context, R.project_explicit(field))
    assert record["accepted"] is (not fault)
    assert record["scalar_derivative_calls"] == 36
    assert record["full_rhs_calls"] == record["full_diagnostic_calls"] == 0

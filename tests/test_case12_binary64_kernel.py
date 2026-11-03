"""Small independent controls; never run the full 482-evaluation campaign here."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.case12_binary64_v1 import kernel as K


def test_binary64_registration_and_uncompressed_budget():
    assert K.ORDERS == (2, 4)
    assert K.EPSILONS == (2**-10, 2**-12)
    assert K.SIGNS == (-1, 1)
    assert K.MODES == (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (-1, -1, 0),
        (-1, 0, -1),
        (0, -1, -1),
    )
    assert K.EVALUATIONS == 2 + 2 * 2 * 58 * 2 + 2 * 2 * 2 * 2 == 482
    assert len(K.ARRAY_SPECS) == 26
    assert sum(math.prod(shape) * 8 for shape in K.ARRAY_SPECS.values()) == 25_506_688
    assert K.ARRAY_SPECS["main_projections"] == (2, 2, 58, 2, 101, 13, 2)
    assert K.ARRAY_SPECS["roundtrip_stats"] == (482, 58, 3)
    assert K.ARRAY_SPECS["negative_field_scales"] == (2, 2, 2, 101)
    assert K.NEGATIVES == (
        ("kappa1_doubled", 18, (17, 18)),
        ("stored_trace_diagnostic", 11, (59, 60, 61)),
    )


@pytest.mark.parametrize(
    "value",
    [np.array([1], dtype=np.float32), [1.0], np.array([np.nan]), np.array([np.inf])],
)
def test_binary64_rejects_invalid_real_array(value):
    with pytest.raises(ValueError):
        K.finite_array(value)


def test_binary64_array_shape_and_complex_pair_roundtrip():
    with pytest.raises(ValueError):
        K.finite_array(np.zeros((2, 3)), (3, 2))
    value = np.array([[1 + 2j, -3j], [0j, -2.5 + 4j]])
    record = K.as_pairs(value)
    assert record.shape == (2, 2, 2)
    np.testing.assert_array_equal(record[0], [[1, 2], [0, -3]])
    np.testing.assert_array_equal(K.from_pairs(record), value)
    with pytest.raises(ValueError):
        K.from_pairs(np.zeros((2, 3)))


@pytest.mark.parametrize("value", ["2/4", "1.0", "01", 1, True])
def test_binary64_rational_input_is_canonical(value):
    with pytest.raises(ValueError):
        K.exact_float(value)


def test_binary64_rational_conversion_retains_sign():
    assert K.exact_float("-7/8") == -0.875
    assert K.exact_float("0") == 0.0


def _coordinates():
    return np.indices((8, 8, 8), dtype=np.float64) / 8


def test_binary64_fft_projection_full_signed_known_signal():
    x, y, _ = _coordinates()
    field = 3 + 2 * np.cos(2 * np.pi * x) + 4 * np.sin(2 * np.pi * (x + y))
    projection = K.project_fields(field)
    expected = np.zeros(13, dtype=np.complex128)
    expected[0], expected[1], expected[7] = 3, 1, 1
    expected[4], expected[10] = -2j, 2j
    np.testing.assert_allclose(projection, expected, rtol=0, atol=3e-15)
    np.testing.assert_allclose(K.reconstruct(projection), field, rtol=0, atol=4e-15)


def test_binary64_offmode_signal_is_not_erased_by_known_mode_fit():
    x, _, _ = _coordinates()
    field = 2 + np.cos(4 * np.pi * x)
    residual = field - K.reconstruct(K.project_fields(field))
    np.testing.assert_allclose(residual, np.cos(4 * np.pi * x), rtol=0, atol=1e-15)
    assert np.max(np.abs(residual)) > 0.99


def test_binary64_shape_projection_has_dc_and_all_twelve_signed_modes():
    coordinates = _coordinates()
    field = np.ones((8, 8, 8))
    for mode in K.W:
        field += np.cos(2 * np.pi * sum(mode[a] * coordinates[a] for a in range(3)))
    field /= 8
    expected = np.full(13, 1 / 16, dtype=np.complex128)
    expected[0] = 1 / 8
    np.testing.assert_allclose(K.project_fields(field), expected, rtol=0, atol=1e-16)


@pytest.mark.parametrize("order", [2, 4])
def test_binary64_modified_wave_numbers_manual_radical(order):
    actual = K.wave_numbers(order)
    k = 4 * math.sqrt(2) if order == 2 else (16 * math.sqrt(2) - 4) / 3
    np.testing.assert_allclose(actual, np.array(K.MODES) * k, rtol=0, atol=2e-15)
    assert np.count_nonzero(actual[0]) == 0


@pytest.mark.parametrize("order", [True, 3, 2.0, "4"])
def test_binary64_unregistered_difference_order_refused(order):
    with pytest.raises(ValueError):
        K.wave_numbers(order)


def test_binary64_recovery_uses_mixed_coefficient_once_and_propagates_uncertainty():
    k = np.asarray(K.MODES, dtype=np.float64) * 2
    signal = np.zeros((101, 13), dtype=np.complex128)
    signal[0] = 7 + 1j * (3 * k[:, 0] - 5 * k[:, 1] + 11 * k[:, 2])
    # Independent full mixed monomial coefficients; none include a hidden 2.
    c = np.array([13, -2, 3, 5, 7, 11, 17, 19, 23, 29], dtype=np.float64)
    for mode, (x, y, z) in enumerate(k):
        signal[58, mode] = (
            13
            + 1j * (-2 * x + 3 * y + 5 * z)
            - 7 * x * x
            - 11 * x * y
            - 17 * x * z
            - 19 * y * y
            - 23 * y * z
            - 29 * z * z
        )
    rhs, constraint, rn, cn = K.recover(signal, np.ones((101, 13)), k)
    np.testing.assert_array_equal(rhs[:, 0], [7, 3, -5, 11])
    np.testing.assert_array_equal(constraint[:, 0], c)
    np.testing.assert_array_equal(rn[:, 0], [1, 0.5, 0.5, 0.5])
    np.testing.assert_array_equal(
        cn[:, 0], [1, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 0.5, 1.5, 0.5]
    )
    assert np.count_nonzero(rhs[:, 1:]) == 0
    assert np.count_nonzero(constraint[:, 1:]) == 0


def test_binary64_symbols_use_successive_first_derivative_multipliers():
    rhs = np.zeros((4, 58, 58))
    constraints = np.zeros((10, 43, 58))
    rhs[:, 2, 5] = [2, 3, 5, 7]
    constraints[:, 4, 7] = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43]
    k = np.asarray(K.MODES, dtype=np.float64) * 3
    answer = K.symbols({"rhs": rhs, "constraints": constraints}, k)
    # x+y mode: a mixed derivative contributes -(3*3)*29, exactly once.
    assert answer[2, 5, 4] == 2 + 24j
    assert answer[62, 7, 4] == 11 - 9 * (23 + 29 + 37) + 90j
    assert answer[62, 7, 10] == answer[62, 7, 4].conjugate()


@pytest.mark.parametrize("order", [2, 4])
def test_binary64_design_has_full_real_coefficient_rank(order):
    fhat = np.full(13, 1 / 16, dtype=np.complex128)
    fhat[0] = 1 / 8
    record = K.design_diagnostics(fhat, order)
    assert record["accepted"]
    assert [r["rank"] for r in record["rows"]] == [4, 10]
    assert [r["shape"] for r in record["rows"]] == [[26, 4], [26, 10]]
    assert all(math.isfinite(r["condition"]) for r in record["rows"])
    assert all(len(r["singular_values"]) == r["coefficients"] for r in record["rows"])


def test_binary64_missing_modes_fail_identification_without_inventing_condition():
    fhat = np.zeros(13, dtype=np.complex128)
    fhat[0] = 1 / 8
    record = K.design_diagnostics(fhat, 2)
    assert not record["accepted"]
    assert [r["rank"] for r in record["rows"]] == [1, 1]
    assert all(r["condition"] is None for r in record["rows"])


def test_binary64_scalar_controls_check_both_paths_and_second_derivatives():
    from science.numerical_relativity import (
        bssn3d,
        fo_ccz4_relaxation_convergence,
        nr3d,
    )

    coordinates = _coordinates()
    f = (
        1
        + sum(
            np.cos(2 * np.pi * sum(mode[a] * coordinates[a] for a in range(3)))
            for mode in K.W
        )
    ) / 8
    context = SimpleNamespace(
        np=np,
        f=f,
        checkpoint=SimpleNamespace(Grid3D=nr3d.Grid3D),
        rhs=SimpleNamespace(_d1=bssn3d._d1),
        diagnostic=fo_ccz4_relaxation_convergence,
    )
    record = K.scalar_controls(context)
    assert record["accepted"]
    assert record["full_state_evaluations"] == 0
    assert record["scalar_comparisons"] == 36
    assert record["compared_grid_slots"] == 36 * 512
    assert {r["derivative_index"] for r in record["stencils"]} == set(range(1, 10))
    assert {r["method"] for r in record["stencils"]} == {
        "actual_rhs_d1",
        "actual_diagnostic_derivative",
    }
    context.rhs = SimpleNamespace(_d1=lambda value, state, axis: np.zeros_like(value))
    assert not K.scalar_controls(context)["accepted"]


def test_binary64_physical_rows_use_current_metric_and_flat_auxiliary_order():
    slots = np.zeros((58, 8, 8, 8))
    slots[:6] = np.array([2, 1, 0, 3, 0, 4])[:, None, None, None]
    grid = (8, 8, 8)
    fields = {
        name: np.full(grid, value, dtype=np.float64)
        for name, value in (
            ("H", 100),
            ("Theta", 104),
            ("det_h", 141),
            ("trace_a", 142),
        )
    }
    for name, count, start in (
        ("M", 3, 101),
        ("A", 3, 108),
        ("B", 9, 111),
        ("D", 18, 120),
        ("P", 3, 138),
    ):
        fields[name] = np.broadcast_to(
            np.arange(start, start + count, dtype=np.float64), (*grid, count)
        ).copy()
    fields["Ghat"] = np.broadcast_to(np.array([1.0, 2.0, 3.0]), (*grid, 3)).copy()
    actual = K.physical_rows(fields, slots)
    expected = np.arange(100, 143, dtype=np.float64)
    expected[5:8] = [2, 3.5, 6]  # h @ Ghat / 2, with non-diagonal h.
    np.testing.assert_array_equal(
        actual, np.broadcast_to(expected[:, None, None, None], actual.shape)
    )


def constraint_fields(data):
    """Small source-shaped real expression for the registered AST mutation."""
    q, mixed = data
    actual_trace = np.trace(mixed, axis1=-2, axis2=-1)
    return actual_trace, actual_trace**2 - np.trace(mixed @ mixed)


def test_binary64_trace_mutant_changes_actual_assignment_without_mutating_source():
    q = np.zeros(58)
    q[17] = 5
    mixed = np.diag([1.0, 2.0, 3.0])
    mutated, audit = K.compile_trace_mutant(constraint_fields)
    assert constraint_fields((q, mixed)) == (6, 22)
    assert mutated((q, mixed)) == (5, 11)
    assert audit["assignment_replacements"] == 1
    assert audit["hamiltonian_trace_squared_input_also_changed"] is True
    assert audit["original_ast"] != audit["mutated_ast"]
    assert constraint_fields((q, mixed)) == (6, 22)


def test_binary64_matrix_gate_zero_and_measurable_rows_are_distinct():
    reference = np.array([[[1.0, 0.0]]])
    measured = np.broadcast_to(reference, (2, 2, 1, 1, 2)).copy()
    nu = np.full_like(measured, 1e-12)
    exact = K.matrix_acceptance(measured, reference, nu)
    assert exact["accepted"]
    assert exact["measurable_rows"] == 0
    assert exact["roundoff_limited_rows"] == 2
    measured[:, 0, 0, 0, 0] += 1e-5
    measured[:, 1, 0, 0, 0] += 1e-7
    shrinking = K.matrix_acceptance(measured, reference, nu)
    assert shrinking["accepted"] and shrinking["measurable_rows"] == 2
    measured[:, 1, 0, 0, 0] = 1 + 3e-6
    assert not K.matrix_acceptance(measured, reference, nu)["amplitude_contraction"]


def test_binary64_matrix_gate_does_not_call_unresolved_nonzero_exact():
    reference = np.array([[[1e-10]]])
    measured = np.broadcast_to(reference, (2, 2, 1, 1, 1)).copy()
    result = K.matrix_acceptance(measured, reference, np.full_like(measured, 1e-9))
    assert result["all_slots"] and result["fine_nonzero_relative"]
    assert not result["fine_nonzero_resolved"]
    assert not result["accepted"]


def test_binary64_zero_reference_slot_failure_is_retained():
    reference = np.zeros((1, 1, 2))
    measured = np.zeros((2, 2, 1, 1, 2))
    measured[:, :, 0, 0, 1] = 1
    result = K.matrix_acceptance(measured, reference, np.full_like(measured, 1e-12))
    assert not result["all_slots"]
    assert not result["accepted"]


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_binary64_matrix_gate_rejects_invalid_resolution(bad):
    reference = np.ones((1, 1, 1))
    measured = np.ones((2, 2, 1, 1, 1))
    with pytest.raises(ValueError):
        K.matrix_acceptance(measured, reference, np.full_like(measured, bad))


@pytest.mark.parametrize("shape", [(1, 2, 1, 1, 1), (2, 2, 1, 0, 1)])
def test_binary64_matrix_gate_rejects_incomplete_shape(shape):
    measured = np.ones(shape)
    with pytest.raises(ValueError):
        K.matrix_acceptance(measured, np.ones(shape[2:]), np.full(shape, 1e-12))


def test_binary64_guard_aborts_before_actual_context_and_keeps_partial_receipt(
    monkeypatch,
):
    records = []
    monkeypatch.setattr(
        K, "build_context", lambda *a, **kw: pytest.fail("context must not be reached")
    )
    reference = {
        "parameters": {},
        "rhs": np.zeros((4, 58, 58)),
        "constraints": np.zeros((10, 43, 58)),
    }

    def stop():
        raise TimeoutError("owned test stop")

    with pytest.raises(TimeoutError, match="owned test stop"):
        K.run(reference, guard=stop, json_sink=lambda n, r: records.append((n, r)))
    assert records == [
        (
            "partial_calls",
            {
                "schema": "case12-binary64-calls-v1",
                "planned": 482,
                "records": [],
                "completed": 0,
                "accepted": False,
            },
        )
    ]


def test_binary64_sink_failure_never_starts_actual_evaluation(monkeypatch):
    coordinates = _coordinates()
    f = (
        1
        + sum(
            np.cos(2 * np.pi * sum(mode[a] * coordinates[a] for a in range(3)))
            for mode in K.W
        )
    ) / 8
    context = SimpleNamespace(np=np, q0=np.zeros(58), scales=np.ones(58), f=f)
    monkeypatch.setattr(K, "build_context", lambda *a, **kw: context)
    monkeypatch.setattr(
        K, "evaluate_state", lambda *a, **kw: pytest.fail("actual field call forbidden")
    )
    reference = {
        "parameters": {},
        "rhs": np.zeros((4, 58, 58)),
        "constraints": np.zeros((10, 43, 58)),
    }
    before = copy.deepcopy(reference)
    records = []

    def stop(name, value):
        assert name == "q0"
        raise OSError("owned sink stop")

    with pytest.raises(OSError, match="owned sink stop"):
        K.run(reference, array_sink=stop, json_sink=lambda n, r: records.append((n, r)))
    assert records[-1][1]["completed"] == 0
    np.testing.assert_array_equal(reference["rhs"], before["rhs"])
    np.testing.assert_array_equal(reference["constraints"], before["constraints"])

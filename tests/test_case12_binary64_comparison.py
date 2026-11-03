"""Synthetic saved-array controls; no original RHS/diagnostic is evaluated."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from experiments.case12_binary64_v1 import comparison as C


class MemoryDirectory:
    def __init__(self, records, parts=()):
        self.records = records
        self.parts = parts

    def __truediv__(self, part):
        return MemoryDirectory(self.records, self.parts + (part,))

    def read_bytes(self):
        assert len(self.parts) == 2 and self.parts[1] == "calls.json"
        return json.dumps(self.records[self.parts[0]], allow_nan=False).encode("utf-8")


def synthetic_indicators(arrays, *, main_B=1.0, mutant_B=1.0):
    """Hand-derived fixture for fhat=(1/8,1/16,...), scales=1."""
    floor = 4096.0 * 2.0**-53
    for path in ("production", "replay"):
        stats = arrays[path, "field_stats"]
        stats[..., 9] = main_B
        stats[..., 10] = 1.0
        rhs = np.empty(C.ARRAY_SHAPES["rhs_resolution"], dtype="<f8")
        con = np.empty(C.ARRAY_SHAPES["constraint_resolution"], dtype="<f8")
        negative = np.empty(C.ARRAY_SHAPES["negative_resolution"], dtype="<f8")
        modal_limits = np.empty(C.ARRAY_SHAPES["modal_limits"], dtype="<f8")
        arrays[path, "negative_field_scales"] = np.full(
            C.ARRAY_SHAPES["negative_field_scales"], mutant_B, dtype="<f8"
        )
        for oi, k in enumerate(
            (4.0 * math.sqrt(2.0), (16.0 * math.sqrt(2.0) - 4.0) / 3.0)
        ):
            for ei, epsilon in enumerate((2.0**-10, 2.0**-12)):
                field_nu = floor * main_B / epsilon
                stats[oi, ei, ..., 8] = field_nu
                rhs[oi, ei] = (
                    np.array([8.0, 16.0 / k, 16.0 / k, 16.0 / k])[:, None, None]
                    * field_nu
                )
                con[oi, ei] = (
                    np.array(
                        [
                            8.0,
                            16.0 / k,
                            16.0 / k,
                            16.0 / k,
                            24.0 / k**2,
                            72.0 / k**2,
                            72.0 / k**2,
                            24.0 / k**2,
                            72.0 / k**2,
                            24.0 / k**2,
                        ]
                    )[:, None, None]
                    * field_nu
                )
                negative[:, oi, ei] = (
                    floor
                    * (main_B + mutant_B)
                    / epsilon
                    / arrays[path, "shape_projection"][None, :, 0]
                )
                # Zero reference matrices still have the preregistered unit
                # floor per block. Axial/mixed modes contain one/two nonzero k.
                rhs_weight = np.array(
                    [1.0]
                    + [1.0 + k] * 3
                    + [1.0 + 2 * k] * 3
                    + [1.0 + k] * 3
                    + [1.0 + 2 * k] * 3
                )
                con_weight = np.array(
                    [1.0]
                    + [1.0 + k + k * k] * 3
                    + [1.0 + 2 * k + 3 * k * k] * 3
                    + [1.0 + k + k * k] * 3
                    + [1.0 + 2 * k + 3 * k * k] * 3
                )
                resolution = field_nu / arrays[path, "shape_projection"][:, 0]
                modal_limits[oi, ei, :, :58] = (
                    64 * epsilon**2 * rhs_weight + 8 * resolution
                )
                modal_limits[oi, ei, :, 58:] = (
                    64 * epsilon**2 * con_weight + 8 * resolution
                )
        arrays[path, "rhs_resolution"] = rhs
        arrays[path, "constraint_resolution"] = con
        arrays[path, "negative_resolution"] = negative
        arrays[path, "modal_limits"] = modal_limits


def synthetic_saved_case(monkeypatch):
    calls = []
    descriptors = [(f"background/fd{order}", order, "baseline") for order in (2, 4)]
    for order in (2, 4):
        for epsilon_index in (0, 1):
            for column in range(58):
                for sign in (-1, 1):
                    descriptors.append(
                        (
                            f"main/fd{order}/eps{epsilon_index}/q{column}/sign{sign}",
                            order,
                            "baseline",
                        )
                    )
    for variant in ("kappa1_doubled", "stored_trace_diagnostic"):
        for order in (2, 4):
            for epsilon_index in (0, 1):
                for sign in (-1, 1):
                    descriptors.append(
                        (
                            f"negative/{variant}/fd{order}/eps{epsilon_index}/sign{sign}",
                            order,
                            variant,
                        )
                    )
    for i, (label, order, variant) in enumerate(descriptors):
        calls.append(
            {
                "index": i,
                "label": label,
                "order": order,
                "variant": variant,
                "started": True,
                "completed": True,
                "hashes": {
                    name: f"{i:064x}"
                    for name in ("input", "rhs", "physical43", "roundtrip")
                },
            }
        )
    base = {"schema": "case12-binary64-calls-v1", "planned": 482, "records": calls}
    records = {"production": copy.deepcopy(base), "replay": copy.deepcopy(base)}
    arrays = {}
    for path in ("production", "replay"):
        for name, shape in C.ARRAY_SHAPES.items():
            # Broadcast zeros keep controls small; only changed arrays allocate.
            arrays[path, name] = np.broadcast_to(np.array(0.0, dtype="<f8"), shape)
        arrays[path, "scales"] = np.ones(58, dtype="<f8")
        shape_projection = np.zeros((13, 2), dtype="<f8")
        shape_projection[:, 0] = 0.0625
        shape_projection[0, 0] = 0.125
        arrays[path, "shape_projection"] = shape_projection
        stats = np.zeros(C.ARRAY_SHAPES["field_stats"], dtype="<f8")
        arrays[path, "field_stats"] = stats
        positive = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1))
        modes = (
            ((0, 0, 0),)
            + positive
            + tuple(tuple(-n for n in mode) for mode in positive)
        )
        arrays[path, "wave_numbers"] = (
            np.array(modes, dtype=np.float64)[None]
            * np.array([4.0 * math.sqrt(2.0), (16.0 * math.sqrt(2.0) - 4.0) / 3.0])[
                :, None, None
            ]
        )
    synthetic_indicators(arrays)

    def load(path, *, allow_pickle, mmap_mode):
        assert allow_pickle is False and mmap_mode == "r"
        assert isinstance(path, MemoryDirectory) and len(path.parts) == 2
        return arrays[path.parts[0], path.parts[1][:-4]]

    monkeypatch.setattr(np, "load", load)
    return MemoryDirectory(records), arrays, records


@pytest.mark.parametrize("field", ["index", "label", "order", "variant", "hashes"])
def test_binary64_compare_rejects_last_call_identity_mismatch(monkeypatch, field):
    directory, _, records = synthetic_saved_case(monkeypatch)
    last = records["replay"]["records"][-1]
    if field == "hashes":
        last[field]["physical43"] = "f" * 64
    elif field in ("index", "order"):
        last[field] += 1
    else:
        last[field] += "-changed"
    if field == "hashes":
        result = C.compare(directory)
        assert result["same_482_ordered_input_and_actual_field_hashes"] is False
        assert result["accepted"] is False
    else:
        with pytest.raises(ValueError, match="registered schedule"):
            C.compare(directory)


@pytest.mark.parametrize(
    "name", ["rhs_coefficients", "constraint_coefficients", "constraint_errors"]
)
def test_binary64_compare_checks_last_recovered_slot_not_just_norm(monkeypatch, name):
    directory, arrays, _ = synthetic_saved_case(monkeypatch)
    changed = arrays["replay", name].copy()
    changed[tuple(size - 1 for size in changed.shape)] = 1.0e-5
    arrays["replay", name] = changed
    result = C.compare(directory)
    row = next(item for item in result["comparisons"] if item["name"] == name)
    assert row["violating_slots"] == 1
    assert row["passed"] is False
    assert result["accepted"] is False


@pytest.mark.parametrize(
    "name", ["main_projections", "negative_projections", "background_projections"]
)
def test_binary64_compare_raw_projection_rounding_uses_actual_field_scale(
    monkeypatch, name
):
    directory, arrays, _ = synthetic_saved_case(monkeypatch)
    if name == "main_projections":
        synthetic_indicators(arrays, main_B=1.0e6)
        change = 1.0e-7  # Below 4096*u*B, far above a projection-only floor.
    elif name == "negative_projections":
        synthetic_indicators(arrays, main_B=1.0e6, mutant_B=1.0e6)
        change = 1.0e-6  # Below both-path nu * fine amplitude * |fhat|.
    else:
        for path in ("production", "replay"):
            arrays[path, "background_fields"] = np.broadcast_to(
                np.array(1.0e6, dtype="<f8"), C.ARRAY_SHAPES["background_fields"]
            )
        change = 1.0e-7
    changed = arrays["replay", name].copy()
    changed[tuple(size - 1 for size in changed.shape)] = change
    arrays["replay", name] = changed
    result = C.compare(directory)
    assert result["accepted"] is True
    row = next(item for item in result["comparisons"] if item["name"] == name)
    assert row["max_absolute_difference"] == change
    assert row["passed"] is True


@pytest.mark.parametrize("fault", ["shape", "float32", "nan", "infinity"])
def test_binary64_compare_rejects_invalid_saved_array(monkeypatch, fault):
    directory, arrays, _ = synthetic_saved_case(monkeypatch)
    name = "constraint_coefficients"
    value = arrays["replay", name].copy()
    if fault == "shape":
        value = value[..., :-1]
    elif fault == "float32":
        value = value.astype(np.float32)
    else:
        value[-1, -1, -1, -1, -1] = np.nan if fault == "nan" else np.inf
    arrays["replay", name] = value
    with pytest.raises(ValueError, match="invalid comparison array"):
        C.compare(directory)


@pytest.mark.parametrize("fault", ["missing", "incomplete", "wrong_plan"])
def test_binary64_compare_refuses_incomplete_482_call_ledger(monkeypatch, fault):
    directory, _, records = synthetic_saved_case(monkeypatch)
    if fault == "missing":
        records["replay"]["records"].pop()
    elif fault == "incomplete":
        records["replay"]["records"][-1]["completed"] = False
    else:
        records["replay"]["planned"] = 481
    with pytest.raises(ValueError):
        C.compare(directory)


def test_binary64_compare_exact_raw_background_requires_bitwise_match(monkeypatch):
    directory, arrays, _ = synthetic_saved_case(monkeypatch)
    changed = arrays["replay", "background_fields"].copy()
    changed[-1, -1, -1, -1, -1] = -0.0
    arrays["replay", "background_fields"] = changed
    result = C.compare(directory)
    assert result["accepted"] is False
    row = next(
        item for item in result["comparisons"] if item["name"] == "background_fields"
    )
    assert row["passed"] is False


def test_binary64_compare_deadline_propagates(monkeypatch):
    directory, _, _ = synthetic_saved_case(monkeypatch)

    def stop():
        raise TimeoutError("synthetic compare deadline")

    with pytest.raises(TimeoutError, match="synthetic compare deadline"):
        C.compare(directory, check=stop)


@pytest.mark.parametrize(
    "fault",
    ["duplicated", "same_swap", "missing_negative", "wrong_epsilon", "boolean_index"],
)
def test_binary64_compare_rejects_both_ledgers_making_same_schedule_error(
    monkeypatch, fault
):
    directory, _, records = synthetic_saved_case(monkeypatch)
    for path in ("production", "replay"):
        rows = records[path]["records"]
        if fault == "duplicated":
            records[path]["records"] = [copy.deepcopy(rows[0]) for _ in range(482)]
        elif fault == "same_swap":
            rows[2], rows[3] = rows[3], rows[2]
            rows[2]["index"], rows[3]["index"] = 2, 3
        elif fault == "missing_negative":
            rows[-1]["variant"] = "baseline"
        elif fault == "wrong_epsilon":
            rows[2]["label"] = rows[2]["label"].replace("eps0", "eps1")
        else:
            rows[0]["index"] = False
    with pytest.raises(ValueError, match="registered schedule"):
        C.compare(directory)


@pytest.mark.parametrize(
    "fault", ["empty", "missing", "extra", "short", "uppercase", "nonstring"]
)
def test_binary64_compare_rejects_same_malformed_hashes_in_both_ledgers(
    monkeypatch, fault
):
    directory, _, records = synthetic_saved_case(monkeypatch)
    for path in ("production", "replay"):
        hashes = records[path]["records"][-1]["hashes"]
        if fault == "empty":
            hashes.clear()
        elif fault == "missing":
            del hashes["physical43"]
        elif fault == "extra":
            hashes["invented"] = "0" * 64
        elif fault == "short":
            hashes["rhs"] = "0" * 63
        elif fault == "uppercase":
            hashes["rhs"] = "A" * 64
        else:
            hashes["rhs"] = 0
    with pytest.raises(ValueError):
        C.compare(directory)


@pytest.mark.parametrize(
    "name, fault",
    [
        ("rhs_resolution", "zero"),
        ("constraint_resolution", "zero"),
        ("negative_resolution", "zero"),
        ("rhs_resolution", "inflated"),
        ("constraint_resolution", "wrong_mixed"),
        ("negative_resolution", "inflated"),
    ],
)
def test_binary64_compare_rebuilds_resolution_instead_of_trusting_agreement(
    monkeypatch, name, fault
):
    directory, arrays, _ = synthetic_saved_case(monkeypatch)
    for path in ("production", "replay"):
        changed = arrays[path, name].copy()
        if fault == "wrong_mixed":
            changed[:, :, 5] *= 2.0
        else:
            index = tuple(size - 1 for size in changed.shape)
            changed[index] = 0.0 if fault == "zero" else changed[index] * 1.01
        arrays[path, name] = changed
    with pytest.raises(ValueError):
        C.compare(directory)


@pytest.mark.parametrize(
    "fault",
    [
        "scale_zero",
        "scale_inconsistent",
        "shape_mode_zero",
        "main_B_zero",
        "field_nu_zero",
        "mutation_B_zero",
        "mutation_B_inconsistent",
        "wave_inconsistent",
    ],
)
def test_binary64_compare_requires_registered_inputs_and_actual_scale_formula(
    monkeypatch, fault
):
    directory, arrays, _ = synthetic_saved_case(monkeypatch)
    for path in ("production", "replay"):
        if fault.startswith("scale"):
            arrays[path, "scales"][57] = 0.0 if fault == "scale_zero" else 2.0
        elif fault == "shape_mode_zero":
            arrays[path, "shape_projection"][12] = 0.0
        elif fault in ("main_B_zero", "field_nu_zero"):
            column = 9 if fault == "main_B_zero" else 8
            arrays[path, "field_stats"][-1, -1, -1, -1, column] = 0.0
        elif fault.startswith("mutation_B"):
            arrays[path, "negative_field_scales"][-1, -1, -1, -1] = (
                0.0 if fault == "mutation_B_zero" else 2.0
            )
        else:
            arrays[path, "wave_numbers"][-1, -1, -1] *= 2.0
    with pytest.raises(ValueError):
        C.compare(directory)


def test_binary64_compare_allows_only_small_relative_resolution_rounding(monkeypatch):
    directory, arrays, _ = synthetic_saved_case(monkeypatch)
    arrays["replay", "constraint_resolution"][-1, -1, -1, -1, -1] *= 1.0 + 2.0**-45
    result = C.compare(directory)
    assert result["accepted"] is True
    assert result["resolution_indicators_recomputed"] is True
    assert result["both_ledgers_match_registered_schedule"] is True


def test_binary64_compare_finite_inputs_cannot_make_infinite_derived_tolerance_pass():
    with pytest.raises(ValueError, match="inconsistent derived indicator"):
        C._tight_numeric_match(
            np.array([1.0e308]), np.array([-1.0e308]), "overflow control"
        )


@pytest.mark.parametrize(
    "fault", ["axis_real_minus_dc", "signed_mode_conjugacy", "mixed_reconstruction"]
)
def test_binary64_compare_registered_redundancy_cannot_pass_by_two_path_agreement(
    monkeypatch, fault
):
    directory, arrays, _ = synthetic_saved_case(monkeypatch)
    for path in ("production", "replay"):
        if fault == "mixed_reconstruction":
            coefficients = arrays[path, "rhs_coefficients"].copy()
            coefficients[-1, -1, 1, -1, -1] = 1.0
            arrays[path, "rhs_coefficients"] = coefficients
        else:
            projections = arrays[path, "main_projections"].copy()
            row, part = (0, 0) if fault == "axis_real_minus_dc" else (100, 1)
            # A unit normalized response only at the final input column.
            projections[-1, -1, -1, 1, row, 1, part] = 2 * 2.0**-12 / 16.0
            arrays[path, "main_projections"] = projections
    result = C.compare(directory)
    assert all(row["passed"] for row in result["comparisons"])
    assert result["same_482_ordered_input_and_actual_field_hashes"] is True
    assert result["registered_redundancies_passed"] is False
    assert result["accepted"] is False
    for path in result["redundancy_controls"]:
        assert path["rows"][-1][fault]["passed"] is False
        assert path["rows"][-1][fault]["violating_slots"] > 0


def test_binary64_compare_redundancy_receipt_has_all_registered_slots(monkeypatch):
    directory, _, _ = synthetic_saved_case(monkeypatch)
    result = C.compare(directory)
    assert result["accepted"] is True
    assert result["registered_redundancies_passed"] is True
    for path in result["redundancy_controls"]:
        assert path["derived_control_not_new_physics_proof"] is True
        assert len(path["rows"]) == 4
        for row in path["rows"]:
            assert row["axis_real_minus_dc"]["checked_slots"] == 58 * 58 * 3
            assert row["signed_mode_conjugacy"]["checked_slots"] == 58 * 101 * 6
            assert row["mixed_reconstruction"]["checked_slots"] == 58 * 101 * 6
            assert row["passed"] is True


@pytest.mark.parametrize("fault", ["zero", "inflated"])
def test_binary64_compare_redundancy_limits_are_bound_to_reference_and_resolution(
    monkeypatch, fault
):
    directory, arrays, _ = synthetic_saved_case(monkeypatch)
    for path in ("production", "replay"):
        value = arrays[path, "modal_limits"]
        value[-1, -1, -1, -1, -1] *= 0.0 if fault == "zero" else 2.0
    with pytest.raises(ValueError, match="registered modal limits"):
        C.compare(directory)

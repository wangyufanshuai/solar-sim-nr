"""Pointwise comparison of two separately executed paths; no norm-only pass."""

from __future__ import annotations

import math
import re

from .runtime import ARRAY_SHAPES, decode, require


def expected_schedule():
    """Independent literal registration, rather than agreement between ledgers."""
    rows = []
    for order in (2, 4):
        rows.append(
            (f"background/fd{order}", order, "baseline", "background", None, None, None)
        )
    for order in (2, 4):
        for ei, epsilon in enumerate((2.0**-10, 2.0**-12)):
            for slot in range(58):
                for sign in (-1, 1):
                    rows.append(
                        (
                            f"main/fd{order}/eps{ei}/q{slot}/sign{sign}",
                            order,
                            "baseline",
                            "main",
                            epsilon,
                            slot,
                            sign,
                        )
                    )
    for variant, slot in (("kappa1_doubled", 18), ("stored_trace_diagnostic", 11)):
        for order in (2, 4):
            for ei, epsilon in enumerate((2.0**-10, 2.0**-12)):
                for sign in (-1, 1):
                    rows.append(
                        (
                            f"negative/{variant}/fd{order}/eps{ei}/sign{sign}",
                            order,
                            variant,
                            "negative",
                            epsilon,
                            slot,
                            sign,
                        )
                    )
    return tuple(rows)


def validate_calls(record):
    require(
        type(record) is dict and set(record) == {"schema", "planned", "records"},
        "call ledger schema",
    )
    require(
        record["schema"] == "case12-binary64-calls-v1"
        and type(record["planned"]) is int
        and record["planned"] == 482,
        "registered call plan",
    )
    require(
        type(record["records"]) is list and len(record["records"]) == 482,
        "complete independent schedule required",
    )
    for index, (row, expected) in enumerate(
        zip(record["records"], expected_schedule(), strict=True)
    ):
        label, order, variant, category, epsilon, slot, sign = expected
        require(
            type(row) is dict
            and {"index", "label", "order", "variant", "started", "completed", "hashes"}
            <= set(row),
            "complete call identity required",
        )
        require(
            type(row["index"]) is int
            and row["index"] == index
            and type(row["label"]) is str
            and row["label"] == label
            and type(row["order"]) is int
            and row["order"] == order
            and type(row["variant"]) is str
            and row["variant"] == variant,
            "call identity disagrees with registered schedule",
        )
        require(
            row["started"] is True and row["completed"] is True,
            "partial scientific call",
        )
        require(
            type(row["hashes"]) is dict
            and set(row["hashes"]) == {"input", "rhs", "physical43", "roundtrip"},
            "four complete actual-field hashes required",
        )
        require(
            all(
                type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None
                for value in row["hashes"].values()
            ),
            "canonical SHA256 digests required",
        )
        for name, value in (
            ("category", category),
            ("epsilon", epsilon),
            ("slot", slot),
            ("sign", sign),
        ):
            if name in row:
                require(
                    type(row[name]) is type(value) and row[name] == value,
                    "optional call identity drift: " + name,
                )
    return True


def _tight_numeric_match(actual, expected, name):
    """Relative arithmetic check: no unit-sized floor can validate zero nu."""
    import numpy as np

    require(
        actual.shape == expected.shape and np.all(np.isfinite(expected)),
        "invalid recomputed indicator: " + name,
    )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        error = np.abs(actual - expected)
        tolerance = (
            4096.0 * 2.0**-53 * np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
        )
    require(
        np.all(np.isfinite(error))
        and np.all(np.isfinite(tolerance))
        and np.all(error <= tolerance),
        "inconsistent derived indicator: " + name,
    )


def _registered_waves():
    import numpy as np

    positive = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1))
    modes = ((0, 0, 0),) + positive + tuple(tuple(-n for n in v) for v in positive)
    waves = np.empty((2, 13, 3), dtype=np.float64)
    for oi, order in enumerate((2, 4)):
        for m, mode in enumerate(modes):
            for a in range(3):
                angle = 2.0 * math.pi * mode[a] / 8.0
                waves[oi, m, a] = (
                    8.0 * math.sin(angle)
                    if order == 2
                    else 8.0 * (8.0 * math.sin(angle) - math.sin(2.0 * angle)) / 6.0
                )
    return waves


def validate_resolution(path, read):
    """Rebuild every positive indicator from saved actual field maxima.

    Maxima themselves are independently generated by the two actual-field
    executions, and checked for exact agreement later. Raw signed grids are
    not archived, so this does not claim a third recomputation of their maxima.
    """
    import numpy as np

    u = 2.0**-53
    q0, scales = read(path, "q0"), read(path, "scales")
    require(
        np.array_equal(scales, np.maximum(1.0, np.abs(q0))),
        "registered positive input scales required",
    )
    fhat = read(path, "shape_projection")
    magnitude = np.hypot(fhat[:, 0], fhat[:, 1])
    require(
        np.all(np.isfinite(magnitude)) and np.all(magnitude > 1.0 / 32.0),
        "all 13 resolved excitation modes required",
    )
    waves = _registered_waves()
    _tight_numeric_match(
        read(path, "wave_numbers"), waves, "registered discrete wave numbers"
    )
    stats = read(path, "field_stats")
    B = stats[..., 9]
    require(
        np.all(B >= 1.0) and np.all(stats[..., 8] > 0.0),
        "positive field resolution and B>=1 required",
    )
    amplitudes = (
        np.array((2.0**-10, 2.0**-12))[None, :, None, None]
        * scales[None, None, :, None]
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        field_nu = 4096.0 * u * B / amplitudes
        modal_nu = field_nu[..., None] / magnitude
    require(
        np.all(np.isfinite(field_nu))
        and np.all(field_nu > 0.0)
        and np.all(np.isfinite(modal_nu))
        and np.all(modal_nu > 0.0),
        "finite positive reconstructed resolution required",
    )
    _tight_numeric_match(stats[..., 8], field_nu, "field resolution")
    rhs_expected = np.empty(ARRAY_SHAPES["rhs_resolution"], dtype=np.float64)
    constraint_expected = np.empty(
        ARRAY_SHAPES["constraint_resolution"], dtype=np.float64
    )
    for oi in range(2):
        # Work in (epsilon,column,row,mode), then place all row/column slots.
        nu = modal_nu[oi]
        rhs_expected[oi, :, 0] = np.swapaxes(nu[:, :, :58, 0], 1, 2)
        constraint_expected[oi, :, 0] = np.swapaxes(nu[:, :, 58:, 0], 1, 2)
        for a, diagonal in ((0, 4), (1, 7), (2, 9)):
            k = abs(waves[oi, a + 1, a])
            rhs_expected[oi, :, a + 1] = np.swapaxes(nu[:, :, :58, a + 1] / k, 1, 2)
            constraint_expected[oi, :, a + 1] = np.swapaxes(
                nu[:, :, 58:, a + 1] / k, 1, 2
            )
            constraint_expected[oi, :, diagonal] = np.swapaxes(
                (nu[:, :, 58:, 0] + nu[:, :, 58:, a + 1]) / (k * k), 1, 2
            )
        for mode, a, b, block, aa, bb in (
            (4, 0, 1, 5, 4, 7),
            (5, 0, 2, 6, 4, 9),
            (6, 1, 2, 8, 7, 9),
        ):
            ka, kb = waves[oi, mode, a], waves[oi, mode, b]
            numerator = np.swapaxes(nu[:, :, 58:, 0] + nu[:, :, 58:, mode], 1, 2)
            numerator = (
                numerator
                + ka * ka * constraint_expected[oi, :, aa]
                + kb * kb * constraint_expected[oi, :, bb]
            )
            constraint_expected[oi, :, block] = numerator / abs(ka * kb)
    for name, expected in (
        ("rhs_resolution", rhs_expected),
        ("constraint_resolution", constraint_expected),
    ):
        actual = read(path, name)
        require(np.all(actual > 0.0), "strictly positive recovered resolution required")
        _tight_numeric_match(actual, expected, name)
    mutation_B = read(path, "negative_field_scales")
    require(
        np.all(mutation_B >= 1.0),
        "registered negative actual-field scale B>=1 required",
    )
    negative_expected = np.empty(ARRAY_SHAPES["negative_resolution"], dtype=np.float64)
    for vi, slot in enumerate((18, 11)):
        amplitude = np.array((2.0**-10, 2.0**-12))[None, :, None] * scales[slot]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            negative_expected[vi] = (
                4096.0 * u * (mutation_B[vi] + B[:, :, slot]) / amplitude
            )[..., None] / magnitude
    negative_actual = read(path, "negative_resolution")
    require(
        np.all(negative_actual > 0.0), "strictly positive negative resolution required"
    )
    _tight_numeric_match(
        negative_actual, negative_expected, "negative resolution from actual maxima"
    )
    modal_limits = read(path, "modal_limits")
    reference_rhs = read(path, "reference_rhs")
    reference_constraints = read(path, "reference_constraints")
    rhs_weights = np.maximum(1.0, np.sum(np.abs(reference_rhs), axis=-1))
    constraint_weights = np.maximum(1.0, np.sum(np.abs(reference_constraints), axis=-1))
    for oi in range(2):
        factors = _symbol_factors(waves[oi])
        # Use explicit block sums, independently of producer matrix products.
        weights = np.empty((101, 13), dtype=np.float64)
        for row in range(101):
            source = (
                rhs_weights[:, row] if row < 58 else constraint_weights[:, row - 58]
            )
            weights[row] = sum(
                source[b] * np.abs(factors[b]) for b in range(len(source))
            )
        for ei, epsilon in enumerate((2.0**-10, 2.0**-12)):
            expected = 64.0 * epsilon**2 * weights[None] + 8.0 * modal_nu[oi, ei]
            _tight_numeric_match(
                modal_limits[oi, ei], expected, "registered modal limits"
            )
    return {
        "modal_nu": modal_nu,
        "field_nu": field_nu,
        "negative_nu": negative_expected,
        "magnitude": magnitude,
    }


def _symbol_factors(wave):
    import numpy as np

    answer = np.empty((10, 13), dtype=np.complex128)
    answer[0] = 1.0
    for axis in range(3):
        answer[axis + 1] = 1j * wave[:, axis]
    for block, (a, b) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)), 4):
        answer[block] = -wave[:, a] * wave[:, b]
    return answer


def _residual_summary(residual, limit, *, coordinates):
    import numpy as np

    absolute = np.abs(residual)
    require(
        absolute.shape == limit.shape
        and absolute.size > 0
        and np.all(np.isfinite(absolute))
        and np.all(np.isfinite(limit))
        and np.all(limit > 0.0),
        "invalid redundancy residual or bound",
    )
    flat = int(np.argmax(absolute))
    index = tuple(int(i) for i in np.unravel_index(flat, absolute.shape))
    violations = int(np.count_nonzero(absolute > limit))
    return dict(
        checked_slots=int(absolute.size),
        coordinates=coordinates,
        max_absolute_residual=float(absolute[index]),
        argmax=list(index),
        limit_at_argmax=float(limit[index]),
        max_allowed_residual=float(np.max(limit)),
        violating_slots=violations,
        passed=violations == 0,
    )


def redundancy_controls(path, read, *, check=lambda: None):
    """Derived algebraic controls; raw inputs remain in the signed projections.

    Each limit is a triangle propagation of the original modal acceptance
    limits through the registered recovery formulas, with no new allowance.
    The physical43 mixed real-part recovery is partly an algebraic identity;
    these controls are not another independent physical validation.
    """
    import numpy as np

    raw = read(path, "main_projections")
    raw_fhat = read(path, "shape_projection")
    fhat = raw_fhat[:, 0] + 1j * raw_fhat[:, 1]
    scales = read(path, "scales")
    limits = read(path, "modal_limits")
    rhs = read(path, "rhs_coefficients")
    con = read(path, "constraint_coefficients")
    waves = _registered_waves()
    rows = []
    for oi, order in enumerate((2, 4)):
        for ei, epsilon in enumerate((2.0**-10, 2.0**-12)):
            check()
            pair = raw[oi, ei]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                signed = pair[..., 0] + 1j * pair[..., 1]
                measured = (
                    (signed[:, 1] - signed[:, 0])
                    / (2.0 * epsilon * scales[:, None, None])
                    / fhat
                )
            require(
                np.all(np.isfinite(measured)), "non-finite normalized signed projection"
            )
            modal = limits[oi, ei]
            axis = _residual_summary(
                measured[:, :58, 1:4].real - measured[:, :58, 0:1].real,
                modal[:, :58, 1:4] + modal[:, :58, 0:1],
                coordinates=["input_slot", "rhs_row", "axis_xyz"],
            )
            conjugacy = _residual_summary(
                measured[:, :, 7:13] - np.conjugate(measured[:, :, 1:7]),
                modal[:, :, 7:13] + modal[:, :, 1:7],
                coordinates=["input_slot", "combined_row", "positive_mode_1_to_6"],
            )
            propagated = np.zeros((58, 101, 10), dtype=np.float64)
            propagated[..., 0] = modal[..., 0]
            for a, diagonal in ((0, 4), (1, 7), (2, 9)):
                k = abs(waves[oi, a + 1, a])
                propagated[..., a + 1] = modal[..., a + 1] / k
                propagated[:, 58:, diagonal] = (
                    modal[:, 58:, 0] + modal[:, 58:, a + 1]
                ) / (k * k)
            for mode, a, b, block, aa, bb in (
                (4, 0, 1, 5, 4, 7),
                (5, 0, 2, 6, 4, 9),
                (6, 1, 2, 8, 7, 9),
            ):
                ka, kb = waves[oi, mode, a], waves[oi, mode, b]
                propagated[:, 58:, block] = (
                    modal[:, 58:, 0]
                    + modal[:, 58:, mode]
                    + ka * ka * propagated[:, 58:, aa]
                    + kb * kb * propagated[:, 58:, bb]
                ) / abs(ka * kb)
            coefficients = np.zeros((58, 101, 10), dtype=np.float64)
            coefficients[:, :58, :4] = np.transpose(rhs[oi, ei], (2, 1, 0))
            coefficients[:, 58:, :] = np.transpose(con[oi, ei], (2, 1, 0))
            factors = _symbol_factors(waves[oi])
            mixed_residual = np.empty((58, 101, 6), dtype=np.complex128)
            mixed_limits = np.empty((58, 101, 6), dtype=np.float64)
            for index, mode in enumerate((4, 5, 6, 10, 11, 12)):
                reconstructed = sum(
                    coefficients[..., b] * factors[b, mode] for b in range(10)
                )
                mixed_residual[..., index] = reconstructed - measured[..., mode]
                mixed_limits[..., index] = modal[..., mode] + sum(
                    propagated[..., b] * abs(factors[b, mode]) for b in range(10)
                )
            mixed = _residual_summary(
                mixed_residual,
                mixed_limits,
                coordinates=[
                    "input_slot",
                    "combined_row",
                    "signed_mixed_mode_4_5_6_10_11_12",
                ],
            )
            rows.append(
                dict(
                    path=path,
                    order=order,
                    epsilon=epsilon,
                    axis_real_minus_dc=axis,
                    signed_mode_conjugacy=conjugacy,
                    mixed_reconstruction=mixed,
                    passed=axis["passed"] and conjugacy["passed"] and mixed["passed"],
                )
            )
    return dict(
        rows=rows,
        accepted=all(row["passed"] for row in rows),
        inputs="saved full signed main_projections, shape_projection, recovered coefficients and bound-validated modal_limits",
        bounds="triangle propagation of original modal limits through registered recovery weights",
        derived_control_not_new_physics_proof=True,
    )


def compare(directory, *, check=lambda: None):
    import numpy as np

    u = 2.0**-53
    exact = {
        "q0",
        "scales",
        "shape_field",
        "reference_rhs",
        "reference_constraints",
        "background_fields",
        "roundtrip_stats",
        "negative_field_scales",
    }

    def read(path, name):
        check()
        value = np.load(
            directory / path / (name + ".npy"), allow_pickle=False, mmap_mode="r"
        )
        require(
            value.shape == ARRAY_SHAPES[name]
            and value.dtype.str == "<f8"
            and np.all(np.isfinite(value)),
            "invalid comparison array: " + name,
        )
        return value

    calls = [
        decode((directory / path / "calls.json").read_bytes())
        for path in ("production", "replay")
    ]
    for record in calls:
        validate_calls(record)
    hash_agreement = all(
        a["hashes"] == b["hashes"]
        and a["index"] == b["index"]
        and a["label"] == b["label"]
        and a["order"] == b["order"]
        and a["variant"] == b["variant"]
        for a, b in zip(calls[0]["records"], calls[1]["records"], strict=True)
    )
    reconstructed = [
        validate_resolution(path, read) for path in ("production", "replay")
    ]
    redundancy = [
        redundancy_controls(path, read, check=check)
        for path in ("production", "replay")
    ]
    stats = [read(path, "field_stats") for path in ("production", "replay")]
    fhat = [read(path, "shape_projection") for path in ("production", "replay")]
    nu_field = stats[0][..., 8] + stats[1][..., 8]
    fhat_min = np.minimum(
        np.hypot(fhat[0][:, 0], fhat[0][:, 1]), np.hypot(fhat[1][:, 0], fhat[1][:, 1])
    )
    require(np.all(fhat_min > 0), "nonzero excitation modes required")
    rows = []
    for name in ARRAY_SHAPES:
        check()
        left, right = (read(path, name) for path in ("production", "replay"))
        difference = np.abs(left - right)
        # Floating-point evaluation/summation floor, never an epsilon^2 allowance.
        limit = 4096 * u * np.maximum(1, np.maximum(np.abs(left), np.abs(right)))
        ignored = 0
        if name in exact:
            passed = bool(np.array_equal(left.view("u8"), right.view("u8")))
            limit = np.zeros_like(limit)
        else:
            if name in (
                "rhs_coefficients",
                "rhs_errors",
                "constraint_coefficients",
                "constraint_errors",
            ):
                kind = name.split("_")[0]
                indicators = [
                    read(path, kind + "_resolution")
                    for path in ("production", "replay")
                ]
                limit = 8 * (indicators[0] + indicators[1])
            elif name == "modal_errors":
                limit = np.broadcast_to(
                    (8 * (reconstructed[0]["modal_nu"] + reconstructed[1]["modal_nu"]))[
                        ..., None
                    ],
                    left.shape,
                )
            elif name == "main_projections":
                magnitude = np.maximum(stats[0][..., 9], stats[1][..., 9])
                limit = np.broadcast_to(
                    (4096 * u * magnitude)[..., None, :, None, None], left.shape
                )
            elif name == "negative_projections":
                scales = read("production", "scales")
                amplitudes = (
                    np.array([scales[18], scales[11]])[:, None, None, None, None]
                    * np.array([2.0**-10, 2.0**-12])[None, None, :, None, None]
                )
                indicators = [
                    read(path, "negative_resolution")
                    for path in ("production", "replay")
                ]
                raw_allowance = sum(
                    indicators[i] * amplitudes * np.hypot(fhat[i][:, 0], fhat[i][:, 1])
                    for i in (0, 1)
                )
                limit = np.broadcast_to(
                    raw_allowance[:, :, :, None, :, :, None], left.shape
                )
            elif name == "background_projections":
                background = read("production", "background_fields")
                magnitude = np.maximum(1, np.max(np.abs(background), axis=(2, 3, 4)))
                limit = np.broadcast_to(
                    (4096 * u * magnitude)[..., None, None], left.shape
                )
            elif name in ("negative_errors", "negative_changes"):
                indicators = [
                    read(path, "negative_resolution")
                    for path in ("production", "replay")
                ]
                limit = np.broadcast_to(
                    (8 * (indicators[0] + indicators[1]))[..., None], left.shape
                )
            elif name == "field_stats":
                # Both paths independently gate all nodes. Tied error maxima may
                # select different locations; record indices but do not gate ties.
                for column in (0, 1, 3, 4):
                    limit[..., column] = 8 * nu_field * (14 if column in (3, 4) else 1)
                difference[..., 2] = 0
                ignored = int(np.prod(left.shape[:-1]))
                limit[..., 9] = 0  # Maxima of bitwise-equal actual fields must agree.
                limit[..., 10] = 0  # each row's pass flag must agree exactly
            require(
                np.all(np.isfinite(difference))
                and np.all(np.isfinite(limit))
                and np.all(limit >= 0),
                "non-finite or invalid comparison arithmetic",
            )
            passed = bool(np.all(difference <= limit))
        rows.append(
            dict(
                name=name,
                passed=passed,
                compared_slots=int(left.size) - ignored,
                tied_argmax_slots_not_compared=ignored,
                max_absolute_difference=float(np.max(difference)),
                max_allowed_difference=float(np.max(limit)),
                violating_slots=int(np.count_nonzero(difference > limit)),
            )
        )
        del left, right, difference, limit
    return dict(
        schema="case12-binary64-two-path-comparison-v1",
        same_482_ordered_input_and_actual_field_hashes=hash_agreement,
        both_ledgers_match_registered_schedule=True,
        resolution_indicators_recomputed=True,
        actual_mutation_field_scales_archived=True,
        redundancy_controls=redundancy,
        registered_redundancies_passed=all(record["accepted"] for record in redundancy),
        comparisons=rows,
        no_epsilon_squared_cross_path_allowance=True,
        shared_actual_backend_not_independent_physics_implementation=True,
        accepted=hash_agreement
        and all(row["passed"] for row in rows)
        and all(record["accepted"] for record in redundancy),
    )

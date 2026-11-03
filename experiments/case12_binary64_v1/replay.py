"""Independent reduction of the registered 8-cube binary64 experiment.

Only the original-field evaluator and state container are shared with the
producer.  Projection, Fourier symbols, coefficient extraction, and empirical
acceptance tests below are independently implemented.  No scientific work is
performed at import time.  This is not an independent physical formulation.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from types import SimpleNamespace

ORDERS = (2, 4)
EPSILONS = (2.0**-10, 2.0**-12)
WAVES = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1))
MODES = ((0, 0, 0),) + WAVES + tuple(tuple(-v for v in k) for k in WAVES)
PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
MUTANTS = (("kappa1_doubled", 18), ("stored_trace_diagnostic", 11))
U = 2.0**-53


def replay_manifest():
    path = Path(__file__).resolve()
    return {
        "schema": "case12-binary64-replay-method-v1",
        "source": "experiments/case12_binary64_v1/replay.py",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "projection": "explicit cosine and sine weighted sums at all 512 nodes",
        "recovery": "independent DC, axial, and mixed-mode linear formulas",
        "oracle": "unshifted full58 J/K and physical43 C0/C1/C2",
        "shared_trusted_base": [
            "original binary64 RHS and diagnostic formulas",
            "kernel.build_context and kernel.evaluate_state",
            "checkpoint state construction and physical43 packing",
            "NumPy, BLAS, libm, and Python runtime",
        ],
        "independent_physical_formulation": False,
        "rigorous_roundoff_certificate": False,
        "planned_state_evaluations": 482,
    }


def evaluation_plan():
    """Complete immutable schedule; no evaluator is called here."""
    result = []
    for order in ORDERS:
        result.append(("background", order, None, None, None, "baseline"))
    for order in ORDERS:
        for epsilon in EPSILONS:
            for slot in range(58):
                for sign in (-1, 1):
                    result.append(("main", order, epsilon, slot, sign, "baseline"))
    for variant, slot in MUTANTS:
        for order in ORDERS:
            for epsilon in EPSILONS:
                for sign in (-1, 1):
                    result.append(("negative", order, epsilon, slot, sign, variant))
    return tuple(result)


def _numpy():
    import numpy as np

    return np


def _real_array(value, *, shape=None):
    np = _numpy()
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float64"):
        raise ValueError("a binary64 ndarray is required")
    if shape is not None and value.shape != shape:
        raise ValueError("unexpected array shape")
    if not np.all(np.isfinite(value)):
        raise ValueError("non-finite binary64 input")
    return value


def _complex_array(value, *, shape=None):
    np = _numpy()
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype("complex128"):
        raise ValueError("a complex128 ndarray is required")
    if shape is not None and value.shape != shape:
        raise ValueError("unexpected complex array shape")
    if not np.all(np.isfinite(value)):
        raise ValueError("non-finite complex input")
    return value


def complex_record(value):
    np = _numpy()
    _complex_array(value)
    return np.stack((value.real, value.imag), axis=-1)


def project_explicit(fields, *, check=lambda: None):
    """Project onto the registered 13 modes without FFT or producer weights."""
    np = _numpy()
    values = _real_array(fields)
    if values.ndim < 3 or values.shape[-3:] != (8, 8, 8):
        raise ValueError("projection requires the registered 8-cube")
    points = np.indices((8, 8, 8), dtype=np.float64) / 8.0
    output = np.empty(values.shape[:-3] + (13,), dtype=np.complex128)
    for m, mode in enumerate(MODES):
        check()
        angle = 2.0 * math.pi * sum(mode[a] * points[a] for a in range(3))
        real = np.sum(values * np.cos(angle), axis=(-3, -2, -1)) / 512.0
        imag = -np.sum(values * np.sin(angle), axis=(-3, -2, -1)) / 512.0
        output[..., m] = real + 1j * imag
    check()
    return output


def reconstruct_explicit(projections, *, check=lambda: None):
    np = _numpy()
    values = _complex_array(projections)
    if values.shape[-1:] != (13,):
        raise ValueError("all 13 signed modes are required")
    points = np.indices((8, 8, 8), dtype=np.float64) / 8.0
    output = np.zeros(values.shape[:-1] + (8, 8, 8), dtype=np.float64)
    for m, mode in enumerate(MODES):
        check()
        angle = 2.0 * math.pi * sum(mode[a] * points[a] for a in range(3))
        output += values[..., m].real[..., None, None, None] * np.cos(angle)
        output -= values[..., m].imag[..., None, None, None] * np.sin(angle)
    check()
    return output


def modified_wave_numbers(order):
    np = _numpy()
    if type(order) is not int or order not in ORDERS:
        raise ValueError("only the registered FD2/FD4 orders are allowed")
    values = np.empty((13, 3), dtype=np.float64)
    for m, mode in enumerate(MODES):
        for axis in range(3):
            angle = 2.0 * math.pi * mode[axis] / 8.0
            if order == 2:
                values[m, axis] = 8.0 * math.sin(angle)
            else:
                values[m, axis] = (
                    8.0 * (8.0 * math.sin(angle) - math.sin(2.0 * angle)) / 6.0
                )
    return values


def assemble_symbols(coefficients, order):
    """C2 stores full mixed monomial coefficients, with no extra factor two."""
    np = _numpy()
    matrix = _real_array(coefficients)
    if matrix.ndim != 3 or matrix.shape[0] not in (4, 10):
        raise ValueError(
            "four first-order or ten second-order coefficient blocks required"
        )
    wave = modified_wave_numbers(order)
    output = np.empty(matrix.shape[1:] + (13,), dtype=np.complex128)
    for m in range(13):
        output[..., m] = matrix[0]
        for axis in range(3):
            output[..., m] += 1j * wave[m, axis] * matrix[1 + axis]
        if len(matrix) == 10:
            for block, (a, b) in enumerate(PAIRS, 4):
                output[..., m] -= wave[m, a] * wave[m, b] * matrix[block]
    return output


def recover_coefficients(symbols, resolution, order, *, second_order=False):
    np = _numpy()
    symbols = _complex_array(symbols)
    if symbols.ndim != 2 or symbols.shape[1] != 13:
        raise ValueError("row-by-13 symbols required")
    resolution = _real_array(resolution, shape=symbols.shape)
    if np.any(resolution < 0) or type(second_order) is not bool:
        raise ValueError("invalid resolution or order selector")
    wave = modified_wave_numbers(order)
    count = 10 if second_order else 4
    answer = np.zeros((count, symbols.shape[0]), dtype=np.float64)
    uncertainty = np.zeros_like(answer)
    answer[0] = symbols[:, 0].real
    uncertainty[0] = resolution[:, 0]
    for axis in range(3):
        k = wave[axis + 1, axis]
        answer[1 + axis] = symbols[:, axis + 1].imag / k
        uncertainty[1 + axis] = resolution[:, axis + 1] / abs(k)
    if second_order:
        for axis, block in ((0, 4), (1, 7), (2, 9)):
            k = wave[axis + 1, axis]
            answer[block] = (answer[0] - symbols[:, axis + 1].real) / (k * k)
            uncertainty[block] = (resolution[:, 0] + resolution[:, axis + 1]) / (k * k)
        for m, a, b, block, aa, bb in (
            (4, 0, 1, 5, 4, 7),
            (5, 0, 2, 6, 4, 9),
            (6, 1, 2, 8, 7, 9),
        ):
            ka, kb = wave[m, a], wave[m, b]
            answer[block] = (
                answer[0]
                - symbols[:, m].real
                - ka * ka * answer[aa]
                - kb * kb * answer[bb]
            ) / (ka * kb)
            uncertainty[block] = (
                resolution[:, 0]
                + resolution[:, m]
                + ka * ka * uncertainty[aa]
                + kb * kb * uncertainty[bb]
            ) / abs(ka * kb)
    return answer, uncertainty


def derivative_shapes(shape_projection, order):
    np = _numpy()
    values = _complex_array(shape_projection, shape=(13,))
    wave = modified_wave_numbers(order)
    result = [reconstruct_explicit(values)]
    result.extend(reconstruct_explicit(values * (1j * wave[:, a])) for a in range(3))
    result.extend(
        reconstruct_explicit(-values * wave[:, a] * wave[:, b]) for a, b in PAIRS
    )
    return np.stack(result)


def design_diagnostics(shape_projection, order):
    """Rank and condition of the complete signed-mode coefficient designs."""
    np = _numpy()
    shape_projection = _complex_array(shape_projection, shape=(13,))
    wave = modified_wave_numbers(order)
    full = np.empty((13, 10), dtype=np.complex128)
    full[:, 0] = 1.0
    for a in range(3):
        full[:, a + 1] = 1j * wave[:, a]
    for j, (a, b) in enumerate(PAIRS, 4):
        full[:, j] = -wave[:, a] * wave[:, b]
    rows = []
    for count in (4, 10):
        weighted = full[:, :count] * shape_projection[:, None]
        matrix = np.concatenate((weighted.real, weighted.imag), axis=0)
        rank = int(np.linalg.matrix_rank(matrix))
        condition = float(np.linalg.cond(matrix))
        accepted = rank == count and math.isfinite(condition)
        rows.append(
            {
                "columns": count,
                "rank": rank,
                "condition_2": condition if accepted else None,
                "accepted": accepted,
            }
        )
    return {
        "order": order,
        "rows": rows,
        "accepted": all(row["accepted"] for row in rows),
    }


def independent_preflight(context, fhat, *, check=lambda: None):
    """Cheap scalar stencil checks, not additional full scientific evaluations."""
    np = _numpy()
    shape = _real_array(context.f, shape=(8, 8, 8))
    rows = []
    designs = []
    for order in ORDERS:
        check()
        designs.append(design_diagnostics(fhat, order))
        grid = context.checkpoint.Grid3D(
            (8, 8, 8),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            boundary="periodic",
            spatial_order=order,
        )
        state = SimpleNamespace(grid=grid)
        exact = derivative_shapes(fhat, order)
        fd = []
        fft = []
        for a in range(3):
            check()
            fd.append(_real_array(context.rhs._d1(shape, state, a), shape=(8, 8, 8)))
            fft.append(
                _real_array(
                    context.diagnostic.derivative(shape, a, order), shape=(8, 8, 8)
                )
            )
        for block, (a, b) in enumerate(PAIRS, 4):
            check()
            fd.append(_real_array(context.rhs._d1(fd[b], state, a), shape=(8, 8, 8)))
            fft.append(
                _real_array(
                    context.diagnostic.derivative(fft[b], a, order), shape=(8, 8, 8)
                )
            )
        for block in range(1, 10):
            scale = max(1.0, float(np.max(np.abs(exact[block]))))
            limit = 4096.0 * U * scale
            fd_error = float(np.max(np.abs(fd[block - 1] - exact[block])))
            fft_error = float(np.max(np.abs(fft[block - 1] - exact[block])))
            pair_error = float(np.max(np.abs(fd[block - 1] - fft[block - 1])))
            rows.append(
                {
                    "order": order,
                    "block": block,
                    "stencil_error": fd_error,
                    "diagnostic_error": fft_error,
                    "cross_error": pair_error,
                    "limit": limit,
                    "accepted": fd_error <= limit
                    and fft_error <= limit
                    and pair_error <= limit,
                }
            )
    return {
        "designs": designs,
        "stencils": rows,
        "scalar_derivative_calls": 36,
        "full_rhs_calls": 0,
        "full_diagnostic_calls": 0,
        "accepted": all(row["accepted"] for row in rows)
        and all(row["accepted"] for row in designs),
    }


def coefficient_gates(actual, resolution, reference, epsilon, *, fine):
    np = _numpy()
    ref = _real_array(reference)
    value = _real_array(actual, shape=ref.shape)
    nu = _real_array(resolution, shape=ref.shape)
    if (
        ref.ndim != 3
        or not all(ref.shape)
        or type(fine) is not bool
        or epsilon not in EPSILONS
        or np.any(nu <= 0)
    ):
        raise ValueError("invalid coefficient gate input")
    weight = np.maximum(1.0, np.sum(np.abs(ref), axis=2))
    error = np.abs(value - ref)
    limits = 64.0 * epsilon * epsilon * weight[..., None] + 8.0 * nu
    nonzero = ref != 0.0
    absolute_ok = bool(np.all(error <= limits))
    resolved = (
        bool(np.all(np.abs(ref[nonzero]) >= 32.0 * nu[nonzero])) if fine else True
    )
    relative_ok = (
        bool(np.all(error[nonzero] <= 1.0e-3 * np.abs(ref[nonzero]))) if fine else True
    )
    return {
        "absolute": absolute_ok,
        "resolved_nonzero": resolved,
        "relative_nonzero": relative_ok,
        "accepted": absolute_ok and resolved and relative_ok,
    }


def convergence_gate(errors, resolution):
    np = _numpy()
    errors = _real_array(errors)
    nu = _real_array(resolution, shape=errors.shape)
    if (
        errors.ndim != 5
        or errors.shape[:2] != (2, 2)
        or not all(errors.shape)
        or np.any(nu <= 0)
    ):
        raise ValueError("two orders and two epsilons required")
    high_error = np.max(np.abs(errors[:, 0]), axis=-1)
    low_error = np.max(np.abs(errors[:, 1]), axis=-1)
    high_nu = np.max(nu[:, 0], axis=-1)
    low_nu = np.max(nu[:, 1], axis=-1)
    truncation = high_error > 64.0 * high_nu
    allowed = high_error / 8.0 + 8.0 * (high_nu + low_nu)
    return {
        "tested_rows": int(np.count_nonzero(truncation)),
        "roundoff_limited_rows": int(np.count_nonzero(~truncation)),
        "failed_rows": int(np.count_nonzero(truncation & (low_error > allowed))),
        "accepted": bool(np.all(~truncation | (low_error <= allowed))),
    }


def _roundtrip_statistics(q, recovered):
    np = _numpy()
    q = _real_array(q, shape=(58, 8, 8, 8))
    recovered = _real_array(recovered, shape=q.shape)
    difference = np.abs(recovered - q)
    ratios = difference / (128.0 * U * np.maximum(1.0, np.abs(q)))
    return np.stack(
        (
            np.max(difference, axis=(1, 2, 3)),
            np.sqrt(np.mean(difference * difference, axis=(1, 2, 3))),
            np.max(ratios, axis=(1, 2, 3)),
        ),
        axis=1,
    )


def _modal_weights(reference, order):
    np = _numpy()
    wave = modified_wave_numbers(order)
    weights = np.maximum(1.0, np.sum(np.abs(reference), axis=2))
    answer = np.empty((reference.shape[1], 13), dtype=np.float64)
    for m in range(13):
        answer[:, m] = weights[0]
        for a in range(3):
            answer[:, m] += abs(wave[m, a]) * weights[a + 1]
        if len(reference) == 10:
            for block, (a, b) in enumerate(PAIRS, 4):
                answer[:, m] += abs(wave[m, a] * wave[m, b]) * weights[block]
    return answer


def _field_statistics(
    plus,
    minus,
    background,
    projections,
    prediction,
    epsilon,
    scale,
    field_weight,
    *,
    check=lambda: None,
):
    np = _numpy()
    odd = (plus - minus) / (2.0 * epsilon * scale)
    difference = odd - prediction
    # Project the actual grid difference, rather than assuming floating-point
    # projection commutes exactly with subtracting the two signed fields.
    off = odd - reconstruct_explicit(project_explicit(odd, check=check), check=check)
    even = plus + minus - 2.0 * background
    magnitude = np.maximum(
        1.0,
        np.maximum(
            np.max(np.abs(plus), axis=(1, 2, 3)),
            np.maximum(
                np.max(np.abs(minus), axis=(1, 2, 3)),
                np.max(np.abs(background), axis=(1, 2, 3)),
            ),
        ),
    )
    nu = 4096.0 * U * magnitude / (epsilon * scale)
    threshold = 64.0 * epsilon * epsilon * field_weight + 8.0 * nu
    maxerror = np.max(np.abs(difference), axis=(1, 2, 3))
    offmax = np.max(np.abs(off), axis=(1, 2, 3))
    accepted = (maxerror <= threshold) & (offmax <= 14.0 * threshold)
    return np.stack(
        (
            maxerror,
            np.sqrt(np.mean(difference * difference, axis=(1, 2, 3))),
            np.argmax(np.abs(difference).reshape(101, -1), axis=1),
            offmax,
            np.sqrt(np.mean(off * off, axis=(1, 2, 3))),
            np.max(np.abs(even), axis=(1, 2, 3)),
            np.sqrt(np.mean(even * even, axis=(1, 2, 3))),
            threshold,
            nu,
            magnitude,
            accepted.astype(np.float64),
        ),
        axis=1,
    )


ARRAY_SPECS = {
    "q0": (58,),
    "scales": (58,),
    "shape_field": (8, 8, 8),
    "shape_projection": (13, 2),
    "wave_numbers": (2, 13, 3),
    "reference_rhs": (4, 58, 58),
    "reference_constraints": (10, 43, 58),
    "background_fields": (2, 101, 8, 8, 8),
    "background_projections": (2, 101, 13, 2),
    "main_projections": (2, 2, 58, 2, 101, 13, 2),
    "negative_projections": (2, 2, 2, 2, 101, 13, 2),
    "rhs_coefficients": (2, 2, 4, 58, 58),
    "rhs_errors": (2, 2, 4, 58, 58),
    "rhs_resolution": (2, 2, 4, 58, 58),
    "constraint_coefficients": (2, 2, 10, 43, 58),
    "constraint_errors": (2, 2, 10, 43, 58),
    "constraint_resolution": (2, 2, 10, 43, 58),
    "modal_errors": (2, 2, 58, 101, 13, 2),
    "modal_limits": (2, 2, 58, 101, 13),
    "field_stats": (2, 2, 58, 101, 11),
    "roundtrip_stats": (482, 58, 3),
    "negative_changes": (2, 2, 2, 101, 13, 2),
    "negative_errors": (2, 2, 2, 101, 13, 2),
    "negative_resolution": (2, 2, 2, 101, 13),
    "negative_limits": (2, 2, 2, 101, 13),
    "negative_field_scales": (2, 2, 2, 101),
}


def negative_gate(changes, errors, resolution, limits, targets, *, fine):
    np = _numpy()
    changes = _complex_array(changes, shape=(101, 13))
    errors = _complex_array(errors, shape=changes.shape)
    resolution = _real_array(resolution, shape=changes.shape)
    limits = _real_array(limits, shape=changes.shape)
    if type(fine) is not bool or tuple(targets) not in ((17, 18), (59, 60, 61)):
        raise ValueError("unregistered negative target rows")
    if np.any(resolution < 0.0) or np.any(limits < 0.0):
        raise ValueError("nonnegative uncertainty and limits required")
    target_change = np.abs(changes[list(targets)])
    target_error = np.abs(errors[list(targets)])
    mask = (target_change > 0.0) & (target_error > limits[list(targets)])
    if fine:
        mask &= target_change > 64.0 * resolution[list(targets)]
    return {
        "target_rows": list(targets),
        "same_slot_witnesses": int(np.count_nonzero(mask)),
        "changed_slots": int(np.count_nonzero(np.abs(changes) > 0.0)),
        "rejected": bool(np.any(mask)),
    }


def run(reference, *, guard, check, array_sink, json_sink):
    """Reevaluate all 482 states, then save independent numerical evidence.

    A candidate failure is data and does not short-circuit later cases.  An
    exception/deadline stops work and saves a truthful partial call ledger;
    it never runs a substitute case or performs an automatic retry.
    """
    from . import kernel

    np = _numpy()
    if type(reference) is not dict or set(reference) != {
        "parameters",
        "rhs",
        "constraints",
    }:
        raise ValueError("reference requires the fixed three-field schema")
    rhs_reference = _real_array(reference["rhs"], shape=(4, 58, 58))
    constraint_reference = _real_array(reference["constraints"], shape=(10, 43, 58))
    for callback in (guard, check, array_sink, json_sink):
        if not callable(callback):
            raise ValueError("callbacks must be callable")
    guard()
    context = kernel.build_context(reference["parameters"], check=check)
    q0 = _real_array(context.q0, shape=(58,))
    scales = _real_array(context.scales, shape=(58,))
    shape = _real_array(context.f, shape=(8, 8, 8))
    if tuple(context.modes) != MODES or tuple(context.W) != WAVES:
        raise ValueError("shared evaluator mode registration drift")
    if not np.array_equal(scales, np.maximum(1.0, np.abs(q0))):
        raise ValueError("column scale registration drift")
    fhat = project_explicit(shape, check=check)
    if np.any(np.abs(fhat) <= 1.0 / 32.0):
        raise ValueError("registered shape projection is unresolved")
    arrays = {
        name: np.zeros(dims, dtype=np.float64) for name, dims in ARRAY_SPECS.items()
    }
    arrays.update(
        {
            "q0": q0.copy(),
            "scales": scales.copy(),
            "shape_field": shape.copy(),
            "shape_projection": complex_record(fhat),
            "wave_numbers": np.stack(
                [modified_wave_numbers(order) for order in ORDERS]
            ),
            "reference_rhs": rhs_reference.copy(),
            "reference_constraints": constraint_reference.copy(),
        }
    )
    saved = set()
    calls = []
    started = 0
    plan = evaluation_plan()
    summary = {
        "schema": "case12-binary64-path-summary-v1",
        "method": "explicit-cos-sin-independent-replay",
        "planned_evaluations": 482,
        "started_evaluations": 0,
        "completed_evaluations": 0,
        "completed": False,
        "accepted": False,
        "empirical_tolerances": True,
        "rigorous_roundoff_certificate": False,
    }

    def save(names):
        for name in names:
            guard()
            if name in saved:
                raise ValueError("duplicate array emission")
            array_sink(name, _real_array(arrays[name], shape=ARRAY_SPECS[name]))
            saved.add(name)

    def evaluate(category, order, epsilon, slot, sign, variant):
        nonlocal started
        guard()
        identity = (category, order, epsilon, slot, sign, variant)
        if started >= len(plan) or plan[started] != identity:
            raise ValueError("scientific evaluation schedule drift")
        q = np.broadcast_to(q0[:, None, None, None], (58, 8, 8, 8)).copy()
        if slot is not None:
            q[slot] += sign * epsilon * scales[slot] * shape
        if category == "background":
            label = f"background/fd{order}"
        elif category == "main":
            label = f"main/fd{order}/eps{EPSILONS.index(epsilon)}/q{slot}/sign{sign}"
        else:
            label = (
                f"negative/{variant}/fd{order}/eps{EPSILONS.index(epsilon)}/sign{sign}"
            )
        index = started
        started += 1
        row = {
            "index": index,
            "label": label,
            "category": category,
            "order": order,
            "epsilon": epsilon,
            "slot": slot,
            "sign": sign,
            "variant": variant,
            "started": True,
            "completed": False,
        }
        calls.append(row)
        json_sink(
            "progress",
            {
                "planned": 482,
                "started": started,
                "completed": sum(c["completed"] for c in calls),
                "active_label": label,
                "phase": "before_actual_evaluation",
            },
        )
        actual_started = time.monotonic()
        result = kernel.evaluate_state(context, q, order, variant=variant, check=check)
        calls[index]["actual_call_seconds_including_guards"] = (
            time.monotonic() - actual_started
        )
        fields = _real_array(result["fields"], shape=(101, 8, 8, 8))
        recovered = _real_array(result["roundtrip"], shape=q.shape)
        arrays["roundtrip_stats"][index] = _roundtrip_statistics(q, recovered)
        if type(result["hashes"]) is not dict or set(result["hashes"]) != {
            "input",
            "rhs",
            "physical43",
            "roundtrip",
        }:
            raise ValueError("evaluator hash schema mismatch")
        for value in result["hashes"].values():
            if (
                type(value) is not str
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("invalid evaluator digest")
        row.update(
            completed=True, hashes=dict(result["hashes"]), metadata=result["metadata"]
        )
        json_sink(
            "progress",
            {
                "planned": 482,
                "started": started,
                "completed": sum(c["completed"] for c in calls),
                "active_label": label,
                "phase": "after_actual_evaluation",
            },
        )
        check()
        return fields

    coefficient_results = []
    negative_results = []
    try:
        save(
            (
                "q0",
                "scales",
                "shape_field",
                "shape_projection",
                "wave_numbers",
                "reference_rhs",
                "reference_constraints",
            )
        )
        preflight = independent_preflight(context, fhat, check=check)
        json_sink("design_and_stencils", preflight)
        for oi, order in enumerate(ORDERS):
            fields = evaluate("background", order, None, None, None, "baseline")
            arrays["background_fields"][oi] = fields
            arrays["background_projections"][oi] = complex_record(
                project_explicit(fields, check=check)
            )
        save(("background_fields", "background_projections"))
        background_max = np.max(np.abs(arrays["background_fields"]), axis=(2, 3, 4))
        background_ok = bool(
            np.all(background_max <= 4096.0 * U * np.maximum(1.0, background_max))
        )
        for oi, order in enumerate(ORDERS):
            basis = derivative_shapes(fhat, order)
            rhs_weight = np.maximum(1.0, np.sum(np.abs(rhs_reference), axis=2))
            constraint_weight = np.maximum(
                1.0, np.sum(np.abs(constraint_reference), axis=2)
            )
            basis_max = np.max(np.abs(basis), axis=(1, 2, 3))
            field_weight = np.concatenate(
                (
                    np.sum(rhs_weight * basis_max[:4, None], axis=0),
                    np.sum(constraint_weight * basis_max[:, None], axis=0),
                )
            )
            symbols_reference = np.concatenate(
                (
                    assemble_symbols(rhs_reference, order),
                    assemble_symbols(constraint_reference, order),
                ),
                axis=0,
            )
            modal_weight = np.concatenate(
                (
                    _modal_weights(rhs_reference, order),
                    _modal_weights(constraint_reference, order),
                ),
                axis=0,
            )
            for ei, epsilon in enumerate(EPSILONS):
                for slot in range(58):
                    minus = evaluate("main", order, epsilon, slot, -1, "baseline")
                    minus_projection = project_explicit(minus, check=check)
                    arrays["main_projections"][oi, ei, slot, 0] = complex_record(
                        minus_projection
                    )
                    plus = evaluate("main", order, epsilon, slot, 1, "baseline")
                    plus_projection = project_explicit(plus, check=check)
                    arrays["main_projections"][oi, ei, slot, 1] = complex_record(
                        plus_projection
                    )
                    odd_projection = (plus_projection - minus_projection) / (
                        2.0 * epsilon * scales[slot]
                    )
                    symbols = odd_projection / fhat[None, :]
                    prediction = np.concatenate(
                        (
                            np.einsum(
                                "br,bijk->rijk", rhs_reference[:, :, slot], basis[:4]
                            ),
                            np.einsum(
                                "br,bijk->rijk", constraint_reference[:, :, slot], basis
                            ),
                        ),
                        axis=0,
                    )
                    stats = _field_statistics(
                        plus,
                        minus,
                        arrays["background_fields"][oi],
                        odd_projection,
                        prediction,
                        epsilon,
                        scales[slot],
                        field_weight,
                        check=check,
                    )
                    arrays["field_stats"][oi, ei, slot] = stats
                    nu = stats[:, 8, None] / np.abs(fhat)[None, :]
                    arrays["modal_errors"][oi, ei, slot] = complex_record(
                        symbols - symbols_reference[:, slot]
                    )
                    arrays["modal_limits"][oi, ei, slot] = (
                        64.0 * epsilon * epsilon * modal_weight + 8.0 * nu
                    )
                    rhs, rhs_nu = recover_coefficients(symbols[:58], nu[:58], order)
                    constraints, constraints_nu = recover_coefficients(
                        symbols[58:], nu[58:], order, second_order=True
                    )
                    arrays["rhs_coefficients"][oi, ei, :, :, slot] = rhs
                    arrays["rhs_resolution"][oi, ei, :, :, slot] = rhs_nu
                    arrays["constraint_coefficients"][oi, ei, :, :, slot] = constraints
                    arrays["constraint_resolution"][oi, ei, :, :, slot] = constraints_nu
                    del minus, plus
                    check()
                for label, ref in (
                    ("rhs", rhs_reference),
                    ("constraint", constraint_reference),
                ):
                    actual = arrays[label + "_coefficients"][oi, ei]
                    arrays[label + "_errors"][oi, ei] = actual - ref
                    gate = coefficient_gates(
                        actual,
                        arrays[label + "_resolution"][oi, ei],
                        ref,
                        epsilon,
                        fine=ei == 1,
                    )
                    coefficient_results.append(
                        {"kind": label, "order": order, "epsilon": epsilon, **gate}
                    )
        save(
            (
                "main_projections",
                "rhs_coefficients",
                "rhs_errors",
                "rhs_resolution",
                "constraint_coefficients",
                "constraint_errors",
                "constraint_resolution",
                "modal_errors",
                "modal_limits",
                "field_stats",
            )
        )
        for vi, (variant, slot) in enumerate(MUTANTS):
            for oi, order in enumerate(ORDERS):
                reference_symbols = np.concatenate(
                    (
                        assemble_symbols(rhs_reference, order),
                        assemble_symbols(constraint_reference, order),
                    ),
                    axis=0,
                )[:, slot]
                for ei, epsilon in enumerate(EPSILONS):
                    minus = evaluate("negative", order, epsilon, slot, -1, variant)
                    pminus = project_explicit(minus, check=check)
                    arrays["negative_projections"][vi, oi, ei, 0] = complex_record(
                        pminus
                    )
                    plus = evaluate("negative", order, epsilon, slot, 1, variant)
                    pplus = project_explicit(plus, check=check)
                    arrays["negative_projections"][vi, oi, ei, 1] = complex_record(
                        pplus
                    )
                    mutated = (pplus - pminus) / (
                        2.0 * epsilon * scales[slot] * fhat[None, :]
                    )
                    raw_baseline = arrays["main_projections"][oi, ei, slot]
                    baseline = (
                        (raw_baseline[1, :, :, 0] + 1j * raw_baseline[1, :, :, 1])
                        - (raw_baseline[0, :, :, 0] + 1j * raw_baseline[0, :, :, 1])
                    ) / (2.0 * epsilon * scales[slot] * fhat[None, :])
                    changes = mutated - baseline
                    errors = mutated - reference_symbols
                    magnitude = np.maximum(
                        1.0,
                        np.maximum(
                            np.max(np.abs(plus), axis=(1, 2, 3)),
                            np.maximum(
                                np.max(np.abs(minus), axis=(1, 2, 3)),
                                np.max(
                                    np.abs(arrays["background_fields"][oi]),
                                    axis=(1, 2, 3),
                                ),
                            ),
                        ),
                    )
                    nu = (
                        4096.0 * U * magnitude / (epsilon * scales[slot])
                        + arrays["field_stats"][oi, ei, slot, :, 8]
                    )[:, None] / np.abs(fhat)[None, :]
                    limits = arrays["modal_limits"][oi, ei, slot]
                    arrays["negative_changes"][vi, oi, ei] = complex_record(changes)
                    arrays["negative_errors"][vi, oi, ei] = complex_record(errors)
                    arrays["negative_resolution"][vi, oi, ei] = nu
                    arrays["negative_limits"][vi, oi, ei] = limits
                    arrays["negative_field_scales"][vi, oi, ei] = magnitude
                    gate = negative_gate(
                        changes,
                        errors,
                        nu,
                        limits,
                        (17, 18) if vi == 0 else (59, 60, 61),
                        fine=ei == 1,
                    )
                    negative_results.append(
                        {"variant": variant, "order": order, "epsilon": epsilon, **gate}
                    )
                    del minus, plus
        save(
            (
                "negative_projections",
                "negative_changes",
                "negative_errors",
                "negative_resolution",
                "negative_limits",
                "negative_field_scales",
                "roundtrip_stats",
            )
        )
        convergence = {
            label: convergence_gate(
                arrays[label + "_errors"], arrays[label + "_resolution"]
            )
            for label in ("rhs", "constraint")
        }
        modal_error = (
            arrays["modal_errors"][..., 0] + 1j * arrays["modal_errors"][..., 1]
        )
        acceptance = {
            "schedule": len(calls) == started == 482,
            "design_and_stencils": preflight["accepted"],
            "background": background_ok,
            "roundtrip": bool(np.all(arrays["roundtrip_stats"][..., 2] <= 1.0)),
            "coefficients": all(row["accepted"] for row in coefficient_results),
            "convergence": all(row["accepted"] for row in convergence.values()),
            "modal": bool(np.all(np.abs(modal_error) <= arrays["modal_limits"])),
            "full_grid": bool(np.all(arrays["field_stats"][..., 10] == 1.0)),
            "negative_controls": all(row["rejected"] for row in negative_results),
        }
        acceptance["accepted"] = all(acceptance.values())
        summary.update(
            {
                "started_evaluations": started,
                "completed_evaluations": sum(c["completed"] for c in calls),
                "rhs_evaluations": len(calls),
                "physical43_evaluations": len(calls),
                "completed": True,
                "accepted": all(acceptance.values()),
                "acceptance": acceptance,
                "coefficient_gates": coefficient_results,
                "convergence": convergence,
                "negative_controls": negative_results,
                "saved_arrays": sorted(saved),
                "source_bridge": replay_manifest(),
                "raw_all_fields_archived": False,
                "driver49_evaluated": False,
                "time_integration_performed": False,
            }
        )
    except BaseException as exc:
        summary.update(
            {
                "started_evaluations": started,
                "completed_evaluations": sum(c["completed"] for c in calls),
                "completed": False,
                "accepted": False,
                "saved_arrays": sorted(saved),
                "exception_type": type(exc).__name__,
            }
        )
        try:
            json_sink(
                "partial_calls",
                {
                    "schema": "case12-binary64-calls-v1",
                    "planned": 482,
                    "records": calls,
                    "completed": sum(c["completed"] for c in calls),
                    "accepted": False,
                },
            )
            json_sink("acceptance", summary)
        except BaseException:
            pass
        raise
    json_sink(
        "calls",
        {"schema": "case12-binary64-calls-v1", "planned": 482, "records": calls},
    )
    json_sink("acceptance", summary)
    return summary

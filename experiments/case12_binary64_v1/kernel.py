"""Unexecuted, fixed 8-cubed binary64 alignment candidate (production path).

Scientific imports and evaluation are lazy. The owner must enforce the resource
gate before calling any numerical API. Original exported sources are never edited.
This module shares actual evaluation with replay, not projection or acceptance.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, replace
from fractions import Fraction
import hashlib
import importlib
import inspect
import time
from types import SimpleNamespace


N = 8
ORDERS = (2, 4)
EPSILONS = (2.0**-10, 2.0**-12)
SIGNS = (-1, 1)
W = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1))
MODES = ((0, 0, 0),) + W + tuple(tuple(-x for x in k) for k in W)
PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
UNIT_ROUNDOFF = 2.0**-53
EVALUATIONS = 482
NEGATIVES = (
    ("kappa1_doubled", 18, (17, 18)),
    ("stored_trace_diagnostic", 11, (59, 60, 61)),
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
    **{
        name: (2, 2, 4, 58, 58)
        for name in ("rhs_coefficients", "rhs_errors", "rhs_resolution")
    },
    **{
        name: (2, 2, 10, 43, 58)
        for name in (
            "constraint_coefficients",
            "constraint_errors",
            "constraint_resolution",
        )
    },
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
FIELD_COLUMNS = (
    "error_max",
    "error_rms",
    "error_argmax_flat",
    "offmode_max",
    "offmode_rms",
    "even_raw_max",
    "even_raw_rms",
    "field_limit",
    "field_resolution",
    "signed_scale_B",
    "passed",
)
ROUNDTRIP_COLUMNS = ("maximum_absolute", "rms", "maximum_ratio_to_limit")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numpy():
    return importlib.import_module("numpy")


def finite_array(value, shape=None):
    np = numpy()
    require(
        isinstance(value, np.ndarray)
        and value.dtype == np.dtype("float64")
        and (shape is None or value.shape == shape)
        and np.all(np.isfinite(value)),
        "finite float64 array with registered shape required",
    )
    return value


def as_pairs(value):
    np = numpy()
    return np.stack((np.asarray(value).real, np.asarray(value).imag), axis=-1).astype(
        "<f8"
    )


def from_pairs(value):
    value = finite_array(value)
    require(value.shape[-1] == 2, "complete complex real/imaginary pair")
    return value[..., 0] + 1j * value[..., 1]


def bytes_hash(value):
    return hashlib.sha256(
        numpy().ascontiguousarray(value, dtype="<f8").tobytes()
    ).hexdigest()


def exact_float(value):
    require(type(value) is str, "canonical rational input")
    rational = Fraction(value)
    require(str(rational) == value, "canonical rational input")
    return float(rational)


def compile_trace_mutant(function):
    """Replace the actual assignment, retaining all other NumPy source AST."""
    source = inspect.getsource(function)
    tree = ast.parse(source)
    original = copy.deepcopy(tree)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "actual_trace"
    ]
    require(len(matches) == 1, "unique actual_trace assignment")
    before = ast.dump(matches[0].value, include_attributes=False)
    expected = ast.parse("np.trace(mixed, axis1=-2, axis2=-1)", mode="eval").body
    require(
        before == ast.dump(expected, include_attributes=False),
        "frozen trace expression",
    )
    matches[0].value = ast.parse("q[17]", mode="eval").body
    ast.fix_missing_locations(tree)
    namespace = dict(function.__globals__)
    exec(compile(tree, "<binary64-stored-trace-mutant>", "exec"), namespace)
    audit = dict(
        name="stored_trace_diagnostic",
        source_function="constraint_fields",
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        original_ast=ast.dump(original, include_attributes=False),
        mutated_ast=ast.dump(tree, include_attributes=False),
        replacement="actual_trace = q[17]",
        assignment_replacements=1,
        hamiltonian_trace_squared_input_also_changed=True,
        original_module_globals_unchanged=True,
    )
    return namespace[function.__name__], audit


def build_context(parameters, *, check=lambda: None):
    """Construct fixed background/context, without evaluating any RHS/diagnostic."""
    check()
    np = numpy()
    prefix = "science.numerical_relativity."
    checkpoint = importlib.import_module(prefix + "fo_ccz4_checkpoint")
    rhs = importlib.import_module(prefix + "fo_ccz4_rhs")
    principal = importlib.import_module(prefix + "fo_ccz4_principal_symbol")
    diagnostic = importlib.import_module(prefix + "fo_ccz4_relaxation_convergence")
    gauge_module = importlib.import_module(prefix + "ccz4_reference")
    profile_module = importlib.import_module(prefix + "fo_ccz4_gauge_profile")
    config_record = {
        "kappa1": "1/10",
        "kappa2": "0",
        "kappa3": "1",
        "cleaning_speed": "1",
        "ordering_mu": "1",
        "shift_strength": "1",
        "tau_inverse": "1",
        "k0": "0",
        "c": "1",
    }
    gauge_record = {
        "lapse_driver": "one-plus-log",
        "shift_driver": "gamma-driver",
        "advective": True,
        "lapse_coefficient": "2",
        "theta_coupling": "2",
        "shift_coefficient": "1",
        "shift_damping": "0",
    }
    background = {
        "alpha": "17/20",
        "chi": "18/25",
        "shift": ["3/100", "-1/25", "1/20"],
        "metric": [
            ["1", "3/25", "-2/25"],
            ["3/25", "634/625", "63/1250"],
            ["-2/25", "63/1250", "101/100"],
        ],
    }
    require(
        type(parameters) is dict
        and parameters["background"] == background
        and parameters["config"] == config_record
        and parameters["gauge"] == gauge_record
        and parameters["profile"] == {"shift_scale": "3", "ordering_scale": "1/5"},
        "fixed case12 background and complete parameters",
    )
    require(parameters["gauge"]["advective"] is True, "actual advective boolean")
    gauge = gauge_module.GaugeConfig(
        **{
            k: exact_float(v)
            if type(v) is str and k not in ("lapse_driver", "shift_driver")
            else v
            for k, v in gauge_record.items()
        }
    )
    config = rhs.FOCCZ4Config(**{k: exact_float(v) for k, v in config_record.items()})
    profile = profile_module.FOCCZ4GaugeProfile(shift_scale=3.0, ordering_scale=0.2)
    q0 = np.zeros(58, dtype=np.float64)
    for index, (i, j) in enumerate(PAIRS):
        q0[index] = exact_float(background["metric"][i][j])
    q0[6], q0[10] = (
        np.log(exact_float(background["alpha"])),
        0.5 * np.log(exact_float(background["chi"])),
    )
    q0[7:10] = [exact_float(x) for x in background["shift"]]
    coordinates = np.indices((N, N, N), dtype=np.float64) / N
    f = np.ones((N, N, N), dtype=np.float64)
    for mode in W:
        f += np.cos(2 * np.pi * sum(mode[a] * coordinates[a] for a in range(3)))
    f /= 8
    mutant, mutant_manifest = compile_trace_mutant(diagnostic.constraint_fields)
    check()
    return SimpleNamespace(
        np=np,
        parameters=copy.deepcopy(parameters),
        q0=q0,
        scales=np.maximum(1, np.abs(q0)),
        f=f,
        modes=MODES,
        W=W,
        gauge=gauge,
        config=config,
        profile=profile,
        checkpoint=checkpoint,
        rhs=rhs,
        principal=principal,
        diagnostic=diagnostic,
        trace_mutant=mutant,
        mutant_manifest=mutant_manifest,
    )


def physical_rows(fields, slots):
    """Pack the actual physical43, using the current perturbed metric for Z."""
    np = numpy()
    h = np.empty((*slots.shape[1:], 3, 3), dtype=np.float64)
    for index, (i, j) in enumerate(PAIRS):
        h[..., i, j] = h[..., j, i] = slots[index]
    z = 0.5 * np.einsum("...ij,...j->...i", h, fields["Ghat"])
    rows = [
        fields["H"],
        *(fields["M"][..., i] for i in range(3)),
        fields["Theta"],
        *(z[..., i] for i in range(3)),
        *(fields["A"][..., i] for i in range(3)),
        *(fields["B"][..., index] for index in range(9)),
        *(fields["D"][..., index] for index in range(18)),
        *(fields["P"][..., i] for i in range(3)),
        fields["det_h"],
        fields["trace_a"],
    ]
    return finite_array(np.stack(rows), (43, N, N, N))


def evaluate_state(context, q, order, variant="baseline", *, check=lambda: None):
    """Exactly one original binary64 RHS and one actual physical43 evaluation."""
    np = context.np
    finite_array(q, (58, N, N, N))
    require(
        type(order) is int
        and order in ORDERS
        and variant in ("baseline", "kappa1_doubled", "stored_trace_diagnostic"),
        "registered actual case",
    )
    check()
    header = dict(
        magic=context.checkpoint.MAGIC,
        version=4,
        dimension=3,
        points=N,
        scenario=0,
        formulation=0,
        solver_mode=1,
        variable_count=58,
        dt=1 / 512,
        time=0.0,
        amplitude=0.0,
        wavelength=1.0,
        step=0,
        spatial_order=order,
    )
    data = context.checkpoint.FOCheckpoint(
        header, tuple(q), context.profile, tau_inverse=1.0
    )
    state = data.to_state()
    roundtrip = finite_array(np.stack(context.checkpoint.state_slots(state)), q.shape)
    config = (
        replace(context.config, kappa1=0.2)
        if variant == "kappa1_doubled"
        else context.config
    )
    check()
    rhs = context.rhs.fo_ccz4_rhs_3d(
        state,
        gauge=context.gauge,
        config=config,
        boundary_mode="periodic",
        gauge_profile=context.profile,
    )
    rhs_fields = finite_array(
        np.stack(context.principal._rhs_components(rhs)), (58, N, N, N)
    )
    check()
    diagnostic = (
        context.trace_mutant
        if variant == "stored_trace_diagnostic"
        else context.diagnostic.constraint_fields
    )
    actual_constraints = physical_rows(diagnostic(data), q)
    fields = np.concatenate((rhs_fields, actual_constraints), axis=0)
    check()
    return dict(
        fields=fields,
        roundtrip=roundtrip,
        hashes=dict(
            input=bytes_hash(q),
            rhs=bytes_hash(rhs_fields),
            physical43=bytes_hash(actual_constraints),
            roundtrip=bytes_hash(roundtrip),
        ),
        metadata=dict(
            order=order,
            variant=variant,
            rhs_calls=1,
            physical43_calls=1,
            driver49_executed=False,
            all_58_auxiliary_slots_independent=True,
        ),
    )


def project_fields(fields):
    """FFT projection; replay uses direct cosine/sine summation instead."""
    np = numpy()
    finite_array(fields)
    require(fields.shape[-3:] == (N, N, N), "registered periodic grid")
    spectrum = np.fft.fftn(fields, axes=(-3, -2, -1)) / N**3
    indices = tuple(np.array([m[a] % N for m in MODES]) for a in range(3))
    return spectrum[(..., *indices)]


def reconstruct(projection):
    np = numpy()
    require(
        projection.shape[-1] == 13 and np.all(np.isfinite(projection)),
        "all13 modal values",
    )
    spectrum = np.zeros((*projection.shape[:-1], N, N, N), dtype=np.complex128)
    for index, mode in enumerate(MODES):
        spectrum[(..., *(x % N for x in mode))] = N**3 * projection[..., index]
    return np.fft.ifftn(spectrum, axes=(-3, -2, -1)).real


def wave_numbers(order):
    np = numpy()
    require(type(order) is int and order in ORDERS, "FD2 or FD4")
    theta = 2 * np.pi * np.asarray(MODES, dtype=np.float64) / N
    return N * (
        np.sin(theta) if order == 2 else (8 * np.sin(theta) - np.sin(2 * theta)) / 6
    )


def symbols(reference, k):
    np = numpy()
    rhs = reference["rhs"][0, :, :, None].astype(np.complex128)
    rhs = np.repeat(rhs, len(k), axis=2)
    con = np.repeat(
        reference["constraints"][0, :, :, None].astype(np.complex128), len(k), axis=2
    )
    for axis in range(3):
        rhs += 1j * reference["rhs"][axis + 1, :, :, None] * k[:, axis]
        con += 1j * reference["constraints"][axis + 1, :, :, None] * k[:, axis]
    for index, (a, b) in enumerate(PAIRS):
        con -= reference["constraints"][index + 4, :, :, None] * (k[:, a] * k[:, b])
    return np.concatenate((rhs, con), axis=0)


def monomials(k):
    np = numpy()
    return np.stack(
        (
            np.ones(len(k)),
            *(1j * k[:, a] for a in range(3)),
            *(-k[:, a] * k[:, b] for a, b in PAIRS),
        )
    )


def design_diagnostics(shape_projection, order):
    """The real coefficient design, including all signed measured shape modes."""
    np = numpy()
    require(
        isinstance(shape_projection, np.ndarray)
        and shape_projection.dtype == np.dtype("complex128")
        and shape_projection.shape == (13,)
        and np.all(np.isfinite(shape_projection)),
        "complete finite complex shape projection",
    )
    factors = monomials(wave_numbers(order))
    rows = []
    for count in (4, 10):
        design = (factors[:count] * shape_projection).T
        real_design = np.concatenate((design.real, design.imag), axis=0)
        singular = np.linalg.svd(real_design, compute_uv=False)
        rank = int(np.linalg.matrix_rank(real_design))
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else None
        rows.append(
            dict(
                coefficients=count,
                shape=[26, count],
                rank=rank,
                singular_values=singular.tolist(),
                condition=condition,
                accepted=rank == count
                and condition is not None
                and bool(np.isfinite(condition)),
            )
        )
    return dict(order=order, rows=rows, accepted=all(row["accepted"] for row in rows))


def scalar_controls(context, *, check=lambda: None):
    """Exercise actual scalar D1 paths, without a full RHS/diagnostic call.

    Every registered mode is present in f. The second derivative is explicitly
    D1 composed with D1; this does not call a separate D2 stencil.
    """
    np = context.np
    fhat = project_fields(context.f)
    designs, stencils = [], []
    for order in ORDERS:
        check()
        designs.append(design_diagnostics(fhat, order))
        basis = reconstruct(monomials(wave_numbers(order)) * fhat)
        grid = context.checkpoint.Grid3D(
            (N,) * 3, (0.0,) * 3, (1.0,) * 3, boundary="periodic", spatial_order=order
        )
        holder = SimpleNamespace(grid=grid)
        methods = (
            ("actual_rhs_d1", lambda value, axis: context.rhs._d1(value, holder, axis)),
            (
                "actual_diagnostic_derivative",
                lambda value, axis: context.diagnostic.derivative(value, axis, order),
            ),
        )
        for name, derivative in methods:
            first = []
            for axis in range(3):
                check()
                first.append(finite_array(derivative(context.f, axis), (N, N, N)))
            values = first + [
                finite_array(derivative(first[b], a), (N, N, N)) for a, b in PAIRS
            ]
            for index, value in enumerate(values, 1):
                check()
                expected = basis[index]
                error = float(np.max(np.abs(value - expected)))
                expected_max = float(np.max(np.abs(expected)))
                limit = 4096 * UNIT_ROUNDOFF * max(1.0, expected_max)
                stencils.append(
                    dict(
                        order=order,
                        method=name,
                        derivative_index=index,
                        compared_grid_slots=N**3,
                        error_max=error,
                        expected_max=expected_max,
                        limit=limit,
                        accepted=error <= limit,
                    )
                )
    return dict(
        schema="case12-binary64-scalar-controls-v1",
        design_convention="real/imaginary stacked actual shape times modified-wave monomials",
        condition_threshold=None,
        condition_must_be_finite=True,
        designs=designs,
        stencils=stencils,
        scalar_comparisons=len(stencils),
        compared_grid_slots=len(stencils) * N**3,
        full_state_evaluations=0,
        accepted=all(row["accepted"] for row in designs + stencils),
    )


def recover(symbol, resolution, k):
    """Explicit full-column recovery, with absolute propagation of nu."""
    np = numpy()
    require(
        symbol.shape == (101, 13) and resolution.shape == (101, 13), "full modal column"
    )
    rhs, con = np.zeros((4, 58)), np.zeros((10, 43))
    rn, cn = np.zeros_like(rhs), np.zeros_like(con)
    rhs[0], rn[0] = symbol[:58, 0].real, resolution[:58, 0]
    con[0], cn[0] = symbol[58:, 0].real, resolution[58:, 0]
    for a in range(3):
        require(k[a + 1, a] != 0, "nonzero registered axis wave number")
        rhs[a + 1] = symbol[:58, a + 1].imag / k[a + 1, a]
        rn[a + 1] = resolution[:58, a + 1] / abs(k[a + 1, a])
        con[a + 1] = symbol[58:, a + 1].imag / k[a + 1, a]
        cn[a + 1] = resolution[58:, a + 1] / abs(k[a + 1, a])
        diagonal = 4 + PAIRS.index((a, a))
        con[diagonal] = (con[0] - symbol[58:, a + 1].real) / k[a + 1, a] ** 2
        cn[diagonal] = (cn[0] + resolution[58:, a + 1]) / k[a + 1, a] ** 2
    for mode_index, (a, b) in zip((4, 5, 6), ((0, 1), (0, 2), (1, 2)), strict=True):
        aa, bb, ab = (4 + PAIRS.index(pair) for pair in ((a, a), (b, b), (a, b)))
        ka, kb = k[mode_index, a], k[mode_index, b]
        con[ab] = (
            con[0] - ka**2 * con[aa] - kb**2 * con[bb] - symbol[58:, mode_index].real
        ) / (ka * kb)
        cn[ab] = (
            cn[0] + ka**2 * cn[aa] + kb**2 * cn[bb] + resolution[58:, mode_index]
        ) / abs(ka * kb)
    return rhs, con, rn, cn


def matrix_acceptance(coefficients, reference, resolution):
    np = numpy()
    finite_array(coefficients)
    finite_array(reference)
    finite_array(resolution)
    require(
        reference.ndim == 3
        and all(reference.shape)
        and coefficients.ndim == 5
        and coefficients.shape[:2] == (2, 2)
        and coefficients.shape == resolution.shape
        and coefficients.shape[2:] == reference.shape
        and np.all(resolution > 0),
        "full order/epsilon matrices",
    )
    weights = np.maximum(1, np.sum(np.abs(reference), axis=-1))
    error = np.abs(coefficients - reference)
    limits = (
        np.asarray(EPSILONS)[None, :, None, None, None] ** 2
        * 64
        * weights[None, None, :, :, None]
        + 8 * resolution
    )
    all_slots = bool(np.all(error <= limits))
    fine_nu, fine_error = resolution[:, 1], error[:, 1]
    nonzero = reference != 0
    resolved = bool(np.all(np.abs(reference)[nonzero] >= 32 * fine_nu[:, nonzero]))
    relative = bool(np.all(fine_error[:, nonzero] <= 1e-3 * np.abs(reference)[nonzero]))
    high_error, low_error = np.max(error[:, 0], axis=-1), np.max(error[:, 1], axis=-1)
    high_nu, low_nu = (
        np.max(resolution[:, 0], axis=-1),
        np.max(resolution[:, 1], axis=-1),
    )
    measurable = high_error > 64 * high_nu
    shrinking = bool(
        np.all(
            low_error[measurable]
            <= high_error[measurable] / 8
            + 8 * (high_nu[measurable] + low_nu[measurable])
        )
    )
    return dict(
        all_slots=all_slots,
        fine_nonzero_resolved=resolved,
        fine_nonzero_relative=relative,
        amplitude_contraction=shrinking,
        measurable_rows=int(np.count_nonzero(measurable)),
        roundoff_limited_rows=int(measurable.size - np.count_nonzero(measurable)),
        accepted=all_slots and resolved and relative and shrinking,
    )


def _run(reference, *, calls, guard, check, array_sink, json_sink):
    """One finite 482-evaluation production path. Failure never changes the plan."""
    require(
        type(reference) is dict
        and set(reference) == {"parameters", "rhs", "constraints"},
        "registered reference keys",
    )
    finite_array(reference["rhs"], (4, 58, 58))
    finite_array(reference["constraints"], (10, 43, 58))
    guard()
    context = build_context(reference["parameters"], check=check)
    np = context.np
    arrays, sent = {}, set()
    for name, shape in ARRAY_SPECS.items():
        arrays[name] = np.zeros(shape, dtype="<f8")

    def save(name):
        guard()
        require(name not in sent, "each complete array saved once")
        finite_array(arrays[name], ARRAY_SPECS[name])
        array_sink(name, arrays[name])
        sent.add(name)

    def evaluate(q, order, variant, label):
        guard()
        require(len(calls) < EVALUATIONS, "no extra actual state evaluations")
        index = len(calls)
        calls.append(
            dict(
                index=index,
                label=label,
                order=order,
                variant=variant,
                started=True,
                completed=False,
            )
        )
        json_sink(
            "progress",
            dict(
                planned=EVALUATIONS,
                started=len(calls),
                completed=sum(c["completed"] for c in calls),
                active_label=label,
                phase="before_actual_evaluation",
            ),
        )
        actual_started = time.monotonic()
        value = evaluate_state(context, q, order, variant, check=check)
        calls[index]["actual_call_seconds_including_guards"] = (
            time.monotonic() - actual_started
        )
        difference = value["roundtrip"] - q
        limit = 128 * UNIT_ROUNDOFF * np.maximum(1, np.abs(q))
        arrays["roundtrip_stats"][index, :, 0] = np.max(
            np.abs(difference), axis=(1, 2, 3)
        )
        arrays["roundtrip_stats"][index, :, 1] = np.sqrt(
            np.mean(difference**2, axis=(1, 2, 3))
        )
        arrays["roundtrip_stats"][index, :, 2] = np.max(
            np.abs(difference) / limit, axis=(1, 2, 3)
        )
        calls[index].update(completed=True, hashes=value["hashes"])
        json_sink(
            "progress",
            dict(
                planned=EVALUATIONS,
                started=len(calls),
                completed=sum(c["completed"] for c in calls),
                active_label=label,
                phase="after_actual_evaluation",
            ),
        )
        return value["fields"]

    arrays.update(
        q0=context.q0.copy(),
        scales=context.scales.copy(),
        shape_field=context.f.copy(),
        reference_rhs=reference["rhs"].copy(),
        reference_constraints=reference["constraints"].copy(),
    )
    fhat = project_fields(context.f)
    require(np.all(np.abs(fhat) > 1 / 32), "all13 known nonzero shape modes")
    arrays["shape_projection"] = as_pairs(fhat)
    arrays["wave_numbers"] = np.stack([wave_numbers(order) for order in ORDERS])
    for name in (
        "q0",
        "scales",
        "shape_field",
        "shape_projection",
        "wave_numbers",
        "reference_rhs",
        "reference_constraints",
    ):
        save(name)
    json_sink(
        "parameters",
        dict(
            schema="case12-binary64-parameters-v1",
            parameters=reference["parameters"],
            actual_gauge=asdict(context.gauge),
            actual_config=asdict(context.config),
            actual_profile=asdict(context.profile),
            mutant=context.mutant_manifest,
            log_chart=dict(
                alpha_target=exact_float(
                    reference["parameters"]["background"]["alpha"]
                ),
                chi_target=exact_float(reference["parameters"]["background"]["chi"]),
                alpha_actual=float(np.exp(context.q0[6])),
                chi_actual=float(np.exp(2 * context.q0[10])),
            ),
            field_columns=list(FIELD_COLUMNS),
            roundtrip_columns=list(ROUNDTRIP_COLUMNS),
            driver49_executed=False,
            time_evolution_executed=False,
        ),
    )
    guard()
    controls = scalar_controls(context, check=check)
    json_sink("scalar_controls", controls)
    guard()
    qbase = np.broadcast_to(context.q0[:, None, None, None], (58, N, N, N)).copy()
    for oi, order in enumerate(ORDERS):
        fields = evaluate(qbase.copy(), order, "baseline", f"background/fd{order}")
        arrays["background_fields"][oi] = fields
        arrays["background_projections"][oi] = as_pairs(project_fields(fields))
    save("background_fields")
    save("background_projections")
    weights_rhs = np.maximum(1, np.sum(np.abs(reference["rhs"]), axis=-1))
    weights_con = np.maximum(1, np.sum(np.abs(reference["constraints"]), axis=-1))
    modal_pass, field_pass = True, True
    for oi, order in enumerate(ORDERS):
        k = arrays["wave_numbers"][oi]
        predicted = symbols(reference, k)
        factors = monomials(k)
        derivative_shapes = reconstruct(factors * fhat)
        field_weights = np.concatenate(
            (
                weights_rhs.T @ np.max(np.abs(derivative_shapes[:4]), axis=(1, 2, 3)),
                weights_con.T @ np.max(np.abs(derivative_shapes), axis=(1, 2, 3)),
            )
        )
        mode_weights = np.concatenate(
            (weights_rhs.T @ np.abs(factors[:4]), weights_con.T @ np.abs(factors)),
            axis=0,
        )
        base = arrays["background_fields"][oi]
        for ei, epsilon in enumerate(EPSILONS):
            for column in range(58):
                check()
                signed = []
                for si, sign in enumerate(SIGNS):
                    q = qbase.copy()
                    q[column] += sign * epsilon * context.scales[column] * context.f
                    fields = evaluate(
                        q,
                        order,
                        "baseline",
                        f"main/fd{order}/eps{ei}/q{column}/sign{sign}",
                    )
                    arrays["main_projections"][oi, ei, column, si] = as_pairs(
                        project_fields(fields)
                    )
                    signed.append(fields)
                minus, plus = signed
                amplitude = epsilon * context.scales[column]
                odd = (plus - minus) / (2 * amplitude)
                even = plus + minus - 2 * base
                response = (
                    from_pairs(arrays["main_projections"][oi, ei, column, 1])
                    - from_pairs(arrays["main_projections"][oi, ei, column, 0])
                ) / (2 * amplitude)
                measured = response / fhat
                B = np.maximum(
                    1,
                    np.maximum.reduce(
                        [
                            np.max(np.abs(item), axis=(1, 2, 3))
                            for item in (minus, plus, base)
                        ]
                    ),
                )
                nu_field = 4096 * UNIT_ROUNDOFF * B / amplitude
                nu_mode = nu_field[:, None] / np.abs(fhat)
                rr, cc, rn, cn = recover(measured, nu_mode, k)
                arrays["rhs_coefficients"][oi, ei, :, :, column] = rr
                arrays["constraint_coefficients"][oi, ei, :, :, column] = cc
                arrays["rhs_resolution"][oi, ei, :, :, column] = rn
                arrays["constraint_resolution"][oi, ei, :, :, column] = cn
                error_mode = measured - predicted[:, column]
                mode_limit = 64 * epsilon**2 * mode_weights + 8 * nu_mode
                arrays["modal_errors"][oi, ei, column] = as_pairs(error_mode)
                arrays["modal_limits"][oi, ei, column] = mode_limit
                modal_pass = modal_pass and bool(
                    np.all(np.abs(error_mode) <= mode_limit)
                )
                field_error = odd - reconstruct(predicted[:, column] * fhat)
                offmode = odd - reconstruct(project_fields(odd))
                limit = 64 * epsilon**2 * field_weights + 8 * nu_field
                flat = np.abs(field_error).reshape(101, -1)
                max_error, off_max = (
                    np.max(flat, axis=1),
                    np.max(np.abs(offmode), axis=(1, 2, 3)),
                )
                row_pass = (max_error <= limit) & (off_max <= 14 * limit)
                arrays["field_stats"][oi, ei, column] = np.stack(
                    (
                        max_error,
                        np.sqrt(np.mean(field_error**2, axis=(1, 2, 3))),
                        np.argmax(flat, axis=1),
                        off_max,
                        np.sqrt(np.mean(offmode**2, axis=(1, 2, 3))),
                        np.max(np.abs(even), axis=(1, 2, 3)),
                        np.sqrt(np.mean(even**2, axis=(1, 2, 3))),
                        limit,
                        nu_field,
                        B,
                        row_pass.astype(float),
                    ),
                    axis=-1,
                )
                field_pass = field_pass and bool(np.all(row_pass))
            guard()
            json_sink(
                f"stage_main_{oi}_{ei}",
                dict(
                    completed_evaluations=len(calls),
                    order=order,
                    epsilon=epsilon,
                    complete_columns=58,
                ),
            )
    arrays["rhs_errors"] = arrays["rhs_coefficients"] - reference["rhs"]
    arrays["constraint_errors"] = (
        arrays["constraint_coefficients"] - reference["constraints"]
    )
    for name in (
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
    ):
        save(name)
    negative_records = []
    for ni, (variant, column, targets) in enumerate(NEGATIVES):
        for oi, order in enumerate(ORDERS):
            reference_symbols = symbols(reference, arrays["wave_numbers"][oi])[
                :, column
            ]
            for ei, epsilon in enumerate(EPSILONS):
                signed = []
                for si, sign in enumerate(SIGNS):
                    q = qbase.copy()
                    q[column] += sign * epsilon * context.scales[column] * context.f
                    fields = evaluate(
                        q,
                        order,
                        variant,
                        f"negative/{variant}/fd{order}/eps{ei}/sign{sign}",
                    )
                    arrays["negative_projections"][ni, oi, ei, si] = as_pairs(
                        project_fields(fields)
                    )
                    signed.append(fields)
                amplitude = epsilon * context.scales[column]
                measured = (
                    from_pairs(arrays["negative_projections"][ni, oi, ei, 1])
                    - from_pairs(arrays["negative_projections"][ni, oi, ei, 0])
                ) / (2 * amplitude * fhat)
                baseline = (
                    from_pairs(arrays["main_projections"][oi, ei, column, 1])
                    - from_pairs(arrays["main_projections"][oi, ei, column, 0])
                ) / (2 * amplitude * fhat)
                B = np.maximum(
                    1,
                    np.maximum.reduce(
                        [
                            np.max(np.abs(item), axis=(1, 2, 3))
                            for item in (*signed, arrays["background_fields"][oi])
                        ]
                    ),
                )
                resolution = (
                    4096
                    * UNIT_ROUNDOFF
                    * (B + arrays["field_stats"][oi, ei, column, :, 9])[:, None]
                    / (amplitude * np.abs(fhat))
                )
                arrays["negative_field_scales"][ni, oi, ei] = B
                change, error = measured - baseline, measured - reference_symbols
                limit = arrays["modal_limits"][oi, ei, column]
                arrays["negative_changes"][ni, oi, ei] = as_pairs(change)
                arrays["negative_errors"][ni, oi, ei] = as_pairs(error)
                arrays["negative_resolution"][ni, oi, ei] = resolution
                arrays["negative_limits"][ni, oi, ei] = limit
                selected = np.array(targets)
                detected = (np.abs(change[selected]) > 0) & (
                    np.abs(error[selected]) > limit[selected]
                )
                if ei == 1:
                    detected &= np.abs(change[selected]) > 64 * resolution[selected]
                negative_records.append(
                    dict(
                        variant=variant,
                        order=order,
                        epsilon_index=ei,
                        target_rows=list(targets),
                        column=column,
                        detected_slots=int(np.count_nonzero(detected)),
                        accepted=bool(np.any(detected)),
                    )
                )
    for name in (
        "negative_projections",
        "negative_changes",
        "negative_errors",
        "negative_resolution",
        "negative_limits",
        "negative_field_scales",
        "roundtrip_stats",
    ):
        save(name)
    background_max = np.max(np.abs(arrays["background_fields"]), axis=(2, 3, 4))
    background_pass = bool(
        np.all(background_max <= 4096 * UNIT_ROUNDOFF * np.maximum(1, background_max))
    )
    rhs_acceptance = matrix_acceptance(
        arrays["rhs_coefficients"], reference["rhs"], arrays["rhs_resolution"]
    )
    constraint_acceptance = matrix_acceptance(
        arrays["constraint_coefficients"],
        reference["constraints"],
        arrays["constraint_resolution"],
    )
    acceptance = dict(
        scalar_controls=controls["accepted"],
        complete_calls=len(calls) == EVALUATIONS and all(c["completed"] for c in calls),
        background=background_pass,
        roundtrip=bool(np.all(arrays["roundtrip_stats"][..., 2] <= 1)),
        rhs=rhs_acceptance["accepted"],
        physical43=constraint_acceptance["accepted"],
        all_modes=modal_pass,
        full_grid_and_offmode=field_pass,
        real_negative_controls=all(r["accepted"] for r in negative_records),
    )
    acceptance["accepted"] = all(acceptance.values())
    require(sent == set(ARRAY_SPECS), "all planned arrays saved")
    result = dict(
        schema="case12-binary64-production-v1",
        acceptance=acceptance,
        rhs=rhs_acceptance,
        physical43=constraint_acceptance,
        negative_controls=negative_records,
        state_evaluations=len(calls),
        rhs_calls=len(calls),
        physical43_calls=len(calls),
        driver49_executed=False,
        time_evolution_executed=False,
        empirical_roundoff_indicator_not_rigorous_bound=True,
        output_array_shapes={n: list(s) for n, s in ARRAY_SPECS.items()},
    )
    json_sink(
        "calls",
        dict(schema="case12-binary64-calls-v1", planned=EVALUATIONS, records=calls),
    )
    json_sink("acceptance", result)
    return result


def run(
    reference,
    *,
    guard=lambda: None,
    check=lambda: None,
    array_sink=lambda name, array: None,
    json_sink=lambda name, record: None,
):
    """Run only after the owner resource gate; preserve partial call receipts."""
    calls = []
    try:
        return _run(
            reference,
            calls=calls,
            guard=guard,
            check=check,
            array_sink=array_sink,
            json_sink=json_sink,
        )
    except BaseException:
        try:
            json_sink(
                "partial_calls",
                dict(
                    schema="case12-binary64-calls-v1",
                    planned=EVALUATIONS,
                    records=calls,
                    completed=sum(c["completed"] for c in calls),
                    accepted=False,
                ),
            )
        except Exception:
            # The owner still writes its reserved terminal receipt if the
            # deadline/output guard no longer permits an ordinary artifact.
            pass
        raise

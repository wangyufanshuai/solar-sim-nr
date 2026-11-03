"""Strict reading of already-qualified case12 reference coefficients.

Importing this module uses only the standard library. ``prepare_reference``
is a worker operation: it validates nine JSON inputs (eight matrices and an
execution record), then imports NumPy to construct binary64 reference arrays.
It does not run an exact-coefficient model or authenticate input provenance;
the caller must first verify the separate experiment manifest.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
import hashlib
import json
import math
import re


RATIONAL_SCHEMA = "nr-case12-complete-rational-matrix-v1"
POLYNOMIAL_SCHEMA = "nr-case12-polynomial-matrix-v1"
RHS_NAMES = ("J", "Ax", "Ay", "Az")
CONSTRAINT_EXPONENTS = (
    (0, 0, 0),
    (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (2, 0, 0), (1, 1, 0), (1, 0, 1),
    (0, 2, 0), (0, 1, 1), (0, 0, 2),
)
CONSTRAINT_NAMES = ("0", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz")
REQUIRED_ALIASES = (
    "volume/baseline__F0.json",
    *("volume/baseline__" + name + ".json" for name in RHS_NAMES),
    "volume/baseline__execution.json",
    *("constraint/C" + str(degree) + ".json" for degree in range(3)),
)
STATE_ORDER = (
    "h00", "h01", "h02", "h11", "h12", "h22", "ln_alpha",
    "beta0", "beta1", "beta2", "ln_phi", "A00", "A01", "A02",
    "A11", "A12", "A22", "K", "Theta", "GammaHat0", "GammaHat1",
    "GammaHat2", "b0", "b1", "b2", "A0", "A1", "A2", "B00",
    "B01", "B02", "B10", "B11", "B12", "B20", "B21", "B22",
    "D000", "D001", "D002", "D011", "D012", "D022", "D100",
    "D101", "D102", "D111", "D112", "D122", "D200", "D201",
    "D202", "D211", "D212", "D222", "P0", "P1", "P2",
)
EXPECTED_PARAMETERS = {
    "schema": "nr-case12-actual-volume-parameters-v1",
    "background": {
        "alpha": "17/20", "chi": "18/25",
        "metric": [
            ["1", "3/25", "-2/25"],
            ["3/25", "634/625", "63/1250"],
            ["-2/25", "63/1250", "101/100"],
        ],
        "shift": ["3/100", "-1/25", "1/20"],
    },
    "config": {
        "c": "1", "cleaning_speed": "1", "k0": "0", "kappa1": "1/10",
        "kappa2": "0", "kappa3": "1", "ordering_mu": "1",
        "shift_strength": "1", "tau_inverse": "1",
    },
    "gauge": {
        "advective": True, "lapse_coefficient": "2",
        "lapse_driver": "one-plus-log", "shift_coefficient": "1",
        "shift_damping": "0", "shift_driver": "gamma-driver",
        "theta_coupling": "2",
    },
    "profile": {"ordering_scale": "1/5", "shift_scale": "3"},
    "independent_state_seeds": 58, "independent_gradient_seeds": 174,
    "state_order": list(STATE_ORDER),
}
_INTEGER = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)\Z")
_RATIONAL = re.compile(r"(?:0|-[1-9][0-9]*|[1-9][0-9]*)(?:/[1-9][0-9]*)?\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(token):
    raise ValueError("nonfinite JSON token: " + token)


def _floating(token):
    raise ValueError("JSON floats are not exact-reference coefficients: " + token)


def decode_json(raw):
    """Decode UTF-8 bytes without duplicate keys or nonfinite constants."""
    require(type(raw) is bytes, "original JSON bytes required")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object,
                       parse_constant=_nonfinite, parse_float=_floating)
    require(type(value) is dict, "JSON object required")
    return value


_OLD_PARENTS = "examples/data/constraint_bridge/parents/"
_NEW_INPUTS = "experiments/case12_binary64_v1/inputs/"
_PARENT_GROUPS = {
    "volume": {
        "prefix": _OLD_PARENTS + "volume_",
        "schema": "nr-case12-actual-volume-linearization-controls-v1",
        "replay_schema": "nr-case12-independent-actual-volume-linearization-v1",
        "flags": ("actual_volume_real_formula_linearization_validated",),
        "acceptance": (
            "accepted", "algebraic_jacobian_bridge", "baseline_equilibrium",
            "chart_chain_rule_controls", "real_parameter_controls",
            "spatial_jacobian_bridge",
        ),
    },
    "coordinate": {
        "prefix": _OLD_PARENTS + "coordinate_",
        "schema": "nr-case12-physical-coordinate-bridge-controls-v1",
        "replay_schema": "nr-case12-independent-physical-coordinate-bridge-v1",
        "flags": ("physical_coordinate_bridge_validated",
                  "zero_frequency_physical39_validated"),
        "acceptance": (
            "accepted", "fixed_constant_source", "inverse_relations",
            "leading_map", "mutation_controls", "positive_frequency_powers",
            "zero_frequency_image",
        ),
    },
    "constraint": {
        "prefix": _NEW_INPUTS + "constraint/",
        "schema": "nr-case12-actual-constraint-diagnostic-controls-v1",
        "replay_schema": "nr-case12-independent-actual-constraint-diagnostic-v1",
        "flags": ("actual_constraint_continuum_real_formula_bridge_validated",
                  "driver_metric_constraint_continuum_bridge_validated"),
        "acceptance": (
            "accepted", "baseline_equilibrium", "complete_physical43_bridge",
            "driver_complete40_bridge", "driver_equilibrium",
            "driver_nine_symmetry_relations", "real_source_negative_controls",
        ),
    },
}
_OLD_PARENT_NAMES = (
    "C0.json", "C1.json", "C2.json",
    "coordinate_qualification-gate.json", "coordinate_worker-result.json",
    "coordinate_independent-replay.json", "coordinate_source_bridge.json",
    "coordinate_parent_provenance.json", "volume_qualification-gate.json",
    "volume_worker-result.json", "volume_independent-replay.json",
    "volume_source_bridge.json", "volume_domain.json",
)
_VOLUME_NAMES = (
    "baseline__F0.json", "baseline__J.json", "baseline__Ax.json",
    "baseline__Ay.json", "baseline__Az.json", "baseline__execution.json",
)
_CONSTRAINT_NAMES = (
    "qualification-gate.json", "worker-result.json", "independent-replay.json",
    "source_bridge.json", "domain.json",
)
PARENT_PATHS = (
    *(_OLD_PARENTS + name for name in _OLD_PARENT_NAMES),
    *(_NEW_INPUTS + "volume/" + name for name in _VOLUME_NAMES),
    *(_NEW_INPUTS + "constraint/" + name for name in _CONSTRAINT_NAMES),
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _finite_metadata_float(token):
    value = float(token)
    require(math.isfinite(value), "nonfinite parent metadata number")
    return value


def _metadata(raw):
    """Receipts contain finite timing floats; coefficient JSON does not."""
    require(type(raw) is bytes, "original parent receipt bytes required")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_object,
                       parse_constant=_nonfinite, parse_float=_finite_metadata_float)
    require(type(value) is dict, "parent receipt object required")
    return value


def _typed_equal(left, right):
    # JSON encoding distinguishes bool/int/float aliases accepted by Python ==.
    options = dict(sort_keys=True, separators=(",", ":"), allow_nan=False)
    return json.dumps(left, **options) == json.dumps(right, **options)


def verify_parent_metadata(raw_by_path):
    """Check frozen receipt states and byte crosslinks, without mathematics.

    Keys are repository-relative POSIX paths. The caller must first authenticate
    all bytes against the separate experiment manifest and unchanged export
    manifest; this function checks the meaning and internal links of those bytes.
    It neither re-runs the parents nor authenticates an arbitrary replacement
    manifest. Historical ``terminal_sha256`` is not a worker-result hash.
    """
    require(isinstance(raw_by_path, Mapping), "parent byte mapping required")
    raw = {}
    for path in PARENT_PATHS:
        require(path in raw_by_path and type(raw_by_path[path]) is bytes,
                "missing original parent bytes: " + path)
        raw[path] = raw_by_path[path]
    hashes = {path: hashlib.sha256(data).hexdigest() for path, data in raw.items()}
    groups, identities = {}, {}
    crosslinks = 0

    def linked(table, name, path):
        nonlocal crosslinks
        require(type(table) is dict and type(table.get(name)) is str
                and _SHA256.fullmatch(table[name]) is not None
                and table[name] == hashes[path], "parent byte crosslink: " + name)
        crosslinks += 1

    def accepted(record, expected):
        require(type(record) is dict and set(record) == set(expected)
                and all(record[name] is True for name in expected),
                "parent acceptance fields")

    for name, spec in _PARENT_GROUPS.items():
        paths = {role: spec["prefix"] + role + ".json"
                 for role in ("qualification-gate", "worker-result", "independent-replay")}
        gate, worker, replay = (_metadata(raw[paths[role]]) for role in paths)
        for receipt in (gate, worker):
            require(receipt.get("schema") == spec["schema"], "parent control schema")
            require(all(receipt.get(field) is True for field in
                        ("passed", "completed", "computation_completed",
                         "old_artifacts_unchanged", *spec["flags"])),
                    "parent control not completed/qualified")
            require(receipt.get("phase") == "complete"
                    and type(receipt.get("errors")) is list and receipt["errors"] == [],
                    "parent terminal errors/phase")
            for field in ("physical_gate", "scientificPayloadWriteback",
                          "full_python_semantics_proved", "new_scientific_evolution"):
                require(receipt.get(field) is False, "parent scope flag changed: " + field)
            if name != "coordinate":
                require(receipt.get("actual_binary64_rhs_executed") is False,
                        "parent is not a binary64 qualification")
                accepted(receipt.get("acceptance"), spec["acceptance"])
            require(_typed_equal(receipt.get("replay"), replay),
                    "embedded parent replay differs")
        require(replay.get("schema") == spec["replay_schema"]
                and replay.get("passed") is True, "parent replay not qualified")
        accepted(replay.get("acceptance"), spec["acceptance"])
        if name != "coordinate":
            require(replay.get("computation_completed") is True
                    and replay.get("actual_binary64_rhs_executed") is False,
                    "parent replay scope/completion")
        parent_digest = gate.get("parent_gate_sha256")
        require(type(parent_digest) is str and _SHA256.fullmatch(parent_digest)
                and worker.get("parent_gate_sha256") == parent_digest
                and replay.get("parent_gate_sha256") == parent_digest,
                "parent upstream identity differs")
        linked(gate.get("artifact_hashes"), "worker-result.json", paths["worker-result"])
        linked(gate.get("artifact_hashes"), "independent-replay.json", paths["independent-replay"])
        groups[name] = (gate, worker, replay)
        identities[name] = {
            "gate_sha256": hashes[paths["qualification-gate"]],
            "worker_sha256": hashes[paths["worker-result"]],
            "replay_sha256": hashes[paths["independent-replay"]],
            "upstream_gate_sha256": parent_digest,
            "passed": True,
        }

    def artifact_and_replay(group, name, path):
        gate, _, replay = groups[group]
        linked(gate.get("artifact_hashes"), name, path)
        linked(replay.get("input_hashes"), name, path)

    for name in _VOLUME_NAMES:
        artifact_and_replay("volume", name, _NEW_INPUTS + "volume/" + name)
    for name in ("source_bridge.json", "domain.json"):
        artifact_and_replay("volume", name, _OLD_PARENTS + "volume_" + name)
        artifact_and_replay("constraint", name, _NEW_INPUTS + "constraint/" + name)
    for name in ("C0.json", "C1.json", "C2.json"):
        artifact_and_replay("coordinate", name, _OLD_PARENTS + name)
    for name in ("source_bridge.json", "parent_provenance.json"):
        artifact_and_replay("coordinate", name, _OLD_PARENTS + "coordinate_" + name)
    for name in _OLD_PARENT_NAMES:
        artifact_and_replay("constraint", "parent-inputs/" + name, _OLD_PARENTS + name)

    volume_domain = _metadata(raw[_OLD_PARENTS + "volume_domain.json"])
    constraint_domain = _metadata(raw[_NEW_INPUTS + "constraint/domain.json"])
    require(_typed_equal(groups["volume"][2].get("domain"), volume_domain)
            and _typed_equal(groups["constraint"][2].get("domain"), constraint_domain),
            "parent domain differs from replay")
    require(identities["constraint"]["upstream_gate_sha256"]
            == identities["volume"]["gate_sha256"], "constraint volume gate mismatch")
    require(constraint_domain.get("parent_gate_sha256")
            == identities["volume"]["gate_sha256"]
            and constraint_domain.get("coordinate_gate_sha256")
            == identities["coordinate"]["gate_sha256"], "constraint domain parent mismatch")
    require(volume_domain.get("schema") == "nr-case12-actual-volume-linearization-domain-v1"
            and constraint_domain.get("schema") == "nr-case12-actual-constraint-domain-v1",
            "parent domain schema")
    for domain in (volume_domain, constraint_domain):
        require(domain.get("actual_binary64_rhs_executed") is False
                and domain.get("full_python_semantics_proved") is False,
                "parent real-formula scope changed")
    require(constraint_domain.get("actual_discrete_constraint_diagnostic_executed") is False
            and constraint_domain.get("binary64_or_discrete_diagnostics_qualified") is False,
            "parent diagnostic scope changed")
    return {
        "schema": "case12-binary64-parent-metadata-verification-v1",
        "passed": True, "metadata_only": True, "parent_files": len(raw),
        "new_input_files": 11, "checked_byte_crosslinks": crosslinks,
        "identities": identities, "input_sha256": hashes,
        "parent_mathematics_rerun": False,
    }


def rational(value):
    """Accept only the canonical integer or reduced rational string."""
    require(type(value) is str and _RATIONAL.fullmatch(value) is not None,
            "canonical rational string required")
    result = Fraction(value)
    require(str(result) == value, "reduced canonical rational required")
    return result


def _coefficient(numerator, denominator):
    require(type(numerator) is str and _INTEGER.fullmatch(numerator) is not None,
            "canonical polynomial numerator")
    require(type(denominator) is str and re.fullmatch(r"[1-9][0-9]*", denominator) is not None,
            "positive canonical polynomial denominator")
    n, d = int(numerator), int(denominator)
    value = Fraction(n, d)
    require(n != 0 and value.numerator == n and value.denominator == d,
            "nonzero reduced polynomial coefficient")
    return value


def _matrix_header(record, shape, schema, keys):
    require(set(record) == set(keys) and record["schema"] == schema,
            "exact matrix schema and fields required")
    observed = record["shape"]
    require(type(observed) is list and len(observed) == 2
            and all(type(size) is int for size in observed)
            and tuple(observed) == shape, "complete registered matrix shape")
    entries = record["entries"]
    require(type(entries) is list and len(entries) == shape[0] * shape[1],
            "one explicit entry per matrix slot")
    return entries


def _entry(entry, offset, columns):
    require(type(entry) is list and len(entry) == 3,
            "three-field matrix entry required")
    row, column, value = entry
    require(type(row) is int and type(column) is int
            and (row, column) == divmod(offset, columns),
            "unique complete row-major coordinates")
    return value


def decode_rational_matrix(raw, shape, *, check=lambda: None):
    """Decode every slot; missing, reordered or duplicate slots are errors."""
    check()
    require(type(shape) is tuple and len(shape) == 2
            and all(type(n) is int and n > 0 for n in shape), "positive shape")
    record = decode_json(raw)
    entries = _matrix_header(record, shape, RATIONAL_SCHEMA,
                             ("schema", "shape", "entries"))
    values = []
    for offset, entry in enumerate(entries):
        if offset % 256 == 0:
            check()
        values.append(rational(_entry(entry, offset, shape[1])))
    check()
    return tuple(tuple(values[i * shape[1]:(i + 1) * shape[1]])
                 for i in range(shape[0]))


def decode_polynomial_matrix(raw, degree, *, check=lambda: None):
    """Read homogeneous C0/C1/C2; mixed monomial coefficients are unchanged."""
    check()
    require(type(degree) is int and degree in (0, 1, 2), "C0/C1/C2 degree")
    record = decode_json(raw)
    entries = _matrix_header(record, (43, 58), POLYNOMIAL_SCHEMA,
                             ("schema", "shape", "variables", "entries"))
    require(type(record["variables"]) is list
            and record["variables"] == ["x", "y", "z"], "ordered x/y/z variables")
    output = []
    for offset, entry in enumerate(entries):
        if offset % 256 == 0:
            check()
        terms = _entry(entry, offset, 58)
        require(type(terms) is list, "explicit polynomial terms list")
        previous, decoded = None, []
        for term in terms:
            require(type(term) is list and len(term) == 5, "five-field polynomial term")
            exponent = tuple(term[:3])
            require(all(type(e) is int and e >= 0 for e in exponent)
                    and sum(exponent) == degree, "registered homogeneous monomial")
            require(previous is None or previous < exponent,
                    "strictly sorted unique monomials")
            decoded.append((exponent, _coefficient(term[3], term[4])))
            previous = exponent
        output.append(tuple(decoded))
    check()
    return tuple(tuple(output[i * 58:(i + 1) * 58]) for i in range(43))


def _typed_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def decode_parameters(raw):
    record = decode_json(raw)
    require(set(record) == {"chart", "operations", "parameters", "requested_parameters"},
            "registered execution fields")
    parameters = record["parameters"]
    require(type(parameters) is dict
            and _typed_json(parameters) == _typed_json(EXPECTED_PARAMETERS),
            "unchanged fixed-case12 parameters and state order")
    require(_typed_json(record["requested_parameters"]) ==
            _typed_json({"kappa1": "1/10", "ordering_scale": "1/5"}),
            "baseline requested source/profile parameters")
    return parameters


def _to_float(value):
    require(type(value) is Fraction, "explicit exact reference coefficient")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("reference coefficient is not finite binary64") from error
    require(math.isfinite(converted) and (value == 0 or converted != 0.0),
            "reference coefficient must remain finite and nonzero when nonzero")
    return converted, str(Fraction.from_float(converted) - value)


def _parameter_conversions(parameters):
    entries = []

    def append(path, value):
        number, error = _to_float(rational(value))
        entries.append([path, value, number.hex(), error])

    background = parameters["background"]
    for name in ("alpha", "chi"):
        append("background." + name, background[name])
    for i, row in enumerate(background["metric"]):
        for j, value in enumerate(row):
            append(f"background.metric.{i}.{j}", value)
    for i, value in enumerate(background["shift"]):
        append(f"background.shift.{i}", value)
    for group in ("config", "gauge", "profile"):
        for name, value in sorted(parameters[group].items()):
            if group == "gauge" and name in ("advective", "lapse_driver", "shift_driver"):
                continue
            append(group + "." + name, value)
    return entries


def prepare_reference(raw_inputs, *, check=lambda: None):
    """Convert qualified input bytes, without computing a reference model.

    ``raw_inputs`` may be an immutable Mapping and may contain other manifest-
    verified metadata. Only REQUIRED_ALIASES are read here. NumPy is imported
    after the complete rational schema, degree, parameter and F0 checks.
    Conversion errors are exact ``binary64_value - rational_value`` strings,
    with one record for every output slot including zero coefficients.
    """
    check()
    require(isinstance(raw_inputs, Mapping), "raw input mapping required")
    require(all(name in raw_inputs and type(raw_inputs[name]) is bytes
                for name in REQUIRED_ALIASES), "all nine reference JSON byte inputs")
    parameters = decode_parameters(raw_inputs["volume/baseline__execution.json"])
    f0_exact = decode_rational_matrix(raw_inputs["volume/baseline__F0.json"],
                                     (58, 1), check=check)
    require(all(value == 0 for row in f0_exact for value in row),
            "qualified exact F0 must be identically zero")
    rhs_exact = tuple(decode_rational_matrix(
        raw_inputs["volume/baseline__" + name + ".json"], (58, 58), check=check)
        for name in RHS_NAMES)
    constraints_exact = tuple(decode_polynomial_matrix(
        raw_inputs["constraint/C" + str(degree) + ".json"], degree, check=check)
        for degree in range(3))
    check()

    import numpy as np

    rhs = np.zeros((4, 58, 58), dtype=np.float64)
    constraints = np.zeros((10, 43, 58), dtype=np.float64)
    rhs_errors, constraint_errors = [], []
    for block, matrix in enumerate(rhs_exact):
        for i, row in enumerate(matrix):
            check()
            for j, value in enumerate(row):
                converted, error = _to_float(value)
                rhs[block, i, j] = converted
                rhs_errors.append([block, i, j, error])
    exponent_index = {exponent: i for i, exponent in enumerate(CONSTRAINT_EXPONENTS)}
    coefficient_lookup = {}
    for matrix in constraints_exact:
        for i, row in enumerate(matrix):
            check()
            for j, terms in enumerate(row):
                for exponent, value in terms:
                    coefficient_lookup[exponent_index[exponent], i, j] = value
    for block in range(10):
        for i in range(43):
            check()
            for j in range(58):
                value = coefficient_lookup.get((block, i, j), Fraction(0))
                converted, error = _to_float(value)
                constraints[block, i, j] = converted
                constraint_errors.append([block, i, j, error])
    f0 = np.zeros((58, 1), dtype=np.float64)
    conversion_errors = {
        "schema": "solar-sim-nr-binary64-reference-conversion-v1",
        "convention": "binary64_value-minus-exact-rational",
        "rhs": {"shape": [4, 58, 58], "entries": rhs_errors},
        "constraints": {"shape": [10, 43, 58], "entries": constraint_errors},
        "f0": {"shape": [58, 1], "entries": [[i, 0, "0"] for i in range(58)]},
        "parameters": _parameter_conversions(parameters),
    }
    for matrix in (rhs, constraints, f0):
        matrix.setflags(write=False)
    check()
    return dict(parameters=parameters, rhs=rhs, constraints=constraints, f0=f0,
                f0_exact_zero=True, conversion_errors=conversion_errors,
                rhs_order=list(RHS_NAMES), constraint_order=list(CONSTRAINT_NAMES),
                input_aliases=list(REQUIRED_ALIASES))

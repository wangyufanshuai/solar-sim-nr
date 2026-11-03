"""Small exact-input controls; no actual RHS or diagnostic evaluation."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from experiments.case12_binary64_v1 import provenance as P


def parent_receipts():
    root = Path(__file__).resolve().parents[1]
    return {name: (root / name).read_bytes() for name in P.PARENT_PATHS}


def change_receipt(raw, group, role, mutate):
    name = P._PARENT_GROUPS[group]["prefix"] + role + ".json"
    record = json.loads(raw[name])
    mutate(record)
    raw[name] = encoded(record)


def test_parent_metadata_matches_all_frozen_receipt_crosslinks():
    raw = parent_receipts()
    summary = P.verify_parent_metadata(MappingProxyType(raw))
    assert summary["passed"] is True and summary["metadata_only"] is True
    assert summary["parent_mathematics_rerun"] is False
    assert summary["parent_files"] == 24 and summary["new_input_files"] == 11
    assert summary["checked_byte_crosslinks"] == 62
    assert summary["input_sha256"] == {
        name: hashlib.sha256(data).hexdigest() for name, data in raw.items()
    }
    for group in P._PARENT_GROUPS:
        assert summary["identities"][group]["passed"] is True


@pytest.mark.parametrize("group,role,field,value", [
    ("volume", "qualification-gate", "passed", False),
    ("volume", "worker-result", "completed", 1),
    ("volume", "worker-result", "computation_completed", False),
    ("volume", "qualification-gate", "errors", ["failure"]),
    ("volume", "qualification-gate", "phase", "running"),
    ("volume", "qualification-gate", "actual_binary64_rhs_executed", True),
    ("constraint", "worker-result", "old_artifacts_unchanged", False),
    ("constraint", "qualification-gate", "actual_constraint_continuum_real_formula_bridge_validated", 1),
    ("constraint", "worker-result", "driver_metric_constraint_continuum_bridge_validated", False),
    ("coordinate", "worker-result", "physical_coordinate_bridge_validated", False),
    ("coordinate", "qualification-gate", "zero_frequency_physical39_validated", False),
    ("coordinate", "independent-replay", "schema", "wrong"),
])
def test_parent_status_and_scope_are_strict(group, role, field, value):
    raw = parent_receipts()
    change_receipt(raw, group, role, lambda record: record.__setitem__(field, value))
    with pytest.raises(ValueError):
        P.verify_parent_metadata(raw)


@pytest.mark.parametrize("group", ["volume", "constraint", "coordinate"])
def test_parent_replay_must_match_both_embedded_records(group):
    raw = parent_receipts()
    change_receipt(raw, group, "worker-result",
                   lambda record: record["replay"]["acceptance"].__setitem__("accepted", 1))
    with pytest.raises(ValueError, match="embedded parent replay"):
        P.verify_parent_metadata(raw)


@pytest.mark.parametrize("group,artifact", [
    ("volume", "worker-result.json"), ("volume", "baseline__Az.json"),
    ("constraint", "independent-replay.json"),
    ("constraint", "parent-inputs/C2.json"),
    ("coordinate", "C1.json"),
])
def test_parent_gate_artifact_crosslinks_are_required(group, artifact):
    raw = parent_receipts()
    change_receipt(raw, group, "qualification-gate",
                   lambda record: record["artifact_hashes"].__setitem__(artifact, "0" * 64))
    with pytest.raises(ValueError, match="crosslink"):
        P.verify_parent_metadata(raw)


@pytest.mark.parametrize("name", [
    P._NEW_INPUTS + "volume/baseline__F0.json",
    P._NEW_INPUTS + "volume/baseline__execution.json",
    P._NEW_INPUTS + "constraint/source_bridge.json",
    P._NEW_INPUTS + "constraint/domain.json",
    P._OLD_PARENTS + "C2.json",
])
def test_parent_payload_original_bytes_not_semantic_json(name):
    raw = parent_receipts()
    raw[name] += b" "
    with pytest.raises(ValueError, match="crosslink"):
        P.verify_parent_metadata(raw)


@pytest.mark.parametrize("replacement", [None, "text", bytearray(b"{}")])
def test_parent_missing_or_nonbyte_receipts_rejected(replacement):
    raw = parent_receipts()
    name = P.PARENT_PATHS[0]
    if replacement is None:
        del raw[name]
    else:
        raw[name] = replacement
    with pytest.raises(ValueError, match="original parent bytes"):
        P.verify_parent_metadata(raw)


@pytest.mark.parametrize("raw", [
    b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}',
    b'{"a":1e999}', b'[]', b'null',
])
def test_parent_metadata_rejects_ambiguous_or_nonfinite_json(raw):
    with pytest.raises(ValueError):
        P._metadata(raw)


def test_parent_finite_timing_float_does_not_relax_exact_coefficient_reader():
    assert P._metadata(b'{"seconds":1.25}')["seconds"] == 1.25
    with pytest.raises(ValueError, match="floats"):
        P.decode_json(b'{"seconds":1.25}')


def encoded(record):
    return (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()


def rational_record(shape=(2, 2)):
    return dict(schema=P.RATIONAL_SCHEMA, shape=list(shape),
                entries=[[i, j, "0"] for i in range(shape[0]) for j in range(shape[1])])


def polynomial_record():
    return dict(schema=P.POLYNOMIAL_SCHEMA, shape=[43, 58], variables=["x", "y", "z"],
                entries=[[i, j, []] for i in range(43) for j in range(58)])


def execution_record():
    return dict(parameters=deepcopy(P.EXPECTED_PARAMETERS), chart={}, operations={},
                requested_parameters={"kappa1": "1/10", "ordering_scale": "1/5"})


def raw_fixture():
    raw = {"volume/baseline__execution.json": encoded(execution_record()),
           "volume/baseline__F0.json": encoded(rational_record((58, 1)))}
    for name in ("J", "Ax", "Ay", "Az"):
        raw["volume/baseline__" + name + ".json"] = encoded(rational_record((58, 58)))
    for degree in range(3):
        raw["constraint/C" + str(degree) + ".json"] = encoded(polynomial_record())
    return raw


@pytest.mark.parametrize("text,expected", [
    ("0", Fraction(0)), ("-7", Fraction(-7)), ("13/17", Fraction(13, 17)),
])
def test_canonical_rationals(text, expected):
    assert P.rational(text) == expected


@pytest.mark.parametrize("value", [
    0, True, 0.5, "1/1", "2/4", "-0", "+1", "01", "0/2", "1/-2",
    "1.0", " 1", "1/0", "NaN", "Infinity", "1e-2",
])
def test_reject_noncanonical_rationals(value):
    with pytest.raises(ValueError):
        P.rational(value)


@pytest.mark.parametrize("raw", [
    b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}',
    b'{"a":1.0}', b'{"a":1e999}', b'[]', b'null',
])
def test_exact_json_rejects_aliases_and_duplicates(raw):
    with pytest.raises(ValueError):
        P.decode_json(raw)


def test_complete_rational_matrix_reads_last_slot():
    record = rational_record()
    record["entries"][-1][2] = "-5/7"
    assert P.decode_rational_matrix(encoded(record), (2, 2)) == (
        (Fraction(0), Fraction(0)), (Fraction(0), Fraction(-5, 7)))


@pytest.mark.parametrize("change", [
    "missing", "duplicate", "reorder", "bool-coordinate", "float-coordinate",
    "shape", "bool-shape", "extra-field", "schema", "noncanonical",
])
def test_rational_matrix_rejects_incomplete_or_wrong_slots(change):
    record = rational_record()
    if change == "missing":
        record["entries"].pop()
    elif change == "duplicate":
        record["entries"][-1] = record["entries"][0]
    elif change == "reorder":
        record["entries"].reverse()
    elif change == "bool-coordinate":
        record["entries"][0][0] = False
    elif change == "float-coordinate":
        record["entries"][0][0] = 0.0
    elif change == "shape":
        record["shape"] = [1, 4]
    elif change == "bool-shape":
        record["shape"][0] = True
    elif change == "extra-field":
        record["ignored"] = 0
    elif change == "schema":
        record["schema"] += "-other"
    else:
        record["entries"][-1][2] = "2/4"
    with pytest.raises(ValueError):
        P.decode_rational_matrix(encoded(record), (2, 2))


def test_polynomial_preserves_mixed_coefficient_without_factor_two():
    record = polynomial_record()
    record["entries"][-1][2] = [[0, 0, 2, "-5", "1"], [1, 1, 0, "7", "3"]]
    result = P.decode_polynomial_matrix(encoded(record), 2)
    assert result[42][57] == (((0, 0, 2), Fraction(-5)), ((1, 1, 0), Fraction(7, 3)))


@pytest.mark.parametrize("terms", [
    [[1, 1, 0, "1", "1"], [1, 1, 0, "2", "1"]],
    [[2, 0, 0, "1", "1"], [0, 0, 2, "1", "1"]],
    [[1, 0, 0, "1", "1"]], [[-1, 1, 2, "1", "1"]],
    [[True, 1, 0, "1", "1"]], [[1, 1, 0, "0", "1"]],
    [[1, 1, 0, "2", "4"]], [[1, 1, 0, "1", "-2"]],
    [[1, 1, 0, 1, "1"]], [[1, 1, 0, "1", "01"]],
    [[1, 1, 0, "1", "1", "extra"]],
])
def test_polynomial_rejects_noncanonical_or_wrong_degree_terms(terms):
    record = polynomial_record()
    record["entries"][0][2] = terms
    with pytest.raises(ValueError):
        P.decode_polynomial_matrix(encoded(record), 2)


@pytest.mark.parametrize("change", ["variables", "missing", "duplicate", "shape"])
def test_polynomial_rejects_layout_drift(change):
    record = polynomial_record()
    if change == "variables":
        record["variables"] = ["z", "y", "x"]
    elif change == "missing":
        record["entries"].pop()
    elif change == "duplicate":
        record["entries"][-1] = record["entries"][0]
    else:
        record["shape"] = [58, 43]
    with pytest.raises(ValueError):
        P.decode_polynomial_matrix(encoded(record), 1)


@pytest.mark.parametrize("change", [
    "state-order", "state-duplicate", "alpha", "bool-int", "float-int",
    "seed-count", "extra-parameter", "request", "missing-record-field",
])
def test_registered_parameters_require_exact_types_and_order(change):
    record = execution_record()
    p = record["parameters"]
    if change == "state-order":
        p["state_order"][0:2] = reversed(p["state_order"][0:2])
    elif change == "state-duplicate":
        p["state_order"][-1] = p["state_order"][0]
    elif change == "alpha":
        p["background"]["alpha"] = "1"
    elif change == "bool-int":
        p["gauge"]["advective"] = 1
    elif change == "float-int":
        p["independent_state_seeds"] = 58.0
    elif change == "seed-count":
        p["independent_gradient_seeds"] = 173
    elif change == "extra-parameter":
        p["extra"] = "0"
    elif change == "request":
        record["requested_parameters"]["kappa1"] = "1/5"
    else:
        del record["chart"]
    with pytest.raises(ValueError):
        P.decode_parameters(encoded(record))


def test_conversion_difference_is_exact_not_decimal_subtraction():
    value, error = P._to_float(Fraction(1, 3))
    assert value == 1 / 3
    assert Fraction(error) == Fraction(-1, 54043195528445952)
    assert P._to_float(Fraction(3, 8)) == (0.375, "0")


@pytest.mark.parametrize("value", [Fraction(10**400), Fraction(1, 10**400)])
def test_conversion_rejects_overflow_and_nonzero_underflow(value):
    with pytest.raises(ValueError):
        P._to_float(value)


def test_prepare_complete_reference_and_immutable_input_mapping():
    raw = raw_fixture()
    for block, name in enumerate(("J", "Ax", "Ay", "Az")):
        record = rational_record((58, 58))
        record["entries"][-1][2] = str(Fraction(block + 1, 3))
        raw["volume/baseline__" + name + ".json"] = encoded(record)
    expected = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    for degree in range(3):
        record = polynomial_record()
        record["entries"][-1][2] = sorted(
            [[*e, str(expected[i]), "1"] for i, e in enumerate(P.CONSTRAINT_EXPONENTS)
             if sum(e) == degree])
        raw["constraint/C" + str(degree) + ".json"] = encoded(record)
    raw["unrelated-manifest-metadata"] = b'{}'
    result = P.prepare_reference(MappingProxyType(raw))
    assert result["rhs"].shape == (4, 58, 58)
    assert result["constraints"].shape == (10, 43, 58)
    assert result["rhs"].dtype.name == "float64"
    assert result["constraints"][:, 42, 57].tolist() == list(expected)
    assert result["rhs"][:, 57, 57].tolist() == [float(Fraction(i, 3)) for i in (1, 2, 3, 4)]
    assert result["constraint_order"] == ["0", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz"]
    assert result["rhs_order"] == ["J", "Ax", "Ay", "Az"]
    assert result["f0_exact_zero"] is True and not result["f0"].any()
    assert result["parameters"] == execution_record()["parameters"]
    assert not result["rhs"].flags.writeable and not result["constraints"].flags.writeable
    errors = result["conversion_errors"]
    assert len(errors["rhs"]["entries"]) == 13456
    assert len(errors["constraints"]["entries"]) == 24940
    assert len(errors["f0"]["entries"]) == 58
    assert errors["rhs"]["entries"][3363] == [0, 57, 57, "-1/54043195528445952"]
    assert errors["constraints"]["entries"][-1] == [9, 42, 57, "0"]
    alpha = next(row for row in errors["parameters"] if row[0] == "background.alpha")
    assert alpha[1] == "17/20"
    assert Fraction(alpha[3]) == Fraction.from_float(float.fromhex(alpha[2])) - Fraction(17, 20)


def test_prepare_rejects_nonzero_f0_before_numpy_import(monkeypatch):
    import builtins

    raw = raw_fixture()
    record = rational_record((58, 1))
    record["entries"][-1][2] = "1/1000"
    raw["volume/baseline__F0.json"] = encoded(record)
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "numpy":
            raise AssertionError("NumPy must not load before exact F0 validation")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="F0"):
        P.prepare_reference(raw)


@pytest.mark.parametrize("kind", ["missing", "not-bytes", "not-mapping"])
def test_prepare_requires_all_original_byte_inputs(kind):
    raw = raw_fixture()
    if kind == "missing":
        del raw["constraint/C2.json"]
    elif kind == "not-bytes":
        raw["constraint/C2.json"] = raw["constraint/C2.json"].decode()
    else:
        raw = list(raw.items())
    with pytest.raises(ValueError):
        P.prepare_reference(raw)


def test_deadline_callback_propagates_during_complete_slot_read():
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("registered deadline")

    with pytest.raises(TimeoutError, match="registered deadline"):
        P.decode_rational_matrix(encoded(rational_record((58, 58))), (58, 58), check=check)
    assert calls == 3

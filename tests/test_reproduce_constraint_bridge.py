"""Light controls for the portable wrapper; no complete scientific AD run."""

from fractions import Fraction as F
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / "examples/reproduce_constraint_bridge.py"
SPEC = importlib.util.spec_from_file_location("portable_constraint_example", PATH)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def matrix(rows=43, cols=580):
    return [[F(0) for _ in range(cols)] for _ in range(rows)]


def scalar(value=F(0), coefficients=None):
    return SimpleNamespace(
        value=value, coefficients={} if coefficients is None else coefficients
    )


@pytest.mark.parametrize("value", ["0", "-1", "601", "nan", "inf", "-inf", "bad"])
def test_timeout_rejects_unbounded_or_invalid(value):
    with pytest.raises(P.argparse.ArgumentTypeError):
        P.timeout_value(value)


def test_cli_modes_and_timeout():
    assert P.arguments(["--verify-inputs-only"]).verify_inputs_only is True
    assert P.arguments(["--output", "new", "--timeout", "5.5"]).timeout == 5.5
    assert P.arguments(["--output", "new"]).timeout == 600
    for args in (
        [],
        ["--verify-inputs-only", "--output", "new"],
        ["--verify-inputs-only", "--worker"],
    ):
        with pytest.raises(SystemExit) as error:
            P.arguments(args)
        assert error.value.code == 2


@pytest.mark.parametrize(
    "name", ["../secret", "/absolute", "C:/absolute", "a\\b", "a/../../b"]
)
def test_input_path_stays_relative(tmp_path, name):
    with pytest.raises(ValueError):
        P.relative_file(tmp_path, name)


def test_json_equality_does_not_equate_boolean_and_integer():
    assert P.same({"n": 1}, {"n": 1})
    assert not P.same({"n": True}, {"n": 1})
    with pytest.raises(ValueError):
        P.encoded({"n": float("nan")})


def test_rows_keep_last_seed_and_exact_zero_slots():
    values, coefficients = P.independent_rows(
        [scalar(F(2, 3), {0: F(-3), 579: F(7, 5)})], 1
    )
    assert values == [[F(2, 3)]]
    assert len(coefficients[0]) == 580
    assert coefficients[0][0] == -3
    assert coefficients[0][579] == F(7, 5)
    assert all(v == 0 for v in coefficients[0][1:579])


@pytest.mark.parametrize(
    "value,terms",
    [
        (1.0, {}),
        (True, {}),
        (F(0), {False: F(1)}),
        (F(0), {-1: F(1)}),
        (F(0), {580: F(1)}),
        (F(0), {0: 1.0}),
        (F(0), {0: F(0)}),
    ],
)
def test_rows_reject_coercions_and_noncanonical_terms(value, terms):
    with pytest.raises(ValueError):
        P.independent_rows([scalar(value, terms)], 1)


def test_independent_oracle_mixed_hessian_is_not_divided_by_two():
    decoded = [[[{} for _ in range(58)] for _ in range(43)] for _ in range(3)]
    decoded[0][0][1] = {(0, 0, 0): F(3)}
    decoded[1][2][3] = {(0, 0, 1): F(-4, 7)}
    decoded[2][42][57] = {(1, 1, 0): F(5, 9), (0, 0, 2): F(8)}
    calls = []

    def decode(record, *, shape, max_degree):
        assert shape == (43, 58)
        assert record == max_degree
        calls.append(max_degree)
        return decoded[max_degree]

    raw = {f"C{i}.json": str(i).encode() for i in range(3)}
    result = P.independent_oracle(raw, SimpleNamespace(decode_matrix=decode))
    assert calls == [0, 1, 2]
    assert result[0][1] == 3
    assert result[2][3 * 58 + 3] == F(-4, 7)
    assert result[42][5 * 58 + 57] == F(5, 9)
    assert result[42][579] == 8
    assert sum(value != 0 for row in result for value in row) == 4
    decoded[2][0][0] = {(1, 0, 0): F(1)}
    with pytest.raises(ValueError, match="homogeneous"):
        P.independent_oracle(raw, SimpleNamespace(decode_matrix=decode))


def test_driver_full_mapping_and_nine_redundancies():
    # Independent literal mapping: source raw49 rows to the corresponding
    # physical43 rows. Z rows 5,6,7 are absent from the driver's40 diagnostics.
    sources = (
        list(range(17))
        + [17, 18, 19, 21, 22, 25, 26, 27, 28, 30, 31, 34, 35, 36, 37, 39, 40, 43]
        + list(range(44, 49))
    )
    targets = list(range(5)) + list(range(8, 43))
    baseline = dict(F0=[[F(i + 1)] for i in range(43)], C=matrix())
    rows = [scalar() for _ in range(49)]
    for source, target in zip(sources, targets, strict=True):
        baseline["C"][target][579] = F(-target - 1)
        rows[source] = scalar(F(target + 1), {579: F(-target - 1)})
    for upper, lower in (
        (18, 20),
        (19, 23),
        (22, 24),
        (27, 29),
        (28, 32),
        (31, 33),
        (36, 38),
        (37, 41),
        (40, 42),
    ):
        rows[lower] = rows[upper]
    result = P.independent_driver(rows, baseline)
    assert len(result["driverAgreement"]) == 40
    assert len(result["driverSymmetry"]) == 9
    assert all(len(row) == 581 for row in result["driverSymmetry"])
    assert not any(v for row in result["driverAgreement"] for v in row)
    assert not any(v for row in result["driverSymmetry"] for v in row)
    rows[42] = scalar(rows[42].value + 1, dict(rows[42].coefficients))
    broken = P.independent_driver(rows, baseline)
    assert broken["driverSymmetry"][8][0] == -1
    assert not any(v for row in broken["driverAgreement"] for v in row)


def test_real_negative_control_requires_same_changed_residual_and_no_outside():
    cases = {case: dict(C=matrix(), R=matrix()) for case in P.CASES}
    cases[P.CASES[1]]["C"][1][579] = F(2)
    cases[P.CASES[1]]["R"][1][579] = F(2)
    cases[P.CASES[2]]["C"][5][0] = F(-3)
    cases[P.CASES[2]]["R"][5][0] = F(-3)
    assert P.independent_controls(cases)["accepted"] is True
    cases[P.CASES[1]]["R"][1][579] = F(0)
    cases[P.CASES[1]]["R"][2][1] = F(2)
    assert P.independent_controls(cases)["accepted"] is False
    cases[P.CASES[1]]["R"][1][579] = F(2)
    cases[P.CASES[2]]["C"][42][3] = F(1)
    assert P.independent_controls(cases)["accepted"] is False


def test_callbacks_interrupt_before_coefficient_extraction():
    def stop():
        raise TimeoutError("test")

    with pytest.raises(TimeoutError):
        P.independent_rows([scalar()], 1, check=stop)


def test_budget_preserves_evidence_and_reserves_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "MAX_OUTPUT_BYTES", 512)
    monkeypatch.setattr(P, "TERMINAL_RESERVE", 128)
    budget = P.Budget(tmp_path, 1, clock=lambda: 0)
    budget.record("sample", {"exact": "2/3"})
    before = (tmp_path / "sample.json").read_bytes()
    used, _ = P.output_census(tmp_path)
    budget.guard(512 - 128 - used)
    with pytest.raises(ValueError):
        budget.guard(513 - 128 - used)
    with pytest.raises(FileExistsError):
        budget.record("sample", {"exact": "3/4"})
    assert (tmp_path / "sample.json").read_bytes() == before
    P.terminal_write(tmp_path, "failed.json", {"passed": False})
    with pytest.raises(TimeoutError):
        P.Budget(tmp_path, 0, clock=lambda: 0).guard()


def test_output_directory_rejects_subdirectories(tmp_path):
    (tmp_path / "not-flat").mkdir()
    with pytest.raises(ValueError, match="regular flat"):
        P.output_census(tmp_path)


def test_fresh_output_only(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("owned by user", encoding="utf-8")
    with pytest.raises(FileExistsError):
        P.supervise(output, 10)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "owned by user"


def fake_supervisor_setup(monkeypatch, output, *, timeout=False, success=True):
    verification = {
        "passed": True,
        "test_input": "immutable",
        "portable_entry_sha256": P.ENTRY_SOURCE_SHA256,
    }
    monkeypatch.setattr(P, "verify_inputs", lambda **kw: (verification, {}))
    monkeypatch.setattr(
        P,
        "load_core",
        lambda: (SimpleNamespace(RECORD_ORDER=("sample",)), None, None, None),
    )
    monkeypatch.setattr(P.importlib.metadata, "version", lambda name: "test-version")

    class Process:
        killed = False
        code = None

        def wait(self, timeout=None):
            if fake_timeout and not self.killed:
                raise subprocess.TimeoutExpired("owned-test-worker", timeout)
            self.code = -9 if self.killed else (0 if success else 1)
            return self.code

        def poll(self):
            return self.code

        def kill(self):
            self.killed = True

    fake_timeout = timeout
    process = Process()

    def factory(command, **kwargs):
        assert command[0] == P.sys.executable
        assert command[1] == "-B"
        assert "--worker" in command
        assert kwargs["env"]["OPENBLAS_NUM_THREADS"] == "1"
        assert kwargs["stdout"] == subprocess.DEVNULL
        if not timeout:
            P.terminal_write(output, "sample.json", {"value": "1/7"})
            P.terminal_write(output, "input-verification.json", verification)
            P.terminal_write(
                output,
                "worker-result.json",
                dict(passed=success, computation_completed=True),
            )
            P.terminal_write(output, "acceptance.json", dict(accepted=success))
            # The fake forward contract must contain acceptance too.
            monkeypatch.setattr(
                P,
                "load_core",
                lambda: (
                    SimpleNamespace(RECORD_ORDER=("sample", "acceptance")),
                    None,
                    None,
                    None,
                ),
            )
            P.terminal_write(
                output,
                "independent-replay.json",
                dict(
                    passed=success,
                    computation_completed=True,
                    errors=[],
                    compared_output_hashes={
                        name: P.digest((output / name).read_bytes())
                        for name in ("sample.json", "acceptance.json")
                    },
                ),
            )
        return process

    return factory, process


@pytest.mark.parametrize("success", [True, False])
def test_supervisor_uses_actual_receipts_and_hashes(tmp_path, monkeypatch, success):
    output = tmp_path / "new"
    factory, process = fake_supervisor_setup(monkeypatch, output, success=success)
    assert P.supervise(output, 10, process_factory=factory) == (0 if success else 1)
    report = json.loads((output / "portable-result.json").read_bytes())
    assert report["passed"] is success
    assert report["owned_worker_stopped"] is True
    assert report["original_full_qualification_rerun"] is False
    assert report["independent_external_reproduction"] is False
    assert process.killed is False
    assert str(tmp_path) not in json.dumps(report)


def test_supervisor_kills_only_owned_worker_and_keeps_failed_receipt(
    tmp_path, monkeypatch
):
    output = tmp_path / "timeout"
    factory, process = fake_supervisor_setup(monkeypatch, output, timeout=True)
    assert P.supervise(output, 0.1, process_factory=factory) == 124
    report = json.loads((output / "portable-result.json").read_bytes())
    assert process.killed is True
    assert report["timed_out"] is True and report["passed"] is False
    assert report["owned_worker_stopped"] is True
    assert (output / "run.json").is_file()


def test_metadata_mode_does_not_request_numerical_work(monkeypatch, capsys):
    called = []
    report = dict(
        passed=True,
        export_files=267,
        exported_sources=254,
        parent_files=13,
        mathematical_source_files=19,
        scientific_functions_executed=False,
    )
    monkeypatch.setattr(
        P, "verify_inputs", lambda: (called.append("verify") or report, {})
    )
    monkeypatch.setattr(
        P, "supervise", lambda *a, **k: pytest.fail("unexpected worker")
    )
    assert P.main(["--verify-inputs-only"]) == 0
    assert called == ["verify"]
    assert json.loads(capsys.readouterr().out)["scientific_functions_executed"] is False

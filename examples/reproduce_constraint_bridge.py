"""Bounded portable reproduction of the case12 continuum constraint bridge.

This reruns the frozen forward and reverse exact derivative implementations.
It does not rerun the historical Windows qualification, its tests or its history
audit. Only NumPy is required by the mathematical worker. Input verification uses
the standard library and does not construct a numerical backend.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ENTRY_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PARENTS = "examples/data/constraint_bridge/parents"
MAX_SECONDS = 600.0
MAX_OUTPUT_BYTES = 64 * 1024 * 1024
TERMINAL_RESERVE = 128 * 1024
SEEDS = 580
CASES = ("baseline", "momentum_trace_mutant", "connection_target_mutant")
SCRIPT_HASHES = {
    "nr_case12_actual_constraint_jet.py": "b982b3415de4656b1c79cafa6cd6997094292e4a0d2dbccdd4c2f579343e137a",
    "nr_case12_actual_constraint_replay.py": "38f6edeb74af0296e665ab6c466dc36c80bfee1816c3355ecc2ce70c5c3fa841",
    "nr_case12_actual_constraint_diagnostic.py": "3bf3891d7a2adb715ddcd3e0478b6beb0e4b1e2931c0b0a43c222a21af8177d8",
    "nr_case12_actual_volume_jet.py": "1ef20ec3a7f7610744af016805b1678f578f0b2f07763575da61c60a3d31a868",
    "nr_case12_actual_volume_linearization.py": "0f0390c1c7ce12317674324157fc3820157ead5ffce69f20ee076e1cd3ca9b57",
    "nr_polynomial_certificate_replay.py": "36ff79bc1f3971f390b779c9898f769ce7c74b99bc469b42f3923fbe1984fae5",
    "nr_profile_exact_coefficients.py": "1c19244c4b4bd787fc389e8158aebc889c4d4ad8d556042866e9c7aa82056dbb",
    "nr_constraint_full_symbol.py": "053b70dd9136117dcfc8f2045b9953dc5ece6bfdf382eb1da12f33b9b8446042",
}
MATH_SOURCES = tuple("scripts/" + n for n in SCRIPT_HASHES) + tuple(
    "science/numerical_relativity/" + n
    for n in (
        "fo_ccz4_rhs.py",
        "fo_ccz4_sbp_volume.py",
        "fo_ccz4_principal_symbol.py",
        "fo_ccz4.py",
        "ccz4_reference.py",
        "fo_ccz4_gauge_profile.py",
        "fo_ccz4_constraint_control.py",
        "fo_ccz4_sbp_nonlinear.py",
        "nr3d.py",
        "fo_ccz4_relaxation_convergence.py",
        "fo_ccz4_checkpoint.py",
    )
)
SCOPE = (
    "portable core reproduction of the fixed-case12 continuum real-formula "
    "constraint bridge: all 43 diagnostics, 580 exact first-variation seeds, "
    "49 driver scalars, 40 driver agreements and 9 symmetry relations"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encoded(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def same(left, right):
    """JSON type-sensitive equality (unlike True == 1)."""
    return encoded(left) == encoded(right)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def relative_file(root, name):
    require(type(name) is str and "\\" not in name, "relative POSIX input path")
    path = PurePosixPath(name)
    require(
        not path.is_absolute()
        and path.parts
        and all(p not in (".", "..") and ":" not in p for p in path.parts),
        "contained input path",
    )
    result = root.joinpath(*path.parts)
    require(result.resolve().is_relative_to(root.resolve()), "contained input file")
    return result


def load_core(root=ROOT):
    # These imports only define classes/functions and pure metadata. NumPy is
    # imported later by an explicitly constructed worker backend.
    for path in (str(root), str(root / "scripts")):
        if path not in sys.path:
            sys.path.insert(0, path)
    return (
        importlib.import_module("nr_case12_actual_constraint_diagnostic"),
        importlib.import_module("nr_case12_actual_constraint_jet"),
        importlib.import_module("nr_case12_actual_constraint_replay"),
        importlib.import_module("nr_polynomial_certificate_replay"),
    )


def verify_inputs(root=ROOT, *, check=lambda: None):
    """Verify the exported archive and the 13 supplied parents, without AD."""
    root = Path(root)
    manifest_bytes = (root / "provenance/export-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    require(manifest.get("schema") == "solar-sim-nr-export-v1", "export schema")
    files = manifest.get("files")
    require(type(files) is list and len(files) == 267, "254 sources and 13 inputs")
    hashes = {}
    for entry in files:
        check()
        name = entry["path"]
        require(name not in hashes, "duplicate export path")
        data = relative_file(root, name).read_bytes()
        require(
            type(entry["bytes"]) is int and len(data) == entry["bytes"],
            "export byte count: " + name,
        )
        actual = digest(data)
        require(actual == entry["sha256"], "export SHA drift: " + name)
        hashes[name] = actual
    for name, expected in SCRIPT_HASHES.items():
        require(hashes.get("scripts/" + name) == expected, "frozen adapter drift")
    require(all(name in hashes for name in MATH_SOURCES), "19-source closure")
    module, actual, reverse, _ = load_core(root)
    raw = {name: (root / PARENTS / name).read_bytes() for name in module.PARENT_FILES}
    provenance = module.parent_provenance_manifest(raw)
    # These frozen functions bind the actual scientific AST and all adaptations.
    bridge, arithmetic = actual.source_manifest(), reverse.replay_manifest()
    require(len(raw) == 13, "13 parents")
    for name, data in raw.items():
        require(
            hashes.get(PARENTS + "/" + name) == digest(data),
            "parent also bound by export",
        )
    check()
    return dict(
        schema="solar-sim-nr-portable-input-verification-v1",
        passed=True,
        export_manifest_sha256=digest(manifest_bytes),
        export_files=len(hashes),
        exported_sources=254,
        parent_files=13,
        mathematical_source_files=19,
        input_hashes=hashes,
        mathematical_source_hashes={n: hashes[n] for n in MATH_SOURCES},
        parent_provenance=provenance,
        source_bridge_sha256=digest(encoded(bridge)),
        reverse_manifest_sha256=digest(encoded(arithmetic)),
        original_history_tree_audited=False,
        scientific_functions_executed=False,
        portable_entry_sha256=digest(
            (root / "examples/reproduce_constraint_bridge.py").read_bytes()
        ),
    ), raw


def output_census(directory):
    """Only regular files are permitted in this fresh, flat output directory."""
    root_stat = directory.stat(follow_symlinks=False)
    require(
        stat.S_ISDIR(root_stat.st_mode)
        and not getattr(root_stat, "st_file_attributes", 0) & 0x400,
        "regular output directory",
    )
    total, hashes = 0, {}
    with os.scandir(directory) as entries:
        for entry in entries:
            info = os.stat(entry.path, follow_symlinks=False)
            require(
                stat.S_ISREG(info.st_mode)
                and not getattr(info, "st_file_attributes", 0) & 0x400,
                "regular flat output files only",
            )
            total += info.st_size
            hashes[entry.name] = None
    return total, hashes


class Budget:
    """Cooperative per-phase guards plus a separately enforced parent timeout."""

    def __init__(self, directory, deadline, *, clock=time.monotonic):
        self.directory, self.deadline, self.clock = Path(directory), deadline, clock

    def check(self):
        if self.clock() >= self.deadline:
            raise TimeoutError("portable worker deadline")

    def guard(self, additional=0):
        self.check()
        require(type(additional) is int and additional >= 0, "nonnegative write size")
        used, _ = output_census(self.directory)
        require(
            used + additional + TERMINAL_RESERVE <= MAX_OUTPUT_BYTES,
            "64 MiB output budget including terminal reserve",
        )
        self.check()

    def record(self, name, value):
        require(
            type(name) is str and name and all(c.isalnum() or c in "_-" for c in name),
            "plain artifact name",
        )
        data = encoded(value)
        self.guard(len(data))
        with (self.directory / (name + ".json")).open("xb") as handle:
            handle.write(data)
        self.guard()


def independent_oracle(raw, codec, *, check=lambda: None):
    """Extract coefficients separately from the forward certificate mapper."""
    powers = (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    )
    block = {exponent: i for i, exponent in enumerate(powers)}
    result = [[Fraction(0)] * SEEDS for _ in range(43)]
    for degree in range(3):
        matrix = codec.decode_matrix(
            json.loads(raw[f"C{degree}.json"]), shape=(43, 58), max_degree=degree
        )
        for i, row in enumerate(matrix):
            check()
            for j, polynomial in enumerate(row):
                for exponent, value in polynomial.items():
                    require(
                        sum(exponent) == degree and exponent in block,
                        "homogeneous parent coefficient",
                    )
                    result[i][58 * block[exponent] + j] = value
    return result


def independent_rows(rows, count, *, check=lambda: None):
    require(type(rows) in (list, tuple) and len(rows) == count, "all scalar rows")
    values, coefficients = [], []
    for row in rows:
        check()
        terms = row.coefficients
        require(
            type(row.value) is Fraction
            and type(terms) is dict
            and all(
                type(index) is int
                and 0 <= index < SEEDS
                and type(value) is Fraction
                and value != 0
                for index, value in terms.items()
            ),
            "exact 580-seed row",
        )
        values.append([row.value])
        coefficients.append([terms.get(index, Fraction(0)) for index in range(SEEDS)])
    return values, coefficients


def independent_driver(rows, baseline, *, check=lambda: None):
    values, coefficients = independent_rows(rows, 49, check=check)
    raw = [[values[i][0], *coefficients[i]] for i in range(49)]
    physical = [[baseline["F0"][i][0], *baseline["C"][i]] for i in range(43)]
    mapping = [(i, i) for i in range(5)] + [(i, i + 3) for i in range(5, 17)]
    for k in range(3):
        index = 0
        for i in range(3):
            for j in range(i, 3):
                mapping.append((17 + 9 * k + 3 * i + j, 20 + 6 * k + index))
                index += 1
    mapping.extend((i, i - 6) for i in range(44, 49))
    require(len(mapping) == 40, "complete driver mapping")
    agreement, symmetry = [], []
    for source, target in mapping:
        check()
        agreement.append(
            [a - b for a, b in zip(raw[source], physical[target], strict=True)]
        )
    for k in range(3):
        for i in range(3):
            for j in range(i + 1, 3):
                check()
                symmetry.append(
                    [
                        a - b
                        for a, b in zip(
                            raw[17 + 9 * k + 3 * i + j],
                            raw[17 + 9 * k + 3 * j + i],
                            strict=True,
                        )
                    ]
                )
    return dict(
        driverF0=values,
        driverC=coefficients,
        driverAgreement=agreement,
        driverSymmetry=symmetry,
    )


def independent_controls(cases):
    controls = {}
    for name, targets in ((CASES[1], [1, 2, 3]), (CASES[2], [5, 6, 7])):
        changes = [
            (i, j)
            for i in range(43)
            for j in range(SEEDS)
            if cases[name]["C"][i][j] != cases["baseline"]["C"][i][j]
        ]
        causal = sum(i in targets and cases[name]["R"][i][j] != 0 for i, j in changes)
        outside = sum(i not in targets for i, _ in changes)
        controls[name] = dict(
            target_rows=targets,
            changed_slots=len(changes),
            changed_outside_target_rows=outside,
            causal_changed_residual_slots=causal,
            original_diagnostic_symbol_rejected=causal > 0,
            accepted=bool(changes) and causal > 0 and outside == 0,
        )
    return dict(
        schema="nr-case12-actual-constraint-real-source-controls-v1",
        controls=controls,
        accepted=all(c["accepted"] for c in controls.values()),
        prefilled_theoretical_pattern=False,
        actual_mutated_source_evaluated=True,
    )


def independent_replay(directory, raw, budget):
    module, actual, reverse, codec = load_core()
    oracle = independent_oracle(raw, codec, check=budget.check)
    cases, statistics, executions, errors = {}, {}, {}, []
    driver_rows = None

    def compare(name, record):
        budget.guard()
        if not same(read_json(directory / (name + ".json")), record):
            errors.append("metadata mismatch: " + name)

    def compare_matrix(name, matrix):
        budget.guard()
        saved = module.decode_matrix(
            read_json(directory / (name + ".json")), module.SHAPES[name]
        )
        if saved != matrix:
            errors.append("matrix mismatch: " + name)
        rows, cols = module.SHAPES[name]
        nonzero = sum(value != 0 for row in matrix for value in row)
        statistics[name] = dict(
            shape=[rows, cols],
            slots=rows * cols,
            nonzero_slots=nonzero,
            all_zero=nonzero == 0,
        )

    for case in CASES:
        budget.guard()
        result = reverse.evaluate_case(
            case=case, guard=budget.guard, check=budget.check
        )
        values, matrix = independent_rows(result["rows"], 43, check=budget.check)
        residual = []
        for i in range(43):
            budget.check()
            residual.append([matrix[i][j] - oracle[i][j] for j in range(SEEDS)])
        cases[case] = dict(F0=values, C=matrix, R=residual)
        for field in ("F0", "C", "R"):
            compare_matrix(case + "__" + field, cases[case][field])
        execution = result["execution"]
        saved = read_json(directory / (case + "__execution.json"))

        def stable(record):
            return {
                k: v
                for k, v in record.items()
                if k not in ("operations", "independent_backend")
            }

        if not same(stable(execution), stable(saved)):
            errors.append("case/source/chart mismatch: " + case)
        executions[case] = execution
        if case == "baseline":
            driver_rows = result["raw_driver"]
        else:
            require(result["raw_driver"] is None, "baseline-only driver")
    for name, matrix in independent_driver(
        driver_rows, cases["baseline"], check=budget.check
    ).items():
        compare_matrix(name, matrix)
    mutations = independent_controls(cases)
    acceptance = dict(
        baseline_equilibrium=statistics["baseline__F0"]["all_zero"],
        complete_physical43_bridge=statistics["baseline__R"]["all_zero"],
        driver_equilibrium=statistics["driverF0"]["all_zero"],
        driver_complete40_bridge=statistics["driverAgreement"]["all_zero"],
        driver_nine_symmetry_relations=statistics["driverSymmetry"]["all_zero"],
        real_source_negative_controls=mutations["accepted"],
    )
    acceptance["accepted"] = all(acceptance.values())
    summary = dict(
        cases=3,
        matrices=13,
        total_slots=206707,
        main_slots=106861,
        mutation_slots=99846,
        required_baseline_zero_slots=53501,
        outputs=22,
        parent_files=13,
        matrices_by_name=statistics,
    )
    for name, record in (
        ("mutations", mutations),
        ("acceptance", acceptance),
        ("stats", summary),
        ("source_bridge", actual.source_manifest()),
        ("parent_provenance", module.parent_provenance_manifest(raw)),
        ("domain", module.domain_metadata()),
    ):
        compare(name, record)
    return dict(
        schema="solar-sim-nr-portable-independent-replay-v1",
        computation_completed=True,
        passed=acceptance["accepted"] and not errors,
        errors=errors,
        scope=SCOPE,
        acceptance=acceptance,
        mutations=mutations,
        stats=summary,
        independent_executions=executions,
        independent_arithmetic=reverse.replay_manifest(),
        shared_trusted_base="actual AST adapters, object-array routing, chart/packing, codecs",
        independent_method="reverse scalar AD and cofactor matrix arithmetic",
        forward_jacobians_used_to_generate_reverse=False,
        original_full_qualification_rerun=False,
        compared_output_hashes={
            n + ".json": digest((directory / (n + ".json")).read_bytes())
            for n in module.RECORD_ORDER
        },
    )


def worker(directory, timeout):
    start = time.monotonic()
    budget = Budget(directory, start + timeout)
    stage, failed_type, passed = "verify-inputs", None, False
    try:
        verification, raw = verify_inputs(check=budget.check)
        budget.record("input-verification", verification)
        module, _, _, _ = load_core()
        produced = []

        def sink(name, record):
            require(
                len(produced) < len(module.RECORD_ORDER)
                and name == module.RECORD_ORDER[len(produced)],
                "complete ordered outputs",
            )
            budget.record(name, record)
            produced.append(name)

        stage = "forward-all-three-cases"
        acceptance = module.build_certificate(
            raw, guard=budget.guard, check=budget.check, sink=sink
        )
        require(tuple(produced) == module.RECORD_ORDER, "all22 forward records")
        # A rejected forward candidate still receives a complete reverse replay.
        stage = "reverse-all-three-cases"
        replay = independent_replay(directory, raw, budget)
        budget.record("independent-replay", replay)
        stage = "final-input-audit"
        after, _ = verify_inputs(check=budget.check)
        require(same(after, verification), "input drift during reproduction")
        passed = acceptance["accepted"] is True and replay["passed"] is True
        budget.guard()
        stage = "complete"
    except Exception as error:
        # Avoid absolute local paths in public receipts. Full failed matrices
        # already saved remain intact; the failure class and stage are retained.
        failed_type = type(error).__name__
    result = dict(
        schema="solar-sim-nr-portable-worker-v1",
        passed=passed,
        computation_completed=stage == "complete",
        stage=stage,
        error_type=failed_type,
        elapsed_seconds=time.monotonic() - start,
    )
    terminal_write(directory, "worker-result.json", result)
    return 0 if passed else (124 if failed_type == "TimeoutError" else 1)


def terminal_write(directory, name, record):
    """Use the reserved space after failure/deadline; never overwrite evidence."""
    data = encoded(record)
    used, _ = output_census(directory)
    require(
        len(data) <= TERMINAL_RESERVE and used + len(data) <= MAX_OUTPUT_BYTES,
        "bounded terminal record",
    )
    with (directory / name).open("xb") as handle:
        handle.write(data)


def runtime_versions():
    try:
        numpy = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy = None
    return dict(
        python=platform.python_version(),
        implementation=platform.python_implementation(),
        numpy=numpy,
    )


def supervise(directory, timeout, *, process_factory=subprocess.Popen):
    """Only this one owned worker may run; kill it at the wall-clock deadline."""
    start = time.monotonic()
    directory = Path(directory)
    directory.mkdir(parents=False, exist_ok=False)
    deadline = start + timeout
    initial = dict(
        schema="solar-sim-nr-portable-run-v1",
        scope=SCOPE,
        timeout_seconds=timeout,
        output_limit_bytes=MAX_OUTPUT_BYTES,
        historical_full_qualification_rerun=False,
        portable_entry_sha256=ENTRY_SOURCE_SHA256,
    )
    terminal_write(directory, "run.json", initial)
    environment = dict(os.environ)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    process, timed_out, error_type, exit_code = None, False, None, None
    before = None
    try:
        before, _ = verify_inputs(
            check=lambda: require(
                time.monotonic() < deadline, "supervisor preflight deadline"
            )
        )
        require(
            before.get("portable_entry_sha256") == ENTRY_SOURCE_SHA256,
            "portable entry changed after it was loaded",
        )
        remaining = deadline - time.monotonic()
        require(remaining > 0, "positive remaining worker budget")
        process = process_factory(
            [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker",
                "--output",
                str(directory.resolve()),
                "--timeout",
                str(remaining),
            ],
            cwd=str(ROOT),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            exit_code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            exit_code = process.wait(timeout=10)
    except (Exception, KeyboardInterrupt) as error:
        error_type = type(error).__name__
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            try:
                exit_code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                error_type = "OwnedWorkerCleanupFailed"
    stopped = process is None or process.poll() is not None
    audit_passed, artifacts = False, {}
    try:
        used, names = output_census(directory)
        require(used <= MAX_OUTPUT_BYTES - TERMINAL_RESERVE, "terminal reserve intact")
        artifacts = {
            name: digest((directory / name).read_bytes()) for name in sorted(names)
        }
        after, _ = verify_inputs()
        audit_passed = before is not None and same(after, before)
        saved_worker = read_json(directory / "worker-result.json")
        replay = read_json(directory / "independent-replay.json")
        acceptance = read_json(directory / "acceptance.json")
        module, _, _, _ = load_core()
        expected = {n + ".json" for n in module.RECORD_ORDER} | {
            "run.json",
            "input-verification.json",
            "independent-replay.json",
            "worker-result.json",
        }
        require(set(names) == expected, "complete portable artifact set")
        require(
            replay.get("errors") == [] and replay.get("computation_completed") is True,
            "complete independent replay",
        )
        require(
            replay["compared_output_hashes"]
            == {n + ".json": artifacts[n + ".json"] for n in module.RECORD_ORDER},
            "replayed artifact identity",
        )
        require(
            same(read_json(directory / "input-verification.json"), before),
            "worker verified same inputs",
        )
        accepted = (
            saved_worker.get("passed") is True
            and saved_worker.get("computation_completed") is True
            and replay.get("passed") is True
            and acceptance.get("accepted") is True
        )
    except Exception as error:
        accepted = False
        error_type = error_type or type(error).__name__
    passed = bool(
        accepted
        and audit_passed
        and stopped
        and exit_code == 0
        and not timed_out
        and error_type is None
    )
    result = dict(
        schema="solar-sim-nr-portable-result-v1",
        passed=passed,
        scope=SCOPE,
        timed_out=timed_out,
        worker_exit_code=exit_code,
        owned_worker_stopped=stopped,
        error_type=error_type,
        input_audit_passed=audit_passed,
        elapsed_seconds=time.monotonic() - start,
        timeout_seconds=timeout,
        output_limit_bytes=MAX_OUTPUT_BYTES,
        artifact_hashes=artifacts,
        runtime=runtime_versions(),
        portable_entry_sha256=ENTRY_SOURCE_SHA256,
        portable_core_reproduction_passed=passed,
        original_full_qualification_rerun=False,
        original_focused_tests_rerun=False,
        original_history_tree_audited=False,
        independent_external_reproduction=False,
        actual_binary64_executed=False,
        discrete_evolution_validated=False,
        boundary_validated=False,
        timeout_note="worker is killed at the deadline; termination and receipt finalization may finish later",
    )
    terminal_write(directory, "portable-result.json", result)
    return 0 if passed else (124 if timed_out or exit_code == 124 else 1)


def timeout_value(value):
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("timeout must be a number") from error
    if not math.isfinite(result) or not 0 < result <= MAX_SECONDS:
        raise argparse.ArgumentTypeError("timeout must be > 0 and <= 600 seconds")
    return result


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-inputs-only",
        action="store_true",
        help="check all exported bytes and parent/source manifests, without numerical imports",
    )
    parser.add_argument(
        "--output", type=Path, help="new directory; its parent must exist"
    )
    parser.add_argument("--timeout", type=timeout_value, default=MAX_SECONDS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.verify_inputs_only and (args.worker or args.output is not None):
        parser.error(
            "--verify-inputs-only cannot be combined with --output or --worker"
        )
    if not args.verify_inputs_only and args.output is None:
        parser.error("--output is required for a reproduction")
    return args


def main(argv=None):
    args = arguments(argv)
    try:
        if args.verify_inputs_only:
            report, _ = verify_inputs()
            print(
                json.dumps(
                    {
                        k: report[k]
                        for k in (
                            "passed",
                            "export_files",
                            "exported_sources",
                            "parent_files",
                            "mathematical_source_files",
                            "scientific_functions_executed",
                        )
                    }
                )
            )
            return 0
        if args.worker:
            return worker(args.output, args.timeout)
        code = supervise(args.output, args.timeout)
        print(
            json.dumps(
                dict(
                    passed=code == 0, exit_code=code, result_file="portable-result.json"
                )
            )
        )
        return code
    except Exception as error:
        print(
            json.dumps(dict(passed=False, error_type=type(error).__name__)),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

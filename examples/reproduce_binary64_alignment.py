"""Unexecuted candidate: guarded case12 binary64 8^3 alignment, not evolution.

Metadata-only modes import no NumPy/scientific modules. A fresh owned worker is
released only after preflight, byte accounting and platform containment succeed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.case12_binary64_v1 import runtime as rt  # noqa: E402
from experiments.case12_binary64_v1.containment import (  # noqa: E402
    WindowsJob,
    cleanup_owned,
    worker_python,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-inputs-only", action="store_true")
    modes.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/binary64-case12"))
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def worker(args):
    rt.require(
        args.deadline is not None and args.manifest_sha256, "worker contract missing"
    )
    rt.require(os.environ.get("NR_BINARY64_OWNED_WORKER") == "1", "supervisor required")
    evidence = rt.Evidence(
        args.output, args.deadline, check_resources=True, terminal_reserved=True
    )
    # Parent first attaches Windows hard-memory job; no NumPy before release.
    release = args.output / "start.json"
    while not release.exists():
        evidence.check()
        time.sleep(0.02)
    release_record = rt.decode(release.read_bytes())
    rt.require(
        release_record["manifest_sha256"] == args.manifest_sha256,
        "worker release identity",
    )
    rt.require(
        release_record["owned_pid"] == os.getpid(),
        "worker PID differs from owned Job member",
    )
    rt.require(
        release_record["python_prefix"] == sys.prefix, "worker environment changed"
    )
    try:
        verified = rt.verify_inputs(ROOT, evidence.check)
        rt.require(
            verified["manifest_sha256"] == args.manifest_sha256,
            "manifest changed at launch",
        )
        from experiments.case12_binary64_v1 import (
            comparison,
            kernel,
            provenance,
            replay,
        )

        rt.require(kernel.ARRAY_SPECS == rt.ARRAY_SHAPES, "kernel output schema drift")
        prepared = provenance.prepare_reference(
            verified["inputs"], check=evidence.check
        )
        import numpy as np

        reference = {key: prepared[key] for key in ("parameters", "rhs", "constraints")}
        evidence.json("reference-conversions", prepared["conversion_errors"])
        evidence.json(
            "environment",
            dict(
                python=sys.version,
                numpy=np.__version__,
                platform=sys.platform,
                executable=sys.executable,
                process_id=os.getpid(),
                python_prefix=sys.prefix,
                single_thread_environment={k: os.environ[k] for k in THREAD_VARIABLES},
            ),
        )

        def source_check():
            for name, module in tuple(sys.modules.items()):
                if name == "science" or name.startswith(
                    ("science.", "scripts.", "experiments.")
                ):
                    filename = getattr(module, "__file__", None)
                    if filename is None:  # namespace package has no source payload
                        continue
                    path = Path(filename).resolve()
                    rt.require(
                        path.is_relative_to(ROOT),
                        "scientific import outside repository",
                    )
                    relative = path.relative_to(ROOT).as_posix()
                    rt.require(
                        relative in verified["raw"]
                        and rt.bounded_read(
                            path, len(verified["raw"][relative]), evidence.check
                        )
                        == verified["raw"][relative],
                        "unbound or changed scientific source: " + relative,
                    )

        summaries = {}
        for label, implementation in (("production", kernel), ("replay", replay)):

            def array_sink(name, value, prefix=label):
                rt.require(np.all(np.isfinite(value)), "nonfinite evidence")
                evidence.array(
                    prefix + "/" + name, np.ascontiguousarray(value, dtype="<f8")
                )

            def json_sink(name, value, prefix=label):
                evidence.json(prefix + "/" + name, value, progress=name == "progress")

            source_check()
            summaries[label] = implementation.run(
                reference,
                guard=evidence.check,
                check=evidence.check,
                array_sink=array_sink,
                json_sink=json_sink,
            )
            source_check()
        cross = comparison.compare(args.output, check=evidence.check)
        evidence.json("comparison", cross)
        source_check()
        final_inputs = rt.verify_inputs(ROOT, evidence.check)
        rt.require(
            final_inputs["manifest_sha256"] == args.manifest_sha256,
            "source/input final drift",
        )
        accepted = (
            all(item["acceptance"]["accepted"] is True for item in summaries.values())
            and cross["accepted"]
        )
        evidence.json(
            "worker-result",
            dict(
                completed=True,
                accepted=accepted,
                manifest_sha256=args.manifest_sha256,
                state_evaluations=964,
                rhs_calls=964,
                physical43_calls=964,
                driver49_executed=False,
                time_evolution_executed=False,
                complete_signed_grid_archive=False,
            ),
        )
        return 0 if accepted else 1
    except BaseException as error:
        # Reserved final receipt is the parent's responsibility; never erase work.
        try:
            evidence.json(
                "worker-failure",
                dict(
                    completed=False,
                    accepted=False,
                    exception_type=type(error).__name__,
                    message=str(error)[:4000],
                ),
            )
        except Exception:
            pass
        return 1


THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def successful_inventory():
    names = {
        prefix + "/" + name + ".npy"
        for prefix in ("production", "replay")
        for name in rt.ARRAY_SHAPES
    }
    names.update(
        {
            "preflight.json",
            "source-input-snapshot.zip",
            "start.json",
            "reference-conversions.json",
            "environment.json",
            "comparison.json",
            "worker-result.json",
            "terminal.json",
        }
    )
    names.update(
        "production/" + name + ".json"
        for name in (
            "parameters",
            "scalar_controls",
            "calls",
            "acceptance",
            "progress",
            "stage_main_0_0",
            "stage_main_0_1",
            "stage_main_1_0",
            "stage_main_1_1",
        )
    )
    names.update(
        "replay/" + name + ".json"
        for name in ("design_and_stencils", "calls", "acceptance", "progress")
    )
    return names


def finalize_terminal(slot, terminal, deadline_check):
    """The durable record stays provisional; only a clean supervisor can commit.

    A failed fsync, close or trailing deadline must never leave a standalone
    PASSED receipt. Consumers require the matching stdout commit and exit 0.
    """
    checks_passed = terminal["accepted"] is True
    terminal.update(
        accepted=False,
        checks_passed=checks_passed,
        acceptance_requires_supervisor_commit=True,
        state="CHECKS_PASSED_AWAITING_SUPERVISOR_EXIT"
        if checks_passed
        else "NOT_QUALIFIED",
        terminal_available=True,
    )
    try:
        digest = slot.write(terminal)
        slot.close()
        if checks_passed:
            deadline_check()
            return digest
    except BaseException as error:
        terminal.update(
            accepted=False,
            state="NOT_QUALIFIED",
            terminal_available=False,
            terminal_unavailable_reason=type(error).__name__ + ": " + str(error)[:1000],
        )
        # Best effort only. Even if this fails, the prior record is provisional.
        try:
            slot.write(terminal)
        except Exception:
            pass
        try:
            slot.close()
        except Exception:
            pass
    return None


def supervise(args):
    started = time.monotonic()
    rt.require(
        1 <= args.timeout_seconds <= rt.MAX_SECONDS,
        "timeout must be within 1..600 seconds",
    )
    deadline = started + args.timeout_seconds

    def deadline_check():
        if time.monotonic() >= deadline:
            raise TimeoutError("bounded preflight deadline")

    verified = rt.verify_inputs(ROOT, deadline_check)
    budget = rt.output_budget(verified)
    deadline_check()
    if args.verify_inputs_only:
        print(
            rt.encode(
                dict(
                    inputs_verified=True,
                    numerical_execution=False,
                    manifest_sha256=verified["manifest_sha256"],
                    budget=budget,
                )
            ).decode(),
            end="",
        )
        return 0
    output = (
        args.output if args.output.is_absolute() else ROOT / args.output
    ).absolute()
    rt.require(
        output.is_relative_to(ROOT / "outputs"),
        "output must stay under this repository's outputs directory",
    )
    rt.safe_path(ROOT, output.relative_to(ROOT).as_posix(), exists=False)
    rt.require(not output.exists(), "output directory must be fresh")
    resource_snapshot = rt.preflight(output.parent)
    deadline_check()
    if args.preflight_only:
        print(
            rt.encode(
                dict(
                    preflight_passed=True,
                    numerical_execution=False,
                    resources=resource_snapshot,
                    budget=budget,
                )
            ).decode(),
            end="",
        )
        return 0
    output.mkdir()
    slot = rt.TerminalSlot(output)
    process, job, result, cleanup, peak = None, None, None, None, 0
    evidence = None
    terminal = dict(
        schema="case12-binary64-terminal-v1",
        accepted=False,
        completed=False,
        manifest_sha256=verified["manifest_sha256"],
        timeout_seconds=args.timeout_seconds,
        output_limit_bytes=rt.MAX_BYTES,
        automatic_retry=False,
        scientific_flags_promoted=False,
        scientificPayloadWriteback=False,
        shadow_only=True,
    )
    try:
        evidence = rt.Evidence(
            output, deadline, check_resources=True, terminal_reserved=True
        )
        evidence.json(
            "preflight",
            dict(
                resources=resource_snapshot,
                parents=verified["parents"],
                budget=budget,
                computation_deadline_includes_preflight_snapshot_and_comparison=True,
                failure_cleanup_may_finish_after_deadline=True,
                memory_enforcement="Windows whole-Job and process commit caps plus sampled RSS"
                if sys.platform == "win32"
                else "Linux sampled RSS watchdog; not a hard virtual-address-space cap",
            ),
        )
        evidence.snapshot(verified)
        job = WindowsJob(memory_bytes=rt.WORKER_BYTES)
        environment = os.environ.copy()
        environment.update({key: "1" for key in THREAD_VARIABLES})
        environment.update(
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONNOUSERSITE="1",
            NR_BINARY64_OWNED_WORKER="1",
            PYTHONPATH=str(ROOT),
            PYTHONPYCACHEPREFIX=str(output / "never-written-bytecode"),
        )
        command = [
            worker_python(environment),
            "-B",
            str(Path(__file__).resolve()),
            "--worker",
            "--output",
            str(output),
            "--deadline",
            repr(deadline),
            "--manifest-sha256",
            verified["manifest_sha256"],
        ]
        terminal["resources_before_spawn"] = rt.preflight(output)
        deadline_check()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=sys.platform != "win32",
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        job.assign(process)
        terminal["resources_before_release"] = rt.preflight(output)
        deadline_check()
        evidence.json(
            "start",
            dict(
                manifest_sha256=verified["manifest_sha256"],
                owned_pid=process.pid,
                python_prefix=sys.prefix,
            ),
            publish=True,
        )
        while process.poll() is None:
            evidence.check()
            try:
                snapshot = rt.resources(process.pid)
            except (OSError, ValueError, KeyError):
                if process.poll() is not None:
                    break
                raise
            peak = max(peak, snapshot["worker_peak_rss_bytes"])
            rt.require(
                snapshot["available_ram_bytes"] >= rt.RUN_FREE_RAM,
                "runtime RAM reserve",
            )
            rt.require(
                max(
                    snapshot["worker_rss_bytes"],
                    snapshot.get("worker_private_bytes", 0),
                )
                <= rt.WORKER_BYTES,
                "worker exceeds 1 GiB",
            )
            time.sleep(0.05)
        evidence.check()
        terminal["exit_code"] = process.returncode
        cleanup = cleanup_owned(process, job)
        terminal["cleanup"] = cleanup
        rt.require(
            cleanup["empty"]
            and cleanup["closed"]
            and not cleanup["unexpected_residuals"],
            "owned container not cleanly empty before sealing",
        )
        evidence.check()
        result_path = output / "worker-result.json"
        if result_path.is_file():
            result = rt.decode(result_path.read_bytes())
        current = rt.verify_inputs(ROOT, deadline_check)
        rt.require(
            current["manifest_sha256"] == verified["manifest_sha256"],
            "final manifest drift",
        )
        for prefix in ("production", "replay"):
            actual = {p.stem for p in (output / prefix).glob("*.npy")}
            rt.require(
                actual == set(rt.ARRAY_SHAPES), "incomplete numerical output inventory"
            )
        rt.require(
            set(evidence.census()) == successful_inventory(),
            "missing or unexpected evidence artifact",
        )
        terminal["completed"] = bool(result and result["completed"] is True)
        terminal["accepted"] = bool(
            process.returncode == 0
            and result
            and result["accepted"] is True
            and result["completed"] is True
            and result["manifest_sha256"] == verified["manifest_sha256"]
        )
        terminal["output_hashes_before_terminal"] = evidence.seal(enforce_deadline=True)
        evidence.check()
    except BaseException as error:
        terminal.update(
            accepted=False,
            exception_type=type(error).__name__,
            message=str(error)[:4000],
        )
    finally:
        try:
            if cleanup is None:
                cleanup = cleanup_owned(process, job)
                terminal["cleanup"] = cleanup
            rt.require(
                cleanup["empty"] and cleanup["closed"], "owned residual processes"
            )
            if cleanup["unexpected_residuals"]:
                terminal["accepted"] = False
        except Exception as error:
            terminal.update(
                accepted=False,
                cleanup_error=type(error).__name__ + ": " + str(error)[:1000],
            )
        finally:
            if job is not None:
                try:
                    job.close()
                except Exception as error:
                    terminal.update(accepted=False, job_close_error=str(error)[:1000])
        if evidence is not None and "output_hashes_before_terminal" not in terminal:
            try:
                terminal["output_hashes_before_terminal"] = evidence.seal(
                    enforce_deadline=False
                )
            except Exception as error:
                terminal.update(
                    accepted=False,
                    seal_error=type(error).__name__ + ": " + str(error)[:1000],
                )
        terminal["progress"] = {}
        for prefix in ("production", "replay"):
            path = output / prefix / "progress.json"
            try:
                if path.is_file():
                    terminal["progress"][prefix] = rt.decode(path.read_bytes())
            except Exception as error:
                terminal["accepted"] = False
                terminal["progress"][prefix] = {
                    "unreadable": True,
                    "error": str(error)[:1000],
                }
        try:
            rt.require(evidence is not None, "evidence recorder unavailable")
            terminal["output_inventory_before_terminal"] = evidence.census()
            terminal["output_bytes_before_terminal"] = sum(
                terminal["output_inventory_before_terminal"].values()
            )
            rt.require(
                terminal["output_bytes_before_terminal"] <= rt.MAX_BYTES,
                "final output limit",
            )
            if terminal["accepted"]:
                sealed = {
                    name: record["bytes"]
                    for name, record in terminal[
                        "output_hashes_before_terminal"
                    ].items()
                }
                sealed["terminal.json"] = rt.RESERVE_BYTES
                rt.require(
                    terminal["output_inventory_before_terminal"] == sealed
                    and set(sealed) == successful_inventory(),
                    "final census differs from sealed inventory",
                )
                evidence.check()
        except Exception as error:
            terminal.update(accepted=False, census_error=str(error)[:1000])
        terminal.update(
            elapsed_seconds=time.monotonic() - started,
            worker_peak_rss_bytes=peak,
            state="PASSED" if terminal["accepted"] else "NOT_QUALIFIED",
            terminal_available=True,
        )
        commit_digest = finalize_terminal(slot, terminal, deadline_check)
    print(
        rt.encode(
            dict(
                schema="case12-binary64-supervisor-commit-v1",
                state="PASSED" if commit_digest else "NOT_QUALIFIED",
                accepted=commit_digest is not None,
                terminal_sha256=commit_digest,
                manifest_sha256=verified["manifest_sha256"],
                output=str(output),
                elapsed_seconds=time.monotonic() - started,
                terminal_available=terminal["terminal_available"],
                terminal_unavailable_reason=terminal.get("terminal_unavailable_reason"),
            )
        ).decode(),
        end="",
    )
    return 0 if commit_digest else 1


def preflight_failure_record(error):
    record = dict(
        state="NOT_STARTED",
        numerical_execution=False,
        exception_type=type(error).__name__,
        message=str(error),
    )
    # A refusal before output creation is otherwise hard to diagnose from the
    # console alone. Keep the registered threshold unchanged and attach a fresh
    # read-only resource observation when the RAM gate is the blocker.
    if str(error) == "need at least 4 GiB available RAM":
        try:
            record["resources"] = rt.resources()
            record["required_available_ram_bytes"] = rt.MIN_FREE_RAM
        except (OSError, ValueError) as observation_error:
            record["resource_observation_error"] = str(observation_error)
    return record


def main(argv=None):
    args = parse_args(argv)
    try:
        return worker(args) if args.worker else supervise(args)
    except (ValueError, OSError, TimeoutError) as error:
        print(rt.encode(preflight_failure_record(error)).decode(), end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

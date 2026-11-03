"""Local serial control runner; this is not a scientific acceptance receipt."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.case12_binary64_v1 import runtime as rt
from experiments.case12_binary64_v1.containment import WindowsJob, cleanup_owned, worker_python


def size(root):
    total = 0
    for directory, _, names in os.walk(root):
        for name in names:
            try:
                total += (Path(directory) / name).stat().st_size
            except FileNotFoundError:
                pass  # pytest fixture cleanup can race this monitoring census.
    return total


def main():
    started = time.monotonic()
    output = ROOT / "outputs" / sys.argv[1]
    rt.require(output.parent == ROOT / "outputs" and not output.exists(), "fresh local output")
    initial = rt.preflight(output.parent)
    output.mkdir()
    receipt = output / "controls-result.json"
    receipt.write_text(json.dumps({"accepted": False, "state": "STARTED"}), encoding="utf-8")
    tests = sys.argv[2:] or [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "tests").glob("test*binary64*.py"))]
    env = os.environ.copy()
    env.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD")})
    env.update(TEMP=str(output), TMP=str(output), PYTHONPATH=str(ROOT))
    child = "import os,sys,json; release=json.loads(sys.stdin.readline()); assert release['pid']==os.getpid() and release['prefix']==sys.prefix; import pytest; sys.exit(pytest.main(sys.argv[1:]))"
    command = [worker_python(env), "-B", "-c", child, "-q", "-o", "cache_dir=" + str(output / "cache"), "--basetemp", str(output / "fixtures"), *tests]
    process = job = reader = None
    error = cleanup = None
    peak = 0
    minimum = initial["available_ram_bytes"]
    log_limit = threading.Event()
    log_errors = []

    def capture():
        try:
            count = 0
            with (output / "pytest.log").open("xb") as log:
                while chunk := process.stdout.read(4096):
                    if count + len(chunk) > 1024**2:
                        log_limit.set()
                        break
                    log.write(chunk)
                    log.flush()
                    count += len(chunk)
        except BaseException as exc:
            log_errors.append(repr(exc))

    try:
        job = WindowsJob(memory_bytes=rt.WORKER_BYTES)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        job.assign(process)
        rt.preflight(output)
        reader = threading.Thread(target=capture, daemon=True)
        reader.start()
        process.stdin.write((json.dumps(dict(pid=process.pid, prefix=sys.prefix)) + "\n").encode())
        process.stdin.flush()
        process.stdin.close()
        while process.poll() is None:
            rt.require(time.monotonic() - started < 600, "600-second control deadline")
            rt.require(not log_limit.is_set() and not log_errors, "bounded log capture failed")
            rt.require(size(output) < 50 * 1024**2, "control output early stop at 50 MiB")
            rt.disk_headroom(output)
            try:
                current = rt.resources(process.pid)
            except (OSError, ValueError):
                if process.poll() is not None:
                    break
                raise
            minimum = min(minimum, current["available_ram_bytes"])
            peak = max(peak, current["worker_peak_rss_bytes"])
            rt.require(minimum >= rt.RUN_FREE_RAM, "runtime RAM reserve")
            rt.require(max(current["worker_rss_bytes"], current["worker_private_bytes"]) <= rt.WORKER_BYTES, "worker memory cap")
            time.sleep(0.1)
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        try:
            cleanup = cleanup_owned(process, job)
        except BaseException as exc:
            error = (error or "") + "; cleanup: " + repr(exc)
        if reader:
            reader.join(timeout=5)
        if process and process.stdout:
            process.stdout.close()
        if reader and reader.is_alive():
            error = (error or "") + "; log reader did not exit"
        if log_errors or log_limit.is_set():
            error = (error or "") + "; output capture failure"
    elapsed = time.monotonic() - started
    accepted = bool(not error and process and process.returncode == 0 and cleanup and cleanup["empty"] and cleanup["closed"] and not cleanup["unexpected_residuals"] and elapsed < 600 and size(output) < 64 * 1024**2 - 16384)
    record = dict(schema="local-bounded-controls-v1", accepted=accepted, tests=tests, scientific_campaign_executed=False, exit_code=None if process is None else process.returncode, elapsed_seconds=elapsed, initial_resources=initial, minimum_sampled_free_ram_bytes=minimum, worker_peak_rss_bytes=peak, job_memory_cap_bytes=rt.WORKER_BYTES, timeout_seconds=600, output_limit_bytes=rt.MAX_BYTES, output_bytes_before_receipt=size(output), cleanup=cleanup, error=error, tested_source_hashes={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in tests}, manifest_sha256=hashlib.sha256((ROOT / rt.MANIFEST_PATH).read_bytes()).hexdigest())
    receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())

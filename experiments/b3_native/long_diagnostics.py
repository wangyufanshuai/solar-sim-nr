"""Bounded native S2-c/S3 diagnostics. See docs/LOW_MEMORY_LONG_DIAGNOSTICS.md."""
from __future__ import annotations
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import time
import numpy as np
from science.numerical_relativity import BSSNState3D, FOCCZ4State3D, GaugeConfig, FOCCZ4Config
from science.numerical_relativity.constraints3d import constraint_report_3d
from kernel.native_evolve import slots, base_fields, state_from_slots, make_rhs, rk4
from experiments.s2_periodic_evolution.robust_diagnostic import noisy_state, rms
from experiments.s2_periodic_evolution.robust_v2 import total_norm
from experiments.b1_gauge_wave import run_b1


class ScientificFailure(ValueError):
    pass


def validate_slots(q):
    if not np.all(np.isfinite(q)):
        raise ScientificFailure("nonfinite evolved slots")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        alpha, chi = np.exp(q[6]), np.exp(2 * q[10])
    if np.any(alpha <= 0) or np.any(chi <= 0):
        raise ScientificFailure("chi/lapse positivity lost in float64 reconstruction")


def checked_rhs(rhs):
    def evaluate(q):
        evaluate.calls += 1
        try:
            validate_slots(q)
            out = rhs(q)
            if not np.all(np.isfinite(out)):
                raise ScientificFailure("nonfinite native RHS")
            return out
        except (ScientificFailure, FloatingPointError) as exc:
            raise ScientificFailure(f"RK4 stage {1 + (evaluate.calls-1) % 4}: {exc}") from exc
    evaluate.calls = 0
    return evaluate


def peak_rss():
    if os.name != "nt":
        import resource
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if __import__('sys').platform == 'darwin' else 1024))
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not query(get_process(), ctypes.byref(counters), counters.cb):
        raise OSError("cannot measure process working set")
    return counters.PeakWorkingSetSize


def windows(rows):
    result = []
    for lo, hi in ((50, 100), (100, 200), (200, 500)):
        selected = [r for r in rows if lo <= r["time"] <= hi]
        if not selected or selected[0]["time"] != lo or selected[-1]["time"] != hi:
            continue
        t = np.array([r["time"] for r in selected])
        fields = {key: float(np.polyfit(t, np.log(np.maximum([r["rms"][key] for r in selected], 1e-300)), 1)[0]) for key in selected[0]["rms"]}
        norm = [total_norm(r["rms"]) for r in selected]
        result.append(dict(start=lo, end=hi, log_slope=float(np.polyfit(t, np.log(norm), 1)[0]), norm_ratio=norm[-1]/norm[0], per_field_log_slope=fields))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("s2", "b1"))
    parser.add_argument("--final-time", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.final_time < 1:
        parser.error("final time must be a positive integer")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive destination preserves previous successful and failed attempts.
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write('{}\n')
    if args.case == "s2":
        initial = noisy_state(8, 4)
        g, dt, per_unit = initial.grid, .0125, 80
        gauge, config = GaugeConfig(theta_coupling=2.), FOCCZ4Config(kappa1=0.)
    else:
        g = run_b1.grid(32, 4)
        initial = FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(g, amplitude=.1))
        dt, per_unit, gauge, config = 1/128, 128, run_b1.GAUGE, run_b1.CONFIG
    q = slots(initial)
    rhs = checked_rhs(make_rhs(spacing=g.spacing, order=4, gauge=gauge, config=config, harmonic=args.case == "b1", threads=1))
    rows, start, completed_steps = [], time.perf_counter(), 0
    result = dict(case=args.case, target=args.final_time, shape=list(g.shape), dt=dt, threads=1, backend="native", status="RUNNING", rows=rows)
    if args.case == "s2":
        rows.append(dict(time=0., rms=rms(initial)))
    reference = json.loads(Path("docs/b1-gauge-wave/long-a0.1.json").read_text())["samples"] if args.case == "b1" else []

    def save():
        result.update(seconds=time.perf_counter()-start, last_completed_step=completed_steps, last_completed_time=completed_steps*dt, peak_rss_bytes=peak_rss())
        if args.case == "s2":
            result["windows"] = windows(rows)
        encoded = json.dumps(result, indent=1, allow_nan=False) + "\n"
        if len(encoded.encode()) > 64*1024*1024:
            raise RuntimeError("64 MiB output budget exceeded")
        args.out.write_text(encoded, encoding="utf-8")

    try:
        for unit in range(1, args.final_time+1):
            for _ in range(per_unit):
                if time.perf_counter()-start >= 600:
                    raise TimeoutError("600 second runtime budget exhausted")
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    candidate = rk4(q, dt, 1, rhs)
                    validate_slots(candidate)
                q = candidate
                completed_steps += 1
            if args.case == "s2":
                state = state_from_slots(q, g)
                rows.append(dict(time=float(unit), rms=rms(state), constraints={k: float(v) for k,v in constraint_report_3d(state.base).norms.items()}))
            else:
                e = max(float(np.max(np.abs(base_fields(q)[f] - getattr(BSSNState3D.gauge_wave(g, amplitude=.1, time=float(unit)), f)))) for f in run_b1.FIELDS)/.1
                row = dict(time=float(unit), error=e)
                if unit <= len(reference):
                    row["python_error"] = reference[unit-1][1]
                    row["relative_difference"] = abs(e-row["python_error"])/abs(row["python_error"])
                rows.append(row)
            save()
            if result["peak_rss_bytes"] > 512*1024*1024:
                raise RuntimeError("512 MiB diagnostic process budget exceeded")
            if unit % 25 == 0 or args.case == "b1":
                print(json.dumps(dict(time=unit, seconds=round(result["seconds"], 2))), flush=True)
        result["status"] = "COMPLETED"
    except (ScientificFailure, FloatingPointError, np.linalg.LinAlgError) as exc:
        result.update(status="SCIENTIFIC_FAILURE", failure=str(exc), failed_step=completed_steps+1, failed_interval=[completed_steps*dt,(completed_steps+1)*dt])
    except (TimeoutError, RuntimeError) as exc:
        result.update(status="RESOURCE_STOPPED", failure=str(exc))
    if args.case == "b1":
        differences = [r["relative_difference"] for r in rows if "relative_difference" in r]
        result["worst_common_relative_difference"] = max(differences) if differences else None
        result["external_independent_reproduction"] = False
    save()
    print(json.dumps({k: v for k,v in result.items() if k not in ("rows", "windows")}), flush=True)


if __name__ == "__main__":
    main()

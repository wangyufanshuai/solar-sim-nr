"""S3 offline saved-state errors, discrete constraints and spatial Fourier energy.

No solver advancement and no replacement for a physical validation gate.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.b1_gauge_wave import run_b1
from experiments.b3_native.long_diagnostics import peak_rss
from kernel.native_evolve import state_from_slots
from science.numerical_relativity import BSSNState3D
from science.numerical_relativity.constraints3d import constraint_report_3d
from science.numerical_relativity.fo_ccz4 import build_fo_ccz4_auxiliary_3d

AMPLITUDE = .1
TIMES = (0., 15., 18., 22.)
CONSTRAINTS = ("hamiltonian", "momentum", "det_h_minus_one", "trace_a",
               "conformal_gamma", "theta")
AUXILIARIES = ("lapse_gradient", "shift_gradient", "metric_gradient",
               "conformal_factor_gradient")


def norms(values):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("finite diagnostic arrays required")
    return dict(l_inf=float(np.max(np.abs(values))),
                rms=float(np.sqrt(np.mean(values*values))))


def spectrum(values):
    """Mean-square x energy, averaged over y,z and all tensor components.

    Transform each transverse/component line separately; do not average away
    transverse errors before transforming. Positive-frequency bins represent
    both signs, except DC and the even-grid Nyquist bin.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 3 or min(values.shape[:3]) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("finite arrays with three spatial axes required")
    nx = values.shape[0]
    transformed = np.fft.rfft(values, axis=0)/nx
    weights = np.full(nx//2+1, 2.)
    weights[0] = 1.
    if nx % 2 == 0:
        weights[-1] = 1.
    power = np.mean(np.abs(transformed)**2, axis=tuple(range(1, values.ndim)))*weights
    total = float(np.sum(power))
    real_energy = float(np.mean(values**2))
    parseval_error = abs(total-real_energy)/max(real_energy, np.finfo(np.float64).tiny)
    if parseval_error > 1e-12:
        raise ValueError("Parseval energy check failed")
    transverse = float(np.mean((values-np.mean(values, axis=(1, 2), keepdims=True))**2))
    k = np.arange(power.size)
    def fraction(energy):
        return float(energy/total) if total else 0.
    return dict(k=k.tolist(), x_power=power.tolist(), mean_square=real_energy,
                parseval_relative_error=parseval_error, dc_fraction=fraction(power[0]),
                fundamental_fraction=fraction(power[1]),
                nyquist_fraction=fraction(power[-1]) if nx % 2 == 0 else None,
                high_k_minimum=nx/4, high_k_fraction=fraction(np.sum(power[k >= nx/4])),
                transverse_mean_square=transverse, transverse_fraction=fraction(transverse))


def inspect_state(state, exact):
    """Compare a state to same-grid analytic data, including discrete controls."""
    raw, fields, constraints, auxiliary = {}, {}, {}, {}
    for name in run_b1.FIELDS:
        delta = getattr(state.base, name)-getattr(exact, name)
        raw["field_error_"+name] = delta
        fields[name] = dict(**norms(delta), l_inf_over_amplitude=norms(delta)["l_inf"]/AMPLITUDE,
                            error_spectrum=spectrum(delta))
    evolved_report, exact_report = constraint_report_3d(state.base), constraint_report_3d(exact)
    for name in CONSTRAINTS:
        evolved, reference = getattr(evolved_report, name), getattr(exact_report, name)
        delta = evolved-reference
        for label, array in (("evolved", evolved), ("analytic_discrete", reference), ("difference", delta)):
            raw[f"constraint_{name}_{label}"] = array
        constraints[name] = dict(evolved=norms(evolved), analytic_discrete=norms(reference),
                                 difference=norms(delta), difference_spectrum=spectrum(delta))
    rebuilt = build_fo_ccz4_auxiliary_3d(state.base)
    for name in AUXILIARIES:
        actual, reference = getattr(state.auxiliary, name), getattr(rebuilt, name)
        delta = actual-reference
        raw["auxiliary_difference_"+name] = delta
        auxiliary[name] = dict(evolved=norms(actual), rebuilt_from_own_base=norms(reference),
                               difference=norms(delta), difference_spectrum=spectrum(delta))
    return dict(fields=fields, constraints=constraints, auxiliary_consistency=auxiliary), raw


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as saved:
        q, current, dt = saved["q"].copy(), float(saved["time"]), float(saved["dt"])
    if q.dtype != np.float64 or q.shape != (58, 32, 5, 5) or not np.all(np.isfinite(q)):
        raise ValueError("expected finite binary64 58x32x5x5 snapshot")
    if current not in TIMES or dt != 1/128:
        raise ValueError("expected registered time in (0,15,18,22) and dt=1/128")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        state = state_from_slots(q, run_b1.grid(32, 4))
    if np.any(state.base.lapse <= 0) or np.any(state.base.chi <= 0):
        raise ValueError("strictly positive reconstructed lapse/chi required")
    return state, current, dt


def run(out, snapshots):
    snapshots = tuple(Path(p) for p in snapshots)
    if len(snapshots) != 4:
        raise ValueError("exactly four registered snapshots required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    def check():
        if time.perf_counter()-start >= 600 or peak_rss() > 512*1024**2:
            raise RuntimeError("600 s or 512 MiB diagnostic ceiling")
    def write(name, data):
        check()
        if sum(p.stat().st_size for p in out.iterdir())+len(data) > 64*1024**2:
            raise RuntimeError("64 MiB diagnostic output ceiling")
        (out/name).write_bytes(data)
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=5).strip()
    clean = not subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True, timeout=5).strip()
    result = dict(source_commit=commit, source_tree_clean=clean, python_version=sys.version,
                  numpy_version=np.__version__, status="DIAGNOSTIC_COMPLETE",
                  scientific_acceptance_established=False, external_independent_reproduction=False,
                  settings=dict(shape=[32,5,5], amplitude=AMPLITUDE, spatial_order=4, dt=1/128,
                                max_seconds=600, max_working_set_bytes=512*1024**2,
                                max_output_bytes=64*1024**2, evolution_steps=0), snapshots=[])
    seen = set()
    for path in snapshots:
        check()
        state, current, dt = load_snapshot(path)
        if current in seen:
            raise ValueError("duplicate snapshot time")
        seen.add(current)
        exact = BSSNState3D.gauge_wave(state.base.grid, amplitude=AMPLITUDE, time=current)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            row, raw = inspect_state(state, exact)
        filename = f"t{int(current)}-arrays.npz"
        payload = io.BytesIO()
        np.savez_compressed(payload, time=current, dt=dt, **raw)
        write(filename, payload.getvalue())
        result["snapshots"].append(dict(time=current, input_path=path.as_posix(),
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), arrays_path=filename, **row))
    result["snapshots"].sort(key=lambda row: row["time"])
    result.update(seconds=time.perf_counter()-start, peak_rss_bytes=peak_rss())
    write("result.json", (json.dumps(result, indent=2, allow_nan=False)+"\n").encode())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, nargs=4, default=[
        Path(f"docs/gauge-stage/baseline-t{t}.npz") for t in (0,15,18,22)])
    args = parser.parse_args()
    result = run(args.out, args.snapshots)
    print(json.dumps({key:result[key] for key in ("status", "source_commit", "seconds", "peak_rss_bytes")}))

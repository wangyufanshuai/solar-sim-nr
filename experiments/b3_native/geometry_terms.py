"""S3 same-input geometry localization; no solver replacement or integration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from experiments.b1_gauge_wave import run_b1
from experiments.b1_gauge_wave.harmonic import harmonic_fo_ccz4_rhs
from experiments.b3_native.gauge_stage_diagnostic import compare_arrays, pack_rhs
from experiments.b3_native.long_diagnostics import peak_rss
from experiments.b3_native.theta_terms import clone_with_globals, adjugate_inverse, python_terms
from kernel.native_evolve import harmonic_correction, state_from_slots
from kernel.native_rhs import GEOMETRY_FIELDS, native_rhs_slots
from science.numerical_relativity.fo_ccz4_rhs import _geometry, _fo_ccz4_volume_rhs_3d, fo_ccz4_rhs_3d


def reciprocal_derivative(values, state, axis):
    """Native periodic stencil: multiply the numerator by its rounded reciprocal."""
    grid = state.grid
    if grid.boundary != "periodic" or grid.spatial_order not in (2, 4):
        raise ValueError("reciprocal ablation supports periodic FD2/FD4 only")
    delta = np.roll(values, -1, axis=axis)-np.roll(values, 1, axis=axis)
    if grid.spatial_order == 4:
        delta = 8.0*delta-(np.roll(values, -2, axis=axis)-np.roll(values, 2, axis=axis))
    factor = 12.0 if grid.spatial_order == 4 else 2.0
    return delta*(1.0/(factor*grid.spacing[axis]))


def python_geometry(q, grid, *, inverse="numpy", derivative="division"):
    """Observe actual Python geometry locals; isolate the optional inverse ablation."""
    if sys.getprofile() is not None:
        raise RuntimeError("geometry observer requires a process without a profiler")
    if inverse not in ("numpy", "adjugate"):
        raise ValueError("unknown inverse variant")
    if derivative not in ("division", "reciprocal"):
        raise ValueError("unknown derivative variant")
    captured = {}
    def observe(frame, event, value):
        if event == "return" and frame.f_code is _geometry.__code__:
            captured.update(frame.f_locals)
    evaluate = _geometry
    if inverse == "adjugate":
        class NumpyView:
            class linalg:
                inv = staticmethod(adjugate_inverse)
            def __getattr__(self, name):
                return getattr(np, name)
        evaluate = clone_with_globals(_geometry, np=NumpyView())
    sys.setprofile(observe)
    try:
        kwargs = {"first_derivative": reciprocal_derivative} if derivative == "reciprocal" else {}
        evaluate(state_from_slots(q, grid), **kwargs)
    finally:
        sys.setprofile(None)
    if not captured:
        raise RuntimeError("Python geometry was not observed")
    captured["d_gamma_hat"] = np.stack(captured["d_gamma_hat"], axis=-2)
    return {name: captured[name] for name, _ in GEOMETRY_FIELDS}


def arithmetic_ablation_rhs():
    """Isolate native inverse/stencil arithmetic in a cloned complete Python RHS."""
    class NumpyView:
        class linalg:
            inv = staticmethod(adjugate_inverse)
        def __getattr__(self, name):
            return getattr(np, name)
    geometry = clone_with_globals(_geometry, np=NumpyView())
    volume = clone_with_globals(_fo_ccz4_volume_rhs_3d, _geometry=geometry)
    volume.__kwdefaults__ = {**volume.__kwdefaults__, "first_derivative": reciprocal_derivative}
    public = clone_with_globals(fo_ccz4_rhs_3d, _fo_ccz4_volume_rhs_3d=volume)
    return clone_with_globals(harmonic_fo_ccz4_rhs, fo_ccz4_rhs_3d=public)


def scalar_contraction(fields, chi, *, sequential):
    """chi * h^ij (R_ij + nabla_i Z_j + nabla_j Z_i), from saved tensors."""
    h, rz = fields["h_inverse"], fields["ricci_z"]
    if not sequential:
        return chi * np.einsum("...ij,...ij->...", h, rz)
    value = np.zeros_like(chi)
    for i in range(3):
        for j in range(3):
            value += h[..., i, j] * rz[..., i, j]
    return chi * value


def tensor_report(reference, native):
    report = []
    for name, _ in GEOMETRY_FIELDS:
        ref, actual = reference[name], native[name]
        if ref.shape != actual.shape or not np.all(np.isfinite(ref)) or not np.all(np.isfinite(actual)):
            raise ValueError("matching finite geometry tensors required: " + name)
        delta = actual - ref
        worst = tuple(int(i) for i in np.unravel_index(np.argmax(np.abs(delta)), delta.shape))
        scale = float(np.max(np.abs(ref)))
        absolute = float(np.max(np.abs(delta)))
        report.append(dict(field=name, reference_max=scale, absolute_difference=absolute,
                           relative_to_tensor_max=absolute/scale if scale else None,
                           changed_elements=int(np.count_nonzero(delta)), worst_index=list(worst),
                           reference_at_worst=float(ref[worst]), native_at_worst=float(actual[worst])))
    return report


def contraction_report(reference, native, chi):
    ref = reference["ricci_z_scalar"]
    actual = native["ricci_z_scalar"]
    seq = scalar_contraction(reference, chi, sequential=True)
    numpy_native = scalar_contraction(native, chi, sequential=False)
    return dict(original_absolute_difference=float(np.max(np.abs(actual-ref))),
                python_tensors_sequential_vs_native=float(np.max(np.abs(actual-seq))),
                python_reduction_order_shift=float(np.max(np.abs(seq-ref))),
                native_tensors_numpy_vs_python=float(np.max(np.abs(numpy_native-ref))),
                python_numpy_reconstruction_error=float(np.max(np.abs(scalar_contraction(reference, chi, sequential=False)-ref))),
                native_sequential_reconstruction_error=float(np.max(np.abs(scalar_contraction(native, chi, sequential=True)-actual))))


def snapshot_stage_inputs(path, grid):
    """Sample ordinary native RK4 inputs from one saved state; do not evolve it."""
    with np.load(path, allow_pickle=False) as saved:
        q, base_time, dt = saved["q"].copy(), float(saved["time"]), float(saved["dt"])
    if q.dtype != np.float64 or q.shape != (58, *grid.shape) or not np.all(np.isfinite(q)):
        raise ValueError("expected finite binary64 58-slot snapshot on the registered grid")
    if not np.isfinite(base_time) or base_time < 0 or dt != 1/128:
        raise ValueError("expected finite nonnegative time and registered dt=1/128")
    kw = dict(spacing=grid.spacing, order=4, gauge=run_b1.GAUGE, config=run_b1.CONFIG, threads=1)
    def ordinary(state):
        raw = native_rhs_slots(state, **kw)
        return harmonic_correction(state, raw, spacing=grid.spacing, order=4,
                                   gauge=run_b1.GAUGE, config=run_b1.CONFIG)
    inputs, controls = [q], []
    for factor in (.5, .5, 1.):
        controls.append(ordinary(inputs[-1]))
        inputs.append(q+factor*dt*controls[-1])
    controls.append(ordinary(inputs[-1]))
    return np.stack(inputs), np.stack(controls), base_time, dt


def run(out, before=None, *, snapshot=None):
    if (before is None) == (snapshot is None):
        raise ValueError("provide exactly one before archive or snapshot")
    out = Path(out)
    source = Path(snapshot if snapshot is not None else before)
    start = time.perf_counter()
    grid = run_b1.grid(32, 4)
    if snapshot is None:
        with np.load(source, allow_pickle=False) as saved:
            states, original = saved["states"].copy(), saved["rhs"].copy()
        base_time, dt = 0., 1/128
    else:
        states, original, base_time, dt = snapshot_stage_inputs(source, grid)
    out.mkdir(parents=True, exist_ok=False)
    if states.shape != (4, 58, *grid.shape) or original.shape != states.shape:
        raise ValueError("expected four registered 58-slot RK inputs and RHS arrays")
    result = dict(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip(),
                  input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  input_path=source.as_posix(), base_time=base_time, dt=dt,
                  input_kind="archived_four_stage_inputs" if snapshot is None else "saved_native_snapshot",
                  control_kind="archived_old_library_rhs" if snapshot is None else "fresh_ordinary_native_rhs",
                  external_independent_reproduction=False, scientific_gate_relative=1e-12,
                  settings=dict(shape=list(grid.shape), amplitude=.1, threads=1, max_seconds=600,
                                max_output_bytes=64*1024**2, max_working_set_bytes=512*1024**2),
                  numpy_version=np.__version__, python_version=sys.version, stages=[])
    if snapshot is None:
        result["before_sha256"] = result["input_sha256"]
    for stage, (q, old) in enumerate(zip(states, original), 1):
        if time.perf_counter()-start > 600 or peak_rss() > 512*1024**2:
            raise RuntimeError("geometry diagnostic resource ceiling")
        kw = dict(spacing=grid.spacing, order=4, gauge=run_b1.GAUGE, config=run_b1.CONFIG, threads=1)
        raw, actual = native_rhs_slots(q, return_geometry=True, **kw)
        normal = native_rhs_slots(q, **kw)
        same_normal = raw.tobytes() == normal.tobytes()
        rhs = harmonic_correction(q, raw, spacing=grid.spacing, order=4, gauge=run_b1.GAUGE, config=run_b1.CONFIG)
        py_rhs, py_terms = python_terms(q, grid)
        ref = python_geometry(q, grid)
        adj = python_geometry(q, grid, inverse="adjugate")
        reciprocal = python_geometry(q, grid, derivative="reciprocal")
        combined = python_geometry(q, grid, inverse="adjugate", derivative="reciprocal")
        evaluate_ablation = arithmetic_ablation_rhs()
        combined_rhs = pack_rhs(evaluate_ablation(state_from_slots(q, grid), gauge=run_b1.GAUGE,
                                                  config=run_b1.CONFIG))
        restored_rhs, restored_terms = python_terms(q, grid)
        after = python_geometry(q, grid)
        chi = state_from_slots(q, grid).base.chi
        row = dict(stage=stage, base_time=base_time,
                   nominal_stage_time=base_time+(0., .5, .5, 1.)[stage-1]*dt,
                   all_58_outputs_bitwise_equal_normal=same_normal,
                   all_58_outputs_bitwise_equal_control=rhs.tobytes()==old.tobytes(),
                   original_python_unchanged=all(ref[n].tobytes()==after[n].tobytes() for n, _ in GEOMETRY_FIELDS)
                     and py_rhs.tobytes()==restored_rhs.tobytes() and py_terms.tobytes()==restored_terms.tobytes(),
                   scalar_equals_theta_observation=ref["ricci_z_scalar"].tobytes()==py_terms[6].tobytes(),
                   strict_rhs_comparison=compare_arrays(py_rhs, rhs),
                   arithmetic_rhs_comparison=compare_arrays(combined_rhs, rhs),
                   tensors=tensor_report(ref, actual), adjugate_tensors=tensor_report(adj, actual),
                   reciprocal_tensors=tensor_report(reciprocal, actual),
                   combined_tensors=tensor_report(combined, actual),
                   contraction=contraction_report(ref, actual, chi),
                   adjugate_contraction=contraction_report(adj, actual, chi),
                   combined_geometry_bitwise_equal_native=all(combined[n].tobytes()==actual[n].tobytes()
                                                              for n, _ in GEOMETRY_FIELDS),
                   combined_contraction=contraction_report(combined, actual, chi))
        if snapshot is None:
            row["all_58_outputs_bitwise_equal_before"] = row["all_58_outputs_bitwise_equal_control"]
        result["stages"].append(row)
        np.savez_compressed(out/f"stage-{stage}.npz", q=q, python_rhs=py_rhs, native_rhs=rhs,
                            arithmetic_rhs=combined_rhs, **{f"{variant}_{n}": v
                            for variant, fields in (("python", ref), ("native", actual), ("adjugate", adj),
                                                    ("reciprocal", reciprocal), ("combined", combined))
                            for n, v in fields.items()})
        if sum(p.stat().st_size for p in out.iterdir()) > 63*1024**2:
            raise RuntimeError("64 MiB output ceiling (report space reserved)")
    result.update(seconds=time.perf_counter()-start, peak_rss_bytes=peak_rss())
    result["instrumentation_pass"] = all(all(row[key] for key in (
        "all_58_outputs_bitwise_equal_normal", "all_58_outputs_bitwise_equal_control",
        "original_python_unchanged", "scalar_equals_theta_observation")) for row in result["stages"])
    result["strict_rhs_pass"] = all(row["strict_rhs_comparison"]["active_slots_pass_relative_1e_minus12"] for row in result["stages"])
    result["diagnostic_arithmetic_rhs_pass"] = all(row["arithmetic_rhs_comparison"]["active_slots_pass_relative_1e_minus12"]
                                                  for row in result["stages"])
    text = json.dumps(result, indent=2, allow_nan=False)+"\n"
    if (result["seconds"] > 600 or result["peak_rss_bytes"] > 512*1024**2
        or sum(p.stat().st_size for p in out.iterdir())+len(text.encode()) > 64*1024**2):
        raise RuntimeError("geometry diagnostic resource ceiling")
    (out/"result.json").write_text(text, encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--before", type=Path)
    inputs.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    result = run(args.out, args.before, snapshot=args.snapshot)
    print(json.dumps({k: v for k, v in result.items() if k != "stages"}))
    # This exit code qualifies the observer only; inspect strict_rhs_pass separately.
    sys.exit(0 if result["instrumentation_pass"] else 2)

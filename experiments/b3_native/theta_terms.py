"""Same-state Theta term isolation; no integration or scientific gate replacement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import hashlib
import subprocess
import time
import numpy as np

from experiments.b1_gauge_wave import run_b1
from experiments.b1_gauge_wave.harmonic import harmonic_fo_ccz4_rhs
from experiments.b3_native.gauge_stage_diagnostic import compare_arrays, pack_rhs
from experiments.b3_native.long_diagnostics import peak_rss
from kernel.native_evolve import harmonic_correction, slots, state_from_slots
from kernel.native_rhs import THETA_TERM_NAMES, native_rhs_slots
from science.numerical_relativity import BSSNState3D, FOCCZ4State3D
from science.numerical_relativity.fo_ccz4_rhs import _fo_ccz4_volume_rhs_3d


def python_terms(q, grid):
    """Observe actual production intermediates, keeping the frozen RHS source unchanged."""
    if sys.getprofile() is not None:
        raise RuntimeError("term observer requires a process without a profiler")
    captured = {}

    def observe(frame, event, value):
        if event == "return" and frame.f_code is _fo_ccz4_volume_rhs_3d.__code__:
            captured.update(frame.f_locals)

    sys.setprofile(observe)
    try:
        result = harmonic_fo_ccz4_rhs(state_from_slots(q, grid), gauge=run_b1.GAUGE,
                                     config=run_b1.CONFIG)
    finally:
        sys.setprofile(None)
    if not captured:
        raise RuntimeError("production volume RHS was not observed")
    alpha, base, aux = captured["alpha"], captured["base"], captured["aux"]
    scalar, z = captured["scalar"], captured["z_upper"]
    cfg = run_b1.CONFIG
    aa = np.einsum("...ij,...ij->...", base.a, captured["a_upup"])
    za = np.einsum("...i,...i->...", z, aux.lapse_gradient)
    terms = np.stack([
        sum(base.shift[..., a] * captured["d_theta"][a] for a in range(3)),
        .5 * alpha * cfg.cleaning_speed**2 * scalar,
        .5 * alpha * cfg.cleaning_speed**2 * (-aa + (2. / 3.) * base.trace_k**2),
        -alpha * cfg.c * base.theta * base.trace_k,
        -alpha * za,
        -alpha * cfg.kappa1 * (2. + cfg.kappa2) * base.theta,
        scalar, aa, za, alpha, base.chi, result.theta,
    ])
    return pack_rhs(result), terms


def sequential_sum(terms):
    total = terms[0].copy()
    for term in terms[1:6]:
        total += term
    return total


def term_report(reference, native):
    if reference.shape != native.shape or reference.shape[0] != len(THETA_TERM_NAMES):
        raise ValueError("expected matching twelve-term arrays")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(native)):
        raise ValueError("finite terms required")
    diff = native - reference
    worst = tuple(int(i) for i in np.unravel_index(np.argmax(np.abs(diff[-1])), diff[-1].shape))
    denominator = abs(float(reference[-1][worst]))
    return {
        "terms": [{"name": name, "reference_max": float(np.max(np.abs(r))),
                   "absolute_difference": float(np.max(np.abs(d)))}
                  for name, r, d in zip(THETA_TERM_NAMES, reference, diff)],
        "worst_theta_index": list(worst),
        "worst_theta_difference": float(diff[-1][worst]),
        "signed_term_differences_at_worst": {name: float(d[worst])
                                               for name, d in zip(THETA_TERM_NAMES[:6], diff[:6])},
        "cancellation_ratio_at_worst": float(np.sum(np.abs(reference[:6][(slice(None), *worst)])))
                                       / denominator if denominator else None,
        "python_reconstruction_error": float(np.max(np.abs(sequential_sum(reference) - reference[-1]))),
        "native_reconstruction_error": float(np.max(np.abs(sequential_sum(native) - native[-1]))),
        "term_delta_reconstruction_error": float(np.max(np.abs(np.sum(diff[:6], axis=0) - diff[-1]))),
    }


def run(out, before=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    g = run_b1.grid(32, 4)
    q = slots(FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(g, amplitude=.1)))
    original = None
    if before is not None:
        with np.load(before, allow_pickle=False) as saved:
            original = {key: saved[key].copy() for key in ("states", "rhs")}
    data = {"settings": {"shape": list(g.shape), "amplitude": .1, "dt": 1/128,
                         "threads": 1, "max_seconds": 600, "max_output_bytes": 64*1024**2},
            "external_independent_reproduction": False,
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip(),
            "numpy_version": np.__version__, "python_version": sys.version,
            "before_comparison_performed": before is not None,
            "before_sha256": hashlib.sha256(Path(before).read_bytes()).hexdigest() if before else None,
            "stages": []}
    k = []
    for stage in range(4):
        if time.perf_counter() - start > 600:
            raise TimeoutError("600 second limit")
        stage_q = q if stage == 0 else q + ([.5, .5, 1.][stage-1]/128) * k[-1]
        raw, terms = native_rhs_slots(stage_q, spacing=g.spacing, order=4, gauge=run_b1.GAUGE,
                                     config=run_b1.CONFIG, threads=1, return_theta_terms=True)
        normal = native_rhs_slots(stage_q, spacing=g.spacing, order=4, gauge=run_b1.GAUGE,
                                  config=run_b1.CONFIG, threads=1)
        same_normal = raw.tobytes() == normal.tobytes()
        native = harmonic_correction(stage_q, raw, spacing=g.spacing, order=4,
                                     gauge=run_b1.GAUGE, config=run_b1.CONFIG)
        expected, reference = python_terms(stage_q, g)
        row = {"stage": stage+1, "all_58_diagnostic_outputs_bitwise_equal_normal":
               same_normal,
               "instrumented_theta_equals_rhs": terms[-1].tobytes() == native[18].tobytes(),
               "comparison": compare_arrays(expected, native), "isolation": term_report(reference, terms)}
        if original is not None:
            row["all_58_outputs_bitwise_equal_before"] = native.tobytes() == original["rhs"][stage].tobytes()
            row["input_bitwise_equal_before"] = stage_q.tobytes() == original["states"][stage].tobytes()
        data["stages"].append(row)
        k.append(native)
        np.savez_compressed(out/f"stage-{stage+1}.npz", python_terms=reference, native_terms=terms)
    data.update(seconds=time.perf_counter()-start, peak_rss_bytes=peak_rss(),
                strict_rhs_pass=all(r["comparison"]["active_slots_pass_relative_1e_minus12"] for r in data["stages"]))
    data["instrumentation_pass"] = all(r["all_58_diagnostic_outputs_bitwise_equal_normal"]
        and r["instrumented_theta_equals_rhs"] and r.get("all_58_outputs_bitwise_equal_before", True)
        and r.get("input_bitwise_equal_before", True) for r in data["stages"])
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    if sum(p.stat().st_size for p in out.iterdir()) + len(text.encode()) > 64*1024**2:
        raise RuntimeError("64 MiB output limit")
    if data["peak_rss_bytes"] > 512*1024**2:
        raise RuntimeError("512 MiB working set exceeded")
    (out/"result.json").write_text(text, encoding="utf-8")
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--before", type=Path)
    args = ap.parse_args()
    result = run(args.out, args.before)
    print(json.dumps({key: value for key, value in result.items() if key != "stages"}))
    sys.exit(0 if result["instrumentation_pass"] else 2)

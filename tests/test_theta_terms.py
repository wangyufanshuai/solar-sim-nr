import sys
import numpy as np
import pytest

from experiments.b1_gauge_wave import run_b1
from experiments.b3_native.theta_terms import python_terms, sequential_sum, term_report
from kernel.native_evolve import slots
from kernel.native_rhs import THETA_TERM_NAMES, native_rhs_slots
from science.numerical_relativity import BSSNState3D, FOCCZ4State3D


def test_native_terms_preserve_all_slots_and_reconstruct_theta():
    g = run_b1.grid(8, 4)
    base = BSSNState3D.gauge_wave(g, amplitude=.1)
    # Exercise transport, damping and Theta*K, which vanish in the initial wave.
    base.shift[..., 0] = .02
    base.theta[:] = .01*np.sin(2*np.pi*g.coordinates[0])
    q = slots(FOCCZ4State3D.from_bssn_state(base))
    from dataclasses import replace
    cfg = replace(run_b1.CONFIG, kappa1=.1, kappa2=.2)
    kw = dict(spacing=g.spacing, order=4, gauge=run_b1.GAUGE, config=cfg, threads=1)
    normal = native_rhs_slots(q, **kw)
    observed, terms = native_rhs_slots(q, return_theta_terms=True, **kw)
    assert observed.tobytes() == normal.tobytes()
    assert terms.shape == (len(THETA_TERM_NAMES), *g.shape)
    assert terms[-1].tobytes() == normal[18].tobytes()
    np.testing.assert_array_equal(sequential_sum(terms), normal[18])
    assert all(np.max(np.abs(terms[i])) > 0 for i in (0, 1, 2, 3, 4, 5))


def test_python_observer_reconstructs_actual_theta_and_keeps_input():
    g = run_b1.grid(8, 4)
    q = slots(FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(g, amplitude=.1)))
    before = q.tobytes()
    rhs, terms = python_terms(q, g)
    assert sys.getprofile() is None
    assert q.tobytes() == before
    np.testing.assert_array_equal(terms[-1], rhs[18])
    np.testing.assert_array_equal(sequential_sum(terms), rhs[18])


def test_observer_refuses_to_replace_an_existing_profiler():
    observer = lambda *args: None
    sys.setprofile(observer)
    try:
        with pytest.raises(RuntimeError, match="without a profiler"):
            python_terms(None, None)
        assert sys.getprofile() is observer
    finally:
        sys.setprofile(None)


def test_term_report_locates_a_curvature_defect_and_preserves_signed_differences():
    reference = np.zeros((12, 2, 2, 2))
    reference[1] = 2.; reference[2] = -1.; reference[-1] = 1.
    native = reference.copy()
    native[1, 1, 0, 1] += .25; native[-1, 1, 0, 1] += .25
    report = term_report(reference, native)
    assert report["worst_theta_index"] == [1, 0, 1]
    assert report["signed_term_differences_at_worst"]["curvature"] == .25
    assert report["cancellation_ratio_at_worst"] == 3
    assert report["native_reconstruction_error"] == 0
    assert report["term_delta_reconstruction_error"] == 0
    with pytest.raises(ValueError, match="finite"):
        term_report(reference, native*np.nan)

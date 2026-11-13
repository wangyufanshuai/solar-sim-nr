import numpy as np
from experiments.b3_native.gauge_stage_diagnostic import compare_arrays, compare_stages, pack_rhs
from experiments.b1_gauge_wave import run_b1
from experiments.b1_gauge_wave.harmonic import harmonic_fo_ccz4_rhs
from kernel.native_evolve import slots, make_rhs
from science.numerical_relativity import BSSNState3D, FOCCZ4State3D


def test_same_state_harmonic_stages_match_without_mutating_input():
    g=run_b1.grid(8,4)
    state=FOCCZ4State3D.from_bssn_state(BSSNState3D.gauge_wave(g,amplitude=.1))
    q=slots(state); before=q.copy()
    rhs=make_rhs(spacing=g.spacing,order=4,gauge=run_b1.GAUGE,config=run_b1.CONFIG,harmonic=True,threads=1)
    comparisons=compare_stages(q,g,1/128,rhs)
    assert len(comparisons)==4
    assert all(c["active_slots_pass_relative_1e_minus12"] for c in comparisons)
    np.testing.assert_array_equal(q,before)
    packed=pack_rhs(harmonic_fo_ccz4_rhs(state,gauge=run_b1.GAUGE,config=run_b1.CONFIG))
    assert packed.shape==(58,*g.shape)


def test_wrong_trace_k_rhs_is_detected_in_canonical_slot():
    reference=np.broadcast_to(np.arange(1,59)[:,None,None,None],(58,5,5,5)).copy().astype(float)
    wrong=reference.copy(); wrong[17]*=-1
    result=compare_arrays(reference,wrong)
    assert not result["active_slots_pass_relative_1e_minus12"]
    assert result["worst_active_relative_difference"]==2
    assert result["slots"][17]["slot"]=="K"


def test_inactive_zero_rhs_is_reported_without_division_by_zero():
    reference=np.zeros((58,5,5,5)); reference[17]=1
    native=reference.copy(); native[25]=1e-16
    result=compare_arrays(reference,native)
    assert result["slots"][25]["reference_max"]==0
    assert result["slots"][25]["absolute_difference"]==1e-16
    assert result["slots"][25]["relative_difference"] is None
    assert result["active_slots_pass_relative_1e_minus12"]

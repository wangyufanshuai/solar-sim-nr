import numpy as np
import pytest
from pathlib import Path
from experiments.b3_native.gauge_restart import CheckedRHS, ResourceStopped, difference, run
from experiments.b3_native.long_diagnostics import ScientificFailure


def test_stage_failure_keeps_input_and_resource_failure_is_distinct():
    q = np.zeros((58,5,5,5)); q[0:3] = 1
    calls = []
    def evaluate(state):
        calls.append(1)
        if len(calls) == 2:
            raise FloatingPointError("deliberate overflow")
        return np.zeros_like(state)
    checked = CheckedRHS(evaluate, lambda: None)
    checked(q)
    with pytest.raises(ScientificFailure, match="RK4 stage 2"):
        checked(q)
    assert checked.failed_stage == 2 and checked.failed_input.tobytes() == q.tobytes()
    def stop():
        raise ResourceStopped("deadline")
    with pytest.raises(ResourceStopped):
        CheckedRHS(evaluate, stop)(q)


def test_restart_preserves_snapshot_and_compares_same_time(tmp_path):
    source = Path(__file__).resolve().parents[1]/"docs/gauge-stage/baseline-t22.npz"
    original = source.read_bytes()
    result = run(tmp_path/"run",source,steps=2)
    assert result["status"] == "COMPLETED" and result["all_trajectories_reached_target"]
    assert result["terminal_states"]["python"]["last_valid_time"] == 22+2/128
    assert "max_slot_difference" in result["terminal_states"]["python"]
    assert source.read_bytes() == original
    assert result["samples"][0]["time"] == 22
    assert result["samples"][-1]["time"] == 22+2/128
    assert result["samples"][0]["trajectories"]["python"]["max_slot_difference"] == 0
    assert result["perturbation"]["delta"] > 0
    assert not result["external_independent_reproduction"]
    with np.load(tmp_path/"run/native-last-valid.npz") as z:
        assert difference(z["q"],z["q"])["base_field_difference_over_amplitude"] == 0


def test_resource_stopped_restart_cannot_be_reported_as_scientific_completion(tmp_path):
    source = Path(__file__).resolve().parents[1]/"docs/gauge-stage/baseline-t22.npz"
    result = run(tmp_path/"stopped",source,steps=2,max_seconds=1e-12)
    assert result["status"] == "RESOURCE_STOPPED"
    assert not result["all_trajectories_reached_target"]
    assert not result["failures"]


def test_terminal_comparison_excludes_stale_native_state():
    from experiments.b3_native.gauge_restart import terminal_states
    from experiments.b1_gauge_wave import run_b1
    with np.load(Path(__file__).resolve().parents[1]/"docs/gauge-stage/baseline-t22.npz") as z:
        q=z["q"].copy()
    result=terminal_states({"native":q,"python":q.copy()}, {"native":0,"python":1},run_b1.grid(32,4),22,1/128)
    assert result["python"]["last_valid_time"] == 22+1/128
    assert "max_slot_difference" not in result["python"]


def test_native_refinement_tracks_substeps_and_shared_sample_times(tmp_path):
    from experiments.b3_native.gauge_restart import run_refinement
    source = Path(__file__).resolve().parents[1]/"docs/gauge-stage/baseline-t22.npz"
    before = source.read_bytes()
    result = run_refinement(tmp_path/"refine",source,coarse_steps=2)
    assert result["status"] == "COMPLETED" and result["all_trajectories_reached_target"]
    assert result["steps"] == {"native":2,"half":4,"quarter":8}
    assert source.read_bytes() == before
    for name,factor in {"native":1,"half":2,"quarter":4}.items():
        with np.load(tmp_path/f"refine/{name}-last-valid.npz") as z:
            assert float(z["time"]) == 22+2/128
            assert float(z["dt"]) == 1/(128*factor)
    last = result["samples"][-1]
    assert last["time"] == 22+2/128
    assert all("difference_from_finest" in r for r in last["trajectories"].values())
    assert last["trajectories"]["quarter"]["difference_from_finest"]["max_slot_difference"] == 0


def test_refinement_keeps_candidate_when_completed_step_reconstruction_fails(tmp_path,monkeypatch):
    import experiments.b3_native.gauge_restart as diagnostic
    source = Path(__file__).resolve().parents[1]/"docs/gauge-stage/baseline-t22.npz"
    original_check = diagnostic.validate_slots
    calls = []
    def fail_first_completed_step(q):
        calls.append(1)
        if len(calls) == 6:  # initial validation, four RHS stages, then candidate validation
            raise ScientificFailure("injected candidate reconstruction failure")
        original_check(q)
    monkeypatch.setattr(diagnostic,"validate_slots",fail_first_completed_step)
    result = diagnostic.run_refinement(tmp_path/"failure",source,coarse_steps=1)
    assert result["status"] == "SCIENTIFIC_FAILURE"
    assert result["failures"]["native"]["phase"] == "completed_step_reconstruction"
    assert result["failures"]["native"]["stage"] is None
    assert (tmp_path/"failure/native-failed-candidate.npz").exists()
    assert not (tmp_path/"failure/native-failed-input.npz").exists()
    with np.load(tmp_path/"failure/native-failed-candidate.npz") as z:
        assert float(z["time"]) == 22+1/128

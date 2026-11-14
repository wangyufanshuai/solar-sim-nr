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

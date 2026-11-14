import numpy as np
import pytest

import experiments.b3_native.gauge_spatial_refinement as diagnostic


def test_short_native_run_samples_exact_endpoint_and_keeps_initial_control(tmp_path):
    result=diagnostic.run(tmp_path/"short",points=(16,),target_time=1/64)
    assert result["status"] == "COMPLETED" and result["all_trajectories_reached_target"]
    row=result["trajectories"][0]
    assert row["completed_steps"] == 1 and row["last_valid_time"] == 1/64
    assert [s["time"] for s in row["samples"]] == [0.,1/64]
    assert max(v["l_inf_over_amplitude"] for v in row["samples"][0]["fields"].values()) == 0
    with np.load(tmp_path/"short"/row["last_valid_arrays"]) as z:
        assert z["q"].shape == (58,16,5,5) and float(z["time"]) == 1/64
    assert result["peak_rss_bytes"] < 512*1024**2


def test_resource_stop_preserves_state_and_skips_later_grids(tmp_path):
    result=diagnostic.run(tmp_path/"stop",points=(32,16,64),target_time=1/16,max_seconds=1e-12)
    assert result["status"] == "RESOURCE_STOPPED"
    assert [r["status"] for r in result["trajectories"]] == ["RESOURCE_STOPPED", "NOT_STARTED_RESOURCE_STOP", "NOT_STARTED_RESOURCE_STOP"]
    assert result["trajectories"][0]["last_valid_time"] == 0
    assert (tmp_path/"stop/nx32-last-valid.npz").exists()
    assert not (tmp_path/"stop/nx16-last-valid.npz").exists()


def test_failed_stage_preserves_input_and_actual_stage_time(tmp_path,monkeypatch):
    calls=[]
    def rhs(q):
        calls.append(1)
        if len(calls) == 3:
            raise FloatingPointError("injected stage-three failure")
        return np.ones_like(q)*1e-8
    monkeypatch.setattr(diagnostic,"make_rhs",lambda **kwargs:rhs)
    result=diagnostic.run(tmp_path/"stage",points=(16,),target_time=1/64)
    row=result["trajectories"][0]
    assert result["status"] == "SCIENTIFIC_FAILURE"
    assert row["completed_steps"] == 0 and row["failure"]["stage"] == 3
    assert row["failure"]["phase"] == "rk_stage" and row["failure"]["failed_stage_time"] == 1/128
    with np.load(tmp_path/"stage/nx16-failed-input.npz") as f, np.load(tmp_path/"stage/nx16-last-valid.npz") as v:
        assert float(f["time"]) == 1/128 and float(v["time"]) == 0
        assert np.max(np.abs(f["q"]-v["q"])) > 0


def test_failed_completed_candidate_is_separate_from_last_valid(tmp_path,monkeypatch):
    def bad_step(q,*args):
        candidate=q.copy()
        candidate[6]=-1000
        return candidate
    monkeypatch.setattr(diagnostic,"rk4",bad_step)
    result=diagnostic.run(tmp_path/"candidate",points=(16,),target_time=1/64)
    row=result["trajectories"][0]
    assert row["failure"]["phase"] == "completed_step_reconstruction"
    assert row["failure"]["stage"] is None
    assert (tmp_path/"candidate/nx16-failed-candidate.npz").exists()
    assert not (tmp_path/"candidate/nx16-failed-input.npz").exists()
    with np.load(tmp_path/"candidate/nx16-failed-candidate.npz") as f, np.load(tmp_path/"candidate/nx16-last-valid.npz") as v:
        assert float(f["time"]) == 1/64 and float(v["time"]) == 0
        assert np.all(f["q"][6] == -1000) and np.all(v["q"][6] > -1)


def test_replay_requires_matching_time_and_keeps_difference(tmp_path):
    path=tmp_path/"replay.npz"
    q=np.ones((58,32,5,5))
    np.savez_compressed(path,q=q,time=15.,dt=1/128)
    assert diagnostic.replay_control(q,path,15.,1/128)["bitwise_equal"]
    changed=q.copy(); changed[0,0,0,0]=1.25
    row=diagnostic.replay_control(changed,path,15.,1/128)
    assert not row["bitwise_equal"] and row["max_slot_difference"] == .25
    with pytest.raises(ValueError,match="match grid"):
        diagnostic.replay_control(q,path,18.,1/128)


@pytest.mark.parametrize("kwargs", [dict(points=(32,32)),dict(points=(8,)),dict(target_time=25.),dict(target_time=.001)])
def test_unregistered_settings_rejected_before_output_creation(tmp_path,kwargs):
    with pytest.raises(ValueError):
        diagnostic.run(tmp_path/"invalid",**kwargs)
    assert not (tmp_path/"invalid").exists()

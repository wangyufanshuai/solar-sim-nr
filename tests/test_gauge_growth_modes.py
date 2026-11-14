from dataclasses import replace

import numpy as np
import pytest

from experiments.b1_gauge_wave import run_b1
from experiments.b3_native.gauge_growth_modes import inspect_state, load_snapshot, run, spectrum
from kernel.native_evolve import slots
from science.numerical_relativity import BSSNState3D, FOCCZ4State3D


def wave(k, nx=32):
    return np.broadcast_to(np.cos(2*np.pi*k*np.arange(nx)/nx)[:,None,None], (nx,5,5))


@pytest.mark.parametrize("k,key,energy", [(0,"dc_fraction",1.), (1,"fundamental_fraction",.5),
                                         (16,"nyquist_fraction",1.), (9,"high_k_fraction",.5)])
def test_known_fourier_modes(k,key,energy):
    report = spectrum(wave(k))
    assert report[key] == pytest.approx(1.)
    assert report["mean_square"] == pytest.approx(energy)
    assert report["x_power"][k] == pytest.approx(energy)
    assert report["parseval_relative_error"] < 1e-14
    assert report["transverse_fraction"] < 1e-28


def test_tensor_components_are_averaged_and_transverse_modes_not_cancelled():
    transverse = np.broadcast_to(np.cos(2*np.pi*np.arange(5)/5)[None,:,None], (32,5,5))
    report = spectrum(transverse)
    assert report["transverse_fraction"] == pytest.approx(1.)
    assert report["dc_fraction"] == pytest.approx(1.)
    assert report["mean_square"] == pytest.approx(.5)
    tensor = np.stack((wave(1),2*wave(16)),axis=-1)
    report = spectrum(tensor)
    assert report["mean_square"] == pytest.approx(2.25)
    assert report["fundamental_fraction"] == pytest.approx(1/9)
    assert report["nyquist_fraction"] == pytest.approx(8/9)


def test_zero_and_odd_grid_energy():
    zero = spectrum(np.zeros((32,5,5)))
    assert zero["mean_square"] == zero["high_k_fraction"] == zero["transverse_fraction"] == 0
    odd = spectrum(wave(15,31))
    assert odd["nyquist_fraction"] is None
    assert odd["x_power"][-1] == pytest.approx(.5)


@pytest.mark.parametrize("array", [np.zeros((32,5)), np.full((32,5,5),np.nan)])
def test_spectrum_rejects_invalid_input(array):
    with pytest.raises(ValueError,match="finite arrays"):
        spectrum(array)


def test_identical_analytic_control_retains_discrete_constraint_baseline():
    exact = BSSNState3D.gauge_wave(run_b1.grid(32,4),amplitude=.1,time=0.)
    state = FOCCZ4State3D.from_bssn_state(exact)
    row, raw = inspect_state(state,exact)
    assert all(v["l_inf"] == 0 for v in row["fields"].values())
    assert all(v["difference"]["l_inf"] == 0 for v in row["constraints"].values())
    assert all(v["difference"]["l_inf"] == 0 for v in row["auxiliary_consistency"].values())
    assert row["constraints"]["hamiltonian"]["analytic_discrete"]["l_inf"] > 0
    np.testing.assert_array_equal(raw["constraint_hamiltonian_evolved"],
                                  raw["constraint_hamiltonian_analytic_discrete"])


def test_constant_auxiliary_offset_is_detected_even_without_a_curl():
    exact = BSSNState3D.gauge_wave(run_b1.grid(32,4),amplitude=.1)
    state = FOCCZ4State3D.from_bssn_state(exact)
    state = replace(state,auxiliary=replace(state.auxiliary,lapse_gradient=state.auxiliary.lapse_gradient+.125))
    row,_ = inspect_state(state,exact)
    assert row["auxiliary_consistency"]["lapse_gradient"]["difference"]["l_inf"] == pytest.approx(.125)
    assert row["auxiliary_consistency"]["lapse_gradient"]["difference_spectrum"]["dc_fraction"] == pytest.approx(1.)
    assert all(v["l_inf"] == 0 for v in row["fields"].values())


def test_snapshot_contract_and_duplicate_time(tmp_path):
    exact = BSSNState3D.gauge_wave(run_b1.grid(32,4),amplitude=.1)
    q = slots(FOCCZ4State3D.from_bssn_state(exact))
    path = tmp_path/"snapshot.npz"
    np.savez_compressed(path,q=q,time=0.,dt=1/128)
    state,t,dt = load_snapshot(path)
    assert t == 0 and dt == 1/128
    np.testing.assert_array_equal(state.base.h,exact.h)
    with pytest.raises(ValueError,match="duplicate"):
        run(tmp_path/"duplicate",[path]*4)
    for values,time,step in ((q.astype(np.float32),0.,1/128), (q[:,:31],0.,1/128),
                             (np.full_like(q,np.nan),0.,1/128), (q,1.,1/128),(q,0.,1/64)):
        np.savez_compressed(path,q=values,time=time,dt=step)
        with pytest.raises(ValueError,match="expected"):
            load_snapshot(path)

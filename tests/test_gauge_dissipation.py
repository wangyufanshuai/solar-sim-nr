import numpy as np
import pytest

from experiments.b3_native.gauge_spatial_refinement import ko6_x, run
from kernel.native_evolve import rk4


def mode(nx,k):
    signal=(-1.)**np.arange(nx) if k == nx//2 else np.cos(2*np.pi*k*np.arange(nx)/nx)
    return np.broadcast_to(signal[None,:,None,None],(58,nx,5,5)).copy()


@pytest.mark.parametrize("nx,k",[(16,1),(32,3),(64,1),(16,8),(32,16),(64,32)])
def test_fourier_sign_and_normalization(nx,k):
    q=mode(nx,k)
    expected=-.1*np.sin(np.pi*k/nx)**6*nx*q
    np.testing.assert_allclose(ko6_x(q,1/nx,.1),expected,rtol=5e-8,atol=2e-13)


def test_constants_are_annihilated_and_zero_strength_is_exact():
    q=np.broadcast_to(np.linspace(.1,5.8,58)[:,None,None,None],(58,32,5,5)).copy()
    assert np.count_nonzero(ko6_x(q,1/32,.1)) == 0
    q[4,12]=100
    assert np.count_nonzero(ko6_x(q,1/32,0.)) == 0


def test_operator_dissipates_energy_and_does_not_mix_slots_or_transverse_axes():
    rng=np.random.default_rng(42)
    q=rng.normal(size=(58,32,5,5))
    assert np.sum(q*ko6_x(q,1/32,.1)) <= 0
    q=np.zeros_like(q); q[3,:,2,4]=mode(32,16)[0,:,0,0]
    actual=ko6_x(q,1/32,.1)
    assert np.count_nonzero(actual)==32
    np.testing.assert_array_equal(actual[3,:,2,4],-3.2*q[3,:,2,4])


def test_fixed_smooth_mode_has_fifth_order_dissipation_strength():
    rates=[np.max(np.abs(ko6_x(mode(n,1),1/n,.1))) for n in (16,32,64)]
    orders=np.log2(np.asarray(rates[:-1])/rates[1:])
    assert np.all(orders>4.9) and np.all(orders<5.01)


@pytest.mark.parametrize("sigma",[.02,.1])
def test_added_scalar_nyquist_mode_rk4_rate(sigma):
    q=mode(32,16); dt=1/128
    z=-sigma*32*dt
    stability=1+z+z*z/2+z**3/6+z**4/24
    actual=rk4(q,dt,1,lambda v:ko6_x(v,1/32,sigma))
    np.testing.assert_allclose(actual,stability*q,rtol=2e-15,atol=0)
    assert 0 < stability < 1


def test_short_comparison_retains_baseline_and_separate_variant_outputs(tmp_path):
    r=run(tmp_path/'short',points=(32,),sigmas=(0.,.02,.1),target_time=1/128)
    assert r['all_trajectories_reached_target']
    assert [t['dissipation_sigma'] for t in r['trajectories']]==[0.,.02,.1]
    for t in r['trajectories']:
        assert len(t['samples'])==2 and t['completed_steps']==1
        assert (tmp_path/'short'/t['last_valid_arrays']).exists()
    assert 'baseline_replay' not in r['trajectories'][1]['samples'][0]
    initial=[]
    for t in r['trajectories']:
        with np.load(tmp_path/'short'/t['samples'][0]['arrays_path']) as z:initial.append(z['q'])
    np.testing.assert_array_equal(initial[0],initial[1])
    np.testing.assert_array_equal(initial[0],initial[2])
    # Compare zero dissipation with this host's unmodified path bitwise.
    # Cross-platform archived initial data keep their measured replay result.
    control=run(tmp_path/'ordinary',points=(32,),target_time=1/128)
    row=control['trajectories'][0]
    with np.load(tmp_path/'ordinary'/row['samples'][0]['arrays_path']) as z:
        assert z['q'].tobytes()==initial[0].tobytes()
    with np.load(tmp_path/'ordinary'/row['last_valid_arrays']) as a, np.load(
        tmp_path/'short'/r['trajectories'][0]['last_valid_arrays']) as b:
        assert a['q'].tobytes()==b['q'].tobytes()
    replay=r['trajectories'][0]['samples'][0]['baseline_replay']
    with np.load(replay['path']) as archive:
        expected=archive['q']
    assert replay['bitwise_equal']==(initial[0].tobytes()==expected.tobytes())
    assert replay['max_slot_difference']==float(np.max(np.abs(initial[0]-expected)))


@pytest.mark.parametrize("dx,sigma",[(0,.1),(np.nan,.1),(.1,-.1),(.1,np.inf),(.1,1)])
def test_invalid_dissipation_parameters_rejected(dx,sigma):
    with pytest.raises(ValueError):ko6_x(mode(32,1),dx,sigma)


def test_unregistered_sigma_suite_rejected(tmp_path):
    with pytest.raises(ValueError,match='registered dissipation'):
        run(tmp_path/'invalid',points=(32,),sigmas=(0.,.05,.1))

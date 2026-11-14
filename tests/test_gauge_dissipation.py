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


@pytest.mark.parametrize("options",[
    dict(points=(32,),fixed_sigma=.1),
    dict(points=(16,32,64),fixed_sigma=.02),
    dict(points=(16,32,64),fixed_sigma=.1,sigmas=(0.,.02,.1)),
])
def test_unregistered_fixed_sigma_study_rejected_before_output(tmp_path,options):
    output=tmp_path/'unregistered'
    with pytest.raises(ValueError):run(output,**options)
    assert not output.exists()


def test_fixed_sigma_refinement_advances_same_time_on_three_grids(tmp_path):
    result=run(tmp_path/'fixed',points=(16,32,64),fixed_sigma=.1,target_time=1/64)
    assert result['status']=='COMPLETED'
    assert result['scientific_acceptance_established'] is False
    assert [row['completed_steps'] for row in result['trajectories']]==[1,2,4]
    assert all(row['last_valid_time']==1/64 for row in result['trajectories'])
    assert all(row['dissipation_sigma']==.1 for row in result['trajectories'])
    assert result['analytic_error_convergence'][0]['time']==1/64
    row=result['trajectories'][1]
    assert row['samples'][0]['fixed_sigma_replay']['max_slot_difference']>=0
    with np.load(tmp_path/'fixed'/row['samples'][0]['arrays_path']) as arrays:
        assert np.max(np.abs(arrays['field_error_h']))==0


def test_analytic_convergence_recovers_known_order_and_handles_zero_errors():
    from experiments.b3_native.gauge_spatial_refinement import convergence_summary
    rows=[]
    for nx in (16,32,64):
        error=(16/nx)**4
        rows.append(dict(nx=nx,samples=[dict(time=1.,
            fields=dict(h=dict(l_inf_over_amplitude=error),zero=dict(l_inf_over_amplitude=0.)),
            constraints=dict(hamiltonian=dict(difference=dict(l_inf=error))),
            auxiliary_consistency=dict(metric_gradient=dict(difference=dict(l_inf=error))))]))
    summary=convergence_summary(rows)
    assert summary[0]['metrics']['h']['orders']==[4.,4.]
    assert summary[0]['metrics']['zero']['orders']==[None,None]
    rows[2]['samples']=[]
    assert convergence_summary(rows)==[]
    assert convergence_summary(rows[:2])==[]


@pytest.mark.parametrize("options",[
    dict(points=(32,),fixed_sigma=.1,timestep_factors=(1,2)),
    dict(points=(16,32,64),fixed_sigma=.1,timestep_factors=(1,2,4)),
    dict(points=(32,),timestep_factors=(1,2,4)),
])
def test_unregistered_temporal_suite_rejected(tmp_path,options):
    with pytest.raises(ValueError):run(tmp_path/'invalid-temporal',**options)
    assert not (tmp_path/'invalid-temporal').exists()


def test_temporal_suite_advances_matching_times_and_distinct_steps(tmp_path):
    result=run(tmp_path/'temporal',points=(32,),fixed_sigma=.1,timestep_factors=(1,2,4),target_time=1/128)
    assert [row['completed_steps'] for row in result['trajectories']]==[1,2,4]
    assert all(row['last_valid_time']==1/128 for row in result['trajectories'])
    assert len({row['last_valid_arrays'] for row in result['trajectories']})==3
    assert result['scientific_acceptance_established'] is False
    assert result['temporal_self_convergence'][0]['time']==1/128
    assert 'fixed_sigma_replay' not in result['trajectories'][1]['samples'][0]


def test_temporal_analysis_cancels_spatial_error_and_recovers_rk4_order(tmp_path):
    from experiments.b3_native.gauge_spatial_refinement import temporal_summary
    from experiments.b1_gauge_wave.run_b1 import FIELDS
    rows=[]
    for factor in (1,2,4):
        # Same spatial bias .5, temporal term .25/factor^4: known independent values.
        error=.5+.25/factor**4
        name=f'factor-{factor}.npz'
        np.savez_compressed(tmp_path/name,time=1.,dt=1/(128*factor),
            **{'field_error_'+field:np.full((2,2,2),error) for field in FIELDS})
        rows.append(dict(timestep_factor=factor,dt=1/(128*factor),samples=[dict(time=1.,arrays_path=name,
                    fields={field:dict(l_inf_over_amplitude=error/.1) for field in FIELDS})]))
    metric=temporal_summary(tmp_path,rows)[0]['metrics']['h']
    assert metric['observed_linf_order']==4.
    assert metric['observed_rms_order']==4.
    assert metric['finest_rk4_error_estimate_over_amplitude']==(.25/4**4)/.1
    rows[2]['samples']=[]
    assert temporal_summary(tmp_path,rows)==[]


def test_temporal_resource_stop_does_not_claim_convergence(tmp_path):
    result=run(tmp_path/'time-stop',points=(32,),fixed_sigma=.1,timestep_factors=(1,2,4),target_time=1/128,max_seconds=1e-12)
    assert result['status']=='RESOURCE_STOPPED'
    assert result['temporal_self_convergence']==[]
    assert not result['all_trajectories_reached_target']

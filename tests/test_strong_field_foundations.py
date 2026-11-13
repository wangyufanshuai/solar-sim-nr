"""Independent polynomial, analytic limit, and angular normalization checks."""
import numpy as np
import pytest
from experiments.strong_field.foundations import (
    prolong_cell_averages, restrict_cell_averages, puncture_psi,
    spin_minus2_l2, sphere_quadrature, project_l2,
)


@pytest.mark.parametrize("n",[8,16,32])
def test_polynomial_transfer_has_derived_cubic_error(n):
    h=1/n
    x=(np.arange(n)+.5)*h
    xf=(np.arange(2*n)+.5)*(h/2)
    # Analytic cell averages, constant in the two transverse directions.
    for power in (0,1,2,3):
        coarse_1d=(np.ones(n) if power==0 else x if power==1 else x*x+h*h/12 if power==2 else x**3+x*h*h/4)
        hf=h/2
        exact=(np.ones(2*n) if power==0 else xf if power==1 else xf*xf+hf*hf/12 if power==2 else xf**3+xf*hf*hf/4)
        coarse=np.broadcast_to(coarse_1d[:,None,None],(n,5,7))
        fine=prolong_cell_averages(coarse)
        np.testing.assert_allclose(restrict_cell_averages(fine),coarse,atol=1e-15,rtol=0)
        # The polynomial is not periodic; keep probes away from the wrap stencil.
        difference=fine[4:-4,4,4]-exact[4:-4]
        target=9*h**3/32 if power==3 else 0
        np.testing.assert_allclose(np.abs(difference),target,atol=2e-15,rtol=0)


def test_coincident_punctures_reduce_to_single_isotropic_mass():
    p=np.array([[3.,4.,0.],[0.,0.,2.]])
    got=puncture_psi(p,masses=(.3,.7),centers=((0.,0.,0.),(0.,0.,0.)))
    np.testing.assert_allclose(got,1+1/(2*np.linalg.norm(p,axis=-1)),rtol=0,atol=1e-15)
    with pytest.raises(ValueError,match="puncture"):
        puncture_psi(np.array([[.5,0.,0.]]))


def test_angular_mode_has_analytic_pole_normalization():
    assert abs(spin_minus2_l2(2,np.array(1.),np.array(0.))-np.sqrt(5/(4*np.pi)))<1e-15
    assert spin_minus2_l2(2,np.array(-1.),np.array(0.))==0
    x,p,w=sphere_quadrature()
    # sin^2(theta) has only m=0 in this spin-2 l=2 basis.
    coefficients=project_l2(1-x*x,x,p,w)
    assert abs(coefficients[0]-np.sqrt(32*np.pi/15))<1e-13
    assert max(abs(v) for m,v in coefficients.items() if m!=0)<1e-13


def test_transfer_rejects_odd_restriction_and_nonfinite_fields():
    with pytest.raises(ValueError):
        restrict_cell_averages(np.ones((7,8,8)))
    q=np.ones((8,8,8)); q[0,0,0]=np.nan
    with pytest.raises(ValueError):
        prolong_cell_averages(q)

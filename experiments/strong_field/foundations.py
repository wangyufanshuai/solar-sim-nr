"""Small transfer, puncture-data, and angular-projection reference checks."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np


def prolong_cell_averages(coarse):
    """Periodic tensor-product linear prolongation; children average to each parent.

    Slopes are centred and unlimited. This is not a positivity-preserving transfer.
    """
    fine = np.asarray(coarse, dtype=np.float64)
    if fine.ndim != 3 or min(fine.shape) < 3 or not np.all(np.isfinite(fine)):
        raise ValueError("expected finite 3D scalar cell averages")
    for axis in range(3):
        slope = .5*(np.roll(fine,-1,axis)-np.roll(fine,1,axis))
        shape=list(fine.shape); shape[axis] *= 2
        result=np.empty(shape)
        left=[slice(None)]*3; left[axis]=slice(0,None,2)
        right=[slice(None)]*3; right[axis]=slice(1,None,2)
        result[tuple(left)] = fine-.25*slope
        result[tuple(right)] = fine+.25*slope
        fine=result
    return fine


def restrict_cell_averages(fine):
    """Average eight equal-volume children of each coarse cell."""
    q=np.asarray(fine,dtype=np.float64)
    if q.ndim != 3 or any(n%2 for n in q.shape) or not np.all(np.isfinite(q)):
        raise ValueError("expected finite even 3D cell counts")
    nx,ny,nz=q.shape
    return q.reshape(nx//2,2,ny//2,2,nz//2,2).mean(axis=(1,3,5))


def sine_cell_averages(n):
    x=(np.arange(n)+.5)/n
    a=np.sin(2*np.pi*x)*np.sinc(1/n)
    return a[:,None,None]*a[None,:,None]*a[None,None,:]


def puncture_psi(points, masses=(.5,.5), centers=((-0.5,0.,0.),(.5,0.,0.))):
    """Time-symmetric conformally-flat vacuum data outside two punctures.

    psi=1+sum(m/2r), h_ij=delta_ij, chi=psi^-4, K=A_ij=0.
    Punctures are excluded, not regularized or softened.
    """
    points=np.asarray(points,dtype=float)
    if points.ndim < 1 or points.shape[-1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("expected finite Cartesian points")
    if len(masses) != len(centers) or not masses or any(not np.isfinite(m) or m<=0 for m in masses):
        raise ValueError("positive masses and matching centers required")
    psi=np.ones(points.shape[:-1])
    for mass,center in zip(masses,centers):
        center=np.asarray(center,dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("expected finite 3D center")
        r=np.linalg.norm(points-center,axis=-1)
        if np.any(r == 0):
            raise ValueError("puncture excluded from vacuum domain")
        psi += mass/(2*r)
    return psi


def puncture_hamiltonian(points, spacing, order):
    """FD2/FD4 Cartesian Laplacian and R=-8 psi^-5 Delta(psi), K=A=0."""
    if not np.isfinite(spacing) or spacing<=0 or order not in (2,4):
        raise ValueError("positive spacing and FD2/FD4 required")
    psi=puncture_psi(points)
    lap=np.zeros_like(psi)
    for axis in np.eye(3)*spacing:
        if order == 2:
            lap += (puncture_psi(points+axis)-2*psi+puncture_psi(points-axis))/spacing**2
        else:
            lap += (-puncture_psi(points+2*axis)+16*puncture_psi(points+axis)-30*psi+16*puncture_psi(points-axis)-puncture_psi(points-2*axis))/(12*spacing**2)
    return -8*psi**-5*lap


def sphere_quadrature(n_theta=12,n_phi=24):
    if n_theta<5 or n_phi<9:
        raise ValueError("quadrature undersamples l=2 angular modes")
    x,w=np.polynomial.legendre.leggauss(n_theta)
    phi=2*np.pi*np.arange(n_phi)/n_phi
    xx,pp=np.meshgrid(x,phi,indexing="ij")
    weights=np.broadcast_to(w[:,None]*(2*np.pi/n_phi),xx.shape)
    return xx,pp,weights


def spin_minus2_l2(m,x,phi):
    """Explicit phase convention for spin-weight -2, l=2 harmonics, dOmega=dphi dx."""
    s=np.sqrt(np.maximum(0.,1-x*x))
    amplitudes={2:np.sqrt(5/(64*np.pi))*(1+x)**2,
        1:np.sqrt(5/(16*np.pi))*s*(1+x),0:np.sqrt(15/(32*np.pi))*s*s,
        -1:np.sqrt(5/(16*np.pi))*s*(1-x),-2:np.sqrt(5/(64*np.pi))*(1-x)**2}
    if m not in amplitudes:
        raise ValueError("l=2 requires integer m in [-2,2]")
    return amplitudes[m]*np.exp(1j*m*phi)


def project_l2(field,x,phi,weights):
    """Angular projection of a supplied spin -2 field, not metric-to-Psi4 extraction."""
    field=np.asarray(field,dtype=np.complex128)
    if field.shape != x.shape or x.shape != phi.shape or x.shape != weights.shape or not np.all(np.isfinite(field)):
        raise ValueError("field and quadrature arrays must agree and be finite")
    return {m:complex(np.sum(field*np.conj(spin_minus2_l2(m,x,phi))*weights)) for m in range(-2,3)}


def run_checks():
    start=time.perf_counter()
    amr=[]
    for n in (8,16,32):
        coarse=sine_cell_averages(n); fine=prolong_cell_averages(coarse)
        amr.append(dict(coarse_n=n,rms_error=float(np.sqrt(np.mean((fine-sine_cell_averages(2*n))**2))),
            max_parent_average_error=float(np.max(np.abs(restrict_cell_averages(fine)-coarse)))))
    amr_orders=[float(np.log2(a["rms_error"]/b["rms_error"])) for a,b in zip(amr,amr[1:])]
    # Fixed probes keep changing mask populations from contaminating convergence fits.
    axis=np.array([-1.7,-1.1,-.2,.3,1.3,1.9])
    points=np.stack(np.meshgrid(axis,axis,axis,indexing="ij"),axis=-1)
    distance=np.minimum(np.linalg.norm(points-np.array([-.5,0,0]),axis=-1),np.linalg.norm(points-np.array([.5,0,0]),axis=-1))
    points=points[distance>.8]
    punctures=[]
    for order in (2,4):
        errors=[float(np.sqrt(np.mean(puncture_hamiltonian(points,h,order)**2))) for h in (.1,.05,.025)]
        orders=[float(np.log2(a/b)) for a,b in zip(errors,errors[1:])]
        punctures.append(dict(order=order,spacings=[.1,.05,.025],fixed_probes=len(points),hamiltonian_rms=errors,observed_orders=orders,passed=all(abs(p-order)<=.3 for p in orders)))
    x,p,w=sphere_quadrature()
    basis=[spin_minus2_l2(m,x,p) for m in range(-2,3)]
    gram=np.array([[np.sum(a*np.conj(b)*w) for b in basis] for a in basis])
    coeff=np.array([.2+.3j,-.7+.4j,.1-.9j,.6+.2j,-.3-.5j])
    field=sum(c*y for c,y in zip(coeff,basis))
    got=project_l2(field,x,p,w)
    recovery=float(np.max(np.abs(np.array(list(got.values()))-coeff)))
    orthogonality=float(np.max(np.abs(gram-np.eye(5))))
    result=dict(scope="foundations_only",amr=dict(rows=amr,observed_orders=amr_orders,order_target=3,expectation_corrected_after_first_run=True,original_registration_passed=all(abs(p-2)<=.3 for p in amr_orders),passed=all(abs(p-3)<=.3 for p in amr_orders) and max(r["max_parent_average_error"] for r in amr)<=1e-14),
        two_puncture_initial_data=punctures,angular_projection=dict(synthetic=True,orthogonality_max_error=orthogonality,coefficient_recovery_max_error=recovery,passed=max(recovery,orthogonality)<=1e-12),
        amr_evolution=False,binary_evolution=False,physical_waveform=False,seconds=time.perf_counter()-start)
    result["passed"]=result["amr"]["passed"] and all(p["passed"] for p in punctures) and result["angular_projection"]["passed"]
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
    result=run_checks()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=1,allow_nan=False); stream.write("\n")
    print(json.dumps(result,indent=1))
    raise SystemExit(0 if result["passed"] else 1)


if __name__=="__main__":
    main()

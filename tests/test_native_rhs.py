"""B3 step 1: the native periodic FO-CCZ4 RHS equals the Python reference to rounding."""

import numpy as np
import pytest
from science.numerical_relativity import Grid3D
from science.numerical_relativity.nr3d import BSSNState3D
from science.numerical_relativity.ccz4_reference import GaugeConfig
from science.numerical_relativity.fo_ccz4_rhs import FOCCZ4State3D, FOCCZ4Config, fo_ccz4_rhs_3d
from science.numerical_relativity.fo_ccz4_checkpoint import state_slots, SYMMETRIC
from kernel.native_rhs import LIBRARY, build, native_rhs_slots

if not LIBRARY.exists():
    build()

def rhs_slots(r):
    return np.stack([*[r.h[...,i,j] for i,j in SYMMETRIC], r.ln_lapse, *[r.shift[...,i] for i in range(3)], r.ln_phi,
        *[r.a[...,i,j] for i,j in SYMMETRIC], r.trace_k, r.theta, *[r.conformal_gamma[...,i] for i in range(3)],
        *[r.b[...,i] for i in range(3)], *[r.lapse_gradient[...,i] for i in range(3)],
        *[r.shift_gradient[...,k,i] for k in range(3) for i in range(3)],
        *[r.metric_gradient[...,k,i,j] for k in range(3) for i in range(3) for j in range(i,3)],
        *[r.conformal_factor_gradient[...,i] for i in range(3)]])

def state(n, order, amp):
    g = Grid3D((n, n+2, n+4), (0.,)*3, (1.,)*3, boundary="periodic", spatial_order=order)
    s = BSSNState3D.minkowski(g, formulation="ccz4")
    x, y, z = g.coordinates; tp = 2*np.pi
    w = lambda a, b, c, ph: np.sin(tp*(a*x + b*y + c*z) + ph)
    h = np.eye(3)[None,None,None] + amp*np.stack([np.stack([w(1,0,1,.1*(i+3*j)) if i<=j else 0*x for j in range(3)],-1) for i in range(3)],-2)
    h = 0.5*(h + np.swapaxes(h,-1,-2)) + 0.0
    h = h / np.cbrt(np.linalg.det(h))[...,None,None]   # det h = 1
    s.h[:] = h
    s.chi[:] = 1 + amp*w(0,1,1,.3); s.lapse[:] = 1 + amp*w(1,1,0,.7)
    s.shift[:] = amp*np.stack([w(1,1,0,.2), w(0,1,1,.4), w(1,0,1,.6)],-1)
    s.trace_k[:] = amp*w(1,1,1,.5); s.theta[:] = .5*amp*w(1,0,1,.9)
    s.a[:] = amp*np.stack([np.stack([w(0,1,1,.2*(i+j)) for j in range(3)],-1) for i in range(3)],-2)
    s.a[:] = 0.5*(s.a + np.swapaxes(s.a,-1,-2))
    s.conformal_gamma[:] = amp*np.stack([w(1,1,i==2,.1*i) for i in range(3)],-1)
    s.b[:] = amp*np.stack([w(1,i==1,1,.3*i) for i in range(3)],-1)
    return FOCCZ4State3D.from_bssn_state(s), g


GAUGE = GaugeConfig(lapse_coefficient=2.0, shift_coefficient=0.75, shift_damping=1.3, theta_coupling=2.0)
CONFIG = FOCCZ4Config(kappa1=0.1, kappa2=0.2, kappa3=0.5, cleaning_speed=1.1, ordering_mu=0.9, shift_strength=1.0, k0=0.05)


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("amp", [1e-3, 0.2])
def test_native_rhs_matches_python_to_rounding(order, amp):
    st, g = state(12, order, amp)
    ref = rhs_slots(fo_ccz4_rhs_3d(st, gauge=GAUGE, config=CONFIG))
    nat = native_rhs_slots(np.stack(state_slots(st)), spacing=g.spacing, order=order, gauge=GAUGE, config=CONFIG)
    per_slot = np.max(np.abs(nat - ref), axis=(1, 2, 3)) / np.max(np.abs(ref), axis=(1, 2, 3))
    assert np.all(np.max(np.abs(ref), axis=(1, 2, 3)) > 1e-10 * np.max(np.abs(ref)))  # every slot exercised
    assert per_slot.max() <= 1e-12

# B1 — harmonic slicing and the nonlinear gauge wave (registration)

Status: **RUN 2026-11-08 — PASSED** (results: [B1_RESULTS_20261108.md](B1_RESULTS_20261108.md)).
Committed before the first run; results are in the separate report.
Code: `experiments/b1_gauge_wave/run_b1.py` (gates in `ACCEPTANCE`) and
`experiments/b1_gauge_wave/harmonic.py` (harmonic slicing).

## Change under test

Harmonic slicing G = f·alpha, implemented as an exact correction on top of the unchanged
constant-coefficient RHS (`harmonic.py`): `+ (f − f alpha) S` for ln(alpha) and
`+ (f − f alpha) d_k(K − 2c Theta) − f alpha A_k S` for A_k, with S = K − K0 − 2c Theta and the
same centred derivative the RHS uses. It requires shift_strength = 0. It lives outside `science/`
because `fo_ccz4_rhs.py` is byte-bound by the frozen S1 registration and the constraint-bridge
export manifest, which therefore stay valid.

Pre-registration smoke check (no evolution): at t = 0 the harmonic RHS matches the exact time
derivative of the gauge wave with observed orders 1.97/1.95 (FD2) and 3.98/3.94 (FD4) between
Nx = 16 and 32, for A = 0.01 and 0.1.

## Test problem

AwA gauge wave, `H = 1 − A sin(2π(x − t))`, exact on every slice; zero shift (shift evolution
off), harmonic slicing f = 1, kappa1 = 0. Grids Nx × 5 × 5, periodic box of length 1 in x.
Error = max over the grid and over chi, h, K, Ã, Theta, GammaHat and lapse of
|numerical − exact|, divided by A.

## Gates (fixed)

| ID | Setup | Pass if |
| --- | --- | --- |
| B1-a | A = 0.01 and 0.1; FD2 and FD4; Nx = 16, 32, 64; dt = 0.1 h; T = 1 | both observed orders in [1.7, 2.3] (FD2) or [3.7, 4.3] (FD4); FD4 error at Nx = 64 ≤ 1e-3; finite |
| B1-b | A = 0.01; FD4; Nx = 32; dt = 0.25 h; T = 100; error sampled every crossing | finite to T = 100; slope of ln(error) over [50, 100] ≤ 0.02 per crossing; error at T = 100 ≤ 1 (i.e. ≤ A) |
| B1-b′ | same with A = 0.1 | reported only: long gauge-wave runs at large amplitude are known to be hard for many formulations |

B1 passes if all eight B1-a cases and B1-b pass. Each case is run once; a changed rerun is a new
registration.

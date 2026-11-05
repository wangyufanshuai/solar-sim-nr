# S2 — periodic FO-CCZ4 evolution and convergence (registration)

Status: **REGISTERED, NOT RUN**. This file and `experiments/s2_periodic_evolution/`
are committed before the first run; results go in a separate dated report.
Code: `linear_wave.py` (exact solution), `run_s2.py` (runs and the acceptance rules in
`ACCEPTANCE`). No source under `science/` changes, so the v1 binary64 manifest is untouched.

## Physics and code under test

`advance_fo_ccz4_rk4_3d` (classical RK4) on periodic grids with `fo_ccz4_rhs_3d`,
FD2 and FD4 centred derivatives, `GaugeConfig(theta_coupling=2)` (advective 1+log lapse,
Gamma-driver shift 0.75, damping 1) and default `FOCCZ4Config()` (kappa1 = 0.1). No
artificial dissipation exists in the RHS.

Test problem: the Apples-with-Apples linearised TT wave,
`gamma_yy = 1 + b, gamma_zz = 1 - b, b = A sin(2 pi (x - t))`, lapse 1, zero shift.
It solves the linearised equations; with this gauge K, Theta and GammaHat are O(A²), so
it is exact to a relative O(A) — 1e-8 at A = 1e-8. Grids are Nx × 5 × 5 on a periodic box
of length 1 in x; y and z derivatives of the data vanish. A pre-registration smoke check
confirmed only that the t = 0 RHS matches the exact time derivative with FD2/FD4 error
ratios 3.94 and 15.8 between Nx = 16 and 32 (no evolution was run).

Error metric: max over the grid of |h_yy − exact| and |Ã_yy − exact|, divided by A,
at T = 1 (one crossing).

## Experiments and acceptance (fixed)

| ID | Setup | Pass if |
| --- | --- | --- |
| S2-a FD2 | Nx = 16, 32, 64; dt = 0.1·h; A = 1e-8; T = 1 | both observed orders log2(E_N/E_2N) in [1.7, 2.3]; all fields finite |
| S2-a FD4 | same | both orders in [3.7, 4.3]; E_64 ≤ 1e-3 |
| S2-b time | FD4, Nx = 32, CFL 0.8, 0.4, 0.2 vs reference CFL 0.1; A = 1e-6 | both orders of the differences to the reference in [3.7, 4.3] |
| S2-c robust | Minkowski + Gaussian noise 1e-10 (seed 42) on 8³, FD2 and FD4, dt = 0.1·h, T = 10, checkpoints every 80 steps | `fo_ccz4_robust_stability_campaign_3d` gate: finite, positive, perturbation L2 growth ≤ 10 at every checkpoint |

S2 passes only if all six checks pass. A larger amplitude in S2-b keeps the RK4 time
differences above round-off; the reference shares the same O(A²) terms, so they cancel.
Each check is run once; a failure is reported as a failure, and any rerun with changed
settings is a new registration.

## Scope

Passing S2 shows correct short-time evolution of a weak wave on a periodic flat background
with the expected spatial and temporal convergence, and no noise blow-up over 10 crossing
times on a coarse grid. It does not test strong fields, boundaries, long-time stability,
black holes or the nonlinear regime.

## Amendment 2026-11-05: S2-c v2 robustness gate (written after run 01)

Run 01 failed S2-c (`S2_RESULTS_20261105.md`). Its diagnostics showed a one-off
constraint transient, plateaus and linear lapse growth. A new gate, written with those
results in view, tests for exponential growth after the transient instead of a fixed
ratio from t = 0: see `experiments/s2_periodic_evolution/robust_v2.py`. Runs: FD2 and FD4
on 8³ to T = 100, FD4 with kappa1 = 0.1 on 8³ to T = 100, FD4 on 12³ to T = 50. Pass if
every run is finite, the slope of ln N(t) over [T/2, T] is ≤ 0.02 per crossing time, and
the Hamiltonian L2 at T is ≤ 10 × its value at t = 1. Run 01's FAILED verdict stands;
a v2 pass would be reported as a pass of this later gate only.

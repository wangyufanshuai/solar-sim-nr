# B2 — SBP-SAT outer boundaries

Status: **B2 quick (B2-a, B2-b) REGISTERED, NOT RUN** — reduced to fit a ~2 h run at the owner's
request (2026-11-08); committed before the run. B2-c and CPBC are deferred.

## Registered gates (B2 quick; `experiments/b2_boundaries/run_b2.py`, ACCEPTANCE)

| ID | Setup | Pass if |
| --- | --- | --- |
| B2-a | sine TT wave (A = 1e-8), exact incoming data on all six faces, Nx = 13, 25, 49 (× 9 × 9), T = 0.25 | both observed orders ≥ 2.6; error at Nx = 49 ≤ 1e-2 (max over all 58 slots, ÷ A) |
| B2-b | pulse (centre 0.8, width 0.1) leaving through x = 1 with flat incoming data there, Nx = 17, 33, T = 0.6 | observed order of the residual ≥ 2; residual at Nx = 33 ≤ 1e-2 (÷ A) |

dt is the largest step with dt·Σ(max characteristic speed/dx) ≤ 0.95 × 0.25 (the driver's cap),
rounded so each time unit has an integer number of steps. The two parts run concurrently with
10 SAT workers each (memory). Each part is run once; a changed rerun is a new registration.

## What exists

- `sbp_boundary_operator.py`: SBP(4,2) first derivative (Mattsson–Nordström 2004), coefficients
  cross-checked against SummationByPartsOperators.jl; scalar SAT energy identity only.
- `fo_ccz4_sbp_volume.py`: all 58 nonlinear FO-CCZ4 volume equations with SBP derivatives.
- `fo_ccz4_sbp_nonlinear.py`: volume + stage-local characteristic SAT on six faces with
  analytic incoming data; no CPBC, no stability theorem; reference grids capped at 9..17 nodes.
- `experiments/b2_boundaries/`: exact linear TT-wave slots with analytic auxiliaries (complex-step
  derivatives; base fields match the periodic S2 construction exactly, auxiliaries match FD4 to
  its truncation error), an uncapped driver subclass and a runner draft.

## Original proposal (superseded by the quick registration above)

| ID | Setup | Proposed gate |
| --- | --- | --- |
| B2-a | sine TT wave, exact incoming data on all faces, Nx = 17, 33, 65 (× 9 × 9), T = 1 | observed order ≥ 2.6 (SBP(4,2): ~3 expected); error at 65 ≤ 1e-2 |
| B2-b | outgoing pulse, flat incoming data on the x = 1 face | reflection falls with resolution (order ≥ 2), ≤ 1e-3 at the finest grid |
| B2-c | sine wave, exact data, long run | no exponential growth (late log-slope ≤ 0.02) |
| later | constraint-preserving boundary conditions | constraint inflow converges to zero |

## Why it is not runnable yet (smoke test, 2026-11-08)

- **Cost.** The driver's CFL cap (dt · Σ max characteristic speed / dx ≤ 0.25, with profile-gauge
  speeds ≈ 2) forces dt ≈ 0.04 h. Each RHS performs one characteristic split per *distinct* face
  point (≈ 6.7 ms each, cached only for byte-identical points); during evolution almost every face
  point differs, so an Nx = 17 RHS costs ≈ 5 s and the planned runs would take ≈ 2 h (17),
  ≈ 8 h (33) and ≈ 30 h (65).
- **Done (2026-11-08): process-parallel SAT** (`parallel_sat.py`, 20 workers), bitwise identical to
  the original RHS (`tests/test_b2_parallel_sat.py`). RHS time 0.64 s (17), 1.49 s (33),
  2.70 s (65), measured while other runs shared the CPU. Estimated wall time: B2-a ≈ 6.5 h,
  B2-b ≈ 6.5 h, B2-c (Nx = 17, T = 10) ≈ 3 h — about 16 h if run one after another.
- **Resolution.** The pulse width 0.08 was under-resolved at Nx = 17 (h = 0.0625; 0.8 A error after
  8 steps); the draft now uses width 0.12. B2-c is shortened to T = 10 on Nx = 17.

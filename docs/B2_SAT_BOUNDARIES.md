# B2 — SBP-SAT outer boundaries (DRAFT, not registered)

Status: **DESIGN DRAFT — NOT REGISTERED, NOT RUN.** Gates below are proposals; they become a
registration only when this file is re-committed with status REGISTERED before a run.

## What exists

- `sbp_boundary_operator.py`: SBP(4,2) first derivative (Mattsson–Nordström 2004), coefficients
  cross-checked against SummationByPartsOperators.jl; scalar SAT energy identity only.
- `fo_ccz4_sbp_volume.py`: all 58 nonlinear FO-CCZ4 volume equations with SBP derivatives.
- `fo_ccz4_sbp_nonlinear.py`: volume + stage-local characteristic SAT on six faces with
  analytic incoming data; no CPBC, no stability theorem; reference grids capped at 9..17 nodes.
- `experiments/b2_boundaries/`: exact linear TT-wave slots with analytic auxiliaries (complex-step
  derivatives; base fields match the periodic S2 construction exactly, auxiliaries match FD4 to
  its truncation error), an uncapped driver subclass and a runner draft.

## Proposed tests

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
  ≈ 8 h (33) and ≈ 30 h (65). **Next task: batch the characteristic split over all face points.**
- **Resolution.** The pulse (width 0.08) is under-resolved at Nx = 17 (h = 0.0625): after
  8 steps the error already reaches 0.8 A. B2-b needs a wider pulse or finer grids.

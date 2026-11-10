# B3 step 1 — native FO-CCZ4 RHS (2026-11-10)

Acceptance (written in [ROADMAP_B.md](ROADMAP_B.md) before the comparison): the native RHS
agrees with the Python RHS to ≤ 1e-12 of the RHS scale in every slot, and is ≥ 20× faster at
Nx = 32. Status: **agreement passed; speed passed with threads (151×), not single-threaded (12×)**.

## What it is

`kernel/fo_ccz4_rhs.cpp` ports `_fo_ccz4_volume_rhs_3d` (science/numerical_relativity/fo_ccz4_rhs.py)
for a periodic grid with FD2/FD4 centred differences and constant gauge coefficients
(`gauge_profile=None`), the case S2 and B1 use. Same equations and term grouping; the 3×3
inverse uses the adjugate formula instead of LU, sums run in a fixed loop order, and the build
uses `-ffp-contract=off` (no fused multiply-adds). One point per OpenMP iteration, derivatives
taken on the fly from the periodic neighbours. `kernel/native_rhs.py` calls it through ctypes on
the canonical 58-slot layout. Not ported: the field-dependent gauge profile, Sommerfeld and
Schwarzschild boundary modes, SBP/SAT. No bound file was changed (the kernel lives outside
`science/`).

## Agreement (tests/test_native_rhs.py)

Generic nonlinear periodic states on a 12 × 14 × 16 grid: every base field varies in all three
directions (h with det h = 1, chi, lapse, shift, K, Theta, Atilde, GammaHat, b), auxiliaries from
the existing FD builder; nonzero kappa1/2/3, cleaning speed, ordering mu, shift damping, k0.

| order | amplitude | max over slots of max|native − python| / max|python| (per slot) | same, global scale |
| --- | --- | --- | --- |
| FD2 | 1e-3 | 8.6e-14 | 7.1e-16 |
| FD2 | 0.2 | 2.2e-15 | 7.8e-16 |
| FD4 | 1e-3 | 1.6e-13 | 7.8e-16 |
| FD4 | 0.2 | 1.8e-15 | 7.2e-16 |

A first test state was too symmetric (shift β^i depending only on x_i, b only on y): four
B_k^i slots were analytically zero and both codes returned rounding noise (~1e-15), which a
per-slot relative measure cannot judge. The test now checks that every slot is exercised.

## Speed (docs/b3-bench-20261110.json)

FD4, Nx³ periodic, best of 3 after a warm-up; Python is NumPy (single-threaded); i9-13900HX,
other applications running (2.7 GiB free memory).

| Nx | Python | native 1 thread | 8 threads | 24 threads |
| --- | --- | --- | --- | --- |
| 32 | 1.53 s | 0.128 s (12×) | 0.018 s (86×) | 0.010 s (151×) |
| 48 | 7.33 s | 0.331 s (22×) | 0.057 s (128×) | 0.031 s (235×) |

## Next (B3 step 2)

Use the native RHS inside the RK4 driver (state update stays in NumPy first), confirm an S2/B1
evolution reproduces the recorded errors, then port the SBP volume operator and the SAT so the
open B2 items can be rerun cheaply.

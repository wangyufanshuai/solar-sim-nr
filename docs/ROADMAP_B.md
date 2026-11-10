# Roadmap B — towards a credible numerical-relativity kernel (2026-11-08)

Goal chosen by the owner: research-grade NR, not only a product. "Top tier" is split in two:

- **Credible NR code** — reproduce published results on the standard tests (AwA suite,
  single black-hole stationarity, head-on collision energy, binary waveforms vs SXS).
- **Top tier in one niche** — verification rigour (exact symbolic checks, pre-registered
  gates, reproducible runs) and provably stable constraint-preserving boundaries for FO-CCZ4
  (SBP-SAT energy estimates), where this repository already has groundwork.

Hardware: one laptop (i9-13900HX, 24 cores, 16 GB RAM, RTX 4060 — FP64 at 1/64 of FP32,
so the GPU is not a shortcut for double-precision NR). Memory is the hard limit.

| Stage | Result to establish (error against a reference) | Estimate |
| --- | --- | --- |
| B1 nonlinear exact solutions | harmonic slicing; AwA gauge wave A = 0.01, 0.1: convergence order and long-run error growth | 2–4 weeks |
| B2 outer boundaries | SBP-SAT constraint-preserving boundaries: outgoing-wave reflections converge, long runs stable, energy estimate | 1–2 months |
| B3 single black hole | C++/OpenMP hot path; moving-puncture Schwarzschild to the trumpet slice; apparent-horizon finder | 3–5 months |
| B4 refinement + head-on | Berger–Oliger (or nested) refinement; Brill–Lindquist head-on; Psi4 extraction; radiated energy vs literature | 3–6 months |
| B5 quasi-circular binary | inspiral–merger–ringdown; mismatch vs SXS ≤ 1e-2, then lower | 6–18 months, probably with rented compute bursts |

Progress (2026-11-08): B1 passed ([B1_RESULTS_20261108.md](B1_RESULTS_20261108.md)); the A = 0.1
long run (reported only) blew up at t ≈ 22.

**B2: not passed, parked (owner's decision, 2026-11-10).** B2-a fails on order (~2 vs 2.6),
B2-b on magnitude (0.25 A vs 1e-2); the face diagnostic points at the SAT treatment of the
incoming speed-1 fields. B2-c has no verdict (run 02 interrupted at t = 3, error flat at
0.031 A). Open items, to be picked up when the native kernel makes reruns cheap: speed-1 SAT
accuracy, B2-c long run. Details: [B2_RESULTS_20261108.md](B2_RESULTS_20261108.md).

**B3 step 1 — native FO-CCZ4 RHS (started 2026-11-10).** A C++/OpenMP port of the periodic
volume RHS (`_fo_ccz4_volume_rhs_3d`, constant gauge coefficients, FD2/FD4), in the new
top-level `kernel/` (outside the bound `science/` files), called through ctypes on the
canonical 58-slot layout. Acceptance: on nonlinear periodic test states the native RHS
agrees with the Python RHS to ≤ 1e-12 of the RHS scale in every slot (rounding only; no
bitwise claim, since summation order differs), and is at least 20× faster at Nx = 32.
The field-dependent gauge profile and the non-periodic boundary modes stay in Python.
Result: agreement ≤ 1.6e-13 per slot; 12× single-threaded, 151× with 24 threads at Nx = 32
([B3_NATIVE_RHS_20261110.md](B3_NATIVE_RHS_20261110.md)).

Rules: one physical result per milestone, each with its error against a reference; gates are
committed before the run; failures are kept; post-hoc gate changes are disclosed; no new
audit/receipt machinery beyond what a result needs.

## Bound sources

`science/numerical_relativity/fo_ccz4_rhs.py` and its neighbours are byte-bound by the frozen S1
registration (v1 manifest `dba2b44c…`) and by the constraint-bridge export manifest. Roadmap B
work therefore goes into new modules (e.g. `experiments/b1_gauge_wave/harmonic.py` adds harmonic
slicing as an exact correction on top of the unchanged RHS) as long as that is possible. When a
bound file genuinely has to change (likely at B3, for performance), S1 and the bridge are
reproduced from the tag `v1.0-freeze`, CI runs their bound tests in a separate job on that tag,
and the change is announced here first.

## Product

The website (Orbit Atlas) stays frozen at v1.0. When a stage here produces a result, its
numbers are exported to the site's read-only `/lab` page; no new product features.

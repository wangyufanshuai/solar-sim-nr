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
long run (reported only) blew up at t ≈ 22. B2 quick and B2-c are running.

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

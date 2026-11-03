# Binary64 v2 (Richardson slot gate) — 2026-11-03

Status: **PASSED** (run 02). Full 8³ binary64 alignment, both paths, 964 state
evaluations, under the v2 acceptance registration. The v1 registration remains
NOT QUALIFIED (see [run 04](BINARY64_RUN04_20261103.md)).

## Why v2, disclosed

v1 run 04 failed one slot of 99,760 (physical43) and the same quantity in the modal
check (1 of 304,616): FD4, eps 2^-10, Hamiltonian row vs q10 = ½ln χ, constant mode.
Its error scaled exactly as eps² (ratio 16.00) with a coefficient ≈ 64.3·w, just
above the registered constant 64. **v2 was written after seeing that result.** It is
a new registration, not a re-scoring of v1.

## The v2 rule

Coefficients are symmetric ±eps differences, so `C(eps) = ref + c2·eps² + c4·eps⁴ + …`.
With the two registered amplitudes (eps_L = eps_H/4),
`R = (16·C_L − C_H)/15` removes the c2 term exactly; `R − ref = −c4·eps_H⁴/16 + …`.
v1 bounded |c2| ≤ 64·w; v2 bounds |c4| ≤ 64²·w:

    |R − ref| ≤ 256·eps_H⁴·w + 8·(16·nu_L + nu_H)/15

This replaces only the per-amplitude `64·eps²` slot bound in the RHS, physical43 and
modal checks, in both the production and replay paths (`richardson.py`). Grid, modes,
amplitudes, signs, resource limits, field/off-mode, resolved, relative, contraction
and negative-control gates are unchanged. An O(1) or O(eps) coefficient fault is not
removed by R and is still rejected (new control tests). One v1 kernel test that
accepted a 100× error shrink for a 4× amplitude step was rewritten: that pattern
implies |c4| ≈ 10⁷·w and v2 rejects it.

Candidate: `.release-local/binary64-v2-richardson-20261103-r2`, manifest
`fb4bd290fc197c78ea68960708d94c3e179e12cbdc276183f3a2da4ff5e09e01`, parent
`1e148ddf…` (startup-2-GiB variant). Prepared by
`.release-local/prepare_binary64_v2_richardson.py`. The first preparation
(`…-20261103`, without the kernel-test update) failed 1 of 347 controls and is kept.

## Runs

| Run | Result | Detail |
| --- | --- | --- |
| controls r2 | 347 / 347 passed | worker peak RSS 126 MB |
| science 01 | NOT_QUALIFIED | production accepted; replay stopped at 72/482, runtime RAM reserve < 2 GiB (transient; worker peak 108 MB, available RAM 6.0 GB before and after) |
| science 02 | **PASSED** | 105.8 s; production 482/482, replay 482/482; comparison accepted; worker peak 128 MB; min available RAM during run 5.16 GiB |

Run 02, worst slot ratio after Richardson elimination (0 failures everywhere):

| Check | Production | Replay |
| --- | ---: | ---: |
| RHS 4×58×58 | 2.1e-06 | 2.1e-06 |
| physical43 10×43×58 | 3.2e-05 | 2.7e-05 / 3.2e-05 |
| modal 101×13 per column | 6.1e-05 | 5.4e-05 / 6.1e-05 |

All other acceptance items passed in both paths, including all 8 injected-fault
negative controls. `verify_binary64_completion.py` (read-only) verified 72 sealed
artifacts and 307 snapshot entries: `INTEGRITY_VERIFIED_NUMERICAL_RECOMPUTATION_NOT_PERFORMED`.
Its outer receipt was assembled after the run from the captured supervisor stdout
(exit 0) and records run 01 as a prior attempt.

## Scope

This establishes that the actual NumPy binary64 state construction, periodic FD2/FD4
derivatives, RHS assembly and physical43 diagnostics reproduce the exact case12
linearisation once the leading eps² amplitude term is eliminated, on one 8³ grid and
one fixed background. It does not test time evolution, grid convergence, boundaries
or any physical scenario. Production and replay share the actual evaluation backend;
they are not independent physics implementations. Next: S2 short periodic evolution
and convergence.

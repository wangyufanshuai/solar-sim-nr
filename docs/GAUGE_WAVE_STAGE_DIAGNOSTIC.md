# S3 large-amplitude stage and roundoff-sensitivity diagnostic

Registered 2026-11-14 before this experiment. Existing B1/S3 gates remain unchanged.
Same 32x5x5 periodic grid, FD4, A=.1, harmonic slicing, zero shift, kappa1=0,
dt=1/128, native threads=1. One invocation is bounded by 600 s / 64 MiB output.

1. Evolve baseline and a second trajectory whose initial h00 slot at (x-index8,y0,z0)
is moved by exactly one representable float toward +infinity. All other slots retain
the same bytes. This deliberately perturbs auxiliary consistency at roundoff scale;
it is a sensitivity test, not a replacement initial-data or science acceptance test.
2. At baseline crossing times 0,1,10,15,17,18,20,22 compare Python harmonic RHS
and native harmonic RHS on the SAME four native RK4 stage input states. Pack all
58 RHS slots, retain per-slot absolute/relative difference and active-slot mask
(max reference magnitude >1e-10 of global RHS scale). Existing RHS comparison
criterion for active slots is <=1e-12 relative. Inactive differences are still reported.
3. At each common crossing report every slot difference and normalized analytic
solution error for both trajectories. Keep original saved Python comparison separate.
4. No positive attribution is pre-assumed. RHS equality on these states cannot prove
all equations correct; sensitivity alone cannot prove the historical discrepancy is
solely roundoff. A detected mismatch remains a failure requiring source inspection.
5. Preserve selected baseline slot snapshots for subsequent offline same-state checks;
no recomputation of evolved auxiliary fields. No external independent reproduction
is claimed by this experiment.


## Measured result

Completed both trajectories to T=22 in 145.07 s, peak working set 46.12 MiB.
The baseline analytic errors are exactly equal to the previously saved native errors
at every common crossing. Same-state comparisons cover all 32 registered stages.
Largest absolute RHS difference is 2.66454e-15. Four stages at T=0 fail the original
1e-12 active-slot relative gate (Theta: up to 5.00895e-12 relative, with absolute
Theta difference only 5.54e-16 and Theta RHS magnitude about 1.11e-4). The later
28 stages pass that relative gate. Overall strict stage comparison remains FAILED;
the criterion has not been widened to absorb cancellation in a small component.

One initial h00 ULP is 1.11022e-16. At T=22 the two native states differ by 2.15135
in the worst canonical slot, an amplification of 1.93776e16. Between T=5 and18,
the fitted log-difference growth rate is 1.78312 per crossing. Analytic normalized
errors at T=22 are 2.09843 (baseline), 21.31915 (one ULP), versus saved Python
13.91410. This establishes severe finite-precision sensitivity on this test. Together
with machine-scale same-state RHS differences it supports roundoff amplification as
an explanation; it does not prove that the historical discrepancy is solely roundoff,
nor validate the unstable long-time trajectory. External reproduction remains absent.

Raw results, four baseline slot snapshots, offline summary and actual-data figure:
gauge-stage/. Re-analyze without evolving:

    python -B -m experiments.b3_native.analyze_stage_diagnostic docs/gauge-stage --null-result docs/s2-null-modes.json --plot

Registration 407dc07; numerical source 9009ade. Three targeted diagnostic tests pass.


## Prospective Theta term isolation (2026-11-14)

Roadmap S3: locate the four T=0 active-slot failures without changing the
1e-12 gate or the original trajectories. Before measuring the new output:
add an optional native diagnostic export of the six additive Theta terms and
its scalar/extrinsic/Z inputs, sharing the production cell routine. Check that
all 58 outputs remain bit-identical to the uninstrumented library on the four
registered T=0 RK4 stage states (32x5x5, dt=1/128, A=.1). Compare each diagnostic
term against the frozen Python equation on the same input bytes. Report maximum
absolute difference, cancellation ratio, reconstruction error and strict slot
comparison. A term localization does not establish a pure-roundoff cause.

No evolution long run is needed. Single process/thread, <=600 s and <=64 MiB
output; <=512 MiB working-set target. Use a new output directory. Original
science sources, gates and saved failure records are retained unchanged.


### Registered inverse-algorithm ablation

Following term isolation, compare the same four T=0 inputs with a Python-only
copy of the frozen functions whose geometry uses the native 3x3 adjugate
formula instead of NumPy LU. Clone function globals locally; do not replace
NumPy or production-module globals. Keep every derivative, RHS term, stage
input and scientific threshold fixed. Report both original and ablated
comparisons, whether the original implementation is bitwise unchanged after
calling the variant, and matrix-inverse error against LU on nonsingular
positive-definite fixtures. Do not assume the variant improves the discrepancy
or adopt it into the solver. Same <=600 s /64 MiB resource envelope.


## Theta term isolation and inverse ablation: measured result

Instrumentation shares the native cell routine. All 58 outputs are bitwise equal
both to ordinary calls and to the original-library four-stage arrays saved before
instrumentation (original kernel at eefba50; registration e9027ff). The frozen
Python science sources are unchanged. Both six-term reconstructions reproduce
Theta exactly on these states.

| T=0 stage | Original worst active-slot relative error | Adjugate variant | Strict 1e-12 variant gate |
| --- | --- | --- | --- |
| 1 | 5.00894654e-12 | 3.98868563e-12 | FAIL |
| 2 | 4.98167522e-12 | 4.95184685e-12 | FAIL |
| 3 | 3.79573324e-12 | 1.17154520e-12 | FAIL |
| 4 | 4.66511481e-12 | 3.77153271e-12 | FAIL |

The dominant absolute mismatch lies in the curvature contribution
0.5*alpha*e^2*(R+2 div Z). Across the four stages its maximum difference is
4.51e-16 to 7.30e-16; lapse and chi are bitwise equal. At each stage's worst
Theta point the six-term cancellation ratio is only 1.000 to 1.076. Therefore
cancellation in the **final six-term accumulation** is not supported as the
dominant explanation at those points. This says nothing definitive about
cancellation inside the geometry construction.

The isolated adjugate-inverse variant reduces some discrepancies but all four
strict stage gates still fail. A change of inverse algorithm alone is insufficient
on these inputs. It is not a solver fix and has not been adopted into production.
Calling the variant leaves the original Python outputs and NumPy/module globals
unchanged. This ablation neither validates long-time stability nor proves that
all historical trajectory differences are solely roundoff.

The final ablation took 0.544 s with peak working set 53.13 MiB. All outputs (including the original-library baseline and earlier diagnostic runs)
total 369640 bytes, below the 64 MiB cap. Numerical source commit: 32a4bfb. Thirty-three related tests passed, including native
RHS, RK4, SBP, harmonic lapse, long-run controls, null-mode diagnostics, strong-field
foundations, nonzero transport/damping, term reconstruction and ablation isolation.

Raw arrays and reports: [theta-terms/](gauge-stage/theta-terms/). Reproduce with a fresh output directory:

```console
python -B -m kernel.native_rhs build
python -B -m experiments.b3_native.theta_terms --out outputs/theta-terms-reproduction --before docs/gauge-stage/theta-terms/before.npz --inverse-ablation
```

Next localization target: the Ricci/Z geometry contractions and derivative
accumulation orders, on the same stage inputs. External independent reproduction
remains absent; the original strict RHS gate remains failed.


## S3 geometry localization registration (2026-11-14)

Before implementation or measurement: use the saved four T=0 RK4 inputs in
`theta-terms/before.npz`, A=0.1, 32x5x5 FD4, single thread. Export actual native
inverse metric, raised metric derivatives, Christoffels and their derivatives,
Ricci, Z derivative and scalar intermediates through an opt-in entry point.
Observe the corresponding frozen Python geometry locals, for both original and
isolated adjugate inverse. Compare tensors and recompute scalar contractions with
native sequential accumulation as a diagnostic only. Verify all 58 native outputs
against saved original-library outputs and ordinary calls; keep original Python
globals and outputs unchanged. No default arithmetic change and no integration.
Keep the strict 1e-12 gate, 600 s, 64 MiB output and 512 MiB working-set ceilings.
The experiment localizes arithmetic discrepancies; it cannot establish long-time
stability, causal uniqueness, or external independent reproduction.


### Geometry localization result and derivative-ablation registration

The ccdd2e7 run preserves every saved and ordinary native RHS bitwise. Original
Python outputs/globals are unchanged. All four strict stage gates still fail.
With isolated adjugate inversion, h_inverse, d_upup, both Christoffels, tilde_gamma
and both Z vectors agree bitwise on these four inputs. The earliest remaining
exported discrepancy is d_conformal_christoffel (max 2.22e-16), followed by
physical-Christoffel derivatives (4.44e-16). Replacing the final scalar contraction
with sequential accumulation changes no values here; upstream geometry differs.
This is a same-state arithmetic localization, not a stability result.
The first observer run compared raw RHS after in-place harmonic correction and
reported a false control failure; retained as initial-control-failure.json.
The corrected run compares before correction, with a four-stage regression test.

Before further implementation/measurement: native periodic d1 multiplies by a
precomputed reciprocal; frozen Python d1 divides by the denominator. Add isolated
Python geometry variants for reciprocal derivative arithmetic, alone and combined
with adjugate inversion; expose actual symmetrized D/P and GammaHat derivatives.
Use exactly the same four saved inputs and limits, no solver changes or evolution.
The original strict RHS gate stays unchanged. Bitwise geometry agreement would
identify these arithmetic choices on these inputs only, not validate the solver.

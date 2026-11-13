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

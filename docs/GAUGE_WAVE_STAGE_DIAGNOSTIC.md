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

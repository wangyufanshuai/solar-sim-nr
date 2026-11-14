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


### Combined geometry arithmetic result and full-RHS ablation registration

At 2d4e320, adjugate inversion plus reciprocal stencil arithmetic reproduces all
397 exported geometry components bitwise at all four saved RK4 inputs. Either
change alone leaves discrepancies. All original native outputs remain unchanged;
the original Python strict RHS gate still fails. Runtime 1.209 s, peak 67.83 MiB.
The agreement covers these saved gauge-wave inputs, not general states or time
integration. Raw arrays and full reports: geometry-terms/arithmetic/.

Before implementation/measurement: clone the frozen volume function with isolated
adjugate geometry and reciprocal first_derivative; retain the original harmonic
wrapper's division arithmetic (matching native_evolve.harmonic_correction).
Compare all 58 full RHS outputs on the same four inputs at the unchanged 1e-12
active-slot relative gate, and show the original comparison alongside the variant.
The variant is diagnostic only; no solver default, science source, long-run gate
or external-reproduction status changes. Retain the same resource ceilings.


### Complete RHS arithmetic ablation: bounded result

Source ad9d661; same saved T=0 four-stage inputs, no evolution. All original
native RHS outputs remain bitwise equal to saved original-library outputs and
ordinary calls. Original Python output/globals/defaults stay unchanged.

| Stage | Original active-slot relative max | Isolated arithmetic variant | Original 1e-12 gate | Variant 1e-12 gate |
| --- | --- | --- | --- | --- |
| 1 | 5.00895e-12 | 6.15618e-14 | FAIL | PASS |
| 2 | 4.98168e-12 | 7.53579e-14 | FAIL | PASS |
| 3 | 3.79573e-12 | 3.60346e-14 | FAIL | PASS |
| 4 | 4.66511e-12 | 3.01284e-14 | FAIL | PASS |

All 397 exported geometry components agree bitwise for the combined inverse/stencil
variant. Full RHS absolute differences remain nonzero, at most 6.94e-18. Thus the
two arithmetic choices are sufficient to account for the strict-gate discrepancy
on these inputs to the registered tolerance, without establishing exact complete
RHS equality, general-state parity or long-time stability. The original gate
remains failed and no diagnostic variant replaces a solver default. In particular,
the T~23.28 failure boundary, S2-c boundedness and external reproduction stay open.
This diagnostic cost 1.569 s, 67.906 MiB peak and 839139 output bytes. Forty related
tests passed, including stencil accuracy, non-diagonal geometry reconstruction,
profiler restoration and whole-RHS ablation isolation. The 267-file constraint
bridge input manifest passed verification without executing its scientific run.

Raw output: [full-rhs/](gauge-stage/geometry-terms/full-rhs/). Reproduce in a fresh
output directory (single-thread BLAS/OpenMP environment recommended):

```console
python -B -m kernel.native_rhs build
python -B -m experiments.b3_native.geometry_terms --out outputs/geometry-reproduction --before docs/gauge-stage/theta-terms/before.npz
```

The process exit status qualifies instrumentation only. Inspect `strict_rhs_pass`
(original, false) and `diagnostic_arithmetic_rhs_pass` (isolated variant, true)
separately. Next: compare registered saved inputs near T=15-18, if available, or
register a resource-bounded acquisition; test the arithmetic explanation before
attributing the later trajectory divergence. No new long evolution was run here.


### Late saved-state arithmetic ablation registration (2026-11-14)

S3 acceptance advanced: localize implementation differences at later native
trajectory states without claiming trajectory agreement or stability.
Before implementation/measurement, use the existing baseline snapshots at T=15,
18 and 22, all shape 58x32x5x5, dt=1/128, FD4, A=.1, harmonic lapse, zero shift.
SHA256 in time order:
- 59321fbc12c4ac7e0a106dbcb4321b3bd4391f41ce4403db0699fec27ab1c276
- 9f8aafb872358a398af05b0b4a6d3bf4b548b0c2a4ae536086c8a8bf311d94a6
- 985410d1ba458b037a57e559ab2d52fde1ed12e47453ffc9165b9c276b3f3edb
Construct four RK4 inputs from ordinary native RHS evaluations of each snapshot;
this samples one step's inputs, with no trajectory advance. Compare original
Python/native and isolated inverse/stencil variants side by side at the unchanged
1e-12 active-slot gate, including the 397 actual geometry intermediates. Save each
input and output. Ordinary calls supply fresh controls here, not archived old-library
outputs. Preserve raw-before-correction and Python-restoration controls.
Each snapshot job retains 600 s / 64 MiB output / 512 MiB working-set ceilings,
single-thread BLAS/OpenMP. No default solver or frozen science edits. No claim of
external independent reproduction or completion of the frozen 8^3 experiment.


### Late same-state result (source 9460109)

All twelve native-generated RK inputs at T=15/18/22 passed the **original**
1e-12 active-slot comparison. Worst original relative differences over four stages
are respectively 5.40275e-14, 4.20536e-14 and 2.08911e-15. The diagnostic arithmetic
variant maxima are 3.77155e-16, 3.33318e-16 and 1.24521e-16. Its 397 geometry
components agree bitwise at every input. Full RHS differences are nonzero in eleven
cases (at most 5.55112e-17); T22 stage4 is bitwise equal. These late controls are
fresh ordinary calls, not an old-library replay. Raw inputs/outputs: geometry-terms/late/.
Original-output preservation and observer controls all pass. Each job took 1.33-1.44 s,
peaked below 66.68 MiB and wrote less than 0.84 MiB. Eighteen related tests passed.
This excludes a sampled late local RHS discrepancy above the registered gate;
it does not establish the cause of accumulated trajectory divergence. The failed
T0 original gate, native failure boundary, S2-c and external reproduction remain open.


### Short identical-snapshot restart registration (2026-11-14)

Before implementation/measurement: restart the registered T22 native snapshot
(hash above) at dt=1/128 toward T24, maximum 256 steps per trajectory. Use the
same additive canonical-slot RK4 for original native RHS, original Python RHS,
and the isolated arithmetic Python RHS. Add a native trajectory with exactly
one upward ULP in slot/index (0,8,0,0). No projection, parameter or threshold edits.
Sample every 16 steps and at failures; retain normalized analytic base-field error,
max slot/base-field difference from the live native baseline, last valid states,
and failing RHS inputs. Record stage-specific finite/positivity/overflow failures.
This is a same-repository restart diagnostic, not frozen multiplicative-driver
parity, a new validated solver, full T0 evolution or external reproduction. The
1% trajectory target is reported separately from the existing 1e-12 local RHS gate.
Limits for the combined four-trajectory job: 600 seconds, 64 MiB output and
512 MiB peak working set, one BLAS/OpenMP thread. Resource interruption is not a
scientific pass. A successful restart does not establish long-time boundedness.


### Identical-snapshot restart result

Source 4da44c6, T22 snapshot, common additive RK4: all four trajectories failed
at stage2 in [23.28125,23.2890625], after 164 valid steps. Status is
SCIENTIFIC_FAILURE, not successful target completion. Runtime 57.490 s, peak
67.508 MiB, raw output 196841 bytes (before terminal analysis). Saved arrays and
original report: restart-t22/. The cf4e2a2 terminal analysis reads these arrays
without rerunning or rewriting the original measurement. At the last valid time,
base-field differences from native divided by A=.1 are .06627 (one ULP), .12436
(original Python), .03922 (arithmetic variant), all above the 1% trajectory target.
The native analytic error divided by amplitude grows from 2.098 at T22 to
1.92635e9 at T23.28125; closeness between implementations is not physical accuracy.
All four failed stage inputs have max(2 ln(phi)) approximately 901.418, beyond
log(max binary64)=709.783, producing chi reconstruction overflow. This establishes
a shared failure on this restart, not the root cause of the earlier instability,
full-trajectory parity, long-time boundedness or external independent reproduction.
Twenty-nine related tests passed, including failure-stage preservation, resource
interruption and exclusion of comparisons across unequal last-valid times.

### Native restart timestep refinement registration

Before implementation/measurement: from the same T22 snapshot, run native-only
additive RK4 at dt=1/128, 1/256, 1/512 toward T24. Preserve per-step failure
intervals and last valid/failed-input arrays; sample at shared 1/8 time intervals.
Compare the last common sampled states with the finest live trajectory and with
the analytic gauge wave. Report failures separately; refinement cannot pass the
original T0 long-wave test or justify resetting invalid states. In particular,
this is a timestep-sensitivity diagnostic on an already inaccurate snapshot,
not a full initial-value convergence study. The combined job retains 600 s,
64 MiB output and 512 MiB peak working-set ceilings and single-thread execution.


### Native timestep-sensitivity result (source b0a4452)

| Native timestep | Failure interval | Failure location |
| --- | --- | --- |
| 1/128 | [23.28125,23.2890625] | RK4 stage2 chi reconstruction overflow |
| 1/256 | [23.27734375,23.28125] | Completed candidate lapse reconstruction underflow to zero |
| 1/512 | [23.279296875,23.28125] | RK4 stage4 exponential overflow |

All three failed before T24. Step refinement did not rescue this already
inaccurate T22 restart. At the last shared regular sample T23.25, analytic errors
normalized by amplitude are about 201.84 in every case. Differences from dt/4
are .05973 (dt) and .004385 (dt/2), using the same amplitude normalization.
Closeness of refined trajectories cannot validate these inaccurate states.
The half-step failed candidate retains finite log slots but min(ln lapse)=-8166.36,
so reconstructed lapse is zero in binary64. This is distinct from a nonfinite-slot
failure or the chi overflow in the coarse trajectory.
Initial source 3893f43 lacked the failed completed-step candidate; its original
report/arrays are retained in restart-refinement-initial/. Source b0a4452 adds this
capture and a regression test. The verification rerun has identical last-valid
arrays for all three trajectories and identical failure intervals; it costs
13.178 s, 82.871 MiB peak and 45496 output bytes. Thirty-one related tests passed.
Raw corrected output: restart-refinement/. No solver arithmetic or thresholds changed.

Reproduction from the repository root, in separate fresh output directories:

```powershell
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
python -B -m experiments.b3_native.geometry_terms --snapshot docs/gauge-stage/baseline-t18.npz --out outputs/late-state-replay
python -B -m experiments.b3_native.gauge_restart --snapshot docs/gauge-stage/baseline-t22.npz --out outputs/restart-replay
python -B -m experiments.b3_native.gauge_restart --refine --snapshot docs/gauge-stage/baseline-t22.npz --out outputs/refinement-replay
```

Restart commands save scientific failures as data; inspect status and
all_trajectories_reached_target instead of treating process exit zero as acceptance.
Next S3 question: identify the pre-T22 growing field/constraint modes and compare
spatial refinement from initial data. A future registered study should separate
discretization growth, gauge evolution and driver rounding before changing a solver.
External reproduction and full frozen 8^3 alignment remain uncompleted.


### Saved-state growth-mode registration (2026-11-14)

Before implementation/measurement, analyze the existing native baseline snapshots
at T=0,15,18,22 (32x5x5, FD4, A=.1, dt=1/128), without advancing a solver.
S3 acceptance moved: quantify where the already observed growing error resides;
this diagnostic has no new physical pass threshold and cannot close S3.

For each saved state compare all seven B1 base fields with the analytic gauge
wave at exactly its saved time. Evaluate physical ADM Hamiltonian/momentum and
BSSN algebraic/Gamma/Theta constraints on both evolved and analytic states using
the same discrete operator. Preserve evolved, analytic-discrete and difference
arrays/norms; analytic discrete residuals need not be zero. Separately compare
all 33 evolved first-order auxiliaries with derivatives reconstructed from their
own base fields. Curl-free alone is insufficient for derivative consistency.

Compute one-sided x Fourier energy of field errors and constraint/auxiliary
differences, with correct DC/Nyquist weighting, component/transverse averaging,
and Parseval checks. Report DC, fundamental, Nyquist and high-k (k>=nx/4)
fractions, and transverse fluctuation energy; retain every spectral bin. Tests
must cover pure DC, fundamental, Nyquist, transverse and tensor-component modes,
plus identical analytic-state controls and malformed snapshots. Do not infer a
pure-gauge or constraint-violating root cause merely from four time slices.

Keep source/input hashes, finite binary64 validation, no frozen-source edits,
600 s / 512 MiB process working set / 64 MiB output ceilings, one process and
one numerical thread. Save raw arrays and a machine-readable diagnostic report.
External independent reproduction, full frozen 8^3 alignment, physical accuracy
and long-time stability remain separate unmet requirements.

# S2-c / S3 low-memory continuation (2026-11-13)

Registered before new runs on branch codex/s3-gauge-wave-evidence.
S2-c acceptance moved: measured late growth and constraints; no claim of infinite-time boundedness.
S3 acceptance moved: independently implemented native RHS/RK4 compared with saved Python trajectories; not external reproduction.

- S2: preserve seed 42, noise 1e-10, FD4, n=8, dt=0.0125, default gauge, kappa1=0. Run T=1 pilot and compare every slot with Python; then T=500, samples every crossing. Report slopes over [50,100], [100,200], [200,500], total and per-field perturbation norms, and Hamiltonian growth from t=1. Original S2-c v2 thresholds remain unchanged and only apply to their registered horizons.
- S3: A=0.1, 32x5x5, FD4, harmonic slicing, zero shift, kappa1=0, dt=1/128, target T=100. Compare common crossing samples against docs/b1-gauge-wave/long-a0.1.json (1% diagnostic agreement target). Keep failure, record last finite step and first rejected RK4 stage, and do not alter acceptance thresholds to achieve continuation.
- Each numerical invocation: maximum 600 seconds, maximum 64 MiB output, one native thread. Pilot before long S2 run; stop on nonfinite values. Record operational stop separately from scientific failure.
- Full 8^3 binary64 registration remains separate, with >=4 GiB available RAM before launch. These native diagnostics do not execute or replace that campaign.


## Measured result (local date 2026-11-14)

S2-c: RESOURCE_STOPPED at 600.02 s, 35,273 accepted steps, integrated T=440.9125,
last crossing sample T=440. Peak process working set 46.25 MiB. T=500 was not reached.
The registered [200,500] fit is therefore unavailable. The explicitly post-hoc [200,440]
log slope is +0.001711745 per crossing; norm ratio over that window is 1.3851.
Hamiltonian L2(T=440)/L2(T=1)=1.02025. Lapse RMS has a measured linear slope
2.6002e-11 per crossing from T=100 to 440. Total perturbation is 56.47 times its T=0 value.
The available trajectory does not establish long-term boundedness; it displays secular lapse
growth. No finite run can prove an infinite-time bound without additional analysis.
Native-vs-saved-Python maximum common per-field RMS relative difference to T=100 is
5.6302e-5. This is a comparison of diagnostics, not bitwise state alignment.

S3: SCIENTIFIC_FAILURE. A=0.1 reaches T=23.28125 (2,980 accepted steps), then RK4
stage 2 overflows in exp during step 2,981, in [23.28125,23.2890625]. Wall time
113.88 s; peak working set 42.67 MiB. Early errors agree: relative difference is 1.68e-12
at T=1, 2.35e-8 at T=10, 0.002791 at T=17. First common sample outside the registered
1% diagnostic target is T=18; difference reaches 0.84919 at T=22. Therefore the native
large-amplitude trajectory comparison FAILS. Both implementations lose validity near this
regime, but their late trajectories and stopping times are not numerically reproduced.
Roundoff amplification is a hypothesis, not an established cause; a field/stage-level
comparison is still needed. This remains a same-project implementation check, with no
external independent reproduction.

Raw data and actual-data figure are tracked in long-diagnostics/. The saved Python S2
reference is included there; no dependency on the original working directory is needed:

    python -B -m experiments.b3_native.analyze_long_diagnostics docs/long-diagnostics --reference docs/long-diagnostics/python-fd4-n8-t100.json --plot

New native runs require a compiled kernel (python -B -m kernel.native_rhs build), and a
new output filename on every invocation. Numerical runner source: d210123 for S2 and
30a0de8 for S3; registration: 85a738a. Nine scoped tests passed before S2 execution;
the four new diagnostic tests passed again after failure-stage reporting was added.

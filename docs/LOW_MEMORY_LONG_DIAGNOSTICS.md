# S2-c / S3 low-memory continuation (2026-11-13)

Registered before new runs on branch codex/s3-gauge-wave-evidence.
S2-c acceptance moved: measured late growth and constraints; no claim of infinite-time boundedness.
S3 acceptance moved: independently implemented native RHS/RK4 compared with saved Python trajectories; not external reproduction.

- S2: preserve seed 42, noise 1e-10, FD4, n=8, dt=0.0125, default gauge, kappa1=0. Run T=1 pilot and compare every slot with Python; then T=500, samples every crossing. Report slopes over [50,100], [100,200], [200,500], total and per-field perturbation norms, and Hamiltonian growth from t=1. Original S2-c v2 thresholds remain unchanged and only apply to their registered horizons.
- S3: A=0.1, 32x5x5, FD4, harmonic slicing, zero shift, kappa1=0, dt=1/128, target T=100. Compare common crossing samples against docs/b1-gauge-wave/long-a0.1.json (1% diagnostic agreement target). Keep failure, record last finite step and first rejected RK4 stage, and do not alter acceptance thresholds to achieve continuation.
- Each numerical invocation: maximum 600 seconds, maximum 64 MiB output, one native thread. Pilot before long S2 run; stop on nonfinite values. Record operational stop separately from scientific failure.
- Full 8^3 binary64 registration remains separate, with >=4 GiB available RAM before launch. These native diagnostics do not execute or replace that campaign.

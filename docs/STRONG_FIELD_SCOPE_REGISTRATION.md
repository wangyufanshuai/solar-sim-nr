# Strong-field minimum-scope registration

Registration date: 2026-11-14. Repository: `solar-sim-nr`, branch
`codex/s3-gauge-wave-evidence`. This document defines the smallest executable
entries for the strong-field follow-on. It is a scope boundary, not a claim
that the repository is an AMR or binary-black-hole evolution code.

## Reproduction entry

From the repository root, with one numerical thread:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
python -B -m experiments.strong_field.foundations --out outputs/strong-field-foundations.json
```

The command is bounded to arrays no larger than `64^3`, one process, at most
600 seconds and 64 MiB of output. It writes a new output path and does not
invoke the frozen full `8^3` binary64 campaign.

## Registered entries and evidence

| Entry | Executable scope | Current evidence | What remains outside scope |
|---|---|---|---|
| AMR transfer | Periodic 2:1 cell-average prolongation with centred slopes and exact eight-child restriction | Parent-average error `1.11e-16`; corrected smooth sine orders `2.9591`, `2.9922`; cubic probe check is explicit | Patch hierarchy, refinement scheduling, time subcycling, flux refluxing, boundary exchange, stability, discontinuities |
| Two punctures | Time-symmetric conformally-flat Brill-Lindquist initial data; fixed physical probes; FD2/FD4 Hamiltonian residual | FD2 orders `2.0008`, `2.0002`; FD4 orders `3.8104`, `3.9085` | Gauge evolution, outer boundaries, apparent horizons, orbit, merger, radiated energy |
| Spin-weighted projection | Explicit spin-weight `-2`, `l=2` quadrature and coefficient recovery for a supplied complex field | Orthogonality error `8.88e-16`; synthetic recovery error `8.95e-16` | Metric-to-`Psi4`, tetrad construction, extraction-radius convergence, strain integration, extrapolation, physical waveform |

The AMR order correction is disclosed in
`experiments/strong_field/README.md`: the first registration expected second
order, but this cell-average transfer cancels the quadratic term and measured
third-order smooth error. The initial result remains preserved in
`docs/strong-field-foundations-initial.json`; the corrected result is in
`docs/strong-field-foundations.json`. The correction must not be reported as a
pre-registered success.

## Promotion gates

No entry may be promoted to a physical application until the following separate
registrations produce reference errors:

1. **AMR evolution:** a multi-level mesh with boundary exchange, subcycling and
   refluxing, plus a manufactured-solution convergence test.
2. **Binary dynamics:** puncture gauge and outer-boundary validation on a single
   hole, then a head-on binary with horizon tracks and radiated-energy comparison.
3. **Waveform:** metric/tetrad `Psi4` extraction at multiple radii, time and
   radius convergence, and comparison against a published or SXS reference.

The current entries are therefore useful for testing data layout, transfer
operators, initial-data residuals and angular algebra. They do not establish
AMR stability, binary evolution, inspiral/merger, or a physical gravitational
waveform.

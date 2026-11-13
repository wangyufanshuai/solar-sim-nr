# S3 follow-on: small strong-field building blocks

Registered before execution, 2026-11-14. These are experimental foundations, not an
AMR evolution engine, inspiralling binary, or astrophysical waveform. The existing
uniform-grid S1/S2/S3 gates and frozen sources are unchanged.

1. AMR transfer: 2:1 cell-average prolongation with centred slopes, followed by exact
volume averaging restriction. Check constant preservation, each-parent average
conservation to 1e-14, and second-order error against analytic periodic cell averages
on 8,16,32 coarse cells per axis. No refinement scheduler, time subcycling, flux
refluxing, boundary exchange, or stability claim is included. Centred slopes are
unlimited and may overshoot near discontinuities.
2. Two punctures: psi=1+sum(m_i/(2r_i)), chi=psi^-4, conformal h=identity,
K=A=0, zero momentum/spin. Outside punctures Delta(psi)=0 analytically and
Hamiltonian R=-8 psi^-5 Delta(psi). Measure FD2 and FD4 residuals at the SAME
fixed physical probes using h=.1,.05,.025, safely away from punctures. Expected
orders 2 +/- .3 and 4 +/- .3. This is initial data only, no evolution or orbit.
3. Wave extraction algebra: explicitly defined spin-weight -2, l=2 angular modes
on Gauss-Legendre cos(theta) and uniform phi quadrature. Orthogonality and
synthetic coefficient recovery <=1e-12. This projects a supplied complex field;
it does not compute Psi4 from a spacetime metric, integrate strain, extrapolate to
infinity, or produce a physical waveform. Quadrature 12x24; all five m modes.

Entry: python -B -m experiments.strong_field.foundations --out NEW_FILE.json
Resource scope: arrays at most 64^3, one process, at most 600 seconds / 64 MiB output.
This command does not invoke the registered full binary64 alignment campaign.


## Registration correction after first observation

The first run is preserved as docs/strong-field-foundations-initial.json and fails the
registered AMR order-2 window: measured orders are 2.959 and 2.992. The expected
order was wrong for this particular 2:1 cell-average transfer. A quadratic cell average
at parent centre x is x^2+h^2/12; both child averages have the SAME curvature term,
so the quadratic term cancels and centred slopes give exact child averages. For a cubic,
coarse average x^3+x h^2/4 gives child error +/-9h^3/32. Thus the smooth transfer has
third-order local accuracy. This derivation is checked with an additional polynomial
probe test, specified here before its execution. The corrected sine order window is
3 +/- .3. This is a disclosed post-hoc correction, not a pre-registered success; the
initial failed result and original expectation remain visible. Existing NR gates are unchanged.

# S2-c centred-derivative null-mode hypothesis (2026-11-14)

Post-hoc hypothesis motivated by the existing lapse RMS linear growth. Original S2-c
acceptance remains unchanged, with no uniform boundedness claim.

For an even periodic grid, each centred first derivative annihilates both a constant
and its Nyquist alternating mode. In three dimensions there are eight parity products
P_b(i,j,k)=(-1)^(b0*i+b1*j+b2*k), b in {0,1}^3. Around Minkowski with kappa1=0,
linearized S=K-2Theta obeys d_t S=-Delta(alpha), using the implemented first-order
auxiliary derivative. A parity mode has zero discrete derivative, so S is constant to
linear order and d_t ln(alpha)=-2S, d_t ln(phi)=K/3. Thus a secular lapse component is
predicted even without exponential growth.

Prospective checks, before evaluating this prediction: compute the eight Fourier/parity
coefficients of seeded S2 noise (seed42, n8, noise1e-10); predict the late lapse RMS slope
as 2*sqrt(sum(mean(S*P_b)^2)) and compare with saved T100..440 slope 2.6001990086e-11.
Also retain the constant-only contribution, not just the total.

Then test all eight derivative null modes directly for FD2 and FD4, and run a fresh
analytic pure-checkerboard K perturbation (h=identity, lapse=chi=1, K=1e-8*P_111,
Theta=0, no other perturbation), n8 FD4, dt=.0125 to T=10. Compare ln(alpha) with
-2*K_initial*t and ln(phi) with K_initial*t/3. Expected max errors <=1e-11, accounting
for O(K^2) nonlinear terms. The initial state is a discretization diagnostic, not a
resolved physical solution. No filtering of the original noise or solver change.
Each invocation <=600 s /64 MiB output, one native thread.


## Measured result

All 16 parity/FD2-or-FD4 combinations have exactly zero centred derivative.
Prediction using only initial seed42 coefficients: 2.60035754098e-11 per crossing.
Saved lapse RMS fit: 2.60019900856e-11; relative difference 6.09693e-5 (0.00610%).
The constant mode alone predicts 3.26346e-12 and cannot account for the full slope;
the seven nonconstant parity modes matter.

The fresh pure P111 evolution completed to T=10 in 12.51 s, peak 42.98 MiB.
At T=10 measured ln(alpha) RMS is 2.0000000000000023e-7 versus predicted 2e-7.
Max error from linearized ln(alpha) is 3.33e-15, and from ln(phi) 1.67e-15;
both meet the pre-specified 1e-11 diagnostic bound. Three targeted tests pass.
Raw result: s2-null-modes.json. Registration 2bc47f5; numerical source b75ae3a.

This provides a discrete linear mechanism for the observed secular lapse component.
It is not a nonlinear stability theorem, an infinite-time bound, or a physical resolved
checkerboard solution. Do not remove these modes from the original data or relabel
the earlier failed fixed-growth test as passed. A future explicitly separate solver
variant can test dissipation against the nonconstant parity modes; mean gauge modes
must still be accounted for, and all original results remain unchanged.


## Independent continuation replay (2026-11-14)

The registered command was rerun into a fresh ignored output path:
`python -B -m experiments.s2_periodic_evolution.null_mode_diagnostic
--out outputs/s2-null-modes-continuation.json`, with one numerical thread.
The result completed in 8.4658 s, peak process working set 43.84 MiB, and
matched every stored seed42 prediction field in `docs/s2-null-modes.json`
bitwise as parsed JSON: predicted slope 2.6003575409805745e-11, observed
slope 2.6001990085587156e-11, relative difference 6.0969341700017e-05,
and constant-only slope 3.2634648734288123e-12. All parity derivatives
remained exactly zero. The pure P111 check reached T=10 with maximum
ln(alpha) error 3.3333383129202843e-15 and maximum ln(phi) error
1.6666659006772512e-15; both remain below the pre-registered 1e-11 bound.
This is a replay of the existing diagnostic and does not alter the solver,
original S2 gates, or the conclusion that infinite-time boundedness is
unproven.

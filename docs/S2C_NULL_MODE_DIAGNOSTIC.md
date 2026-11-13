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

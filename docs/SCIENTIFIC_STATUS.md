# Scientific status

**Status: research-alpha.** The available evidence concerns specific
fixed-background mathematical and source-interpretation checks. It does
not make the repository a validated general-purpose numerical-relativity
solver.

## Historical local evidence

Two local campaigns completed on 2026-10-15 before this standalone
repository was created. Their results were checked through complete
coefficient comparisons, separate scalar-arithmetic reconstruction and
read-only terminal audits. They were not external reproductions.

| Campaign | Established local scope |
| --- | --- |
| Actual continuum volume linearization | First variation of the registered real-formula volume RHS in all 58 independent states and 174 independent spatial gradients agrees with the accepted fixed-case12 linear symbol. |
| Actual constraint diagnostic bridge | First variation of the complete 43-row physical diagnostics in 580 independent state/gradient/Hessian coefficients agrees with accepted C0/C1/C2 coefficients; all 49 raw driver scalars satisfy the registered correspondence and symmetry checks. |

These campaigns concern exact rational interpretations of source formulas
at a constant background. They do not show that executing the original
NumPy binary64 program produces the same exact values.

The constraint campaign's original gate identity is
`285b84e81940b3465853a4752d1b7db4319a4bf2f6b4216998968e4669165c51`.
Its volume parent is
`ba5bff599828bf9185db40c72e6421c337fe89b60fab5acde4329f534b4502f0`.
The polynomial coordinate parent is
`466bf5a2a13dadf05bda60b9cbbad4e60ce43f442bcb8fdda0cf61fb2a4dd708`.
These identify historical receipts; a recorded digest is not itself a
reproduction or an external endorsement.

## Fixed background and diagnostic meaning

The demonstrated background is case12:

```text
alpha = 17/20
chi   = 18/25
beta  = (3/100, -1/25, 1/20)
h     = L L^T
L     = [[1, 0, 0], [3/25, 1, 0], [-2/25, 3/50, 1]]
```

The remaining dynamical and auxiliary background fields and all
background spatial derivatives vanish. The metric is positive definite
with determinant one. Perturbations retain all independent state
components, including off-constraint metric, trace and auxiliary fields.
They are not projected onto the constraint surface.

The 43 physical rows are Hamiltonian H, momentum M, Theta, lower-index Z,
the A/B/D/P definition defects, metric determinant and conformal
extrinsic-curvature trace. The actual raw connection difference is
converted by `Z_i = h_ij Ghat^j / 2` before extracting coefficients.
The driver exposes 49 raw scalars: 40 independent quantities plus nine
redundant symmetric D components. Its API does not itself contain Z.

The 580 seeds comprise 58 states, 174 first spatial derivatives and 348
independent symmetric Hessians. This is first order in perturbation
amplitude and second order in spatial derivatives, not a nonlinear
second-order perturbation calculation. Mixed Hessian coefficients are
extracted from every polynomial monomial, without direction sampling.

## Historical acceptance and portable target

The original constraint campaign saved 13 complete matrices containing
206,707 scalar slots. The 53,501 required baseline/background/agreement
slots were exactly zero. Two genuine source mutations each produced
18 changed coefficients and 18 same-slot nonzero discrepancies, with no
changes outside their registered target rows.

The mutations replace the physical extrinsic trace by stored K, or replace
the contracted Christoffel target by inverse-metric divergence. The trace
mutation also changes the Hamiltonian's trace-squared expression; only
the complete first variation at this zero-curvature background establishes
that the detected changes are confined to momentum rows.

The [portable example](../examples/reproduce_constraint_bridge.py) targets
this complete diagnostic calculation. It recomputes the original-source
baseline and both mutations with forward coefficients, then recomputes
them using reverse scalar differentiation. It also executes the baseline
driver diagnostic. It does not merely check that an old residual file
contains zeros.

Its success must be reported as a portable recomputation at the stated
commit and environment. It does not repeat the original campaign's
138 controls, 696-source snapshot, 29 historical directories or Windows
Job supervisor. The portable report therefore keeps
`original_full_qualification_rerun=false`.

## Common trusted base

Forward and reverse algorithms use different scalar differentiation and
small-matrix inverse/determinant arithmetic. They share:

- Original source selection and the narrowly registered AST adaptations.
- Exact literal interpretation, centered logarithmic/exponential chart
  and state/output packing.
- Object-array routing and the continuum spatial-derivative interpretation.

The original FD2/FD4-FFT and SBP derivative operators are explicitly
replaced for this check. A private driver adapter avoids binary64 field
conversion and calls the adapted state constructor. These substitutions
are part of the certificate's scope, not proof that the original discrete
program has executed correctly.

## Claims that remain open

| Area | What this repository's demonstrated bridge does not establish |
| --- | --- |
| Floating-point implementation | Agreement and stability of actual binary64 state, RHS and diagnostic execution |
| Time evolution | A validated periodic Cauchy evolution or time-integration convergence result |
| Discretization | Complete FD/SBP/SAT energy, consistency or convergence validation |
| Boundaries | A well-posed physical initial-boundary-value problem or constraint-preserving boundary implementation |
| Background range | Variable-background or nonlinear stability from a fixed-case12 first variation |
| Constraint dynamics | An inverse for all 43 constraints at zero frequency, a damping theorem or a loss-free Euclidean estimate for the 43 constraints |
| Black holes and AMR | Validated puncture/excision evolutions, AMR transfers, binary mergers or waveform accuracy |
| Independent reproduction | A confirmed reproduction by an external team or independent physical implementation |

Earlier principal-symbol and energy work has its own restricted domains;
the two source bridges do not automatically broaden those domains.
Archived modules and examples are not evidence that the listed open gates
have been completed. Long-lived evolution, resolution studies, observable
error budgets and independent comparisons require their own records.

See [reproducibility](REPRODUCIBILITY.md) for the operational contract and
[the main README](../README.md) for the minimal entry point.

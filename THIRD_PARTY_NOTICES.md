# Third-party notices and scientific references

Original Solar Sim NR code and accompanying new documentation are distributed
under the MIT license. Existing third-party notices remain in their source files.
Python packages are installed as dependencies; their source distributions are not
vendored into this repository and retain their respective licenses.

## SummationByPartsOperators.jl coefficient tables

The diagonal-norm SBP second-derivative coefficient table in
[`science/numerical_relativity/sbp_second_derivative.py`](science/numerical_relativity/sbp_second_derivative.py)
is attributed there to Hendrik Ranocha / SummationByPartsOperators.jl. The original
MIT copyright and permission notice is preserved verbatim in that byte-identical
source file and must be retained with the coefficient table.

Upstream: <https://github.com/ranocha/SummationByPartsOperators.jl>

## Scientific references

FO-CCZ4 formulation: M. Dumbser et al., *A strongly hyperbolic first-order CCZ4
formulation of the Einstein equations and its solution with discontinuous
Galerkin schemes*, Physical Review D 97, 084053 (2018),
<https://doi.org/10.1103/PhysRevD.97.084053>.

References to Einstein Toolkit, SpECTRE and GRChombo in the research sources are
methodological comparisons and citations. They do not establish endorsement,
code equivalence, or scientific qualification of this implementation.

This release excludes the parent astronomy application's textures, catalogs,
database files, web assets, binaries and downloaded content packs.

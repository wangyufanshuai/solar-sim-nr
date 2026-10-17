# Validation record

This page records local release checks performed on 2026-10-17. The
repository remains **research-alpha**. Software controls, a portable
scientific recomputation, hosted CI and independent external reproduction
are separate results.

## Environment and completed controls

The local environment was Windows, CPython 3.12.3 and NumPy 2.2.6, with a
new virtual environment. These results describe that environment; they do
not establish equivalent behavior on every supported platform.

| Check | Local result | Scope |
| --- | --- | --- |
| Dependency installation and `pip check` | Passed | Pinned environment consistency |
| Core dependency imports | Passed | Importability, not scientific validation |
| Six Python Schwarzschild controls | 6 passed in 0.62 seconds | Registered local controls; not a black-hole evolution |
| Portable release controls, initial registration | 33 passed in 0.44 seconds | Small software and provenance controls; did not catch the complete reverse-wrapper defect described below |
| Portable release controls, corrected registration | 39 passed in 0.56 seconds | Includes six added controls, including a real small reverse graph |
| Three C++ component tests with MSVC | 3 passed in 0.43 seconds | Topology, spatial operators and Schwarzschild domain classification |

The first CMake attempt using MinGW failed during its trivial compiler
check, before any scientific component ran. Its `build/native` directory
was retained locally. A separate fresh configuration using MSVC
19.44.35224 and OpenMP 2.0 built and passed the three CTest components.
The compiler-configuration failure is not a failed relativity experiment,
and the CTest success is not evidence of a complete evolution solver.

## First clean-checkout reproduction: failed

The first complete portable attempt used commit
`e1a52cb9635865e33c7859bf00893b8b4cef88e9` and entry-point SHA256
`33756354b65fa03bb777961b217ed8b91b16459831f4e710a51ca7a22eb004ae`.
It **failed** after 7.219 seconds, with worker exit status 1. It did not
reach the 600-second deadline.

The forward calculation completed all 22 registered payload records, and
its saved acceptance fields were true. The complete attempt nevertheless
failed during `reverse-all-three-cases`: the new portable wrapper required
a `dict` for scalar coefficients, whereas the frozen reverse backend
returns a read-only `MappingProxyType`. The worker recorded `ValueError`.
The finalizer then recorded `FileNotFoundError` because no independent
replay report had been completed.

The final report correctly kept `passed=false` and
`portable_core_reproduction_passed=false`; the input audit passed and the
owned worker stopped. Forward acceptance alone does not qualify the
portable recomputation.

All 26 JSON files from this failed attempt remain in the ignored local
directory `.release-local/clean-checkout/outputs/constraint-bridge`.
This is a private local evidence location, not a file bundle distributed
with the public repository. No successful reverse result is inferred or
filled in for the failed attempt.

## Corrected clean-checkout reproduction: passed locally

Commit `8f7200594e55a2922de2f71d60ddf0d27064cbf4` corrects the portable
wrapper to accept the reverse backend's immutable coefficient mapping,
while retaining strict coordinate and exact-coefficient checks. It also
adds regression controls. The repair changes the new integration layer;
it does not change the physical formulas, frozen reference inputs or
scientific acceptance thresholds.

A second fresh checkout used a separate `.venv-minimal` environment with
CPython 3.12.3 and NumPy 2.2.6 as its only third-party dependency. Its
`pip check` passed. The complete calculation ran into a new output
directory, without reusing the first attempt's calculated payload.
The fresh checkout's separate unit-test run passed all 39 controls in
0.49 seconds, and the checkout remained clean.

| Result | Observed value |
| --- | --- |
| Entry-point SHA256 | `8448e2d57ba5f6504c9f0a0a208c96aae40e17b49bbb335817e36e1f7ed3744e` |
| Process exit | 0 |
| Overall elapsed time | 9.859 seconds |
| Worker elapsed time and final stage | 8.125 seconds; `complete` |
| Final result | `passed=true`, `portable_core_reproduction_passed=true` |
| Input audit / worker cleanup | Passed / owned worker stopped |
| Reverse reconstruction | Passed, completed, `errors=[]` |
| Saved output | 27 JSON files; 3,502,632 bytes |
| Complete scientific matrices | 13 matrices; 206,707 scalar slots |
| Required baseline/background/agreement zeros | All 53,501 slots exactly zero |
| Each source mutation | 18 changed slots, 18 same-slot causal discrepancies, zero changes outside registered target rows |

The local evidence directory is
`.release-local/clean-checkout-r2/outputs/constraint-bridge`, also ignored
and not included in the public source repository. The final receipts have
these SHA256 identities:

| Receipt | SHA256 |
| --- | --- |
| `portable-result.json` | `036c50432b3b6b4ec930777acebda2de9a2a8555ea1ba626b8db62ed9cc3fc27` |
| `worker-result.json` | `4ae9b30dd42d0042bc3ece005c6241db0815f646a10fba9de890cd790076762f` |
| `independent-replay.json` | `a81fb9de97d1a1104528c6d69e7a9e6dee64801b1acec8b9a23883056cfbbc5e` |

All 26 artifact hashes listed by the final report were independently
rechecked against the saved bytes.

The first attempt remains failed. The second result establishes a local
portable recomputation in the stated environment, not an external
reproduction. These elapsed times are observations, not performance
guarantees or estimates for black-hole simulations.

## Hosted CI status

Hosted CI has not yet been checked for this revision. The configured
Windows/Linux workflow and C++ component job are not recorded as passing
until their actual remote results are available.

## Preservation checks

The 267 exported source/input identities agree with
[the export manifest](../provenance/export-manifest.json). Inspection of
the original research workspace found its 696 registered sources, Git
HEAD, tracked status and tracked binary diff unchanged. This supports
preservation of the historical baseline, not a new execution of its full
qualification campaign.

The portable example does not rerun the original 138 controls, historical
source inventories or Windows Job supervisor. It does not establish
binary64 equivalence, time integration, FD/SBP/SAT convergence, physical
boundary stability, AMR, waveform accuracy or external reproduction.
See [scientific status](SCIENTIFIC_STATUS.md) and
[reproducibility instructions](REPRODUCIBILITY.md) for the exact contract.

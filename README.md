# Solar Sim NR

**Research-alpha numerical relativity code and reproducible exact-coefficient checks.**

Solar Sim NR studies a first-order CCZ4 research kernel. This repository
extracts the numerical-relativity sources from a larger application and
provides a small entry point for reproducing a specific scientific check:
the first variation of actual physical-constraint diagnostics at one fixed
background.

This is not a production black-hole simulator. A passing example does not
establish time-evolution accuracy, binary64 equivalence, boundary stability,
AMR correctness or gravitational-wave accuracy. See
[scientific status](docs/SCIENTIFIC_STATUS.md) for the exact scope.

## Minimal reproduction

Use **Python 3.12** in a fresh environment. The example requires
**NumPy 2.2.6**; it does not require SymPy, SciPy, a GPU or an external
numerical-relativity solver.

Create an isolated environment from the repository root. On Windows
PowerShell:

```console
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-minimal.txt
.venv\Scripts\python -B examples/reproduce_constraint_bridge.py --verify-inputs-only
New-Item -ItemType Directory -Force outputs | Out-Null
.venv\Scripts\python -B examples/reproduce_constraint_bridge.py --output outputs/constraint-bridge
```

On Linux or macOS, create the environment with `python3.12 -m venv .venv`
and use `.venv/bin/python` in the remaining commands. No shell activation
is required; create the output parent with `mkdir -p outputs` instead of
the PowerShell `New-Item` command. See the [environment instructions](docs/REPRODUCIBILITY.md#environment)
for the core and development dependency sets.

The metadata command after installation checks input provenance and source
metadata without running the scientific calculation. The final command
performs the complete portable calculation. Use a fresh output directory
for each attempt and retain a failed attempt's records. The final report
is `outputs/constraint-bridge/portable-result.json`; a complete success
requires exit status 0 and `portable_core_reproduction_passed=true`.

The example has a 600-second worker deadline and a 64 MiB evidence-output limit.
The latter is a disk-output limit, not a RAM limit. Runtime depends on the
machine; reaching a limit is a failed or incomplete attempt, not permission
to omit coefficients or loosen acceptance. Worker termination and final
receipt checks can finish after the deadline.

[Reproducibility instructions](docs/REPRODUCIBILITY.md) explain the input
bundle, outputs, dependencies, failure semantics and provenance boundaries.

## What the example checks

The portable example re-executes the source-bound real-formula diagnostic
using 580 independent first-variation coefficients: 58 state components,
174 spatial gradients and 348 symmetric spatial Hessians. It checks:

- The complete physical 43-row constraint map against previously accepted
  polynomial coefficients, including mixed spatial derivatives.
- All 49 raw driver diagnostic scalars, their 40 independent counterparts
  and nine symmetric duplicates.
- Two actual source mutations that must produce detectable disagreements.
- Agreement with a separately implemented reverse scalar differentiation
  algorithm and independent small-matrix arithmetic.

The two algorithms share source adaptation, array routing, chart and
packing code. This is a disclosed common trusted base, not independent
implementations of the physical equations.

The example is a **new portable recomputation**. It is not a rerun of the
original full qualification campaign, its complete test registration,
historical source inventory or Windows Job supervision.

## Repository guide

| Location | Purpose |
| --- | --- |
| [examples/reproduce_constraint_bridge.py](examples/reproduce_constraint_bridge.py) | Minimal portable entry point |
| [examples/data/constraint_bridge/](examples/data/constraint_bridge/) | Frozen parent inputs and reproduction data |
| [docs/SCIENTIFIC_STATUS.md](docs/SCIENTIFIC_STATUS.md) | Established claims and remaining scientific gates |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | How to inspect and reproduce this check |
| [docs/VALIDATION.md](docs/VALIDATION.md) | Actual local results, including retained failed attempts |
| [tests/](tests/) | Repository controls; not a substitute for the full example |
| [provenance/export-manifest.json](provenance/export-manifest.json) | Exported source and parent-input identities |
| [CITATION.cff](CITATION.cff) | Software citation metadata |

The source archive contains research experiments of different maturity.
Some historical native campaign and checkpoint tools depend on Windows,
original machine paths or campaign inputs that are not included. They are
outside the portable example and its CI scope. Presence in this repository
does not establish that a module is independently runnable or validated.

## Contributing

For a reproducibility report, include the commit, Python and NumPy
versions, operating system, exact command, exit status and generated
report. Keep incomplete evidence and describe the failed stage. Do not
change a reference matrix or threshold merely to make a check pass.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup. Scientific
changes should state the equation or observable affected, the
expected behavior and the validation needed. Numerical stability,
convergence and performance are separate questions.

## License and citation

See [LICENSE](LICENSE) for the MIT license. Citation metadata is provided
in [CITATION.cff](CITATION.cff). There is no software DOI or release-paper
claim in this repository's citation metadata.

## 中文简述

这是研究阶段的广义相对论数值内核仓库。最小示例重新计算固定背景下的完整约束诊断，
并进行精确系数比较及独立标量算法重放；它不代表黑洞演化、AMR、边界适定性或波形精度
已经通过，也不等于重跑旧工作区的全部资格检查。

# Reproducing the constraint diagnostic bridge

This guide describes the portable exact-coefficient example. It does not
describe a black-hole simulation or a rerun of the entire historical
qualification environment.

## Environment

Use Python 3.12 and NumPy 2.2.6 in an isolated environment. For this
example, the only third-party numerical dependency is NumPy. On Windows
PowerShell, from the repository root:

```console
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-minimal.txt
```

On Linux or macOS:

```console
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-minimal.txt
```

The commands below use `python` as a short name for the selected environment's
interpreter. Replace it with `.venv\Scripts\python` on Windows or
`.venv/bin/python` on Linux/macOS; activation is optional.

| Dependency file | Scope |
| --- | --- |
| [requirements-minimal.txt](../requirements-minimal.txt) | NumPy for the portable example |
| [requirements-core.txt](../requirements-core.txt) | Minimal dependencies plus SciPy, SymPy and mpmath for other kernels |
| [requirements-dev.txt](../requirements-dev.txt) | Core dependencies plus the pinned pytest dependency set |

For repository controls, install the development file with the same
interpreter and run `python -B -m pytest -q`. These controls and the full
example are separate checks. A configured CI workflow is not evidence that
a hosted run has passed.

SymPy, SciPy, GPU libraries and an installed external relativity solver
are not required for this entry point. Other archived research modules
may have additional dependencies; this minimal dependency statement does
not apply to every file in the archive.

Record the repository commit, platform, Python and NumPy versions with
any result. A numerical dependency pin does not pin all standard-library,
operating-system or hardware behavior.

## Input provenance

The [data directory](../examples/data/constraint_bridge/) contains the
portable reproduction inputs. Thirteen parent JSON files are preserved
as original bytes so their registered SHA256 identities remain meaningful.
The scientific entry point uses a fixed source subset with recorded
source and AST identities; it must reject an unregistered input change.

The [export manifest](../provenance/export-manifest.json) records the
original byte identities. The repository archives 246 NR Python/C++ source files and eight
exact-check scripts. That archive is a research code collection, not an
assertion that every source file was executed by the portable example.
The example's selected source set and the historical campaign's much
larger source/history inventory are distinct.

The historical `native_campaign`, `native_checkpoint`,
`native_continuation` and related campaign harnesses retain Windows or
original-machine assumptions and references to campaign material not
bundled here. They are preserved as research sources, not exposed as
portable entry points or included in the portable example's CI scope.
Installing the core dependency set does not make every archived harness
independently executable.

Two historical gate receipts each contain three supervisor-command path
fields from the original machine. These six paths are retained only as
non-executable historical provenance because editing them would change
the original receipt hashes. They are not installation instructions,
must not be copied into commands, and are not needed to run this example.
The new entry point resolves its own files relative to this repository.

## Inspect before computing

From the repository root:

```console
python -B examples/reproduce_constraint_bridge.py --verify-inputs-only
```

This checks all 267 exported files: 254 sources and 13 parent inputs,
together with JSON qualifications and source/AST metadata for the selected
19-source mathematical closure. It does not import NumPy.
It does not execute the diagnostic mathematics and is not a scientific
pass. A successful metadata check does not establish a derivative identity
or prove that a full run fits the time budget.

## Run the complete portable check

Create the parent directory first: `New-Item -ItemType Directory -Force outputs`
in PowerShell, or `mkdir -p outputs` in a POSIX shell. The requested
run directory itself must not exist. Then run:

```console
python -B examples/reproduce_constraint_bridge.py --output outputs/constraint-bridge
```

The calculation must include all of the following:

1. Exact extraction of every C0/C1/C2 polynomial coefficient into the
   complete 43-by-580 target, including mixed Hessian coefficients.
2. Actual source-bound baseline diagnostic evaluation and both registered
   source mutations using the forward scalar backend.
3. Actual baseline driver evaluation with all 49 raw scalars, the 40-row
   correspondence and nine symmetry comparisons.
4. Reverse scalar differentiation and independent small-matrix arithmetic
   to reconstruct the complete calculations.
5. Exact matrix, source, provenance, acceptance and causal-negative checks.

No axis-only or finite-direction sample is an acceptable replacement.
No projected trace/determinant or reconstructed auxiliary fields may be
used to remove independent perturbations. The accepted real-formula
interpretation and source adaptations remain explicit.

## Outputs and acceptance

The scientific payload contains 13 complete matrices and 206,707 scalar
slots. Three cases each provide background values, 580-column derivatives
and discrepancies from the unchanged oracle. Four driver matrices retain
its raw outputs and complete comparisons. The underlying registered
payload includes 22 matrix/metadata records.

The complete portable output adds five operational records, for 27 JSON
files in total:

| Record | Meaning |
| --- | --- |
| `run.json` | Invocation scope and resource limits |
| `input-verification.json` | Checked archive, source and parent identities |
| `independent-replay.json` | Reverse reconstruction, comparisons and checked payload hashes |
| `worker-result.json` | Scientific worker completion and failure stage |
| `portable-result.json` | Final completion, input audit, worker exit and scope flags |

A successful report requires exact zero of all 53,501 required
background/baseline/agreement slots. Each genuine source mutation must
change a coefficient and disagree with the original oracle at the same
registered slot; changes outside the registered affected rows fail the
control. Original historical counts are reference results, not values
that may be inserted without computation.

Keep the whole output directory, including provenance, execution records,
matrices and independent reconstruction results. Report the actual
machine-readable completion state and process exit status. A metadata-only
check, a partially written directory or a failed attempt does not qualify
the portable scientific result.

For a full run, success requires exit status 0 and
`portable_core_reproduction_passed=true` in `portable-result.json`.
Inspect `passed`, `timed_out`, `owned_worker_stopped` and
`input_audit_passed` there, and preserve the independent replay record.

| Exit status | Meaning |
| --- | --- |
| 0 | Requested operation passed; metadata-only success is not a scientific pass |
| 1 | Verification, calculation, resource or finalization failure |
| 2 | Invalid command-line arguments |
| 124 | Worker deadline reached |

## Resource and failure policy

The portable entry point enforces a maximum 600-second worker deadline and
64 MiB of evidence output, including reserved space for final records.
`--timeout` may reduce the deadline but cannot increase it above 600
seconds. The output limit concerns disk artifacts rather than resident
memory. The limit is not a runtime promise for every machine.

The parent process supervises one owned worker and kills it at the
deadline. Termination waits and final input/hash checks can finish later,
so the command's total elapsed time can exceed 600 seconds. This is not a
portable hard memory or process-tree limit.

Do not increase the accepted limits, silently drop coefficients or adjust
the reference matrix to make an attempt pass. Preserve an incomplete
attempt and identify the phase that failed. There is no automatic retry
or permission to combine outputs from separate attempts into one result.

The new runner's resource handling is portable. It does not reproduce
the original Windows Job ownership model, disk-reservation policy or
complete historical audit. The report must retain
`original_full_qualification_rerun=false` even when this example succeeds.

## What is independent

The second calculation uses a reverse scalar graph and independent
3-by3 cofactor/adjugate matrix arithmetic rather than the forward jet's
coefficient propagation and elimination. It must not generate its answer
from saved forward coefficients.

Both sides still share source AST selection/adaptation, object-array
routing, chart and packing. This is algorithmic cross-checking within a
shared physical-source implementation. It is not an external reproduction
or proof of the full Python/NumPy execution semantics.

## Reporting a result

Include these items in a reproducibility issue or report:

- Commit and local modifications, if any.
- Operating system, Python and NumPy versions.
- Exact invocation and whether it was metadata-only or a complete run.
- Exit status, completion/acceptance fields and elapsed time.
- Input/source identities and the complete generated report directory.
- Any resource limit, interruption or failed stage.

Remove unrelated personal data from a public report, while preserving
the actual checked input identities. If a historical receipt is modified
for a separate publication purpose, identify it as a derived copy and
do not give it the original receipt's digest.

The original local campaign's larger test and historical evidence totals
must not be presented as tests rerun by this portable command. Repository
unit tests likewise do not substitute for the complete scientific example.

See [scientific status](SCIENTIFIC_STATUS.md) for claims this result cannot
support and [the README](../README.md) for project orientation.

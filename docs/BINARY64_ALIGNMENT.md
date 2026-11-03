# Case12 binary64 alignment candidate

**This document registers the experiment, not an acceptance result.**
Actual results require a [separate run receipt](BINARY64_RESULTS.md).
It does not change the established claims in
[scientific status](SCIENTIFIC_STATUS.md), and it is not a binary64
acceptance result.

This candidate follows the [historical design note](../experiments/case12_binary64_v1/protocol-note.md),
originally `NOTE-20261015-case12-binary64-periodic-bridge-next.md`. That note proposed
the experiment; it did not report an execution. The independent repository
keeps the original research workspace and its evidence unchanged.

## Question and scope

The exact-coefficient work connected a rational interpretation of actual
volume and constraint formulas to fixed-case12 linear symbols. This next
experiment asks whether the original NumPy binary64 program, its actual
state conversions and its periodic discrete derivatives reproduce those
linearizations within a registered, resolvable finite-amplitude window.

The calculation includes all 58 RHS fields and all 43 physical constraint
fields, at every node of a single periodic 8-by-8-by-8 grid. It does not
time-step the state. The 49 raw SBP-driver diagnostics are outside this
execution: `driver49_executed=false`. Their earlier exact-formula evidence
must not be relabeled as a binary64 run.

A successful future result would establish only this finite registered
alignment check. It would not establish spatial convergence, time
integration, nonlinear stability, a new uniform hyperbolicity proof,
physical boundary stability, AMR, black-hole evolution or waveform
accuracy. It would not convert the original failed or incomplete
experiments into passes.

## Actual code paths

The positive candidate must execute the original float64 paths, including
their checks and NumPy operations:

| Role | Source/API |
| --- | --- |
| State restoration and slot round trip | [fo_ccz4_checkpoint.py](../science/numerical_relativity/fo_ccz4_checkpoint.py): `FOCheckpoint.to_state`, `state_slots` |
| RHS and geometry | [fo_ccz4_rhs.py](../science/numerical_relativity/fo_ccz4_rhs.py): `fo_ccz4_rhs_3d`, its volume implementation and geometry helpers, with `boundary_mode="periodic"` |
| Complete RHS packing | [fo_ccz4_principal_symbol.py](../science/numerical_relativity/fo_ccz4_principal_symbol.py): `_rhs_components` |
| Physical constraints | [fo_ccz4_relaxation_convergence.py](../science/numerical_relativity/fo_ccz4_relaxation_convergence.py): `constraint_fields`, `derivative`, `symmetric`, `christoffel` |
| Production derivative | [bssn3d.py](../science/numerical_relativity/bssn3d.py): periodic `_d1` |

The physical output wrapper keeps H, M3, Theta, Z3, A3, B9, D18, P3,
determinant and trace. It computes `Z_i = h_ij Ghat^j / 2` with the actual
current float64 metric. The 69-row principal-symbol diagnostic is not a
replacement for these 43 rows. Exact-jet AST adaptations are not an
implementation of this positive binary64 path.

The background remains case12: `alpha=17/20`, `chi=18/25`,
`beta=(3/100,-1/25,1/20)` and `h=L L^T`, with
`L=((1,0,0),(3/25,1,0),(-2/25,3/50,1))`. Other background dynamical and
auxiliary fields vanish. Exact parent values are converted individually
to binary64, and their actual stored values are recorded. Rounded values
must not be described as exact rationals.

The gauge and formulation parameters are fixed:

```text
GaugeConfig:
  advective=true, lapse_coefficient=2, shift_coefficient=1,
  shift_damping=0, theta_coupling=2
FOCCZ4GaugeProfile:
  shift_scale=3, ordering_scale=0.2
FOCCZ4Config:
  kappa1=0.1, kappa2=0, kappa3=1, tau_inverse=1,
  cleaning_speed=1, ordering_mu=1, shift_strength=1, k0=0, c=1
```

The profile's state-dependent coefficients must be evaluated through the
original implementation. All 58 slots remain independent, including
off-constraint perturbations. Symmetric tensor entries share their single
registered slot; auxiliary fields are not reconstructed, determinant and
trace are not projected, and lapse/chi are not clipped. The actual
`exp`/`log` round trip for the logarithmic lapse and conformal-factor
coordinates is recorded. Invalid or non-finite states fail.

## Complete finite registration

Use the unit periodic cube, `N=8`, nodes `x_i=i/8`, both FD2 and FD4, and
one fixed spatial profile:

```text
W = ((1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1))
f(x) = (1 + sum(cos(2*pi*k.x) for k in W)) / 8
mode order = (DC, W, -W)
epsilon_H = 2^-10
epsilon_L = 2^-12
s_a = max(1, abs(q0_a))
q = q0 +/- epsilon*s_a*f*e_a, for every a = 0,...,57
```

All 13 signed/DC modes, both amplitudes and both signs are retained. The
actual sampled profile and its measured nonzero Fourier coefficients are
saved; coefficient recovery cannot silently substitute theoretical
sampling values.

Each implementation records the measured Fourier-weighted real design
matrices (26-by-4 for the RHS and 26-by-10 for constraints), requires ranks
4 and 10 respectively, and records finite condition numbers. This is a
fixed design check, not an adaptive conditioning threshold. The registered
stencil controls also compare the two actual first-derivative routines and
their compositions on all 512 nodes: 36 method/order/axis-composition
records use the fixed `4096*u` comparison gate.

| Evaluations | Per complete implementation path |
| --- | ---: |
| Two orders, two amplitudes, 58 columns, two signs | 464 |
| Complete background at each order | 2 |
| Two registered mutations, two orders, two amplitudes, two signs | 16 |
| Total | 482 |

The independent path repeats all 482 actual state evaluations. The full
registration therefore has 964 state evaluations. Each comprises one
58-field RHS evaluation and one 43-field diagnostic evaluation, for
1,928 scientific calls in total. Started and completed counts must be
recorded. A partial run cannot claim complete alignment. No pilot,
additional amplitude, extra mode or adaptive search is included.

This is a single-grid experiment. Comparing FD2 and FD4 here does not
measure a spatial convergence order.

## Discrete reference and recovery

For grid spacing `h=1/8` and `theta=2*pi*k/8`, the modified wave numbers
are

```text
ktilde_FD2 = sin(theta)/h
ktilde_FD4 = (8*sin(theta)-sin(2*theta))/(6*h)

L58(k) = J + i*sum(ktilde_a*A_a)
C43(k) = C0 + i*sum(ktilde_a*C1_a)
             - sum(ktilde_a*ktilde_b*C2_ab for a<=b)
```

`C2_ab` is the corresponding monomial coefficient of the accepted C2
polynomial. A mixed xy/xz/yz coefficient receives no extra factor of two
or one half. The reference uses two applications of D1, including on the
diagonal; it must not substitute continuous wave numbers or the different
direct D2 stencil.

DC separates J/C0, axis modes separate the three first-derivative blocks
and diagonal second-derivative blocks, and mixed modes separate the three
mixed blocks. Recovery saves four 58-by-58 matrices and ten 43-by-58
matrices per order/amplitude combination: 38,396 coefficient positions.
All four combinations, all zero entries and their complete error matrices
are retained. Conjugate-mode consistency, redundant real/imaginary
relations, full-grid response errors and off-mode content are also
checked. A small error at selected modes alone is insufficient.

## Floating-point acceptance

The implementation fixes the following numerical gates. They are not
proved rounding-error bounds. The code and separate experiment manifest
must be frozen before execution; static source checks do not establish
that a resolvable window exists.

The coefficient check uses reference row scales `w_i=max(1,sum_j abs(Mref_ij))`,
unit roundoff `u=2^-53`, and a measured resolution indicator

```text
nu = 4096*u*B/(epsilon*s_a*abs(fhat_k))
B = max(1, max_grid(abs(Fplus_i)), max_grid(abs(Fminus_i)),
           max_grid(abs(Fbase_i)))
```

The indicator must be propagated through the actual coefficient-recovery
formula, including wave-number divisions. It is not an interval enclosure
or a rigorous IEEE-754 error bound. The implemented slotwise gate is
`abs(error) <= 64*epsilon^2*w_i + 8*nu_recovered`, with fine-amplitude
nonzero reference signals at least 32 times their resolution indicator
and relative error at most `1e-3`. Reference-zero slots still receive
their complete absolute checks.

For each recovered coefficient row, coarse maximum error exceeding
`64*nu_H` requires fine maximum error at most
`error_H/8 + 8*(nu_H+nu_L)`. Other rows are labeled roundoff-limited;
unresolved nonzero coefficients prevent complete acceptance. The
full-field indicator omits the Fourier-coefficient division. Full-field
and modal error gates use `64*epsilon^2` times their independently
assembled reference weights plus eight times the corresponding resolution
indicator. Off-mode maximum amplitude is limited to 14 times the full-field
gate. Background and state-round-trip checks are separate. Even responses,
odd responses and linear reconstruction errors remain saved. Gates cannot
be relaxed after observing an outcome.

## Genuine fault controls

Two actual changes are rerun through the original NumPy paths:

- `kappa1_doubled`: kappa1 changes from 0.1 to 0.2, with registered input
  column 18 (Theta). The unchanged physical diagnostic also executes.
- `stored_trace_diagnostic`: the original `actual_trace` assignment is
  replaced by stored slot 17, with registered input column 11 (Atilde00).
  This changes the whole assignment, including its Hamiltonian use; no
  nonlinear unaffected-row claim is assumed. The original RHS also runs.

Both controls retain all 101 output fields and compare against the
unchanged original reference. Acceptance requires an actual changed input
or source, a resolvable target change, and a detected disagreement at both
orders and amplitudes. Results must not be created by editing a predicted
matrix or adding a desired residual. An undetected mutation fails the
control; it does not authorize a larger mutation or another input column.

## Independent computation and low-memory storage

Production projection uses FFT. The second path independently rebuilds
and evaluates every registered state, uses direct discrete cosine/sine
projection, and separately implements coefficient recovery, reference
assembly, error summaries and acceptance. It must not read the production
projections as its scientific answer. Both paths share the original
physical source, state containers, NumPy and platform floating-point
libraries; this is not an independent implementation of general
relativity.

Process states sequentially rather than retaining all 964 grid states.
Retain complete registered projections, recovered matrices and errors in
fixed-shape arrays, together with all-node error summaries and ordered
field hashes. Little-endian float64 storage, explicit shapes, complex
real/imaginary layout and no-pickle decoding must be registered. Small
entries are never thresholded to zero.

The main signed projection array per path has shape
`(2,2,58,2,101,13,2)` and 9,747,712 bytes of raw float64 data. Background
and mutation projections, both paths' full matrices, metadata, controls
and terminal space also count toward the output budget. File-by-file
uncompressed bounds must fit before launch; expected compression is not
a capacity guarantee.

Field hashes and all-node norms are not complete raw-grid archives. This
experiment does not claim to save every signed 101-field grid. The
independent path must actually recompute those grids and check every node.
Different projection summation orders need not be bitwise identical;
their comparison uses a separately frozen tolerance tighter than the
scientific truncation allowance. Exact reference/background/state fields
and all 482 ordered state/actual-output hashes must match between paths.
The call ledger is also checked against the registered sequence, including
all labels, indices, mutations and four SHA-256 fields; two identical but
incomplete ledgers cannot pass. Each actual call records its elapsed time
including guards. Positive field scales and modal/coefficient resolution
indicators are independently rebuilt from saved field maxima, amplitudes,
state scales and measured Fourier coefficients. Zero, negative or non-finite
indicators are invalid, and both paths cannot inflate them together.
The saved maxima are compared between independently executed paths; without
all signed raw grids, this is not a third recomputation of those maxima.
Compact axis-real/DC, conjugacy and signed-mixed reconstruction controls
use triangle propagation of the original modal limits. Some reconstruction
relations follow directly from coefficient recovery; these are consistency
controls, not additional physical validation.
Other arrays use registered roundoff/resolution comparisons; recorded
argmax indices are not compared because equal maxima can have different
indices. Unexpected nondeterminism in the same source path must
be recorded and investigated rather than silently ignored.

## Resource and provenance gates

The runner fixes a **600-second deadline** covering preflight, input
verification, snapshot creation, calculation, cross-comparison, normal cleanup,
sealing and successful receipt writing. Failure
cleanup and final receipt writing may finish after that deadline. A
`--timeout-seconds` value from 1 through 600 can shorten it; it cannot
increase it. The evidence-output cap is **64 MiB**.

Launch requires **4 GiB of available system RAM**, with a **2 GiB**
runtime reserve. Installed RAM and swap do not replace available RAM;
an unknown or insufficient reading blocks launch. The startup gate is sampled
again immediately before spawning and releasing the worker. Snapshot checks
also monitor supervisor RAM, including a 1 GiB supervisor RSS/private-byte cap.
On Windows an owned Job
enforces hard **1 GiB committed-memory** limits for the worker and the whole
owned Job, with
additional RSS/private-byte sampling. On Linux a watchdog samples RSS
against 1 GiB; it is not a hard address-space limit. Resource monitoring
supports 64-bit Python on Windows and Linux. Output-volume free space must
be at least **4 GiB + 32 MiB + the full 64 MiB batch** before launch.
Fresh runtime checks preserve **4 GiB + 32 MiB** on that volume;
Windows also preserves **3 GiB** on the system volume.

The registered two-path raw array payload is **51,013,376 bytes**, across
26 arrays per path, including the mutation field scales needed to rebuild
their resolution indicators. Capacity accounting adds every NPY header,
an exact uncompressed `ZIP_STORED` bound with an **8 MiB snapshot cap**,
a **4 MiB JSON allowance** and a **2 MiB final-receipt reserve**.
The manifest's final byte calculation must fit
64 MiB before launch. This is a capacity bound, not measured output or RAM
usage. The terminal file is allocated, flushed and fsynced before worker
release with an initial unsuccessful receipt. Finalization overwrites the
same held file at its fixed size, even if a later disk or directory scan
fails; hardware/write failures still prevent acceptance. Each path's small
`progress.json` is atomically overwritten; other evidence is write-once.
Before a successful seal, cleanup must confirm that the owned Windows Job
or Linux process group is empty, including descendants whose leader has
exited. Unexpected residuals, failed queries or failed handle closure reject
the result. The final census must exactly match the sealed inventory plus
the reserved terminal file. Filesystem scan errors and trailing deadline
checks fail closed.

Preflight and input verification use only the standard library and do
not run scientific functions. One owned worker handles the complete
registered calculation. No automatic retry, partial-result stitching,
deletion of earlier evidence, or larger resource allowance is included.
An interruption, unresolved coefficient, incomplete replay, failed
control or limit violation preserves an unsuccessful result.

A separate experiment manifest must bind the new entry point, production
and independent implementations, controls, this protocol, numerical
constants, exact parent/reference data and their source identities. The
original [export manifest](../provenance/export-manifest.json) and all
original archived scientific sources remain unchanged. Earlier continuum
true flags are not repurposed as binary64 acceptance flags.
Source reads reject oversized files before allocation and read at most one
byte beyond the registered size to detect growth during the read. Manifest
and export metadata each have a 1 MiB raw cap; registered source bytes plus
the manifest are bounded before loading. These checks precede scientific imports.

One metadata file has an explicit text representation: the existing export
manifest is CRLF in the current Windows working directory and LF in its
already committed Git blob. This candidate binds its LF representation to
that existing blob's SHA-256, and archives that representation without
rewriting the working file. All scientific source files, parent receipts
and the historical protocol note remain bound to their original raw bytes.

After manifest authentication, metadata-only validation checks three
historical qualification groups (volume, coordinate and constraint),
their actual completion/acceptance fields, embedded replay identities and
62 byte-digest crosslinks across 24 parent files. This includes the 11
newly copied raw inputs and the 13 existing exported parents. Worker
hashes are read from `artifact_hashes["worker-result.json"]`; historical
`terminal_sha256` has a different meaning. The metadata check does not
rerun the parents or authenticate an arbitrary replacement manifest.

The [reference reader](../experiments/case12_binary64_v1/provenance.py)
does not generate a scientific reference model. It reads nine original
JSON records: F0/J/Ax/Ay/Az, their execution parameters, and C0/C1/C2.
Those records contain eight matrices. It checks every row-major slot,
canonical rational coefficient, polynomial variable/term order and
homogeneous degree, together with the complete fixed-case12 parameter
and state order. F0 must be exactly zero. It returns the four RHS and ten
constraint coefficient blocks without changing mixed-derivative factors.

The reader records the exact rational difference between each converted
binary64 coefficient and its parent rational value, including every zero
slot and numeric background/configuration parameter. Non-finite conversion
or a nonzero reference rounded to zero is rejected. This decoding is not
provenance authentication: manifest verification must occur first. NumPy
is imported only by the worker's reference-conversion operation, after
the exact input checks.

## Commands and result semantics

The entry point is [reproduce_binary64_alignment.py](../examples/reproduce_binary64_alignment.py).
The commands below use its implemented interface. They are not a report
that the program or experiment has run.

```console
python -B examples/reproduce_binary64_alignment.py --verify-inputs-only
python -B examples/reproduce_binary64_alignment.py --preflight-only
python -B examples/reproduce_binary64_alignment.py --output outputs/binary64-case12
```

Use the selected virtual environment's Python, with the repository's
minimal NumPy dependency installed. Metadata modes import only the
standard library and do not evaluate scientific functions. `--verify-inputs-only`
checks source/input identity and historical receipt consistency;
`--preflight-only` additionally checks resources and output capacity.
Outputs must remain inside the repository's `outputs` directory. The
selected parent must exist, and `outputs/binary64-case12` must not exist.
Successful metadata verification or preflight is not scientific acceptance.
A scientific launch requires frozen implementation and a passing resource
preflight. On Windows, the worker launches the base CPython executable with
the virtual-environment identity preserved, avoiding a launcher child before
Job assignment. Its release handshake checks the actual PID and Python prefix
before scientific imports. Controls and run receipts remain separate evidence.

Exit code 0 denotes successful metadata-only handling or a fully accepted
scientific run, according to the selected mode. A started unsuccessful run
returns 1; invalid arguments or a failure before the bounded run starts
return 2. Acceptance uses two stages: the fixed terminal file always keeps
`accepted=false` and `acceptance_requires_supervisor_commit=true`. A candidate
whose checks passed has state `CHECKS_PASSED_AWAITING_SUPERVISOR_EXIT`.
Only after writing, flushing, fsync, closing and the final deadline check
does the supervisor emit a compact stdout commit with `PASSED`,
`accepted=true` and the SHA-256 of the complete padded terminal file.
Qualification requires **all three**: normal exit 0, that explicit final
stdout commit, and a matching terminal digest. A terminal file alone can
never qualify. A late write/close/deadline failure cannot leave a standalone
successful receipt, even if a corrective failure write also fails.
Failed runs retain `NOT_QUALIFIED` or a provisional terminal;
prelaunch failures print `NOT_STARTED`. A failure while reserving terminal
storage can leave a fresh directory or incomplete file; it is preserved.
If the destination cannot safely accept the terminal receipt, the console
reports `terminal_available=false` and its reason. A missing receipt never
counts as a pass. Exit code alone does not identify a scientific pass.

Pass counts, runtime, memory peaks and coefficient errors belong in the
execution records. Actual results, including failures, must be derived
from completed receipts; this registration cannot stand in for them.

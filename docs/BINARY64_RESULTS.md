# Binary64 execution records

This result log is separate from the frozen experiment registration.
It records bounded controls and numerical outcomes, including failures.

## Local controls, 2026-10-19

The first run completed 335 controls with two Linux-signal mock failures on
Windows. Its failed receipt is retained locally. A separate startup attempt
was refused before testing when available memory fell below 4 GiB.

After fixing those mocks and the Windows virtual-environment worker launch,
the second executed run passed 339 controls in 5.86 seconds including supervision.
Its peak worker RSS was 130,539,520 bytes, with an empty, closed Windows Job
at completion. These are unit and integration controls, not the 964-state
scientific experiment. The initial runner sampled the outer virtual-environment
launcher, so its old memory peak is not a scientific or actual-worker measurement.

## Scientific execution

The first full-entry launch was refused at preflight with `NOT_STARTED`,
exit 2 and `numerical_execution=false`: available RAM was below 4 GiB.
No scientific output directory was created. A subsequent direct resource
sample reported 2,308,276,224 available bytes. This is not a failed physics
test, a partial numerical pass, or permission to lower the startup gate.

No completed scientific result is recorded here yet. Frozen registration:
`dba2b44c3e010b2a0f56514ae7af291b8b6c42d2b5613583272a02b69cd89e02`.
The 600-second, 64 MiB output, 1 GiB worker and 4 GiB available-RAM startup
limits remain unchanged. The first failed controls and launch console are
retained under local ignored output/release directories. Nothing has been
retried after a started scientific failure; no scientific campaign has started.

The final registration differs from the passing controls registration only
in README/status/registration prose. Its numerical, runtime and test source
bytes are unchanged. The source snapshot's registered size is 8,379,829 bytes;
the complete output upper bound is 65,691,317 bytes, below 67,108,864 bytes.

## Continuation readiness, 2026-10-28

A read-only continuation check authenticated all 304 registered files and
the unchanged manifest. Those 305 files also match their raw bytes in the
2026-10-23 preservation snapshot. The eight test-source hashes still match
the earlier 339-control receipt; those controls were not rerun today.

The resource query reported 2,835,197,952 available RAM bytes, below the
unchanged 4,294,967,296-byte startup floor. Output and system disk space
passed their registered capacity requirements. Preflight therefore refused
execution. No scientific worker or new scientific output directory was
created, and no numerical acceptance claim changed.

The machine-readable local record is
`.release-local/continuation-readiness-20261028.json` (ignored, not included
in the public source repository). This readiness check does not modify the
sealed 2026-10-23 backup or replace the earlier refused-launch record.

## Isolated startup-2-GiB continuation, 2026-11-03

The separately registered resource variant passed all 339 bounded controls.
Two actual scientific workers subsequently started, reaching durable production
progress checkpoints of 31/482 and 73/482 completed evaluations. Both were
stopped by the unchanged 3 GiB system-volume disk guard, with empty, closed owned
Windows Jobs and preserved partial artifacts. Neither completed the required
964-state two-path experiment; both remain **NOT QUALIFIED**. The original v1
registration and all 304 bound input files remain unchanged.

See [the continuation report](BINARY64_CONTINUATION_20261103.md) for exact
lineage, local evidence links, scope limits and the remaining resource blocker.
Earlier dated observations above remain historical records of those runs.

## Run 04, 2026-11-03

Production path completed all 482 evaluations; replay stopped at 477/482 when
the 2 GiB runtime RAM reserve was crossed. Production acceptance failed on one
of 99,760 physical43 slots (FD4, eps 2^-10, error/limit 1.005); the error scales
exactly as eps^2, so it is the expected amplitude term exceeding the registered
constant 64, not a code fault. **NOT QUALIFIED.** See
[the run 04 report](BINARY64_RUN04_20261103.md).

## v2 Richardson registration, 2026-11-03

A new registration replaced the per-amplitude 64·eps² slot bound with Richardson
elimination of the eps² term (written after run 04; disclosed). Science run 02
**PASSED**: both paths 482/482, comparison accepted, all negative controls detected,
worst slot ratio 6.1e-05. Run 01 stopped on a transient RAM-reserve breach. See
[the v2 report](BINARY64_V2_RICHARDSON_20261103.md).

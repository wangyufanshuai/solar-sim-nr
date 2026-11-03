# Binary64 continuation — 2026-11-03 (Asia/Tokyo)

## Material Passport

- Mode: run and read-only evidence validation.
- Repository: `E:/xuexi/solar-sim-nr`, branch `codex/research-alpha`.
- HEAD at entry: `583e2e5ecfa871c5462e1316a61fd7f7716f6bb6`.
- Status: **PARTIAL / RESOURCE-STOPPED / NOT QUALIFIED**.
- Original v1 manifest:
  `dba2b44c3e010b2a0f56514ae7af291b8b6c42d2b5613583272a02b69cd89e02`.
- Isolated startup-2-GiB manifest:
  `1e148ddf37b916da22428c665c366daebaff8496fc27875c1f6f33a9ae98d0e7`.
- Scope: binary64 alignment candidate only; no live dynamics writeback,
  scientific promotion, time evolution, or physical-case qualification.
- Raw receipt timestamps use UTC; their local calendar date is November 3
  in the client timezone, Asia/Tokyo.

## Actual results

The original 304 registered inputs and both candidate identities were rechecked.
No scientific formula, numerical threshold, grid, amplitude, stencil, negative
control, time limit, or output limit changed. The isolated candidate retains
the authorized 2 GiB startup floor, 2 GiB runtime reserve, 1 GiB worker limit,
600-second deadline, 64 MiB output limit and 3 GiB system-volume disk floor.
The original 4-GiB v1 candidate is unchanged.

| Evidence | Observed result |
| --- | --- |
| Continuation 01 | Metadata preflight refused: C: below 3 GiB. No scientific worker. |
| Completion-receipt controls | 10 metadata control groups passed, including corrupt commits, partial terminals, deadline, cleanup, scope flags and rejection of the real refused attempt. No numerical evaluator imported. |
| Full bounded controls, continuation 02 | **339 passed**, pytest 9.05 s; supervisor 9.656 s; peak worker RSS 131,088,384 bytes; owned Job empty and closed. |
| Actual scientific run 02 | **NOT QUALIFIED**, supervisor exit 1 after 9.391 s; production checkpoint 31 completed / 32 started of 482; C: dropped below 3 GiB. |
| Actual scientific run 03, after user's next continuation | **NOT QUALIFIED**, supervisor exit 1 after 26.828 s; production checkpoint 73 completed / 74 started of 482; same disk-floor refusal. |

Run 03 reused the passing control receipt after rehashing its eight test sources
and verifying the unchanged complete candidate manifest. It did not rerun the
339 tests. Each scientific invocation used a fresh, distinct directory;
there was no retry loop, checkpoint resume, accumulation of partial counts,
threshold reduction, or replacement of earlier failure receipts.

Both interrupted workers were terminated through their owned Windows Jobs.
Both cleanup records report `empty=true`, `closed=true`, and
`unexpected_residuals=false`. Their sampled peak RSS values were 73,445,376 and
75,132,928 bytes respectively. These are partial-run observations, not proof
that the complete experiment fits any specified machine.

The last durable progress checkpoints do not prove the exact number of calls
completed between the last checkpoint and termination. Full call ledgers were
not finalized, the replay path did not start, and the required 964 total states
were not completed. Neither run establishes floating-point numerical agreement
or a physics failure. Both remain resource-stopped, unqualified experiments.

## Preservation and verification

The saved partial-output inventories, every sealed artifact hash, and all 305
source/input snapshot entries were read back and matched. Runs 02 and 03 retained
12,293,022 and 12,293,024 bytes respectively. Original registered inputs remained
unchanged. The new completion gate correctly rejected the failed run; its
passing-run integrity path has not been exercised on a real successful scientific
run. It checks evidence and does not perform another numerical recomputation.

- [Final lineage and artifact hashes](../.release-local/binary64-startup2g-20261029/outputs/continuation-summary-20261103.json)
- [Passing 339-control receipt](../.release-local/binary64-startup2g-20261029/outputs/controls-startup2g-20261103-02/controls-result.json)
- [Raw pytest log](../.release-local/binary64-startup2g-20261029/outputs/controls-startup2g-20261103-02/pytest.log)
- [Run 02 terminal](../.release-local/binary64-startup2g-20261029/outputs/binary64-startup2g-20261103-02/terminal.json)
- [Run 03 terminal](../.release-local/binary64-startup2g-20261029/outputs/binary64-startup2g-20261103-03/terminal.json)
- [Run 03 captured supervisor stdout and exit lineage](../.release-local/binary64-startup2g-20261029/outputs/resume-20261103-03/receipt.json)
- [Run 03 rejected completion gate](../.release-local/binary64-startup2g-20261029/outputs/resume-20261103-03/completion-gate.json)

Local `.release-local/` evidence is ignored by Git and is not part of a public
reproduction package. No push, frontend modification, user-file cleanup, or
termination of another task's process was performed.

## Remaining dependency

All three scientific milestones remain incomplete. Full binary64 acceptance
must precede short evolution/convergence and the final analytic physical case.
The current blocker is C: fluctuating around the unchanged 3 GiB floor.
Reserve approximately **5 GiB free on C:** before another supervised attempt
to provide practical headroom; this recommendation does not replace or lower
the registered 3 GiB hard guard. Any later attempt needs a fresh directory and
must preserve these interrupted runs.

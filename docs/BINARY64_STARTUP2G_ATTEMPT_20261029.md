# Binary64 startup-2-GiB resource variant — 2026-10-29

Status: **NOT_STARTED / resource preflight refused**. No scientific worker,
NumPy experiment, time evolution, or scientific qualification was performed.
The three proposed numerical milestones remain incomplete.

The user explicitly requested a trial with 2 GiB available physical RAM. A
separate local candidate was prepared at
`../.release-local/binary64-startup2g-20261029/`. The original v1 registration
and its 304 registered inputs were rechecked and preserved byte for byte.

## Registered change

- Parent manifest SHA256:
  `dba2b44c3e010b2a0f56514ae7af291b8b6c42d2b5613583272a02b69cd89e02`.
- Resource-variant manifest SHA256:
  `1e148ddf37b916da22428c665c366daebaff8496fc27875c1f6f33a9ae98d0e7`.
- Startup available-RAM floor: **4 GiB → 2 GiB** in the isolated copy only.
- Runtime available-RAM reserve: **2 GiB**, unchanged. A worker allocating
  memory can still cause a stop if this reserve is crossed.
- Worker cap: **1 GiB**; deadline: **600 seconds**; output cap: **64 MiB**;
  system-volume free-space floor: **3 GiB**, all unchanged.
- Only the copied `runtime.py` startup constant and its error text changed.
  Scientific sources, input matrices, numerical thresholds, grid, modes,
  amplitudes, negative controls, and production/replay computations are identical.
- The variant does not qualify the original 4-GiB v1 registration.

## Actual observations

Input verification passed for all 304 bound files. The registered output upper
bound is 65,691,921 bytes, below 64 MiB. Six standard-library resource-contract
checks passed, including rejection one byte below the RAM and disk boundaries.
These are mocked boundary controls, not numerical validation. The full existing
339-control suite was **not rerun**: its launcher was refused before spawning
a worker because system-volume space was below 3 GiB.

One subsequent live **metadata-only preflight** exited 2 and returned:

```json
{"exception_type":"ValueError","message":"need 3 GiB system-volume free space","numerical_execution":false,"state":"NOT_STARTED"}
```

Immediately before that preflight, available RAM was 2,369,519,616 bytes
(about 2.21 GiB), system-volume free space was 2,283,765,760 bytes
(about 2.13 GiB), and the output volume had 68,121,042,944 bytes free.
Thus the observed refusal was the original system-disk guard. Memory also
fluctuated below 2 GiB during preparation; this remains an independent runtime
constraint. No scientific output directory was created, no owned worker was
spawned, and no automatic retry or resource-threshold reduction followed.

## Evidence and continuation

- [Preparation and original-input preservation](../.release-local/binary64-startup2g-20261029/preparation.json)
- [Bound resource-attempt receipt](../.release-local/binary64-startup2g-20261029/outputs/resource-preflight01/receipt.json)
- [Raw preflight stdout](../.release-local/binary64-startup2g-20261029/outputs/resource-preflight01/preflight.stdout.json)
- [Raw preflight stderr](../.release-local/binary64-startup2g-20261029/outputs/resource-preflight01/preflight.stderr.txt)

These are local evidence links; `.release-local/` is ignored by Git and is not
part of a portable or published reproduction. No push or frontend modification
was performed for this attempt.

After system-volume free space reaches at least 3 GiB and available RAM reaches
at least 2 GiB, use a fresh evidence directory, rerun the bounded controls,
and make a single supervised full binary64 attempt with the registered limits.
Preserve this refusal and any future failures. Only an accepted full numerical
result permits dependent short-evolution and convergence experiments to begin.

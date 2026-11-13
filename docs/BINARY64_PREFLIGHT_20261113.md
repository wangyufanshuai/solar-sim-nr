# Binary64 8³ preflight receipt — 2026-11-13

The registered binary64 candidate at commit a57ef77 was checked in a
temporary worktree with the original Git bytes preserved. The metadata-only
command completed with exit 0 and manifest SHA-256
dba2b44c3e010b2a0f56514ae7af291b8b6c42d2b5613583272a02b69cd89e02.

The resource-only command was then run with a fresh output path. It returned
exit 2 with need at least 4 GiB available RAM. The observed workstation
snapshot was about 2.07 GiB available RAM, 12.46 GiB free on the system
volume and 79.18 GiB free on the output volume. The entry point reported
numerical_execution=false and state=NOT_STARTED. No NumPy import, worker
release, array allocation or scientific evolution occurred.

The registered envelope remains 600 seconds, 64 MiB output, a 1 GiB worker
commit cap and at least 4 GiB available RAM. The refusal is the expected
resource gate. A successful metadata check or resource preflight is not
binary64 scientific acceptance.

The low-memory control suite was run separately with the four registered
runtime, supervisor, containment and kernel test modules. It passed 98 tests
in 0.94 s. These are control checks, not the 482-evaluation scientific run.


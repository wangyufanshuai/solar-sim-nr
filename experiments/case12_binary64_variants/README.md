# Binary64 8³ registration variants

The original registration in `../case12_binary64_v1` is never edited. Each variant is
generated as an isolated, separately hashed copy under `.release-local/` (ignored by Git).

| Variant | Manifest SHA256 | Change | Result |
| --- | --- | --- | --- |
| v1 | `dba2b44c…89e02` | original (4 GiB startup RAM floor) | not run to completion |
| startup2g | `1e148ddf…0d0e7` | startup RAM floor 4 → 2 GiB only | run 04 NOT QUALIFIED ([report](../../docs/BINARY64_RUN04_20261103.md)) |
| v2 Richardson | `fb4bd290…09e01` | slot bound: Richardson elimination of eps² | run 02 PASSED ([report](../../docs/BINARY64_V2_RICHARDSON_20261103.md)) |

v2 was registered after seeing the startup2g run 04 result; see the report.

## Reproduce

From the repository root (Windows; the supervisor uses a Windows Job):

```
python experiments/case12_binary64_variants/prepare_startup2g.py
python experiments/case12_binary64_variants/prepare_v2_richardson.py
cd .release-local/binary64-v2-richardson-20261103
python -B .release-local/run_bounded_controls.py controls-01
python -B examples/reproduce_binary64_alignment.py --preflight-only --output outputs/run-01
python -B examples/reproduce_binary64_alignment.py --output outputs/run-01
```

Each scientific run needs a fresh output directory, at least 2 GiB available RAM for the
whole run (about 2 minutes), and 3 GiB free on the system volume. Do not retry into an
existing directory.

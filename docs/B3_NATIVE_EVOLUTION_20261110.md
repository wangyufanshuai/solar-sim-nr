# B3 step 2 — native RK4 reproduces S2 and B1 (2026-11-10)

Registration: [ROADMAP_B.md](ROADMAP_B.md), commit before the runs (pass if every recorded error
is reproduced to ≤ 1 % relative and every gate verdict is unchanged). Status: **PASSED** — worst
relative difference 5.4e-8; all errors agree to 8 significant digits, so all verdicts are the
same. Numbers: [b3-repro-20261110.json](b3-repro-20261110.json).

`kernel/native_evolve.py`: classical RK4 on the 58 canonical slots with the native RHS; B1's
harmonic-slicing correction is applied on the slots exactly as in
`experiments/b1_gauge_wave/harmonic.py`. Runner: `experiments/b3_native/reproduce.py`;
references: the recorded Python outputs `outputs/s2-20261105-01`, `outputs/b1-20261108-01`.
Regression test: `tests/test_native_evolve.py` (3 RK4 steps vs `advance_fo_ccz4_rk4_3d`).

| case | worst relative difference | Python wall time | native wall time (16 threads) |
| --- | --- | --- | --- |
| S2-a FD2, Nx 16/32/64 | 6.1e-14 | 28 / 59 / 250 s | 0.14 / 0.57 / 2.5 s |
| S2-a FD4, Nx 16/32/64 | 1.6e-10 | 16 / 66 / 317 s | 0.15 / 0.56 / 2.4 s |
| S2-b FD4 Nx 32, CFL 0.8/0.4/0.2 vs 0.1 | 2.5e-8 | 15 / 28 / 53 s | 0.07 / 0.12 / 0.24 s |
| B1-a, A 0.01 and 0.1, FD2/FD4, Nx 16/32/64 | 1.7e-10 | 28–306 s each | 0.2–3.4 s each |
| B1-b, A 0.01, Nx 32, T = 100 | 5.4e-8 (error at T = 100); late slope 4.0e-8 | 4578 s | 38.7 s |

The largest differences are where the measured quantity is smallest relative to the fields (the
S2-b time-step differences at A = 1e-6 and the 100-crossing run), i.e. accumulated rounding, as
expected for two codes that sum in different orders.

## What this enables

The registered S2 and B1 suites, which took about 40 minutes and 76 minutes in Python, now take
about 10 seconds and under a minute. Next (B3 step 3): native SBP volume operator and SAT so the
parked B2 items can be rerun, then the single-black-hole setup.

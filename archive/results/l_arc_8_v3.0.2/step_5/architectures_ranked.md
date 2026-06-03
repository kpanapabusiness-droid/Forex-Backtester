# l_arc_8_v3.0.2 — Step 5 architectures ranked

_Generated: 2026-05-25T22:17:37.757204+00:00Z_

**Verdict:** `FAIL`

Candidate cluster: c2 (Bimodal). Arch set per Amendment 5: {A1, A4}.
Exit slate (Bimodal canonical): {sl_only, sl_partial_close_1r_runner_trail, sl_plus_tp_2r}.

## Top-K candidates by worst-fold ratio

| Rank | Config | Worst-fold ratio | Worst-fold DD% | Mean fold ROI% | Base verdict | Amended verdict |
|---:|---|---:|---:|---:|---|---|
| 1 | `A1::a1_sl1.5_exit-sl_partial_close_1r_runner_trail_exp-max_per_currency_2` | -0.8860888007390031 | 23.88% | -7.16% | fail | fail |
| 2 | `A1::a1_sl1.5_exit-sl_partial_close_1r_runner_trail_exp-unlimited` | -0.8890681496325287 | 31.04% | -5.99% | fail | fail |
| 3 | `A1::a1_sl2.0_exit-sl_partial_close_1r_runner_trail_exp-max_per_currency_2` | -0.9093355262549794 | 23.87% | -3.12% | fail | fail |

## Amendment 3 evaluation (top-K)

| Config | Amended verdict | Chained DD% | k_safe | k_hard | r_safe% (deploy) | r_hard% (deploy) | r_safe intrinsic% | r_hard intrinsic% | safe capped | hard capped | scalable_safe | scalable_hard | primary_failure_mode |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|---|
| `A1::a1_sl1.5_exit-sl_partial_close_1r_runner_trail_exp-max_per_currency_2` | fail | 69.2391% | — | — | — | — | — | — | N | N | N | N | PrimaryFailureMode.STEP5_CHAINED_DD_ABOVE_GATE |
| `A1::a1_sl1.5_exit-sl_partial_close_1r_runner_trail_exp-unlimited` | fail | 73.4713% | — | — | — | — | — | — | N | N | N | N | PrimaryFailureMode.STEP5_CHAINED_DD_ABOVE_GATE |
| `A1::a1_sl2.0_exit-sl_partial_close_1r_runner_trail_exp-max_per_currency_2` | fail | 46.1419% | — | — | — | — | — | — | N | N | N | N | PrimaryFailureMode.STEP5_CHAINED_DD_ABOVE_GATE |

## Step 6 (auto-dispatch per Amendment 4)

- Not dispatched (no PASS-tier candidate cleared §3 constraints #1-9).

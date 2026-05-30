# Arc 10 v3.0.2 — Path Analytics (EET, descriptive)

> Read-only aggregation over the EET v3.0.2 WFO path frames. No config / signal / exit-policy change; no re-simulation of an alternative policy. The deployed policy `sl_partial_close_1r_runner_trail @ 3.5xATR` is replayed over the recorded per-bar path via the canonical `simulate_path` (runner legs byte-identical to live).

## Units & conventions

- **2.0R frame** = recorded `mae/mfe_so_far_r`, R-units of Step-1 base SL=2.0xATR. Distribution tables (terminal MAE/MFE) are in this frame; the dispatch's `mfe_p50 ~ 5.31R` cohort anchor is 2.0R-frame.
- **3.5R frame** = deployed risk unit (SL=3.5xATR). Realized R, give-back, leg decomposition are 3.5R-frame (scale = 2.0/3.5 = 0.5714).
- **outcome (win/loss)** is defined on the **deployed** realized R (`sl_partial_close_1r_runner_trail @ 3.5xATR`), not the raw Step-1 sl_only `final_r`.
- **terminal_mae/mfe (primary)** computed over the **deployed-policy held window** (entry -> deployed exit bar). The literal `is_held` (2.0 sl_only) window is reported as `_held` columns; it truncates earlier and floors losers near the 2.0 SL, so it is NOT the primary cut. See flag below.
- **fold** = OOS calendar year (fold k -> 2010+k-1); **holdout** = fold 12 (2021-01-01..2026-04-30). Tagged by `entry_time` year.

## Sample sizes

| segment | n |
|---|---|
| fold | 2059 |
| holdout | 1093 |

| segment | fold | n |
|---|---|---|
| fold | 1 | 201 |
| fold | 2 | 182 |
| fold | 3 | 179 |
| fold | 4 | 195 |
| fold | 5 | 192 |
| fold | 6 | 190 |
| fold | 7 | 171 |
| fold | 8 | 190 |
| fold | 9 | 176 |
| fold | 10 | 195 |
| fold | 11 | 188 |
| holdout | 12 | 1093 |

Outcome split (deployed policy):

| outcome | n |
|---|---|
| loss | 903 |
| win | 2249 |

Cluster split:

| cluster | archetype | n |
|---|---|---|
| 0 | v_shape_recovery | 1493 |
| 1 | monotonic_down | 1658 |
| 2 | monotonic_down | 1 |


---

# Task A — Entry / post-entry MAE

## A — load-bearing read: win-vs-loss terminal MAE gap (2.0R frame)

Negative = dip below entry. `terminal_mae_r` over deployed-held window. If winners dip as deep as losers, a lower limit fill is plausibly free; if winners dip *less*, a lower limit adverse-selects.

| percentile | win | loss | gap_win_minus_loss |
|---|---|---|---|
| p1 | -7.1557 | -3.2119 | -3.9439 |
| p5 | -4.0922 | -2.6088 | -1.4834 |
| p10 | -3.1738 | -2.3882 | -0.7856 |
| p25 | -1.9543 | -2.1305 | 0.1762 |
| p50 | -1.0257 | -1.9283 | 0.9026 |
| p75 | -0.4242 | -1.8257 | 1.4015 |
| p90 | -0.1781 | -1.7742 | 1.5962 |
| p95 | -0.1083 | -1.7601 | 1.6518 |
| p99 | -0.0333 | -1.0272 | 0.9939 |

Same gap on the **early dip** (worst MAE within first 3 bars, the limit-entry-relevant window):

| percentile | win | loss | gap_win_minus_loss |
|---|---|---|---|
| p1 | -1.5778 | -2.5638 | 0.9859 |
| p5 | -1.1337 | -1.3464 | 0.2127 |
| p10 | -0.9144 | -1.0706 | 0.1562 |
| p25 | -0.5883 | -0.7324 | 0.1441 |
| p50 | -0.3318 | -0.4325 | 0.1006 |
| p75 | -0.1794 | -0.2373 | 0.0579 |
| p90 | -0.0872 | -0.1152 | 0.0280 |
| p95 | -0.0533 | -0.0766 | 0.0233 |
| p99 | -0.0207 | -0.0197 | -0.0009 |

## A1 — terminal_mae_r distribution (2.0R frame)

| cut | n | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled (deployed-held) | 3152 | -6.5825 | -3.6578 | -2.8469 | -2.0713 | -1.7516 | -0.6743 | -0.2415 | -0.1373 | -0.0428 | -1.6141 |
| win (deployed-held) | 2249 | -7.1557 | -4.0922 | -3.1738 | -1.9543 | -1.0257 | -0.4242 | -0.1781 | -0.1083 | -0.0333 | -1.4522 |
| loss (deployed-held) | 903 | -3.2119 | -2.6088 | -2.3882 | -2.1305 | -1.9283 | -1.8257 | -1.7742 | -1.7601 | -1.0272 | -2.0174 |
| pooled (is_held / 2.0 sl_only) | 3152 | -2.4924 | -1.7465 | -1.5386 | -1.2846 | -1.1016 | -1.0042 | -0.6746 | -0.3352 | -0.1071 | -1.1601 |

## A2 — bars_to_mae_trough distribution (deployed-held window)

| cut | n | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled | 3152 | 0.0 | 0.0 | 0.0 | 4.0 | 16.0 | 37.0 | 68.0 | 92.0 | 149.0 | 26.7 |
| win | 2249 | 0.0 | 0.0 | 0.0 | 2.0 | 11.0 | 31.0 | 62.2 | 85.0 | 134.5 | 22.3 |
| loss | 903 | 2.0 | 4.0 | 7.0 | 14.0 | 27.0 | 49.0 | 82.0 | 103.9 | 176.0 | 37.5 |

## A3 — limit-fill feasibility grid (2.0R frame)

Fraction of trades whose post-entry dip reaches >= X R below entry within the first k bars (offset 0..k-1, entry bar included). Computed separately for wins and losses.

**WINS** (n=2249):

| dip_>=R | within_1bar | within_2bar | within_3bar |
|---|---|---|---|
| 0.1000 | 0.7786 | 0.8519 | 0.8808 |
| 0.2500 | 0.4024 | 0.5465 | 0.6301 |
| 0.5000 | 0.1232 | 0.2334 | 0.3148 |

**LOSSES** (n=903):

| dip_>=R | within_1bar | within_2bar | within_3bar |
|---|---|---|---|
| 0.1000 | 0.8228 | 0.8915 | 0.9225 |
| 0.2500 | 0.4629 | 0.6368 | 0.7287 |
| 0.5000 | 0.1584 | 0.3123 | 0.4297 |

## A4 — gap-at-open (entry open vs prior signal-bar close, 2.0R frame)

Resolved gap for 3152/3152 trades. Fraction with entry open ABOVE prior signal-bar close (gap up): **0.9734**.

| cut | n | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gap_open_r pooled | 3152 | -0.0291 | 0.0021 | 0.0039 | 0.0081 | 0.0151 | 0.0263 | 0.0476 | 0.0611 | 0.1246 | 0.0199 |
| gap_open_r win | 2249 | -0.0158 | 0.0021 | 0.0038 | 0.0080 | 0.0151 | 0.0271 | 0.0487 | 0.0617 | 0.1284 | 0.0201 |
| gap_open_r loss | 903 | -0.0554 | 0.0020 | 0.0040 | 0.0083 | 0.0150 | 0.0254 | 0.0433 | 0.0590 | 0.1032 | 0.0192 |


---

# Task B — TP1 / TS placement (MFE + give-back)

## B1 — terminal_mfe_r distribution (2.0R frame)

`_full` = max over full 240-bar window (== pool `mfe_r`; matches the 5.31R cohort anchor). `_dep` = over deployed-held window.

| cut | n | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pooled (full 240-bar) | 3152 | 0.0632 | 0.3168 | 0.6073 | 1.5603 | 3.3699 | 5.8732 | 8.7702 | 10.9753 | 16.9055 | 4.2547 |
| pooled (deployed-held) | 3152 | 0.0332 | 0.1659 | 0.3899 | 1.2756 | 2.4384 | 3.8411 | 5.7140 | 7.0524 | 10.5839 | 2.8438 |
| fold (full 240-bar) | 2059 | 0.0590 | 0.3364 | 0.6226 | 1.5757 | 3.3703 | 5.9760 | 8.9731 | 11.1778 | 18.4365 | 4.3715 |
| holdout (full 240-bar) | 1093 | 0.0734 | 0.3124 | 0.5878 | 1.4881 | 3.3629 | 5.7233 | 8.1864 | 10.3854 | 15.5243 | 4.0348 |
| c0 v_shape (full 240-bar) | 1493 | 0.5335 | 1.3163 | 1.9420 | 3.4102 | 5.3090 | 7.8953 | 10.5941 | 12.7148 | 19.0449 | 6.0536 |
| c1 monotonic (full 240-bar) | 1658 | 0.0365 | 0.1649 | 0.3635 | 0.8592 | 1.9642 | 3.5120 | 5.4131 | 6.7563 | 11.4799 | 2.6338 |

## B2 — conditional continuation past +1R

Of trades that reached threshold T, the fraction that went on to reach higher thresholds. Reported in BOTH frames: **3.5R-deployed** (the frame the +1R partial actually keys off — tests prematurity directly) and **2.0R-recorded** (the cohort-anchor frame).

**3.5R deployed frame:**

| reached_>= | n | ->2R | ->3R | ->5R |
|---|---|---|---|---|
| 1R | 2256 | 0.6702 | 0.4145 | 0.1414 |
| 2R | 1512 | nan | 0.6184 | 0.2110 |
| 3R | 935 | nan | nan | 0.3412 |

**2.0R recorded frame:**

| reached_>= | n | ->2R | ->3R | ->5R |
|---|---|---|---|---|
| 1R | 2633 | 0.8150 | 0.6555 | 0.3828 |
| 2R | 2146 | nan | 0.8043 | 0.4697 |
| 3R | 1726 | nan | nan | 0.5840 |

## B3 — trail give-back (3.5R frame)

`give_back_r = runner_peak_mfe_r - runner_exit_r` for trades whose runner leg exits via the trail (reason=`partial_then_trail`). 3.5R frame.

| cut | n | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| give_back_r (trail exits) | 2161 | 1.0015 | 1.0071 | 1.0145 | 1.0403 | 1.0991 | 1.2010 | 1.3563 | 1.4824 | 1.9008 | 1.1595 |

## B4 — exit_reason mix (deployed policy, 3.5R frame)

| exit_reason_deployed | n | share | mean_realized_r | median_realized_r |
|---|---|---|---|---|
| SL_no_partial | 880 | 0.2792 | -1.0000 | -1.0000 |
| partial_then_runner_SL | 5 | 0.0016 | 0.0000 | 0.0000 |
| partial_then_time_exit | 90 | 0.0286 | 0.4841 | 0.2503 |
| partial_then_trail | 2161 | 0.6856 | 1.0000 | 0.8349 |
| time_exit_no_partial | 16 | 0.0051 | -0.3949 | -0.5409 |

## B5 — realized-R decomposition: partial leg vs runner leg (3.5R frame)

| cut | n | mean_partial_contrib | mean_runner_contrib | mean_total |
|---|---|---|---|---|
| all trades | 3152 | 0.3579 | 0.0603 | 0.4182 |
| partial fired | 2256 | 0.5000 | 0.4772 | 0.9772 |
| no partial | 896 | 0.0000 | -0.9892 | -0.9892 |


---

### Flags
- `terminal_mae/mfe_held` (literal `is_held`, 2.0 sl_only window) truncates at the 2.0xATR SL; deployed-losers floor near -1R (2.0 frame), distorting a win-vs-loss comparison. The **deployed-held** cut is the meaningful one and is used for the load-bearing read.
- give-back is only defined for `partial_then_trail` exits; other reasons have no trail leg to give back.
- this is descriptive only; no entry/exit change is proposed or tested here (per dispatch role boundary).


---

## Band pullbacks (3.5R)

> Descriptive only — no trail re-optimisation or WFO. Per-band deepest **close-based** retracement from running peak while traversing band k=[kR,(k+1)R], 3.5R deployed frame, over the deployed-policy held window. `intra_band_pullback_r` is exactly what a peak-anchored, bar-close-updated trail of width W keys off. Completer p90 ~ minimum safe trail width for that band; reverser p50 ~ how early a tighter trail would catch faders. Where completer-p90 < current 1.0R the trail is wider than needed in that band.

### Headline

| band | completion_rate | completer_pullback_p90 | reverser_pullback_p50 |
|---|---|---|---|
| 1.0000 | 0.4245 | 0.8817 | 1.0936 |
| 2.0000 | 0.4224 | 0.8958 | 1.1053 |
| 3.0000 | 0.4107 | 0.9163 | 1.1119 |
| 4.0000 | 0.4224 | 0.9606 | 1.0881 |
| 5.0000 | 0.4853 | 0.8938 | 1.1531 |

### Band attrition

| band | n_entered | n_completed | completion_rate |
|---|---|---|---|
| 1.0000 | 2186.0000 | 928.0000 | 0.4245 |
| 2.0000 | 928.0000 | 392.0000 | 0.4224 |
| 3.0000 | 392.0000 | 161.0000 | 0.4107 |
| 4.0000 | 161.0000 | 68.0000 | 0.4224 |
| 5.0000 | 68.0000 | 33.0000 | 0.4853 |

> **THIN-BAND FLAG (n_entered<100):** band5(n=68) — p90/p95/max unreliable.

> band 4 completer cohort thin: n=68.

> band 4 reverser cohort thin: n=93.

> band 5 completer cohort thin: n=33.

> band 5 reverser cohort thin: n=35.

### Band 1 = [1R, 2R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 928 | 0.3136 | 0.5053 | 0.7144 | 0.8817 | 0.9499 | 4.1647 | 0.5338 |
| reverser | 1258 | 1.0378 | 1.0936 | 1.1909 | 1.3120 | 1.4467 | 3.1636 | 1.1438 |

bars_in_band:

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 928 | 9.0 | 18.5 | 31.0 | 49.0 | 62.6 | 134.0 | 23.3 |
| reverser | 1258 | 11.0 | 19.0 | 31.0 | 49.0 | 61.1 | 210.0 | 24.0 |

### Band 2 = [2R, 3R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 392 | 0.3464 | 0.5273 | 0.7280 | 0.8958 | 0.9730 | 4.1647 | 0.5726 |
| reverser | 536 | 1.0403 | 1.1053 | 1.2135 | 1.3577 | 1.4800 | 3.5766 | 1.1610 |

bars_in_band:

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 392 | 8.0 | 17.0 | 29.0 | 46.0 | 56.0 | 120.0 | 21.7 |
| reverser | 536 | 10.0 | 18.0 | 28.0 | 43.0 | 58.0 | 99.0 | 21.7 |

### Band 3 = [3R, 4R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 161 | 0.2955 | 0.4761 | 0.6866 | 0.9163 | 0.9998 | 4.1647 | 0.5568 |
| reverser | 231 | 1.0412 | 1.1119 | 1.2153 | 1.4257 | 1.5865 | 2.6207 | 1.1701 |

bars_in_band:

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 161 | 6.0 | 11.0 | 18.0 | 37.0 | 44.0 | 89.0 | 15.6 |
| reverser | 231 | 9.5 | 16.0 | 26.5 | 40.0 | 62.0 | 149.0 | 21.2 |

### Band 4 = [4R, 5R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 68 | 0.3575 | 0.5640 | 0.7577 | 0.9606 | 1.3295 | 4.1647 | 0.6470 |
| reverser | 93 | 1.0293 | 1.0881 | 1.1931 | 1.3406 | 1.5052 | 1.7499 | 1.1363 |

bars_in_band:

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 68 | 4.0 | 11.0 | 21.0 | 32.5 | 42.2 | 83.0 | 14.8 |
| reverser | 93 | 8.0 | 13.0 | 17.0 | 26.6 | 43.6 | 81.0 | 15.6 |

### Band 5 = [5R, 6R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 33 | 0.4229 | 0.6331 | 0.8016 | 0.8938 | 1.2844 | 2.0204 | 0.6587 |
| reverser | 35 | 1.0724 | 1.1531 | 1.4054 | 1.9047 | 2.2325 | 5.4987 | 1.4044 |

bars_in_band:

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 33 | 4.0 | 9.0 | 20.0 | 27.8 | 30.0 | 36.0 | 12.5 |
| reverser | 35 | 5.0 | 11.0 | 16.0 | 23.0 | 27.5 | 47.0 | 12.1 |


---

## Runner EV by band (3.5R)

> Descriptive only — no second-partial policy added, no WFO. Runner leg (partial_fired, n=2256). `runner_final_r` = per-unit runner realized R at deployed exit (3.5R frame). "reached kR" = runner LIVE peak `runner_peak_mfe_r` >= k (running-max => first live cross of kR). `diff = E[runner_final | reached kR] - kR`: **diff<0 => banking at kR beats holding by |diff| per runner unit**; the most-negative band is the best single second-partial level. `tail_share` = fraction of reached-kR runners whose final R >= kR+2 (the upside a bank-at-kR partial gives up).

| band_kR | n_reached | E_runner_final_r | kR | diff_E_minus_kR | p25 | p50 | p75 | p90 | mean | tail_share_ge_kR_plus_2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1R | 2179 | 1.0076 | 1.0000 | 0.0076 | 0.2128 | 0.6712 | 1.4344 | 2.5132 | 1.0076 | 0.0638 |
| 2R | 926 | 1.9715 | 2.0000 | -0.0285 | 1.1805 | 1.6214 | 2.4500 | 3.4112 | 1.9715 | 0.0594 |
| 3R | 392 | 2.9564 | 3.0000 | -0.0436 | 2.1789 | 2.7003 | 3.3632 | 4.3250 | 2.9564 | 0.0714 |
| 4R | 161 | 3.9373 | 4.0000 | -0.0627 | 3.1173 | 3.5399 | 4.3186 | 5.4905 | 3.9373 | 0.0683 |
| 5R | 68 | 4.8320 | 5.0000 | -0.1680 | 4.0338 | 4.5626 | 5.4400 | 6.6076 | 4.8320 | 0.0882 |

> **THIN-BAND FLAG (k>=4, n_reached<100):** 5R(n=68) — E / percentiles / tail_share unreliable.

### Decision summary

> k* taken over reliable (n_reached>=100) bands; raw most-negative diff is 5R (|diff|=0.1680, n=68) but THIN — not actionable.

| metric | value | note |
|---|---|---|
| best second-partial level k* (reliable) | 4R | most-negative diff among non-thin bands; n_reached=161 |
| per-unit EV gain |diff| at k* | 0.0627 | R per runner unit banked vs held |
| tail_share (final>=k*R+2) at k* | 0.0683 | upside fraction the partial gives up |

> Read: among reliable bands, banking the runner at **4R** gives the largest per-unit EV gain (**0.0627R**) over holding, surrendering the **0.0683** tail of reached-4R runners finishing >= k*R+2. All reliable-band gains are small (|diff| <= 0.07R) and the per-unit edge grows monotonically with k while n thins, so the runner trail is effectively EV-neutral at every reliable band — no band shows a material banking edge. Candidate (laddered second partial) gated on this table and deferred post-live per protocol; v3.0.2 locked.


---

## Uncensored forward-path pullbacks (3.5R)

> Descriptive only — full 240-bar recorded forward path, NO exit policy applied. 3.5R frame. Metric identical to the deployed-held `band_pullbacks` (peak-anchored `mfe_3p5 - close_3p5`), censor removed, so RAW-vs-censored is apples-to-apples. **RAW completion past a band is price-GEOMETRY POTENTIAL, NOT realizable** — the SL-FLOOR pass (hard -1R SL only, no trail/partial/time) is the realizable counterpart. No exit change, no WFO; v3.0.2 locked.

### Headline — natural vs censored

| band | completion_RAW | completion_SL_floor | completion_deployed_held_prior | completer_p90_RAW | completer_p90_SL_floor | completer_p90_deployed_held_prior |
|---|---|---|---|---|---|---|
| 1 | 0.6702 | 0.6525 | 0.4245 | 1.9539 | 1.7234 | 0.8817 |
| 2 | 0.6184 | 0.6339 | 0.4224 | 1.8714 | 1.8370 | 0.8958 |
| 3 | 0.5872 | 0.6056 | 0.4107 | 1.9351 | 1.9480 | 0.9163 |
| 4 | 0.5811 | 0.5744 | 0.4224 | 2.0170 | 2.0238 | 0.9606 |
| 5 | 0.5611 | 0.5625 | 0.4853 | 2.0585 | 1.9806 | 0.8938 |
| 6 | 0.6201 | 0.6429 | nan | 2.1452 | 2.1452 | nan |

> Load-bearing read: if completer_p90 (RAW) **tapers** with k a laddered trail revives; if it stays ~flat the taper thesis stays dead — but at the true uncensored width, not the trail-bounded ~0.9R of the deployed-held cut. Deployed-held completion (~0.42/band) is trail-censored; RAW band-1 completion is the natural 1R→2R rate (anchors to B2 full-240 ~0.67).

### RAW pass

#### Band attrition

| band | n_entered | n_completed | completion_rate |
|---|---|---|---|
| 1 | 2256 | 1512 | 0.6702 |
| 2 | 1512 | 935 | 0.6184 |
| 3 | 935 | 549 | 0.5872 |
| 4 | 549 | 319 | 0.5811 |
| 5 | 319 | 179 | 0.5611 |
| 6 | 179 | 111 | 0.6201 |

#### Band 1 = [1R,2R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 1512 | 0.4227 | 0.7586 | 1.3314 | 1.9539 | 2.4508 | 8.4367 | 0.9777 |
| reverser | 744 | 2.1320 | 3.0956 | 4.4483 | 6.1114 | 7.3551 | 17.9903 | 3.5078 |

#### Band 2 = [2R,3R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 935 | 0.4219 | 0.7260 | 1.2481 | 1.8714 | 2.3773 | 8.9407 | 0.9609 |
| reverser | 577 | 1.5119 | 2.4762 | 3.7918 | 5.3529 | 6.6942 | 17.8849 | 2.9136 |

#### Band 3 = [3R,4R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 549 | 0.4129 | 0.6866 | 1.2412 | 1.9351 | 2.3984 | 9.3563 | 0.9644 |
| reverser | 386 | 1.3224 | 2.2016 | 3.6152 | 5.1756 | 6.0617 | 98.1878 | 2.9033 |

#### Band 4 = [4R,5R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 319 | 0.4122 | 0.7196 | 1.2782 | 2.0170 | 2.3461 | 8.9407 | 0.9720 |
| reverser | 230 | 1.0919 | 2.0678 | 3.1577 | 4.4359 | 5.2239 | 15.1934 | 2.4044 |

#### Band 5 = [5R,6R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 179 | 0.4687 | 0.7481 | 1.1546 | 2.0585 | 2.5139 | 8.9407 | 1.0329 |
| reverser | 140 | 1.1704 | 2.1845 | 3.2305 | 4.4156 | 5.8984 | 9.5687 | 2.4325 |

#### Band 6 = [6R,7R]  _(thin: reverser)_

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 111 | 0.5082 | 0.7564 | 1.2218 | 2.1452 | 2.6921 | 8.9407 | 1.1087 |
| reverser | 68 | 1.3303 | 2.6798 | 3.8274 | 5.4458 | 8.0136 | 28.0172 | 3.4264 |

### SL_FLOOR pass

#### Band attrition

| band | n_entered | n_completed | completion_rate |
|---|---|---|---|
| 1 | 1557 | 1016 | 0.6525 |
| 2 | 1016 | 644 | 0.6339 |
| 3 | 644 | 390 | 0.6056 |
| 4 | 390 | 224 | 0.5744 |
| 5 | 224 | 126 | 0.5625 |
| 6 | 126 | 81 | 0.6429 |

#### Band 1 = [1R,2R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 1016 | 0.4045 | 0.7094 | 1.1973 | 1.7234 | 1.9275 | 4.1647 | 0.8521 |
| reverser | 541 | 2.0997 | 2.3289 | 2.6305 | 2.8529 | 2.9353 | 3.8558 | 2.3126 |

#### Band 2 = [2R,3R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 644 | 0.4060 | 0.7281 | 1.2473 | 1.8370 | 2.1988 | 8.9407 | 0.9273 |
| reverser | 372 | 1.9587 | 2.9894 | 3.4257 | 3.7421 | 3.9352 | 13.7549 | 2.6976 |

#### Band 3 = [3R,4R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 390 | 0.4028 | 0.6949 | 1.2338 | 1.9480 | 2.3977 | 8.9407 | 0.9435 |
| reverser | 254 | 1.6517 | 2.5776 | 3.9413 | 4.4618 | 4.6745 | 19.3948 | 2.7549 |

#### Band 4 = [4R,5R]

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 224 | 0.3981 | 0.7406 | 1.2867 | 2.0238 | 2.6066 | 8.9407 | 1.0063 |
| reverser | 166 | 1.1694 | 2.1115 | 3.4850 | 4.5496 | 5.1284 | 11.1151 | 2.4075 |

#### Band 5 = [5R,6R]  _(thin: reverser)_

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 126 | 0.4667 | 0.7233 | 1.1403 | 1.9806 | 2.3857 | 8.9407 | 1.0012 |
| reverser | 98 | 1.2678 | 2.1845 | 3.2412 | 4.7773 | 5.9133 | 6.8635 | 2.4412 |

#### Band 6 = [6R,7R]  _(thin: completer, reverser)_

| cohort | n | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|
| completer | 81 | 0.5476 | 0.7539 | 1.2128 | 2.1452 | 3.0127 | 8.9407 | 1.1553 |
| reverser | 45 | 1.8976 | 2.6644 | 3.6661 | 5.0287 | 5.5131 | 6.2283 | 2.8759 |


---

## Trade distribution analytics (3.5R)

> Descriptive, read-only re-aggregation of the deployed-policy frame (realized R = `sl_partial_close_1r_runner_trail @ 3.5xATR`, both legs, 3.5R frame; win = R>0). Calendar is EET-local (Europe/Athens, the 5ers EET convention); EET-year matches the fold-tagged year for all 3152 trades, day-of-week shifts for 452 late-Friday-UTC bars onto the next EET trading day. **NOT a pruning signal:** per-pair realized R is descriptive only — selecting pairs on realized R is outcome-selection against the ex-ante population rule; any pair-set change is a separate ex-ante-justified + WFO question, never a backtest filter. No exit change, no WFO; v3.0.2 locked.

### Headline

- **Best pairs (total R):** GBPCAD (66.8R, n=125), EURGBP (65.0R, n=118), GBPNZD (64.6R, n=104), CHFJPY (61.9R, n=117), EURJPY (60.8R, n=109).
- **Worst pairs (total R):** NZDCHF (24.3R, n=123), GBPCHF (26.4R, n=109), NZDUSD (27.5R, n=91), AUDNZD (34.2R, n=107), AUDCAD (34.3R, n=131).
- **Top-5 pairs = 24.2% of all realized R** (of 1318.2R total); 0 pair(s) net-negative.
- **Busiest month:** 4 (n=296, 1.13x uniform); **sparsest:** 12 (n=221, 0.84x).
- **Frequency verdict:** monthly-count CV=0.31, month-of-year chi-square p=0.0968 -> **~random-uniform**.

### 1. Per pair (sorted by total R)

> Flag: pairs with n<30 are directional-only. `rank_gap = rank_total_r - rank_mean_r`: large negative = high-volume-low-edge; large positive = low-volume-high-edge.

| pair | n | n_win | win_rate | mean_r | total_r | mean_win_r | mean_loss_r | profit_factor | share_of_trades | share_of_total_r | rank_total_r | rank_mean_r | rank_gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GBPCAD | 125 | 95 | 0.7600 | 0.5346 | 66.8294 | 1.0193 | -1.0000 | 3.2276 | 0.0397 | 0.0507 | 1 | 5 | -4 |
| EURGBP | 118 | 89 | 0.7542 | 0.5505 | 64.9543 | 1.0505 | -0.9841 | 3.2761 | 0.0374 | 0.0493 | 2 | 3 | -1 |
| GBPNZD | 104 | 81 | 0.7788 | 0.6212 | 64.6017 | 1.0752 | -0.9779 | 3.8724 | 0.0330 | 0.0490 | 3 | 1 | 2 |
| CHFJPY | 117 | 87 | 0.7436 | 0.5289 | 61.8840 | 1.0561 | -1.0000 | 3.0628 | 0.0371 | 0.0469 | 4 | 6 | -2 |
| EURJPY | 109 | 82 | 0.7523 | 0.5576 | 60.7759 | 1.0704 | -1.0000 | 3.2510 | 0.0346 | 0.0461 | 5 | 2 | 3 |
| AUDUSD | 112 | 90 | 0.8036 | 0.5234 | 58.6247 | 0.8958 | -1.0000 | 3.6648 | 0.0355 | 0.0445 | 6 | 8 | -2 |
| NZDCAD | 133 | 93 | 0.6992 | 0.4322 | 57.4799 | 1.0374 | -0.9750 | 2.4738 | 0.0422 | 0.0436 | 7 | 12 | -5 |
| USDCAD | 104 | 81 | 0.7788 | 0.5373 | 55.8758 | 0.9738 | -1.0000 | 3.4294 | 0.0330 | 0.0424 | 8 | 4 | 4 |
| USDJPY | 106 | 80 | 0.7547 | 0.5246 | 55.6121 | 1.0077 | -0.9615 | 3.2245 | 0.0336 | 0.0422 | 9 | 7 | 2 |
| EURCAD | 111 | 85 | 0.7658 | 0.4787 | 53.1336 | 0.9259 | -0.9835 | 3.0779 | 0.0352 | 0.0403 | 10 | 10 | 0 |
| EURCHF | 124 | 85 | 0.6855 | 0.4103 | 50.8816 | 1.0522 | -0.9885 | 2.3198 | 0.0393 | 0.0386 | 11 | 15 | -4 |
| EURAUD | 118 | 82 | 0.6949 | 0.4146 | 48.9179 | 1.0198 | -0.9640 | 2.4096 | 0.0374 | 0.0371 | 12 | 14 | -2 |
| GBPJPY | 100 | 75 | 0.7500 | 0.4871 | 48.7124 | 0.9828 | -1.0000 | 2.9485 | 0.0317 | 0.0370 | 13 | 9 | 4 |
| GBPAUD | 107 | 73 | 0.6822 | 0.4297 | 45.9769 | 1.0898 | -0.9877 | 2.3692 | 0.0339 | 0.0349 | 14 | 13 | 1 |
| CADJPY | 110 | 78 | 0.7091 | 0.4078 | 44.8546 | 0.9853 | -1.0000 | 2.4017 | 0.0349 | 0.0340 | 15 | 16 | -1 |
| AUDCHF | 131 | 94 | 0.7176 | 0.3411 | 44.6839 | 0.8583 | -0.9730 | 2.2412 | 0.0416 | 0.0339 | 16 | 21 | -5 |
| CADCHF | 133 | 92 | 0.6917 | 0.3330 | 44.2891 | 0.9196 | -0.9832 | 2.0986 | 0.0422 | 0.0336 | 17 | 23 | -6 |
| EURNZD | 113 | 80 | 0.7080 | 0.3870 | 43.7315 | 0.9591 | -1.0000 | 2.3252 | 0.0359 | 0.0332 | 18 | 19 | -1 |
| USDCHF | 109 | 76 | 0.6972 | 0.3874 | 42.2239 | 0.9799 | -0.9772 | 2.3094 | 0.0346 | 0.0320 | 19 | 18 | 1 |
| EURUSD | 93 | 68 | 0.7312 | 0.4426 | 41.1595 | 0.9505 | -0.9389 | 2.7536 | 0.0295 | 0.0312 | 20 | 11 | 9 |
| NZDJPY | 112 | 77 | 0.6875 | 0.3671 | 41.1106 | 0.9884 | -1.0000 | 2.1746 | 0.0355 | 0.0312 | 21 | 20 | 1 |
| AUDJPY | 101 | 70 | 0.6931 | 0.4036 | 40.7653 | 1.0060 | -0.9567 | 2.3746 | 0.0320 | 0.0309 | 22 | 17 | 5 |
| GBPUSD | 101 | 67 | 0.6634 | 0.3403 | 34.3706 | 0.9915 | -0.9430 | 2.0720 | 0.0320 | 0.0261 | 23 | 22 | 1 |
| AUDCAD | 131 | 88 | 0.6718 | 0.2615 | 34.2536 | 0.8732 | -0.9904 | 1.8043 | 0.0416 | 0.0260 | 24 | 26 | -2 |
| AUDNZD | 107 | 74 | 0.6916 | 0.3196 | 34.2002 | 0.9021 | -0.9864 | 2.0506 | 0.0339 | 0.0259 | 25 | 24 | 1 |
| NZDUSD | 91 | 61 | 0.6703 | 0.3027 | 27.5438 | 0.9433 | -1.0000 | 1.9181 | 0.0289 | 0.0209 | 26 | 25 | 1 |
| GBPCHF | 109 | 70 | 0.6422 | 0.2424 | 26.4233 | 0.9213 | -0.9760 | 1.6942 | 0.0346 | 0.0200 | 27 | 27 | 0 |
| NZDCHF | 123 | 76 | 0.6179 | 0.1976 | 24.3015 | 0.9269 | -0.9817 | 1.5267 | 0.0390 | 0.0184 | 28 | 28 | 0 |

> **Rank disagreement (|gap|>=7):** EURUSD(total#20/mean#11)

### 2. Per year

| year | n | win_rate | mean_r | total_r |
|---|---|---|---|---|
| 2010 | 201 | 0.6816 | 0.3087 | 62.0582 |
| 2011 | 182 | 0.7198 | 0.4040 | 73.5277 |
| 2012 | 179 | 0.7318 | 0.4688 | 83.9126 |
| 2013 | 195 | 0.7179 | 0.5529 | 107.8094 |
| 2014 | 192 | 0.7396 | 0.4301 | 82.5759 |
| 2015 | 190 | 0.6526 | 0.2947 | 55.9912 |
| 2016 | 171 | 0.7251 | 0.4413 | 75.4689 |
| 2017 | 190 | 0.7579 | 0.5878 | 111.6904 |
| 2018 | 176 | 0.6534 | 0.2320 | 40.8291 |
| 2019 | 195 | 0.7333 | 0.4324 | 84.3113 |
| 2020 | 188 | 0.7553 | 0.4822 | 90.6514 |
| 2021 | 212 | 0.7123 | 0.4328 | 91.7610 |
| 2022 | 179 | 0.6760 | 0.3426 | 61.3169 |
| 2023 | 189 | 0.6720 | 0.3647 | 68.9271 |
| 2024 | 191 | 0.7592 | 0.5242 | 100.1169 |
| 2025 | 251 | 0.7610 | 0.4639 | 116.4298 |
| 2026 | 71 | 0.5775 | 0.1520 | 10.7937 |

Fold vs holdout:

| segment | n | win_rate | mean_r | total_r |
|---|---|---|---|---|
| fold | 2059 | 0.7154 | 0.4220 | 868.8261 |
| holdout | 1093 | 0.7100 | 0.4111 | 449.3454 |

### 3. Per calendar month (pooled across years)

| month | n | win_rate | mean_r | total_r | expected_n_if_uniform | n_over_expected |
|---|---|---|---|---|---|---|
| 1 | 238 | 0.6891 | 0.3642 | 86.6716 | 262.6667 | 0.9061 |
| 2 | 261 | 0.6782 | 0.3827 | 99.8919 | 262.6667 | 0.9937 |
| 3 | 263 | 0.7148 | 0.3809 | 100.1673 | 262.6667 | 1.0013 |
| 4 | 296 | 0.7297 | 0.4766 | 141.0765 | 262.6667 | 1.1269 |
| 5 | 254 | 0.6772 | 0.3709 | 94.2112 | 262.6667 | 0.9670 |
| 6 | 273 | 0.6850 | 0.3836 | 104.7198 | 262.6667 | 1.0393 |
| 7 | 271 | 0.6679 | 0.2885 | 78.1727 | 262.6667 | 1.0317 |
| 8 | 276 | 0.7464 | 0.4884 | 134.7902 | 262.6667 | 1.0508 |
| 9 | 262 | 0.7710 | 0.5168 | 135.3941 | 262.6667 | 0.9975 |
| 10 | 286 | 0.7552 | 0.4789 | 136.9529 | 262.6667 | 1.0888 |
| 11 | 251 | 0.7092 | 0.4426 | 111.0818 | 262.6667 | 0.9556 |
| 12 | 221 | 0.7330 | 0.4301 | 95.0415 | 262.6667 | 0.8414 |

### 4. Frequency / clustering

| metric | value |
|---|---|
| active months (span, incl. zeros) | 195 |
| monthly count mean | 16.1641 |
| monthly count median | 15.0000 |
| monthly count CV (std/mean) | 0.3136 |
| max-month / median-month | 1.8667 |
| month-of-year chi-square stat (dof=11) | 17.3934 |
| month-of-year chi-square p-value | 0.09677 |

> Verdict: **~random-uniform** (CV=0.31; month-of-year chi-square p=0.0968). Monthly arrival is close to uniform.

### 5. Pair x year — DIRECTIONAL ONLY (most cells n<30)

Trade count matrix:

| pair | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUDCAD | 10 | 9 | 6 | 7 | 4 | 8 | 8 | 12 | 10 | 10 | 3 | 7 | 5 | 9 | 7 | 13 | 3 |
| AUDCHF | 6 | 9 | 11 | 6 | 7 | 5 | 8 | 9 | 8 | 9 | 7 | 10 | 11 | 4 | 9 | 10 | 2 |
| AUDJPY | 7 | 8 | 5 | 9 | 5 | 8 | 5 | 5 | 6 | 7 | 5 | 7 | 6 | 6 | 4 | 8 | 0 |
| AUDNZD | 5 | 6 | 5 | 7 | 3 | 5 | 8 | 7 | 10 | 7 | 8 | 4 | 7 | 12 | 7 | 4 | 2 |
| AUDUSD | 9 | 9 | 6 | 3 | 8 | 9 | 4 | 7 | 4 | 12 | 4 | 8 | 7 | 7 | 4 | 8 | 3 |
| CADCHF | 10 | 8 | 12 | 7 | 8 | 6 | 7 | 9 | 6 | 8 | 9 | 7 | 7 | 7 | 6 | 12 | 4 |
| CADJPY | 10 | 6 | 7 | 10 | 6 | 8 | 3 | 6 | 7 | 6 | 4 | 7 | 8 | 2 | 7 | 12 | 1 |
| CHFJPY | 9 | 7 | 3 | 11 | 5 | 9 | 8 | 8 | 7 | 7 | 9 | 6 | 7 | 5 | 7 | 6 | 3 |
| EURAUD | 4 | 6 | 4 | 3 | 10 | 11 | 12 | 4 | 9 | 6 | 5 | 8 | 5 | 11 | 7 | 12 | 1 |
| EURCAD | 6 | 4 | 10 | 7 | 6 | 5 | 4 | 5 | 5 | 9 | 7 | 9 | 7 | 4 | 8 | 13 | 2 |
| EURCHF | 6 | 2 | 12 | 10 | 9 | 7 | 4 | 7 | 5 | 7 | 12 | 11 | 4 | 8 | 7 | 11 | 2 |
| EURGBP | 8 | 7 | 9 | 7 | 5 | 6 | 0 | 6 | 7 | 3 | 13 | 7 | 4 | 9 | 11 | 13 | 3 |
| EURJPY | 8 | 5 | 3 | 9 | 5 | 9 | 4 | 9 | 3 | 8 | 5 | 4 | 6 | 8 | 8 | 10 | 5 |
| EURNZD | 6 | 5 | 3 | 6 | 10 | 6 | 9 | 8 | 7 | 4 | 7 | 11 | 6 | 7 | 7 | 7 | 4 |
| EURUSD | 5 | 4 | 3 | 5 | 8 | 7 | 5 | 6 | 6 | 8 | 5 | 5 | 6 | 5 | 9 | 4 | 2 |
| GBPAUD | 7 | 7 | 2 | 6 | 8 | 5 | 8 | 6 | 7 | 10 | 6 | 9 | 7 | 4 | 9 | 5 | 1 |
| GBPCAD | 7 | 7 | 7 | 8 | 7 | 9 | 7 | 9 | 5 | 5 | 8 | 10 | 7 | 7 | 7 | 11 | 4 |
| GBPCHF | 5 | 5 | 4 | 7 | 11 | 5 | 4 | 8 | 6 | 5 | 7 | 10 | 7 | 7 | 5 | 12 | 1 |
| GBPJPY | 8 | 6 | 7 | 9 | 6 | 4 | 3 | 6 | 5 | 8 | 4 | 8 | 6 | 9 | 4 | 4 | 3 |
| GBPNZD | 7 | 3 | 6 | 4 | 4 | 4 | 6 | 7 | 5 | 11 | 5 | 8 | 11 | 7 | 5 | 10 | 1 |
| GBPUSD | 4 | 8 | 5 | 6 | 9 | 4 | 7 | 6 | 5 | 7 | 6 | 4 | 3 | 7 | 6 | 9 | 5 |
| NZDCAD | 9 | 9 | 5 | 11 | 9 | 10 | 11 | 7 | 5 | 6 | 8 | 10 | 4 | 7 | 6 | 13 | 3 |
| NZDCHF | 10 | 9 | 8 | 7 | 8 | 4 | 6 | 6 | 4 | 5 | 7 | 7 | 6 | 9 | 10 | 10 | 7 |
| NZDJPY | 10 | 7 | 6 | 7 | 3 | 9 | 7 | 4 | 6 | 4 | 9 | 7 | 6 | 11 | 6 | 7 | 3 |
| NZDUSD | 6 | 5 | 7 | 5 | 8 | 6 | 6 | 4 | 6 | 5 | 5 | 8 | 4 | 4 | 4 | 7 | 1 |
| USDCAD | 6 | 6 | 6 | 5 | 9 | 4 | 6 | 6 | 9 | 8 | 4 | 6 | 5 | 5 | 12 | 5 | 2 |
| USDCHF | 3 | 4 | 12 | 5 | 5 | 5 | 7 | 8 | 5 | 5 | 8 | 8 | 11 | 5 | 6 | 10 | 2 |
| USDJPY | 10 | 11 | 5 | 8 | 6 | 12 | 4 | 5 | 8 | 5 | 8 | 6 | 6 | 3 | 3 | 5 | 1 |

Total R matrix:

| pair | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUDCAD | 6.7 | 4.0 | 0.5 | 0.7 | 0.3 | 5.3 | 2.5 | -2.2 | 1.1 | 3.2 | 1.7 | 2.4 | 1.6 | 1.1 | 0.3 | 7.6 | -2.6 |
| AUDCHF | 4.3 | 4.4 | -0.4 | 0.1 | 7.0 | 2.5 | 6.1 | -1.4 | 1.8 | 8.0 | 0.5 | 1.2 | -0.1 | -2.0 | 6.7 | 4.7 | 1.2 |
| AUDJPY | 2.6 | 4.4 | 2.6 | 5.4 | 0.1 | 1.0 | 1.4 | 2.9 | 5.3 | 2.4 | -0.1 | 1.0 | 3.1 | 3.6 | 3.6 | 1.5 | 0.0 |
| AUDNZD | -0.6 | -4.3 | 0.4 | -0.2 | 1.1 | 1.5 | 3.1 | 5.1 | 6.3 | 4.7 | 1.6 | -0.9 | 5.5 | -0.8 | 4.3 | 4.9 | 2.5 |
| AUDUSD | 8.1 | 4.7 | 2.3 | 1.2 | 4.3 | -0.1 | 3.8 | 5.7 | -2.1 | 8.7 | 1.0 | 3.9 | 4.0 | 4.5 | 1.1 | 7.3 | 0.4 |
| CADCHF | 1.4 | 5.6 | 5.9 | -0.6 | 9.8 | 1.3 | 7.9 | 4.2 | -1.1 | -4.8 | 3.9 | 1.1 | 0.8 | 7.1 | 4.0 | -1.0 | -1.1 |
| CADJPY | 1.1 | 3.4 | 2.9 | 4.0 | 5.7 | -2.8 | 3.1 | 1.3 | -1.6 | -1.9 | 3.2 | 6.7 | 5.9 | 3.0 | 7.2 | 3.0 | 0.6 |
| CHFJPY | 2.6 | 1.5 | 2.6 | 10.7 | 2.1 | -0.5 | 0.5 | 2.2 | 5.8 | 3.8 | 6.0 | 8.3 | 5.3 | 5.4 | 1.6 | 3.7 | 0.2 |
| EURAUD | -0.6 | 0.1 | 3.4 | 0.8 | 1.4 | 5.2 | 7.8 | 1.6 | 1.9 | 1.0 | 4.9 | 6.3 | 0.3 | 4.2 | 6.1 | 5.5 | -1.0 |
| EURCAD | 7.6 | -0.3 | 2.7 | 5.9 | -0.2 | 2.4 | 3.6 | 6.0 | 3.7 | 0.6 | 2.2 | -1.6 | 1.1 | 3.0 | 6.9 | 10.1 | -0.6 |
| EURCHF | 0.8 | 1.9 | 17.8 | 2.2 | 3.6 | 2.2 | -0.1 | 9.3 | -5.0 | 3.7 | 13.1 | -2.5 | -0.4 | -0.0 | 0.1 | 5.0 | -0.7 |
| EURGBP | -0.9 | 3.2 | 2.8 | 10.3 | 2.3 | 5.6 | 0.0 | 6.7 | 1.7 | 0.3 | 6.9 | 3.1 | 0.5 | 3.9 | 8.2 | 9.4 | 1.0 |
| EURJPY | 7.5 | 1.0 | 2.4 | 6.1 | 2.3 | 3.5 | 2.2 | 7.4 | -3.0 | 2.8 | 0.0 | 4.2 | 3.1 | 6.4 | 2.6 | 8.6 | 3.8 |
| EURNZD | 0.7 | -2.2 | 3.1 | 1.5 | -2.2 | -1.2 | 1.9 | 10.6 | 5.0 | -1.4 | 2.0 | 8.3 | 1.5 | 1.1 | 4.0 | 9.6 | 1.5 |
| EURUSD | 2.3 | 1.8 | -1.0 | 2.9 | 1.5 | 2.7 | 2.9 | 7.7 | 3.1 | 6.7 | 5.5 | -3.1 | -0.2 | 3.2 | -1.0 | 5.0 | 1.1 |
| GBPAUD | 1.6 | 9.1 | 0.4 | 0.4 | 3.8 | 1.0 | -0.5 | 4.1 | 5.7 | 5.4 | 2.6 | 6.1 | -1.1 | 3.1 | 5.5 | -0.2 | -1.0 |
| GBPCAD | 4.8 | 3.4 | 1.6 | 9.4 | 3.3 | 1.7 | -2.3 | 10.0 | 1.1 | 1.1 | 4.9 | 7.8 | 4.8 | 4.2 | 4.1 | 7.5 | -0.6 |
| GBPCHF | -3.4 | 0.8 | -0.7 | 4.0 | 8.2 | 4.5 | -2.5 | 9.4 | -3.0 | 1.7 | 4.9 | 4.9 | -1.4 | -0.2 | 1.8 | -1.6 | -1.0 |
| GBPJPY | 1.5 | 1.9 | 2.8 | 6.4 | 4.1 | 0.2 | -1.6 | 3.6 | -1.3 | 8.5 | 4.0 | -1.1 | 4.2 | 9.3 | 4.3 | 0.1 | 1.9 |
| GBPNZD | -0.0 | 3.4 | 6.0 | 1.7 | 1.4 | 5.4 | 1.9 | 5.4 | 4.0 | 12.9 | -3.6 | 8.6 | 2.6 | -0.2 | 7.9 | 8.2 | -1.0 |
| GBPUSD | 0.3 | -1.2 | 0.6 | 3.5 | -3.7 | 0.5 | 3.8 | 4.1 | 3.7 | 1.5 | 4.8 | 2.1 | -1.4 | 4.2 | 2.0 | 6.7 | 3.0 |
| NZDCAD | 8.2 | 7.4 | 4.6 | 10.9 | 3.2 | 6.8 | 7.6 | 3.7 | -2.3 | -3.9 | 5.6 | 4.6 | -4.0 | 2.2 | -3.9 | 5.5 | 1.4 |
| NZDCHF | 7.0 | 4.3 | -0.5 | 1.9 | 4.8 | 0.9 | 6.0 | -0.4 | -0.7 | 2.5 | 5.0 | 4.4 | -1.4 | -5.7 | 1.1 | -5.4 | 0.4 |
| NZDJPY | 0.8 | 2.3 | 4.4 | 5.3 | 2.5 | 0.9 | 4.8 | 4.6 | -4.0 | 1.0 | 5.2 | -1.0 | 3.1 | 8.0 | 3.6 | 1.1 | -1.4 |
| NZDUSD | 4.3 | 1.4 | 1.7 | 5.2 | 4.2 | -0.9 | 3.0 | -0.9 | 3.9 | 4.8 | 4.2 | -1.5 | -2.5 | 0.3 | -1.1 | 2.5 | -1.0 |
| USDCAD | -3.0 | 6.3 | 3.6 | 3.4 | 5.2 | 1.8 | 7.6 | 2.0 | 2.7 | 5.2 | -2.3 | 4.4 | 3.7 | -0.4 | 10.6 | 2.5 | 2.8 |
| USDCHF | -1.9 | 2.8 | 12.3 | 1.6 | 5.4 | -0.6 | 2.3 | -3.7 | -0.3 | 2.2 | -1.7 | 7.0 | 9.1 | -1.7 | 6.9 | 2.0 | 0.5 |
| USDJPY | -1.6 | 2.5 | -0.6 | 3.0 | 0.7 | 5.2 | -1.1 | 2.8 | 8.5 | 3.6 | 4.6 | 7.0 | 13.5 | 2.3 | 1.8 | 2.8 | 0.6 |

### 6. Day-of-week of entry (EET trading day)

| dow | weekday | n | win_rate | mean_r | total_r |
|---|---|---|---|---|---|
| 0 | Mon | 630 | 0.7238 | 0.4135 | 260.5095 |
| 1 | Tue | 606 | 0.7211 | 0.4482 | 271.6108 |
| 2 | Wed | 632 | 0.7215 | 0.4449 | 281.1627 |
| 3 | Thu | 674 | 0.6884 | 0.3824 | 257.7275 |
| 4 | Fri | 562 | 0.7171 | 0.4093 | 230.0176 |
| 5 | Sat | 45 | 0.6889 | 0.3470 | 15.6135 |
| 6 | Sun | 3 | 0.6667 | 0.5100 | 1.5299 |

> **n<30 (directional-only):** Sun(n=3) — these are late-Friday-UTC 4H bars rolled onto the next EET trading day.

### 7. Concentration

Top-5 by total R (=24.2% of all realized R):

| pair | n | win_rate | mean_r | total_r | profit_factor |
|---|---|---|---|---|---|
| GBPCAD | 125 | 0.7600 | 0.5346 | 66.8294 | 3.2276 |
| EURGBP | 118 | 0.7542 | 0.5505 | 64.9543 | 3.2761 |
| GBPNZD | 104 | 0.7788 | 0.6212 | 64.6017 | 3.8724 |
| CHFJPY | 117 | 0.7436 | 0.5289 | 61.8840 | 3.0628 |
| EURJPY | 109 | 0.7523 | 0.5576 | 60.7759 | 3.2510 |

Bottom-5 by total R:

| pair | n | win_rate | mean_r | total_r | profit_factor |
|---|---|---|---|---|---|
| NZDCHF | 123 | 0.6179 | 0.1976 | 24.3015 | 1.5267 |
| GBPCHF | 109 | 0.6422 | 0.2424 | 26.4233 | 1.6942 |
| NZDUSD | 91 | 0.6703 | 0.3027 | 27.5438 | 1.9181 |
| AUDNZD | 107 | 0.6916 | 0.3196 | 34.2002 | 2.0506 |
| AUDCAD | 131 | 0.6718 | 0.2615 | 34.2536 | 1.8043 |

> No pair has net-negative total R.


---

## Concurrency / correlation / loss-clustering (3.5R)

> Descriptive PORTFOLIO risk surface over the deployed-policy EET frame. Mark-to-market = deployed running R (banked +1R partial + open leg at close, 3.5R frame), 1R=0.5% account. Common clock = union of 28 H4-EET cache grids over the trade span; **open positions are forward-filled across the clock within their interval** (a held position keeps its last mark over a bar its pair lacks), so concurrency / exposure / open-book equity are interval-consistent, not bar-presence (the latter would drop open positions at sparse weekend-boundary bars and manufacture spurious drawdowns). No config/exit/pair change, no WFO; v3.0.2 locked. **Not a tuning trigger** — informs the live-DD picture and kill-criteria docs only.

### Step 0 — DD framing verdict

> **The 7.80% worst-fold DD is PER-TRADE-SEQUENTIAL — concurrency is UNMODELLED.** `scripts/l_arc_10_v3/step_5.py:_equity_curve` compounds trades one at a time in `signal_bar_time` order (`eq += eq*0.005*R`, full realized R per trade); the Amendment-3 holdout rerun (`amendment_3_addendum._equity_curve`) uses identical math. No portfolio equity curve sums simultaneously-open positions. The 7.80% (`fundednext_cost_sweep` `worst_dd_at_rbase`=0.07799, central 1.5x-spread/0.5-slip cell) therefore does NOT see overlapping correlated open risk. **This probe decomposes the concurrency risk the gate omitted.** Open-book figures below are linear (1R=0.5%, additive) vs the gate's per-trade compounding — magnitudes are comparable for a risk-surface read, not byte-identical.

### Headline

- peak concurrency **25** open trades (mean 9.1, p99 20); N>=10 45% of bars
- worst currency concentration: net-long **EUR** peak 5.25% (13 trades same-direction)
- open-book max DD (pooled) **9.22%** / worst-fold portfolio DD **9.22%** vs sequential-gate 7.80%
- loss arrival: runs-test p=5.35e-37, monthly dispersion 1.87 (Poisson=1) -> **CLUSTERED**
- worst currency-shock: CHF 5 losses in 28h

### 1. Concurrency (open trades per H4 bar)

| metric | value |
|---|---|
| open trades p50 | 9.0 |
| open trades p90 | 14.0 |
| open trades p99 | 20.0 |
| open trades mean | 9.14 |
| peak concurrency | 25 |
| total H4 bars in span | 25742 |
| fraction of time N>=5 | 0.8656 |
| fraction of time N>=8 | 0.6399 |
| fraction of time N>=10 | 0.4535 |
| fraction of time N>=15 | 0.0921 |

Concurrency-at-entry (already-open trades when each fires): p50=9, p90=14, p99=20, mean=9.1, max=24.

### 2. Currency exposure (risk-weighted net, % of account)

> net = Σ(+base / −quote) over open trades × open_units × 0.5%. Ranked by peak |net|. One adverse move in the top currency hits all same-direction legs at once.

| ccy | peak_net_long_pct | peak_net_short_pct | peak_abs_net_pct | peak_net_long_trades | peak_net_short_trades | median_abs_net_pct |
|---|---|---|---|---|---|---|
| EUR | 5.2500 | -0.0000 | 5.2500 | 13 | 0 | 0.7500 |
| CHF | 1.0000 | 5.2500 | 5.2500 | 2 | 11 | 0.5000 |
| JPY | 0.0000 | 5.0000 | 5.0000 | 0 | 12 | 0.7500 |
| GBP | 3.7500 | 1.5000 | 3.7500 | 8 | 3 | 0.5000 |
| AUD | 3.7500 | 2.0000 | 3.7500 | 8 | 4 | 0.5000 |
| CAD | 3.0000 | 3.0000 | 3.0000 | 6 | 8 | 0.5000 |
| NZD | 3.0000 | 2.2500 | 3.0000 | 6 | 5 | 0.5000 |
| USD | 3.0000 | 2.5000 | 3.0000 | 6 | 6 | 0.5000 |

### 3. Correlated joint drawdown

- Pooled open-book equity max DD: **9.22%** (relative) / 10.17 account-pp peak-to-trough.

- Per-fold (OOS-year, reset) portfolio DD:

| fold | n_trades | portfolio_dd_pct |
|---|---|---|
| 1 | 201 | 9.22 |
| 2 | 182 | 9.09 |
| 3 | 179 | 4.55 |
| 4 | 195 | 7.05 |
| 5 | 192 | 7.46 |
| 6 | 190 | 6.85 |
| 7 | 171 | 6.73 |
| 8 | 190 | 8.26 |
| 9 | 176 | 5.87 |
| 10 | 195 | 8.45 |
| 11 | 188 | 7.26 |
| 12 | 1093 | 6.03 |

- **Worst-fold portfolio DD = 9.22%** vs the per-trade-sequential gate figure **7.80%** (Step-0). Concurrency AMPLIFIES the DD beyond the gate number.

- Pairwise per-bar MtM-increment correlation, co-open trades:

| group | n_pairs | mean_corr | median_corr |
|---|---|---|---|
| shares a currency leg | 12035 | 0.1899 | 0.3185 |
| no shared leg | 13677 | -0.0005 | 0.0106 |

> 25712 co-open pairs (overlap>=8 bars). Shares-a-leg vs no-shared-leg quantifies how much shared currency legs co-move the open book.

- Top-5 open-book drawdown episodes:

| rank | peak_ts | trough_ts | depth_pct | n_open_at_trough | dominant_ccy | trades_at_trough |
|---|---|---|---|---|---|---|
| 1 | 2010-04-30 13:00 | 2010-05-17 01:00 | 9.224 | 7 | CHF | 48,52,57,58,59,60,62 |
| 2 | 2011-03-07 06:00 | 2011-03-17 02:00 | 7.056 | 10 | AUD | 222,223,224,226,228,229,230,231 |
| 3 | 2010-07-28 05:00 | 2010-09-09 05:00 | 6.083 | 9 | EUR | 106,118,120,121,123,125,128,129 |
| 4 | 2010-11-04 10:00 | 2010-12-30 02:00 | 5.649 | 19 | JPY | 171,174,179,181,182,185,188,189 |
| 5 | 2011-07-04 21:00 | 2011-07-18 13:00 | 4.243 | 14 | CAD | 272,274,275,278,289,294,295,296 |

### 4. Loss clustering — temporal

| metric | value |
|---|---|
| runs_observed | 998 |
| runs_expected | 1289.6 |
| runs_z | -12.71 |
| runs_p (Wald-Wolfowitz) | 5.35e-37 |
| max_losing_streak | 10 |
| dispersion_index_month (Poisson=1) | 1.873 |
| dispersion_index_week (Poisson=1) | 1.898 |
| base_loss_rate | 0.2865 |

> Verdict: **CLUSTERED**. Runs-test z=-12.7, p=5.35e-37 (p>0.05 would mean win/loss order is random; here losses arrive in non-random clusters); monthly dispersion 1.87, weekly 1.90 (Poisson=1; >1.25 = over-dispersed/bursty). Max losing streak 10 trades.

Losing-streak length distribution (count of streaks):

| streak_len | count |
|---|---|
| 1 | 284 |
| 2 | 118 |
| 3 | 52 |
| 4 | 23 |
| 5 | 10 |
| 6 | 7 |
| 7 | 1 |
| 8 | 2 |
| 10 | 2 |

### 5. Loss clustering — shared currency leg (root cause)

> Currency-shock window = >=K losing trades sharing one currency leg exiting within T H4 bars. Sweep:

| K | T_bars | n_shock_events |
|---|---|---|
| 3 | 4 | 34 |
| 3 | 8 | 56 |
| 3 | 12 | 71 |
| 4 | 4 | 5 |
| 4 | 8 | 9 |
| 4 | 12 | 14 |
| 5 | 4 | 1 |
| 5 | 8 | 4 |
| 5 | 12 | 4 |

Worst currency-shock events (K>=3, T=8 bars / 32h):

| ccy | n_losses | span_hours | start | trades |
|---|---|---|---|---|
| CHF | 5 | 28 | 2017-10-25 13:00 | 1453,1440,1457,1451,1463 |
| CHF | 5 | 4 | 2021-07-08 05:00 | 2175,2169,2158,2134,2168 |
| EUR | 5 | 32 | 2023-11-28 14:00 | 2620,2625,2622,2626,2623 |
| JPY | 5 | 24 | 2025-01-29 18:00 | 2850,2847,2849,2839,2848 |
| CAD | 4 | 0 | 2021-07-08 05:00 | 2134,2178,2158,2175 |

> Co-exposed conditional loss rate: when a trade loses, the fraction of concurrently-open trades that also lose — **sharing a currency leg 0.3107** (n=5968) vs **no shared leg 0.1884** (n=6582); base loss rate 0.2865. The load-bearing contrast is shared-vs-no-shared: a co-open trade sharing a currency leg with a loser is **1.65x** more likely to also lose than one with no shared leg — direct evidence that shared currency legs, not time alone, drive joint losses.

### 6. Same-pair concurrency

- Bars with the same pair open >1x simultaneously: 18425 (71.58% of span). Overlapping same-pair trade-pairs: 758.

- Gap between consecutive same-pair entries (H4 bars): p50=178, p25=87, min=20 (n=3124).

- Stacked same-pair co-outcome: both win 0.734, both lose 0.156, split 0.111 (n=758 pairs; independence would give both-lose≈0.082).


> **Thin-cell flags:** per-fold portfolio DD for short OOS years and any K/T shock cell with n_shock_events small are directional-only; same-pair stacked-pair co-outcome (n=758) is directional if <30.


---

## Portfolio daily DD vs 5% (3.5R)

> Descriptive risk surface. REUSES the concurrency probe's portfolio-equity reconstruction (imported `build_mtm` + `build_portfolio`, interval-consistent forward-filled), bucketed by EET trading day (00:00 EET reset, the Amendment-6 boundary). The per-trade-sequential gate never measured portfolio daily DD; the binding 5ers limit is 5% daily. Two references reported (DAY-START = drop below opening equity, 5ers-like; DAY-HIGH = drop from running intraday peak). **The 5ers daily basis (balance vs equity, reset time) is externally unconfirmed and parameterised here — the live figure is whichever basis 5ers confirms.** No config/exit/sizing change, no WFO; v3.0.2 locked. Not a tuning trigger.

### Headline

- **Worst daily DD (close-mark): DAY-START 4.66% (2010-05-06), DAY-HIGH 4.89% (2010-05-06).** Conservative intrabar-MAE bound (day-start): **11.17%**.
- **Margin to the 5% limit:** 0.34pp (close-mark day-start); -6.17pp under the conservative MAE bound.
- Days over 4%: 1 (day-start) / 1 (day-high). Days over 5%: 0 (day-start) / 0 (day-high), of 4750 trading days.
- **Worst day split:** realized (closed) -0.62% vs floating (open MtM) 5.27% (realized = -13% of the drop); dominant adverse currency **GBP**; 13 open / 4 SL closes that day.

### 1. Daily DD distribution (account-%)

| reference | p50 | p90 | p99 | max | mean |
|---|---|---|---|---|---|
| day-start | 0.0462 | 0.3087 | 0.9107 | 4.6554 | 0.1153 |
| day-high | 0.0918 | 0.3779 | 1.0365 | 4.8917 | 0.1599 |
| conservative MAE (day-start) | 0.9408 | 2.7214 | 6.7126 | 11.1729 | 1.3399 |

Days exceeding threshold (of 4750 trading days):

| threshold_pct | days_daystart | days_dayhigh | days_conservative |
|---|---|---|---|
| 2 | 6 | 7 | 827 |
| 3 | 3 | 3 | 391 |
| 4 | 1 | 1 | 227 |
| 5 | 0 | 0 | 146 |

### 2. Worst-10 days (by DAY-START close-mark DD)

| day | dd_daystart_pct | dd_dayhigh_pct | dd_cons_daystart_pct | n_open | n_closes | n_sl_closes | realized_drop_pct | floating_drop_pct | dominant_ccy |
|---|---|---|---|---|---|---|---|---|---|
| 2010-05-06 | 4.6554 | 4.8917 | 8.1225 | 13 | 5 | 4 | -0.6177 | 5.2730 | GBP |
| 2010-04-27 | 3.5030 | 3.5030 | 7.3998 | 17 | 5 | 0 | -1.3110 | 4.8139 | AUD |
| 2011-03-15 | 3.2211 | 3.2211 | 4.5571 | 11 | 1 | 0 | -0.1016 | 3.3227 | AUD |
| 2010-04-23 | 2.2718 | 2.2718 | 6.7718 | 16 | 0 | 0 | -0.0000 | 2.2718 | JPY |
| 2011-03-16 | 2.1983 | 2.9285 | 3.6295 | 10 | 0 | 0 | -0.0000 | 2.1983 | AUD |
| 2010-05-11 | 2.0779 | 2.0779 | 4.5432 | 8 | 0 | 0 | -0.0000 | 2.0779 | GBP |
| 2010-10-19 | 1.7922 | 1.7922 | 3.7637 | 15 | 2 | 2 | 0.8235 | 0.9688 | EUR |
| 2011-01-04 | 1.7584 | 2.5326 | 7.1277 | 20 | 5 | 0 | -1.0800 | 2.8384 | JPY |
| 2010-04-12 | 1.7193 | 1.8895 | 5.3876 | 13 | 5 | 0 | -2.1967 | 3.9159 | AUD |
| 2011-01-28 | 1.6068 | 1.6068 | 4.9234 | 11 | 2 | 0 | -0.9127 | 2.5195 | JPY |

### 3. Realized vs floating (worst-10 days)

> Closed losses are unrecoverable; floating may reverse before the daily close — which matters depends on the (unconfirmed) 5ers basis. Split is of the day-start trough drop.

| day | total_drop_pct | realized_drop_pct | floating_drop_pct | realized_share_pct |
|---|---|---|---|---|
| 2010-05-06 | 4.6554 | -0.6177 | 5.2730 | -13.2685 |
| 2010-04-27 | 3.5030 | -1.3110 | 4.8139 | -37.4252 |
| 2011-03-15 | 3.2211 | -0.1016 | 3.3227 | -3.1533 |
| 2010-04-23 | 2.2718 | -0.0000 | 2.2718 | -0.0000 |
| 2011-03-16 | 2.1983 | -0.0000 | 2.1983 | -0.0000 |
| 2010-05-11 | 2.0779 | -0.0000 | 2.0779 | -0.0000 |
| 2010-10-19 | 1.7922 | 0.8235 | 0.9688 | 45.9470 |
| 2011-01-04 | 1.7584 | -1.0800 | 2.8384 | -61.4215 |
| 2010-04-12 | 1.7193 | -2.1967 | 3.9159 | -127.7654 |
| 2011-01-28 | 1.6068 | -0.9127 | 2.5195 | -56.7989 |

### 4. Conservative intrabar bound vs close-mark

> Worst-day DAY-START DD: close-mark **4.66%** vs intrabar-MAE upper bound **11.17%** — a 6.52pp uncertainty band from unrecorded intrabar lows + the simultaneity/cumulative-MAE conservatism. True worst-day daily DD lies between these. Even the upper bound BREACHES 5% (6.17pp over).

### 5. Currency attribution of the daily-DD tail

> Days with day-start DD > 3% (3 days), grouped by dominant adverse currency at the trough:

| dominant_ccy | n_days | max_dd_pct | mean_dd_pct |
|---|---|---|---|
| AUD | 2 | 3.5030 | 3.3620 |
| GBP | 1 | 4.6554 | 4.6554 |

> Cross-check: the max-DD episode (concurrency probe) was CHF-dominated; the daily-DD tail spreads across currencies.


> **Note:** figures are linear (1R=0.5%, additive open+closed MtM) vs the gate's per-trade compounding — risk-surface comparable, not byte-identical. Live-relevant basis is whichever 5ers confirms.

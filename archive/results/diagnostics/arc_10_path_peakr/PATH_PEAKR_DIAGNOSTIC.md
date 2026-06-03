# Arc 10 (DLR) — Path / Peak-R Diagnostic

> READ-ONLY forward-path geometry diagnostic. Does NOT modify Arc 10 config / risk / canonical numbers. Informs FUTURE-arc architecture choices (stop width, A3 deferred-entry, A4 differentiated-exit).


## A. DESCRIPTIVE — by realised-R bucket

Per-trade forward paths on the l_arc_10 frame (see FRAME_PROVENANCE at the bottom). Percentiles in ATR-at-entry units; `t_to_peak` in bars. Realised-R buckets are **SL-honest deployed-R** (1R = 3.5 ATR; see the exit-fidelity finding below).

| realized_bucket | n | pct_pool | prepeak_maxdip_atr_p10 | prepeak_maxdip_atr_p25 | prepeak_maxdip_atr_p50 | prepeak_maxdip_atr_p75 | prepeak_maxdip_atr_p90 | fulllife_mae_atr_p10 | fulllife_mae_atr_p25 | fulllife_mae_atr_p50 | fulllife_mae_atr_p75 | fulllife_mae_atr_p90 | terminal_peak_atr_p10 | terminal_peak_atr_p25 | terminal_peak_atr_p50 | terminal_peak_atr_p75 | terminal_peak_atr_p90 | t_to_peak_p10 | t_to_peak_p25 | t_to_peak_p50 | t_to_peak_p75 | t_to_peak_p90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| <0 | 1660 | 50.29 | -9.37 | -6.44 | -4.04 | -1.28 | -0.51 | -20.33 | -13.87 | -9.47 | -6.20 | -4.45 | 0.65 | 1.50 | 3.13 | 7.37 | 12.30 | 1.00 | 11.00 | 94.50 | 195.00 | 232.00 |
| 0-1 | 1013 | 30.69 | -4.35 | -2.97 | -1.82 | -0.92 | -0.40 | -16.02 | -9.70 | -4.81 | -2.37 | -1.18 | 4.15 | 5.13 | 7.07 | 10.94 | 16.70 | 23.00 | 48.00 | 115.00 | 198.00 | 232.00 |
| 1-2 | 534 | 16.18 | -2.95 | -2.15 | -1.22 | -0.54 | -0.27 | -9.15 | -4.20 | -2.22 | -1.05 | -0.45 | 8.44 | 9.55 | 12.34 | 15.70 | 21.15 | 43.00 | 76.00 | 137.50 | 208.00 | 235.00 |
| 2-3 | 75 | 2.27 | -2.69 | -2.05 | -1.25 | -0.44 | -0.23 | -3.76 | -2.36 | -1.46 | -0.54 | -0.25 | 15.26 | 16.55 | 18.77 | 22.11 | 27.36 | 61.80 | 104.00 | 163.00 | 217.00 | 233.00 |
| 3-5 | 19 | 0.58 | -2.07 | -1.72 | -1.08 | -0.69 | -0.24 | -2.07 | -1.72 | -1.08 | -0.69 | -0.24 | 24.52 | 28.27 | 31.19 | 42.74 | 49.74 | 82.60 | 109.00 | 193.00 | 232.50 | 237.20 |


**Terminal-peak (deployed-R) × realised-R cross-tab (trade counts):**

| peak_atr_bucket | <0 | 0-1 | 1-2 | 2-3 | 3-5 |
|---|---|---|---|---|---|
| <0.5R | 1169 | 374 | 1 | 0 | 0 |
| 0.5-1R | 324 | 429 | 263 | 0 | 0 |
| 1-2R | 144 | 177 | 237 | 62 | 2 |
| 2-3R | 19 | 28 | 24 | 9 | 10 |
| 3R+ | 4 | 5 | 9 | 4 | 7 |


## B. STOP-WIDTH READ — is 3.5×ATR right?

`pct_prepeak_dip_below_-3.5atr` = share of the group whose deepest dip BEFORE their unconstrained peak pierces the −3.5 ATR stop line (the unstopped-path geometry; dips in ATR).

| group | n | pct_prepeak_dip_below_-3.5atr | prepeak_dip_atr_p50 | prepeak_dip_atr_p25 | prepeak_dip_atr_p10 |
|---|---|---|---|---|---|
| <0 | 1660 | 59.28 | -4.04 | -6.44 | -9.37 |
| 0-1 | 1013 | 13.33 | -1.82 | -2.97 | -4.35 |
| 1-2 | 534 | 2.06 | -1.22 | -2.15 | -2.95 |
| 2-3 | 75 | 1.33 | -1.25 | -2.05 | -2.69 |
| 3-5 | 19 | 0.00 | -1.08 | -1.72 | -2.07 |
| ALL winners (realized>0) | 1631 | 8.58 | -1.54 | -2.66 | -3.44 |
| ALL | 3301 | 34.26 | -2.22 | -4.51 | -7.62 |


- For **winners (realised>0, SL-honest)**, **8.58%** have a pre-peak dip beyond −3.5 ATR; the −3.5 ATR line sits at the **~91th percentile** of winners' pre-peak dip distribution.


**Stop-width sweep — SL-honest (primary) vs fast-replay, realised in EACH stop's own R units (1R = sl_mult×ATR):**

| sl_mult | honest_mean_R | honest_win_rate_pct | honest_n_full_loss | honest_total_R | replay_mean_R | replay_win_rate_pct | replay_minus_honest_mean_R |
|---|---|---|---|---|---|---|---|
| 2.50 | -0.01 | 48.68 | 1682 | -17.89 | 0.60 | 78.73 | 0.60 |
| 3.00 | 0.01 | 49.14 | 1664 | 19.43 | 0.52 | 75.34 | 0.51 |
| 3.50 | -0.01 | 49.41 | 1646 | -19.27 | 0.41 | 71.55 | 0.42 |
| 4.00 | -0.00 | 49.62 | 1625 | -4.51 | 0.33 | 68.04 | 0.33 |


## C. EXIT-HEADROOM — the A4 ceiling

`gap = terminal_peak (unconstrained) − realised` (SL-honest captured P&L, same frame). Upper bound on what a perfect differentiated exit could chase.

| metric | value |
|---|---|
| n | 3301.000 |
| total_foregone_depR | 7988.262 |
| mean_foregone_depR | 2.420 |
| median_foregone_depR | 1.828 |
| gap_atr_p10 | 3.273 |
| gap_atr_p25 | 4.444 |
| gap_atr_p50 | 6.399 |
| gap_atr_p75 | 10.455 |
| gap_atr_p90 | 15.856 |
| gap_depR_p10 | 0.935 |
| gap_depR_p25 | 1.270 |
| gap_depR_p50 | 1.828 |
| gap_depR_p75 | 2.987 |
| gap_depR_p90 | 4.530 |
| winners_n | 1631.000 |
| winners_total_foregone_depR | 3788.838 |
| winners_mean_foregone_depR | 2.323 |


## D. DEFERRED-ENTRY / EARLY-PATH SEPARABILITY

RF (Appendix-A defaults: n_estimators=200, max_depth=6, min_samples_leaf=50, random_state=42, n_jobs=1), 5-fold TimeSeriesSplit, chronological by entry_time. Features at bar N: running MFE(N), running MAE(N), close(N), velocity — NO lookahead. Target1 = SL-honest realised R>0; Target2 = terminal peak ≥ 2R (7 ATR).

| N | n | auc_y_win | auc_y_win_std | auc_y_peak2r | auc_y_peak2r_std |
|---|---|---|---|---|---|
| 0 | 3301 | 0.601 | 0.037 | 0.530 | 0.028 |
| 1 | 3301 | 0.636 | 0.029 | 0.548 | 0.019 |
| 2 | 3301 | 0.673 | 0.036 | 0.556 | 0.026 |
| 3 | 3301 | 0.697 | 0.022 | 0.562 | 0.018 |
| 5 | 3301 | 0.745 | 0.029 | 0.595 | 0.009 |
| 8 | 3301 | 0.796 | 0.016 | 0.614 | 0.017 |


**Deployment-honesty block (mandatory) — naive defer-enter-at-N (best N=8):**

AUC at N=8 (0.796) is materially above bar-0 (0.601). Defer rule: skip if running MAE(N) below threshold; admitted trades pay the late-entry tax of forgoing the 0→N move. Admit-only mean is shown WITH the reject-pool + foregone-R tax — never in isolation.

| N | mae_thr_depR | n_admit | n_reject | admit_only_mean_depR_NO_tax | foregone_mean_from_late_entry_depR | admit_net_mean_depR_late_entry | reject_pct_were_winners | reject_mean_realized_depR | full_pool_mean_depR | net_edge_vs_full_pp |
|---|---|---|---|---|---|---|---|---|---|---|
| 8 | -1.000 | 3025 | 276 | 0.084 | 0.069 | 0.016 | 0.362 | -0.995 | -0.006 | 0.022 |
| 8 | -0.750 | 2666 | 635 | 0.168 | 0.147 | 0.021 | 13.543 | -0.737 | -0.006 | 0.027 |
| 8 | -0.500 | 2082 | 1219 | 0.268 | 0.259 | 0.010 | 26.415 | -0.474 | -0.006 | 0.016 |
| 8 | -0.350 | 1566 | 1735 | 0.349 | 0.346 | 0.003 | 33.833 | -0.327 | -0.006 | 0.009 |
| 8 | -0.250 | 1128 | 2173 | 0.415 | 0.425 | -0.010 | 38.564 | -0.224 | -0.006 | -0.004 |


---
## ⚠ Exit-fidelity finding (load-bearing for B, C, D)

The fast Step-5 replay `simulate_path` (`sl_partial_close_1r_runner_trail`) only applies the −3.5 ATR SL to the **runner, after** the +1R partial fires (its `sl_breach > tp1_i` guard) — it **ignores SL breaches that occur before the partial**. On the unstopped paths, **22.1% of trades reach +1R only after their low has already pierced −3.5 ATR**; a live SL-honest engine stops these at −1R first.

- **Replay (optimistic):** mean **0.414 R_dep**, win **71.6%**.
- **SL-honest (live-faithful, this doc's primary):** mean **-0.006 R_dep**, win **49.4%**.

The **live `MultiPairBacktester` driver IS SL-honest** — its always-on intra-bar SL (`bar.low_bid <= sl_price`) stops the position at −1R before any partial (see `core/sim/exit_policies/sl_partial_close_1r_runner_trail.py` docstring, Stage 1). So the LIVE EA / engine takes the correct trades; the optimism is confined to the FAST REPLAY used for Step-5 *ranking*. The canonical Step-5 `realized_r_3p5` is the replay value, and the same optimism is visible in the canonical pool: **30.9% of canonical winners** pierced −3.5 ATR within their deployed window; forcing those to −1R moves canonical mean realised **0.418 → 0.005 R_dep** and win **71.4% → 49.3%** (deployed-window proxy; the strict pre-tp1 measure needs per-bar paths, which are gone for the canonical pool).

> **Scope:** This is a READ-ONLY diagnostic; it does NOT alter any canonical number. It is **flagged for the canonical/engine owners** — the live deployment is unaffected (live engine is SL-honest), but the Step-5 ranking/gate replay over-counts winners and any future arc using `simulate_path` inherits the bias. Recommend the fast replay censor pre-tp1 SL breaches (one-line fix: treat `sl_breach < tp1_i` as a −1R stop).


---
## Canonical v3.0.2 cross-validation (frame representativeness)

Apples-to-apples: l_arc_10 **replay** realised vs canonical **replay** `realized_r_3p5` (both ignore pre-tp1 SL), plus exit-agnostic geometry stats. Confirms the l_arc_10 frame is the same trade population as the live EET pool.

| metric | l_arc_10 | canonical_v302 |
|---|---|---|
| n_trades | 3301.000 | 3152.000 |
| REPLAY win% (apples-to-apples) | 71.554 | 71.352 |
| REPLAY mean R (apples-to-apples) | 0.414 | 0.418 |
| median terminal_peak ATR | 6.678 | 6.740 |
| p90 terminal_peak ATR | 17.387 | 17.540 |
| total foregone depR vs REPLAY (A4 ceiling) | 6603.847 | 6345.194 |


**Replay realised-R bucket shares (%):**

| realized_bucket | l_arc_10_replay_pct | canonical_v302_replay_pct |
|---|---|---|
| <0 | 28.17 | 28.49 |
| 0-1 | 45.96 | 45.37 |
| 1-2 | 21.66 | 21.73 |
| 2-3 | 3.48 | 3.52 |
| 3-5 | 0.73 | 0.86 |
| 5+ | 0.00 | 0.03 |


---
## VERDICTS

**1. Stop width OK? — YES, 3.5×ATR is adequate; the result is risk-normalised-neutral across 2.5–4.0.** SL-honest mean R is 2.5=-0.005 / 3.0=+0.006 / 3.5=-0.006 / 4.0=-0.001 — a total spread of only 0.012R, i.e. **noise**; there is no material risk-normalised case to re-tune the stop in either direction. Win-rate rises monotonically with width (48.7%→49.6%) but that is purely mechanical (a wider stop is breached less often) and is exactly offset in R because each loss is a wider −1R. Only **8.6% of SL-honest winners** dip beyond −3.5 ATR before peaking (the −3.5 line sits ~p91 of their pre-peak dip distribution), so 3.5×ATR is NOT cutting eventual winners short. The real stop-width story is the **exit-fidelity finding**: the stop is load-bearing (SL-honest vs replay mean = -0.006 vs +0.414 R), and the replay that justified the gate effectively ignored it on 22% of trades.


**2. A3 deferred-entry viable? — NO (marginal at best); the AUC lift is largely mechanical, not tradeable.** Early-path AUC for `realised>0` climbs 0.60→0.80 from N=0→8, but that rise is mostly **contemporaneous-outcome leakage**: by bar 8 a large share of trades have already partialled or stopped (median t-to-peak ≈ 113 bars; many losers resolve in <8), so 'observing the path to N' increasingly means 'observing the result'. The honest test is the defer-enter-at-N simulation: the net edge vs the full pool is positive at 4/5 thresholds but **tiny (max +0.027 R/trade)**, it rejects 13–39% of trades of which a rising share are real winners, and it leaves the pool at ≈break-even (full pool -0.006 R SL-honest). Deferred entry does NOT rescue the edge and is not worth an A3 build on this evidence; true pre-entry separability (N=0 AUC 0.60) is weak.


**3. A4 differentiated-exit headroom? — a real but bounded ceiling; treat the headline total as an over-count.** The full-pool gap (7988 R_dep, 2.42/trade) is **inflated by stopped-then-recovered losers** whose unconstrained peak is fantasy (they were correctly stopped and could not be held to it). The meaningful figure is the **winners-only ceiling ≈2.32 R_dep/trade** over 1631 winners — and even that is an upper bound assuming capture of the unconstrained peak, which no causal exit achieves; the runner-trail already harvests part of it. Net: there is moderate, genuine upside for a smarter runner/exit on the WINNER subset, but it is far smaller than the raw total suggests. **Any A4 work must be measured against an SL-honest baseline** (exit-fidelity finding) or the apparent improvement will be illusory.


> **Overarching note.** The single most consequential output of this diagnostic is the **exit-fidelity finding**, not the three architecture reads: under a faithful always-on 3.5×ATR stop the l_arc_10 frame's gross per-trade edge is ≈0 R (vs +0.41 R under the replay the canonical Step-5/gate uses), and the same optimism is present in the canonical pool. This is flagged for the canonical/engine owners to verify on the live EET pool; it is out of scope for this read-only diagnostic to adjudicate.


---
## FRAME_PROVENANCE

- **Per-bar source = `results/l_arc_10/step_1/trade_paths.parquet`** (3301 trades, full unstopped forward path 0→240 bars/trade). The **UTC-convention precursor** of the canonical v3.0.2 (EET) DLR pool.
- **Why not the canonical pool directly:** the v3.0.2 per-bar paths are **unrecoverable** here — the `H4_5ers_eet` cache and the sha-`05dea9` frame were removed 2026-05-31 and raw HistData is a 4.6 MB stub (not the 52 GB source), so the forward paths cannot be regenerated. l_arc_10 is the only surviving per-bar artefact.
- **Representativeness:** same DLR signal, same 28-pair universe, same 2010-2026 span, same R-unit recording (1R_pool = 2.0×ATR_entry), near-identical size (3301 vs 3152) and exit mix (~85% stoploss / ~15% time-exit); the cross-validation above confirms the replay realised-R, terminal-peak and bucket shares match the canonical pool.
- **Units:** ATR = pool-R × 2.0; deployed-R (1R_dep = 3.5×ATR = the SL) = pool-R × (2/3.5). The 3.5×ATR stop line = −3.5 ATR = −1.0 R_dep.
- **Realised R = SL-honest** (`sl_honest_realized`, mirrors the live driver's always-on intra-bar SL); the optimistic fast replay (`simulate_path`) is retained as a comparison column. Deterministic (random_state=42, n_jobs=1). Artefact SHAs in `manifest_sha256.txt`.

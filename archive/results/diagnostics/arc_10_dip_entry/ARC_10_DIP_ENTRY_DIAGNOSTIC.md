# Arc 10 (DLR) — Dip Distribution + Deferred Entry + Path Export

> READ-ONLY extension of the Arc 10 path/peak-R diagnostic. Does NOT modify Arc 10 config / risk / exit / canonical numbers. Informs future-arc entry/exit mechanics. Realised R = **SL-honest** deployed exit (always-on intra-bar SL; see the prior diagnostic's exit-fidelity finding — the fast Step-5 replay was optimistic and over-counted winners).


## A. PRE-PEAK DIP DISTRIBUTION

Depth (positive ATR) of the deepest dip BEFORE the unstopped terminal peak; and `first_bar_low` = entry-bar dip depth (drives the C1 limit-entry sim).

| group | metric | n | p10 | p25 | p50 | p75 | p90 | p95 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL_winners | prepeak_dip | 1638 | 0.320 | 0.727 | 1.548 | 2.674 | 3.439 | 5.090 | 25.293 | 1.991 |
| ALL_winners | first_bar_low | 1638 | 0.099 | 0.204 | 0.387 | 0.643 | 0.970 | 1.253 | 2.633 | 0.477 |
| ALL_trades | prepeak_dip | 3301 | 0.413 | 0.916 | 2.217 | 4.509 | 7.621 | 9.845 | 25.875 | 3.296 |
| ALL_trades | first_bar_low | 3301 | 0.113 | 0.236 | 0.436 | 0.739 | 1.128 | 1.531 | 8.093 | 0.561 |


**Histogram (counts / % per ATR bin):**

| group | metric | bin_atr | n | pct |
|---|---|---|---|---|
| ALL_winners | prepeak_dip | 0-0.5 | 282 | 17.22 |
| ALL_winners | prepeak_dip | 0.5-1.0 | 267 | 16.30 |
| ALL_winners | prepeak_dip | 1.0-1.5 | 244 | 14.90 |
| ALL_winners | prepeak_dip | 1.5-2.0 | 213 | 13.00 |
| ALL_winners | prepeak_dip | 2.0-2.5 | 172 | 10.50 |
| ALL_winners | prepeak_dip | 2.5-3.0 | 168 | 10.26 |
| ALL_winners | prepeak_dip | 3.0-3.5 | 152 | 9.28 |
| ALL_winners | prepeak_dip | 3.5+ | 140 | 8.55 |
| ALL_winners | first_bar_low | 0-0.5 | 1047 | 63.92 |
| ALL_winners | first_bar_low | 0.5-1.0 | 443 | 27.05 |
| ALL_winners | first_bar_low | 1.0-1.5 | 101 | 6.17 |
| ALL_winners | first_bar_low | 1.5-2.0 | 41 | 2.50 |
| ALL_winners | first_bar_low | 2.0-2.5 | 5 | 0.31 |
| ALL_winners | first_bar_low | 2.5-3.0 | 1 | 0.06 |
| ALL_winners | first_bar_low | 3.0-3.5 | 0 | 0.00 |
| ALL_winners | first_bar_low | 3.5+ | 0 | 0.00 |
| ALL_trades | prepeak_dip | 0-0.5 | 445 | 13.48 |
| ALL_trades | prepeak_dip | 0.5-1.0 | 444 | 13.45 |
| ALL_trades | prepeak_dip | 1.0-1.5 | 368 | 11.15 |
| ALL_trades | prepeak_dip | 1.5-2.0 | 286 | 8.66 |
| ALL_trades | prepeak_dip | 2.0-2.5 | 228 | 6.91 |
| ALL_trades | prepeak_dip | 2.5-3.0 | 214 | 6.48 |
| ALL_trades | prepeak_dip | 3.0-3.5 | 185 | 5.60 |
| ALL_trades | prepeak_dip | 3.5+ | 1131 | 34.26 |
| ALL_trades | first_bar_low | 0-0.5 | 1902 | 57.62 |
| ALL_trades | first_bar_low | 0.5-1.0 | 960 | 29.08 |
| ALL_trades | first_bar_low | 1.0-1.5 | 267 | 8.09 |
| ALL_trades | first_bar_low | 1.5-2.0 | 113 | 3.42 |
| ALL_trades | first_bar_low | 2.0-2.5 | 40 | 1.21 |
| ALL_trades | first_bar_low | 2.5-3.0 | 9 | 0.27 |
| ALL_trades | first_bar_low | 3.0-3.5 | 3 | 0.09 |
| ALL_trades | first_bar_low | 3.5+ | 7 | 0.21 |


- **Of winners, 33.5% dip ≤1 ATR, 61.4% dip ≤2 ATR, 82.2% dip ≤3 ATR before peaking.**


## B. SL-WIDTH GRID (SL-honest, 1.0–4.0 ×ATR)

`mean_R` and DD are **risk-normalised** (1R = sl_mult×ATR), so a tighter stop's larger position size is already reflected. `lot_multiplier_vs_3p5` = 3.5/sl (same $-risk → more lots at tighter SL) shown for transparency. `worst_fold_trailing_dd` is **per-trade-sequential** at r_base=0.5% (approximation — understates concurrent-portfolio DD; cross-SL relative only).

| sl_mult_atr | lot_multiplier_vs_3p5 | mean_R | win_rate_pct | pct_winners_stopped_before_peak | worst_fold_trailing_dd_pct | total_R |
|---|---|---|---|---|---|---|
| 1.000 | 3.500 | -0.063 | 42.805 | 48.620 | 22.452 | -207.006 |
| 1.500 | 2.333 | -0.036 | 46.804 | 35.663 | 19.989 | -119.473 |
| 2.000 | 1.750 | -0.017 | 47.713 | 26.413 | 18.384 | -55.741 |
| 2.500 | 1.400 | -0.004 | 48.682 | 18.793 | 13.349 | -13.458 |
| 3.000 | 1.167 | 0.008 | 49.197 | 12.869 | 15.126 | 27.928 |
| 3.500 | 1.000 | 0.001 | 49.621 | 8.547 | 19.765 | 4.340 |
| 4.000 | 0.875 | 0.013 | 50.167 | 5.737 | 18.826 | 43.686 |


## C. DEFERRED ENTRY (all trades taken; only the FILL changes)

C1 = limit-buy at entry−L×ATR (fills iff price dips to it in the window; no-fills are foregone and counted as 0 R in `net_mean_R_vs_full`). C2 = enter at market at bar N. `mean_R` is over FILLED trades; `net_mean_R_vs_full` is the pool-level mean (no-fills = 0) — the honest basis.

| variant | param | fill_window | fill_rate_pct | mean_entry_improve_atr | mean_R | win_rate_pct | worst_fold_dd_pct | net_mean_R_vs_full | note |
|---|---|---|---|---|---|---|---|---|---|
| baseline (bar-0 market) | - | - | 100.000 | 0.000 | 0.001 | 49.621 | 19.765 | 0.001 | all trades filled at bar-0 |
| C1 limit -L*ATR | L=0.25 | 1bar | 73.190 | 0.250 | 0.012 | 49.959 | 11.967 | 0.009 | 885 no-fills foregone (counted as 0 in net) |
| C1 limit -L*ATR | L=0.25 | 3bar | 85.520 | 0.250 | 0.025 | 50.443 | 14.212 | 0.021 | 478 no-fills foregone (counted as 0 in net) |
| C1 limit -L*ATR | L=0.5 | 1bar | 42.381 | 0.500 | 0.017 | 50.179 | 12.066 | 0.007 | 1902 no-fills foregone (counted as 0 in net) |
| C1 limit -L*ATR | L=0.5 | 3bar | 65.889 | 0.500 | 0.021 | 50.345 | 12.930 | 0.014 | 1126 no-fills foregone (counted as 0 in net) |
| C1 limit -L*ATR | L=0.75 | 1bar | 24.174 | 0.750 | -0.008 | 48.120 | 9.492 | -0.002 | 2503 no-fills foregone (counted as 0 in net) |
| C1 limit -L*ATR | L=0.75 | 3bar | 48.894 | 0.750 | -0.013 | 48.823 | 14.519 | -0.006 | 1687 no-fills foregone (counted as 0 in net) |
| C1 limit -L*ATR | L=1.0 | 1bar | 13.299 | 1.000 | -0.012 | 47.608 | 5.465 | -0.002 | 2862 no-fills foregone (counted as 0 in net) |
| C1 limit -L*ATR | L=1.0 | 3bar | 34.989 | 1.000 | -0.004 | 48.745 | 12.391 | -0.001 | 2146 no-fills foregone (counted as 0 in net) |
| C1 limit -L*ATR | L=1.5 | 1bar | 5.211 | 1.500 | -0.046 | 44.767 | 4.654 | -0.002 | 3129 no-fills foregone (counted as 0 in net) |
| C1 limit -L*ATR | L=1.5 | 3bar | 19.327 | 1.500 | -0.039 | 46.552 | 7.866 | -0.008 | 2663 no-fills foregone (counted as 0 in net) |
| C2 delay N bars | N=1 | market@N | 100.000 | 0.057 | 0.024 | 50.500 | 17.387 | 0.024 | enter at bar-N close; all trades taken (few short-path drops) |
| C2 delay N bars | N=2 | market@N | 100.000 | 0.062 | 0.023 | 50.560 | 19.784 | 0.023 | enter at bar-N close; all trades taken (few short-path drops) |
| C2 delay N bars | N=3 | market@N | 100.000 | 0.063 | 0.025 | 50.560 | 16.844 | 0.025 | enter at bar-N close; all trades taken (few short-path drops) |


## D. PER-TRADE PATH EXPORT

- `per_trade_path_long.csv` — one row per (trade_id, t≤240): mfe_atr_t, mae_atr_t, close_atr_t, is_winner, terminal_peak, prepeak_dip, arc10_realised_R. Substrate for an every-trade ATR-vs-t overlay.

- `path_envelope.csv` — per t: p10/25/50/75/90 envelope of running MFE & MAE (ATR) across all trades and across winners, plus n alive. Aggregate plot without rendering 5k+ lines.


**Envelope (median unless noted), selected t:**

| t | n_trades_alive | all_mae_p50 | all_mfe_p50 | win_mae_p50 | win_mfe_p50 | win_mfe_p90 |
|---|---|---|---|---|---|---|
| 0 | 3301 | -0.44 | 0.33 | -0.39 | 0.38 | 1.15 |
| 1 | 3301 | -0.60 | 0.49 | -0.49 | 0.61 | 1.65 |
| 2 | 3301 | -0.73 | 0.63 | -0.58 | 0.79 | 2.04 |
| 3 | 3301 | -0.84 | 0.75 | -0.65 | 1.03 | 2.41 |
| 5 | 3301 | -1.05 | 0.96 | -0.79 | 1.40 | 3.12 |
| 8 | 3301 | -1.32 | 1.17 | -0.90 | 1.79 | 3.82 |
| 13 | 3300 | -1.63 | 1.51 | -1.03 | 2.48 | 4.78 |
| 21 | 3299 | -2.02 | 1.91 | -1.16 | 3.32 | 6.16 |
| 34 | 3296 | -2.55 | 2.46 | -1.36 | 4.33 | 7.93 |
| 55 | 3292 | -3.18 | 3.15 | -1.59 | 5.43 | 9.88 |
| 89 | 3287 | -4.05 | 4.02 | -1.91 | 6.65 | 12.58 |
| 144 | 3278 | -5.18 | 5.18 | -2.45 | 8.13 | 15.80 |
| 233 | 3262 | -6.71 | 6.55 | -3.20 | 9.51 | 19.96 |


---
## INTERPRETATION

**A — dip shape.** Winners pull back **shallowly** before running: median pre-peak dip 1.55 ATR, p90 3.44 ATR; 61% never dip beyond 2 ATR and 82% never beyond 3 ATR before peaking. The entry-bar dip is even shallower — 91% of winners trade <1 ATR below entry on bar 0 — so a below-signal limit can only fill on a thin sliver of the favourable trades.

**B — SL width.** The 2.5–4.0 region is flat (mean-R spread 0.017 R) but the full 1.0–4.0 grid spans 0.076 R because **tighter stops (1.0–2.0 ATR) are strictly worse** (mean R -0.063 at 1.0 → -0.017 at 2.0 → +0.001 at 3.5). A tighter stop sizes up the SAME trades but converts the extra size into extra −1R stops more than one-for-one (it stops out trades that the wider stop would have ridden to the partial). There is no risk-normalised case to tighten.

**C — deferred entry.** Best net-of-foregone variant is `C2 delay N bars N=3` at net **+0.025 R** vs baseline +0.001 R — a gain of +0.023 R/trade. The benefit is **small and consistent** (delaying 1–3 bars or a tight 0.25–0.5 ATR limit catches the slight continuation of the DLR pullback for a ~0.05–0.06 ATR cheaper average fill), but **economically marginal** — it is within the reconstruction's approximation error and below typical round-trip costs. **Larger limits (L≥0.75 ATR) go net-negative**: fill rates collapse (≤25–49%) and the no-fill trades that ran away without dipping are exactly the winners, so the foregone-run tax dominates.


---
## VERDICTS

**1. Dip distribution shape:** winners pull back **shallowly** before running — median pre-peak dip 1.55 ATR, 34% ≤1 ATR / 61% ≤2 ATR / 82% ≤3 ATR; the 3.5 ATR stop sits in the far tail (only 8.5% of winners dip that far pre-peak). Entry-bar dips are tiny (winner median 0.39 ATR). The signal enters near a local low and rarely revisits it — consistent with a genuine swing-low-rejection entry, and the reason below-signal limits mostly miss.


**2. Is a tighter SL better? NO — and tighter is strictly worse.** The 2.5–4.0 region is risk-normalised-flat (spread 0.017 R, confirming the prior 0.012 R plateau) and tightening to 1.0–2.0 ATR actively degrades mean R (-0.063 at 1.0 vs +0.001 at 3.5). Sizing up on a tighter stop is paid back one-for-one (and worse) in extra −1R stops. 3.5×ATR is fine; no change warranted.


**3. Does deferred entry improve net fill? NOT MEANINGFULLY.** A tight limit (0.25–0.5 ATR) or a 1–3 bar delay yields a small, consistent net gain (best +0.025 R, `C2 delay N bars N=3`, vs baseline +0.001 R = +0.023 R/trade) — the DLR pullback tends to extend a hair — but it is **economically immaterial** (within reconstruction error, below costs) and does NOT change the pool's ≈break-even character. Larger below-signal limits are net-negative: they fill on a minority and forgo precisely the runaway winners. Deferred entry is not a lever worth deploying on this evidence; the dominant fact remains the exit-fidelity finding from the prior diagnostic.


---
## FRAME_PROVENANCE & METHOD

- **Frame:** reused `results/l_arc_10/step_1/trade_paths.parquet` (3301 trades, unstopped 0→240-bar paths) — the UTC-precursor of the canonical v3.0.2 (EET) pool; the v3.0.2 per-bar paths are unrecoverable (H4 cache + frame gone). Prior diagnostic cross-validated this frame to the canonical pool within <0.6 pp (same population).
- **Per-bar OHLC reconstruction:** the frame stores cumulative running mae/mfe + per-bar close only. Reconstructed low/high = the running extreme on bars that set a new extreme, else the close (clamped high≥close≥low). EXACT at extreme bars; reproduces the original cumulative running extrema for entry at bar 0 (verified) so the bar-0 baseline is faithful. For deferred entry (bar k>0) it is approximate on non-extreme bars (close proxies the bar's high/low), which slightly UNDER-detects intrabar SL/partial touches post-entry — conservative.
- **Exit:** SL-honest deployed `sl_partial_close_1r_runner_trail` @ 3.5×ATR (50% at +1R, runner trails 1R off rolling peak, 240-bar stop), always-on intra-bar SL matching the live driver. Units: ATR = pool-R×2; R_dep = ATR/3.5. Deterministic (seed 42; no RNG). Artefact SHAs in `manifest_sha256.txt`.

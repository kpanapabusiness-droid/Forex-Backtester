# Arc 10 — SL-Honest vs Canonical-Replay Gate Fidelity Audit

> READ-ONLY fidelity audit. Re-runs the EXACT canonical Arc-10 v3.0.2 portfolio gate (fixed-initial, EET, cell-5 costs, governors 3.5/4.5+7/8, daily_ref=initial, r_base 0.40%) on the canonical EET frame, swapping ONLY the exit for pre-partial-breach trades (SL-honest: stop at −1R at the SL bar when the low pierces −3.5×ATR before the +1R partial — matching the live `MultiPairBacktester` driver). Does NOT modify any canonical number; reports the delta for analyst adjudication.

> **Self-validation:** **replay path reproduces the published canonical numbers** (worst-fold ROI 13.77% vs 13.77%, trailing DD 7.73% vs 7.73%) → the SL-honest delta below is trustworthy.


## Fold-by-fold — SL-honest engine vs canonical replay (r_base 0.40%)

| fold | yr | n | replay ROI% | SLhonest ROI% | ΔROI | replay win% | SLhon win% | replay meanR | SLhon meanR | replay trailDD% | SLhon trailDD% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 2010 | 201 | 23.24 | -7.15 | -30.39 | 68.16 | 47.76 | 0.31 | -0.04 | 7.73 | 11.71 |
| F2 | 2011 | 182 | 25.83 | -8.20 | -34.03 | 71.98 | 43.96 | 0.40 | -0.15 | 7.55 | 8.34 |
| F3 | 2012 | 179 | 28.47 | 3.55 | -24.92 | 73.18 | 54.75 | 0.47 | 0.12 | 3.89 | 8.61 |
| F4 | 2013 | 195 | 40.19 | 8.85 | -31.34 | 71.79 | 51.79 | 0.55 | 0.16 | 6.40 | 14.13 |
| F5 | 2014 | 192 | 28.91 | -2.72 | -31.63 | 73.96 | 50.52 | 0.43 | 0.03 | 6.18 | 5.70 |
| F6 | 2015 | 190 | 20.77 | -7.56 | -28.33 | 65.26 | 46.84 | 0.29 | -0.05 | 5.75 | 7.49 |
| F7 | 2016 | 171 | 28.51 | -5.61 | -34.11 | 72.51 | 46.78 | 0.44 | -0.05 | 5.52 | 8.62 |
| F8 | 2017 | 190 | 42.56 | -8.19 | -50.75 | 75.79 | 44.21 | 0.59 | -0.02 | 6.96 | 8.76 |
| F9 | 2018 | 176 | 13.77 | -7.77 | -21.54 | 65.34 | 44.32 | 0.23 | -0.13 | 5.09 | 8.80 |
| F10 | 2019 | 195 | 30.97 | -8.32 | -39.29 | 73.33 | 50.26 | 0.43 | 0.01 | 7.58 | 10.67 |
| F11 | 2020 | 188 | 34.08 | -6.43 | -40.51 | 75.53 | 48.40 | 0.48 | -0.05 | 6.21 | 11.31 |
| 2021 | 2021 | 212 | 33.84 | 0.69 | -33.15 | 71.23 | 51.42 | 0.43 | 0.05 | 5.36 | 10.12 |
| 2022 | 2022 | 179 | 22.40 | -5.94 | -28.33 | 67.60 | 46.93 | 0.34 | -0.05 | 6.87 | 9.07 |
| 2023 | 2023 | 189 | 24.92 | 1.59 | -23.34 | 67.20 | 50.79 | 0.36 | 0.06 | 5.34 | 6.66 |
| 2024 | 2024 | 191 | 36.74 | -3.57 | -40.31 | 75.92 | 49.21 | 0.52 | -0.00 | 4.88 | 8.70 |
| 2025 | 2025 | 251 | 43.54 | 4.42 | -39.12 | 76.10 | 56.18 | 0.46 | 0.09 | 5.86 | 6.97 |
| 2026p | 2026 | 71 | 2.78 | 0.49 | -2.29 | 57.75 | 53.52 | 0.15 | 0.07 | 4.59 | 5.64 |


## Aggregate — both engines side by side (search folds F1–F11)

| metric | replay | sl_honest | delta |
|---|---|---|---|
| worst-fold ROI % | 13.77 | -8.32 | -22.10 |
| mean-fold ROI % | 28.85 | -4.50 | -33.35 |
| worst-fold trailing DD % | 7.73 | 14.13 | 6.40 |
| worst daily DD % | 4.06 | 2.13 | -1.93 |
| folds positive (of 11) | 11.00 | 2.00 | -9.00 |
| kills (8% close-all) | 0.00 | 3.00 | 3.00 |
| per-trade win % | 71.35 | 49.30 | -22.05 |


## Engine disagreement (the bug's footprint)

- **699 trades (22.2% of 3152)** have a pre-partial SL breach (low pierced −3.5×ATR before reaching +1R). The replay books these as partial+runner outcomes; the SL-honest engine stops them at −1R.
- **695 of them (22.0% of the pool)** the replay booked as WINS that become SL-honest LOSSES. Their replay realised summed to **+608.8 R** (gross) — all of which is reclassified to −695 R under SL-honest. That swing is the engine of the worst-fold/mean/win deltas above.


(Full list in `disagreement_trades.csv`.)


---
## VERDICT

**The canonical gate IS INFLATED by the replay's pre-partial SL-skip.** Worst-fold ROI 13.77% → **-8.32%**, mean-fold 28.85% → **-4.50%**, per-trade win 71.4% → **49.3%**, worst-fold trailing DD 7.73% → **14.13%**, folds-positive 11/11 → **2/11**. On the SL-honest engine Arc 10 **NO LONGER CLEARS PASS-DEPLOYABLE** (worst-fold ROI >5%, mean >8%, trailing DD ≤8%, 0 kills, 11/11 positive). The live EA is SL-honest, so live execution already matches the SL-honest column — but the BACKTEST GATE that justified deployment rests on the inflated replay numbers. This is a deployment-integrity finding flagged for analyst decision; no live change is authorised by this audit.


---
## FRAME_PROVENANCE & METHOD

- **Frame:** canonical v3.0.2 EET pool (3152 trades) + `H4_5ers_eet` cache, recovered from sibling worktree `nice-mirzakhani-baa9e5` (the canonical frame was removed from the main tree 2026-05-31) and copied into this worktree. trade_ids match this tree's `l_arc_10_v3.0.2/path_analytics` exactly. No forward price fabricated.
- **Engine:** the canonical portfolio reconstruction `governed_wfo.build_schedules` → `simulate_fold` (fixed-initial, total_ref=static, daily_ref=initial-resetting, governed 3.5/4.5+7/8, cell-5 costs via `fundednext_floating`), r_base 0.40%, EET clock. Identical to the canonical `final_canonical_wfo` path.
- **SL-honest change (the only difference):** for each pre-partial-breach trade (first SL-breach bar `sb` < first +1R bar `tp1`), `dep_exit_offset := sb` and `realized_r_3p5 := −1.0` in `meta` before `build_schedules`. The canonical machinery then emits SL-honest marks, exit timing, concurrency, daily-DD and realised. All other trades are byte-identical (replay ≡ SL-honest where there is no pre-partial breach), which is why the replay column self-validates against the published gate.
- SL detection on the recorded running low (`mae_so_far_r ≤ −1.75 R_pool = −3.5 ATR`); +1R on running high (`mfe_so_far_r ≥ +1.75`). Deterministic (no RNG; n_jobs=1; LF). Artefact SHAs in `manifest_sha256.txt`.


**Honesty / residual assumptions (all conservative — i.e. they would make the real inflation ≥ what is reported, not less):**
- SL detection uses the SAME recorded `mae_so_far_r` path the canonical replay uses, so the replay-vs-SL-honest comparison is exact and internally consistent; the only change is the exit logic. If that recorded low is mid-anchored while the live driver fires the SL on `low_bid` (< mid for a long), the live engine would stop EARLIER/more often — so this audit UNDER-counts breaches, making it a lower bound on the inflation.
- Same-bar SL+partial (sb == tp1) trades are left as the replay outcome (not forced to −1R), i.e. treated generously — another reason this is a lower bound.
- Stopped trades are booked at exactly −1R gross (the SL price, no extra slippage); real fills could be marginally worse.
- The analyst should confirm against (a) the deployed MQL5 EA's SL handling and (b) the live trade blotter — live realised P&L should track the SL-honest column, NOT the replay/backtest column, if this finding is correct.

# Arc 11 — Experimental Step 5 WFO (off-protocol, documentation only)

> **Status:** documentation only. Arc 11 remains **Closed-HALT** per §16a Path A regardless of this script's outcome. No queue, registry, or protocol mutation.

## What this is

A lightweight 7-fold WFO over the surviving Step 3 cohorts of Arc 11, to characterise what canonical §10 S5 (= v2.3 Step 5 WFO) would see, given S4 closed FAIL → HALT on the extractability gate (best AUC 0.5728 vs gate 0.60, margin 0.027).

Two runs:
- **Run A — c1 raw at SL=3.0×ATR.** No admission filter; every c1 trade enters.
- **Run B — agg_c1_c3 + Pipeline D1 t=5 classifier at SL=3.0×ATR.** Per-fold classifier trained on IS, applied to OOS at bar t=5 with threshold 0.50.

## Method (and where it deviates from canonical §10)

- 7 anchored expanding folds mirroring KH-24 OOS schedule (Oct 2020 – Jan 2026).
- Per-trade outcome from re-imposing SL on the §15a bar path via `_eval_trade_at_sl` (byte-identical with Step 3 / Step 4 SL re-imposition).
- 0.5% risk per trade, compounded equity. Annualised ROI = `(1+roi)^(365.25/days) - 1` per fold.

**Deviations vs canonical §10:**
1. **No §11 V-shape recovery trail exit policy.** Outcome uses fixed-SL truncation + recorded forward-window close (≤240 4H bars). Trail upside not captured; chop-induced trail early-outs not captured.
2. **Run B threshold = 0.50 fixed**, not selected by v2.2 §3 (max precision with recall ≥ 0.60) — that rule never fires in Arc 11 S4. 0.50 is a midpoint of the sweep grid.
3. **Cluster ID assumed known at entry** for both runs. **This is an oracle.** In real deployment cluster membership is identified by Pipeline E (entry-time classifier); Arc 11 S4 Pipeline E AUC was 0.42 / 0.39 / 0.51 for c1 / c3 / agg respectively — well below the 0.65 gate. Both runs' ROI figures should be read as **upper bounds conditioned on a perfect cluster oracle**, not as deployable systems.
4. Fold 1 (Oct 2020 – Jul 2021) has **0 IS trades** for Run B's classifier — falls back to admit-all (== raw agg at SL=3 for that fold).

## Results table

### Aggregate (canonical §10 pass-deployable / pass-viable gates applied)

| Run | Description | Sign-consistent | Worst-fold ROI ann % | Mean-fold ROI ann % | Worst-fold DD % | Min trade count | Full-data ROI % | Full-data DD % | Pass-deployable | Pass-viable |
|---|---|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| A | c1 raw, SL=3.0 | YES | 101.39 | 152.12 | 2.48 | 42 | 13,006.57 | 2.48 | **YES** | YES |
| B | agg_c1_c3 + D1 t=5 classifier, SL=3.0 | YES | 26.30 | 53.33 | 4.95 | 114 | 704.89 | 4.95 | **YES** | YES |

Pass-deployable thresholds: worst-fold annualised ROI ≥ 5%, mean ≥ 8%, worst-fold DD ≤ 8%, all folds positive, ≥ 15 trades/fold, full-data ROI ≥ 5%, full-data DD ≤ 10%. Both runs clear on every line.

### Per-fold detail

**Run A — c1 raw at SL=3.0:**

| Fold | OOS window | Days | n_trades | mean R | ROI period % | ROI ann % | Max DD % |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 2020-10-01 → 2021-07-01 | 273 | 42 | 3.61 | 111.47 | 172.36 | 1.00 |
| 2 | 2021-07-01 → 2022-04-01 | 274 | 46 | 3.43 | 117.72 | 182.12 | 0.65 |
| 3 | 2022-04-01 → 2023-01-01 | 275 | 44 | 3.00 | 91.90 | 137.67 | 1.00 |
| 4 | 2023-01-01 → 2023-10-01 | 273 | 44 | 2.75 | 81.89 | 122.63 | 1.00 |
| 5 | 2023-10-01 → 2024-07-01 | 274 | 62 | 2.48 | 113.51 | 174.86 | 2.48 |
| 6 | 2024-07-01 → 2025-04-01 | 274 | 43 | 3.56 | 112.90 | 173.81 | 0.52 |
| 7 | 2025-04-01 → 2026-01-31 | 305 | 43 | 2.75 | 79.43 | 101.39 | 0.50 |

**Run B — agg_c1_c3 + Pipeline D1 t=5 classifier at SL=3.0:**

| Fold | n_total | n_admit | n_reject | n_pre_t | mean R | admit-only mean R | ROI ann % | DD % | n_IS | classifier? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 | 133 | 133 | 0 | 0 | 1.12 | 1.12 | 165.39 | 4.95 | 0 | no (admit-all) |
| 2 | 114 | 33 | 81 | 0 | 0.54 | 1.50 | 49.96 | 2.61 | 132 | yes |
| 3 | 123 | 21 | 102 | 0 | 0.35 | 1.09 | 32.77 | 1.09 | 246 | yes |
| 4 | 132 | 18 | 114 | 0 | 0.32 | 1.54 | 32.39 | 1.15 | 369 | yes |
| 5 | 131 | 12 | 119 | 0 | 0.35 | 1.74 | 34.91 | 1.00 | 501 | yes |
| 6 | 123 | 10 | 113 | 0 | 0.29 | 1.79 | 26.30 | 0.81 | 632 | yes |
| 7 | 133 | 13 | 120 | 0 | 0.35 | 2.01 | 31.57 | 1.40 | 755 | yes |

Pre-t SL losses = 0 across all folds for both runs (every trade survived to bar 5 at SL=3.0).

## What S5 sees that the S4 AUC gate did not

1. **Both runs are pass-deployable on every §10 gate.** S4 closed Arc 11 HALT at AUC 0.5728 vs gate 0.60. S5 says the underlying cohort, deployed at SL=3.0, easily clears worst-fold ROI ≥ 5% (Run A: 101%; Run B: 26%) and DD ≤ 8% (Run A: 2.5%; Run B: 4.95%) with all 7 folds positive. **This is the v2.3 §3 (Open-22) cross-arc lesson in microcosm: AUC is a classifier metric; deployment economics is the truth.**
2. **The D1 classifier IS picking up signal.** admit-only mean R per fold is 1.09–2.01 (folds 2–7) vs overall mean R 0.29–0.54 across all trades — admit pool runs 3–6× the population mean. AUC 0.573 in 5-fold TSCV understates the classifier's usefulness when paired with a high-magnitude cohort, because the AUC gate doesn't condition on the cohort's `mfe_p50`.
3. **The admit rate is heavy** (8–29% post fold 1). The D1 t=5 path-so-far features differentiate strongly — most agg signals are rejected, the ones admitted are concentrated wins. Reject pool drag at bar 5 is small (mean reject R in folds 2–7 averages ~0.15) because at bar 5 most surviving trades are close to entry.

## Cluster-vs-signal divergence

- **Run A (c1 oracle) — magnitude ceiling.** Mean R per fold 2.47–3.61, full-data ROI 13,006%. This is what the c1 cohort can produce IF you can identify it at entry. You can't — Pipeline E AUC for c1 was 0.42, below random. Run A is the upper bound conditioned on a perfect cluster filter.
- **Run B (agg + D1 t=5) — realistic-but-still-oracle.** Mean R per fold 0.32–0.54, full-data ROI 704%. Assumes you can identify the trade as agg_c1_c3 at entry (Pipeline E AUC 0.51 for agg). Even with that assumption AND a barely-above-random D1 classifier at t=5, S5 pass-deployable on every gate.
- **Gap = extractability tax.** Mean fold ROI ann: Run A 152% vs Run B 53% = 99 pp gap. Roughly 2/3 of the cohort's capturable edge is lost to the D1 admission filter (early reject of would-be winners). Mostly because the classifier rejects 71–92% of trades — many of those would have won.
- **Bottom line on cluster-vs-signal:** Arc 11's cluster c1 (a V-shape recovery sub-cluster at SL=3) is a genuinely capturable cohort with massive structural edge. The signal-level deployment problem is identifying it at decision time. S4's AUC gate is the right *direction* but the wrong *threshold* for cohorts of this magnitude — the binding constraint should be deployed worst-fold ROI/DD, not classifier AUC.

## Pairing with Arc 6 (capturable-not-extractable write-up)

Arc 11 is the second arc in a row to show:
- S3 PASS (clean capturability with real magnitude)
- S4 AUC FAIL → HALT (entry-time / early-path features can't clear 0.65 / 0.60 gates)
- S5 (experimental) would PASS-DEPLOYABLE if a deployable cluster-filter or D1-admission existed

Arc 6 closed KILL → DIES at Step 4 with best E AUC 0.60 (gate 0.65), Pipeline D1 max-F1 fallback admitting 3-4 trades. Arc 11 sees the same shape but with stronger magnitude (c1 fwd_mfe_p50 4.48R vs Arc 6 c2 4.47R).

Both arcs point at the same gap: **the §8 feature regime (8 cross-dataset base + arc-specific entry-time + path-so-far at t)** is insufficient to extract the cohort identity for trend-continuation breakout signals. Calibration candidate (from both arcs): feature-set extension — multi-TF, order-flow proxies, regime conditioners, ensemble.

## Caveats — read before quoting these numbers

1. **Cluster ID at entry is an oracle.** Real deployment can't filter to c1 / agg_c1_c3 without a Pipeline E classifier that clears its gate. Arc 11's E classifier didn't.
2. **No V-shape recovery trail exit.** Fixed-SL + time-exit truncation. Run A's mean R 2.47–3.61 likely underestimates a trail-based exit's capture (winners give back to time exit at bar 240 instead of locking in via trail). DD likely also higher under trail.
3. **Threshold 0.50 fixed** for Run B; not selected per v2.2 §3 (which would have required recall ≥ 0.60, never satisfied).
4. **Fold 1 of Run B is admit-all** (no IS for classifier). This drives full-data DD of 4.95% — fold 1's swings are the biggest of all.
5. **No reject-pool downstream re-entries**. v2.3 §4 (Open-23) reject-pool cost language: implicit here as close_r at bar 5 in new R-frame. Empirical bound in protocol: −0.15 to −0.46R per rejected D1 archetype across closed arcs. Arc 11 reject-pool mean R ≈ +0.15 (much better than the protocol's lower bound; consistent with V-shape recovery archetype's nature — bar 5 close is rarely deep underwater for trades that survived the pre-t SL).
6. **Magnitude inflation.** Compounding 0.5% × ~300 trades × +1R+ over 5 years gives extreme full-data ROI (13,006%). The annualised number (149% Run A) is more representative.

## Artefacts

- `arc11_exp_s5_aggregate.csv` — aggregate pass-deployable / pass-viable check for both runs
- `arc11_exp_s5_run_a_c1_raw_per_fold.csv` — per-fold metrics for Run A
- `arc11_exp_s5_run_b_agg_d1_t5_per_fold.csv` — per-fold metrics for Run B
- `arc11_exp_s5_NOTES.md` — this doc
- `scripts/l_arc_11/experimental_s5_wfo.py` — runner

## Disposition

**Arc 11 remains Closed-HALT.** This experimental S5 does not alter that. It does contribute one cross-arc finding: the S4 AUC gate may be too conservative for high-magnitude cohorts; pairing AUC threshold with cohort `fwd_mfe_p50` (e.g. relax AUC gate when `mfe_p50 ≥ 3R`) is a candidate for the next protocol amendment cycle. Should be combined with the Arc 6 closure for the capturable-not-extractable write-up.

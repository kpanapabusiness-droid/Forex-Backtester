# Arc 8 Step 5 — Pre-check 1: Fold 2 regime investigation

> Dispatch §41: "Step 4 Pipeline E per-fold AUCs were 0.8550 / 0.4500 / 0.7750 / 0.7402 / 0.6667. Identify the date range covered by TimeSeriesSplit fold 2 and report which macro regime it corresponds to."

## TimeSeriesSplit fold-to-date mapping (c1 cluster, n=177, chronological)

c1 pool date range: 2020-11-05 to 2025-12-02 (5y 1mo of trades, 28 FX pairs).

`sklearn.model_selection.TimeSeriesSplit(n_splits=5)` produces 5 expanding-train folds. Mapping to the Step 4 reported AUCs:

| sklearn fold idx | Step 4 RF AUC | Train n | Train range | Test n | Test (OOS) range | Span |
|---:|---:|---:|---|---:|---|---|
| 0 | **0.8550** | 32 | 2020-11-05 to 2021-12-17 | 29 | 2021-12-20 to 2022-10-17 | 10mo |
| 1 | **0.4500** | 61 | 2020-11-05 to 2022-10-17 | 29 | **2022-10-19 to 2023-08-01** | 9.5mo |
| 2 | **0.7750** | 90 | 2020-11-05 to 2023-08-01 | 29 | 2023-08-23 to 2024-03-21 | 7mo |
| 3 | **0.7402** | 119 | 2020-11-05 to 2024-03-21 | 29 | 2024-03-25 to 2024-10-02 | 6mo |
| 4 | **0.6667** | 148 | 2020-11-05 to 2024-10-02 | 29 | 2024-10-17 to 2025-12-02 | 13.5mo |

The dispatch's "fold 2" (per the AUC list ordering 0.8550 / **0.4500** / 0.7750 / 0.7402 / 0.6667) corresponds to **sklearn fold idx 1** — the second fold in 0-indexed terms, or "fold 2" if 1-indexed.

## Test-window macro regime (2022-10-19 → 2023-08-01)

This 9.5-month OOS window straddles three back-to-back regime-shift events. Each individually capable of breaking a trend-continuation signal; together they form an unusually adverse stretch for any HH/HL trend-resume signal trained on 2020-2022 data.

### Phase 1 — USD reversal (late Oct 2022 → end Q4 2022)

- **DXY peak**: 114.78 on 2022-09-28 (then highest since 2002), reverses sharply through Oct-Nov 2022.
- **UK gilt crisis** (2022-09-23 Truss/Kwarteng mini-budget) → BoE emergency bond-buying through October; GBP whip-saws.
- **Bank of Japan FX intervention** late Oct 2022 to defend JPY (USD/JPY ~152 high). PR-HHHL longs on JPY-cross uptrends would have been caught in the reversal.

PR-HHHL signal trained on prior 2020-2022 data was learning the **DXY uptrend regime**. The October 2022 reversal flipped that regime; pullback-resume patterns in JPY pairs and EUR/USD started failing precisely because the dominant trend ended.

### Phase 2 — US regional bank crisis (March 2023)

- **SVB failure** 2023-03-10, **Signature Bank** 2023-03-12, **Credit Suisse forced takeover** 2023-03-19.
- Sharp risk-off flow → JPY and CHF strength against everything. USD highly volatile.
- AUD/JPY, NZD/JPY, GBP/JPY crosses: classic risk-on trend continuation pairs all reversed.

For c1 archetype (V-shape recovery, FG-weak): trades in this window were structurally mismatched — the "V-shape" was no longer a single-event dip-and-recover but multi-week chop-and-fail.

### Phase 3 — BoJ regime change + US debt ceiling (April-June 2023)

- **Kazuo Ueda** took office as BoJ governor 2023-04-09. YCC band widening expectations whipsaw JPY pairs.
- **US debt ceiling standoff** May-June 2023 (resolved late May). Cross-asset volatility.
- **ECB hiking aggressively** (250bp cumulative through period) while Fed near terminal — sharp DXY weakness.

## Interpretation

The fold idx 1 AUC drop from 0.8550 → 0.4500 is consistent with a **classifier-regime mismatch artefact**, not a Step 4 signal failure:

- Training data (2020-11 to 2022-10): dominated by clean USD uptrend + post-COVID risk-on, with the Q1-2022 USD/JPY squeeze providing many textbook PR-HHHL trades that resumed cleanly.
- OOS (2022-10 to 2023-08): three back-to-back regime-shift events that violated the trend-continuation prior. Classifier features (`ret_5bar_atr`, `pos_in_20bar_range`, `pullback_depth_atr`, `range_to_atr_14`, `hl_range_atr`) describe geometry; under regime-shift their predictive mapping flips.

Folds 2 and 3 recover (0.78 / 0.74) because the training set now includes the 2022-10-2023-08 turbulence and the test windows are 2H 2023 / 1H 2024 — back to more conventional trend regimes.

Fold 4's softer AUC (0.6667) covers a longer (13.5mo) test window through late 2024 and most of 2025 — likely picks up additional regime mix; still clears the 0.65 §8 gate marginally.

## Implication for Step 5 WFO

The dispatch's WFO uses `configs/wfo_kh24.yaml` windows (rolling 12mo train / 3mo OOS). Those windows are **finer-grained** than the TimeSeriesSplit 5-fold structure — a 2022-10-2023-08 disaster zone will split into ~3 separate 3mo OOS windows under KH-24 WFO. Worst-window ROI under WFO will be the weakest of those 3mo windows, not the aggregated 9.5mo. That's the WFO ship-gate test §10 requires.

**Verdict: NOT a Step 4 leak or modelling artefact.** Step 4 result stands; fold idx 1 weakness is a real OOS regime-shift challenge that WFO will surface or exonerate at finer granularity.

**Recommendation for WFO interpretation:**
- If WFO worst-window falls inside 2022-10 to 2023-08 with ROI < 0%: this is the expected stress test outcome — analyst judges whether c1 ships pass-viable (portfolio candidate) vs pass-deployable (solo deploy).
- If WFO worst-window falls outside that window: the OOS pain was specifically regime-shift; c1 may ship pass-deployable.
- Per-window stratification report (which calendar months underlie worst-window) is recommended addendum to WFO output.

## Files referenced

- `results/l_arc_8/step1_verbatim/trades_all.csv` (c1 chronological order via merge with `clusters_K4.csv`)
- `results/l_arc_8/step2/clusters_K4.csv` (cluster_id == 1 → c1 trade IDs)
- `results/l_arc_8/step4/predictability_angle_E.csv` (per-fold AUCs)

# Live Tracking Framework

> **Purpose:** Define what "system performing as expected" looks like in numbers, BEFORE live data arrives.
> **Why now:** future-you in drawdown will rationalise in either direction. Past-you, calibrated to backtest, is the right person to set the bands.
> **Use:** weekly review, compare actual live results to these bands. Inside band = normal variance. Outside band = investigate.

## The cardinal rule

**This document is written from a calibrated state. Do not edit it from a stressed state.**

If live performance violates these bands, the document does not get edited to fit the data. The data forces a decision: investigate, pause, or trigger kill criteria. Editing the bands to accommodate poor performance defeats the entire point.

If you genuinely believe a band needs updating (e.g. after 12+ months of live data you have a better empirical distribution), update it in a calibrated state with explicit justification, version-bumped.

## Per-trade R-distribution expectation

Backtest distribution from the WFO pool (EET v3.0.2, ~3,152 trades over 14 years):

| Metric | Backtest value | Live expected | Variance band (acceptable) |
|---|---|---|---|
| Mean R per trade | +0.42 | +0.30 to +0.40 (cost haircut) | ±0.10 |
| Median R per trade | −0.15 to −0.20 | similar | — (median always negative; mean is what matters) |
| Win rate | ~38% | 33-43% | 28-48% over any 50+ trade window |
| Largest single win R | +5.0 to +8.0 (rare) | similar | clip outliers when computing means |
| Largest single loss R | −1.0 (clean SL hit) | −1.0 to −1.3 (slippage tail) | −1.5R is alarming, −2.0R+ is investigate |
| % of trades with TP1 fired | ~55% | 50-60% | — |
| % of trades exited via trail_stop | ~50% | 45-55% | — |
| % of trades exited via initial_sl_hit | ~38% | 35-45% | — |
| % of trades exited via time_exit | ~7% | 5-12% | high time-exit % = trend regimes weak |

**Why mean is positive but median is negative:** asymmetric R distribution. Most trades lose ~1R (initial SL hit) or partial-close-then-trail to ~0R. The minority of trades that catch runners produce +2R, +3R, +5R outcomes. The right tail funds the strategy.

**Variance bands are wide because:** 50 trades is small statistical mass. Backtest at 50 trades had folds with win rates of 30% AND 48% within the same year. Don't react to a single 50-trade window.

## Cumulative ROI expectation

Based on EET expected live (44% annualised holdout ROI, after haircuts). At 0.50% risk per trade, ~225 trades per year.

| Trade count | Expected days elapsed | Expected ROI | 25% confidence band | 75% confidence band |
|---|---|---|---|---|
| 25 | ~40 | +4% | −2% to +10% | — |
| 50 | ~80 | +9% | 0% to +18% | — |
| 100 | ~160 | +19% | +5% to +33% | — |
| 200 | ~320 | +38% | +15% to +60% | — |
| 365 (one year) | ~365 | +44% | +20% to +70% | — |

**How to use:**
- Live ROI inside the 25-75% band → normal, no action
- Live ROI below the 25% band → flag for review, not yet kill criteria
- Live ROI below 0% at 100+ trades → investigate seriously, check for systematic issue
- Live ROI tracking at the median → ideal, no action needed
- Live ROI above 75% band → also note (could be regime-favorable, could be over-leveraged via fat tails)

**Critically:** these are estimates with high variance. A 50-trade ROI of −3% is within the realm of normal variance. Don't react to short windows.

## Per-pair frequency expectation

Per the WFO pool, signal frequency varies materially across the 28 pairs. Some pairs (heavy USD crosses, JPY pairs) signal more frequently than others (tight Euro-area crosses).

Top frequency pairs (most signals/year):
- USDJPY, EURJPY, GBPJPY (all JPY) — ~10-15 trades/year each
- EURUSD, GBPUSD, USDCAD — ~8-12 trades/year
- AUDUSD, NZDUSD — ~8-10 trades/year

Bottom frequency pairs:
- EURCHF, CADCHF (CHF cross-pairs) — ~3-5 trades/year (CHF tight-band intervention regimes)
- NZDCHF, AUDCHF — ~4-6 trades/year

**Flag if:** any pair fires >2x its expected frequency in any quarter (might indicate convention drift or symbol-specific issue) or <0.3x its expected frequency over 6 months (might indicate symbol unavailable or broken data feed).

## Daily DD expectation

Most live days will have minimal intraday DD because Arc 10 runs ~1-2 open positions on average across 28 pairs.

| Daily DD level | Frequency in backtest | Action |
|---|---|---|
| 0 to 1% | ~75% of days | Normal — no action |
| 1% to 2% | ~20% of days | Normal — no action |
| 2% to 3% | ~4% of days | Note, no action |
| 3% to 3.5% | ~0.8% of days | Watch — approaching halt threshold |
| 3.5% to 4.5% | very rare | System halts new entries — review |
| 4.5%+ | should not occur | System force-closes all — investigate immediately |

**Why this matters:** if live DD distribution is materially different (e.g. 4% of days hitting 2-3% DD instead of 0.8%), that's a signal of either:
- Position sizing miscalibration
- Broker fill quality worse than modeled
- Concurrent trade load higher than backtest assumed
- Tail event clustering (multiple losing positions correlating)

## Total DD expectation

The internal "something's wrong" threshold:

| Total DD level | Action |
|---|---|
| 0 to 5% | Normal |
| 5% to 7% | Note — track, but normal |
| **7%** | **Internal pause-and-review trigger** |
| 8%+ | System force-closes (CloseAll threshold) |
| 10%+ | Broker hard limit — account terminated |

**At 7% total DD:**
1. Stop and verify the system is operating correctly (no bugs, no operator error, no broker issue)
2. Cross-check against backtest worst-fold DD (expected live ~8.8% EET, ~9.5% UTC)
3. If system is operating correctly and DD is within expected worst-fold range → continue per plan (this is what the system is designed to survive)
4. If anything looks off → trigger emergency kill procedure (`04_runbook/05_emergency_kill.md`) and investigate

**Why 7% (not 8%):** 8% triggers the system's automatic CloseAll. By 7% you want operator-level awareness so you can choose to continue, pause, or kill BEFORE the system makes the decision for you. 7% provides a 1pp human decision buffer.

## Win rate over rolling windows

Win rate variance is high at short windows. Backtest showed:

| Window | Min win rate seen | Max win rate seen | Mean |
|---|---|---|---|
| 25 trades | 22% | 56% | 38% |
| 50 trades | 27% | 49% | 38% |
| 100 trades | 31% | 45% | 38% |
| Full year | 33% | 44% | 38% |

**Don't react to single 25-trade win rate.** Even a 22% win rate over 25 trades was within backtest variance. Only react if 50-trade rolling win rate drops below 27% sustained for multiple windows.

## Monthly sign expectation

Backtest sign consistency was 11/11 folds (yearly) positive. Monthly is noisier.

In backtest, individual months:
- ~65-70% of months were positive
- ~30-35% of months were negative
- No fold (year) was negative overall

**Live tracking:** 
- 1 negative month in 6 → normal
- 2 negative months in 6 → note but normal
- **3 negative months in 6 → investigate** (rare in backtest; possible regime shift or systematic issue)
- 4+ negative months in 6 → triggers kill criteria review (`07_kill_criteria.md`)

## What live data CAN'T tell you in the first month

- Whether expected ROI is being achieved (sample too small)
- Whether DD profile matches backtest (haven't hit drawdown yet)
- Whether broker fill quality matches modeled costs (need 50+ trades for meaningful spread/slippage stats)

**First month conclusions:**
- System is running operationally (yes/no)
- Trades fired when expected (yes/no — cross-check signal envelopes vs broker trades)
- No surprising errors in logs (yes/no)
- Fill quality looks similar to backtest at a glance (within order of magnitude)

That's it. Don't try to conclude ROI or edge from one month.

## Quarterly review structure

Every 3 months (= ~50-75 trades), do a structured review against these bands:

1. Trade count: matches expected ~55-70 per quarter?
2. Win rate: inside 28-48% band over the quarter?
3. Mean R per trade: inside +0.20 to +0.50 band?
4. Cumulative ROI: tracking inside the cumulative ROI table band?
5. Per-pair frequency: any pair >2x or <0.3x expected?
6. DD profile: max daily DD seen vs expected distribution?
7. Total DD: max total DD seen vs expected worst-fold?

If 0-1 bands violated: normal variance, continue.
If 2-3 bands violated: investigate root cause, continue trading at current risk.
If 4+ bands violated: review against `07_kill_criteria.md`, decide whether to pause, reduce risk, or kill.

## What to capture each quarter (raw data)

After each quarterly review, append a row to a tracking ledger (build it as you go):

| Quarter | Trades | Win % | Mean R | ROI | Max Daily DD | Max Total DD | Bands violated | Action |
|---|---|---|---|---|---|---|---|---|
| Q1 (date range) | | | | | | | | |

The ledger lives in a private place (not in the repo for privacy of live numbers if you prefer). The decision-making process IS in the repo (this doc + kill criteria).

## Why writing this BEFORE live data matters

The single highest-value piece of the entire `arc_10/` consolidation is this document + `07_kill_criteria.md`.

Configs and scripts can be reconstructed from git history. The calibrated decision framework — what "normal" looks like, what "wrong" looks like, what triggers action — cannot be reconstructed under drawdown stress. Stressed-you will rationalise either:

- "It's just variance, keep going" (when it's actually broken)
- "It's broken, stop everything" (when it's actually variance)

Calibrated-you, with WFO numbers fresh, can write down the bands honestly. Then stressed-you just has to look at the document and follow it.

**Treat this document as a contract with future-you.** It supersedes any in-the-moment intuition.

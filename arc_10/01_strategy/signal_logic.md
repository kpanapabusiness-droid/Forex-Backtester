# Signal Logic — Arc 10 DLR

> **Audience:** Someone who needs to understand what triggers a trade and why.
> **No MQL/Python code here.** Pure conceptual explanation. For code, see `signals/lchar_dlr_long.py`.

## What the signal looks for

Arc 10 trades **swing-low rejections** on the H4 timeframe with daily-bar structural context. The setup type is "v-shape recovery" — price has formed a clear swing low (L1) below a prior pivot (L0), then rejected upward off that L1 with structural support from D1 context.

**In plain English:** the system waits for a clean two-bar swing pattern where price dipped, formed a low (L1), then immediately rejected upward to close strongly above the dip. That bar — the "rejection bar" — is the signal bar. The trade enters long at the next bar open.

## The five gates a bar must pass

For a signal to fire on bar T (H4 close), all five conditions must hold simultaneously.

### Gate 1: Swing structure (L1 below L0)

A pivot-low L1 must exist within the trailing H4 window, sitting structurally below a prior pivot-low L0. "Structurally below" means L1's low is lower than L0's low by at least a configured proximity margin (in units of ATR).

This is the "pullback after pullback" pattern — the system trades second-leg recoveries, not first-attempt bounces.

### Gate 2: L1-to-ATR proximity

The distance from L1 to the current bar's price (in ATR units) must fall within a defined band. Too close to L1 → the rejection hasn't traveled far enough yet, the move hasn't developed. Too far → momentum has run away, we're chasing.

### Gate 3: Reject buffer

There must be a measurable upward rejection from L1 — the current bar's close must be above L1 by at least a configured buffer (in ATR units). This filters out signals where price merely paused at L1 instead of rejecting away from it.

### Gate 4: Upper fraction

The signal bar itself must close in its upper N% of its own range (configured upper_fraction, e.g. 0.55 means top 55%). This ensures the H4 closed strongly, not weakly. A doji or weak close → no signal.

### Gate 5: D1 trend context

The D1 (daily) chart must show alignment with the long direction. Specifically, recent D1 closes must hold above a configured D1-derived support level (one-bar-lagged to avoid lookahead).

When all five hold, the signal fires on bar T's close. The trade enters at bar T+1's open.

## What the signal does NOT use

- **No indicators** in the traditional NNFX sense. No MA crosses, no RSI, no MACD, no momentum filters. The signal is structural, not indicator-based.
- **No volume.** Volume was a veto-only feature in earlier work; Arc 10 doesn't use it (microstructure metrics don't port across brokers).
- **No fundamental data.** News filter blocks entries near scheduled high-impact news but doesn't generate signals.
- **No machine learning at decision time.** All gates are explicit thresholds. ML was used in the discovery phase (clustering trade archetypes) but the deployed signal is a deterministic rule set.

## Why long-only

Two reasons:

1. Empirical: The DLR pattern is genuinely asymmetric — sell-side swing-highs don't cluster the same way swing-lows do across the 28-pair universe. Sell signals exist in theory but underperform the buy side significantly.

2. Operational: Long-only halves the implementation complexity and the cost surface (no separate short-side calibration). Worth the constraint.

A short-side companion strategy is theoretically interesting but not pursued in Arc 10. Future arc territory.

## Why H4 timeframe

Tested against H1, H2, H4, D1 in earlier research (KGL/KH arc). H4 produced the best worst-fold ratio under WFO. Lower timeframes have higher signal counts but worse per-trade R-multiples and higher transaction costs as a percentage of edge. D1 has clean structure but signal frequency too low for meaningful WFO statistics.

H4 is the sweet spot.

## Why these 28 pairs

The standard 28-pair major + cross universe:
- AUD, CAD, CHF, EUR, GBP, JPY, NZD, USD — all crosses among these 8 currencies
- Excludes minor crosses (e.g. NOK, SEK, ZAR) due to broker support and liquidity concerns

These 28 pairs are the broadest correlated-but-distinct universe that all major prop firm brokers offer. Trading the full set captures diversification (~3,000 trades over 14 years).

## What the signal does NOT guarantee

- Profitability on any single trade. Many trades stop out at SL.
- Profitability on any single fold or month. Sign consistency was 11/11 folds positive in backtest, but individual months can be negative.
- Edge in markets fundamentally different from 2010-2024 (e.g. all-out central bank intervention regime, broker-feed corruption, etc.). The system assumes broadly continuous market microstructure.

## What the signal DOES guarantee (mechanically)

- Deterministic output for any given input. Same H4/D1 panels → same signal decision. No randomness, no model drift.
- Byte-identical signal output between the live sidecar and the historical lab (proven by Phase 2 parity).
- Bounded loss per trade (SL is the hard cap at signal-time ATR distance).
- Bounded daily/total drawdown (equity guard halts new entries at threshold).

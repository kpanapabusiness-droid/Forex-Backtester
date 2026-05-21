# signal_spec_pullback_resume_hhhl_long_v0.1

> Standalone signal spec. Authored by analyst, referenced by `results/ARC_QUEUE.md` Arc 8. Read by CC orchestrator at arc-open per dispatch reading list.

## Identification

| Field | Value |
|---|---|
| Name | `signal_pullback_resume_hhhl_long_v0.1` |
| Family | Trend continuation (structural) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | n/a (single-TF) |
| Pair set | 28 FX (KH-24 set) |
| Hypothesis | Trend strength + pullback geometry + trigger-bar geometry are entry-time observable → Pipeline E should clear 0.65 AUC |

## Trigger (locked at arc open — L_ARC_PROTOCOL §1.8)

**Swing definitions (3-bar local extreme):**
- Swing-high at bar k iff `high[k] > max(high[k-3..k-1])` AND `high[k] > max(high[k+1..k+3])`
- Swing-low at bar k iff `low[k] < min(low[k-3..k-1])` AND `low[k] < min(low[k+1..k+3])`

**1. Trend established in window `t-30..t-1`:**
- Identify all swing-highs in window; require ≥ 2; require sequence strictly ascending (HH structure)
- Identify all swing-lows in window; require ≥ 2; require sequence strictly ascending (HL structure)
- Right-edge constraint: most recent identifiable swing-high at most bar `t-4`; same for swing-lows

**2. Pullback:**
- `close[t-1] ≤ most_recent_swing_high − 0.5 × ATR(14)[t-1]`

**3. Resume trigger at bar t:**
- `close[t] > open[t]` (bullish close)
- `close[t] > high[t-1]` (breaks prior bar high)
- `(close[t] − low[t]) / (high[t] − low[t]) ≥ 0.5` (close in upper half)

**4. Spacing & entry:**
- ≥ 20 bars since last signal on this pair
- Entry: bar t+1 open per `docs/SPREAD_SEMANTICS_LOCK.md` round-trip convention

## Configuration

| Field | Value |
|---|---|
| Initial SL (Step 1 sim) | `entry − 2.0 × ATR(14)_4H[t]` |
| SL sweep at Step 3 | Default `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0} × ATR_4H` |
| Forward window | 240 bars (4H) |
| Exposure cap | Max 1 open position per pair |
| Risk per trade | 0.5% × reset floor balance |
| Spread | Real per-bar MT5 bid/ask; `configs/spread_floors_5ers.yaml` fallback only when raw = 0 |
| Data window | 2020-10-01 → 2026-01-31 |
| Arc config target | `configs/wfo_l_arc_8.yaml` |
| ATR period | 14 |

## Pool-size prior

Estimate 2,500–4,000 trades / 5y / 28 pairs. If Step 1 returns < 500, arc dies on §5 floor per §16a. Fallback relaxation (≥ 1 HH and ≥ 1 HL) is a future-arc design candidate, NOT a within-arc adjustment.

## Step 1 right-edge swing audit (mandatory)

Both swing-high and swing-low identification use k+1..k+3 lookahead within the detection window only. Confirm at Step 1 that all swings used in trigger evaluation are at most bar `t-4` in any selected signal. If standard 5/5 lookahead spot-check shows any future-bar dependency in trigger-evaluation fields, halt — engine-touching, do not patch.

## Step 1 co-fire matrix (mandatory)

Report co-fire %:
- **KH-24** (`kb_exhaustion_bar`): bearish single-bar exhaustion vs PR-HHHL bullish resume — independence expected. If > 10%, flag.
- **Arc 9** (`signal_inside_bar_break_trend_long_v0.1`) if Step 1 landed: both bullish-close trend-continuation triggers; some overlap expected. Open-05 portfolio note.
- **Arc 10** (`signal_d1_swing_low_rejection_long_v0.1`) if Step 1 landed: expected low (different anchor TF). Open-05 note.
- **Arc 11** (`signal_swing_high_breakout_trend_long_v0.1`) if Step 1 landed: **expected 15-30%** — pullback-resume that clears a prior swing-high fires both. Quantify cluster-by-cluster overlap at Step 3 if feasible.

## Hypothesis notes (informational, not gating)

PR-HHHL is the pool-size anchor of the Arc 8-11 parallel batch. Highest expected trade count, broadest feature coverage. If E fails here, informs whether entry-time-features hypothesis fails on trend-continuation signals broadly.

Path-shape expectation: Stepwise climber on clean pullback-resume; whipsaw on failed resume. Cluster heterogeneity at Step 2 is expected.

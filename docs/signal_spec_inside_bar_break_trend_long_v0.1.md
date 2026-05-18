# signal_spec_inside_bar_break_trend_long_v0.1

> Standalone signal spec. Authored by analyst, referenced by `results/ARC_QUEUE.md` Arc 9. Read by CC orchestrator at arc-open per dispatch reading list.

## Identification

| Field | Value |
|---|---|
| Name | `signal_inside_bar_break_trend_long_v0.1` |
| Family | Trend continuation (compression-and-break) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | n/a (single-TF) |
| Pair set | 28 FX (KH-24 set) |
| Hypothesis | Compression geometry (parent-bar range, inside-bar range, range ratio) + break-bar strength are entry-time observable → Pipeline E should clear 0.65 AUC |

## Trigger (locked at arc open — L_ARC_PROTOCOL §1.8)

**Swing-low definition (3-bar local low):**
- Swing-low at bar k iff `low[k] < min(low[k-3..k-1])` AND `low[k] < min(low[k+1..k+3])`

**1. Trend filter (pure structural, no MA):**
- Identify all swing-lows in window `t-30..t-1`
- Right-edge constraint: most recent identifiable swing-low at most bar `t-4`
- Require ≥ 1 swing-low exists in window
- Require `close[t-1] > min(swing_lows in window)` — price holds above the recent swing-low chain

**2. Inside bar at t-1 (strict nest in mother bar t-2):**
- `high[t-1] < high[t-2]`
- `low[t-1] > low[t-2]`

**3. Break trigger at bar t:**
- `close[t] > high[t-1]` (breaks inside-bar high)
- `close[t] > open[t]` (bullish close)

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
| Arc config target | `configs/wfo_l_arc_9.yaml` |
| ATR period | 14 |

## Pool-size prior

Estimate 1,500–2,500 trades / 5y / 28 pairs. Inside-bar events are rarer than pullbacks. If Step 1 returns < 500, arc dies on §5 floor per §16a. Relaxation candidates (drop bullish-close requirement, drop spacing rule) are future-arc design candidates, NOT within-arc adjustments.

## Smaller-pool risk

With 1,500–2,500 trades, individual clusters at Step 3 may not clear `size_fraction ≥ 0.10` (requires ~150–250 trades per cluster). §16a handles disposition: size_fraction failure near 0.10 with strong magnitude (`fwd_mfe_p50 ≥ 3.0R`) triggers HALT Path B; otherwise KILL.

## Step 1 right-edge swing audit (mandatory)

Swing-low identification uses k+1..k+3 lookahead within the detection window only. Confirm at Step 1 that all swing-lows used for trend filter are at most bar `t-4` in any selected signal. If standard 5/5 lookahead spot-check shows any future-bar dependency, halt — engine-touching, do not patch.

## Step 1 co-fire matrix (mandatory)

Report co-fire %:
- **KH-24** (`kb_exhaustion_bar`): bearish single-bar exhaustion vs IB-trend bullish break — independence expected. If > 10%, flag.
- **Arc 8** (`signal_pullback_resume_hhhl_long_v0.1`) if Step 1 landed: both bullish-close trend-continuation triggers. Some overlap expected. Open-05 note.
- **Arc 10** (`signal_d1_swing_low_rejection_long_v0.1`) if Step 1 landed: expected low (different anchor TF, different trigger geometry). Open-05 note.
- **Arc 11** (`signal_swing_high_breakout_trend_long_v0.1`) if Step 1 landed: modest overlap possible (IB-break that clears prior swing-high). Open-05 note.

## Hypothesis notes (informational, not gating)

IB-trend is the narrowest-feature-class candidate of the Arc 8-11 batch. If E clears here but fails Arc 8 (broader features), that argues compression-specific geometry carries disproportionate predictive signal.

Path-shape expectation: Stepwise climber or early-peak hold family on clean breaks; whipsaw on failed breaks.

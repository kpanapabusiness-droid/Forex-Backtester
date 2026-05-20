# signal_spec_three_bar_reversal_trend_long_v0.1

> Standalone signal spec. Authored by analyst, referenced by `results/ARC_QUEUE.md` Arc 12.

## Identification

| Field | Value |
|---|---|
| Name | `signal_three_bar_reversal_trend_long_v0.1` |
| Family | Trend continuation (multi-bar sequence) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | n/a (single-TF) |
| Pair set | 28 FX (KH-24 set) |
| Hypothesis | 3-bar sequence geometry (pullback bar → recovery bar → break bar) carries entry-time information that single-bar triggers (Arc 8/9) miss → Pipeline E should clear 0.65 AUC |

## Trigger (locked at arc open — L_ARC_PROTOCOL §1.8)

**Swing-low definition (3-bar local low):**
- Swing-low at bar k iff `low[k] < min(low[k-3..k-1])` AND `low[k] < min(low[k+1..k+3])`

**1. Trend filter (pure structural, no MA — same as Arc 9):**
- Identify all swing-lows in window `t-30..t-1`
- Right-edge constraint: most recent identifiable swing-low at most bar `t-4`
- Require ≥ 1 swing-low exists in window
- Require `close[t-1] > min(swing_lows in window)`

**2. Bar t-2 (pullback bar):**
- `close[t-2] < open[t-2]` (bearish close)

**3. Bar t-1 (recovery bar):**
- `low[t-1] > low[t-2]` (higher low than pullback bar)
- `close[t-1] > close[t-2]` (higher close than pullback bar)

**4. Bar t (break trigger):**
- `close[t] > high[t-1]` (breaks recovery bar high)
- `close[t] > open[t]` (bullish close)
- `(close[t] − low[t]) / (high[t] − low[t]) ≥ 0.5` (close in upper half)

**5. Spacing & entry:**
- ≥ 20 bars since last signal on this pair
- Entry: bar t+1 open per `docs/SPREAD_SEMANTICS_LOCK.md`

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
| Arc config target | `configs/wfo_l_arc_12.yaml` |
| ATR period | 14 |

## Pool-size prior

Estimate 1,500–2,500 trades / 5y / 28 pairs. The 3-bar sequence is more restrictive than Arc 9's inside-bar pattern but the trend filter is identical. If Step 1 returns < 500, arc dies on §5 floor per §16a.

## Step 1 right-edge swing audit (mandatory)

Swing-low identification uses k+1..k+3 lookahead within the detection window only. Confirm at Step 1 that all swing-lows used for trend filter are at most bar `t-4`. If standard 5/5 lookahead spot-check shows any future-bar dependency, halt.

## Step 1 co-fire matrix (mandatory)

Report co-fire %:
- **KH-24** (`kb_exhaustion_bar`): bearish exhaustion vs 3BR bullish break — independence expected. If > 10%, flag.
- **Arcs 8/9/10/11** if Step 1 landed: report each. Expected highest with Arc 9 (both use same trend filter + bullish break geometry). Open-05 note.
- **Arcs 13/14/15/16** if Step 1 landed: report each. Expected lower (different families).

## Hypothesis notes (informational, not gating)

3BR is the cleanest test of "multi-bar sequence vs single-bar trigger." If it clears E where Arc 8 (PR-HHHL) fails — or vice versa — that's a concrete signal-design lesson. If both clear, the 3-bar sequence is redundant complexity over PR-HHHL. If both fail, the trend-continuation feature class is dead at 4H regardless of trigger granularity.

Path-shape expectation: Stepwise climber on clean continuation; whipsaw on failed break. Cluster heterogeneity at Step 2 expected.

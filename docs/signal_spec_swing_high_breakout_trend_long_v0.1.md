# signal_spec_swing_high_breakout_trend_long_v0.1

> Standalone signal spec. Authored by analyst, referenced by `results/ARC_QUEUE.md` Arc 11. Read by CC orchestrator at arc-open per dispatch reading list.

## Identification

| Field | Value |
|---|---|
| Name | `signal_swing_high_breakout_trend_long_v0.1` |
| Family | Trend continuation (structural breakout at historical reference) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | n/a (single-TF) |
| Pair set | 28 FX (KH-24 set) |
| Hypothesis | Break magnitude relative to meaningful historical reference (`H_ref`), reference freshness, and trigger-bar geometry are entry-time observable → Pipeline E should clear 0.65 AUC |

## Trigger (locked at arc open — L_ARC_PROTOCOL §1.8)

**Swing definitions (3-bar local extreme):**
- Swing-high at bar k iff `high[k] > max(high[k-3..k-1])` AND `high[k] > max(high[k+1..k+3])`
- Swing-low at bar k iff `low[k] < min(low[k-3..k-1])` AND `low[k] < min(low[k+1..k+3])`

**1. Trend filter (structural, no MA — same convention as Arc 9):**
- Identify all swing-lows in window `t-30..t-1`
- Require ≥ 1 swing-low exists in window
- Require `close[t-1] > min(swing_lows in window)` — price holds above the recent swing-low chain

**2. Reference swing-high:**
- Identify all swing-highs in window `t-20..t-1`
- Right-edge constraint: most recent identifiable swing-high at most bar `t-4`
- `H_ref` = most recent identifiable swing-high in window
- Require `H_ref` exists in window (if no swing-high in last 20 bars, no signal)

**3. Break trigger at bar t:**
- `close[t] > H_ref + 0.10 × ATR(14)_4H[t]` (decisive close above reference with buffer)
- `close[t] > open[t]` (bullish close)
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
| Arc config target | `configs/wfo_l_arc_11.yaml` |
| ATR period | 14 |

## Pool-size prior

Estimate 1,500–3,000 trades / 5y / 28 pairs. If Step 1 returns < 500, arc dies on §5 floor per §16a.

## Step 1 right-edge swing audit (mandatory)

Both swing-high (for `H_ref`) and swing-low (for trend filter) use k+1..k+3 lookahead within the detection window only. Confirm at Step 1 that:
- `H_ref` is at most bar `t-4` in any selected signal
- All swing-lows used for trend filter are at most bar `t-4`

If standard 5/5 lookahead spot-check shows any future-bar dependency in trigger-evaluation fields, halt — engine-touching, do not patch.

## Step 1 co-fire matrix (mandatory)

Report co-fire %:
- **KH-24** (`kb_exhaustion_bar`): bearish single-bar exhaustion vs SHB bullish breakout — independence expected. If > 10%, flag.
- **Arc 8** (`signal_pullback_resume_hhhl_long_v0.1`) if Step 1 landed: **expected 15-30%** — pullback-resume that also clears prior swing-high fires both. Not disqualifying if path-shape clusters differ; quantify cluster-by-cluster overlap at Step 3 if feasible.
- **Arc 9** (`signal_inside_bar_break_trend_long_v0.1`) if Step 1 landed: lower expected (IB-break + prior swing-high break are independent conditions in general).
- **Arc 10** (`signal_d1_swing_low_rejection_long_v0.1`) if Step 1 landed: expected low (different anchor TF). Open-05 note.

## Arc 8 differential note (informational, not gating)

SHB and PR-HHHL differ in the reference broken: PR-HHHL clears `high[t-1]` (single prior bar); SHB clears most recent swing-high (multi-bar historical reference). Hypothesis: SHB's reference carries more information, producing cleaner Pipeline E features (`break_magnitude_atr = (close[t] − H_ref) / ATR`, `H_ref_freshness_bars`, `prior_leg_length`). Three informative outcomes:
- Both clear E → trend-continuation broadly extractable
- Arc 11 only → reference magnitude matters; design future signals around historical levels
- Arc 8 only → single-bar trigger is sufficient; complexity not needed

## Hypothesis notes (informational, not gating)

Path-shape expectation: Stepwise climber or Monotone ascent family on breaks that hold; whipsaw on breaks that fail (`H_ref` retest + reversal). Failure mode is the source of cluster heterogeneity at Step 2.

`H_ref` age uncontrolled (4-19 bars). Stale references = different setups than fresh. Should surface as cluster heterogeneity at Step 2; worth watching in Step 1 distributions.

# signal_spec_failed_breakdown_reversal_uptrend_long_v0.1

> Standalone signal spec. Authored by analyst, referenced by `results/ARC_QUEUE.md` Arc 15.

## Identification

| Field | Value |
|---|---|
| Name | `signal_failed_breakdown_reversal_uptrend_long_v0.1` |
| Family | Failed-pattern reversal (long after downside fakeout in uptrend context) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | n/a (single-TF) |
| Pair set | 28 FX (KH-24 set) |
| Hypothesis | Failed-breakdown geometry (depth, retake speed, ATR-normalised metrics) is entry-time observable. Tests whether Arc 6 failed-breakout-up died for class-specific reasons or trigger-specific reasons — Arc 15 is the symmetric setup → Pipeline E should clear 0.65 AUC |

## Trigger (locked at arc open — L_ARC_PROTOCOL §1.8)

**Swing definitions (3-bar local extreme):**
- Swing-high at bar k iff `high[k] > max(high[k-3..k-1])` AND `high[k] > max(high[k+1..k+3])`
- Swing-low at bar k iff `low[k] < min(low[k-3..k-1])` AND `low[k] < min(low[k+1..k+3])`

**1. Uptrend context:**
- Identify all swing-lows in window `t-30..t-1`; require ≥ 1 swing-low exists
- Identify all swing-highs in window `t-30..t-1`; require ≥ 1 swing-high exists
- Require `close[t-1] > min(swing_lows in window)` — price holds above recent swing-low chain
- Right-edge constraint: all identifiable swings at most bar `t-4`

**2. Reference swing-low:**
- Identify swing-lows in window `t-15..t-1`
- `L_ref` = most recent identifiable swing-low in window (at most bar `t-4`)
- Require `L_ref` exists in window

**3. Failed breakdown (within last 3 bars):**
- Require some bar `k ∈ {t-3, t-2, t-1}` such that:
  - `low[k] < L_ref − 0.10 × ATR(14)[k]` (price broke below L_ref with buffer)
  - `close[k] > L_ref` (closed back above L_ref on same bar)
- Record `k_fakeout` = the most recent qualifying bar

**4. Trigger at bar t:**
- `close[t] > high[t-1]` (bullish continuation)
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
| Arc config target | `configs/wfo_l_arc_15.yaml` |
| ATR period | 14 |

## Pool-size prior

Estimate 800–1,800 trades / 5y / 28 pairs. Multiple compound conditions (uptrend + reference swing-low + fakeout in last 3 bars + trigger geometry) cut pool aggressively. If Step 1 returns < 500, arc dies on §5 floor per §16a — this is the highest pool-floor-risk signal of the Arc 12-16 batch.

## Smaller-pool risk

With 800–1,800 trades, clusters at Step 3 may not clear `size_fraction ≥ 0.10` (requires ~80–180 per cluster). §16a handles disposition: size_fraction failure near 0.10 with strong magnitude (`fwd_mfe_p50 ≥ 3.0R`) → HALT Path B; else KILL.

## Step 1 right-edge swing audit (mandatory)

Both swing-high and swing-low identification use k+1..k+3 lookahead within the detection window only. Confirm at Step 1 that:
- All swing-lows used for trend filter are at most bar `t-4`
- All swing-highs used for trend filter are at most bar `t-4`
- `L_ref` is at most bar `t-4`

If standard 5/5 lookahead spot-check shows any future-bar dependency, halt.

## Step 1 fakeout-bar validity audit (mandatory, novel)

The fakeout condition allows `k ∈ {t-3, t-2, t-1}`. Confirm at Step 1:
- `k_fakeout` is always strictly less than `t` (no same-bar fakeout)
- For all signals, the fakeout bar has both `low[k] < L_ref` and `close[k] > L_ref` (intrabar low must be below L_ref AND close must be above — single-bar wick-and-reverse pattern)
- L_ref is older than `k_fakeout` (the level being faked existed before the fakeout)

If any of these fail, halt — fakeout definition is structurally broken.

## Step 1 co-fire matrix (mandatory)

Report co-fire %:
- **KH-24** (`kb_exhaustion_bar`): bearish exhaustion vs FBR bullish failed-breakdown reversal — independence expected. If > 10%, flag.
- **Arc 6** (closed): structural cousin. Arc 6 was failed-breakout-up reversal; Arc 15 is failed-breakdown-down reversal. Co-fire near zero by direction.
- **Arcs 8/9/10/11/12** if Step 1 landed: report each. Expected low — FBR triggers after a fakeout, not on clean continuation. Open-05 note.
- **Arcs 13/14/16** if Step 1 landed: report each. Expected very low with Arc 14 (MRS) despite both being non-Arc-8 family — different geometry (stretch vs fakeout).

## Hypothesis notes (informational, not gating)

FBR is the controlled test of Arc 6's failure mode. Arc 6 died at Step 4 with Pipeline E both clusters AUC ~0.60 (below 0.65 gate). Two interpretations:

- **Class-specific:** failed-pattern reversal signals can't be discriminated on entry-time features at all. FBR will die the same way.
- **Trigger-specific:** Arc 6's failed-breakout-up trigger had specific geometry issues (e.g., breakout was already overextended by the time it failed). FBR's failed-breakdown trigger may have cleaner geometry because the prior context is an uptrend, not an overextended breakout.

If FBR clears E where Arc 6 didn't, that's a direction-asymmetry signal worth documenting in the closure doc and shelved-arcs register.

Path-shape expectation: bimodal — clean continuation after fakeout (Stepwise climber) vs failed continuation (early peak then deeper decline). The bimodality risk (Open-13 / Open-14 territory) may bite at Step 2; watch silhouette scores. Aggregate evaluation may be the deciding §7 path here.

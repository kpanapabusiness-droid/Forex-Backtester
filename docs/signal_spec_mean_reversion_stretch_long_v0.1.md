# signal_spec_mean_reversion_stretch_long_v0.1

> Standalone signal spec. Authored by analyst, referenced by `results/ARC_QUEUE.md` Arc 14.

## Identification

| Field | Value |
|---|---|
| Name | `signal_mean_reversion_stretch_long_v0.1` |
| Family | Counter-trend reversion within trend (the only non-continuation signal in current batches) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | n/a (single-TF) |
| Pair set | 28 FX (KH-24 set) |
| Hypothesis | Stretch magnitude (ATR-normalised distance below recent high) + recovery-bar geometry are entry-time observable → Pipeline E should clear 0.65 AUC. Diagnostic-value high: tests whether reversal-class signals can clear E at all (Arcs 5/6/7 all failed Step 3/4 in this family) |

## Trigger (locked at arc open — L_ARC_PROTOCOL §1.8)

**Swing-low definition (3-bar local low):**
- Swing-low at bar k iff `low[k] < min(low[k-3..k-1])` AND `low[k] < min(low[k+1..k+3])`

**1. Trend context (weaker than Arc 9/11 — admits more pullback depth):**
- Identify all swing-lows in window `t-50..t-1`
- Right-edge constraint: most recent identifiable swing-low at most bar `t-4`
- Require ≥ 1 swing-low exists in window
- Require `close[t-1] > min(swing_lows in window)` — price holds above 50-bar swing-low chain

**2. Stretch condition:**
- Let `H_10 = max(high[t-10..t-1])` (10-bar high in window strictly before signal bar)
- Require `close[t-1] < H_10 − 1.5 × ATR(14)[t-1]` — close at least 1.5 ATR below 10-bar high

**3. Reversal trigger at bar t:**
- `close[t] > open[t]` (bullish close)
- `close[t] > close[t-1]` (higher close than prior bar)
- `(close[t] − low[t]) / (high[t] − low[t]) ≥ 0.6` (close in upper 40% of bar)

**4. Spacing & entry:**
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
| Arc config target | `configs/wfo_l_arc_14.yaml` |
| ATR period | 14 |

## Pool-size prior

Estimate 2,000–3,500 trades / 5y / 28 pairs. Pullbacks are common; weaker trend filter admits more candidates than Arc 9/11. If Step 1 returns < 500, arc dies on §5 floor per §16a.

## Step 1 right-edge swing audit (mandatory)

Swing-low identification uses k+1..k+3 lookahead within the detection window only. Confirm at Step 1 that all swing-lows used for trend filter are at most bar `t-4`. If standard 5/5 lookahead spot-check shows any future-bar dependency, halt.

## Step 1 co-fire matrix (mandatory)

Report co-fire %:
- **KH-24** (`kb_exhaustion_bar`): KH-24 is bearish exhaustion in trend, MRS is bullish reversal after stretch — opposite by construction. Co-fire expected very low. If > 5%, flag.
- **Arcs 8/9/10/11/12** if Step 1 landed: report each. Expected low — MRS triggers at pullback bottoms, not continuation tops. The mechanical opposite of Arc 8/9/11/12 trigger geometry. Open-05 note.
- **Arcs 13/15/16** if Step 1 landed: report each.

## Hypothesis notes (informational, not gating)

MRS is the highest-risk signal in the Arc 12-16 batch by ship-probability prior, but the highest diagnostic-value signal if it fails. Three independent reversal/recovery signals died at Step 3/4 in Arcs 5-7. MRS is the cleanest fourth test:

- If MRS dies at Step 3 same way Arc 3/5 died (path quality clean but §2 fails on shape_tag) → reversal-class signals consistently produce bimodal/scattered path-shape clusters, even with clean stretch geometry. Strong methodological conclusion.
- If MRS dies at Step 4 same way Arc 6/7 died (E fails AUC 0.65) → reversal-class signals can't be discriminated on entry-time features at all. Even stronger methodological conclusion: reversal class is dead on this protocol surface, future work goes to richer D1 or hybrid pipelines (not in this batch).
- If MRS clears E → Arc 5-7 failures were trigger-specific, not class-specific. Reversal signals reopen as a viable line.

Path-shape expectation: V-shape recovery family (Arc 7 pattern) on clean reversions; whipsaw bottom or stair-step decline on failures.

Co-fire with KH-24 expected near-zero by trigger geometry; this is a deliberately complementary signal to the current live system.

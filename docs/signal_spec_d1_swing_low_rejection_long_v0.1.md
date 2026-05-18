# signal_spec_d1_swing_low_rejection_long_v0.1

> Standalone signal spec. Authored by analyst, referenced by `results/ARC_QUEUE.md` Arc 10. Read by CC orchestrator at arc-open per dispatch reading list.

## Identification

| Field | Value |
|---|---|
| Name | `signal_d1_swing_low_rejection_long_v0.1` |
| Family | Multi-TF trend continuation (HTF-anchored level rejection) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | D1 (one-day lag mandatory per L_ARC_PROTOCOL §1.4 / §1a) |
| Pair set | 28 FX (KH-24 set) |
| Hypothesis | HTF context (D1 swing-low identity + D1 HL structure) + LTF rejection geometry are entry-time observable → Pipeline E should clear 0.65 AUC |

## Trigger (locked at arc open — L_ARC_PROTOCOL §1.8)

**D1 anchor identification (one-day lag enforced — KH-24 convention; see `KH24_SYSTEM_LOCK.md` / `scripts/phase_kgl_v2_4h_wfo.py`):**

1. **D1 swing-low at day d:** `low[d] < min(low[d-3..d-1])` AND `low[d] < min(low[d+1..d+3])` (3-bar D1 local low)
2. **Most recent identifiable D1 swing-low** at 4H bar t = `L_1`. Search window: D1 bars closing strictly before 4H bar t's open. Right-edge constraint: most recent identifiable L_1 at most D1[d_t − 4] where d_t is the D1 bar containing 4H bar t.
3. **Prior D1 swing-low** = `L_0` (next-most-recent before L_1)
4. **D1 HL structure:** require both `L_1` and `L_0` exist within last 30 D1 bars at 4H bar t, AND `L_1 > L_0` (strictly ascending)
5. **L_1 freshness:** D1 bar containing L_1 not older than 20 D1 bars at 4H bar t

**4H test/reject (bar t = signal bar):**

6. **Test (proximity):** `low[t] ≤ L_1 + 0.25 × ATR(14)_4H[t]` (price comes within 0.25 ATR of D1 swing-low)
7. **Reject (close back above):** `close[t] > L_1 + 0.10 × ATR(14)_4H[t]` (decisive close above level with buffer)
8. **Trigger-bar geometry:**
   - `close[t] > open[t]` (bullish close)
   - `(close[t] − low[t]) / (high[t] − low[t]) ≥ 0.6` (close in upper 40% of bar)

**Spacing & entry:**

9. ≥ 20 4H bars since last signal on this pair
10. Entry: bar t+1 open per `docs/SPREAD_SEMANTICS_LOCK.md` round-trip convention

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
| Arc config target | `configs/wfo_l_arc_10.yaml` |
| ATR period | 14 (both 4H and D1 where used) |

## Pool-size prior

Estimate 1,000–2,000 trades / 5y / 28 pairs. Multi-condition + multi-TF cuts pool relative to single-TF signals. If Step 1 returns < 500, arc dies on §5 floor per §16a.

## Smaller-pool risk

With 1,000–2,000 trades, individual clusters at Step 3 may not clear `size_fraction ≥ 0.10` (requires ~100–200 trades per cluster at the smaller pool end). §16a handles disposition mechanically.

## Step 1 D1 lag verification (mandatory, lookahead-critical)

**Implementation:** D1 swing-low identification uses `merge_asof` backward against pre-shifted D1 dates, matching the KH-24 convention in `scripts/phase_kgl_v2_4h_wfo.py`. Same-day D1 close MUST NOT be available at 4H bar t.

**NaN-perturbation test:** in addition to the standard 5/5 audit, verify on 3 explicit synthetic test cases that swapping D1[d_t] data for NaN produces identical Arc 10 signal output. If any signal disappears under this perturbation, the lag is broken → halt with halt summary (engine-touching, do not patch).

**Right-edge swing audit:** D1 swing identification uses k+1..k+3 lookahead within the D1 swing-detection window only. Most recent identifiable L_1 must be at most D1[d_t − 4]. Confirm at Step 1 that L_1 selected for any signal is never at D1[d_t − 3] or later.

## Step 1 co-fire matrix (mandatory)

Report co-fire %:
- **KH-24** (`kb_exhaustion_bar`): bearish single-bar exhaustion + D1 regime filter vs DLR bullish rejection at D1 swing-low — independence expected (different trigger direction, different D1 condition). If > 10%, flag.
- **Arc 8** (`signal_pullback_resume_hhhl_long_v0.1`) if Step 1 landed: report overlap %. DLR HTF-anchored vs PR-HHHL intra-TF structural — moderate overlap expected. Open-05 note.
- **Arc 9** (`signal_inside_bar_break_trend_long_v0.1`) if Step 1 landed: expected low. Open-05 note.
- **Arc 11** (`signal_swing_high_breakout_trend_long_v0.1`) if Step 1 landed: expected low (different anchor TF, different trigger geometry). Open-05 note.

## Hypothesis notes (informational, not gating)

DLR is the only Arc 8-11 candidate bringing HTF context. If E clears it but fails Arc 8/9/11, that's a feature-class signal worth noting in the Step 4 halt summary for cross-arc synthesis.

L_1 freshness + L_0/L_1 ascending requirement is the source of pool-size compression. Relaxing freshness from 20 → 30 D1 bars is a future-arc candidate, not within-arc.

Path-shape expectation: V-shape recovery family (price tests level, rejects, runs) — different cluster signature from Arc 8 (Stepwise climber expected) or Arc 9 (early-peak after compression expected). Arc 7 showed V-shape is capturable but not extractable on entry-time features alone; DLR's HTF features are the variable being tested.

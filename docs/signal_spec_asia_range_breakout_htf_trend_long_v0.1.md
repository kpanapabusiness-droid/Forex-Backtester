# signal_spec_asia_range_breakout_htf_trend_long_v0.1

> Standalone signal spec. Authored by analyst, referenced by `results/ARC_QUEUE.md` Arc 13.

## Identification

| Field | Value |
|---|---|
| Name | `signal_asia_range_breakout_htf_trend_long_v0.1` |
| Family | Session breakout × HTF trend (multi-TF, session-anchored) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | D1 (one-day lag mandatory per L_ARC_PROTOCOL §1.4 / §1a) |
| Pair set | 28 FX (KH-24 set) |
| Hypothesis | Session structure (Asia compression → London/NY directional break) + HTF trend context are entry-time observable and orthogonal to swing-based price-action signals → Pipeline E should clear 0.65 AUC |

## Trigger (locked at arc open — L_ARC_PROTOCOL §1.8)

**Session definitions (MT5 server time = GMT+2 winter / GMT+3 summer, DST-aware):**
- **Asia session bars (day d):** 4H bars closing at 04:00 and 08:00 server time on day d
- **London/NY session bars (day d):** 4H bars closing at 12:00 and 16:00 server time on day d

**D1 anchor (one-day lag enforced per §1.4):**
1. **D1 trend filter:** `close_D1[d_t − 1] > close_D1[d_t − 5]` (close five D1 bars ago, strictly before bar t). Structural weak-trend filter; no MA.

**Asia range identification at 4H bar t:**
2. Let `t_asia_high` = max(high) of Asia session bars on day d_t
3. Let `t_asia_low` = min(low) of Asia session bars on day d_t
4. Asia range exists only if both Asia bars are present in data for day d_t (no missing bars)

**Trigger at 4H bar t (must be a London/NY session bar):**
5. Bar t closes at 12:00 or 16:00 server time on day d_t
6. `close[t] > t_asia_high + 0.10 × ATR(14)_4H[t]` (decisive break above Asia high with buffer)
7. `close[t] > open[t]` (bullish close)

**Spacing & entry:**
8. Max one signal per pair per day (the earlier of the two London/NY bars wins if both qualify)
9. Entry: bar t+1 open per `docs/SPREAD_SEMANTICS_LOCK.md`

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
| Arc config target | `configs/wfo_l_arc_13.yaml` |
| ATR period | 14 (4H and D1 where used) |

## Pool-size prior

Estimate 1,500–3,000 trades / 5y / 28 pairs (one chance per pair per day × ~250 trading days × 28 pairs × ~15–30% trigger rate). If Step 1 returns < 500, arc dies on §5 floor per §16a.

## Step 1 D1 lag verification (mandatory, lookahead-critical)

D1 close at `d_t − 1` must reference the D1 bar closing strictly before 4H bar t's open. Same-day D1 close MUST NOT be available. Use KH-24's D1 lag convention as reference (`scripts/phase_kgl_v2_4h_wfo.py`, `merge_asof` backward).

**NaN-perturbation test:** swap D1[d_t] data for NaN on 3 explicit synthetic test cases; if any signal disappears, the lag is broken → halt (engine-touching, do not patch).

## Step 1 session-time alignment audit (mandatory, novel)

Confirm at Step 1:
1. All bars classified as "Asia session" have timestamps closing 04:00 or 08:00 server time
2. All bars classified as "London/NY session" have timestamps closing 12:00 or 16:00 server time
3. DST transitions handled correctly: 5ers MT5 server transitions GMT+2 ↔ GMT+3 on European DST schedule. Sample 5 bars from days within 7 days of a DST transition; confirm session classification is consistent
4. Days with missing Asia bars (broker downtime, holiday) produce no signal

If audit fails, halt — session boundary handling is non-trivial and must not be papered over.

## Step 1 co-fire matrix (mandatory)

Report co-fire %:
- **KH-24** (`kb_exhaustion_bar`): bearish exhaustion vs ARB bullish session break — independence expected. If > 10%, flag.
- **Arcs 8/9/10/11/12** if Step 1 landed: report each. Expected low across the board (ARB is the only session-anchored signal). Open-05 note.
- **Arcs 14/15/16** if Step 1 landed: report each.

## Hypothesis notes (informational, not gating)

ARB is the highest-distance signal in the Arc 12-16 batch from the swing-based price-action family. If it clears E where Arcs 8/9/10/11/12 don't (or vice versa), that's the strongest cross-arc feature-class signal of the second batch.

Time-of-day features (hour of bar close, day of week) are entry-time observable and may carry signal. Pipeline E should explore these in Step 4.

Path-shape expectation: bimodal cluster split likely — clean breakouts (Stepwise climber) vs failed breakouts (early peak then reversal). The breakout-fail problem applies to ARB as much as to Arc 11 (SHB); ARB's session-anchored reference may be a better discriminator than SHB's swing-high reference because it's a more cohesive setup (range identity).

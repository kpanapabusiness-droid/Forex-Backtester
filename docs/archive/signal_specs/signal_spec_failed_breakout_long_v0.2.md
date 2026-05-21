# Signal spec — failed-breakout reversal (long) — v0.2

> Out-of-registry structural signal spec used by Arc 6 (`discovery/lomega_regime_conditional`).
> Status as-of 2026-05-17: arc closed DIES at Step 4 deployability. Signal NOT permanently eliminated. See `docs/arc_results/ARC_6_RESULT.md`.

---

## v0.1 → v0.2 erratum

The v0.1 literal definition `swing_low_N = min(low[t-N..t-1])` is mathematically unsatisfiable in combination with the breakout-bar requirement `close[t*] < swing_low_N` for `t* ∈ [t-M..t-1]`:

- `t*` falls inside the swing-low window `[t-N..t-1]` (since N = 20 ≥ M = 5)
- By definition of min: `low[t*] ≥ swing_low_N`
- By OHLC invariant: `close[t*] ≥ low[t*] ≥ swing_low_N`
- Therefore `close[t*] < swing_low_N` cannot hold for any `t*`

v0.2 corrects this by anchoring `swing_low_N` to bars strictly preceding the M-bar breakout window:

```
swing_low_N = min(low[t-N-M..t-M-1])
```

The N = 20-bar swing evidence is preserved; the total signal span becomes N + M = 25 bars instead of 20. Verified at Arc 6 Step 1 (no-lookahead spot check 5/5); locked as-built throughout Arc 6 (which failed at Step 4 deployability — see `docs/arc_results/ARC_6_RESULT.md`). Erratum discovered during Step 1 no-lookahead validation.

---

## Structural story

A swing low at the N-bar lookback is an objective level traded as support by many participants. When price breaks below it on a closed bar (the breakout bar), short-side participants are drawn in expecting continuation. If price then reclaims the level — closing back above `swing_low_N` on a bullish-close bar within M bars — the breakout failed; trapped shorts are forced to cover, providing a short-term long edge.

Spec captures the failed-breakout reversal mechanically: closed-bar break of an objective swing low, followed by a closed-bar bullish reclaim of the same level within a short M-bar window.

---

## Event definition

For each pair × bar `t` on the signal timeframe:

1. **Swing-low level defined** (corrected per v0.2 erratum):

   ```
   swing_low_N = min(low[t-N-M..t-M-1])
   ```

2. **Breakout bar identified** — there exists `t* ∈ [t-M..t-1]` such that:

   ```
   close[t*] < swing_low_N − magnitude_atr_floor × ATR(14)[t*-1]
   ```

   Smallest such `t*` is the breakout bar; ATR is measured at the bar preceding the breakout bar to avoid lookahead.

3. **Reclaim bar identified** — bar `t` is the candidate signal bar if:

   ```
   close[t]  > swing_low_N
   close[t]  > open[t]            (bullish close)
   t - t*    ≤ M                  (reclaim within M bars)
   ```

4. **Entry** — long at bar `t+1` open per `SPREAD_SEMANTICS_LOCK`.

---

## Parameters (v0.2 defaults; no sweeps performed at Arc 6)

| Parameter | Value | Notes |
|---|---|---|
| N (swing lookback) | 20 bars | |
| M (reclaim window) | 5 bars | |
| `magnitude_atr_floor` | 0.25 × ATR(14) | minimum breakout magnitude below `swing_low_N` |
| ATR period | 14 | computed on the signal TF |
| Signal TF | 4H | |
| Direction | long only | |
| Multi-TF | 4H only | no higher-TF regime filter at v0.2 |

---

## Lookahead guarantees

- `swing_low_N` uses bars strictly preceding the breakout window `[t-M..t-1]`.
- Breakout bar magnitude floor uses `ATR(14)[t*-1]` — ATR reading from before the breakout bar's own close.
- Reclaim test at bar `t` uses `close[t]` and `open[t]` — both observable at bar `t` close.
- Entry is bar `t+1` open, per `SPREAD_SEMANTICS_LOCK`.

Spot-check at Arc 6 Step 1: 5/5 sampled signals verified — every input to the event predicate is observable at-or-before the bar it gates.

---

## Backtester configuration as used in Arc 6

| Field | Value |
|---|---|
| Pair set | all 28 FX (KH-24 list) |
| Entry | bar `t+1` open per `SPREAD_SEMANTICS_LOCK` |
| Initial SL (simulation) | `entry_price − 2.0 × ATR(14)_4H[t]` |
| Forward window | 240 bars |
| Exposure cap | max 1 open position per pair |
| Risk per trade | 0.5% × reset floor balance |
| Spread | `configs/spread_floors_5ers.yaml` |
| Data window | 2020-10-01 → 2026-01-31 |
| Arc config | `configs/wfo_l_arc_6.yaml` |

These are arc-side backtester defaults under L_ARC_PROTOCOL v2.1.2, not part of the signal spec itself — included here for traceability.

---

## Result reference

Arc 6 disposition: **DIES at Step 4 deployability** (Pipeline E AUC 0.600 / 0.590 vs 0.65 floor; Pipeline D1 mechanically clears AUC ≥ 0.60 but threshold sweep collapses to max-F1 fallback at sub-1% recall). Full closure: `docs/arc_results/ARC_6_RESULT.md`.

The signal is not permanently eliminated — path quality is clean at v2.1.2 §2 (c2 Stepwise, mfe_p50 4.47R, ww_pp 0.000); entry-time predictability under the 27-feature catalogue at 4H on majors is too thin to filter on. Re-attack candidates: richer feature regime, multi-TF entry context, ensemble approaches, or different signal TF.

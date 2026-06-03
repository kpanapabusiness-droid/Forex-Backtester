# Exit Policy — sl_partial_close_1r_runner_trail

> **Audience:** Someone who needs to understand what happens after a trade enters.
> **For code:** `core/sim/exit_policies/sl_partial_close_1r_runner_trail.py` (lab) and `deployment/ea/include/ExitPolicyEngine.mqh` (live).

## The three-stage policy at a glance

Every trade flows through three potential stages, in order:

1. **Stage 1 — Initial SL hit:** If price retraces to the initial stop level before TP1, the full position closes at the stop. Trade ends.
2. **Stage 2 — TP1 partial close:** If price reaches the +1R level (entry price + initial SL distance), close 50% of the position at that level. Lock in the partial profit.
3. **Stage 3 — Runner trail:** The remaining 50% runs with a trailing stop. The trail is the rolling high minus 1 ATR (1×R, where R = initial SL distance). When price closes below the trail level, the runner exits.

Plus three escape valves:

- **Time exit:** If the trade is still open after 240 H4 bars (~40 days), force close.
- **Daily DD halt:** If account daily drawdown breaches threshold, halt new entries and (at higher threshold) force close all.
- **Total DD halt:** If account total drawdown breaches threshold, same as daily but with bigger threshold.

## Stage 1 — Initial SL

**Where it sits:** At entry, the SL is placed at `entry_price - (3.5 × ATR_at_signal_bar)`. ATR is computed using Wilder's smoothing on bid prices over 14 H4 bars.

**Why 3.5 × ATR:** This came out of capturability analysis in the L_ARC_PROTOCOL Step 3. Tested 2.0, 2.5, 3.0, 3.5, 4.0× multipliers; 3.5× produced the best balance of "wide enough to absorb normal noise" vs "tight enough to bound loss". Locked at 3.5× in v3.0.2.

**Mechanically:** The EA places the SL on the broker via `PositionModify()` at entry. If price touches the SL, the broker fires the close — the EA detects on next OnTick and logs the event as `initial_sl_hit`.

**Why broker-side SL (not EA-internal):** Crash safety. If the EA process dies between bars, the position is still protected by the broker's SL order. Trade-off documented in `phase_1_build_intent.md §6.3`.

## Stage 2 — TP1 partial close

**Where it sits:** At `entry_price + (1.0 × initial_SL_distance)`, i.e. +1R. The "1R" here equals the SL distance itself (3.5 × ATR), since R is defined relative to the SL distance.

**What happens:** When price reaches TP1, the EA actively sends a market sell order for 50% of the original lot size. This is an EA-initiated close, not a broker SL. Logs as `partial_close` event.

**After the partial:** The remaining 50% is the "runner" — it goes into Stage 3 trail.

**Why partial close (not full TP):** Empirically, the half-close-at-1R approach extracts ~60% of the strategy's total return at very low further risk, while the trailing runner captures the remaining ~40% of return at much higher variance. Splitting into partial + trail is the optimal balance found in the discovery phase.

## Stage 3 — Runner trail

**Where it sits:** After TP1 fires, the runner has a trail SL. The trail level is `peak_high_bid_since_entry - (1.0 × R)`, where R = initial SL distance (3.5 × ATR).

**How it ratchets:** Each H4 bar close after TP1, the EA checks if the latest H4 high pushed peak_high_bid higher. If yes, the trail level rises with it. If no, trail level stays.

**Critically:** the trail SL ratchets **upward only**. Once it's at level X, it never moves down — only up if new highs appear.

**How it fills:** Same mechanism as initial SL. EA places trail SL on broker via `PositionModify()`. When price touches trail level intra-bar, broker fires close. EA detects on next OnTick. Logs as `trail_stop` (inferred from position state — see Option 1 + Bug A fix in commit 74748ef).

**Why 1×R trail (not 0.5R, 2R, etc):** Tested in discovery. 1×R is wide enough to survive normal pullbacks but tight enough to capture meaningful runner R-multiples. Locked at 1×R in v3.0.2.

## Escape valve — Time exit at 240 bars

**Why it exists:** A trade that has been open for 240 H4 bars (~40 calendar days, accounting for weekends) has likely exhausted its directional thesis. Keeping it open consumes margin and adds exposure without expected edge.

**Mechanically:** When `bar_ord >= 240` and the position is still open, the EA sends a market close order. Logs as `time_exit` event.

**Empirically:** ~3-5% of trades exit via time-exit in backtest. Net contribution is mildly positive (slight average-positive R at time-exit moment), but the cap is more about bounding exposure than capturing edge.

## Escape valve — Daily / Total DD halt

**Daily DD thresholds:**
- `Daily_DD_Halt_Pct = 0.035` (3.5%): no new entries
- `Daily_DD_CloseAll_Pct = 0.045` (4.5%): force close all open positions

**Total DD thresholds:**
- `Total_DD_Halt_Pct = 0.07` (7%): no new entries
- `Total_DD_CloseAll_Pct = 0.08` (8%): force close all open positions

The thresholds sit inside the prop firm's hard limits (5% daily, 10% total for both 5ers and FundedNext), providing a safety margin.

**How DD is computed:** Daily DD measures drawdown from the EET-day's starting equity (FundedNext) or UTC-day's starting equity (5ers). Total DD measures drawdown from the all-time equity high.

**What happens when halt fires:** The EA logs an `equity_block` event and does not enter new trades. Existing positions continue to manage (trail, partial, time exit) — they're not force-closed until the higher threshold (`CloseAll`) trips.

**What happens when CloseAll fires:** The EA force-closes every open position via market orders, regardless of P&L. This is the nuclear option. Recovery requires manual intervention to re-enable trading.

## What is NOT in the exit policy

- **No fixed R-multiple TP beyond TP1.** The runner has no upper TP — it trails until stopped out, time-exited, or DD-halted.
- **No breakeven-stop adjustment.** Some systems move SL to entry after TP1. Arc 10 does not — the trail picks up where the initial SL leaves off, and the trail is wider than entry initially.
- **No re-entry logic.** Once a trade exits, the system waits for the next fresh signal on that pair. No "average down", "scale in", or "re-enter on pullback".
- **No correlation-based exits.** Each pair manages its own positions independently.
- **No news-based exits.** News filter blocks new entries near scheduled news but doesn't close existing positions.

## How the lab simulator vs live EA differ

Both implement the same logic. The differences are in fill convention:

| Event | Lab simulator | Live EA |
|---|---|---|
| Entry | Fill at next-bar open_bid | Market order at next-bar first tick (~near broker ask) |
| TP1 partial | Fill at +1R level exact | Market order at +1R level (~near broker bid, typically clean) |
| Trail SL hit | Fill at next-bar open_bid (after trail-breach close) | Broker SL fires at trail level (~at trail SL, modulo intra-bar slippage) |
| Time exit | Fill at bar 240 close | Market order at first tick when `bar_ord >= 240` |

**Net effect on live vs backtest:** The trail-exit convention difference is the largest. Broker fills at trail level; lab fills at next-bar open (which is often worse than trail level). Live execution is therefore ~0-0.3R per trail trip BETTER than lab predicts.

This was quantified in the Phase 2 EET parity work and is documented as a known tolerance (not a bug). The lab's headline numbers are conservative vs expected live performance on the trail-exit dimension.

## What can go wrong

- **Gap-through-SL:** If price gaps below SL on Sunday open or during a news event, fill is at the gap-open price, not the SL level. Loss exceeds 1R. This is the "tail risk" not modeled in haircuts. Mitigation: position sizing keeps single-trade max loss to ~0.40% of account at worst (the operating risk tier), so even a 2-3R gap is survivable.
- **Broker SL ignored:** Hypothetically, a broker could decline to honor the SL order during a freeze. Not seen in practice with 5ers or FundedNext, but possible. Mitigation: EA also tracks position state and would close manually if it detected the SL wasn't filling.
- **EA crashes mid-position:** Recovery logic reconstructs position state from broker on restart. Logs as `recovery_reconstructed` event. Trail SL is re-anchored at the last known peak (conservative).
- **Sidecar dies but EA stays alive:** EA detects stale heartbeat (>10 minutes by default), blocks new entries, manages existing positions normally. Watchdog auto-restarts sidecar within 5 minutes.

## Configuration summary

These are the EA input parameters that control the exit policy:

| Parameter | Value | Effect |
|---|---|---|
| `Risk_Per_Trade` | 0.0050 (FN) / 0.0040 (5ers) | Position size as % of equity |
| `SL_ATR_Multiplier_Expected` | 3.5 | Initial SL distance in ATR units |
| `Time_Exit_Bars` | 240 | Time exit at this many H4 bars |
| `Daily_DD_Halt_Pct` | 0.035 | Block new entries at 3.5% daily DD |
| `Daily_DD_CloseAll_Pct` | 0.045 | Force close all at 4.5% daily DD |
| `Total_DD_Halt_Pct` | 0.07 | Block new entries at 7% total DD |
| `Total_DD_CloseAll_Pct` | 0.08 | Force close all at 8% total DD |

Do not change these without re-running the cost sweep and updating the deployment artifacts.

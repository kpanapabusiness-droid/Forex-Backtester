# Signal Lifecycle — From H4 Close to Closed Trade

> **Audience:** Someone tracing what happens on each step from market data to broker fill.
> **For code:** `deployment/sidecar/sidecar.py` (sidecar), `deployment/ea/Arc10_DLR_Sidecar_EA.mq5` (EA).

## The full chain

```
H4 boundary closes (e.g. 12:00 UTC for UTC convention)
    ↓
Sidecar wakes (every 4h, after H4 close + 10s buffer)
    ↓
Sidecar fetches H4 + D1 bars for 28 pairs from MT5
    ↓
Sidecar computes signal logic per pair (DLR gates)
    ↓
For any pair that fires signal: emit envelope JSON to signals_out/
    ↓
Sidecar writes sidecar.heartbeat with timestamp + pair list
    ↓
Sidecar enters sleep until next H4 boundary
    ↓
EA detects envelope on next tick (polling)
    ↓
EA validates envelope (config_hash, schema, freshness)
    ↓
EA computes position size from Risk_Per_Trade and SL distance
    ↓
EA places market order with SL set
    ↓
Broker fills entry, returns ticket
    ↓
EA writes trade_log.csv (entry event)
    ↓
EA monitors position on every tick
    ↓
On each new H4 bar (per-position bar tracking):
    EA checks TP1 level → if breached, send partial close
    EA checks trail logic → if applicable, modify broker SL
    EA checks time_exit (bar_ord >= 240) → if so, force close
    EA writes trade_log.csv events (partial_close, trail_modify)
    ↓
Trade exits via one of:
    - Broker SL fires (initial_sl_hit OR trail_stop, inferred by state)
    - EA time_exit (active close at bar 240)
    - EA equity_guard_force (active close on DD breach)
    ↓
EA writes trade_log.csv (exit event with strategic reason + fill price)
    ↓
EA moves processed envelope to signals_processed/
```

## Detail per stage

### Sidecar wake & data fetch

The sidecar service runs continuously via NSSM. Inside its main loop:

1. `compute_next_h4_close()` — calculates next H4 boundary in UTC, returns target sleep time
2. `sleep(target - now + 10s buffer)` — sleeps until 10s after the H4 boundary
3. On wake, `_loop_iteration()`:
   - For each of 28 pairs: `copy_rates_from_pos(symbol, TIMEFRAME_H4, 0, 300)` — fetch last 300 H4 bars
   - For each pair: `copy_rates_from_pos(symbol, TIMEFRAME_D1, 0, 100)` — fetch last 100 D1 bars
   - Pass H4 + D1 panels into `signals.lchar_dlr_long.compute_signal()`
   - If signal fires: build envelope dict, write to `signals_out/<PAIR>_<iso_timestamp>.json`
4. Write `sidecar.heartbeat` with current timestamp + pair list
5. Write `sidecar_state.json` with `last_processed_bar_utc` per pair
6. Return to sleep

**Critical: the boundary_convention flag determines anchor logic.** UTC → H4 boundaries at 00/04/08/12/16/20 UTC. EET → broker-local midnight boundaries (= UTC 22/02/06/10/14/18 winter, 21/01/05/09/13/17 summer).

### Envelope format

The JSON envelope written to `signals_out/` looks like:

```json
{
  "pair": "EURUSD",
  "signal_bar_close_utc": "2026-05-29T08:00:00Z",
  "entry_bar_open_utc": "2026-05-29T12:00:00Z",
  "direction": "long",
  "entry_price_estimate": 1.16245,
  "sl_at_entry_price": 1.13615,
  "sl_distance": 0.02630,
  "atr14_at_signal": 0.00749,
  "config_hash": "4467366b...840821e",
  "boundary_convention": "utc",
  "schema_version": "1.3",
  "audit": {
    "L1_value": 1.14820, "L0_value": 1.15390,
    "L1_age_d1_bars": 6, "L0_age_d1_bars": 12,
    "L1_to_atr_proximity": 0.83, "reject_buffer_atr": 0.31,
    "upper_fraction": 0.62
  }
}
```

The audit block is the lab's full audit fingerprint — what was the signal's full context at firing time. Used for Phase 2 parity comparison.

### EA polling & validation

The EA's `SignalPoller` runs on every OnTick:

1. Check if poll interval elapsed (default 5s minimum between polls)
2. Read `signals_out/` directory
3. For each `.json` file found:
   - Parse JSON
   - Validate schema (all required fields present)
   - Check `config_hash` matches `Expected_Config_Hash` input
   - Check `signal_bar_close_utc` is not stale (more than 2 hours old → reject)
   - Check sidecar heartbeat is fresh (default <600s old)
4. If all checks pass: process the envelope (enter trade)
5. If any check fails: move envelope to `signals_failed/`
6. After processing: move envelope to `signals_processed/`

### EA entry placement

Position sizing:
```
risk_dollars = current_equity × Risk_Per_Trade
sl_distance_pips = (entry_price - sl_at_entry_price) / pip_size
lot_size = risk_dollars / (sl_distance_pips × dollar_per_pip)
```

Round to broker lot precision (typically 0.01). Cap at broker's max position size.

Place the order:
```mql5
trade.Buy(lot_size, symbol, market_ask_price, sl_at_entry_price, 0, comment)
```

The SL is set on the order at entry. The broker holds it.

### EA per-bar management

The EA uses `last_processed_h4_bar` per position (topology fix). On each tick, for each open position:
1. Get current H4 bar time for the position's pair: `iTime(pair, PERIOD_H4, 0)`
2. If different from `last_processed_h4_bar` → new bar has closed → process bar logic
3. Bar logic:
   - Update peak_high_bid if new high
   - Check trail level: if `tp1_fired` AND `close_bid <= trail_level` → close runner via market sell
   - Check time_exit: if `bar_ord >= Time_Exit_Bars` → close via market sell
   - Update trail SL on broker: `PositionModify(ticket, new_trail_level, 0.0)`
   - Update `last_processed_h4_bar`

### Trade log entries

Every event writes a row to `trade_log.csv`. Schema:

```csv
timestamp_utc, event, signal_id, pair, ticket, direction, entry_price, 
sl_initial, sl_distance, r_atr, initial_lots, current_lots, peak_high_bid, 
trail_sl, tp1_fired, tp1_bar_ord, bar_ord, partial_price, fill_price, 
reason, daily_dd_pct, total_dd_pct, equity, note
```

`event` types:
- `entry` — new position opened
- `partial_close` — TP1 hit, half closed
- `trail_modify` — trail SL moved (bookkeeping; one per H4 bar after TP1)
- `exit` — position closed
- `recovery_reconstructed` — position rebuilt after EA restart

`reason` (only on `exit` events) values:
- `trail_stop` — runner exited via trail SL
- `initial_sl_hit` — full position exited via initial SL (before TP1)
- `time_exit` — exited at bar 240
- `equity_guard_force` — force-closed by DD halt
- `external_close` — broker-side close that doesn't match any of the above (manual intervention?)

## What happens on Friday close / weekend

- Sidecar's last cycle of the week fires at the Friday H4 close (Fri 20:00 UTC for UTC convention)
- After that, sidecar enters sleep until next H4 close, which is Monday 00:00 UTC (for UTC) — sidecar sleeps through the weekend
- Trades that were open at Friday close stay open over the weekend
- On Sunday/Monday market reopen, sidecar wakes at the next H4 boundary
- For the weekend timestamp edge case: a signal that fired on Friday's last H4 bar has `entry_bar_open_utc` projected to Sunday/Monday reopen (handled by `_project_entry_bar_open()`)

## What happens when sidecar dies

- EA continues running. On next signal poll: heartbeat is stale → block new entries
- Existing positions continue to be managed normally (trail, partial, time exit)
- Watchdog detects stale heartbeat within 5 minutes
- Watchdog restarts NSSM service: `nssm restart Arc10Sidecar5ers` (or `...FundedNext`)
- Sidecar reboots, computes next H4 boundary, sleeps
- Heartbeat refreshes on next cycle
- EA sees fresh heartbeat, resumes entry processing

## What happens when EA dies

- Broker positions still exist on the broker's side, with SL still set
- On EA restart: recovery logic reads ea_positions.json
- For each known position: query broker for ticket → if still open, reconstruct state (peak_high_bid, trail_sl_current, tp1_fired flag, bar_ord)
- Logs `recovery_reconstructed` event per position
- EA resumes normal management

## What happens when MT5 restarts (Sunday reboot, say)

- VPS reboots Sunday at scheduled time (operator action)
- MT5 services auto-start when Windows boots (configured in NSSM)
- Both MT5 terminals launch, auto-login (if "remember password" is set; otherwise manual)
- Sidecar services auto-start (NSSM)
- Sidecars probe MT5 connection
- If MT5 not ready: sidecar logs error, retries
- Once MT5 ready: sidecar fetches data, runs cycle
- EA on attached chart picks up where it left off
- Recovery logic handles any positions that survived the restart

## What happens when broker disconnects

- MT5 shows "No connection" — symbol prices freeze
- Sidecar's `copy_rates_from_pos` returns stale or empty data → logs error per pair, continues to next pair
- Heartbeat may still be written (if some pairs succeed)
- EA detects stale price feed (no ticks for >N seconds) → block new entries
- Once broker reconnects: prices resume, sidecar resumes normal cycles, EA resumes entry processing

## What happens when news event fires

- Sidecar fetches ForexFactory calendar at startup + every 4 hours
- For each signal: check news filter
- If high-impact news scheduled within `News_Window_Sec` (default 120s) of entry_bar_open_utc → defer entry by `News_Delay_Buffer_Sec` (default 5s after news passes)
- If news within delay max (default 3600s): defer
- If news beyond delay max: discard the signal entirely
- All news-related decisions logged in trade_log.csv with `reason` field

## What's not in the lifecycle

- No human intervention. The system runs end-to-end without operator action between weekly checks.
- No order modification beyond SL trail. Position sizes, take profit levels, etc. are not adjusted.
- No correlation-based decisions. Each pair's lifecycle is independent.
- No regime detection. The system trades the same logic regardless of trending/ranging/volatile market state. The DLR signal's own gates handle context implicitly.

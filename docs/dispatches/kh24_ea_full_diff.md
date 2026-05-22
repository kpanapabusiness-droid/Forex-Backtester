# KH-24 v3 vs Deployed EA — Full Diff (PR-E.1.6 contract)

This document is the **contract** for PR-E.1.6 per dispatch discipline. Code changes match this doc exactly. EA references cite line numbers in [`reference/kh24_ea/KH24_EA.mq5`](../../reference/kh24_ea/KH24_EA.mq5) — the committed MT5 EA source.

**Branch context:** PR-E.1.6 is cut from main but subsumes PR-E.1.5's three fixes (forward-ported in the first commits of this branch). All EA corrections land here in a single PR.

---

## Summary table

| § | Component | EA verdict | v3 status | Action | Confidence |
|---|---|---|---|---|---|
| A | Signal c1-c6, c8, c9 OHLC reference | bid-side single OHLC | mid OHLC | **FIX** | HIGH |
| B | Trail mechanics (price ref) | bar close (bid) | mid_close | **FIX** | HIGH |
| B | Trail mechanics (exit timing) | close-driven, next-bar-open fill | intra-bar wick on next bar | **FIX (engine ext.)** | HIGH |
| C | kijun_d1 exit | prev D1 close < prev D1 Kijun, bid-side | matches post PR-E.1.5 | KEEP | — |
| D | H1 CIR filter | bid-side single OHLC | matches post PR-E.1.5 | KEEP | — |
| E | Sizing (account ref) | `AccountInfoDouble(ACCOUNT_BALANCE)` (live, compounds) | reset-floor (ratchet up, no compounding) | **FIX** | HIGH |
| F | Exposure cap semantics | per-currency at 2 | total at 2 | **FIX (config)** | HIGH |
| G | News filter | high-impact ±3min blackout via MT5 calendar | not implemented | **DEFER** (needs calendar data) | MEDIUM |
| H | SL anchor post-fill | corrected to realised entry price | anchored to close_ask of signal bar | **DEFER** (needs driver ext.) | LOW-MEDIUM |

**Fixes in this PR:** 5 (Sections A, B (×2), E, F).
**Carried forward unchanged from PR-E.1.5:** 2 (Sections C, D).
**Deferred to future PR:** 2 (Sections G, H).

---

## Section A — Signal c-conditions OHLC reference

| Aspect | MT5 EA (ground truth) | v3 (current) | Mismatch? |
|---|---|---|---|
| File | `KH24_EA.mq5` `EvalSignal` lines 318-384 | `core/strategies/kh24/signal.py` `evaluate_kh24_signal` | — |
| Bar reference | `r[1]` = just-closed H4 bar via `CopyRates(sym, PERIOD_H4, 0, need4h, r)` with `ArraySetAsSeries(r, true)` | `df_h4` row at signal time (same conceptual bar) | No (same bar) |
| open/high/low/close field source | `MqlRates.open/high/low/close` — MT5 returns single-side OHLC (bid by broker convention) | `(open_bid + open_ask) / 2`, `mid_high`, `mid_low`, `mid_close` | **YES** |
| ATR(14) input | `WilderATR(sym, PERIOD_H4, 14, 1)` — bid OHLC | `wilder_atr(mid_high, mid_low, mid_close, 14)` | **YES** |
| Kijun(26) input | `Kijun(sym, PERIOD_H4, 26, 1)` — bid OHLC | `kijun(mid_high, mid_low, 26)` | **YES** |
| D1 series (for C8/C9) | `d[1].close` + `Kijun(PERIOD_D1, 26, 1)` + `WilderATR(PERIOD_D1, 14, 1)` — bid OHLC | Built from mid OHLC | **YES** |

**Effect:** For any non-zero spread, mid OHLC is roughly bid + half-spread (per corner). The c-conditions are mostly ratio-invariant (C2 |close-open|/ATR; C3 (close-low)/(high-low); C6 (close-close[-10])/ATR), but boundary cases shift slightly. The PR-E.2 bisect Step 1 (signal-only) produced 262 trades vs expected ~340 from the deployed EA's signal-only universe (extrapolated from published 214 post-CIR-filter), suggesting a ~25% trade-count drift attributable to signal-side OHLC reference + the rest from filters.

**Fix scope:** `core/strategies/kh24/signal.py` — replace every `mid_*` reference with the bid-side single column. Helper functions (`_build_d1_lag1_arrays`) updated to use `close_bid`, `high_bid`, `low_bid` for D1 series. ~15 LOC.

---

## Section B — Trail mechanics

| Aspect | MT5 EA (ground truth) | v3 (current) | Mismatch? |
|---|---|---|---|
| File | `ProcessExits` lines 387-470 + `ExecClose` 475-493 | `core/sim/trailing_stop.py` + `core/sim/multipair_backtester.py` | — |
| Bar-close reference price | `bc = CopyClose(sym, PERIOD_H4, 1, 1, closes)` — bid-side single | `(close_bid + close_ask) / 2.0` (mid) | **YES** (H1 from round-2 diagnostic) |
| Activation condition | `bc >= entry_price + 2.0 * atr_at_entry` | `close_price >= self.activation_close_threshold` (same formula on mid) | No on formula; **YES** on input (mid vs bid) |
| Trail level computation | `ntl = max_close - 1.5 * atr_at_entry`; ratchet up only | `proposed_trail = close_price - 1.5 * ATR`; ratchet up only | No on formula; **YES** on input |
| Max-close tracking | `if(bc > max_close) max_close = bc` (bid) | `highest_close_since_activation` updated on mid | **YES** on input |
| Exit trigger | EA line 449: `if(bc <= trail_level && trail_active) pending_close = true` — **bar close drives exit** | v3: trail updates `current_sl_price` at bar close; driver's `_check_exits` on NEXT bar fires SL when `low_bid <= current_sl_price` — **intra-bar wick drives exit** | **YES** (H2 from round-2 diagnostic) |
| Fill timing | `ExecClose` runs on first tick of NEXT bar; fills at MARKET (broker bid for a long) | v3 fills at the trail SL price on the next bar's intra-bar wick | **YES** — different price AND different bar position |
| Broker SL during trail | EA freezes broker SL at initial hard stop `entry - 2×ATR` FOREVER (never updated). Trail level is software-only. | v3 effectively replaces sl_price with the trail level (via `effective_sl`) when trail is active | Functionally similar in result, but architectural difference matters when trail exit didn't fire and a hard SL did (priority semantics differ) |

**Effect (H1):** v3 activates the trail EARLIER (mid_close > bid_close for any non-zero spread) and ratchets HIGHER on the highest-mid versus highest-bid. So v3's trail stop is closer to the entry → exits big winners short.

**Effect (H2):** EA exits when a BAR's CLOSE falls to trail level; fills the close at the next bar's open. v3 exits when the NEXT BAR's LOW (wick) touches the trail level; fills at the trail level itself. The EA's fill is typically NICER (next-bar open is usually somewhere between the close that triggered the exit and the next bar's wick low). v3 effectively pays a premium on every trail exit.

**Fix scope:**

1. **`core/sim/trailing_stop.py`:**
   - `update_all_at_close`: read `close_bid` instead of computing mid. ~3 LOC.
   - Add `pending_exit` flag on `TrailState`: set to True if `update_at_close` finds `close_bid <= current_sl_price` AND `activated == True`. Returned from `update_all_at_close` so the driver can act on it.

2. **`core/sim/multipair_backtester.py`:**
   - After `_check_exits` (intra-bar SL/TP) + before `mark_to_market`, add a step that consults `trail_manager` for any `pending_exit` flags and closes them at the CURRENT bar's `close_bid` (this matches EA semantics: trail exit detected at bar close, fills at the close price — effectively the "first tick of next bar" since the H4 bar boundary IS where the close is published). For a long, the close is at `close_bid` (selling into the bid). ~30 LOC.
   - Alternative: queue trail exits into the existing `_pending` mechanism (as a special order) and let the next-bar fill machinery handle them. Cleaner. ~50 LOC.

3. **Behaviour decision: close-at-bar-close vs queue-for-next-bar-open.** The EA's `ExecClose` runs at first tick of next bar. That's effectively the new bar's open price. Implementing as "queue close to fill at next bar's open_bid" matches EA exactly. Using the current bar's `close_bid` is close but not identical. I'll implement the queue-for-next-bar variant.

---

## Section C — kijun_d1 exit (PR-E.1.5 fix carried forward)

| Aspect | MT5 EA | v3 (post PR-E.1.5) | Match? |
|---|---|---|---|
| File | `KH24_EA.mq5` lines 456-469 | `core/strategies/kh24/exits/kijun_d1.py` | — |
| Reference price | `d[1].close` (lag-1 D1 close, bid) | `d1_close_lag1` (bid-side) | **MATCH** ✓ |
| Comparand | `Kijun(PERIOD_D1, 26, 1)` (bid) | `d1_kijun_lag1` (bid-side) | **MATCH** ✓ |
| Condition | `d[1].close < kjd1` | `d1_close_lag1[t] < d1_kijun_lag1[t]` | **MATCH** ✓ |
| Lag | shift=1 (strictly prior D1) | merge_asof backward on H4_date − 1d | **MATCH** ✓ |
| Fill | next bar open (long: bid) | `long_exit_market_price(bar)` = current bar's `close_bid` | **MISMATCH** (same as trail in Section B) |

**Note:** kijun_d1's fill timing has the same mismatch as the trail — EA queues `pending_close` and fills at next-bar open. v3 fills at the current bar's `close_bid`. Will be addressed by the trail's queue-for-next-bar engine extension; once the engine supports deferred close orders, the kijun_d1 predicate can emit one instead of an immediate ExitDecision.

**Fix scope in PR-E.1.6:** small — wire kijun_d1 to use the same queue-for-next-bar mechanism. ~10 LOC in `core/strategies/kh24/exits/kijun_d1.py` once the driver supports it.

---

## Section D — H1 CIR filter (PR-E.1.5 fix carried forward)

| Aspect | MT5 EA | v3 (post PR-E.1.5) | Match? |
|---|---|---|---|
| File | `H1CloseInRange` lines 295-313 + call site 773-785 | `core/strategies/kh24/filters/h1_cir.py` | — |
| H1 bar reference | shift=1 (just-closed H1) | `H4_index + 3h` (same physical bar) | **MATCH** ✓ |
| OHLC source | bid-side single OHLC (MT5 CopyRates) | bid-side single OHLC (`df_h1["high_bid"]` etc) | **MATCH** ✓ |
| Formula | `(close − low) / (high − low)` | identical | **MATCH** ✓ |
| Threshold | 0.28 | 0.28 | **MATCH** ✓ |
| Direction | block if `cir > 0.28` | pass if `cir <= 0.28` | **MATCH** ✓ (equivalent) |
| Doji handling | EA returns 0.5 → blocks (0.5 > 0.28) | v3 NaN → blocks | **MATCH** ✓ |

**Fix scope:** none. Carries forward unchanged.

---

## Section E — Sizing (`CalcLots`)

| Aspect | MT5 EA (ground truth) | v3 (current) | Mismatch? |
|---|---|---|---|
| File | `KH24_EA.mq5` `CalcLots` lines 229-252 | `core/sim/risk/reset_floor.py` | — |
| Account reference | `AccountInfoDouble(ACCOUNT_BALANCE)` (line 231) — live balance, COMPOUNDS with realised PnL | `self._floor` — reset-floor balance that ratchets UP on winning days but DOES NOT compound on losing days | **YES** |
| Risk percent | `RiskPercent = 1.0%` (input parameter, locked to 1.0%) | `risk_pct = 0.01` | No on value |
| SL distance | `sl_pips = (2.0 × ATR) / pip_size` — in PIPS | `sl_distance = abs(entry − sl)` — in PRICE units | Different units but reconcilable |
| Pip-value awareness | `lots = risk_amt / (sl_pips × pip_value_per_lot)`, where `pip_value_per_lot` = `SymbolInfoDouble(SYMBOL_TRADE_TICK_VALUE) × pip_size / SymbolInfoDouble(SYMBOL_TRADE_TICK_SIZE)` (returns account-currency value per pip per standard lot) | `units = (floor × rp) / sl_distance` — gives units of base currency; risk-on-SL in QUOTE currency = floor × rp | **YES** (units differ + USD-equivalence missing for non-USD-quote pairs) |
| Lot rounding | `lots = MathFloor(lots / step) * step` with `step = SymbolInfoDouble(SYMBOL_VOLUME_STEP)`; reject if below `vmin` | v3 returns raw float units (no lot rounding) | Yes, but minor |

**Effect:** Across 5+ years of cumulative PnL, the EA's position sizes COMPOUND with equity. A profitable run sizes up the next trade; a drawdown sizes down. v3's floor ratchets up on profit but never down → over multi-year stretches, v3 underestimates winning fold sizes (no compounding gain) AND in absolute USD terms misprices non-USD-quote pairs (no cross-rate conversion).

**Fix scope:**

- New module: `core/sim/risk/live_balance.py` with `LiveBalanceRisk(starting_balance, risk_pct)` exposing `risk_size(account, entry_price, sl_price)`. Sizing function reads the CURRENT account balance (`account.balance`) at call time.
- `core/strategies/kh24/kh24.py`: switch from `ResetFloorAccount` to `LiveBalanceRisk` for KH-24. Keep `ResetFloorAccount` available for L-arc work (its convention is 5ers reset-floor at 0.5% risk).
- KH24Config gets a `risk_model: str` field defaulting to `"live_balance"` for KH-24; future arcs can pick `"reset_floor"`.
- ~80 LOC including tests.

**Cross-currency simplification:** Still document that non-USD-quote pair sizing uses quote-currency risk (no USD cross-rate). For KH-24 mostly USD-quote pairs in the universe — partial coverage. This is a known limitation worth surfacing in the comparison but not a fix in this PR.

---

## Section F — Exposure cap semantics

| Aspect | MT5 EA (ground truth) | v3 (current) | Mismatch? |
|---|---|---|---|
| File | `CountCurrencyExposure` lines 273-289 + call site 753-770 | `core/sim/account.py` `Account.exposure_check` + `core/strategies/kh24/kh24.py` `KH24Config.exposure` | — |
| Per-pair cap | Implicit: `FindPosition(sym) != 0` blocks new entry on same pair | `max_concurrent_per_pair = 1` | **MATCH** ✓ |
| Per-currency cap | `CountCurrencyExposure(base_ccy) >= 2 OR CountCurrencyExposure(quote_ccy) >= 2` → block. Cap = 2 PER CURRENCY across all open positions. | `max_concurrent_per_currency = None` (DISABLED!) | **YES** |
| Total cap | EA has NO total cap — total open count can be > 2 as long as no single currency reaches the cap | `max_concurrent_total = 2` (WAY too restrictive) | **YES** |
| Counting semantics | Each open position contributes 1 to BOTH base and quote currency counts | v3 `_currency_concurrency` matches (Counter increments both base and quote per position) | **MATCH** ✓ (semantics match; just toggled wrong) |

**Effect:** v3's KH24Config caps at **2 TOTAL** open positions across the entire 28-pair universe. The EA caps at **2 PER CURRENCY** — which in practice allows many more concurrent positions. Example: with 4 open positions on EURUSD, GBPJPY, AUDCAD, NZDCHF, no single currency has more than 1 open position; EA accepts; v3 blocks the 3rd and 4th.

This explains a large fraction of the trade-count drift: v3 blocks signals that EA would accept due to the wildly stricter total cap.

**Fix scope:** one-line change in `core/strategies/kh24/kh24.py`:

```python
exposure: ExposureRules = field(
    default_factory=lambda: ExposureRules(
        max_concurrent_total=None,            # EA has no total cap
        max_concurrent_per_pair=1,            # EA implicit
        max_concurrent_per_currency=2,        # EA ExposureCap=2
    )
)
```

`core/sim/account.py` exposure-check logic itself is correct; only the config is wrong.

---

## Section G — News filter (DEFERRED)

| Aspect | MT5 EA (ground truth) | v3 (current) |
|---|---|---|
| File | `IsNewsBlackout` lines 499-528 + call site 538-552 | (none) |
| Mechanism | Queries MT5 economic calendar via `CalendarValueHistory` for events ±`NewsBufferMins=3` minutes from `TimeCurrent()` | n/a |
| Filtered events | `CALENDAR_IMPORTANCE_HIGH` only | n/a |
| Currency match | Event affects either base or quote currency of the symbol | n/a |
| Behaviour on hit | Sets `news_delayed=true`; entry RETRIES on next tick (not cancelled) | n/a |

**Effect:** EA suppresses entries within 3 minutes of high-impact news. v3 has no equivalent → v3 fires signals during news windows that the EA blocks.

**Fix scope:** Substantial — needs economic-calendar data integration. Either:
- Pre-fetch a static calendar dataset (e.g. via `forexfactory` CSV, dukascopy, or MT5 calendar export) and bake into v3's data layer
- Run-time API call to a calendar service (out of scope — no internet during deterministic backtests)

**Decision: DEFER to a future PR.** Document in this PR's comparison.md that v3 makes trades the EA would have blocked during news; expect ROI to deviate downward in folds with high news activity (especially NFP days, FOMC meetings).

---

## Section H — SL anchor post-fill (DEFERRED)

| Aspect | MT5 EA (ground truth) | v3 (current) |
|---|---|---|
| File | `ExecEntry` lines 534-616 — line 595: `corrected_sl = entry_price - SL_ATR_MULT * atr` after fill | `core/strategies/kh24/kh24.py` strategy callable — line ~205: `sl_price = entry_proxy − 2×ATR` using `close_ask` of current bar |
| When SL is computed | EA: pre-fill estimate from current ask (`ask - 2*ATR`), then POST-FILL corrected to `entry_price - 2*ATR` once the broker confirms the fill price | v3: pre-computed on the signal bar before fill, never corrected |
| Drift | Small — typically (next-bar open_ask) ≈ (current close_ask) on FX majors during normal hours; bigger gap during news / weekend opens | Same drift, NOT corrected post-fill |

**Effect:** v3's SL is anchored to a proxy entry price (current bar's close_ask) rather than the realised entry price (next bar's open_ask). For a typical 1-pip overnight gap, the SL distance is off by ~1 pip. Across 168+ trades, cumulative ~ small fraction of a percent.

**Fix scope:** Driver extension to support deferred SL setting. The strategy emits an Order with `sl_atr_mult=2.0` (no absolute price); driver fills at next-bar open, then sets `sl_price = realised_entry − sl_atr_mult × atr_at_entry`. Requires changes to `Order` dataclass + driver `_fill_pending_entries`. ~40 LOC.

**Decision: DEFER to a future PR.** The drift is small (~1-2pp aggregate per fold worst case) compared to the headline issues in Sections A, B, E, F. If Sections A/B/E/F together close the verdict gap, Section H stays open as a polish issue.

---

## Implementation order

1. **Commit this diff doc** (the contract).
2. **Section F (exposure cap)** — one-line config change. Fast. Tests should mostly pass already.
3. **Section A (signal bid OHLC)** — switch references in `signal.py`. Existing signal tests should still pass (synthetic data has bid==ask).
4. **Section E (live-balance sizing)** — new `LiveBalanceRisk` module, swap in `kh24.py`. Tests for live-balance compounding behaviour.
5. **Section B (trail mechanics)** — engine extension for deferred close orders. Trail uses bid_close; emits "pending exit" flag; driver queues for next-bar open fill. Largest change in this PR.
6. **Test sweep** — all PR-E.1 / PR-E.1.5 tests still pass.
7. **Re-run Mode A anchor** — apply verdict.
8. **If PASS:** Mode B + final docs + open PR. **If HALT:** diagnostic_round_3.

---

## Open questions / pending chat decision

None. The EA is unambiguous on Sections A, B, E, F. Sections G and H are deferred with clear rationale. Proceeding.

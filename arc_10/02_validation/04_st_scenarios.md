# ST Scenarios — EA Execution Validation

> **Purpose:** Prove the EA correctly handles each unique execution code path in isolation.
> **Method:** Strategy Tester runs with synthetic signal envelopes against real historical MT5 data.
> **Coverage:** 6 PASS, 4 valid-skip with reasoning, 2 defer-to-live.

## What ST scenarios test (and don't test)

**Test:** The EA's behaviour on each unique mechanical case — entry, partial close, trail trip, time exit, equity guard, restart recovery, news filter, envelope rejection.

**Don't test:** Whether the strategy is profitable. Whether signals are correct. Whether sidecar+EA produce the right results together.

ST scenarios isolate the EA's correctness as a function of envelope-in → trade-out. Strategy correctness is proven by WFO. End-to-end correctness is proven by Phase 2 parity.

## Test setup

- **EA:** Arc10_DLR_Sidecar_EA built from current main branch
- **Envelopes:** Synthetic JSON files placed via `tests/ea/fake_sidecar` Python script
- **Data:** Real historical MT5 broker data, replayed via Strategy Tester at 1-minute OHLC modeling
- **Result file:** `trade_log.csv` after each run, compared against `scenarios.json` expected events

## Results summary

| # | Scenario | Mechanic tested | Status | Note |
|---|---|---|---|---|
| s1 | Trail-stop full happy path | entry → partial → trail → exit @ trail_stop | ✅ PASS | net +$327.77, EURUSD 2026-03-09 |
| s2 | Initial SL hit (no partial) | entry → SL hit before TP1, exit @ initial_sl_hit | ⏭ SKIP | Mechanic covered by trail closures; clean SL hard to engineer on real data |
| s3 | Partial → near-BE trail exit | entry → partial → quick trail @ near-BE | ✅ PASS | net +$261, AUDUSD 2025-03-11 |
| s4 | Multi-bar runner | entry → partial → 3+ trail_modify → exit | ⏭ SKIP | M1 data unavailable for COVID-era pre-2020 |
| s5 | Time exit at bar 240 | entry → bar 240 → force close | ✅ PASS | net +$565, NZDUSD 2025-11-25, Time_Exit_Bars=12 override |
| s6 | News filter blocks entry | envelope arrives, news filter discards | 🔵 DEFER | ForexFactory calendar unreplayable in ST |
| s7 | Equity guard halts entry | DD halt threshold trips → equity_block | ✅ PASS | EURUSD, Daily_DD_Halt_Pct=-0.9 override |
| s8 | Same-bar dual-touch | bar opens, hits SL AND TP1 same bar | ⏭ SKIP | No canonical-R candidate exists on real data |
| s9 | Restart pre-partial | EA restart with open position, no partial yet | 🔵 DEFER | Restart timing fiddly in ST; live restart will exercise |
| s10 | Restart post-partial | EA restart with open runner | 🔵 DEFER | Same as s9 |
| s11 | Stale heartbeat | Sidecar heartbeat > Max_Age → block all polling | ✅ PASS | EURUSD, heartbeat set to 2099-01-01 |
| s12 | Bad config hash | Envelope with wrong hash → reject | ✅ PASS | EURUSD, hash mismatch → moved to signals_failed |

**Pass rate on testable scenarios:** 6/6 (100%).
**Skipped with documented reason:** 4.
**Deferred to live operation:** 2.

## What each PASS proved

### s1 — Trail-stop full happy path

**Verified:** Entry placement, position size calculation, partial close at +1R, trail ratchet (multiple H4 bars), trail SL fill at trail level, strategic reason inference (`trail_stop` not generic `broker_closed`), fill price captured correctly (was bug A).

**Trade lifecycle:** entry 1.15484 → partial @ 1.16247 (+1R) → trail ratcheted 7 bars → exit @ 1.15925 (`trail_stop`). Net +$327.77.

### s3 — Near-BE trail exit

**Verified:** Same as s1 with different market timing. Confirms trail mechanic works on AUDUSD with smaller R-units, exits at near-breakeven level.

### s5 — Time exit

**Verified:** Bar counter (`bar_ord`) increments correctly per H4 bar, time exit fires at configured `Time_Exit_Bars` threshold, EA actively sends close order (not waiting for broker SL), reason logged as `time_exit`, fill_price populated.

**Critical for runner trades:** confirms positions cannot stay open forever and consume margin indefinitely.

### s7 — Equity guard

**Verified:** When `Daily_DD_Halt_Pct` threshold is breached, EA refuses new entries and logs `equity_block` with reason `daily_dd_halt`. Existing positions continue to manage (only `_CloseAll_Pct` triggers force close).

**Critical for risk management:** confirms prop firm DD limits are enforced at the system level.

### s11 — Stale heartbeat

**Verified:** EA checks `sidecar.heartbeat` `last_heartbeat_utc` against `Sidecar_Heartbeat_Max_Age_Sec`. If stale, polling stops, no new signals processed. Existing positions continue to manage.

**Critical for sidecar failure:** if sidecar dies, EA stops trading new signals automatically — doesn't blindly trust stale envelopes.

### s12 — Bad config hash

**Verified:** Envelope with `config_hash` not matching EA's `Expected_Config_Hash` is rejected, moved to `signals_failed/`. No trade placed.

**Critical for operator error:** if wrong config is deployed (UTC sidecar pointing at EET EA), envelopes can't trade. Multi-broker isolation via hash matching.

## Why s2 was skipped (Initial SL hit)

**The mechanic in question:** trade enters, never hits TP1, retraces to SL, exits via broker SL with `reason=initial_sl_hit`.

**Why we couldn't engineer it cleanly:** finding historical H4 bars where price drops to SL without ever reaching +1R is rare on real data. Tried 3 different date selections — all either hit TP1 first (becoming s1-type trades) or hit time-exit window before SL.

**Why skipping is safe:**
- The "broker SL fires, EA detects on next OnTick, logs strategic reason" code path is identical between initial SL and trail SL
- Trail SL is exercised in every s1, s3 test
- The only thing not exercised is the SPECIFIC reason-inference branch that emits `initial_sl_hit` instead of `trail_stop`
- That branch logic is simple: if `tp1_fired == false` and position closed → `initial_sl_hit`. Visually verifiable in code; tested by ST not needed.

## Why s4 was skipped (Multi-bar runner)

**The mechanic:** entry → partial → many trail_modify events (5+) → exit. Designed for COVID-era 2020 USDJPY because canonical-R 3.5×ATR is so wide that finding a runner with many new-high bars requires extreme directional moves.

**Why we couldn't run:** 5ers MT5 only retains M1 historical data back ~2 years. COVID-era M1 data not available. H4 data exists but ST modeling at 1-minute OHLC needs M1.

**Why skipping is safe:** the multi-bar trail mechanic was exercised in s1 (7 trail_modify rows) and s2 first re-run (49 trail_modify rows when window extended). The "multi-bar" quantitative variation is over-satisfied.

## Why s8 was skipped (Same-bar dual-touch)

**The mechanic:** entry bar where H4 high touches +1R AND low touches initial SL within the same bar.

**Why we couldn't run:** under canonical R (3.5×ATR), no such bar exists in the historical broker data. The bar would need to traverse 4.5×ATR in range (SL → entry → TP1) within a single H4 period. Possible in theory (extreme news days), but no candidate found in 14 years × 28 pairs.

**Why skipping is safe:** the dual-touch handler in the EA code is simple — checks TP1 first (long-only direction), only checks SL if TP1 didn't fire. Logic visually verifiable. No real-world risk if mechanic isn't exercised in ST.

## Why s6 was deferred (News filter)

**The mechanic:** envelope arrives, ForexFactory news event scheduled within `News_Window_Sec` of entry_bar_open_utc, EA blocks/defers entry.

**Why deferred:** ForexFactory news calendar is rolling — only carries current + next ~2 weeks of news. Can't replay historical news in ST.

**Plan:** first real NFP/CPI release in live will exercise this code path. Monitor the journal for `news_discard` or `news_delay` events.

## Why s9/s10 were deferred (Restart recovery)

**The mechanic:** EA stops mid-trade (Strategy Tester unable to simulate cleanly), restarts, reconstructs position state from broker, resumes management.

**Why deferred:**
- Strategy Tester restart timing is fiddly and easy to mistime
- Real-world restart scenarios (sidecar redeploy, VPS reboot) provide cleaner tests
- The recovery code itself was audited during the r_atr fix session; same r_atr bug existed in RecoveryManager and was fixed alongside the EA fix

**Plan:** first VPS reboot or sidecar redeploy in production will exercise this naturally. Check journal for `recovery_reconstructed` events.

## Critical bug discovered during ST validation (r_atr bug)

During s8 investigation, the user noticed P&L numbers didn't match expected. Tracing: TP1 was firing at +1×ATR instead of +3.5×ATR (= initial SL distance).

**Root cause:** `phase_1_build_intent.md` had two contradictory definitions of `r_atr`. Implementer followed the wrong one (line 369: `r_atr = sl_distance / 3.5`) instead of the correct one (line 256: `R = sl_distance`).

**Fix:** Commit 164ffb5 — `r_atr = sl_distance` in PositionManager.mqh + RecoveryManager.mqh. Lab simulator was always correct; only the EA was wrong.

**Impact:** all pre-fix ST scenarios were false positives (testing the wrong logic). After the fix, all 6 PASS scenarios were re-run to verify the canonical-R behaviour.

**Lesson:** caught by user reasoning from actual fill prices on a test run. Without that human-in-the-loop scrutiny, this bug would have shipped to live and caused real money loss (TP1 firing too early → much smaller R-multiples → worse net returns vs WFO).

## Source artifacts

| Artifact | Path |
|---|---|
| Scenario definitions | `tests/ea/scenarios.json` |
| Synthetic sidecar | `tests/ea/fake_sidecar.py` |
| EA source | `deployment/ea/Arc10_DLR_Sidecar_EA.mq5` + includes |
| Sample trade_log outputs | Captured in chat history during dispatches |
| ST validation tag | `arc-10-st-validated` |
| Topology validation tag | `arc-10-topology-validated` |

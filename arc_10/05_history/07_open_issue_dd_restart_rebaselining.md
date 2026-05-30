# OPEN-001 — EA Restart Re-baselines DD Reference

> **Status:** ✅ RESOLVED — 2026-05-30
> **Discovered:** 2026-05-30
> **Resolved:** 2026-05-30 (same day; fix prioritized on first weekend after go-live)
> **Severity:** Medium (real silent-failure risk; bounded by restart probability)
> **Fix commit:** `b386287` — "OPEN-001: operator-set static total-DD floor (fail-loud, no live capture) (#242)" on `main`
> **Owner:** Keanu

---

## The bug

The total-DD floor (`g_arc_eq_total_floor`) was snapshotted at every `OnInit()` from live equity (`AccountInfoDouble(ACCOUNT_EQUITY)`, `EquityGuards.mqh:100`). It was neither an input nor read from disk. Any event that re-ran `OnInit` — Windows reboot, MT5 crash + relaunch, manual recompile/reattach — re-baselined the floor to whatever equity was at that moment.

**Failure mode:** if the EA re-initialized while the account was drawn down, the floor dropped with equity, dragging the EA's halt threshold below the broker's static termination point. The broker terminates against the static initial balance regardless of EA state, so the account could hit the broker limit before the EA's (now-sunken) halt ever fired. Protection fails silently.

**Worked case (the one we hit live):** 5ers floor = $10,000 (static). Account drew to $9,770 on day 1. Under the old code, a restart here would have re-baselined the floor to $9,770 and sunk the halt with it. Under the fix, the floor stayed pinned at $10,000 (verified — see deployment evidence).

## The fix

Total-DD floor is now **solely an operator-set input**, `Initial_Equity_Floor` (default `0`). The EA uses it directly; **no live-equity capture anywhere.**

- **Fail-loud guard:** if `Initial_Equity_Floor < 5000` (sentinel/implausible) the EA refuses to trade, fires `Alert()`, journals `ERROR FLOOR_FAIL` + `source=sentinel-fail`, and the on-tick path short-circuits (no div-by-zero). It never falls back to live capture and never hardcodes a balance.
- **Scale-up = manual.** Broker scale-up (e.g. $100k → $125k) is handled by the operator editing the input and reattaching. No separate handler, no confirm flag, no JSON state file — the act of setting the input is the confirmation. Scale-ups occur ~once per several months; manual is appropriate.
- **No persisted state file.** Rejected deliberately. MT5's saved profile re-persists the input across terminal restart (verified live), and fail-loud covers the only path where it wouldn't (recompile-to-defaults). One source of truth, one code path.

> **Note on the superseded design.** An earlier draft of this issue (formerly tracked at `arc_10/04_runbook/10_open_issue_dd_restart_rebaselining.md`, describing the OPEN / fix-deferred state) proposed a persisted `equity_state.json` plus a `Manual_Scale_Up_Balance` / `Manual_Scale_Up_Confirm` input pair. That JSON-state-file approach was **rejected** in favour of the operator-set input + fail-loud above — simpler, one source of truth, no state-file corruption path. This document is the reconciliation of that earlier draft to the resolved reality, relocated to `05_history/` now that the issue is closed.

### Daily-DD — corrected, not changed

The daily day-start reference is unchanged in behavior: a fixed equity snapshot captured at `OnInit` and re-captured once per EET day rollover, held constant between. Daily-DD measures loss from that fixed point. A mid-day restart re-snapshots (accepted; daily risk is bounded to one day by the natural rollover) — it is NOT persisted and NOT live-tracked. The only edit was correcting PR/comment wording that had described it as "floating/equity-based," phrasing that could have invited a future incorrect "fix." Threshold (`0.045`), EET rollover, and floating loss computation were untouched.

## Correction to the original risk estimate

The original analysis estimated ~1–3% probability of a materially bad outcome over 6 months, partly on the premise that "~2 NSSM restarts/week" were re-firing the EA. **That premise was wrong: NSSM wraps the Python sidecar, not the EA terminal (confirmed by the EA developer).** The EA's `OnInit` re-fires only on terminal restart, crash auto-restart, or manual reattach/recompile — rarer and operator-correlated. Actual pre-fix risk was lower than documented. The fix was cheap enough to ship regardless.

## Deployment evidence (2026-05-30, both terminals, markets closed)

- Source diff: exactly 3 files changed (`Arc10_DLR_Sidecar_EA.mq5`, `EquityGuards.mqh`, `HeartbeatWriter.mqh`); all trading-logic includes byte-identical (SAME) — confirmed floor-fix-only, no incidental trading-logic changes.
- Recompiled per terminal, 0 errors. Both EAs halted on first reload (default floor `0` → fail-loud fired the alert) — protection confirmed working before any floor was set.
- **FundedNext:** `equity init: floor=100000.00 source=input`; daily day-start `100000.00` init-snapshot; 0 positions.
- **5ers:** `equity init: floor=10000.00 source=input`; daily day-start `9770.07` init-snapshot (real current equity, account drawn down day 1); 1 position (USDCHF) recovered cleanly.
- **Persistence test (FundedNext, flat):** full MT5 close + reopen → returned `floor=100000.00 source=input`, no FLOOR_FAIL. **MT5 profile persists the input across terminal restart.** Re-entry not required on restart.
- **Alert reach:** the fail-loud `Alert()` was observed in MT5 on both terminals during recompile — confirmed visible to operator at the terminal.

## Test coverage

`scenarios.json` extended to 19 scenarios; `pytest tests/ea/test_fake_sidecar.py` 23/23 pass.
- s13 restart → floor unchanged
- s14 drawdown + restart → floor unchanged (the bug case)
- s15 input 100000 → 125000 → floor updates (scale-up path)
- s16 scaled floor + restart → keeps 125000
- s17 floor = 0 → fail-loud, refuses to trade + alerts
- s18 daily day-start fixed across ticks + EET-rollover re-snapshot
- s19 mid-day restart → daily re-snapshots (not persisted, not live)

(Strategy Tester GUI runs are manual; specs authored. pytest envelope layer green.)

## Residual / follow-ups (not blocking; logged)

1. **Sidecar does not yet read the EA `ea.heartbeat` `status` field.** The fail-loud condition writes `status: halted_floor_unset` to the heartbeat, but `sidecar.py` doesn't consume it — so on an *unattended* VPS a floor-fail halt is currently signalled only by the in-terminal `Alert()` + journal token, not pushed to the operator. **Follow-up:** wire the always-up sidecar to grep the journal `FLOOR_FAIL` token or read the heartbeat status, to get an unattended alert path. Tracked in `04_open_items.md`.
2. **Profile-persistence is now relied upon** for restart-survival. Verified once (FundedNext). If MT5 behavior changes or a hard crash ever fails to retain the profile, fail-loud still protects (EA halts rather than re-baselines) — the cost is a manual floor re-entry, not a safety failure.

## Closeout

Bug eliminated. Both live accounts deployed with correct static floors, restart-safe, fail-loud verified. Daily-DD semantics pinned. Risk note corrected. Doc reconciled from the stale `04_runbook/10` draft (OPEN / fix-deferred) to this RESOLVED record in `arc_10/05_history/` for the record.

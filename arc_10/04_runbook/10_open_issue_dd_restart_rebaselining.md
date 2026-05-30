# Open Issue — EA Restart Re-baselines DD References

> **Status:** OPEN — known risk, fix deferred
> **Discovered:** 2026-05-30 via `ea_dd_calculation_audit.md`
> **Severity:** Medium — real risk, but bounded by restart probability
> **Owner:** Keanu
> **Target fix:** before risk ramp to 0.50% on funded account, OR if frequency of EA restarts becomes operationally significant, OR **immediately upon first broker scale-up event** (whichever comes first)
> **Related docs:**
> - `arc_10/05_history/05_pre_mortem.md` (failure mode #7 — partially related but inaccurate on mechanism)
> - Audit report: `ea_dd_calculation_audit.md` (out of repo, on local)

---

## What is the issue

The Arc 10 EA captures its drawdown reference values at `OnInit()` — i.e. every time the EA initializes. Two values are affected:

1. **`initial_equity_floor`** — used as the reference for total DD calculations. Captured once at OnInit. **Not persisted to disk.**
2. **`day_start_equity`** — used as the reference for daily DD calculations. Captured at OnInit and refreshed at each broker-day boundary (EET for FundedNext, UTC for 5ers). **Not persisted to disk.**

When the EA is re-initialized (Windows reboot, NSSM service restart, MT5 crash + relaunch, manual recompile + reattach), both values are recaptured **from current equity**, regardless of what they were previously.

### Why this matters

The broker (5ers and FundedNext) enforces drawdown limits against the **static initial account balance** — set when you opened the account or last scaled it up. The broker doesn't care if your EA restarted.

So if the EA restarts mid-drawdown, the EA's "100% headroom" view diverges from the broker's "drawn-down headroom" reality.

### Worked example

Initial balance: **$100,000**.
Broker DD limit: **10% of initial** = $90,000 floor (static until scale-up).
EA Total DD halt threshold: **8% of initial_equity_floor**.
EA Total DD CloseAll: **8.5% of initial_equity_floor**.

**Normal operation:**
- EA captures `initial_equity_floor = $100,000` at OnInit.
- EA halts at equity $92,000 (8% drawdown).
- Broker terminates at $90,000 (10% drawdown).
- EA halt fires 2pp before broker termination. Safety margin works as designed.

**Drawdown + restart scenario:**
1. Account starts at $100,000.
2. Account drifts to $94,000 (6% drawdown from start).
3. EA / VPS / sidecar restart event occurs (NSSM auto-restart, Windows update reboot, manual recompile, etc.).
4. EA re-initializes. `OnInit()` captures `initial_equity_floor = $94,000` (current equity).
5. EA now thinks: "8% halt threshold = $94,000 × 0.92 = $86,480."
6. **EA's halt threshold ($86,480) is now BELOW the broker's account-termination point ($90,000).**
7. If drawdown continues from $94,000, the account hits the broker's $90,000 termination point before the EA's $86,480 halt fires.
8. **Account terminated. EA never halted.** Protection failed silently.

The same scenario applies to daily DD, though the impact is smaller because daily DD has a natural reset at each broker-day boundary regardless of EA restart.

### Why this wasn't caught earlier

- The pre-mortem doc (`05_history/05_pre_mortem.md` #7 "Equity calculation drift") flagged equity-related risk but described it as caching at H4 boundaries (incorrect mechanism) and as drift of <0.5% (incorrect magnitude). The actual issue is much larger and structurally different.
- ST scenarios test EA behavior in normal operation. Restart-during-drawdown wasn't an ST scenario (it's now documented as a gap to add).
- Backtests don't model EA restarts (the backtester doesn't have a notion of "the EA crashed").

---

## How likely is this to happen

**Inputs to the probability estimate:**

- EA restart frequency: historically ~2 NSSM service restarts per week per broker in normal operation (per `sidecar_state.json` restart_count). Most of those happen during low-volatility periods (weekends, off-hours) when an active drawdown is unlikely.
- Drawdown frequency: based on backtest, ~20% of trading days have any drawdown > 1% from day-start. Days with drawdown > 5% from initial are <2%.
- Conditional probability: P(EA restart | active drawdown > 5%) is harder to estimate without live data.

**Rough estimate over a 6-month live operation period:**

| Scenario | Probability |
|---|---|
| EA restarts at least once during the period | ~100% |
| Account is in any drawdown > 3% at some point | ~80% |
| Account is in drawdown > 6% at some point | ~30% |
| **EA restart happens DURING a > 6% drawdown** | ~5-10% |
| Restart re-baselines AND further drawdown breaches broker limit before next operator intervention | ~1-3% over 6 months |

So: **bad outcome is unlikely but not negligible.** Not "happens every month" — more like "happens during one bad week per year of operation, IF drawdown timing aligns with restart timing."

This is a real risk that needs fixing, but it's not on fire.

---

## Why it's not being fixed right now

Three reasons:

1. **Currently on demo.** Restart re-baselining on a demo account costs nothing. The risk only materializes on a real-money account.

2. **Risk ramp is conservative.** Week 1-3 use 0.20% → 0.30% → 0.50% risk per trade. At those levels, accumulated drawdown is much slower, so the time window in which a restart could land mid-drawdown is small.

3. **Other priorities take precedence right now.** First weeks of live trading need observation and weekly review, not EA modifications. Modifying the EA introduces new bugs; deferring lets the current validated state run as-is.

**When this changes:** before risk scales to 0.50% on funded capital (Week 4+ of live), OR if frequency of EA restarts becomes operationally significant during the first 4 weeks (e.g. > 5 restarts in a week consistently).

---

## Manual mitigations until fixed

Four compensating controls operator can apply:

### Mitigation 1: Watch restart_count weekly

The Sunday weekly check (per `arc_10/04_runbook/08_sunday_weekly_check.md`) includes restart_count comparison week-over-week.

**Add to weekly review prompt:**
> If restart_count increased by > 2 in the past week AND account is currently in drawdown > 3%, manually verify EA's `initial_equity_floor` matches the broker's initial balance (or last scale-up balance), not current equity.

How to verify: read the EA's journal log around EA init, look for the line that captures equity floor. It should match the broker's account-history initial balance, not the current floating equity at restart time.

### Mitigation 2: Manual halt override

If a restart happens during a known drawdown, operator can manually halt the EA via MT5:
- Right-click chart → Expert Advisors → Remove
- Investigate the floor capture
- If wrong, fix manually and reattach with corrected inputs

This is reactive and requires operator vigilance. Not ideal but works until the proper fix is in.

### Mitigation 3: Pre-Challenge floor verification

When the FundedNext Challenge account is purchased and the EA is attached:
- Note the EA's reported `initial_equity_floor` from the journal log at attach time
- Verify it matches the Challenge starting balance (e.g. $100K)
- Save this verification in `arc_10/06_live_reports/setup_verification.md`

This becomes the canonical floor for the lifetime of that account. If the EA ever shows a different floor without an operator-confirmed scale-up event, that's the bug surfacing.

### Mitigation 4: Handling broker scale-up events (detach + reattach procedure)

When the broker scales the account up (e.g. promoted from $100K to $125K, broker DD limit moves from $90K to $112.5K), the EA's `initial_equity_floor` does NOT automatically update. It still references the original $100K. The EA's halt threshold becomes:

- Old (correct, but stale): 8% of $100K = halt at $92K → 2pp safety margin before broker's $90K termination
- After scale-up (now wrong): 8% of $100K = halt at $92K → but broker termination is now at $112.5K → **EA halt fires AFTER broker termination already happened**

The EA is now ineffective at protecting the new account size. Operator MUST update the floor manually.

**Procedure when broker scale-up event occurs:**

1. **Confirm the scale-up.** Verify in broker portal AND MT5 that the account balance has updated to the new amount (e.g. $125K).

2. **Wait for weekend market closure.** Do NOT do this with open positions. Friday 22:00 UTC → Sunday 22:00 UTC.

3. **Verify no open positions.** Check MT5 Trade tab; should be empty.

4. **Note the current account balance.** Record exact value (e.g. $125,000.00). This will become the new floor.

5. **Take a screenshot** of the journal entry showing the OLD floor (so you have a record of what changed):
   - Right-click chart → Expert Advisors → Properties (verify current inputs)
   - Or look at recent journal entries for `equity init: floor=...` lines

6. **Detach the EA.** Right-click chart → Expert Advisors → Remove. Verify journal shows clean detach.

7. **Reattach the EA.** Drag the EA from Navigator onto the chart. Use the SAME input parameters as before (Risk_Per_Trade, magic number, all paths). The reattach triggers a fresh `OnInit()`.

8. **Verify new floor in journal.** Expected log line: `equity init: floor=125000.00 day_start_utc=...`. Confirm it matches the new balance.

9. **Record the event** in `arc_10/06_live_reports/setup_verification.md`:

```markdown
## Scale-up event: <YYYY-MM-DD>

- Broker: <FundedNext / 5ers>
- Old account balance: $<old>
- New account balance: $<new>
- Old EA floor: $<old> (from journal at <timestamp>)
- New EA floor: $<new> (from journal at <timestamp>)
- Procedure: detach + reattach during weekend closure
- Verified by: <operator>
- Notes: <any anomalies>
```

10. **Verify operationally** for first H4 cycle after market reopen — confirm sidecar still cycling, EA still alive, heartbeats fresh.

**Fragility warning:**

Mitigation 4 only fixes the floor at the moment of reattach. If the EA subsequently restarts for any reason (Windows reboot, NSSM service restart, MT5 crash + auto-relaunch), it will recapture whatever current equity is at THAT moment — which won't be $125K anymore if the account has drifted up or down.

**Example failure mode:**
- Day 1: scale-up to $125K → operator reattaches → floor = $125K ✓
- Day 14: account drifts to $128K (profits)
- Day 15: VPS reboots for Windows update → EA OnInit → floor = $128K (now wrong direction — too high, halts too aggressively)
- Or: Day 30: account drifts to $122K (drawdown) → restart → floor = $122K (wrong direction — halts too late)

So Mitigation 4 must be REPEATED whenever a restart happens after a scale-up, until the proper code fix lands.

**This is why broker scale-up is a definite trigger for fixing the bug.** Once you've scaled even once, the cost of manual mitigation per restart event compounds quickly. See "Decision criteria for fix now" section below — scale-up is upgraded from "trigger to consider" to "trigger to immediately schedule."


---

## What the fix looks like (for when it's time)

### Code changes required

**File 1: `deployment/ea/include/EquityGuards.mqh`** (or wherever equity init lives)

Add state persistence:

```mql5
// Pseudocode — actual MQL5 will differ slightly

// New input parameter:
input string Equity_State_Path = "Arc10_5ers\\equity_state.json";

// New struct:
struct EquityState {
    double initial_equity_floor;
    datetime initial_floor_captured_at;
    double broker_scale_up_balance;  // null = never scaled up
    datetime broker_scale_up_at;
};

void InitializeEquityFloor() {
    EquityState state;
    if (LoadEquityState(Equity_State_Path, state)) {
        // Restored from disk — use persisted floor
        initial_equity_floor = state.initial_equity_floor;
        Print("Restored equity floor from state: ", initial_equity_floor);
    } else {
        // Fresh deploy — capture current
        initial_equity_floor = AccountInfoDouble(ACCOUNT_EQUITY);
        SaveEquityState(Equity_State_Path, initial_equity_floor);
        Print("Captured fresh equity floor: ", initial_equity_floor);
    }
}
```

**File 2: New input parameter for manual scale-up event**

```mql5
input double Manual_Scale_Up_Balance = 0;  // 0 = no scale-up event
input bool Manual_Scale_Up_Confirm = false;  // require explicit confirmation
```

When operator confirms a broker scale-up (e.g. promoted from $100K to $125K), they:
1. Set `Manual_Scale_Up_Balance = 125000`
2. Set `Manual_Scale_Up_Confirm = true`
3. Reattach the EA
4. EA detects the change, updates `initial_equity_floor = 125000`, persists to state file
5. Operator sets `Manual_Scale_Up_Confirm = false` again to avoid accidental re-trigger

**File 3: `deployment/ea/include/RecoveryManager.mqh`**

Already handles position recovery. Extend to call `InitializeEquityFloor()` from the recovery flow.

### Schema changes

**New state file:** `Arc10_<broker>/equity_state.json`

```json
{
    "initial_equity_floor": 100000.00,
    "initial_floor_captured_at_utc": "2026-05-29T12:00:00Z",
    "broker": "FundedNext",
    "scale_up_events": []
}
```

After a scale-up:

```json
{
    "initial_equity_floor": 125000.00,
    "initial_floor_captured_at_utc": "2026-08-15T09:00:00Z",
    "broker": "FundedNext",
    "scale_up_events": [
        {
            "from": 100000.00,
            "to": 125000.00,
            "operator_confirmed_at_utc": "2026-08-15T09:00:00Z",
            "reason": "broker_promotion"
        }
    ]
}
```

### Tests required

New ST scenarios to add to `tests/ea/scenarios/`:

| Scenario | Setup | Expected |
|---|---|---|
| s13 | EA initialized, captures floor, restarted → floor restored from disk | Floor unchanged after restart |
| s14 | EA initialized at $100K, account drifts to $94K, EA restarted | Floor still $100K, not $94K |
| s15 | Operator confirms scale-up to $125K | Floor updates to $125K, persists |
| s16 | EA at $125K floor, current equity $120K, restart | Floor restored as $125K, not $120K |
| s17 | State file corrupt / missing on EA init | Falls back to live capture + logs warning |

### Estimated effort

- MQL5 code: ~80 lines (state file IO, init logic, scale-up handler)
- State file schema: ~5 lines JSON
- ST scenarios: 5 new × ~30 lines each = ~150 lines
- Documentation update: pre-mortem #7 correction + new `arc_10/05_history/...` entry
- PR review + redeploy: half a day operator time

**Total: ~1-2 days CC work + ~3 hours operator time for redeploy + soak.**

### Compatibility / safety

- Fully backwards compatible: if state file doesn't exist (fresh deploy), behaves identically to current EA
- No effect on entry / exit / position logic — purely equity-reference tracking
- New ST scenarios validate restart behavior end-to-end
- Phase 2 parity not affected (sidecar doesn't compute equity references; this is purely EA-side)

---

## Decision criteria for "fix now"

Trigger the fix dispatch when ANY of these become true:

1. **EA has restarted > 5 times in a single week** without operator-initiated cause. Frequency suggests environmental instability that compounds the risk.

2. **Risk ramp is about to scale to 0.50%** on a funded account. At full risk, position sizes are 2.5× Week 1 levels and drawdown velocity is higher.

3. **Account profit accumulation triggers a broker scale-up event.** ★ **HIGHEST PRIORITY TRIGGER.** ★ The static reference becomes structurally wrong at scale-up — every restart event afterward requires manual Mitigation 4 procedure, which is operationally fragile. Schedule fix immediately upon first scale-up notification from broker.

4. **First incident occurs.** Either an actual close call where drawdown approached broker limit during/after a restart, OR a near-miss where the operator notices a floor mismatch in the weekly check.

5. **Operator decides the risk warrants it** — even without trigger above, this can be prioritized any time other work is paused.

---

## Tracking

| Field | Value |
|---|---|
| Issue ID | OPEN-001 (first formal open-issue under this convention) |
| Discovered | 2026-05-30 |
| Discovery method | EA DD calculation audit (CC dispatch, no code changes) |
| Severity | Medium |
| Frequency estimate | ~1-3% probability of materially bad outcome over 6 months |
| Mitigation | Manual via weekly check + restart_count monitoring + operator vigilance |
| Fix complexity | Medium (~1-2 days CC + 3hr operator) |
| Fix urgency | Before risk = 0.50% on funded, OR upon trigger criteria above |
| Related issues | Pre-mortem #7 inaccurate mechanism description (cosmetic doc fix, separately) |

---

## What to do with this doc

- **Sits in repo at `arc_10/05_history/06_open_issue_dd_restart_rebaselining.md`** (proposed location — alternatively a new `arc_10/05_history/open_issues/` subfolder if more open issues accumulate).
- **Referenced from `arc_10/05_history/04_open_items.md`** as a tracker entry.
- **Reviewed monthly** during the first month of live trading, weekly review thereafter.
- **Closed when fix lands** — at which point the doc gets a "Resolved" section appended and stays as historical record.

---

## Honest assessment

The bug exists. The fix is straightforward. The risk over the next 4 weeks of live trading at conservative risk levels is low (estimated 1-2% probability of bad outcome). Deferring is defensible but not zero-cost.

The right time to fix is between Week 3 (risk ramp completing) and Week 5 (before any meaningful profit accumulation that could trigger scale-up). That's a 1-2 week window in roughly a month.

Operator decision: address it before live, accept it during ramp, OR defer to scheduled fix window. Current decision: defer to scheduled fix window.

This doc preserves the analysis so future-you (or future-CC) can resume from full context when the fix is scheduled.

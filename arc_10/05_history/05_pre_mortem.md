# Pre-Mortem

> **Purpose:** List the most likely ways Arc 10 could fail, before live deployment. Future-you, when something goes wrong, can check this list. If it's a predicted failure mode → continue per plan. If it's a surprise → stop and investigate.

## Why write this

When a strategy starts struggling, the first question is: "is this normal variance, a predicted failure mode, or a surprise?"

Without a pre-mortem, every failure feels like a surprise. Every drawdown becomes an emergency. Every losing month becomes evidence the system is broken.

With a pre-mortem, you have a calibration: most failures will fit one of these patterns. Recognizing the pattern lets you respond per plan instead of improvising.

## The most likely failure modes (in rough order of probability)

### 1. Worst-fold-equivalent drawdown period

**What it looks like:** total DD approaches 8-9%. Lasts 2-8 weeks. Multiple consecutive losing weeks. Win rate drops to ~30%.

**Why it happens:** the EA-faithful backtest at 0.40% has worst-fold DD of 8.21% trailing (F10 2019) / 5.49% from-initial (F5 2014); FundedNext measures from-initial. Hitting a worst-fold-equivalent period in any single year is roughly expected.

**Confidence this is the issue:** if total DD is between 5-8.5%, system is running, no operational errors → very likely this is just a worst-fold-equivalent period.

**Response:** continue per plan. The system halts entries at 7% and CloseAll at 8% automatically. If it breaches 8.5%, kill criteria fire (`07_kill_criteria.md`). Otherwise: this is what the system was designed to survive.

### 2. Multi-week quiet period

**What it looks like:** few or no trades for 2-3 weeks. Signal_processed/ count low. Account balance unchanged.

**Why it happens:** Arc 10's signal frequency varies. Quiet markets (low ATR, sideways action, holiday weeks) produce fewer signals across all 28 pairs simultaneously.

**Confidence this is the issue:** if sidecar logs show normal cycle completion every 4h, heartbeats fresh, just no signals firing across 28 pairs.

**Response:** wait. This is normal. Don't tinker with thresholds.

### 3. Live R-distribution slightly worse than backtest

**What it looks like:** mean R per trade in live is +0.20 instead of +0.40. Cumulative ROI tracks below median but inside band.

**Why it happens:** the haircuts in the cost sweep are estimates. Live broker fill quality, slippage tail, gap-through-SL events — all contribute. Some pessimistic assumptions may be slightly understated.

**Confidence this is the issue:** if mean R is +0.20 to +0.30 (vs expected +0.30 to +0.40) at 100+ trades, this is within haircut error.

**Response:** continue. This is what haircuts are for — actual live being slightly worse than ideal backtest is expected.

### 4. Broker disconnect coverage gap

**What it looks like:** sidecar logs show "MT5 not initialized" errors. Some H4 cycles miss completely. Signal that should have fired didn't.

**Why it happens:** brokers periodically disconnect, restart MT5, do maintenance. Sidecar can't fetch data during disconnect. If disconnect spans an H4 boundary, that cycle is missed.

**Confidence this is the issue:** if errors are intermittent (a few per week), self-recover, no pattern.

**Response:** monitor. Acceptable if rare. If frequent (multiple per day): contact broker support, consider broker change.

### 5. News-event gap-through-SL

**What it looks like:** single trade with -1.5R to -2R loss instead of expected -1R. Trade closed during a news event.

**Why it happens:** SL fills at next-available tick after news spike. If price gaps over SL level, fill is worse than SL price.

**Confidence this is the issue:** if trade closure timestamp coincides with major news event (NFP, CPI, central bank announcement).

**Response:** log the event. Acceptable if rare (1-3 per year). The news filter blocks new entries near news, but existing positions aren't closed pre-news (deliberate — closing positions before news is its own risk).

### 6. EA execution latency miss

**What it looks like:** envelope written to signals_out/, no broker trade follows. Or trade fills at noticeably worse price than envelope's entry_price_estimate.

**Why it happens:** EA polls every 5 seconds. If broker spread widens during the 5s window, the trade fills at a worse price than the signal expected.

**Confidence this is the issue:** if pattern correlates with broker spread spikes (visible in logs).

**Response:** acceptable if rare. If frequent: reduce `Signal_Poll_Min_Interval_Sec` to 2-3 seconds (faster polling = more responsive but more CPU).

### 7. EA restart re-baselines the total-DD floor (OPEN-001 — RESOLVED 2026-05-30)

> **Mechanism/magnitude corrected 2026-05-30.** This item previously read "Equity calculation drift" and described the EA caching equity at H4 boundaries with <0.5% drift. **Both were wrong.** The real risk was the total-DD floor being snapshotted from live equity at every `OnInit`, which a mid-drawdown restart would re-baseline downward — sinking the EA's halt below the broker's static termination point (silent protection failure). It was **not** driven by NSSM restarts: NSSM wraps the Python sidecar, not the EA terminal; `OnInit` re-fires only on terminal restart, crash auto-restart, or manual reattach/recompile. The original ~1–3%/6mo estimate was therefore overstated (it assumed NSSM was hitting the EA).

**What it looks like:** after a restart taken while the account is drawn down, the EA's journal shows a total-DD `floor` below the broker's static initial balance. The EA's halt/close-all thresholds are now computed off the sunken floor, so the broker could terminate the account before the EA's halt fires.

**Why it happened:** `g_arc_eq_total_floor` was captured at every `OnInit` from `AccountInfoDouble(ACCOUNT_EQUITY)` (`EquityGuards.mqh:100`) — not an input, not persisted. Any re-init re-baselined it to whatever equity was at that moment.

**Status — RESOLVED by `b386287` (#242).** The floor is now solely an operator-set input, `Initial_Equity_Floor` (default `0`), used directly with **no live-equity capture anywhere**. A fail-loud guard (`< 5000` → refuse to trade + `Alert()` + journal `FLOOR_FAIL`) prevents an unset/implausible floor from silently mis-protecting. Scale-up is a manual input edit; MT5's saved profile persists the input across restart (verified live). Full record: [`07_open_issue_dd_restart_rebaselining.md`](07_open_issue_dd_restart_rebaselining.md).

**Response (if a floor mismatch is ever observed post-resolution):** confirm the journal shows `source=input` and `floor` = the broker's static balance. A `FLOOR_FAIL`/halt means the floor is unset — set `Initial_Equity_Floor` and reattach. This is now an operator-config check, not a code bug.

### 8. Hidden bug surfacing under specific conditions

**What it looks like:** unexpected behavior — wrong direction trade, wrong lot size, wrong SL placement, missing TP1 partial. Trade log doesn't match what should have happened.

**Why it happens:** despite ST validation + Phase 2 parity, edge cases exist that weren't tested.

**Confidence this is the issue:** if the unexpected behavior is reproducible (e.g. happens on every JPY pair under certain conditions) rather than random.

**Response:** stop trading. Investigate. Bug fix + re-validation. Don't restart until fixed and tested.

### 9. Regime shift — strategy edge degrades

**What it looks like:** cumulative ROI below band at 200+ trades. Win rate persistently low across multiple windows. No identifiable bug — system is operating correctly but producing bad results.

**Why it happens:** markets change. Strategies built on historical patterns can lose edge when those patterns no longer hold. Volatility regime shifts. Central bank policy regime shifts. Microstructure changes.

**Confidence this is the issue:** if no operational issues, no bugs, but performance is sustained outside expected bands. The hardest failure mode to diagnose.

**Response:** triggers kill criterion #3 (ROI below band at 200+ trades). Stop trading. Deep investigation. Possibly the strategy needs re-validation against newer data, or Lomega discovery engine to find new signal candidates.

### 10. FundedNext rule change invalidating deployment

**What it looks like:** FundedNext changes terms — removes swap-free, tightens daily DD, prohibits weekend holds, etc.

**Why it happens:** prop firms change rules. Often after market events that hurt their P&L.

**Confidence this is the issue:** broker email notification or FAQ change.

**Response:** reassess. Some changes are manageable (small DD tightening can be absorbed). Others (swap-free removal) invalidate the cost sweep economics and require switching brokers or accepting reduced risk.

### 11. Operator error during maintenance

**What it looks like:** after a config change, restart, or update — system behaves unexpectedly. EA inputs wrong. Sidecar pointing at wrong broker. Service not auto-starting.

**Why it happens:** complex multi-component system. Easy to mis-configure under time pressure.

**Confidence this is the issue:** if anomaly started immediately after a maintenance action.

**Response:** revert the change. Verify against `04_vps_setup_guide.md` configuration. Restart carefully.

### 12. VPS-level issue

**What it looks like:** VPS unresponsive. RDP fails. Sidecar logs show extended gaps.

**Why it happens:** VPS provider outage, network issue, Windows update gone wrong, hard drive failure.

**Confidence this is the issue:** contact Contabo support for VPS health status.

**Response:** depends on cause. Provider outage: wait. Disk failure: restore from snapshot. Network: investigate. May need to redeploy from `04_vps_setup_guide.md` if VPS is unrecoverable.

## Less likely but possible

### 13. Broker fraud / insolvency

Rare but real. If broker goes bankrupt or fraudulent, account is lost. Mitigation: prop firms historically more stable than retail brokers; choose well-established firms.

**Response:** unrecoverable for that account. Move to backup broker.

### 14. Black swan market event

CHF-2015-equivalent. Single day with multi-R moves on multiple pairs. Could exceed 10% account DD in hours.

**Response:** broker terminates account at 10% DD. Outside system control.

### 15. Code drift via dependency update

Sidecar uses MetaTrader5 Python lib + various libraries. If a `pip install -U` updates a dependency, behavior could subtly change.

**Mitigation:** lock dependency versions (`requirements.txt` with `==` pinning). Don't auto-update.

## What NOT in the pre-mortem (because unlikely to be the issue)

- "Strategy logic was wrong all along" — extensively WFO-validated across 14 years and 11 folds. If this fails, it's regime shift (#9), not logic.
- "Cost modeling was wildly wrong" — modeled conservatively. If costs are an issue, they're 10-20% off, not 100%+ off. Covered by haircut framework.
- "Wrong pairs" — 28-pair universe well-validated. Adding/removing pairs requires explicit re-validation, not first-line response.
- "Wrong timeframe" — H4 chosen via comparison across H1/H2/H4/D1. Switching is not in scope.

## How to use this list when something goes wrong

1. Observe the failure symptom (drawdown, missing trades, weird behavior, etc.)
2. Scan this list for matching patterns
3. If matches → respond per "Response" section for that mode
4. If no match → STOP and investigate. This is a surprise, not a predicted failure.
5. After resolving (or after the live event passes): consider whether to add this as a new pattern to the list

**Surprises are dangerous because you have no playbook.** Predicted failures are routine because the playbook is written.

The goal of this pre-mortem is to maximize the percentage of live events that are "predicted" rather than "surprise."

## Reviewing this document

Annually, or after any significant incident:
- Did the failure mode appear in this list?
- If yes: was the response correct?
- If no: should it be added?
- Are any items in this list now obviously wrong / shouldn't be there?

Update the document with explicit version-bumped reasoning, only in a calibrated state (not during drawdown).

# Kill Criteria

> **Purpose:** Define explicit triggers for stopping Arc 10 deployment, written from a calibrated state.
> **Why now:** future-you in drawdown is the wrong person to decide whether to keep going. Past-you, calibrated, is the right one. Write the rules now, follow them later.
> **Status:** These are HARD triggers. Hitting one means stop trading. Re-evaluation is required before restarting.

## The cardinal rule

**If a kill criterion is hit, the system stops trading. Period. No "let me wait one more week."**

The reason to write this now: in the moment, you will rationalise continuing. "It's just a bad week." "The next trade will recover." "I'll close manually if it gets worse." These rationalisations are how blown accounts happen.

When a kill criterion fires:
1. Run emergency kill procedure (`04_runbook/05_emergency_kill.md`)
2. Stop trading
3. Cool down for at least 7 days (no re-evaluation during the cool-down)
4. Then evaluate per the "Restart criteria" section below
5. If restart conditions aren't met → system stays stopped

## What counts vs what doesn't

**Counts toward kill criteria:** automated EA trades on the live account.

**Does NOT count:** demo trades, backtest trades, trades from prior systems (KH-24), manual trades (which should never happen — see emergency kill if you ever placed one).

**Aggregation:** combined across both 5ers and FundedNext if both are live. The kill criteria apply to the overall Arc 10 deployment, not per-broker. Reason: bad performance on either signals the same underlying issue.

## The kill criteria (in priority order)

### #1 — System-level: prop firm hard limit hit

If either 5ers or FundedNext kills the account (10% total DD or 5% daily DD breach), the system stops automatically — broker terminates account.

**Action:** investigate root cause before opening any new account. Do not buy another Challenge until you understand what happened. This is the only "involuntary kill" — the others are deliberate.

### #2 — Operator-level: total DD ≥ 8.5% (past the auto-CloseAll)

EA-faithful worst-fold DD at the 0.40% operating tier is **8.21% trailing / 5.49% from-initial** (FundedNext, the primary firm, measures from-initial). Hitting 8.5% total means you're past the worst backtested trailing fold and past the system's own protection — "this is beyond what the system is designed to survive."

System auto-halts new entries at 7%, auto-CloseAll at 8%. The 8.5% threshold gives a 0.5pp buffer past CloseAll for cases where positions slipped through CloseAll before fully exiting.

**Action:** kill, cool down, investigate.

### #3 — Variance-level: cumulative ROI below 25% confidence band at 200+ trades

The live-tracking framework (`06_live_tracking_framework.md`, re-derived from `07_canonical_wfo.md`) specifies a 25% low band at 200 trades (~one year) of approximately **+14% ROI** — the worst observed fold (F9 2018). If actual cumulative ROI at 200 trades is well below this (e.g. cumulative +5% or negative), this exceeds normal variance.

**Threshold:** cumulative ROI < +5% at 200+ trades.

**Action:** kill, cool down, investigate. Possible causes: regime shift, broker fill degradation, hidden bug, market structure change.

**Why 200 trades:** below 200, sample noise is high enough that bad luck can produce low ROI. At 200+ trades, sustained underperformance is signal not noise.

### #4 — Consistency-level: 4+ negative months in any 6-month window

Backtest had ~30-35% negative months. 2 negative months in 6 is normal. 3 is unusual but possible. **4 in 6 is outside backtest experience and signals systematic issue.**

**Threshold:** 4 negative months in any rolling 6-month window.

**Action:** kill, cool down, investigate.

### #5 — Drawdown-duration-level: total DD > 5% for > 60 consecutive days

Backtest had drawdown periods that lasted weeks. Few exceeded 60 days. If account is in 5%+ DD for 60+ consecutive days without recovery, the system isn't reverting to its expected positive expectancy.

**Threshold:** total DD > 5% sustained for 60+ days.

**Action:** kill, cool down, investigate.

### #6 — Process-level: 3 consecutive kill criteria warnings without trigger

Even if no kill criterion is hit outright, getting close repeatedly (e.g. DD spiking to 7-7.9% three times in 6 months without breach) is a signal that the system is operating closer to its limits than designed.

**Threshold:** 3 incidents in 6 months where DD reaches 7% (system halt threshold) or worst weekly performance < −3% ROI.

**Action:** review (not full kill). Consider risk reduction. If issues continue: kill.

### #7 — Trust-level: any unexplained discrepancy you can't resolve

Phantom trade. Lost signal. Sidecar reporting fresh heartbeat but no trades for 2 weeks despite signals expected. Account balance off by an amount that doesn't reconcile to trade log.

**Action:** kill, cool down, investigate. Don't trade a system you don't fully understand.

This is the catch-all: any "I don't know what just happened" event triggers kill.

## What does NOT trigger kill

Things that LOOK alarming but are within expected variance:

- **Single bad day:** even −3% in one day is rare but possible in backtest. Not kill criteria.
- **Single bad week:** −2% in one week is uncomfortable but within variance.
- **Single bad month:** monthly variance is high. One negative month is normal.
- **Approaching but not breaching DD:** the system halts at 7%, CloseAll at 8%. Approaching these activates system protection, not kill criteria.
- **Lower trade frequency than expected for a month:** Arc 10 is a signal-frequency-variable system. Some months are quiet.
- **Higher trade frequency than expected:** as long as DD is controlled, more signals is fine.
- **Backtest expectation deviation in early sample:** 25-50 trades is too small to draw conclusions. Don't react.
- **Broker rule changes:** these may require config changes but don't auto-trigger kill. Reassess against new rules.

## Restart criteria (after a kill)

If you killed the system, here's what's required before restarting:

### Mandatory before any restart

1. **Cool-down period:** minimum 7 calendar days from kill date. No restart during this period. Use it to investigate without time pressure.

2. **Root cause identified:** you must be able to articulate WHY the kill happened. "Bad luck" is not a root cause; "this kill criterion fired because X" is. Document in `05_history/02_decisions_log.md`.

3. **Decision documented:** write a new entry in the decisions log explaining what changed (if anything) and why restart is justified.

### Restart conditions vary by kill type

| Kill type | Restart conditions |
|---|---|
| #1 prop firm hard limit | Cannot restart same account; open new account ONLY after root cause + fix |
| #2 8.5%+ DD | Restart at reduced risk (0.30%); return to the 0.40% operating tier after 50 trades clean |
| #3 ROI below band at 200+ trades | Significant investigation required; may need re-validation against new data; consider whether strategy is still alive |
| #4 4 negative months in 6 | Investigate regime; consider risk reduction; restart only if you understand cause |
| #5 prolonged DD | Same as #4 |
| #6 repeated near-misses | Reduce risk before restart |
| #7 unexplained discrepancy | Bug must be fixed and verified before restart |

### If you can't articulate WHY → don't restart

The hardest case: you killed the system, the cool-down passes, and you don't have a clear root cause. The temptation is to attribute it to bad luck and restart.

**Don't.** If you can't explain why, you can't predict if it'll happen again. Better to leave the system dormant for an extra month while investigating than to restart blindly.

If still no clear root cause after 30 days: strongly consider that the strategy no longer has edge (regime shift, validation drift, market structure change) and that further investigation is needed before deploying any version of it.

## What to do during cool-down

Productive activities (no trading):
- Read `04_incident_response.md` and `05_emergency_kill.md` carefully
- Review trade logs in detail — what does the actual loss path look like?
- Cross-check live trades against backtest expectations on the same period
- Investigate any unusual log entries
- Check broker fill quality (compare actual fills vs SL levels)
- If suspecting bug: re-run sidecar against historical data, compare to backtest
- Talk to anyone with relevant context (broker support if relevant)

Unproductive activities (avoid):
- Re-tuning parameters to fit the bad period
- Looking for new strategies to deploy "instead"
- Doubling down on the existing strategy at higher risk to "make it back"

## Why these specific numbers

| Criterion | Threshold | Rationale |
|---|---|---|
| DD limit | 8.5% | EA-faithful worst-fold 8.21% trailing / 5.49% from-init @ 0.40%; 8.5% sits just past the 8% auto-CloseAll |
| ROI band | < +5% at 200+ trades (low band ~+14%) | Below the worst-fold floor, sample large enough that bad ROI is signal |
| Negative months | 4 in 6 | Backtest rarely had 3, never had 4. Outside experience. |
| DD duration | 60 days | Backtest had recovery within 60 days in nearly all cases |
| Near-misses | 3 in 6 months | Pattern of near-misses = operating too close to limits |

These thresholds are calibrated to the backtest. If a backtest re-validation produces different numbers, update thresholds in a calibrated state with explicit version bump.

## What this document is NOT

- **It is NOT a stop-loss for individual trades.** That's the EA's `SL_ATR_Multiplier_Expected` = 3.5×ATR. This document is the kill switch for the entire strategy deployment.

- **It is NOT performance targets.** Performance targets are aspirational; kill criteria are floor conditions.

- **It is NOT optional.** If a kill criterion fires, the system stops. There's no "but I think it'll recover."

- **It is NOT permanent.** Restart is allowed under explicit conditions documented above. But the kill itself is non-negotiable.

## Pre-commitment

By deploying Arc 10 live, the implicit contract is: you will follow these criteria. The criteria exist precisely because in-the-moment decision-making fails under drawdown stress.

If you find yourself in 2027 trying to talk yourself out of a kill: re-read this paragraph. The Arc 10 deployment process explicitly anticipated this moment. Past-you knew that future-you would want to keep going. Past-you decided the rules anyway, calibrated to the data.

Follow the rules.

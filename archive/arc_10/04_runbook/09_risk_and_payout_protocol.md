# Risk & Payout Protocol — Arc 10 v3.0.2 (FundedNext)

> **Purpose:** Pre-commit risk levels, payout timing, and scaling behaviour from a calibrated state, so no sizing or withdrawal decision is ever made emotionally or mid-drawdown.
> **Why now:** future-you, up 8% or down 6%, is the wrong person to decide whether to raise risk or take a payout. Past-you, calibrated to the EA-faithful WFO, is the right one. Write the rules now, follow them later.
> **Primary firm:** FundedNext (Stellar 2-Step). 5ers is secondary.
> **Source of truth for all numbers:** `arc_10/02_validation/07_canonical_wfo.md` → `results/l_arc_10_v3.0.2_ea_faithful/` CSVs.
> **Status:** Logic LOCKED. Two FundedNext mechanics remain pending final confirmation (§6) — flagged inline; the safe assumption applies until confirmed.
> **Companion docs:** `07_kill_criteria.md` (hard stops), `06_live_tracking_framework.md` (performance bands), `../02_validation/07_canonical_wfo.md` (the numbers).

## The cardinal rule

**Risk steps and payouts are triggered by measured equity thresholds, not by judgment in the moment.**

You do not decide "I feel ready to raise risk." You pre-commit the trigger here, once, cold. Live-you only executes it. Do not edit this document from a stressed state. Revisions happen calibrated, justified, version-bumped — never to accommodate a bad week or a tempting balance.

---

## 1 — The governing facts (FundedNext, confirmed)

The DD floor and limit structure that everything below is built on:

- **DD basis: equity** (floating counts intraday). FundedNext measures the Maximum Loss Limit **from-initial** — this is the live basis that matters most.
- **Maximum Loss Limit (MLL):** 10% of initial balance. Floor sits at **initial − 10%** (e.g. $90k on a $100k account). Profit *expands* allowable loss room: unwithdrawn profit adds to the MLL (at $120k equity, allowable loss = $10k + $20k = $30k; you can fall to $90k). **The floor is fixed at initial − 10% within a tier; the room above it grows with retained profit.**
- **Daily Loss Limit:** 5% of initial, resets **00:00 server time** (server = GMT+3 summer / GMT+2 winter — DST-switching; EA day-anchor must track server midnight, not a fixed UTC offset). Intraday profit increases that day's available daily room.
- **Internal governors (operator cushions inside FundedNext's limits, NOT FundedNext's numbers):** daily 3.5% halt / 4.5% close-all; total 7% halt / 8% close-all. Validated: −9% halt test confirmed entries blocked under deep drawdown.

**EA-faithful WFO result (floating-equity sizing, costed, governed, EET — the live-matched basis):** see `../02_validation/07_canonical_wfo.md`. At the **0.40% operating tier**, worst-fold DD is **5.49% from-initial / 8.21% trailing**, worst daily **4.11%**, **0 kills** across 11 search folds + 6 holdout years. From-initial is FundedNext's actual basis and sits comfortably under the 8% in-system target. This is the basis the system runs on.

**The core principle:** within a tier, risk capacity grows with retained profit — the floor is fixed and profit expands the room. Across a scale, this resets (§4). Risk is therefore a **sawtooth**: build within a tier, reset at each scale.

---

## 2 — Risk ladder

Risk per trade is a function of banked buffer and tier, not of confidence, streak, or balance milestones.

| Stage | Condition | Risk/trade | Rationale |
|---|---|---|---|
| **Initial launch** | first-ever live deployment, £0 buffer | **0.30–0.40%** | No live track record + thinnest buffer. De-risk the front-loaded high-concurrency period; ramp toward the 0.40% operating tier as the first clean weeks accrue. |
| **Operating tier** | post-ramp, normal running | **0.40%** | The validated operating level. EA-faithful worst-fold DD 5.49% from-initial / 8.21% trailing, daily 4.11%, 0 kills — the only level clearing **both** hard limits on the conservative trailing basis. |
| **Through scaling** | post-scale, new tier | **0.40% carries** | % risk and % DD are scale-invariant; the system is live-validated by then. A new tier resets the *buffer*, not the validation — so 0.40% holds; re-launch at 0.30% only if a scale coincides with a regime you have no live data for. |
| **0.50% (marginal upgrade)** | see §7 | **gated, deferred** | Real upside on an expanding-room account, but it FAILS the conservative trailing/daily basis (10.89% trailing, 5.16% daily). Evidence-gated only — never an automatic buffer-triggered step. |

**Hard rules:**
- **0.40% is the operating ceiling for routine running.** 0.50% is not a normal buffer-triggered step; it is a marginal, evidence-gated upgrade (§7).
- Risk never rises during an open drawdown, even if a balance spike crosses a threshold intrabar. Triggers read realized, day-close equity — not floating.
- The 0.30–0.40% launch ramp is a one-time first-deployment de-risk; it is not re-applied at every scale (the % behaviour is identical at any tier; only the very first live period lacks a track record).

---

## 3 — Within-tier banking & payout

**Key confirmed fact: a qualifying cycle requires 4% growth, NOT a withdrawal.** You do not have to withdraw to bank a qualifying cycle or to make scaling progress. This unlocks **bank-and-hold**: leave profit in, let it expand your MLL room, carry maximum safety through the entire tier.

**The optimal within-tier play:**
1. **Do not withdraw mid-tier.** Retained profit expands the MLL room (§1) and carries the buffer. Withdrawing mid-tier shrinks both — it pulls equity back toward the floor and reduces allowable loss. There is no scaling reason to withdraw (qualification is performance-based), so the only reason to withdraw mid-tier is income need.
2. **Scaling is the income event.** At a scale, profit is realized (§4) — that is the natural payout point. Hold through the tier, get paid at the scale, restart larger. No mid-tier withdrawal required.
3. **If income is needed mid-tier:** withdraw only profit above the buffer line that holds the operating tier. Never withdraw below it. Mechanics: rewards requested at cycle ends (21d first, then 14d); min request $20 (carried forward if under); crypto methods cap $2,000/request, RiseWorks unlimited; reward split 80% (→90% post-scale).

**The emotion-removal point:** the temptation is to withdraw early ("take some off the table"). On this structure, mid-tier withdrawal is strictly negative — it shrinks both the buffer and the loss room, for no scaling benefit. Bank-and-hold to the scale event is the disciplined default.

---

## 4 — Scaling (the sawtooth)

**Qualification requirements (all must be met):**
- **4 qualifying Performance Reward cycles** — a cycle qualifies on **≥4% account growth within that cycle** (performance, not withdrawal).
- A sub-4% cycle **does not count but does not reset** progress — continue until 4 qualifying cycles are banked.
- **Account held ≥2 months minimum.**
- Then close all positions and contact Support — scaling is **manual**.

**Cycle structure:** first cycle 21 days, then 14-day cycles. So 4 qualifying cycles ≈ 2+ months minimum (21 + 14 + 14 + 14 ≈ 9 weeks), aligning with the 2-month floor. Required pace: ≥4% per ~2-week cycle to stay on the scaling track. `[PENDING CONFIRM: is the 4% measured on closed balance or equity at cycle start/end? With ~40-day positions straddling 14-day boundaries, this determines whether a cycle's growth is driven by when trades close (balance) or when profit accrues (equity).]`

**At the scale event (working model — `[PENDING FINAL CONFIRM]`):**
- Accumulated profit is **realized / paid out** (transferred to wallet, withdrawn in full).
- Account **restarts at the new balance** = prior tier balance × 1.25 (e.g. 100k → 125k), with **floating profit zeroed**.
- Both Daily and Max Loss Limits **re-base proportionally** to the new balance (5% / 10% of 125k).
- Repeats at +25% per qualifying scale up to **$4M** max.

**Consequence — risk capacity is a sawtooth, not a monotonic compound:**
- *Within* a tier: floor fixed, room expands with retained profit → capacity grows.
- *At* a scale: profit paid out, balance steps +25%, limits re-base, buffer resets to zero → capacity drops back to a fresh (larger) tier's starting point.
- Net trend is upward and the tiers get larger, but each scale is a reset, not a carry. **This is why §2 holds 0.40% rather than escalating risk across scales** — there is no compounding buffer to "spend" on higher risk; each tier starts fresh.

**Strategic shape:** hold-and-bank through each tier (max safety, expanding room) → qualify on 4×4% over 2 months → scale event realizes profit (your income) → restart larger → repeat. Scaling is simultaneously the growth mechanism and the payout mechanism.

---

## 5 — Pre-committed triggers (executable summary)

| Trigger (measured, day-close equity) | Action |
|---|---|
| First-ever launch | risk = 0.30–0.40% ramp, withdraw nothing |
| Ramp clean, normal running | risk = 0.40% (operating tier) |
| Mid-tier, income needed | withdraw only profit above buffer line; else bank-and-hold |
| 4 qualifying cycles (≥4% each) + ≥2 months | close all, contact Support, scale |
| Scale processed | profit realized; restart at +25% balance; limits re-base; risk holds 0.40% |
| Considering risk > 0.40% (i.e. 0.50%) | freeze; run §7 daily-DD sweep + require live gap-event evidence; no live discretion |
| Any kill criterion (`07_kill_criteria.md`) | stop — overrides everything here |

Risk never rises during open drawdown. Withdrawal never drops equity below the buffer line. Risk never exceeds 0.40% without the §7 re-gate.

---

## 6 — Parameter status

**Confirmed (FundedNext, Stellar 2-Step):**
- DD basis equity; MLL = 10% of initial, floor fixed at initial−10% within tier, room expands with retained profit.
- Daily 5%, reset 00:00 server (GMT+2/+3 DST).
- Qualifying cycle = ≥4% growth, **withdrawal not required**.
- Scaling: 4 qualifying cycles + ≥2 months + manual; +25%/step to $4M; sub-4% cycle doesn't count, doesn't reset.
- Cycle cadence 21d then 14d. Min reward $20 (carried if under); crypto cap $2k/request, RiseWorks unlimited; split 80%→90% post-scale, +15% challenge-phase reward on scale.
- Internal governors 3.5/4.5 daily, 7/8 total — operator cushions, −9% halt validated.
- Operating risk 0.40% (EA-faithful canonical, `../02_validation/07_canonical_wfo.md`).

**Pending final confirmation (safe assumption applies until confirmed):**
1. `[CONFIRM]` Scale-event profit handling — working model: profit paid out, account restarts flat at +25% balance, limits re-base. (Alternative: profit rolls into new balance. The two give different post-scale buffers — confirm before first scale.)
2. `[CONFIRM]` Cycle 4%-growth measurement basis — closed balance vs equity at cycle boundaries (§4).
3. `[CONFIRM]` Live gap-event behaviour at 0.50% — whether the tick EA actually caps daily under 5% on a real spike (the §7 gate to the marginal upgrade).

None block deployment at 0.30–0.40%. Items 1–2 gate the first scale; item 3 gates any step to 0.50%.

---

## 7 — Risk beyond the 0.40% operating tier (0.50% marginal upgrade — deferred candidate)

Real upside exists: on an account where the floor is fixed and room expands with retained profit, higher risk deep in buffer is high-EV and compounds within a tier. **But 0.50% is gated, not discretionary — and on the conservative basis it FAILS.**

**Why 0.50% is not a free knob:** DD scales ~linearly with risk, but **breach behaviour does not.** From-initial DD headroom (6.85% at 0.50%) is the *misleading* number — it looks like room to push. The binding constraints are the **trailing DD** (10.89% gov-on at 0.50% > 10% hard limit) and the **5% daily limit** (5.16% at 0.50% > 5%), where concurrency and correlated-currency clusters go non-linear: the single-currency cluster that is a ~4–5% floating daily event at 0.40% tips over the daily line at 0.50%. Total-DD room hides this. The worst case is **F1 2010 (the Flash Crash)** — a single-bar gap that fired the close-all and reached 5.16% daily at 0.50% (`governor_log.csv`, `per_fold.csv`).

**The gate for 0.50%:**
- Re-run the **daily-DD and governor-firing sweep** (the costed-governed simulator, swept across r_base) — not the total-DD analysis. Daily/trailing breach probability is the thing that kills accounts and the thing that goes non-linear.
- **Require live gap-event evidence:** a real spike day must confirm the tick EA caps daily under 5% before 0.50% is considered. This is the live-only residual (intrabar tick resolution) the backtest cannot close.
- Buffer-gated: only considered deep in a tier's buffer, never at launch.
- Never a live discretionary call. "Scale further if it feels right" is the sentence that blows the account; "scale further after a daily-DD sweep + a confirmed live gap-event" is the same upside with discipline intact.

**Note on governors:** the daily-DD sweep already proved tightening the close-all is pointless — the worst daily event (F1 2010, 5.16%) is a single-bar gap, threshold-invariant; lowering the 4.5% close-all does not catch it sooner and only costs ROI on benign days. **Governors stay 3.5/4.5 daily, 7/8 total.**

**Do not bend the system to the 4%-per-cycle pace.** No close-timing or sizing changes to make a cycle "look good" — that is discretionary drift the whole protocol exists to prevent. Note the pace requirement, measure against it, do not trade to it.

---

## Version

- **v0.4** — re-based on the EA-faithful (floating-equity, live-matched) canonical WFO (`../02_validation/07_canonical_wfo.md`). **0.40% is now the operating tier; 0.50% is a marginal, evidence-gated upgrade that FAILS the conservative trailing (10.89%) and daily (5.16%) basis** — replaces the prior "0.50% validated ceiling / 3.75% from-initial" framing, which rested on a superseded whole-period continuous-equity figure. From-initial at 0.50% is 6.85% (not 3.75%). FundedNext mechanics (static MLL, bank-and-hold, sawtooth scaling, cadence, reward split) unchanged. §7 reworked to gate 0.50% on a daily-DD sweep + live gap evidence. Gap (single-currency bounded) + slippage (~5× margin) findings added as the risk-surface basis for the 0.40% call.
- v0.3 — full FundedNext model. Confirmed: qualification = 4% growth not withdrawal (bank-and-hold unlocked); scaling sawtooth; cycle cadence + reward mechanics. (Superseded: held 0.50% as the validated ceiling on the whole-period 3.75% from-initial figure.)
- v0.2 — MLL-stays-at-initial (later refined to the profit-expanding-room model); daily reset, scaling step, reward split confirmed.
- v0.1 — initial draft, post whole-period DD analysis.
- Cardinal rule throughout: written calibrated, followed live, never edited stressed.

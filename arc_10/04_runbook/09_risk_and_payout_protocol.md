# Risk & Payout Protocol — Arc 10 v3.0.2 (FundedNext)

> **Purpose:** Pre-commit risk levels, payout timing, and scaling behaviour from a calibrated state, so no sizing or withdrawal decision is ever made emotionally or mid-drawdown.
> **Why now:** future-you, up 8% or down 6%, is the wrong person to decide whether to raise risk or take a payout. Past-you, calibrated to the whole-period DD analysis, is the right one. Write the rules now, follow them later.
> **Primary firm:** FundedNext (Stellar 2-Step). 5ers is secondary.
> **Status:** Logic LOCKED. Two FundedNext mechanics remain pending final confirmation (§6) — flagged inline; the safe assumption applies until confirmed.
> **Companion docs:** `07_kill_criteria.md` (hard stops), `06_live_tracking_framework.md` (performance bands).

## The cardinal rule

**Risk steps and payouts are triggered by measured equity thresholds, not by judgment in the moment.**

You do not decide "I feel ready to raise risk." You pre-commit the trigger here, once, cold. Live-you only executes it. Do not edit this document from a stressed state. Revisions happen calibrated, justified, version-bumped — never to accommodate a bad week or a tempting balance.

---

## 1 — The governing facts (FundedNext, confirmed)

The DD floor and limit structure that everything below is built on:

- **DD basis: equity** (floating counts intraday).
- **Maximum Loss Limit (MLL):** 10% of initial balance. Floor sits at **initial − 10%** (e.g. $90k on a $100k account). Profit *expands* allowable loss room: unwithdrawn profit adds to the MLL (at $120k equity, allowable loss = $10k + $20k = $30k; you can fall to $90k). **The floor is fixed at initial − 10% within a tier; the room above it grows with retained profit.**
- **Daily Loss Limit:** 5% of initial, resets **00:00 server time** (server = GMT+3 summer / GMT+2 winter — DST-switching; EA day-anchor must track server midnight, not a fixed UTC offset). Intraday profit increases that day's available daily room.
- **Internal governors (operator cushions inside FundedNext's limits, NOT FundedNext's numbers):** daily 3.5% halt / 4.5% close-all; total 7% halt / 8% close-all. Validated: −9% halt test confirmed entries blocked under deep drawdown.

**Whole-period analysis result (costed, governed, EET, r_base 0.5%):** from-initial DD **3.75%** over 16 years vs the 8% in-system target — structurally unbreachable once a buffer exists. This is the basis the system runs on.

**The core principle:** within a tier, risk capacity grows with retained profit — the floor is fixed and profit expands the room. Across a scale, this resets (§4). Risk is therefore a **sawtooth**: build within a tier, reset at each scale.

---

## 2 — Risk ladder

Risk per trade is a function of banked buffer and tier, not of confidence, streak, or balance milestones.

| Stage | Condition | Risk/trade | Rationale |
|---|---|---|---|
| **Initial launch** | first-ever live deployment, £0 buffer | **0.30–0.40%** | No live track record + thinnest buffer. De-risk the front-loaded high-concurrency period until both exist. |
| **Steady-state** | buffer ≥ [CONFIRM threshold, ROI %], held ≥10 trading days | **0.50%** | Full validated risk. The whole-period 3.75% DD result is at 0.50%. |
| **Through scaling** | post-scale, new tier | **0.50% carries** | % risk and % DD are scale-invariant; the system is live-validated by then. A new tier resets the *buffer*, not the validation — so 0.50% holds; only re-launch at 0.30–0.40% if a scale coincides with a regime you have no live data for. |
| **Beyond 0.50%** | see §7 | **deferred** | Real upside on an expanding-room account, but gated on a daily-DD sweep — never a discretionary live call. |

**Hard rules:**
- Never exceed 0.50% without a re-gate (§7). 0.50% is the validated ceiling.
- Risk never rises during an open drawdown, even if a balance spike crosses a threshold intrabar. Triggers read realized, day-close equity — not floating.
- The 0.30–0.40% launch is a one-time first-deployment de-risk; it is not re-applied at every scale (the % behaviour is identical at any tier; only the very first live period lacks a track record).

---

## 3 — Within-tier banking & payout

**Key confirmed fact: a qualifying cycle requires 4% growth, NOT a withdrawal.** You do not have to withdraw to bank a qualifying cycle or to make scaling progress. This unlocks **bank-and-hold**: leave profit in, let it expand your MLL room, carry maximum safety through the entire tier.

**The optimal within-tier play:**
1. **Do not withdraw mid-tier.** Retained profit expands the MLL room (§1) and carries the buffer that holds your 0.50% risk. Withdrawing mid-tier shrinks both — it pulls equity back toward the floor and reduces allowable loss. There is no scaling reason to withdraw (qualification is performance-based), so the only reason to withdraw mid-tier is income need.
2. **Scaling is the income event.** At a scale, profit is realized (§4) — that is the natural payout point. Hold through the tier, get paid at the scale, restart larger. No mid-tier withdrawal required.
3. **If income is needed mid-tier:** withdraw only profit above the buffer line that holds 0.50% risk. Never withdraw below it. Mechanics: rewards requested at cycle ends (21d first, then 14d); min request $20 (carried forward if under); crypto methods cap $2,000/request, RiseWorks unlimited; reward split 80% (→90% post-scale).

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
- Net trend is upward and the tiers get larger, but each scale is a reset, not a carry. **This is why §2 holds 0.50% rather than escalating risk across scales** — there is no compounding buffer to "spend" on higher risk; each tier starts fresh.

**Strategic shape:** hold-and-bank through each tier (max safety, expanding room) → qualify on 4×4% over 2 months → scale event realizes profit (your income) → restart larger → repeat. Scaling is simultaneously the growth mechanism and the payout mechanism.

---

## 5 — Pre-committed triggers (executable summary)

| Trigger (measured, day-close equity) | Action |
|---|---|
| First-ever launch | risk = 0.30–0.40%, withdraw nothing |
| Buffer ≥ threshold, held ≥10d | risk → 0.50% |
| Mid-tier, income needed | withdraw only profit above buffer line; else bank-and-hold |
| 4 qualifying cycles (≥4% each) + ≥2 months | close all, contact Support, scale |
| Scale processed | profit realized; restart at +25% balance; limits re-base; risk holds 0.50% |
| Considering risk > 0.50% | freeze; run §7 daily-DD sweep; no live discretion |
| Any kill criterion (`07_kill_criteria.md`) | stop — overrides everything here |

Risk never rises during open drawdown. Withdrawal never drops equity below the buffer line. Risk never exceeds 0.50% without a re-gate.

---

## 6 — Parameter status

**Confirmed (FundedNext, Stellar 2-Step):**
- DD basis equity; MLL = 10% of initial, floor fixed at initial−10% within tier, room expands with retained profit.
- Daily 5%, reset 00:00 server (GMT+2/+3 DST).
- Qualifying cycle = ≥4% growth, **withdrawal not required**.
- Scaling: 4 qualifying cycles + ≥2 months + manual; +25%/step to $4M; sub-4% cycle doesn't count, doesn't reset.
- Cycle cadence 21d then 14d. Min reward $20 (carried if under); crypto cap $2k/request, RiseWorks unlimited; split 80%→90% post-scale, +15% challenge-phase reward on scale.
- Internal governors 3.5/4.5 daily, 7/8 total — operator cushions, −9% halt validated.

**Pending final confirmation (safe assumption applies until confirmed):**
1. `[CONFIRM]` Scale-event profit handling — working model: profit paid out, account restarts flat at +25% balance, limits re-base. (Alternative: profit rolls into new balance. The two give different post-scale buffers — confirm before first scale.)
2. `[CONFIRM]` Cycle 4%-growth measurement basis — closed balance vs equity at cycle boundaries (§4).
3. `[CONFIRM]` Steady-state buffer threshold that unlocks 0.50% (set as ROI %, not currency).

None block deployment at 0.30–0.40%. Items 1–2 gate the first scale; item 3 gates the step to 0.50%.

---

## 7 — Risk beyond 0.50% (deferred candidate)

Real upside exists: on an account where the floor is fixed and room expands with retained profit, higher risk once deep in buffer risks *expanded* room against a fixed floor — high-EV, and the upside compounds within a tier. **But it is gated, not discretionary.**

**Why it is not a free knob:** DD scales ~linearly with risk, but **breach behaviour does not.** From-initial total DD headroom (3.75% at 0.50%) is the *misleading* number — it looks like room to push. The binding constraint is the **5% daily limit**, where concurrency and correlated-currency clusters go non-linear: the same EUR/GBP concurrent cluster that is a ~4.7% floating daily event at 0.50% becomes a daily breach at higher risk. Total-DD room hides this.

**The gate for any risk above 0.50%:**
- Re-run the **daily-DD and governor-firing sweep** (the costed-governed simulator, swept across r_base) at each candidate level — not the total-DD analysis. Daily breach probability is the thing that kills accounts and the thing that goes non-linear.
- Buffer-gated: only considered when deep in a tier's buffer, never at launch.
- Post-live, on real fills — the modeled give-back and governor timing must be confirmed against live behaviour first.
- Never a live discretionary call. "Scale further if it feels right" is the sentence that blows the account; "scale further after a daily-DD sweep confirms breach probability at that level" is the same upside with discipline intact.

**Do not bend the system to the 4%-per-cycle pace.** No close-timing or sizing changes to make a cycle "look good" — that is discretionary drift the whole protocol exists to prevent. Note the pace requirement, measure against it, do not trade to it.

---

## Version

- **v0.3** — full FundedNext model. Confirmed: qualification = 4% growth not withdrawal (bank-and-hold unlocked); scaling sawtooth (profit realized at scale, limits re-base, buffer resets per tier); cycle cadence + reward mechanics. §2 ladder reworked (0.50% holds through scaling, no escalation across tiers); §3 bank-and-hold default; §4 sawtooth; §7 beyond-0.50% deferred candidate (daily-DD-sweep-gated). 3 items pending final confirm.
- v0.2 — MLL-stays-at-initial (later refined to the profit-expanding-room model); daily reset, scaling step, reward split confirmed.
- v0.1 — initial draft, post whole-period DD analysis.
- Cardinal rule throughout: written calibrated, followed live, never edited stressed.

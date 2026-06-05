# arc 3008 — Month-end London-4pm-fix flow dislocation-and-reversion (long)

**Chat:** 3000–3999 · **Date:** 2026-06-05 · **Disposition:** KILL (real, beats-null, but net-negative)

## (a) Log synthesis (FRESH EYES, honest-era only)
22 arcs, 3 chats, honest engine. ONE net-positive long-only component ever found: the weekend
gap-fill on JPY crosses (arc 1006, PORTFOLIO; audited thinner in arc 1009 — gap-specific excess
~+0.36pp, threshold-fragile). It is a **discrete-flow-event reversion**. EDGE<COST is the binding
constraint, now closed from BOTH sides (edge-side: every entry/TF/universe/regime/exit/stop-removed;
cost-side: exits/frequency 1004, spread-timing 3007). Directional space exhaustively closed under
capture+drift lenses. Calendar/microstructure flow: turn-of-month drift sub-cost (1005), gotobi
absent (1008), round-numbers arbitraged (1010), triangulation ≈0 (3005) — pattern: a famous
*because* ≠ a surviving edge, and H4 nets intraday flows → confirm at H1. arc-3004 escalation
stands (structural unlock = operator). My pre-shorts lane: a 2nd net-positive *decorrelated*
long-only PORTFOLIO component, or a novel structural mechanism with a *because*.

## (b) Idea + why (log-seeded)
Arc 1010 named the one untested discrete-flow lane verbatim: *"month-end fixing-flow
dislocation-and-reversion, distinct from arc 1005's turn-of-month drift."* The **London 4pm
WM/Reuters fix** is the most-documented FX flow event: at month-end, equity-index investors
rebalance FX hedges at the 16:00 London benchmark; the mechanical, predictable flow pushes price
INTO the fix and partially REVERSES after (Melvin & Prins 2015; Evans 2018). The long-only side =
buy an abnormal DOWN push into the fix, bet on post-fix reversion UP. This is the gap-fill archetype
on a **decorrelated, calendar-timed intraday** event → candidate 2nd PORTFOLIO component. Distinct
from arc 1005 (multi-day turn-of-month *drift*, not the fix-window *dislocation-reversion*).

## (c) Observation (gross bp, then honest ATR/R; IS 2010–2020, 7 USD majors, H1)
Fix-hour bar = the 15:00–16:00 London bar (DST-robust via Europe/London tz), ending at the 16:00 fix.

**Bp lens (`arc3008_observe_fix.py`):** a real, month-end-SPECIFIC reversion.
- corr(push, +2h return): **−0.121 on month-end** vs −0.035 non-ME.
- BIG down-push (top quintile) into the month-end fix → **+4.31 bp over 2h, frac+ 0.576** (n=59);
  non-ME big down-push only +1.44 bp, frac+ 0.509. The reversion is month-end-specific.
- BUT thin (~5 big-push events/yr across 7 pairs) and tail-driven; cross-pair scattered.

**Honest ATR/R lens (`arc3008_capture.py`, BUILT `observe_long_capture`, next-bar-open entry, SL=2·ATR):**
- gross fwd-3h **drift +0.11 to +0.12R** at every threshold k∈{0.25,0.5,1.0} (the reversion).
- capture 0.24–0.26 (uninformative here: +1R = 2·ATR over a 6h H1 horizon is far; drift is the lens).
- **Cross-section is a NOISE signature** (arc-1010 tell): only USDCAD (+0.17–0.35R) and USDCHF
  (+0.28–0.40R) robustly positive across thresholds; GBPUSD/NZDUSD/USDJPY negative. 2/7 robust.
- On this H1 construction 1R = 2·H1-ATR ≈ H4-ATR (half the H4 R), so the spread cost-in-R is ~2× the
  H4 figure → cost hurdle ~0.10–0.20R → **+0.12R drift sits right at EDGE≈COST.**

It beats the placebo + shows +0.12R gross drift (not a coin-flip), so per §5f it earns the honest
engine before a FAIL, with the reversion-appropriate exit (short time-exit; arc 2004: reversion
entries prefer stop-removed/time-exit).

## (d)/(f)/(g) Honest WFO (full 10-fold IS; OOS held PRISTINE — IS never cleared)
`arc3008_wfo.py` — ArcFoldRunner → MultiPairBacktester, FundedNext costs ON, SL-honest. Best-version
exit/SL sweep (§5f), 7 USD majors, H1. **Every config × both thresholds is net-NEGATIVE and NOT
all-folds-positive:**

| threshold | exit/SL | IS mean | worst fold | neg folds | maxDD | n |
|---|---|---|---|---|---|---|
| 0.5 | partial/runner SL2.0 | −1.32% | −5.54% | 7/10 | 5.97% | 221 |
| 0.5 | time3 SL2.0 | −1.13% | −4.07% | 7/10 | 4.21% | 160 |
| 0.5 | **time3 SL6.0 (stop-off)** | **−0.28%** | −1.48% | 5/10 | 1.53% | 160 |
| 0.5 | time6 SL6.0 (stop-off) | −0.37% | −1.30% | 7/10 | 1.41% | 143 |
| 0.25 | partial/runner SL2.0 | −1.34% | −5.65% | 6/10 | 5.97% | 243 |
| 0.25 | time3 SL2.0 | −1.04% | −4.07% | 7/10 | 4.21% | 176 |
| 0.25 | **time3 SL6.0 (stop-off)** | **−0.22%** | −1.48% | 5/10 | 1.53% | 176 |
| 0.25 | time6 SL6.0 (stop-off) | −0.32% | −1.30% | 5/10 | 1.41% | 159 |

**The arc-2004/3004 reversion signature reproduces exactly:** removing the SL-first 2·ATR stop
(time3 SL6.0) halves DD (5.97→1.53%) and lifts mean from −1.3% toward zero (−0.28%) — the take-the-loss
tax bites a buy-into-weakness entry — but **no exit/SL/threshold reaches mean-positive or
all-folds-positive.** The stop is a real drag, not the wall; the residual edge ≈ cost.

**Fair NULL (random-entry, matched fire-rate, SAME time3 SL6.0 exit/SL/universe — BUILT
`build_null_signal_evaluation`):** REAL −0.28% (5/10 neg) vs NULL −0.73% (9/10 neg). **The signal
BEATS random entry → a REAL but SUB-COST edge** (the arc-0/1000 signature): the month-end fix
reversion genuinely exists (beats placebo in §c AND beats the random null here), but its gross
magnitude (+0.12R/3h) does not clear FundedNext cost + take-the-loss on this H1 construction (where
1R = 2·H1-ATR ≈ H4-ATR, so cost-in-R is ~2× the H4 figure).

## Verdict
**KILL** (§11: beats-null-but-net-negative). A REAL, month-end-specific, mechanistically-clean,
decorrelated reversion that nonetheless lands SUB-COST — so it is NOT a PORTFOLIO component
(PORTFOLIO requires mean-POSITIVE net of cost, e.g. the gap-fill +0.36–0.69pp; this is −0.22 to
−0.28% at best). You cannot diversify a net-negative component positive (arcs 3000/3001). OOS held
pristine (IS never cleared; ~3 trades/pair/yr is too thin for the all-folds judge anyway — the
arc-2001/2003 majors-gap-fill discipline).

## Tooling
BUILT `MonthEndFixReversionLongSignal` (`discovery/tools/fix_flow_signals.py`) — mask + ATR geometry
only, conforms to `SignalModule`. Reused BUILT `observe_long_capture`, `make_time_exit_predicate`,
`build_null_signal_evaluation`. No canonical-core change.

## Convergence with arc 1011/1012 (chat 1000s, landed concurrently)
The 1000s chat independently tested **month-end reversion long on USD majors** and reached
**PORTFOLIO** (the 2nd net-positive component): arc 1011 `MonthEndReversionLongSignal` is a **D1
multi-bar over-extension INTO month-end** (mean +0.23% IS, mechanism-controlled +0.249 ATR excess
vs a random day, beats the fair null +0.56pp, threshold-robust, leave-one-pair-out all+; 7/10 folds
→ PORTFOLIO not all-folds-positive). Mine is the **H1 intraday 16:00-fix-WINDOW dislocation** of the
same broad idea, and it is **sub-cost (KILL)**. **Two chats converged: month-end reversion is REAL
on USD majors.** The disposition split is *explained by my lesson #2* — the D1 version's R (2·D1-ATR)
is large, so cost-in-R is low and the +reversion drift clears; my H1 version's R (2·H1-ATR ≈ H4-ATR)
is small, so the same spread costs ~2× in R and eats the drift. **The capturable expression is the
D1 over-extension (their arc-1011 PORTFOLIO component), NOT the intraday fix window (my KILL).** My
arc CORROBORATES theirs (real, mechanism-controlled month-end reversion) and adds the negative
boundary: H1/fix-window resolution is the wrong, cost-dominated way to harvest it.

## Threads / lessons
1. **Month-end London-4pm-fix reversion is REAL but sub-cost on H1 majors** — beats both the non-ME
   placebo (corr −0.121 vs −0.035; big-down-push +4.3bp/2h vs +1.4bp) AND a fair random null
   (−0.28% vs −0.73%), yet net-negative after FundedNext cost + take-the-loss. The 4th documented
   microstructure/flow effect to be real-but-uncapturable on honest H4/H1 data (gotobi 1008,
   round-numbers 1010, triangulation 3005, now the fix) — a famous *because* with a measurable,
   placebo-confirmed, null-beating footprint STILL dies at the cost floor.
2. **The H1 cost-in-R penalty is a structural tax on intraday reversion edges** — 1R = 2·H1-ATR ≈
   H4-ATR (half the H4 R), so the same spread costs ~2× in R terms. An intraday reversion needs a
   *larger* gross drift than an H4 signal to clear cost; +0.12R/3h is not enough. Future intraday
   ideas should weigh this before believing a small gross drift.
3. **Reproduces arc-2004 on a NEW reversion family** — removing the SL-first stop halves DD and lifts
   mean toward zero (buy-into-weakness = big adverse excursion = stop-tax), but never manufactures
   the edge. Stop-removed/time-exit is the right construction for a reversion entry, necessary but
   not sufficient. Two independent reversion families (weekend gap-fill majors, month-end fix) now
   share this exact A/B signature.
4. **The discrete-liquidity-EVENT lane (arc 1010's last named long-only thread) is now mapped:**
   turn-of-month drift sub-cost (1005), gotobi absent (1008), round-numbers arbitraged (1010),
   month-end **D1** over-extension reversion mean-POSITIVE → **PORTFOLIO (arc 1011, the 2nd
   decorrelated long-only component!)**, month-end **fix-window H1** reversion real-but-sub-cost
   (this arc, KILL). There are now **TWO net-positive long-only PORTFOLIO components** (arc-1006
   JPY-cross weekend gap-fill + arc-1011 D1 month-end reversion). **Arc 2006 (2000s, landed
   concurrently) already ran the 2-way combination of those two → KILL** (a mutually-negative 2015
   fold + tail-correlation blocks an all-folds-positive combined book; its lesson: pick a 3rd
   component by its ROI on the book's NEGATIVE folds, not by avg-corr). My net-negative fix-window
   variant is **NOT** a usable 3rd component. The arc-3004 escalation + FLAG-1 shorts/second-leg
   unlock remain the highest-leverage operator move. **Surviving pre-shorts
   untested sub-lanes:** the fix's STRONGER leg is the UP-push short (blocked by long-only = FLAG-1
   again — the up-push into the fix reverts DOWN, the better side); and multi-leg/relative-value
   constructions (need shorts).

## Tooling
BUILT `MonthEndFixReversionLongSignal` (`discovery/tools/fix_flow_signals.py`) — mask + ATR geometry
only, conforms to `SignalModule`; registered in TOOL_REGISTRY. Reused BUILT `observe_long_capture`,
`make_time_exit_predicate`, `build_null_signal_evaluation`. No canonical-core change.

## FLAGS (code not merged)
None new. Carries the standing FLAG-1 (long-only blocks the stronger side — here the UP-push-into-fix
SHORT, which reverts down; operator/human-gated canonical-core change) and the
`A1Config.time_exit_bars`-unwired flag (arcs 1005/3004; worked around via the BUILT `ExitPredicate`).
Drivers scratch `_disco3_work/arc3008_observe_fix.py`, `arc3008_capture.py`, `arc3008_wfo.py`.

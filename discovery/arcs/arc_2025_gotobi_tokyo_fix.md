# arc_2025 — Gotobi (五十日) Tokyo-fix settlement flow

**Chat:** 2000s | **Date:** 2026-06-05 | **Disposition:** KILL (obs cheap-kill, §5d — real but sub-cost) | **OOS:** untouched

## (a) Log read

(Carried from arc 2024, same chat, re-anchored.) Corpus state: the 4-component PORTFOLIO book is the
deepest result (mean-positive, ~3 independent bets, temporally robust, NOT all-folds-positive); the
leg-hunt / portfolio route is PROVEN closed (arc 2019/2022/3021 portfolio-math); option-B (a thick
fold-resolving standalone) is closed on 4 constructions (fbr-depth 1025, cross-sectional me 2018,
thick-by-design session 1026, me-window 2024); the deployability lever is the operator path-A gate call.
The fix-flow family already has one data point: **arc 3008 — the month-end WMR 16:00-London fix reversion
is REAL (beats placebo + null) but SUB-COST on H1 majors → KILL.** Open question this arc attacks: is there
a *different*, intra-month fix flow that is large enough to clear cost?

## (b) Idea + because

The **Gotobi (五十日) effect** is a documented, calendar-anchored, regime-ORTHOGONAL forced flow distinct
from the month-end fix: on days divisible by 5 (5/10/15/20/25 + month-end), Japanese importers/corporates
settle trade invoices in USD, creating USD-buying demand into the **09:55 JST Tokyo fix** → USD-vs-JPY
tends to DRIFT UP through the Tokyo morning into the fix. It is (i) genuinely novel vs the corpus (3008 was
the month-end *London* WMR fix; this is the intra-month *Tokyo* fix), (ii) calendar-driven → orthogonal to
the price-move-triggered risk-off factor that ties every existing component together, and (iii) thick
(~6 events/month). If real AND > FundedNext cost, a long-USDJPY-into-the-Tokyo-fix on gotobi days is a
candidate decorrelated component.

## (c)/(d) Observe + cheap-kill (§5d — gross; NOT a gate)

Driver: `discovery/_disco2_work/arc_2025_gotobi_obs.py`. Timing: 09:00 JST = 00:00 UTC, 09:55 JST = 00:55
UTC → the H1 bar timestamped 00:00 UTC (09:00–10:00 JST) SPANS the fix; its open→close return is the cheap
proxy for the pre-fix Tokyo-morning drift. 4 JPY pairs (USDJPY + EUR/GBP/AUD-JPY crosses), H1, IS 2010-2020,
gotobi vs non-gotobi (simple day∈{5,10,15,20,25,month-end} rule). Cost proxy: 1.5×spread + 0.5p slippage +
~0.5p commission-equiv.

```
pair    non-gotobi mean   GOTOBI mean   frac+    gotobi EXCESS   ~RT cost   verdict     per-yr+   2015     2018
USDJPY     -0.134p          +0.101p     0.513      +0.235p        2.96p     SUB-COST    6/11    +0.505   -0.942
EURJPY     -0.619p          +0.178p     0.520      +0.797p        3.87p     SUB-COST    8/11    +0.398   +1.049
GBPJPY     -0.970p          +0.114p     0.504      +1.084p        6.37p     SUB-COST    7/11    +1.685   +0.171
AUDJPY     -0.868p          +0.277p     0.497      +1.145p        4.64p     SUB-COST    6/11    +6.214   +0.708
```

## (e) Diagnose

1. **The Gotobi effect is REAL and mechanism-consistent.** On ALL 4 pairs the gotobi-day fix-bar mean is
   POSITIVE while the non-gotobi Tokyo-morning mean is NEGATIVE — i.e. the normal Tokyo morning drifts down,
   but on gotobi days the documented USD-settlement demand flips it up. The gotobi EXCESS is POSITIVE on all
   4 pairs (+0.24 to +1.14p), strongest on the JPY crosses (EUR/GBP/AUD-JPY) where the USD-buy → JPY-sell
   also lifts the cross. EURJPY is even +2018 (+1.05p) and 8/11 folds positive. The mechanism is confirmed
   in 2010-2020 FX data.

2. **DECISIVE: it is ~10–60× SUB-COST.** The gotobi-day drift is +0.10 to +0.28p; the H1 round-turn cost is
   ~3–6p. The drift — and even the gotobi EXCESS — is an order of magnitude or two below the cost line on
   every pair. frac+ ≈ 0.50–0.52 (a tiny mean drift, not a reliable directional bet).

3. **Resolution-invariant (M1 won't rescue it).** The excess sits in a window that is mostly *pre*-fix
   (09:00→09:55 of the 09:00–10:00 bar); a finer M1 entry that captures only the pre-fix leg would at best
   modestly increase the +0.2–1.1p excess, still far below the 3–6p RT cost. Sub-cost is robust to
   resolution (cf. arc 2002: the gap continuation was TF-invariant).

## (f)/(g) Verdict — KILL (obs cheap-kill, §5d)

A real, documented, mechanism-confirmed forced flow, but ~10–60× sub-cost on FundedNext H1 → not tradeable.
No non-coin-flip, above-cost entry exists, so §5f does not bite (nothing to put on the engine). OOS untouched.

## (i) What this closes + lesson

- **Closes the Gotobi / intra-month Tokyo-fix flow** — real but sub-cost, directly complementing arc 3008
  (month-end London WMR fix, also real-but-sub-cost). **Generalized lesson:** the *fix-flow family* (both
  the month-end WMR London fix AND the intra-month Tokyo gotobi fix) is REAL but **sub-cost** — fix-driven
  over-extensions are sub-pip-to-~1-pip, an order of magnitude below FX spread+commission, so they cannot be
  captured net on a retail-cost broker. Fix-flow as a tradeable-edge family is closed (2 independent fixes,
  same verdict).
- Reinforces the unified theory in a new way: even a genuinely orthogonal, calendar-driven, thick forced
  flow is sub-cost — the corpus's tradeable edges (gap/me/fbr) survive cost only because they condition on a
  *large* price displacement (≥1 ATR), whereas pure calendar-fix flows are too small. Operative frontier
  unchanged (operator path-A gate call). Components UNCHANGED.

**Tooling:** no new tool (self-contained obs driver). No TOOL_REGISTRY append. **FLAGS:** none.
Driver: `_disco2_work/arc_2025_gotobi_obs.py`.

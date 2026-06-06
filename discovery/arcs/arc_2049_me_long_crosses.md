# arc_2049 — Month-end reversion LONG on NON-USD crosses (a decorrelated +2014/15/16 leg?)

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** KILL (obs cheap-kill, §5d) · **Disposition:** KILL

## Log synthesis (step a)
Continuing from arc 2048 (same chat; flow-efficacy regime closed). Pulled main (merged concurrent 1000s arc 1049). State: the 4-component reversion book is mean-positive, never all-folds-positive; under honest §5f exits it collapses to **me_long-SOLO** (1046). The unfound deployable lever is a **+2014/15/16 OR +2018 regime-orthogonal component** (me_short already gives a fragile +2018). me_long (USD majors) DIES in the contiguous **2014/15/16 strong-USD block** (1012); arc 2048 just closed the flow-efficacy/state axis for separating it. Universe-portability is a documented hard prior — **corpus edges are USD-major-specific (1018 gap, 2031/1022 fbr) — except the gap-fill, which is *natively* JPY-cross** (1006). me_**short** broad-cross was tested (1021: JPY carries 2016, XCROSS adds 2015, disqualified on robustness); **me_LONG on crosses for +2014/15/16 is genuinely untested.**

## Idea (step b — with a mechanism)
me_long fails 2014/15/16 because on a USD major a big DOWN move into month-end IS the USD trend → the WMR reversion gets overrun. On a **non-USD cross** (EURJPY/EURGBP/…) a down-move into month-end is NOT USD-driven → the same forced month-end rebalancing reversion might HOLD in 2014/15/16 → a decorrelated **+2014/15/16 leg** (the precise unfound spec, 1015/1020). *Because:* me is a FLOW edge and its sibling flow (gap-fill) ports to JPY crosses — so unlike the structural fbr (2031, dead on crosses), me_long *might* port. **Falsifiers (§5d):** (1) month-end EXCESS drift ≈ 0 vs a matched random-day control (no real flow); (2) the 2014/15/16 block NOT positive (wrong-sign / thin-tail regime-luck). Either → KILL.

## What I did (steps c–d — cheap-kill observation)
Population = D1, 6 non-USD crosses — JPY group {EURJPY, GBPJPY, AUDJPY} + XC group {EURGBP, EURAUD, AUDNZD}, 2010–2020. Driver: `discovery/_disco2_work/arc_2049_me_long_crosses_obs.py`. Ex-ante (canonical `_month_end_into_move` + `observe_long_capture`): **me_long fire** = last trading day of month with `into = (close[i]−close[i−2])/ATR ≤ −1.0`; honest +1R capture + 2-bar drift; **control** = the SAME big-down move on a NON-month-end day (matched-magnitude baseline → month-end EXCESS).

## What happened — FALSIFIED both ways

| population | n | capture | 2-bar drift (ATR) |
|---|---|---|---|
| month-end long (all crosses) | 153 | 0.5294 | **−0.044** |
| — JPY group | 74 | 0.5946 | −0.106 |
| — XC group | 79 | 0.4684 | +0.015 |
| random-day control (same big-down) | 3259 | 0.4854 | **−0.036** |
| **month-end EXCESS** | | | **−0.008** |

1. **Falsifier 1 — no real flow: month-end EXCESS drift = −0.008 ATR** (ME −0.044 vs control −0.036). The month-end timing is NOT load-bearing on crosses — a big down-move into month-end reverts no more than the same move on a random day, vs the **+0.249 ATR month-end excess on USD majors** (1011). The JPY group's higher *capture* (0.5946) carries *negative* drift (−0.106) = the take-the-loss-label-vs-forward-move artifact, not a reversion.
2. **Falsifier 2 — 2014/15/16 NET NEGATIVE:** block drift −0.132 (2014 −0.092 / 2015 +0.282 / 2016 −0.809) — same 2014-16-bad/2015-ok pattern as USD majors, **2016 worse on crosses**. Per-pair, **5 of 6 pairs negative** in the block (EURJPY −0.95, GBPJPY −0.38, EURGBP −0.29, AUDNZD −0.18, AUDJPY −0.02); the lone positive (EURAUD +1.18) is **n=5 thin-tail** — the 2011/3011/2031 pair-mix/thin-tail confound. Not the +2014/15/16 leg.

## Why it fails (the mechanism)
The WMR month-end rebalancing reversion is **USD-major-specific** — the forced rebalancing flow is concentrated in USD-vs-major hedge adjustment; non-USD crosses are the residual, thinner cross of two such legs, and the month-end timing adds essentially nothing (EXCESS ≈ 0). This **generalizes arc 1018's lesson** ("stripping the USD/carry exposure removes the edge — edge ≡ factor") from the gap-fill to the month-end flow, and confirms 2031/1022 (corpus edges don't port off USD majors) on the FLOW axis. It also refines 1021: me_**short** cross flow was at least real-but-disqualified, whereas me_**long** cross flow has ~0 month-end excess on the drift (economic) measure — the down/long side does not port even as far as the up/short side did. The strong-USD-2014/16 hole is NOT a USD-major artifact removable by changing universe — the crosses carry their own (worse) 2016 hole, so the "decorrelate from USD by changing universe" route is closed for me_long as it was for the gap-fill (1018) and fbr (2031).

## Verdict + what this closes
**KILL (obs cheap-kill, §5d).** Month-end EXCESS ≈ 0 (no tradeable cross flow) and the 2014/15/16 block is net-negative/thin-tail → **§5f does not bite** (no real month-end edge to put on the engine; the capture excess is a label artifact with negative drift). No engine / null / council spent (matches 2047/2048/2031/1018). Closes the **me_long-on-non-USD-crosses** route to the +2014/15/16 leg; the month-end reversion is USD-major-specific (flow-axis confirmation of universe-specificity). me_long UNCHANGED (PORTFOLIO, USD majors); the +2014/15/16 component remains unfound. Components UNCHANGED (all 4 PORTFOLIO; deploy core = me_long-solo per 1046). Lever = operator path-A. Deployable-system count = 0.

**Tooling:** no new BUILT tool — canonical loader + BUILT `_month_end_into_move`/`observe_long_capture`; no TOOL_REGISTRY append (inline scratch). **FLAGS:** none. OOS untouched.

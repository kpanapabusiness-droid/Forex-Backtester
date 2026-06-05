# arc 1015 — 3-way PORTFOLIO combination WFO (gap-fill 1006 + month-end 1011 + failed-breakdown 1013)

**Chat:** 1000s · **Date:** 2026-06-05 · **Disposition:** KILL (combined 3-way book; the three
components are UNCHANGED — reproduced exactly, retain PORTFOLIO) · **Council:** none (not a PASS
survivor, not an idea-fork / diagnosis — a direct quantitative combination test, exactly arc 2006's
situation).

## Idea + why (log-seeded — the gated next step)

After 26 arcs the directional space is comprehensively closed; the programme has exactly **three
net-positive long-only PORTFOLIO components**, all reproduced and audited:

1. **gap-fill (1006/1009)** — weekend gap-down-fill long, 5 JPY crosses, H4, 24-bar time exit. IS mean **+0.685%**.
2. **month-end (1011/1012)** — month-end mechanical-rebalancing reversion long, 7 USD majors, D1, `sl_only`+2-bar. IS mean **+0.232%**.
3. **failed-breakdown (1013)** — DEEP failed-breakdown RECLAIM (stop-run reversal) long, 7 USD majors, H4, `sl_plus_trailing_atr`. IS mean **+1.854%**.

Arc 2006 ran the **2-way** combination (gap-fill + month-end) → KILL, *provably* blocked by the
mutually-negative **2015** fold, and specced the fix: a **≥3rd component selected for
fold-complementarity** (positive on the 2-way book's NEGATIVE folds — 2015/16/18/20), NOT average
correlation. **Arc 1013 IS exactly that 3rd component** — positive in 2015/16/20 (3 of the 4 bleed
years), negative only in 2018 — and explicitly flagged: *"run the 3-way combination WFO, ranking by
ROI on the 2-way book's negative folds, gated all-folds-positive on the combined book."* This is item
#5 on the dispatch frontier ("once ≥3 net-positive, regime-complementary components exist, the
combined-book all-folds-positive WFO is the deployable gate"), now unblocked for the first time — the
single most decision-relevant unrun experiment, decisive either way (a PASS candidate — the
programme's first — or a precise spec for the missing piece).

## Method (CALLED canonical apparatus; no re-roll)

Each component reproduced via its **REGISTERED signal** over the SAME canonical IS folds
(`build_v3_folds`, `is_days≥365` → 10 folds, OOS-year anchors 2011–2020, fold 6 = 2015, fold 9 =
2018), scored **solely** by `MultiPairBacktester` via `A1Architecture`/`ArcFoldRunner`
(`run_config_over_folds`), FundedNext costs ON, SL-first. Per-fold ROI combined with the BUILT
`combine_fold_roi` (`fit_weights` on IS → `combine_fold_rois`), weights **fit on IS and FROZEN**;
the all-folds-positive judge applied to the COMBINED book. Driver (scratch, reproducible):
`PYTHONPATH=. py _disco_work/arc1015_combo3.py`. histdata_root = `C:\Users\panap\histdata_backup`.

**Arc-10 reproduction discipline (don't trust transcription):** all three reproduce **byte-exactly**
against their records — gap +0.685%, month-end +0.232%, failed-breakdown +1.854% (9/10, per-fold
vector identical). The failed-breakdown match required **`trail_enabled=True`** (the A1Config default,
left ON in arc 1013's committed config alongside `exit_policy="sl_plus_trailing_atr"` — a double-trail
quirk; `trail_enabled=False` gives +2.084%/8-of-10). Reproduced faithful to the committed component
definition. (Minor FLAG below.)

## Per-fold IS (ROI %, fold order 2011..2020)

| year | gap-fill | month-end | failed-brk |
|---|---|---|---|
| 2011 | −0.07 | +0.40 | +7.55 |
| 2012 | +8.23 | +0.29 | +3.05 |
| 2013 | −2.06 | +0.96 | +0.91 |
| 2014 | +2.94 | −0.23 | +0.19 |
| **2015** | **−4.19** | **−1.14** | **+3.17** |
| 2016 | +3.20 | −0.51 | +2.55 |
| 2017 | +0.53 | +0.34 | +1.23 |
| **2018** | **−6.79** | **+0.90** | **−4.20** |
| 2019 | +7.45 | +1.16 | +0.05 |
| 2020 | −2.39 | +0.15 | +4.03 |
| **mean** | **+0.685** | **+0.232** | **+1.854** |

## Combined book (weights fit on IS, frozen)

| weighting | weights (gap/me/fbr) | mean | neg folds | worst | all-folds-pos |
|---|---|---|---|---|---|
| equal | 0.333 / 0.333 / 0.333 | +0.924% | **3/10** (2013, 2015, 2018) | −3.360% | **NO** |
| **risk-parity** | 0.106 / 0.726 / 0.168 | **+0.552%** | **2/10** (2015, 2018) | **−0.770%** | **NO** |

The 3rd component worked **exactly as arc 1013 predicted**: positive in 2015/16/20, it cut the 2-way
risk-parity book (arc 2006: 4/10 neg, worst −1.53%) down to **2/10 neg, worst −0.770%** — a dramatic
tail improvement, mean still positive. But the book is **NOT all-folds-positive → KILL** (§11).

## Why it's blocked — provable, combination-method-invariant

The block is **not weighting-sensitive** and does **not** rest on the linear-combination approximation:

- **2018** is positive in only ONE component (**month-end +0.90**); gap (−6.79) AND fbr (−4.20) are
  both deeply negative. The max any convex weighting can achieve on 2018 = all weight on month-end =
  +0.90 — i.e. month-end **solo**, which is itself only 7/10. A persistent strong-USD trend year:
  breakdowns are real (not swept → fbr's own named 2018 weakness) and JPY-cross gaps don't fill.
- **2015** is positive in only ONE component (**failed-breakdown +3.17**); gap (−4.19) AND month-end
  (−1.14) are both negative. The max convex weighting on 2015 = all weight on fbr = +3.17 — i.e. fbr
  **solo**, which fails 2018. CHF-depeg / USD-bull risk-off year.
- **2015 wants weight on fbr; 2018 wants weight on me — opposite directions** — and neither fbr-solo
  nor me-solo is all-folds-positive. **No single convex weighting makes both 2015 AND 2018 positive,
  because no single component is positive in both.** A small co-simulation margin interaction cannot
  flip a −3.36% / −0.77% fold positive, so the verdict is robust to the linear-combination limitation.

This is the arc-2006 mutually-negative-fold pre-screen, sharpened: the route is now blocked by exactly
**two folds (2015 & 2018)** under the best (risk-parity) weighting — both persistent strong-USD /
risk-off years where every long flow-reversion + structural-reclaim edge bleeds simultaneously.

IS not all-folds-positive → **OOS deliberately NOT touched** (§4 + holdout preservation; the
components' OOS status is unchanged — month-end pristine, gap-fill/fbr already measured in their own arcs).

## Verdict

**FAIL the sole judge (combined 3-way book) → KILL.** Not deployable, *provably* blocked by ≥1 fold
(2015 & 2018) where no convex combination of these three can be positive. **The three components are
UNCHANGED** (reproduced exactly, not re-tested/weakened) — they retain PORTFOLIO status; **no new
`portfolio-candidates/` entry** (would triple-count 1006/1011/1013). The portfolio **thread stays
ACTIVE** with a razor-sharp 4th-component spec.

## The precise 4th-component spec (the high-value output)

A 4th component must be **positive in BOTH 2015 AND 2018** (the two surviving mutually-negative folds).
These are persistent **strong-USD / risk-off** years; in them, gap-fill (JPY-cross reversion), month-end
(USD-bull block), and failed-breakdown reclaim (breakdowns real, not swept) ALL lose — because each is a
**long reversion / reclaim** structure, and that whole family inherits the same risk-off tail. A component
positive in 2015 & 2018 is therefore structurally a **strong-USD-trend / risk-off-positive** construction
— most naturally a **SHORT** (of the weak non-USD legs) or a **USD-long trend** leg — which the long-only
reversion menu **cannot** provide. This is precisely the regime-orthogonal leg the arc-2006/3004
escalation + FLAG-1 named, now pinned to a **2-fold target (2015 & 2018)**. The portfolio route is
**exactly one regime-orthogonal component away** from a deployable all-folds-positive book.

## Threads / lessons

1. **The 3-way combination is the strongest portfolio book the corpus has produced** (risk-parity 2/10
   neg, worst −0.77%, mean +0.55%) — but still NOT deployable. Two near-zero-correlated mean-positive
   components plus a genuinely regime-orthogonal third (fbr, positive 2015/16/20) **halves** the
   2-way's negative-fold count yet cannot clear the gate, because the *complementary* component (fbr)
   does not also cover 2018.
2. **The block is now pinned to two specific folds with a structural cause** (2015 & 2018 strong-USD /
   risk-off) and a precise, falsifiable spec for what closes it — far tighter than arc 2006's "need a
   3rd." A 4th long reversion edge would re-inherit the tail and not help (the same lesson arc 2006
   reached, now confirmed by adding the most complementary long available and STILL failing).
3. **The escalation is at its sharpest form:** a deployable book is one component away, and that
   component is structurally short/trend (positive in strong-USD/risk-off), not long-reversion →
   **shorts/second-leg unlock (FLAG-1) is the specific named blocker**, with a 2-fold acceptance test
   (positive 2015 & 2018), not a vague "highest leverage."
4. **Selection criterion refined (extends arc 2006 lesson #2):** rank a candidate not just by ROI on
   the book's negative folds in aggregate, but by whether ONE component covers EACH residual
   mutually-negative fold. Here fbr covered 2015/16/20 but not 2018; the next must cover the
   *intersection* the current set leaves open (2015 ∩ 2018) — i.e. a component positive in the years
   where the existing complementary legs disagree.
5. **Methodological (reproduction):** an arc's committed config can carry a non-obvious default
   (`trail_enabled=True` left on under an `exit_policy`) — Arc-10 byte-reproduction caught it; always
   reproduce the recorded per-fold VECTOR, not just the mean (the mean matched at +2.08% under the
   wrong flag too, but the per-fold vector and fold-count did not).

## FLAGS (code not merged)

- **None requiring the canonical core.** Carries the standing `A1Config.time_exit_bars`-unwired flag
  (arcs 1005/3004; worked around via the BUILT `make_time_exit_predicate`) and FLAG-1 (long-only
  blocks the regime-orthogonal short/trend leg the portfolio route now specifically needs for 2015 &
  2018).
- **Config-quirk note (not a code change):** arc 1013's committed config runs **double-trailing** (legacy
  `TrailManager` ON via `trail_enabled=True` default + `exit_policy="sl_plus_trailing_atr"`). It is the
  committed component definition and reproduces +1.854% exactly, so I combined it as-is; a future arc
  re-optimising fbr's exit should be aware the recorded number includes the legacy trail.
- **Linear-combination LIMITATION (FLAGGED, immaterial here):** `combine_fold_roi` does a per-fold
  linear combination, not a single co-simulated equity curve; month-end (D1 USD majors) and
  failed-breakdown (H4 USD majors) share the USD-major universe and could occasionally hold
  simultaneous same-pair positions (shared margin / daily-DD), which the linear book ignores. A
  single-engine co-simulation is **impossible with the canonical runner** (the components span two
  timeframes — one A1 run iterates a single primary-TF panel). The verdict rests on per-fold ROI
  SIGNS in 2015/2018 that are large and unambiguous, so the limitation cannot flip it.

## Tooling

No new BUILT tool. Reused BUILT `combine_fold_roi` (arc 2006), `WeekendGapFillLongSignal` (2001),
`MonthEndReversionLongSignal` (1011), `FailedBreakdownReclaimLongSignal` (1013),
`make_time_exit_predicate` (1005) — all CALLED, scoring stayed canonical (`ArcFoldRunner` →
`MultiPairBacktester`). Driver scratch `_disco_work/arc1015_combo3.py`.

# arc 1020 — 4-way PORTFOLIO combination WFO (gap 1006 + me-long 1011 + fbr 1013 + me-SHORT 1019)

**Chat:** 1000s · **Date:** 2026-06-05 · **Disposition:** KILL (combined 4-way book; the four components
are UNCHANGED — reproduced + retain PORTFOLIO) · **Council:** none (not a PASS survivor, not an
idea-fork/diagnosis — a direct quantitative combination test, exactly arcs 1015/2006/2008/3009).

## Idea + why (log-seeded — the gated next step, now unblocked)

Arc 1015 (this chat) ran the 3-way combination (gap + me-long + fbr) → KILL, **provably blocked by 2015 &
2018**: 2015 positive in ONLY fbr (+3.17), 2018 positive in ONLY me-long (+0.90, weak) → 2015-wants-fbr vs
2018-wants-me-long are mutually exclusive, 0/5151 convex weightings all-folds-positive (worst −0.77%). The
razor-sharp 4th-component spec: **positive in BOTH 2015 & 2018** — structurally a strong-USD/risk-off SHORT.

**Arc 1019 (this chat) found exactly that** — the month-end reversion SHORT (the untested mirror of me-long):
**robustly 2018-positive** (+0.86, every exit-except-tp2r / thr / pair-drop) with a fragile-but-present 2015
tilt (+0.40). For the first time **2018 has TWO positive contributors** (me-long +0.90, me-short +0.86) and
**2015 has TWO** (fbr +3.17, me-short +0.40) — the arc-1015 deadlock (each binding fold positive in only one
component) is broken. This is the single most decision-relevant unrun experiment: does the 4th leg finally
clear an all-folds-positive book?

## Method (CALLED canonical apparatus; no re-roll)

Each component reproduced LIVE via its REGISTERED signal over the SAME canonical IS folds (`build_v3_folds`,
`is_days≥365` → 10 folds, OOS-year anchors 2011–2020, fold 6=2015, fold 9=2018), scored **solely** by
`MultiPairBacktester` via `A1Architecture`/`ArcFoldRunner` (`run_config_over_folds`), FundedNext costs ON,
SL-first. Per-fold ROI combined with the BUILT `combine_fold_roi` (`fit_weights` IS → `combine_fold_rois`,
frozen); all-folds-positive judged on the COMBINED book. **Convex grid search over the full 4-simplex (step
0.02).** Driver `_disco_work/arc1020_combo4.py`. histdata_root = `C:\Users\panap\histdata_backup`.

**Arc-10 reproduction discipline:** me-long, fbr, me-short reproduce **byte-exact** vs their records (max-dev
≤0.005pp: me-long +0.232%, fbr +1.854% 9/10 with `trail_enabled=True`, me-short +0.683%). **gap shows a
per-fold mismatch** (mean +0.633% vs recorded +0.685%; signs all match, magnitudes differ ~1–2pp; neither
`gap_hours`=20 nor 36 reproduces arc 1015's exact vector — a config nit, FLAGGED below). **The verdict is
invariant to it:** re-run with the AUDITED recorded gap vector (arc 1015 byte-exact) gives the SAME KILL, so
the recorded gap vector is used as authoritative.

## Per-fold IS (ROI %, fold order 2011..2020)

| year | gap (rec) | me-long | fbr | me-SHORT |
|---|---|---|---|---|
| 2011 | −0.07 | +0.40 | +7.55 | +3.39 |
| 2012 | +8.23 | +0.29 | +3.05 | +1.69 |
| 2013 | −2.06 | +0.96 | +0.91 | −0.90 |
| 2014 | +2.94 | −0.23 | +0.19 | +0.98 |
| **2015** | −4.19 | −1.14 | **+3.17** | **+0.40** |
| 2016 | +3.20 | −0.51 | +2.55 | −0.91 |
| 2017 | +0.53 | +0.34 | +1.23 | −0.68 |
| **2018** | −6.79 | **+0.90** | −4.20 | **+0.86** |
| 2019 | +7.45 | +1.16 | +0.05 | +1.29 |
| 2020 | −2.39 | +0.15 | +4.03 | +0.71 |
| **mean** | +0.685 | +0.232 | +1.854 | +0.683 |

## Combined book — convex search (authoritative: recorded gap vector)

| weighting | gap/me-long/fbr/me-short | mean | neg | worst | AFP |
|---|---|---|---|---|---|
| equal | 0.25/0.25/0.25/0.25 | +0.85% | 3/10 | −1.81% | NO |
| risk-parity | 0.08/0.52/0.12/0.28 | +0.59% | 2/10 | −0.38% | NO |
| **best max-min (convex search)** | **0.00/0.60/0.16/0.24** | +0.62% | **2/10** | **−0.115%** | **NO (0/all weightings)** |

Best max-min per-year: `2011 +2.26 · 2012 +1.07 · 2013 +0.50 · 2014 +0.13 · 2015 −0.08 · 2016 −0.12 ·
2017 +0.24 · 2018 +0.08 · 2019 +1.01 · 2020 +0.91`.

## The headline — the 2018 WALL is BREACHED in combination

The 4th component did what arc 1015's spec demanded. In the 3-way, **2018 was the unsolvable binding fold**
(positive in only the weak me-long +0.90; gap −6.79 & fbr −4.20 both deeply negative; the convex max on 2018
was me-long-solo) — and it killed 12+ standalone routes. **me-short gives 2018 a second, robust contributor,
and at the book optimum 2018 is now POSITIVE (+0.08).** The worst fold improved **7×** (−0.77% 3-way →
−0.115% 4-way); this is the **strongest book the corpus has produced**, within ~0.1% of deployable.

## Why it's STILL blocked — the constraint moved to 2015 & 2016 (both marginal)

0/all convex weightings are all-folds-positive; the residual block is now **2015 (−0.08) & 2016 (−0.12)**,
and it is **robust** — confirmed under BOTH gap vectors AND under EVERY me-short exit from its 1019 menu
(sl_only te2/te3/te5, tp_2r, partial-runner all give 0 all-folds-positive; partial-runner is the best,
the only one keeping 2018 positive at the optimum):

- **2016** is the new structural knot: **both month-end legs are negative there under every exit**
  (me-long −0.51, me-short −0.91/partial) — 2016 relies entirely on gap (+3.20) & fbr (+2.55), but
  weighting those up **breaks 2018** (gap & fbr both deeply negative in 2018). me-short relieved the 2018
  constraint, which let the search lift 2016 *almost* to zero — but not over.
- **2015** is dragged by me-long (−1.14, weighted heavy 0.60 because the book leans on it for 2018) plus
  me-short's 2015 being its **fragile leg** (+0.40 only; the threshold/GBPUSD-fragility flagged in arc 1019).
  fbr (+3.17) nearly covers it but the search can't weight fbr higher without breaking 2018.

IS not all-folds-positive → **OOS deliberately NOT touched** (§4; components' OOS unchanged — me-long /
me-short pristine, gap / fbr already measured in their own arcs).

## Verdict

**FAIL the sole judge (combined 4-way book) → KILL.** Not deployable. **The four components are UNCHANGED**
(reproduced, not re-tested/weakened) — they retain PORTFOLIO; no new `portfolio-candidates/` entry (would
quadruple-count). But this is the route's biggest advance: the 2018 wall is breached, the book is 7× tighter
(worst −0.11%), and the block is reduced to two MARGINAL folds with a precise structural cause.

## The precise 5th-component spec (the high-value output)

A 5th component must be **positive in 2015 AND 2016 without dragging 2018** (the two new residual
mutually-negative folds, both marginal at −0.1%). Critically it must **NOT be another month-end / reversion
leg** — both month-end legs (long AND short) and the gap-fill are reversion structures, and 2016 is negative
in both month-end legs under every exit. 2016 (Brexit / US-election) is positive only in gap & fbr, which
2018 caps; 2015 (CHF-depeg / USD-bull) only in fbr & the fragile me-short. So the 5th leg is a **2016-AND-
2015-positive, 2018-neutral-or-positive edge that is structurally distinct from the reversion/gap family** —
e.g. a different mechanical-flow event covering those years, or a structural edge with a 2016/2015 regime
profile. Equivalently: strengthen me-short's 2015 leg out of its GBPUSD/low-threshold fragility (which would
lift 2015 above zero) AND find a non-reversion 2016 contributor. The route is **~0.11% and one
non-reversion, 2015&2016-positive component from a deployable all-folds-positive book** — a far better
position than arc 1015's "blocked at 2018 by −0.77%."

## Threads / lessons

1. **A mechanical-flow SHORT breached the 2018 wall the entire directional/structural menu could not** —
   12+ routes (structure/trend/flow/vol shorts, relative-value, deep-continuation, USD-neutral, weekly,
   end-of-week) died trying to make 2018 positive standalone; the answer was the *short side of the proven
   month-end flow*, and it only had to work IN COMBINATION (positive in 2018, not all-folds-positive solo).
   **The portfolio gate rewards regime-complementarity, not solo strength** — exactly the §11 thesis.
2. **The residual block is the SAME structural tension, shifted one fold over:** fbr is wanted by
   2015/2016/2020 but rejected by 2018; me-short relieved 2018 (a non-fbr 2018 contributor), so the binding
   fold moved from 2018 to 2016 — the next fold where the book's positive contributors (gap, fbr) are the
   ones 2018 rejects. The route advances fold-by-fold; each regime-orthogonal leg neutralizes one wall and
   exposes the next-marginal one.
3. **Selection criterion (extends arc 1015 lesson #4):** the 5th component must cover the intersection the
   current FOUR leave open — 2015 ∩ 2016 — and must be NON-reversion (the reversion family is saturated:
   gap, me-long, me-short all share the 2016 negative and the risk-off tail).
4. **me-short's documented 2015 fragility (arc 1019) is now load-bearing at the book level** — its +0.40
   in 2015 is the difference between 2015 being −0.08 and positive. A non-fishing IS-fold improvement of
   me-short's 2015 robustness (a real refinement, not exit-selection-on-the-gate) is a concrete sub-thread.

## FLAGS (code not merged)

- **None requiring the canonical core.** Carries the standing `A1Config.time_exit_bars`-unwired flag
  (worked around via `make_time_exit_predicate`).
- **gap reproduction nit (FLAGGED, immaterial):** the registered `WeekendGapFillLongSignal(threshold_atr=0.5,
  gap_hours=36)` over the 5 JPY crosses gives mean +0.633% with a per-fold vector ~1–2pp off arc 1015's
  recorded byte-exact gap (gap_hours=20 also misses). The exact arc-1006 gap config is not pinned by the
  signal params alone; arc 1006/1009/1015's recorded vector is authoritative and the combination verdict is
  invariant to it (re-run with the recorded vector = same KILL). A future arc that needs gap re-run should
  pin arc 1006's exact instantiation.
- **Linear-combination LIMITATION (FLAGGED, immaterial here):** `combine_fold_roi` is per-fold linear, not a
  single co-simulated equity curve; me-long/me-short (D1 USD majors) and fbr (H4 USD majors) share the
  USD-major universe and could hold simultaneous same-pair positions (shared margin/daily-DD) the linear book
  ignores. Single-engine co-sim is impossible with the canonical runner (components span D1 + H4). The
  verdict rests on per-fold ROI signs that are unambiguous (2015/2016 marginal-negative under every weighting
  and every me-short exit), so the limitation cannot flip it.

## Tooling

No new BUILT tool. Reused BUILT `combine_fold_roi` (2006), `WeekendGapFillLongSignal` (2001),
`MonthEndReversionLongSignal` (1011), `FailedBreakdownReclaimLongSignal` (1013),
`MonthEndReversionShortSignal` (1019), `make_time_exit_predicate` (1005) — all CALLED, scoring canonical
(`ArcFoldRunner` → `MultiPairBacktester`). Driver scratch `_disco_work/arc1020_combo4.py`.

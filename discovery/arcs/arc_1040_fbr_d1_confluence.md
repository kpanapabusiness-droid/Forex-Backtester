# arc 1040 — fbr × Daily (D1) structural confluence: does HTF support rescue fbr's 2018 fold?

**Chat:** 1000s · **Range:** 1000-1999 · **Date:** 2026-06-06 · **Verdict: KILL** (refinement; fbr UNCHANGED, PORTFOLIO) · OOS never touched.

## Frame (step a, fresh eyes — honest-era log)
Resumed 1000s at highest-in-range 1039 → opened 1040. Corpus state read from DISCOVERY_LOG + LESSONS +
TOOL_REGISTRY: 4 PORTFOLIO components (gap 1006 / me_long 1011 / fbr 1013 / me_short 1019); a mean-positive,
cost-robust (κ=3.32, arc 3022) 4-way book that is NOT all-folds-positive (binding folds 2015/2016 at the
optimum, arc 1020; 2018 breached by me_short); the §11 independent-verification programme COMPLETE (signal
2034/2035/1037, outcome 2036/2037/2038/1038/1039, cost 2039); the edge-hunt declared structurally closed by
~6 recent arcs (1032/1033/1034/1035/2032/2033); RV/market-neutral definitively closed WITH shorts enabled
(2003/2010/2018 — doubled-cost-vs-coin-flip). Per §2 + the arc-3004 council warning ("apparatus incapable is
a seductive, search-ending conclusion"), I made a genuine fresh-mechanism attempt rather than defaulting to a
diagnostic.

## Idea + because (log-seeded, the corpus's top near-miss)
The per-year component table (`validate_4way_book.py`) shows **fbr — the corpus's strongest edge — is one
fold from all-folds-positive PASS**: its SOLE clearly-negative IS fold is 2018 (−4.20%; all other years
positive). arc 2014 diagnosed fbr-2018: in a strong-USD regime, failed breakdowns become REAL breakdowns
(the reclaim fails). HYPOTHESIS: a reclaim occurring **AT a Daily (D1) swing-low level** — support defended
on a higher timeframe — may HOLD even in 2018, where generic H4-level reclaims fail. This is the
"multi-timeframe structure as a setup" sub-lane the corpus explicitly flagged untested (log lines 1275/1464),
and the arc-1013 template (structure × magnitude) extended with a cross-TF axis. It attacks the single
highest-value target in the corpus: if D1-confluence rescues 2018 without breaking the other folds, fbr goes
from 9/10 PORTFOLIO to an all-folds-positive PASS candidate. Distinct from prior fbr refinements: 1022
(cross-universe), 2030 (SAME-TF touch count), 1034 (gap × structure) — none tested a HIGHER-TF level.

## Method
- **Obs cheap-kill (§5d, gross take-the-loss capture, no cost):** canonical `Panel.from_pairs` (H4+D1, 7 USD
  majors) + committed `FailedBreakdownReclaimLongSignal` (K40/shadow1.25) fires + BUILT `observe_long_capture`
  restricted to fbr fire bars. Attached a no-lookahead D1 level (`D1 low_bid.shift1.rolling(20).min`, read from
  the most recent D1 bar dated STRICTLY before the fire day) and the H4 sweep extreme; bucketed by
  |sweep − D1_level| in D1-ATR units. Driver: `_arc1040_work/obs_fbr_d1_confluence.py`.
- **§5f engine confirmation (non-coin-flip entry → engine before FAIL):** built
  `discovery/tools/failed_breakdown_d1_confluence.py` (BUILT) = fbr AND the D1-confluence mask (band 0.5 D1-ATR);
  scored per-calendar-year (build_v3_folds folds oos_start.year≥2011, `ArcFoldRunner`→`A1Architecture`→
  `MultiPairBacktester`, FundedNext) vs generic fbr at fbr's committed exit + 2 alternatives. IS only.
  Driver: `_arc1040_work/engine_confluent_fbr.py`.

## What happened
**Obs — D1-confluence is a REAL gradient but does NOT lift 2018.** Base fbr IS capture 0.5865 (n=237). By
distance to the nearest prior D1 level: AT-level (≤0.5 ATR) cap **0.6075** (n=107, drift_med +0.77) · 0.5–1.0
**0.7429** · 1.0–2.0 0.5472 · far (>2 ATR) **0.4524** (drift_med −1.49). Fires near a D1 support capture
better; fires far from one are coin-flip-to-negative → confluence carries real structural information.
**But the 2018 cell — the entire point — reads 0.500** (confluent n=12, per-pair mixed: EURUSD 0/2, USDJPY
0/1, USDCHF 2/4): D1-confluence lifts 2018 only from 0.474 to a dead coin-flip, no positive edge.

**Engine (§5f) — confirms the obs, decisively:**

| | mean ROI | pos folds | **2018** | 2015 (book-binding) | 2016 | n_trades |
|---|---|---|---|---|---|---|
| fbr generic (committed `sl_plus_trailing_atr`) | +2.08% | 8/10 | **−4.39%** | +3.36% | +4.04% | 237 |
| fbr D1-conf [`sl_plus_trailing_atr`] | +1.42% | **9/10** | **−2.57%** | +1.00% | +4.61% | ~107 |
| fbr D1-conf [`sl_only`] | −0.12% | 5/10 | −6.16% | −3.28% | −0.36% | ~84 |
| fbr D1-conf [`sl_plus_tp_3r`] | +0.50% | 7/10 | −4.27% | −0.67% | +4.34% | ~95 |

The committed trailing exit is best (consistent with fbr's IS-selected exit). D1-confluence **mitigates**
fbr-2018 (−4.39 → −2.57, ~41% loss reduction) and flips 2019 positive (9/10), **but 2018 stays clearly
negative across all three exits → NOT all-folds-positive → NOT a PASS candidate.** And it is not a book
improvement: it WEAKENS the book's actually-binding 2015 fold (+3.36 → +1.00) while halving the pool
(237 → ~107, worse fold resolution per arc 2030), and helps 2018 — which was already breached at the book
optimum (arc 1020). Net for the book: worse where it binds.

## Why (diagnosis)
D1-level confluence offers *partial* additional defense (the −4.39→−2.57 mitigation is real; HTF support
does absorb some stop-runs even in 2018), but **strong-USD 2018 overruns structure at EVERY timeframe** — a
D1 "support" in a one-way trending regime is just a pause, broken like the H4 one. This is the THIRD
level-significance proxy tested on fbr — shadow-depth (intrinsic, arc 1025), same-TF touch-count (arc 2030),
and now cross-TF D1 confluence — and all three are collinear (each thins the pool for a modest capture lift)
and NONE rescues 2018. Confirms arc 2014's strong-USD diagnosis AND arc 1025's "fbr-2018 is regime-intrinsic,
density-invariant" — now also TF-invariant. fbr-2018 is not a level-quality deficit a better filter can fix.

## Verdict
**KILL the refinement** (§5d obs + §5f engine; no council needed — not a survivor). fbr UNCHANGED (PORTFOLIO,
8–9/10, 2018 the sole drag). No new component. The "multi-timeframe structure as a setup" sub-lane is now
TESTED and CLOSED for the reclaim family. Operative lever unchanged = operator path-A (gate governance, arc
1032) + deployment risk-appetite (1033/2033). OOS never touched.

## Lesson (new)
Cross-timeframe (D1) structural confluence is a REAL but COLLINEAR level-significance proxy on fbr — the
third after shadow-depth and same-TF touch-count — and like the others it thins the pool more than it adds
edge. It *partially* mitigates fbr's strong-USD-2018 loss (~41%) but cannot flip the fold positive: a
strong-USD trend overruns support at every timeframe, so fbr-2018 is regime-intrinsic, NOT a level-quality
deficit. Generalizes arc 2030 (touch-count collinear) and arc 1025 (depth↔strength coupled) to the
cross-timeframe axis: NO level-significance filter rescues fbr-2018. When a filter mitigates but does not
flip a binding fold AND weakens the actually-binding fold, it is a KILL, not a book swap.

## Tooling
BUILT `discovery/tools/failed_breakdown_d1_confluence.py` (`FailedBreakdownReclaimD1ConfluenceLongSignal`) —
fbr restricted to H4 reclaims at a prior D1 swing-low level (no-lookahead cross-TF confluence mask). Registered
in TOOL_REGISTRY. Reusable for cross-TF structural confluence on any H4 mask-based signal.
**FLAGS:** none. No canonical-core change. OOS NOT touched.

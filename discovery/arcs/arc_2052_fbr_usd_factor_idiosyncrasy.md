# arc 2052 — fbr × cross-sectional USD-FACTOR-trend conditioner (is fbr's 2018 failure a USD-factor property?)

**Chat 2000s · VERDICT: KILL (obs cheap-kill / diagnostic, §5d).** No engine/null/council; OOS untouched;
components UNCHANGED (all 4 PORTFOLIO; honest deploy core = me_long-solo, arc 1046). A REAL mechanism
finding banked (fbr's edge is USD-factor-idiosyncratic) but it does NOT separate the binding 2018 fold.

## Because
`fbr` (1013) is a PAIR-SPECIFIC stop-run reclaim. arc 2014 found its 2018 failure is a near-uniform −1R
wipeout across ALL 7 USD majors SIMULTANEOUSLY, and entry-time-unconditionable on SINGLE-PAIR axes (even
the "best" per-pair balanced context wipes out in 2018). That simultaneity is the tell of an UNTESTED
axis: are the 2018 fires **USD-FACTOR-WIDE** (the whole dollar trending → an INFORMATIONAL breakdown, the
reclaim fails) while good-year fires are **PAIR-IDIOSYNCRATIC** (a real stop-grab, USD calm → reclaim
holds)? arc 2014 tested per-pair trend; the cross-sectional USD-FACTOR trend (the common factor the corpus
keeps blaming for every strong-USD-year failure) was never tested directly on fbr. *Because:* a true
stop-grab reclaim is a pair-specific liquidity event; a USD-factor breakdown shows the same move across the
whole complex. Condition fbr on the move being idiosyncratic → keep good-year grabs, drop the 2018
falling-knives → a potential solo PASS (the first survivor).

## Method (observation only; no engine/P&L)
H4, 7 USD majors, fbr K=40/shadow≥1.25 (canonical), IS 2010-2020 (OOS preserved). Built a signed USD-factor
return (`mean over pairs of sgn_usd·logret`, sgn_usd = −1 XXXUSD / +1 USDXXX, USD-up positive); the L=40-bar
USD move z-scored (250-bar vol). For each fbr fire, the **aligned** USD-trend-z = signed so POSITIVE means
the broad USD moved in the SAME direction the pair's down-move implies (XXXUSD down→USD-up→+z; USDXXX
down→USD-down→−z) — i.e. HIGH = a USD-factor-wide breakdown. Honest +1R capture + 24-bar drift via canonical
`observe_long_capture` restricted to fires, joined to aligned-z + year. Driver
`discovery/_disco2_work/arc_2052_fbr_usd_factor_obs.py` (scratch). fbr reproduced exactly (n=235, cap
0.5872 ≡ 1013's 0.582).

## Results — a real texture that does NOT rescue 2018
**Falsifier 2 — the USD-factor conditioner IS real (cross-sectionally).** Edge by aligned USD-trend-z tercile:

| tercile | n | capture | drift ATR |
|---|---|---|---|
| LOW (idiosyncratic) | 80 | **0.6875** | +0.313 |
| MID | 78 | 0.6026 | +0.626 |
| HIGH (USD-factor-wide) | 77 | **0.4675** | −0.516 |

fbr's edge genuinely concentrates in pair-idiosyncratic stop-grabs (cap 0.69) and collapses to
coin-flip-negative (cap 0.47, drift −0.52) when the move is USD-factor-wide — a clean confirmation that
fbr is a pair-specific liquidity event, killed by the common dollar factor.

**Falsifier 3 — but the conditioner does NOT separate the binding 2018 fold (decisive).** Keeping
aligned_z ≤ median improves the pooled edge (cap 0.6695, drift +0.469) and most good folds, BUT per-year:

| year | raw drift | kept drift | |
|---|---|---|---|
| 2014 | +1.02 | +2.45 | strong-USD (already +) |
| 2015 | +0.80 | +1.47 | strong-USD (already +) |
| **2018** | **−1.28** | **−3.36** | **strong-USD — WORSE** |
| 2012 | +0.30 | **−0.60** | good fold FLIPS NEG |
| 2016 | −0.90 | −1.76 | worse |

**2018 gets WORSE under the filter** (the idiosyncratic 2018 fires are the *worst*), and 2012 flips
negative — no free lunch (arc 2014's recurring result). 2018's failure is NOT a USD-factor-alignment
property of individual fires.

**Falsifier 1 — per-year aligned-z does not rank fbr success.** 2014 aligned_z +0.27 (low) yet +1.02;
2018 +0.51 (mid-high) and −1.28; but 2016 +0.12 (low) and −0.90; 2020 +0.60 (high) and −1.53. The
per-fire tercile separation is real, but the per-YEAR mapping is noisy → exactly why the binding fold
isn't a clean function of the conditioner.

## Mechanism / why it KILLs for the deploy purpose
The USD-factor trend is a REAL conditioner on fbr's edge cross-sectionally (idiosyncratic grabs work,
USD-wide breakdowns don't — confirming fbr is pair-specific liquidity), but **2018's failure is a
time-localized REGIME where all reclaims fail, not a collection of USD-factor-aligned fires.** Even the
idiosyncratic 2018 fires wipe out (kept drift −3.36 < raw −1.28). So conditioning fbr on USD-factor
idiosyncrasy makes 2018 WORSE, not better, and flips a good fold (2012) negative. This SHARPENS arc 2014:
2018 is entry-time-unconditionable on the **cross-sectional USD-factor axis** too, not just per-pair — the
most natural conditioner given the corpus's central "USD factor" finding still cannot lift the fold.

## What this closes
The "condition fbr on the USD-factor (cross-sectional) regime to fix 2018" lane — the last categorically-
new conditioning axis for fbr's solo-PASS route (after per-pair trend 2014, level-significance proxies
1025/2030/1040, universe 2031/1049, finer-TF 2051). fbr-solo-PASS is now closed on the USD-factor axis;
its 2018 hole is time-localized-regime-intrinsic. fbr stays a PORTFOLIO component, USD majors, H4.
me_long-solo deploy object UNCHANGED (1046). Deployable-system count = 0.

## NEW lesson
fbr's reclaim edge is **USD-factor-conditional**: cross-sectionally it is strong on pair-idiosyncratic
stop-grabs (cap 0.69, drift +0.31, USD calm) and collapses to coin-flip-negative when the down-move is
USD-factor-wide (cap 0.47, drift −0.52) — a clean, novel confirmation that fbr is a pair-specific
liquidity event, arbitraged away when it coincides with a common-factor (dollar) breakdown. BUT a real
cross-sectional conditioner need NOT map onto a binding YEAR-fold: 2018's failure is a time-localized
regime (every reclaim fails, the idiosyncratic ones WORST), so filtering on USD-factor idiosyncrasy makes
2018 worse and flips a good fold negative (no free lunch). Distinguish a fire-level texture (real, useful
for understanding mechanism) from a fold-level separator (what a solo-PASS needs) — they are not the same,
and the corpus's binding folds keep being the latter, not the former.

## Tooling
No new BUILT tool — single-use diagnostic scan (matching 2047-2051); canonical `Panel.from_pairs` +
`observe_long_capture` + BUILT `_atr_shift1_mid` + inline USD-factor index + fbr fire logic. No
TOOL_REGISTRY append. No canonical change, no FLAG, no council, OOS untouched. Driver
`discovery/_disco2_work/arc_2052_fbr_usd_factor_obs.py`.

## Threads / handoff
fbr-solo-PASS is closed on every conditioning axis (per-pair trend, level-significance, universe,
finer-TF, cross-sectional USD-factor). The corpus's binding folds (2014/2018) are time-localized regimes,
not fire-level-conditionable properties. The surviving edges are information-free MECHANICAL forced flows
keyed to a LARGE surprise displacement at a NON-directional-flow time, living ON the USD factor at H4/D1;
their strong-USD failure folds are intrinsic and un-conditionable by any entry-time tell (per-pair OR
cross-sectional). 2018/2014 regime-orthogonal leg unfound (~22 routes); deployability lever = operator
path-A (arcs 2019/3021/1046) + the arc-2033/2045 vehicle wall. Useful banked texture: fbr's edge IS
USD-factor-idiosyncratic (cap 0.69 idiosyncratic vs 0.47 USD-wide) — a real mechanism datum for any future
fbr work.

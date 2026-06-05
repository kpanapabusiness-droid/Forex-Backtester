# arc 2020 — fbr toward solo-PASS: does M1 reclaim-QUALITY confirmation separate the 2018 failures?

**chat:** 2000s | **date:** 2026-06-05 | **disposition:** KILL (obs cheap-kill; closes council M1-confirm thread; fbr UNCHANGED, still PORTFOLIO)

## Step (a) — log read + synthesis (FRESH EYES, honest-era only)
Pulled main, read DISCOVERY_PROTOCOL, the full Tier-1 ledger (arcs 0–2019 across all 3 chats),
LESSONS, TOOL_REGISTRY. STOP absent. Highest arc-id in my range (2000–2999) = **2019** → resume at **2020**.

State of the corpus:
- **The 4-component PORTFOLIO book** (gap-fill long 1006 · month-end-long `me` 1011 · failed-breakdown-reclaim
  `fbr` 1013 · month-end-short `me_short` 1019) is mean-positive (t=2.66, P(mean<0)=0.004), ~3 independent
  bets (ENB 3.32/4, arc 2019), negative-tail-decorrelated — but NOT all-folds-positive (blocked by marginal
  2015/2016/2018).
- **Arc 2019 (my last, council-driven)** concluded the book's AFP failure is a **gate-resolution artifact**
  (thin legs trip the every-calendar-year gate); a 5th decorrelated REVERSION leg CANNOT make it AFP;
  **edge-hunting for the book is closed; the lever is the operator gate-governance call.** The one open
  edge-spec is arc-2017 **option B: a standalone THICK enough that folds RESOLVE** (none known).
- Mapped dead: shallow directional long/short (all mechanisms, H1/H4/D1/W1, majors+crosses), calendar flow,
  vol state, intraday/session structure (3016), relative-value/market-neutral (2010/2018), every short
  construction except `me_short`. Data backup is **FX-only** (28 pairs; no gold/metals/indices) → no new-instrument route.

## Step (b) — idea (observe, don't guess)
The programme's single highest-VALUE open target is its actual goal — a **deployable solo PASS** — and the
corpus crown jewel **`fbr` (arc 1013) is 9/10 IS, one fold (2018) from it** (OOS neg years 2022/25 share the
strong-USD signature). Arc 2019's `/llm-council-discovery` explicitly raised — and left UNRESOLVED — *"refine
fbr-2018 via M1 reclaim-confirm."* That lever (entry-quality at finer resolution) is **genuinely distinct**
from what closed fbr-2018 before: arc **2014** tested *daily regime-gating* (downtrend strength/persistence)
and arc **2017** was *diagnostic-only* (per-fold CI). Neither asked whether the 2018 failures are distinguishable
**AT ENTRY TIME** by the M1 micro-structure of the reclaim.

Hypothesis (falsifiable, pre-registered): *in the strong-USD year the "reclaim" H4 bars are weak/marginal at
M1 (price barely holds above the swept low), and an M1-reclaim-QUALITY filter — applied UNIFORMLY across all
years, no regime knowledge — improves the worst fold without destroying the 9 good folds.* Counter-prior
(arc 2014's "near-total 2018 wipeout, entry-time-unconditionable"): the reclaim is real at entry but fails
FORWARD (sold again days later in the USD trend) → finer ENTRY resolution cannot help.

## Step (c)/(d) — characterize + observe (no engine; gross take-the-loss capture)
Reproduced fbr exactly (`FailedBreakdownReclaimLongSignal` K=40, shadow≥1.25) on the 7 USD majors,
2010–2020 → **237 raw fires** (matches the corpus's standard fbr obs count, arc 1022). For every fire I
measured M1 reclaim-quality WITHIN the H4 reclaim bar (no-lookahead: all M1 ≤ signal-bar close; entry is
t+1 open): `pierce_count`, `last_pierce_frac` (late pierce = fresh/weak), `hold_min_margin_atr` (worst dip
above the swept level after the last M1 pierce), `time_above_after_frac`, `close_margin_atr` (decisiveness
at bar close), `frac_mid_above`. Joined honest per-trade capture + 24-bar forward drift.
Script: `discovery/_disco2_work/arc_2020_m1_reclaim_obs.py`.

### Result — both questions answered NO; decisive cheap-kill
**2018 is the failing fold, confirmed:** capture 0.474 vs other-years 0.596; **forward drift −1.282 vs +0.271**.

**Q1 — entry-time M1 tell in 2018? NO.** 2018 reclaim-quality vs other years is MIXED and SMALL:
`hold_min_margin` −0.095, `close_margin` −0.105 (marginally weaker) but **`frac_mid_above` +0.104 HIGHER**
(2018 price spent *more* of the bar above the swept level), `last_pierce_frac`/`time_above` ≈ identical.
The one big delta — `pierce_count` 39 vs 66 — is M1-density/vol confounded and its sign vs outcome is WRONG
(corr −0.091: more pierces → lower capture). **The 2018 reclaims do not look weaker/falser at M1; if anything
they look MORE decisive at the bar level — and still get sold.**

**Q2 — does M1-quality predict the outcome? NO.** Every metric |corr| ≤ 0.10 with capture, ≤ 0.074 with
drift (≈ zero). Terciles flat/non-monotone (top close-margin tercile cap 0.62 vs base ~0.58).

**No filter rescues 2018.** Sweeping each metric as a uniform keep-rule: requiring HIGH reclaim-decisiveness
makes 2018 capture *worse* (close_margin q.50 → 2018 cap 0.300; hold_min q.50 → 0.273; frac_above q.50 →
0.273) — the "high-quality" 2018 reclaims fail MORE. The only rule that "lifts" 2018 (pierce_count hi q.50 →
cap 0.833) does it by thinning to **n=6** (un-scalable thin-fold regime-luck, arc-1017/3010 tell) while
gutting good folds to 7/11. There is NO M1-confirm filter that raises 2018 without destroying the 9 good
folds — reproducing arc 2014's "any gate removing the neg pockets DESTROYS good folds," now at M1 resolution.

## Diagnosis + verdict
fbr's 2018 failure is a **FORWARD-reversal phenomenon** — the failed-breakdown reclaim is structurally real
*at entry* (2018 reclaims are if anything MORE decisive at M1), but does not HOLD over the following days in
the strong-USD trend (the failed breakdown becomes a real breakdown). Finer ENTRY resolution cannot see, and
cannot filter, a forward failure. This **independently confirms arc 2014 (daily regime gates) and arc 2017
(per-fold CI) via a THIRD, genuinely-different lever (M1 micro-structure of the reclaim itself)**, and
**definitively closes the council's "refine fbr-2018 via M1 reclaim-confirm" thread** (arc 2019).

§5f does not bite: this is an obs cheap-kill of a *refinement* (the M1-confirm lever shows no tell and no
within-sample predictive power → there is no reasoned best-version filter to run on the engine). **fbr itself
is UNCHANGED — still PORTFOLIO (arc 1013).** No engine/null/council spent.

**Implication for the route:** fbr-2018 is now triangulated as mechanism-intrinsic / entry-unconditionable
across three independent levers (regime-gating 2014, CI-decomposition 2017, M1-microstructure 2020). This
*strengthens* arc 2019's conclusion — the solo-PASS-via-fbr-2018-fix route is closed; the deployability lever
is the operator's gate-governance call, not more entry-side edge-hunting. 12th+ dead route to the 2018 leg,
and the last entry-resolution lever on the corpus's crown jewel.

## Threads
- The solo-PASS-via-fbr route is closed at the entry side (this arc, final entry-resolution lever).
- Open only: operator gate-governance call (path A: fold/gate resolution above noise floor; arcs 2016/2017/
  2019/1023) OR arc-2017 option B (a THICK fold-resolving standalone — none known on FX; data is FX-only).

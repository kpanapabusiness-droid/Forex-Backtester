# Arc 2001 — Weekend-Gap-Fill Long

> **Arc id:** 2001 · **Chat:** 2000–2999 (continuous) · **Date:** 2026-06-04
> **Final verdict:** **FAIL (full-IS sub-cost + HEAVY diagnosis council → KILL).** The FIRST long in the
> programme with a real, monotone, mechanistically-clean edge — but it fails on CAPTURABILITY, not on
> absence of edge: the tradeable honest entry (i+1, after the gap bar) sits inside the gap's adverse
> continuation, the event is thin (~177 big-gap trades/decade, ~13–17/fold), and the *stronger* raw
> signal (the UP-gap SHORT) is structurally inaccessible to the long-only apparatus. 9th EDGE<COST result.
> **Lever tested:** a NON-price-direction price-structure axis no prior arc touched — weekend/weekly-open
> GAP behavior, attacking EDGE<COST with a discrete, sometimes-large move.

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). Canonical apparatus called; the
gap signal is an experiment tool (`discovery/tools/gap_signals.py`, registered). **OOS (2021+) deliberately
NOT touched** — see §(g).

## (a) Log read — FRESH EYES

Pulled main (`077671c`, after self-debloat from arc 2000). Honest-era corpus = 8 arcs (0, 1000–1004, 3000,
2000), ALL FAIL. Accumulated finding (across 3 chats): long-only spot-FX **price-direction** at H4/D1 does
not clear FundedNext costs — invariant across entry construction, timeframe (H4/D1), the full 28-pair
universe (majors+crosses+coupled), both mechanism families (continuation+reversion), and payoff/exit
structure. Constraint = **EDGE < COST** (crosses have +0.10R gross drift but wider spreads eat it; majors
have ~0 gross edge). Claimed lanes: calendar-flow (1000s), portfolio/selection (3000s). LESSONS.md empty.
No STOP.

**Open lane for the 2000s range:** something that *raises per-trade gross edge or cuts cost/frequency*
enough to clear EDGE<COST, on an axis that is neither calendar nor portfolio. The price corpus is
price-only (no rates/fundamentals), so the accessible non-direction edges are price-structure or
time-structure. The untouched price-structure axis: **weekend/weekly-open GAPs** — discrete, sometimes
large moves with a documented partial-fill tendency.

## (b) Idea + observation (observe first, drift metric)

**Observation** (H4 majors, IS 2010–2020): at each weekly open (index time-gap > 20h) compute
gap = (open_mid − prior_close_mid)/ATR, then forward drift (mean k-bar return in ATR) by gap bin:

| gap bin | n | fwd6 (mean ATR / frac+) | fwd30 |
|---|---|---|---|
| **DOWN >1·ATR** | 177 | **+0.454 / 0.59** | +0.531 / 0.58 |
| down .5–1 | 415 | +0.185 / 0.57 | +0.362 / 0.53 |
| flat \|.1\| | 1034 | +0.019 / 0.51 | −0.080 |
| up .5–1 | 358 | −0.197 / 0.43 | −0.411 |
| **UP >1·ATR** | 142 | **−0.574 / 0.36** | −0.235 |

A clean, **monotone & symmetric gap-fill**: the bigger the gap, the stronger the reversion toward the
prior close (down-gaps drift up, up-gaps drift down). This is the **only conditioning variable in the whole
programme to push the directional hit-rate clearly above 0.50 with a sizable drift.** The long-only
tradeable side = buy big DOWN gaps. Did NOT cheap-kill at observation — it earned characterization.

## (c) Characterize — the MAE problem

Canonical pool, threshold {0.5,0.75,1.0} × SL {1.0,1.5,2.0}, hold 30. Key finding: the down-gap's
**adverse continuation is large** — median MAE −1.05 to −1.30R, P(MAE≤−0.5R)=0.77–0.89. The honest entry is
i+1 (signal fires gap-bar close, fill next bar open), so it MISSES the gap-bar's own initial fill and enters
INTO the continuation. Honest +1R-before-SL capture only 0.45–0.47 (< 0.50). meanFinalR is positive GROSS
but small (+0.07 to +0.18R) — a *real* edge (first non-cross-trend positive-gross long), but small. Tight
SL (1.0) → 77–80% stop-out (the MAE blows through it).

## (d) Triage → full IS WFO (the lucky-fold correction)

3-fold triage (2013/16/19) looked near-break-even/positive (best: thr0.5 SL2.0 tp_2r mean **+1.29%**, 1/3
neg) → NOT deeply negative → proceeded (no cheap-kill). But the **full 10-fold IS WFO** corrected it:

| config | n_neg | worst | mean IS |
|---|---|---|---|
| thr0.5 SL2.0 tp_2r | 7/10 | −8.17% | −1.67% |
| **thr1.0 SL1.0 partial (best)** | 6/10 | −4.57% | **−0.97%** |
| thr0.5 SL2.0 partial | 7/10 | −7.05% | −2.37% |
| thr0.5 SL2.0 tp_3r | 7/10 | −12.65% | −2.53% |

The 3-fold triage folds were **favorable-fold luck** (2013/16/19 = low-vol orderly-reversion years). The
full IS is sub-cost (mean −1 to −2.5%, 6–7/10 folds negative) for *every* threshold/SL/exit. **Lesson: a
3-fold triage can be misleadingly positive; the full IS WFO is the honest judge — run it before getting
excited.** NOT all-folds-positive.

## (e) Diagnose + HEAVY council

**Diagnosis:** real gap-fill edge, but (i) the i+1 entry sits in the adverse continuation (large MAE → stop
before fill), and (ii) the surviving gross edge is small and cost-eaten. Fork: (A) one mechanistic
refinement — delayed/confirmation entry past the continuation; or (B) accept EDGE<COST and FAIL.

**Council (`/llm-council-discovery`, HEAVY/evaluative, 5 lenses + 3 anonymized reviewers + chairman).**
Full transcript: [`../results/arc_2001_weekend_gap_fill_long/council_transcript.md`](../results/arc_2001_weekend_gap_fill_long/council_transcript.md).
**Verdict: KILL — FAIL on IS now; do NOT touch OOS; do NOT build the refinement.** Reasoning (CC commits,
no override):
1. **Option A is contamination as scoped.** A reversal-confirmation entry invented + tuned on IS can only
   be validated by spending the pristine OOS — protocol §4 violation. When IS fails cleanly, FAIL on IS and
   preserve OOS.
2. **The sample can't support the judge even if the edge is real.** ~13–17 IS trades/fold → ~6–8/OOS-year
   on a filtered subset; at that count all-folds-positive is a coin-flip *regardless of true edge*.
3. **The drift table is hindsight-framed.** The exciting 0.59 frac-positive / monotonicity is measured from
   the gap-bar OPEN — a price the engine cannot trade. The *tradeable* i+1 edge is the small one (+0.07..
   +0.18R) that 8 prior arcs proved costs eat. "The mechanism survives; the trade does not."

Council agreement was deep (Soundness lens + all 3 reviewers); the Mechanism/Refinement lenses were right
that confirmation-entry is *where the edge lives* but answered the wrong question (a correct mechanism you
cannot honestly validate at this n is not a tradeable edge). Alternative-framing's regime-filter reframe was
unanimously judged the weakest (a regime filter is *more* parameter-hungry, thinning a sub-thin sample).

## (g) Validation — IS only; OOS PRESERVED PRISTINE

The full IS WFO is NOT all-folds-positive (6–7/10 folds negative, every config). There is no finished
system that clears IS → per protocol §4 + the council, **OOS (2021+) was deliberately NOT measured.** Two
reasons beyond "IS failed": (i) measuring an IS-failed system on OOS adds nothing; (ii) had I tuned a
refinement to lift IS, validating it on OOS would be contamination, and the OOS folds (~6–8 trades/year)
are too thin to be a meaningful all-folds-positive judge anyway. Preserving OOS is the correct discipline.

## Final verdict — FAIL (real edge, uncapturable)

The **weekend-gap-fill long is not deployable** — but it fails differently from the prior eight: it has a
real, monotone, mechanistically-clean edge that fails on **CAPTURABILITY**, not absence of edge. The
tradeable i+1 entry sits in the gap's adverse continuation; the event is thin (~177/decade, ~13–17/fold);
the stronger raw signal (the UP-gap short) is structurally inaccessible to the long-only apparatus; and the
small surviving gross edge is cost-eaten. 9th EDGE<COST result.

## Convergence with arc 1006 (chat 1000s, landed on main mid-arc)

Chat 1000s **independently** arrived at weekend gap-down-fill (via a GENERATIVE council) and tested it on
**JPY CROSSES** (gap<−0.5·ATR, 24-bar time exit): the run's **MOST promising result** — the FIRST
mean-**POSITIVE** IS edge (+0.69%), beats random — but fold-fragile → FAIL (worst IS −6.79%, worst OOS
−4.13%, n=396). **Two chats, two universes, one convergence: weekend gap-fill is the realest edge the
programme has found.** KEY CROSS-FINDING: gap-fill is mean-**POSITIVE on JPY crosses** (1006) but
mean-**NEGATIVE on majors** (this arc) → **JPY crosses are the better gap-fill universe** (bigger weekend
gaps from the JPY session-break + carry unwind, stronger reversion); H4 majors are too efficient. 1006's
fleet steer: a decorrelated COMBINATION of ≥2 net-positive gap-type edges could cut fold-variance while
preserving positive mean → plausibly reach all-folds-positive. **This arc's distinct adds:** (1) majors are
the WEAKER gap universe — do not pursue gap-fill on majors; (2) **FLAG-1** — the long-only apparatus blocks
the STRONGER short side, a constraint now independently relevant to the programme's most promising thread;
(3) the OOS-pristine discipline + the n-too-thin-for-the-judge arithmetic (HEAVY council), which bounds how
far any single thin gap cohort can be pushed.

## Lessons (candidate for LESSONS.md)

1. **Weekend gaps are the first real, monotone, mechanistically-clean edge in the programme** — gap size
   scales reversion symmetrically (down-gaps fill up, up-gaps fill down), the cleanest "because"
   (weekend-illiquidity repricing toward the prior close) found so far. But it FAILS on capturability:
   thin frequency + the honest i+1 entry landing in the adverse continuation (median MAE −1.1 to −1.3R) +
   long-only access to only the weaker side + small cost-eaten gross edge.
2. **A 3-fold triage can be favorable-fold luck.** The gap-fill's 3-fold triage (2013/16/19) was
   near-break-even-positive, but the full 10-fold IS WFO was 6–7/10 negative. The triage folds were
   low-vol orderly-reversion years. **Run the full IS WFO before believing a positive triage** (the same
   2013-regime-luck that already burned arcs 1000/1003).
3. **OOS-pristine discipline, applied.** When IS fails and the only "rescue" is a refinement that must be
   tuned on IS then validated on OOS, the correct move is FAIL-on-IS + preserve-OOS, not burn the holdout.
   Confirmed by the HEAVY council (contamination + sample-too-thin-for-the-judge).
4. **9th EDGE<COST result** — but the first whose binding constraint is CAPTURABILITY/frequency/access, not
   "no edge." This sharpens the programme's negative result: a *real* H4 edge exists; it is the cost +
   long-only + event-rarity stack that defeats it.

## Threads / FLAGS

- **Closed:** weekend-gap-fill LONG on H4 majors (real but uncapturable: thin + i+1-into-continuation +
  cost). The delayed/confirmation-entry refinement is NOT worth pursuing on this sample (council: curve-fit
  trap at n~13/fold; would contaminate OOS).
- **FLAG-1 (operator / human-gated CODE — do NOT self-merge):** the UP-gap **SHORT** side is the *stronger*
  raw signal (UP >1·ATR: −0.57 ATR drift, 0.64 directional accuracy vs the down-side's 0.59) but the
  apparatus is **long-only** (`build_arc_pool` raises NotImplementedError for short; A1 emits Direction.LONG
  only). This is the first concrete evidence that the **long-only constraint may be blocking the programme's
  best signal.** Adding short support is a canonical-core code change → flagged for the operator, NOT merged.
- **FLAG-2 (future-arc seed):** the upstream question — *is H4 + FundedNext cost structure GENERICALLY
  hostile to fill/mean-reversion entries?* If yes, a single cheap diagnostic could retire a whole class of
  future arcs. A separate investigation, not a rescue of this one.
- **Open (carried):** the 2000s lane still needs an EDGE>COST construction. Given the gap finding, the
  highest-value untested directions are (a) a LOWER-timeframe gap test where the event is more populous (M1
  cache exists; H1 aggregable) — more events could make the same edge statistically tradeable, and (b)
  whatever the FLAG-2 diagnostic implies.

## Reproduction

Experiment tool committed + registered: `discovery/tools/gap_signals.py` (`WeekendGapFillLongSignal`).
Drivers in scratch (`_disco2000_work/arc2001_gap_observe.py`, `arc2001_gap_characterize.py`,
`arc2001_gap_triage.py`, `arc2001_gap_wfo_is.py`); `PYTHONPATH=. py _disco2000_work/<script>.py`. Data:
`histdata_root=C:\Users\panap\histdata_backup`, `cache_root=...\data\cache`, tf H4 5ers_eet, 8 majors. Pool:
hold 30, IS 2010–2020. Engine: `A1Architecture`+`ArcFoldRunner`, FundedNext costs at
`build_fold_stats_from_run`, `trail_enabled=False`. IS folds `build_v3_folds` (is_days≥365); OOS NOT run.

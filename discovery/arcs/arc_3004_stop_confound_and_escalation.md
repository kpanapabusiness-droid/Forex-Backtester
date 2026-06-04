# Arc 3004 — The Stop Confound (council-driven) → Clean Escalation

> **Arc id:** 3004 · **Chat:** 3000–3999 (continuous) · **Date:** 2026-06-05
> **Final verdict:** **FAIL — and the directional space is now CLEANLY closed.** The one untested confound
> across all 13 prior arcs (every arc used a 2·ATR SL-first stop) was lifted: removing the stop improves
> mean fold ROI by ~4–5pp (the take-the-loss tax is real) but the strongest directional signal is **still not
> all-folds-positive** (~breakeven, regime-dependent). → **ESCALATION** (see below): within this apparatus,
> the long-only single-instrument directional space is exhausted; the binding constraint is structural.
> **Idea source:** LIGHT generative council (`/llm-council-discovery`) at a genuine stuck-point.

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). Engine/measurement **called, never
re-rolled**; the time-exit predicate is a BUILT experiment tool **reused** from arc 1005 (geometry only).

---

## (a) Log read + synthesis — FRESH EYES

Pulled `origin/main`. Honest-era corpus = 14 arcs across 3 chats (0; 1000–1005; 2000; 3000–3003), all FAIL.
The 3000s side (3000–3003) had systematically closed directional entries (both metrics), the instrument
universe, the volume data axis, and regime detection; the 1000s/2000s chats closed timeframe, exits, calendar,
convexity. The price/volume/structure/regime directional space looked exhausted — a genuine stuck-point /
idea-fork. Per protocol §5b/§7, convened the LIGHT generative council before another lone guess.

## (b) Idea formation — the GENERATIVE COUNCIL (`/llm-council-discovery`)

Full transcript: [`../results/arc_3004_stop_confound/council_transcript.md`](../results/arc_3004_stop_confound/council_transcript.md).
5 isolated lenses → 3 anonymous peer reviews → chairman. The council did its job — it surfaced a confound I
(and all 13 arcs) had missed.

- **Mechanism / Steelman-Devil (BEAR):** "directional FX has no edge" is mechanistically true — the apparatus
  forbids the inputs that drive FX *direction* (rates, central-bank flow, positioning); the price residual
  after informed flow clears is a coin-flip; FX's durable edges are *relative* (carry, value, cointegration),
  needing a second leg the apparatus forbids. → escalate.
- **Refinement / Soundness (the surfaced confound):** **every one of the 13 arcs used a 2·ATR SL-first stop.**
  The take-the-loss tax demonstrably converts a positive *gross* drift into negative *honest* expectancy
  (+0.019R raw → −0.055R, arc 3003). No arc ever tested a **stop-free / time-exit-only** construction. So
  "directional space is closed" is *confounded* with "a tight symmetric stop truncates a slow drift before it
  accrues." Soundness named the meta-trap: **escalation is the *more* seductive lie because it ends the
  search** — demand a cheap falsification first.
- **Peer review (decisive sharpening):** the **cost** is a wall *independent* of the stop — a time-exit
  holding N bars still pays the full per-round-turn spread, so a stop-removed test could flip a fold positive
  yet stay sub-cost; the stop and cost confounds must be lifted *together* and read as two numbers.
- **Chairman:** **investigate-X-first, do NOT escalate yet.** Re-score the strongest +gross-drift entry under
  a wide/disabled SL + time-exit (a non-partial-runner policy, so the CI-locked take-the-loss invariant is
  untouched), costs ON. Pre-committed rule: stop-removed **and** all-folds-positive → the stop was the wall
  (directional space reopens); stop-removed but still sub-cost → EDGE<COST is structural → escalate with a
  *clean* proof.

**CC commitment (generative/light weight, but the convergence was strong and the test cheap+dispositive):**
committed to the chairman's measure-then-decide. The synthesized idea = the stop-confound test.

## (c)+(g) The clean stop-confound test (the experiment the council demanded)

Entry: post-up-spike continuation on the 12 trending crosses (arc 3001's pool, the strongest +gross-drift cell,
gross mean final_r +0.088R). A/B, full IS WFO, FundedNext costs ON, scored only by `MultiPairBacktester`.
"Stop removed" = pure **time-exit** via the BUILT `make_time_exit_predicate` (arc 1005) with a wide 6·ATR SL
used only for position-sizing (drift/cost ratio is position-size-invariant, so the wide SL does not bias the
net sign). Mechanism note: `A1Config.time_exit_bars` is **defined but UNWIRED** (FLAG, independently found by
arc 1005) — so a time exit must be supplied as an `ExitPredicate`, which A1 *does* wire.

| config | per-fold (2011…2020) | worst | mean | median | neg | AFP |
|---|---|---|---|---|---|---|
| BASELINE 2·ATR partial/runner (all-13-arc geometry) | −8/−4/+17/−15/+3/−28/+21/−14/−21/+5 | −28.07% | −4.52% | −6.03% | 6/10 | False |
| stop-removed, time-exit N=6 | −8/−5/+9/−6/−1/−11/+10/−5/−4/−1 | −10.66% | −2.13% | −4.06% | 8/10 | False |
| stop-removed, time-exit N=12 | −7/−2/+5/−9/−1/−11/+7/−6/−12/−0 | −11.87% | −3.46% | −3.81% | 8/10 | False |
| stop-removed, time-exit N=24 | −6/+5/+7/−10/−4/−15/+15/−6/−13/−0 | −14.95% | −2.77% | −4.93% | 7/10 | False |
| stop-removed, time-exit N=48 | −6/+9/+9/−5/+7/−20/+13/−5/−1/+5 | −20.43% | **+0.62%** | +2.21% | 5/10 | False |

**Reading the two numbers (per the pre-committed rule):**
1. **STOP isolated:** removing the 2·ATR stop *does* help — mean fold ROI rises from −4.52% to ~0% (N=48
   even +0.62%), and the deepest drawdowns shrink. The take-the-loss tax was a real ~4–5pp drag. The
   Refinement/Soundness lenses were right that the stop mattered.
2. **COST/REGIME isolated:** but it is **NOT the binding wall.** Even with the stop gone and the full drift
   captured, **no horizon is all-folds-positive** — the best (N=48) still has **5/10 negative folds, worst
   −20.43%** (the chop years 2014/2016/2018 stay deeply negative regardless of stop). The residual edge after
   FundedNext cost is ~breakeven and **regime-dependent** — exactly the skeptics' prediction.

This is the chairman's "stop-removed but still sub-cost" branch, now established with a **clean, unconfounded**
proof: the directional space is closed not because a stop was killing a winner, but because the gross drift
barely matches cost and remains regime-dependent.

## ESCALATION — the honest deliverable (operator decision required)

See [`../ESCALATION_apparatus_capability.md`](../ESCALATION_apparatus_capability.md) for the standalone,
operator-facing note. In brief: **14 arcs across 3 chats, plus this council-driven test that lifts the one
remaining confound, are strong evidence that the fixed apparatus — LONG-ONLY, SINGLE-INSTRUMENT, per-trade
SL/TP, price+tick-volume only, FundedNext costs — cannot express a deployable (all-folds-positive) edge on FX
H4/D1.** The mechanistic reason: the apparatus can only bet on *direction*, which is the most-arbitraged,
~0.49-coin-flip property of liquid FX; the durable FX edges (carry, relative-value, cointegration) are
*relative/market-neutral* and require a structural unlock the apparatus forbids. **Minimal unlocks for the
operator to consider (ranked): (1) a second simultaneous leg** (enables market-neutral cross-spread /
cost-mutualising relative-value — the one lever that doesn't require beating 0.50 trade-by-trade); **(2)
shorting** (doubles the expressible signal, enables mean-reversion both sides); **(3) a different execution/cost
regime** (raw-spread ECN — FundedNext's per-round-turn cost is the explicit hurdle, fixed by protocol §1 and
thus an escalation item, not a CC lever). This is a RECOMMENDATION for the operator's return; the discovery
loop is NOT halted (no STOP) — it continues to probe the few remaining thin in-apparatus ideas.

## Final verdict — FAIL (directional space cleanly closed; escalation raised)

The post-up-spike directional signal is not deployable (not all-folds-positive) even with the stop confound
lifted. More importantly, the arc converts the corpus's accumulating negative result into a *clean* structural
conclusion: the binding constraint is the apparatus, not the entry/exit/regime/instrument tuning.

## Lessons (candidate for LESSONS.md compression)

1. **The 2·ATR SL-first stop was a real ~4–5pp drag but NOT the binding wall.** Removing it (pure time-exit,
   wide sizing-only SL) lifts mean fold ROI from −4.52% to ~0% on the strongest +gross-drift signal, but the
   result is still not all-folds-positive (best N=48: 5/10 neg, worst −20%, mean +0.62%). The take-the-loss
   tax is real; the deeper wall is that the gross drift ≈ cost and is regime-dependent. **This retires the
   "the stop was the wall" hypothesis with a clean, unconfounded test** — the directional space is closed for
   the right reason.
2. **The directional space is closed structurally, not by tuning.** Across 14 arcs + this confound-lifting
   test, no entry / exit / instrument / timeframe / regime / stop-geometry variation makes a long-only
   single-instrument FX bet all-folds-positive. The apparatus expresses only *direction*, which liquid FX
   prices to a coin-flip; durable FX edges are relative/market-neutral. → escalation (second leg / shorts /
   cost regime).
3. **TOOLING DISCIPLINE WORKED (and a near-miss).** The time-exit predicate I needed already existed
   (`make_time_exit_predicate`, arc 1005); I started to rebuild it, the Write correctly failed (file present),
   and I **called the BUILT tool** instead. *Lesson: check `TOOL_REGISTRY.md` BUILT FIRST, before writing —
   I checked too late.* The cross-chat tool registry compounds: arcs 1005/2000 built time-exit + trend-entry
   tools that I reused this arc.
4. **The council earns its keep at a genuine stuck-point.** It surfaced a confound (the universal 2·ATR stop)
   that I had not seen after 4 arcs, and its measure-then-decide discipline prevented escalating on confounded
   data (the Arc-10 lesson — internal consistency ≠ correctness — applied to a research conclusion).

## Threads / what didn't help

- **Closed (clean):** the stop-geometry confound — removing the per-trade stop does not rescue a directional
  long; the residual is ~breakeven + regime-dependent.
- **Escalated:** apparatus capability → operator (second leg / shorts / cost regime). See the escalation note.
- **Remaining thin in-apparatus threads (low conviction, for completeness before fully concluding):** the
  Mechanism lens's cross-rate triangulation lag (synthetic mid from the two USD legs vs the quoted cross) —
  likely an M1 microstructure effect below the cross spread, but the one genuinely-different *non-directional*
  in-apparatus idea named; a candidate for a future arc as a cheap-kill. Portfolio/selection remains
  blocked (no net-positive component, and the honest cost-mutualising version needs the multi-leg unlock).

## Flags (code NOT merged — human-gated, per protocol §9)

- **`A1Config.time_exit_bars` is defined (line 78) but UNWIRED** — never consumed by `A1Architecture`
  (`_build_a1_strategy` does not pass it to the Order/engine). A time exit must instead be supplied as an
  `ExitPredicate`. Independently flagged by arc 1005; **re-flagged here** as it caused a silent no-op in an
  early version of this arc's test (all `time_exit_bars` values gave identical results). Fix is human-gated:
  either wire `time_exit_bars` into the A1 strategy/Order, or remove the dead parameter to prevent silent
  no-ops. Not merged.

## Reproduction

- **Data:** `histdata_root = C:\Users\panap\histdata_backup`, `cache_root = data/cache`,
  `boundary_convention = "5ers_eet"`, TF H4. **Pairs:** 12 trending crosses (EURJPY GBPJPY AUDJPY NZDJPY
  CADJPY CHFJPY EURAUD EURNZD EURCAD GBPAUD GBPNZD GBPCAD).
- **Entry:** `_disco3_work/arc3001_signal.PostUpSpikeContinuationLong(spike_atr=2.0, refractory=6)`.
- **Exit (test):** `discovery/tools/time_exit_predicate.make_time_exit_predicate({p: panel.pair_dfs[p]}, n_bars=N)`
  injected via `dataclasses.replace` onto each pair's `PerPairSignalState.exit_predicate`; A1Config
  `exit_policy="sl_only", sl_atr_mult=6.0` (wide, sizing only). Baseline:
  `exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0`.
- **Driver (scratch):** `_disco3_work/arc3004_clean.py`. Full IS folds `build_v3_folds`; judge
  `judge_all_folds_positive`. Council via `/llm-council-discovery` (5 lenses + 3 peer reviews + chairman).

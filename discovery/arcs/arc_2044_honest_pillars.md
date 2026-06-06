# arc 2044 — Honest-§5f re-derivation of the path-A robustness pillars (discharges arc-1043 FLAG F2)

**Chat:** 2000s · **Range:** 2000–2999 · **Disposition:** **DIAGNOSTIC → KILL** (no new
component; the path-A deploy case's THREE remaining robustness pillars — diversification/ENB
[2019], temporal-stability [2021], cost-robustness-κ [3022] — are recomputed under honest §5f
exits for the operator's call) · **Council:** none (a measurement resolving a measurement, as
arcs 1023/2016/2017/2019/3021/3022/1032/2043)

---

## (a) Log read / synthesis

Pulled main; read DISCOVERY_LOG (both tiers, arcs 0–2043 + 1043), LESSONS, TOOL_REGISTRY; STOP
absent. State (honest-era only): single-condition shallow directional prediction is **closed
ground** (H1/H4/D1/W1, long+short); the edge-hunt is **structurally closed** (arc 3021 path-B
proof — a shared USD factor floors P(AFP) at any N; MENU M1/O1/L1/Q1/G1/S1 exhausted; ~18 routes to
a +2015/+2018 leg dead, every short construction dead). **4 PORTFOLIO components** — gap (1006),
me_long (1011), fbr (1013, load-bearing), me_short (1019). The **4-way book is mean-positive but
never all-folds-positive** (combination-invariant, blocked by 2015 & 2018; arcs 1020/2008/3009/1015).
**§11 independent verification COMPLETE** end-to-end (signal 2034/2035/1037 + gross outcome
2036/2037/2038/1038/1039 + cost 2039). Sole deployability lever = the **operator path-A
gate-governance call** (arc 1032: coarsening the calendar gate yields no meaningful-resolution AFP →
path-A = adopt a mean/CI gate).

**The live thread — exit-honesty (2040→2041→2042/1042→2043/1043).** §5f nested-WFO exit selection
showed the committed component headlines are full-sample-best exit picks: **fbr −40% (stays +),
me_short & gap SIGN-FLIP mean-negative, only me_long exit-robust**. Arc 2043/1043 discharged FLAG F3:
the book's **t=2.66 mean-significance pillar does NOT survive honest exits** (honest t≤0.96, all CIs
span zero). Arc 1043 then raised **F2, explicitly owed:** *"arcs 2021 (temporal), 3022 (cost-κ),
2019 (ENB/diversification, P(mean<0)=0.004) were ALL computed on the COMMITTED-exit series → suspect
by the same exit-optimism mechanism; honest-exit re-run owed before path-A leans on them."*

## (b) Idea (the *because*)

Path-A would deploy the book on its MEAN, supported by FOUR robustness pillars: mean-significant
(2023/2019 — now demolished by 2043/1043), **diversified ~3-independent-bet sound portfolio (2019:
ENB 3.32, negative-tail, "AFP-failure is PURELY thinness, not a diversification/tail/mean problem")**,
**temporally stable (2021: LATE-half still +0.404%)**, and **cost-robust (3022: break-even κ=3.32,
70% of gross survives)**. All three remaining pillars were computed on the COMMITTED-exit series, in
which all four legs are mean-positive. The exit-honesty thread flipped two of four legs (gap,
me_short) mean-NEGATIVE. **A "sound diversified positive 3-bet portfolio," a "temporally stable
edge," and a "3.3× cost cushion" all rest on four positive legs — none was ever recomputed once two
go negative.** This arc does that recompute (F2), with the committed series as a fidelity anchor
(reproduce ENB 3.32, LATE +0.40%, break-even κ=3.32).

## (c)/(f) Method

- Scoring 100% canonical (`ArcFoldRunner` → `A1Architecture` → `MultiPairBacktester`, FundedNext,
  A1 default `risk_pct=0.005` = the linear deployable regime; ratios risk-INVARIANT, arc 1024).
  Selection via BUILT `nested_exit_selection` (2040); book combination via BUILT `combine_fold_roi`
  (2006); cost re-net via BUILT `scaled_fundednext` (3022) through the **canonical chokepoint**
  `build_fold_stats_from_run` (no engine re-run — the gross `RunResult` is captured per grid cell and
  re-netted at each κ). ENB/temporal arithmetic is inline numpy over canonical per-fold numbers (no
  new BUILT tool, as arc 2043's significance battery).
- Honest grids reproduce 1042/2043 exactly: gap = pure-time × horizon{12,18,24,36,48} × SL{1.5,2,2.5};
  me_long/fbr/me_short = 6 registry exits × 3 SL (me_long 2-bar horizon fixed as mechanism). 3 nested
  metrics; **the honest no-lookahead series = the 8 EVALUABLE folds (2013–2020)** (warmup folds 2011/12
  use the frozen full-sample pick = not honest). Primary metric = `afp_then_mean` (gate-aligned;
  `mean_roi` grabs the fbr trailing_swing §5f trap, arc 2040).
- **OOS NEVER touched** (book fails IS AFP; §5g operator firewall). Driver
  `_disco_work/arc2044_honest_pillars.py`, output `_disco_work/arc2044_out.txt`.

## Results

### Fidelity anchors — ALL reproduce (Arc-10 independent-reproduction discipline)
- Committed component means: gap **+0.685%**, me_long **+0.232%**, fbr **+2.084%** (trail-OFF book
  config ≡ arc-2040/2043 anchor), me_short **+0.683%** — exact.
- **ENB committed (n=10): 3.31/4, top-eigenvalue share 38.9%, CI[2.00,3.27]** — reproduces arc 2019's
  **ENB 3.32, top-eig 38.9%, CI[2.02,3.33]** essentially exactly.
- **Temporal committed (n=10): EARLY +0.775%, LATE +0.412% (P≤0=0.056), decay +0.361% CI[−0.614,+1.348]
  P(decay>0)=0.76** — reproduces arc 2021's LATE +0.404%, decay +0.365% CI[−0.636,+1.326] P=0.76.
- **Cost committed (n=10): break-even κ=3.39** (arc 3022's 3.32); κ=0 worst-fold negative, not
  all-folds-positive cost-free (arc 3022's "AFP-failure is cost-independent"). *(My κ=0 shows 1/10 neg
  vs 3022's 2/10 — a single borderline fold flipped by the trail-OFF fbr +2.084 vs 3022's double-trail
  +1.854; immaterial — the break-even and the "not-AFP-even-cost-free" conclusion hold.)*
  ⇒ the recompute machinery is anchored; the honest numbers below rest on a reproduced baseline.

### F2 — every remaining pillar is materially WEAKER under honest §5f exits

**[A0] Two of four legs are mean-NEGATIVE under honest exits** (8 eval folds): gap **−0.854%**,
me_long **+0.203%**, fbr **+1.001%**, me_short **−0.405%** (confirms 2041/2042/1042/2043).

**Pillar 1 — DIVERSIFICATION / ENB (arc 2019).** The decorrelation STRUCTURE largely survives but
the "sound POSITIVE portfolio" claim does not:
- Honest ENB ≈ **2.7–2.9/4** (mean_roi 2.68 / afp 2.89 / worst 2.90; top-eig 43–49%), vs committed
  3.31 (n=10) / 3.03 (n=8 eval) — still ~3 statistically decorrelated bets (me_long·fbr −0.57,
  me_long·me_short −0.55), tail co-movement NEGATIVE (worst-5 −0.179, best-5 −0.304), max
  co-negativity 3/4. So the legs ARE decorrelated — but decorrelation is **necessary-not-sufficient.**
- **[A4] DECISIVE — the 4-leg book is mean-DRAGGED BELOW its 2 honest legs.** Under honest exits the
  RP 4-leg book (mean +0.08…+0.12%, pos 4–5/8) is **strictly worse than the 2-leg me_long+fbr book**
  (mean **+0.36…+0.41%**, pos 5–7/8, equal-or-better worst-fold in 2/3 metrics) across all 3 metrics.
  ⇒ arc 2019's celebrated "diversification" is, under honest exits, **diluting 2 positive legs with 2
  now-negative ones** — diversifying INTO negative-mean legs hurts the book. The "AFP-failure is purely
  thinness, not a diversification/mean problem" framing inverts: the honest mean problem (2 neg legs)
  is real, and the honest book's best expression is the 2-leg me_long+fbr (still NOT all-folds-positive
  — afp 6/8 worst −0.21% — so no new survivor).

**Pillar 2 — TEMPORAL STABILITY (arc 2021).** The "edge is temporally robust, LATE-half stays
+0.404%" claim does NOT survive:
- Honest RP book LATE-half (2017–2020) = **mean_roi −0.109% / afp −0.093% / worst +0.055%**
  (P(≤0) 0.41–0.65) vs committed +0.412%. The honest book's positive mean is **EARLY-loaded**
  (2013–2016 EARLY +0.18…+0.28%, P(≤0) 0.13–0.19). The decay CI still spans zero (low power, n=4/half)
  — so this is "honest LATE ≈ 0/slightly-negative," not a proven decay, but it removes arc 2021's
  positive-LATE reassurance. (Window note: honest eval is 2013–2020 so EARLY=13–16/LATE=17–20, vs
  2021's 11–15/16–20 — the honest series structurally loses the strong 2011/12 warmup folds.)

**Pillar 3 — COST ROBUSTNESS κ (arc 3022).** The cushion roughly QUARTERS:
- Honest break-even **κ ≈ 1.31–1.49** (afp 1.31 / mean_roi 1.38 / worst 1.49) vs committed 3.39.
  The honest book survives only ~1.3–1.5× FundedNext cost (vs 3.3×); at κ=1 (actual FN) it is
  +0.08…+0.12%, so a ~40–50% cost increase wipes it. **AFP-failure remains cost-independent** —
  3/8 neg even cost-free (κ=0), confirming arc 3022's structural point (the per-year gate, not cost,
  is the wall) on the honest series too.

## (h)/(g) Verdict — DIAGNOSTIC → KILL (no new component; components UNCHANGED)

**Findings.**
1. **F2 discharged.** All three remaining path-A robustness pillars are materially weaker under honest
   §5f exit accounting: **diversification** survives as structure (ENB ~2.8) but the 4-leg book is
   mean-dragged below its 2 honest legs (gap+me_short are now drags, not diversifiers);
   **temporal stability** is gone (honest LATE-half ≈ 0/negative vs committed +0.41%);
   **cost cushion** quarters (break-even κ ≈ 1.3–1.5 vs 3.3). With arc 2043/1043 (significance gone),
   **the entire committed-exit-based path-A robustness profile is an exit-optimism artifact** —
   because two of the four legs flip mean-negative once exits are honest.
2. **Unified honest picture of the book:** the honest book is best expressed as the **2-leg
   me_long+fbr** (the only exit-robust legs), weakly mean-positive (~+0.4% 2-leg / ~+0.1% dragged
   4-leg), **not statistically significant** (2043), **early-loaded** (no robust late-half),
   **thin cost cushion** (~1.3× FN), and **still NOT all-folds-positive** (the per-year gate fails even
   cost-free) — materially weaker than the committed-exit headline the path-A case was built on.
3. **Anchors reproduced** (ENB 3.32, LATE +0.40%, break-even κ≈3.3) — the recompute is trustworthy.

**FAIR caveat (§8, as 2040/2041/2042/1042/2043 — why FLAG not downgrade).** The committed exits are
defensible FIXED, mechanism-motivated choices (gap's overshoot-harvest horizon 1006/1007, me_long's
2-bar fix-flow, fbr's trailing-atr, me_short's partial-runner). Under the **fixed-mechanism reading**
the four legs are positive and the three pillars hold (ENB 3.32, LATE +0.41%, κ 3.39). Under the
**honest-§5f reading** (exit = nested hyperparameter) two legs go negative and all three pillars
weaken as above. The nested negativity is also partly small-n (8 eval folds) adaptive-selection
variance. So — exactly as the significance pillar (2043) — the **entire robustness profile is
reading-dependent**, and which standard governs is the operator's path-A call.

**FLAG F1 (docs only, no code change — discharges arc-1043 F2).** The path-A deploy case's three
remaining robustness pillars do NOT survive honest §5f exit accounting: (i) the 4-way book's
diversification is mean-DRAG under honest exits (the honest 4-leg book is worse than its 2 honest legs
me_long+fbr; gap & me_short are negative-mean drags), (ii) the book is not temporally stable (honest
LATE-half ≈ 0/negative vs committed +0.41%), (iii) the cost cushion quarters (break-even κ ≈ 1.3–1.5
vs 3.3). Combined with 2043/1043 (significance gone), path-A should weigh the honest book as a
**weakly mean-positive, not-significant, early-loaded, thin-cushion 2-leg (me_long+fbr) book that is
still not all-folds-positive** — NOT the committed-exit "significant, diversified, temporally-stable,
3.3×-cost-robust 4-way book."

**Components UNCHANGED** (all 4 stay PORTFOLIO; a chat does not unilaterally re-disposition a
committed component — the honest 2-leg book is still not AFP, no survivor). Lever unchanged = operator
path-A gate-governance call. No canonical change, no new BUILT tool, no council, **no OOS** (book
fails IS AFP; §5g operator firewall).

## (i) NEW lesson

For a thin mean-positive PORTFOLIO book whose deploy case rests on a SUITE of robustness pillars
(significance, diversification, temporal stability, cost cushion), those pillars are NOT independent
of the exit standard: if the book's component headlines are full-sample-best exit picks, the ENTIRE
profile is exit-optimistic, not just the mean. Concretely — once §5f exit-honesty flips two of four
legs mean-negative: (1) **diversification inverts from help to drag** (ENB / decorrelation survives,
but decorrelating INTO negative-mean legs makes the multi-leg book worse than its positive legs alone
— decorrelation is necessary-not-sufficient, and a committed-exit ENB silently assumes all legs
positive); (2) **temporal stability evaporates** (the late-half positivity was carried by the
now-negative legs + warmup folds the honest series loses); (3) **the cost cushion shrinks
proportionally to the halved mean** (break-even κ ≈ mean/cost-drag → ~quarters). Re-derive the WHOLE
robustness suite on the honest no-lookahead series at diversification-optimal weights before leaning
on any of it; here it converts the book's deploy case from "significant, diversified, stable,
cost-robust 4-way" to "weakly-positive, not-significant, early-loaded, thin-cushion 2-leg
(me_long+fbr), still not all-folds-positive." The autonomous programme has now characterized the book
on EVERY axis under BOTH the committed and the honest-exit reading; the deployability decision is the
operator's path-A call, with the full pillar suite now quantified as reading-dependent.

## (k) Re-orient

Detail persisted (this doc + DISCOVERY_LOG append, committed + pushed). Tools: REUSED
`nested_exit_selection` (2040) + `combine_fold_roi` (2006) + `scaled_fundednext` (3022) through the
canonical `build_fold_stats_from_run`; ENB/temporal are inline numpy (no new BUILT tool, as 2043).
Driver `_disco_work/arc2044_honest_pillars.py`. No canonical change, no council, no OOS. Next: resume
at arc 2045 (2000s range) — though the edge-hunt is structurally closed and the book is now fully
characterized under both exit readings; the remaining lever is the operator's path-A call.

# arc 2043 — Honest-§5f book-mean SIGNIFICANCE recompute (discharges arc-1042 FLAG F3)

**Chat:** 2000s · **Range:** 2000–2999 · **Disposition:** **DIAGNOSTIC → KILL** (no new
component; the book's mean-positive *significance* is recomputed under honest exits for the
operator's path-A call) · **Council:** none (a measurement resolving a measurement, as
arcs 1023/2016/2017/2019/3021/3022/1032)

---

## (a) Log read / synthesis

Pulled main; read DISCOVERY_LOG (both tiers, arcs 0–2042 + 1042), LESSONS, TOOL_REGISTRY; STOP
absent. State (honest-era only): single-condition shallow directional prediction is **closed
ground** (H1/H4/D1/W1, long+short); the edge-hunt is **structurally closed** (arc 3021 path-B
proof — a shared USD factor floors P(AFP) at any N; MENU M1/O1/L1/Q1/G1/S1 exhausted; ~18 routes
to a +2015/+2018 leg dead, including every short construction). **4 PORTFOLIO components** exist —
gap (1006), me_long (1011), fbr (1013, the load-bearing leg), me_short (1019, the only robust
+2018 leg). The **4-way book is mean-positive but never all-folds-positive** (provably
combination-invariant, blocked by 2015 & 2018; arcs 1020/2008/3009/1015). It is a sound
~3-independent-bet portfolio (ENB 3.32, arc 2019) whose AFP failure is a noise-floor/gate-resolution
artifact (arcs 2016/2017), not a diversification/tail/cost problem. **§11 independent verification
COMPLETE** end-to-end (signal 2034/2035/1037 + gross outcome 2036/2037/2038/1038/1039 + cost 2039,
all honest). The sole deployability lever is the **operator path-A gate-governance call** (arc 1032
quantified: coarsening the calendar gate yields no meaningful-resolution AFP pass → path-A = adopt a
mean/CI gate).

**The live fresh thread — exit-honesty (my chat's 2040→2041→2042, then 1000s' 1042).** §5f
nested-WFO exit selection showed the committed component headlines are full-sample-best exit picks:
fbr −40% (stays +), **me_short & gap SIGN-FLIP mean-negative**, only me_long is exit-robust. Arc 1042
recomputed the honest book MEAN (≈ +0.27% RP vs committed +0.59%) and raised **FLAG F3, explicitly
owed:** *"arc-1023/2019's t=2.66 significance pillar likely does not survive honest exits (estimated
t≈1.2); exact recompute owed."* That estimate was a mean-haircut guess, never a computed sd.

## (b) Idea (the *because*)

Path-A would deploy the book MEAN, and its entire statistical case rests on arc 1023's **t=2.66
(p≈0.026), fold-bootstrap CI [+0.22%,+1.09%], P(mean≤0)=0** and arc 2019's risk-parity CI
[+0.120%,+1.088%], P(mean<0)=0.004 — all computed on the **COMMITTED-exit** series. The exit-honesty
thread just halved the mean and flipped 2 of 4 legs negative. **The exact significance of the book
mean under each component's HONEST §5f nested-selected exit has never been computed** — only
estimated (1042, t≈1.2). This arc computes it: apply the EXACT arc-1023 across-fold significance
battery (t-stat, worst-fold z, fold-bootstrap CI seed 123 / B=20000, P(mean≤0), binomial sign) to
the honest §5f per-fold book series, with the committed series as a fidelity anchor / independent
reproduction. It resolves whether the path-A deploy case's "statistically mean-positive" pillar
survives honest exit accounting.

## (c)/(f) Method

- Scoring 100% canonical (`ArcFoldRunner` → `A1Architecture` → `MultiPairBacktester`, FundedNext,
  A1 default `risk_pct=0.005` = the linear deployable regime; **t is risk-INVARIANT there**, arc
  1024, so directly comparable to arc 1023). Selection via BUILT `nested_exit_selection` (arc 2040);
  book combination via BUILT `combine_fold_roi` (arc 2006). Significance is pure numpy arithmetic
  over the canonical per-fold numbers (arc-1023's method verbatim). **No gate reimplemented.**
- Honest grids (reproducing 1042): gap = pure-time (`exit_policy=None`) × horizon{12,18,24,36,48} ×
  SL{1.5,2,2.5} (the overshoot-harvest horizon is gap's chosen hyperparameter, arc 1007); me_long /
  fbr / me_short = 6 registry exits × 3 SL (me_long's 2-bar reversion horizon held fixed as
  mechanism, arc 2024). 3 nested metrics (mean_roi / afp_then_mean / worst_then_mean).
- **The honest reading is 8 EVALUABLE folds (2013–2020).** The nested walk-forward's first 2 folds
  (2011, 2012) are warmup — no in-sample exit choice exists, so the tool falls back to the *frozen
  full-sample* pick = NOT honest. The genuinely no-lookahead honest series is therefore the 8
  evaluable folds. To separate the exit effect from this n-change, the committed book is reported on
  BOTH the full 10 folds (arc-1023/2019 basis) AND the same 8 evaluable folds.
- **OOS NEVER touched** (the book fails IS AFP; the combined-book holdout is the operator's §5g
  firewall, arc 2022/1032). No new BUILT tool (reused 2040/2006; significance is inline arithmetic,
  as arc 1023). Driver `_disco_work/arc2043_honest_book_significance.py`, output `_disco_work/arc2043_out.txt`.

## Results

### Fidelity anchors — ALL reproduce (Arc-10 independent-reproduction discipline)
- Committed per-component means reproduce arc 1042/1020 **exactly**: gap +0.685%, me_long +0.232%,
  fbr **+2.084%** (the trail-OFF book config = arc-2040/1042 anchor, not arc-1020's +1.854%
  double-trail), me_short +0.683%.
- **Committed book at arc-2016 weights {gap 0, me_long .65, fbr .2, me_short .15}, n=10: t=+2.66,
  CI[+0.254,+1.184], P(≤0)=0.000** — **reproduces arc 1023's t=2.66 / CI[+0.22,+1.09] / P=0 EXACTLY.**
- Committed RP weights (gap .078/me_long .531/fbr .107/me_short .284), n=10: mean +0.594%, t=+2.16,
  CI[+0.108,+1.126], P(≤0)=0.005 — reproduces arc 2019's RP CI[+0.120,+1.088], P(mean<0)=0.004.
  ⇒ the recompute machinery is trustworthy; the honest numbers below rest on a reproduced anchor.

### F3 — the t=2.66 pillar does NOT survive honest §5f exits (the headline)

**Every honest §5f book cell (3 metrics × 3 weightings) is NOT statistically significant** — t ≤
+0.96, **all 9 CIs span zero**, P(mean≤0) 0.15–0.49:

| honest book (8 eval folds) | mean | sd | t | CI | P(≤0) |
|---|---|---|---|---|---|
| mean_roi · RP commit-wts (best) | +0.337% | 0.996% | **+0.96** | [−0.284,+1.002] | 0.154 |
| mean_roi · RP honest-refit | +0.088% | 0.552% | +0.45 | [−0.253,+0.464] | 0.322 |
| afp_then_mean · RP commit-wts | +0.034% | 0.629% | +0.15 | [−0.394,+0.409] | 0.409 |
| worst_then_mean · RP commit-wts | +0.109% | 0.575% | +0.54 | [−0.257,+0.482] | 0.285 |
| (worst cell) afp · EQUAL | −0.014% | 1.480% | −0.03 | [−1.015,+0.876] | 0.489 |

⇒ **F3 RESOLVED: under honest §5f exit accounting the book mean is +0.09…+0.34% RP but
statistically indistinguishable from zero (best t≈0.96; arc-1042's t≈1.2 estimate was close,
slightly optimistic).** The "statistically-significant mean-positive" pillar (arc 1023/2019)
collapses to "weakly mean-positive, not significant."

### Significance is fragile to BOTH the exit standard AND the fold-count (separated)

| book | n | t | CI | sig? |
|---|---|---|---|---|
| committed RP | 10 | +2.16 | [+0.108,+1.126] | borderline (P=0.005) |
| committed RP | 8 eval | **+1.28** | [−0.077,+0.704] | **NO** (P=0.072) |
| honest RP commit-wts (best metric) | 8 eval | +0.96 | [−0.284,+1.002] | NO |
| committed arc-2016 wts (gap=0) | 10 | +2.66 | [+0.254,+1.184] | YES |
| committed arc-2016 wts (gap=0) | 8 eval | +2.50 | [+0.117,+0.665] | YES |

- **The n-effect alone is large:** dropping the 2 strong warmup folds (2011/2012 — gap +8.23% in
  2012, fbr +8.97% in 2011) takes committed RP from t=2.16 → **1.28 (not sig)**. The honest
  no-lookahead exit series *structurally* lives on only 8 folds → it can never reach the committed
  10-fold t=2.66.
- **The exit-standard effect:** on the same 8 folds, honest selection further weakens RP (1.28 →
  0.96).
- **The arc-2016 hand-weighted book (gap=0) stays significant** (t≈2.5–2.66, robust to n) — but that
  is a *specific convex weighting that zero-weights the volatile/exit-fragile gap leg*. The
  diversification-optimal RP book and equal-weight book do not.

### Part 3 — the committed significance was a DIVERSIFICATION (portfolio) effect, not any single leg

**No single component is individually significant — even on COMMITTED exits** (8 eval folds): gap
t=−0.10, me_long +0.72, fbr +1.11, me_short +0.68 (all CIs span zero). The committed book's t=2.66
came *purely from combining decorrelated legs* (book sd ≪ component sds) — exactly the ENB-3.32
portfolio effect (arc 2019). Under honest exits: gap −0.85…−1.14% and me_short −0.03…−0.41% go
negative (confirms 2041/2042/1042), me_long stays robust +0.20% (t=0.72), fbr stays + but high-sd
(mean_roi metric picks the trailing_swing exit, sd 12.2% — the §5f trap; conservative metrics give
fbr +1.0–1.2%, sd 2.3–2.7%, t≈1.25). None individually significant under honest exits.

## (h)/(g) Verdict — DIAGNOSTIC → KILL (no new component; components UNCHANGED)

**Findings.**
1. **F3 discharged.** The book's mean-positive deploy case is **NOT statistically significant under
   honest §5f exit accounting** (best t≈0.96, all CIs span zero, P(mean≤0) 0.15–0.49). The arc-1023
   t=2.66 / arc-2019 P(mean<0)=0.004 pillar required BOTH the full 10 folds AND the committed-exit
   (or gap-zero) reading.
2. **Both confounds quantified and separated:** the honest no-lookahead exit series structurally
   loses 2 (strong) folds → committed RP already drops to t=1.28 (not sig) at n=8; the honest exit
   standard weakens it further to t=0.96. Significance at the conventional level rests on using all
   10 folds with a favorable exit/weight reading.
3. **The committed significance was a pure diversification effect** — no leg is individually
   significant even committed; the book's t came from combining decorrelated thin legs. Honest exits
   flip 2 of those legs negative, eroding the combination.
4. **Reproduced arc 1023 (t=2.66) and arc 2019 (RP CI/P) EXACTLY** — the recompute is anchored.

**FAIR caveat (§8, identical in spirit to 2040/2041/2042/1042 — why FLAG not downgrade).** The
committed exits are defensible FIXED, mechanism-motivated choices (gap's overshoot-harvest horizon
1006/1007, me_long's 2-bar fix-flow, fbr's trailing-atr, me_short's partial-runner). As FIXED choices
they reproduce t=2.16 (RP, n=10) / t=2.66 (arc-2016 wts). The nested-§5f negativity is also partly
small-n (8 eval folds) adaptive-selection variance. So the significance of the book mean is
**reading-DEPENDENT:**
- **Fixed-mechanism reading** (exit chosen a priori on mechanism): book is significantly mean-positive
  (t=2.16–2.66, n=10).
- **Honest-§5f reading** (exit = nested hyperparameter): book is NOT significantly mean-positive
  (t≤0.96, n=8, every CI spans zero).

This reading-dependence is the operator input. **Which standard governs is the path-A gate-governance
call** — the same lever the corpus keeps converging on, now with the significance pillar quantified
under both readings.

**FLAG F1 (docs only, no code change — discharges arc-1042 F3).** arc-1023/2019's t=2.66 significance
pillar does NOT survive honest §5f exit accounting: the honest book mean (+0.09…+0.34% RP) is
statistically indistinguishable from zero (best t≈0.96, all CIs span zero, P(mean≤0) 0.15–0.49). The
committed t=2.66 requires the full 10 folds AND the fixed-exit/gap-zero reading; the genuinely
no-lookahead 8-fold honest series is not significant under any metric/weighting. Path-A should weigh
the book's deploy case as *weakly mean-positive (≈+0.2–0.3%/yr, IS≈OOS per 1042) but not
statistically significant under honest exits* — not the headline t=2.66.

**Components UNCHANGED** (all 4 stay PORTFOLIO; a chat does not unilaterally re-disposition a
committed component). Lever unchanged = operator path-A gate-governance call. No canonical change, no
new BUILT tool, no council, **no OOS** (book fails IS AFP; §5g operator firewall).

## CONVERGENCE — independent concurrent reproduction (arc 1043, 1000s chat)

The 1000s chat resolved the SAME arc-1042 F3 concurrently (arc 1043, landed on main at push). **Same
verdict, complementary method** (Arc-10 independent-reproduction):
- **Agreement:** both reproduce the committed anchor (RP t=+2.16, CI strictly >0 ≡ arc 1023) and both
  conclude the honest §5f book mean is **positive but NOT statistically significant — every CI spans
  zero** under every metric/weighting. F3 is resolved the same way by two chats with different code +
  seeds.
- **Complementary, not redundant:** arc 1043 held **n=10 constant** (fold-bootstrap of all 10 IS folds
  incl. warmup, seed 42) → honest t ∈ [0.83, 1.52], isolating the *pure exit effect* (committed
  2.16 → honest ~1.3 at constant n). This arc used the **8 EVALUABLE folds** (seed 123) — the
  genuinely no-lookahead honest series, since warmup folds use the frozen full-sample pick = not honest
  → honest t ≤ 0.96, and explicitly **decomposed the two confounds** (n-effect: committed RP
  2.16→1.28 at n=8; exit-effect on top: →0.96) plus the **Part-3 diversification finding** (no leg
  individually significant even committed). arc 1043 added the valuable **F2**: arcs 2021 (temporal),
  3022 (cost-κ), 2019 (ENB) were ALL computed on committed-exit series → owed an honest-exit recheck.
- **Combined statement:** the honest series is non-significant whether measured at n=10-with-warmup
  (1043's ~1.3) or n=8-honest-evaluable (this arc's ≤0.96); the significance pillar is gone under
  honest exits by two independent paths.

## (i) NEW lesson

For a thin, mean-positive PORTFOLIO book whose edge is a **diversification effect across thin
decorrelated legs** (no leg individually significant), the book-mean's statistical significance is
**triply fragile**: (1) to the exit standard (honest §5f selection halves the mean & flips legs
negative → t 2.66→≤0.96), (2) to fold-count (the honest no-lookahead exit series structurally loses
the warmup folds → committed RP itself drops to t=1.28 at n=8), and (3) to the weighting (the
gap-zero arc-2016 convex pick stays t≈2.5, the diversification-optimal RP/equal books do not). The
"statistically-significant mean-positive" framing that anchors a deploy case must be recomputed on
the **honest no-lookahead series at the diversification-optimal weights** — not on the committed
full-sample-exit headline — before it is leaned on. Here that recompute removes the significance
pillar: the honest book is weakly mean-positive but indistinguishable from zero. The autonomous
programme has now characterized the book on every axis (noise 2016/17/1023/2019, time 2021, cost
3022, gate-coarsening 1032, risk geometry 1033, vehicle 2033, §11 honesty 2034–2039, exit-honesty
2040–2042/1042, and now significance-under-honest-exits) — the deployability decision is the
operator's path-A call, now with the significance pillar quantified as reading-dependent.

## (k) Re-orient

Detail persisted (this doc + DISCOVERY_LOG append, committed + pushed). Tools: REUSED
`nested_exit_selection` (2040) + `combine_fold_roi` (2006); significance is inline numpy (arc-1023
method), no new BUILT tool. Driver `_disco_work/arc2043_honest_book_significance.py`. No canonical
change, no council, no OOS. Next: resume at arc 2044 (2000s range) — though the edge-hunt is
structurally closed and the book is now fully characterized; the remaining lever is the operator's.

# arc 1013 — Failed-breakdown RECLAIM ("stop-run reversal") long, multi-factor

**Chat:** 1000s · **Range:** 1000–1999 · **Disposition:** **PORTFOLIO** · **Council:** none (PORTFOLIO, not a PASS survivor — arc-1011 precedent)

> DEEP-arc dispatch: the shallow single-trigger directional slice is closed; multi-factor directional
> is the open frontier (§5a novel-mechanism carve-out). This arc builds a conjunctive, structurally
> grounded entry and develops the best version on IS before any verdict.

---

## 1. READ + SYNTHESIZE THE LOG (step a)

Honest-era corpus = trial (arc 0) + 24 arcs across 3 chats. State at entry:

**Closed (shallow directional):** the long base is a ~0.4877 coin-flip on H4 majors, and that is
**instrument-invariant** (28-pair scan, crosses ≤ majors — 3000/3001), **timeframe-invariant**
(H1≡H4≡D1 — 1002/2002), and **metric-robust** (capture AND mean-forward-drift agree — 3001).
Momentum / breakout / mean-reversion / trend-following all net ≈ cost. **Regime conditioning failed
three ways**: dispersion (1000), vol-level (1001), Kaufman-ER (3003 — strong trends *invert*/revert).
Volume = magnitude not direction (3002). Exits (1004/2000), the 2·ATR stop (3004/2004), and the cost
side (2005/3007) are all real but secondary ~5–10 pp drags, **not the wall** — direction is the wall.

**The only net-positive edges** are discrete flow-event *reversions* where direction is NOT a
coin-flip: weekend gap-fill JPY crosses (1006/1009, PORTFOLIO) and month-end D1 reversion USD majors
(1011/1012, PORTFOLIO). Their 2-way combo **KILLed** (2006) — mutually-negative 2015 fold + tail
correlation; the route needs a ≥3rd component positive in risk-off years.

**Flow/structure mostly dry:** gotobi absent (1008), round-numbers arbitraged (1010), triangulation
~0 (3005), turn-of-month sub-cost (1005), month-end *fix-window* sub-cost (3008), breakout-RETEST
*continuation* arbitraged (3006).

**Open frontier (my dispatch):** DEEP multi-factor directional. The capturability lesson is decisive
— every dead reversion (gap-fill, RSI, month-end-fix) entered *into* the adverse move; the one
mechanistic gap nobody exploited is a **confirmation that the adverse move is already over at entry.**

## 2. THE IDEA + WHY (step b — ideation shown, per dispatch)

**Structural *because*.** Liquid FX carries predictable resting **sell-stop clusters just below
visible swing lows** (retail protective stops; breakout-sellers' stops). A bar that **pierces** a
K-bar swing low (running those stops — a liquidity grab that supplies fills for larger players to
absorb) and then **closes back above** the level is a **failed breakdown**: the down-move was
*liquidity-driven, not informational*, and the **reclaim is the tell** it has failed. This is a
documented order-flow behaviour (stop-runs / liquidity sweeps), not a chart shape.

**Why several conditions must interact (the multi-factor logic).** A bare reclaim is not enough —
the *because* only holds in a specific conjunction, and each layer has a reason:
- **Structure** (K-bar swing low) — the level must be one with stops actually resting below it
  (a *visible* low). Larger K = more significant level = denser stops.
- **Sequence/path** (pierce → reclaim, ex-ante) — the failed-breakdown is a *two-state* event, not a
  point-in-time read; the reclaim is the confirmation that fixes the capturability wall.
- **Sweep depth** (moderate, not a multi-ATR collapse) — a shallow grab is a liquidity sweep; a deep
  break is informational (real sellers) and the reclaim there is just a pause. Conditioning OUT the
  deep breaks is the hypothesis that separates grab from genuine break.
- **Session** — the reclaim needs *participants* to drive the reversal; an Asia-hours reclaim is thin
  noise, a London/NY reclaim has institutional flow behind it.
- **MTF context** (D1 not in a genuine downtrend) — don't fade a real informational down-leg; the
  stop-run edge should live where the higher TF is balanced/up.

**Why it is NOT closed ground (fresh-eyes + DRY-prior check, §5a).**
- It is the **inverse of arc 3006** (break-and-HOLD *continuation* → this is break-and-FAIL
  *reversal*) — a different mechanism, not a re-cut.
- It is **not indicator mean-reversion** (RSI/z, 3000/3001 — dead): it is anchored to a structural
  level + a sequence, with an order-flow because.
- It is **not arc 0's bare pierce-the-dip** (close>SMA50 & pierce prior-5-low): arc 0 entered *into*
  the dip (sub-cost, lands in the adverse excursion) with NO reclaim, NO session, NO MTF, NO depth.
  The **reclaim + conjunction** is the whole mechanistic difference and the documented *because*.

**Rejected at ideation (and why):**
- A 3rd flow-reversion component (the portfolio route's named hunt) — would inherit the same 2015
  tail (arc 2006 lesson: a reversion-flavoured leg won't fix the mutually-negative fold). A directional
  *reversal* with a different regime signature is a better swing if it clears solo, and the dispatch
  asks for DEEP directional, not another flow edge.
- Short side (the strongest stop-run leg is arguably the up-sweep short) — long-only lane holds until
  the operator flips shorts; long failed-breakdown reclaim is the tradeable side now.

## 3. OBSERVATION (step b/c)

H4 + D1, 7 USD majors, IS 2010–2020. Base capture **0.4860** (≈ the 0.4877 reference). Drivers:
`_disco_work/arc1013_{observe,refine,control}.py`.

**The bare reclaim is ≈ coin-flip** (K=10 cap 0.4881, K=20 0.4931, K=40 0.4965) — the corpus prior holds.
**My hypothesised conjunctions mostly failed or INVERTED** (the data led, not my priors):
- **Session** (UTC hour): scattered, arc-1010 noise signature — no clean separation.
- **D1 MTF context**: *inverted* — the reclaim is BETTER when D1 ≤ SMA50 (cap 0.498–0.504) than in a
  D1 uptrend (0.475–0.477). "Don't fade a downtrend" was wrong; I dropped it.
- **Sweep depth**: also inverted, *informatively* — the EDGE is in DEEP pierces. A wick that spikes
  >1 ATR below a K-bar swing low and still closes back above it has cap **0.55–0.56** (K=40 depth>1.0:
  0.5496, drift +0.222; K=60: 0.5636, drift +0.301), broad across **6–7/7 pairs**, both lenses agreeing.

**The signal refined to: deep wick-rejection that sweeps & reclaims a swing low.** Sharpening with the
lower-shadow magnitude: shadow≥1.25 cap 0.5865 (n=237), shadow≥1.5 cap 0.6107 (n=131) — the highest
directional captures in the corpus.

**DECISIVE CONTROL (arc-1011 discipline — is structure load-bearing, or generic wick-bounce?).** Split
magnitude-matched rejection bars by whether they swept a swing low:

| rejection magnitude | (A) AT swept swing low | (B) elsewhere (control) |
|---|---|---|
| shadow≥1.0 | n=507 cap **0.5227** drift +0.049 | n=1577 cap 0.4807 drift −0.049 |
| shadow≥1.25 | n=237 cap **0.5865** drift +0.146 | n=634 cap 0.4905 drift −0.217 |
| shadow≥1.5 | n=131 cap **0.6107** drift +0.203 | n=296 cap 0.5068 drift −0.176 |

The same-magnitude wick AT a swept swing low captures 0.52→0.61; the SAME wick elsewhere is a coin-flip
with *negative* drift, and the excess GROWS with rejection size. **The structure (sweeping a real
stop-cluster low and reclaiming it) IS the mechanism — not the generic wick** (generic deep-down
reversion is dead, 3000/3001). The *because* holds.

## 4. BEST-VERSION on the HONEST ENGINE (§5f) + soundness

Built `discovery/tools/failed_breakdown_signals.py :: FailedBreakdownReclaimLongSignal(K, min_shadow_atr)`
(EXPERIMENT: mask + ATR geometry only; scoring canonical). Pool floor PASS (n=131–237 IS). Drivers:
`_disco_work/arc1013_{wfo,null_robust,oos,loo}.py`. Folds: `build_v3_folds` IS + `build_oos_year_folds`
(2021+), scored solely by `MultiPairBacktester` (FundedNext, SL-first), judged `judge_all_folds_positive`.

**§5f exit menu (K=40, shadow=1.25, SL=2.0)** — EVERY exit is mean-POSITIVE (unprecedented for a
directional long; the corpus's only positives were flow-reversions at +0.23–0.69%):

| exit | mean | neg/10 | worst | maxDD |
|---|---|---|---|---|
| sl_only | +1.43% | 2 | −4.63% | 6.40% |
| sl_plus_tp_3r | +1.57% | 2 | −4.63% | 5.54% |
| **sl_plus_trailing_atr** | **+1.85%** | **1** | −4.20% | 4.87% |
| sl_partial_close_1r_runner_trail | +1.37% | 2 | −2.88% | 4.05% |
| sl_only+time8 | +1.59% | 2 | −2.68% | 3.88% |

**Committed config (on IS, no OOS peek): K=40, shadow≥1.25, `sl_plus_trailing_atr`, SL=2.0** — the
most-populated robust cell (minN=16); best-mean exit but the choice is robust (all exits mean-positive);
shadow≥1.25 is the mechanistically-motivated rejection gate (control excess grows with shadow; shadow=1.0
dilutes to 4/10 neg).

**Soundness (the dispatch's mandated measurement integrity):**
- **Fair same-exit NULL (decisive):** REAL +1.854% vs random-entry NULL avg **−1.107%** (matched
  fire-rate, same exit/pairs; 3 seeds) → **excess +2.96pp**. The null is firmly NEGATIVE — the structure
  is the entire edge, not exit geometry or regime drift (cleanest null in the corpus; cf. arc 1009's
  gap-fill null +0.327% which halved its headline).
- **K/shadow robustness:** mean-positive across K∈{40,60}×shadow∈{1.0,1.25,1.5}; sweet spot shadow≥1.25
  (1–2/10 neg at both K); shadow=1.0 dilutes (4–5/10) — the rejection gate is load-bearing, not one cell.
- **Leave-one-pair-out:** mean +1.18% to +1.93% dropping ANY pair (EURUSD strongest contributor, USDJPY a
  mild drag) — broad-based, not one-pair regime-luck.

**ONE-SHOT frozen OOS (2021+):** mean **+0.936%**, 3/6 neg (2022 −1.39%, 2025 −2.76%, 2026 −0.57% n=5),
maxDD 4.20%. **Mean-positive out-of-sample** — the edge PERSISTS forward (where arc 2005 died OOS −2.26%).

**Per-fold (committed config), fold→OOS-year (anchor: fold 6 = 2015, arc 2006):**
`IS 2011:+7.55 2012:+3.05 2013:+0.91 2014:+0.19 2015:+3.17 2016:+2.55 2017:+1.23 2018:−4.20 2019:+0.05 2020:+4.03`
· `OOS 2021:+1.63 2022:−1.39 2023:+1.10 2024:+7.61 2025:−2.76 2026:−0.57`.
The negatives cluster in **persistent strong-USD / risk-off years** (2018, 2022, 2025) where breakdowns
DON'T fail — a coherent regime characterization, the arc-1012-class regime drag (not exit-fixable; not
fished).

## 5. VERDICT — PORTFOLIO (mean-positive IS+OOS, beats null decisively, robust; NOT all-folds-positive)

The failed-breakdown reclaim is a **real, mechanism-controlled, null-beating, broad-based, OOS-persistent
long-only edge** — the strongest and cleanest directional result in the 25-arc corpus (IS +1.85% vs
gap-fill +0.69% / month-end +0.23%; the only one that beats a NEGATIVE null and survives OOS mean-positive).
It is NOT all-folds-positive (the 2018 strong-USD regime fold; OOS 2022/25 share that signature) → it is
not a solo survivor (not PASS), but a strong **PORTFOLIO component** (§11).

**Why this is the high-value 3rd component (arc-2006 spec).** Arc 2006 KILLed the 2-way book (gap-fill +
month-end) on a mutually-negative fold and tail-correlation in 2015/16/18/20, specifying a 3rd selected by
its ROI on the book's NEGATIVE folds. This component is a **different mechanism family** (structural
stop-run reversal vs flow-event reversion → genuinely decorrelated) and is **POSITIVE in 2015 (+3.17),
2016 (+2.55), 2020 (+4.03)** — three of the four years that book bled — sharing only **2018** with it. The
3-way combination (2000s' arc-2006 machinery) should reduce the book's mutually-negative folds from four to
≈one (2018) → a concrete, quantified advance for the portfolio route.

**Threads / lessons.**
1. **DEPTH closes the directional coin-flip — the shallow slice was hiding a deep-tail structural edge.**
   25 arcs found direction ≈ coin-flip, but every prior cut was shallow/single-trigger; the multi-factor
   conjunction (swing-low structure × pierce-reclaim sequence × deep-rejection magnitude) reaches cap
   0.55–0.61. **Depth, not breadth, was the missing axis** — the dispatch's thesis, confirmed.
2. **The RECLAIM fixes the capturability wall.** Every dead reversion (gap-fill 2001/2002, RSI 3000/3001,
   month-end-fix 3008) entered INTO the adverse move; the reclaim confirms the move is over AT entry — the
   first reversion whose honest i+1 entry is mean-positive net of cost.
3. **My session/MTF priors were wrong; the data inverted them.** Showing the ideation (dispatch ask): I
   hypothesised active-session + D1-uptrend conjunctions; both failed/inverted, and DEPTH (which I'd guessed
   *down*-weighted as "informational") was the real conditioner. Develop-on-IS-let-data-lead earned its keep.
4. **Cleanest null in the corpus** (real beats a −1.1% null by +2.96pp) — a directional structural edge, not
   the exit/drift artifacts that thinned the flow components.
5. **The regime weakness is honest and named:** persistent strong-USD/risk-off trend years (2018/2022/2025)
   where breakdowns are real, not swept. A future refinement could *condition out* those regimes — but only
   with a pre-registered, causal regime measure tested fresh (NOT fished to flip 2018; arc-1012 trap).

**Disposition: PORTFOLIO.** Recorded at `discovery/portfolio-candidates/arc_1013_failed_breakdown_reclaim_long/`.

**FLAGS (code not merged):** none requiring the canonical core. Built + registered (BUILT, direct to main)
`FailedBreakdownReclaimLongSignal` (EXPERIMENT signal tool). Carries the standing `A1Config.time_exit_bars`
-unwired flag (arcs 1005/3004; worked around via the BUILT `make_time_exit_predicate`). Drivers in scratch
`_disco_work/arc1013_*.py` (reproducible from this doc; obs cached at `_disco_work/obs_h4_usd.parquet`).

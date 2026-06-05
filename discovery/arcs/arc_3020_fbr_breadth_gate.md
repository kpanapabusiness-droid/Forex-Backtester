# arc 3020 — fbr × cross-pair BREADTH gate (target fbr's 2018 hole)

**Chat:** 3000s · **Date:** 2026-06-05 · **Verdict:** FAIL (cheap-kill at observation) → **KILL**
**Disposition:** KILL · **passed:** N · **Component touched:** arc-1013 fbr (UNCHANGED, still PORTFOLIO)

> Can the corpus's single real fold-level obstacle — `fbr`'s −2018 (arc 2017) — be conditioned out by
> a CROSS-PAIR BREADTH gate (fire only when few other USD majors are simultaneously firing an fbr)?
> **No.** The motivating premise is falsified: 2018 fbr fires are NOT more clustered than normal years.
> A real secondary texture exists (solo fires drift +0.52 vs clustered −0.4) but it does not reach
> all-folds-positive and its 2018 "fix" is n=8 regime-luck — reproducing arc-2014's "any gate that
> fixes 2018 breaks good folds." **5th independent confirmation that fbr-2018 is unconditionable.**

---

## Log reading (step a — FRESH EYES, honest-era only)

Pulled main; read DISCOVERY_PROTOCOL, the full DISCOVERY_LOG Tier-1 (arcs 0→3020), LESSONS, TOOL_REGISTRY.
No `discovery/STOP`. Resuming 3000s at 3020 (highest in-range = 3019).

State after ~55 arcs across 3 chats (honest-era), the converged picture:
- **Closed ground:** single-condition shallow directional prediction (momentum/breakout/mean-reversion/
  trend), H1/H4/D1/W1, majors+crosses, **long AND short**, every exit, stop-removed — forward drift ≈ cost
  everywhere. Regime conditioning failed 4 ways (dispersion/vol-level/Kaufman-ER/vol-expansion). Intraday
  & relative-value sub-cost (H1 cost wall + doubled-cost-vs-coin-flip).
- **Four net-positive PORTFOLIO components, none all-folds-positive:** gap-fill 1006 (+0.685% IS),
  me_long 1011 (+0.232%), **fbr 1013 (+1.854%, 9/10 — STRONGEST + the ONLY fold-resolving edge)**,
  me_short 1019/3017 (+0.683%, first robustly +2018).
- **Portfolio route** (2-/3-/4-way: 2006, 2008/3009/1015, 1020/2015) all FAIL all-folds-positive; best
  4-way worst-fold −0.115%, blocked by MARGINAL 2015 & 2016.
- **Decisive recent diagnostics (2016/2017/1023/2019):** the 4-way book is a sound ~3-independent-bet
  (ENB=3.32), **mean-positive** (t=2.66, P(mean<0)=0.004) PORTFOLIO whose all-folds-FAIL is purely
  **below its noise floor** — every worst fold is a single thin-leg noise dip, not a co-drawdown.
  arc 2019: "edge-hunting for the book is closed; the lever is the operator gate-governance call." arc
  2017: the route's **single statistically-real fold obstacle is `fbr`'s −2018** (the only CI<0 negative
  fold in the whole corpus); arc 2017 names the only productive remaining spec as **option B = a THICK,
  fold-RESOLVING standalone** (the thin components make the per-year gate vacuous).
- **fbr-2018 attacked & found mechanism-intrinsic on 3 axes:** 2014 (per-pair downtrend regime-gate →
  INVERTED, 2018 is 18/19 trades −1R near-total wipeout, any gate that drops 2018 destroys good folds),
  2020 (M1 reclaim-quality → no entry-time tell), 1025 (depth/density → degrades monotonically, fbr
  doesn't thicken). Also 3013 (level axis → all levels −2018). 2018-leg unfound in EVERY short
  (1014/2009/2011/3011/3010/1016/2013/3012/1017) and the named non-reversion 2015&2016 leg (3019 shock-
  continuation) died OOS (epoch-dependent).

**Open thread I picked:** arc 2017's option B (a thick fold-resolving edge) + improving `fbr` toward a
solo-PASS — the single highest-leverage target (fbr is 9/10 IS, ONE fold from the corpus's first
standalone PASS, and the only fold-resolving edge). The 2018 hole has been hit on per-pair-regime / M1 /
depth / level axes but **NOT on the cross-pair BREADTH axis** — genuinely untried.

## Idea + why (a *because* tied to fbr's known 2018 failure mode)

arc 2014 diagnosed 2018 as the regime where the failed breakdown becomes a REAL breakdown (the reclaim
doesn't hold in strong-USD risk-off). **Mechanism hypothesis:** in a USD-wide risk-off move MANY USD
majors print a failed-breakdown-reclaim on the SAME few days (the dollar gaps everything down together),
and those broad/clustered reclaims are the ones that fail (it's a real USD trend, not an idiosyncratic
liquidity grab); in normal regimes fbr fires idiosyncratically (1–2 pairs) at genuine per-pair support
pivots and the reclaim holds. **If so, a LOW-BREADTH gate** (fire only when few other majors are
simultaneously firing) **drops 2018's clustered false-reclaims while keeping good-fold idiosyncratic
winners → possibly fbr's first all-folds-positive (solo-PASS).** Cross-pair breadth is a distinct
entry-time axis from 2014/2020/1025/3013.

## Method (CALLED canonical; observation only — §5d)

`Panel.from_pairs` (H4, cached) on the 1013 universe (7 USD majors). Built `FailedBreakdownReclaimLong
Signal(K=40, shadow=1.25)` (BUILT, arc 1013), honest long capture + 24-bar drift via BUILT
`observe_long_capture` restricted to the fire mask. IS window 2010–2020 only. **BREADTH** per fire =
count of OTHER majors firing an fbr within ±{0.17, 1, 3, 7} days; primary proxy = ±3 days. Binned
solo(0) / low(1) / high(≥2); compared 2018 vs non-2018 breadth, and per-year drift under breadth
filters. CHARACTERIZATION ONLY (gross, take-the-loss capture + drift; NOT a gate). Driver:
`_disco3_work/arc3020_fbr_breadth.py`.

## What happened — premise FALSIFIED; secondary texture real but no solo-PASS

**Reproduction (Arc-10 discipline):** 237 IS fbr fires, unconditional capture **0.5865**, drift mean
+0.146 / median +0.256 — reproduces arc 1013 / 3013 exactly.

**Breadth separates DRIFT (real secondary texture), in the hypothesized direction:**

| breadth bin (±3d) | n | cap | drift mean | drift med |
|---|---|---|---|---|
| solo (0 others) | 140 | 0.579 | **+0.519** | +0.275 |
| low (1 other) | 60 | 0.583 | −0.354 | +0.130 |
| high (≥2 others) | 37 | 0.622 | **−0.452** | −0.837 |

→ clustered fbr fires DO have worse drift (the "USD-wide move = less-reliable reclaim" texture is real).

**But the 2018 premise is FALSE — 2018 is NOT more clustered:**

| | n | cap | drift | breadth(±3d) | share high(≥2) |
|---|---|---|---|---|---|
| non-2018 | 218 | 0.596 | +0.271 | 0.560 | 0.156 |
| **2018** | 19 | 0.474 | −1.282 | **0.737** | **0.158** |

2018's breadth (0.74) and high-share (0.158) are ~identical to normal years (0.56 / 0.156). The
hypothesized broad-clustering signature of 2018 **does not exist** → breadth cannot select the 2018 regime.

**Decisive per-year test — breadth filtering does NOT yield all-folds-positive:**
- Drop high-breadth (keep ≤1): **6/11 years drift>0** (WORSE than all-fires 7/11); 2018 stays −1.85.
- SOLO-only (breadth==0, 140/237 fires): **8/11 years drift>0** — makes 2018 positive (+0.59) BUT
  BREAKS 2010 (−0.32), 2012 (−1.00), 2016 (−1.65). The 2018 "fix" is **n=8 (regime-luck**, arc-1017/3010
  tell), and the solo edge is **carried by 2 pairs** (EURUSD +2.20, USDCAD +1.17; AUDUSD/USDCHF/USDJPY
  flat-to-negative, USDJPY n=35 ≈ 0.00). Trading a thin 2018-fix for 2010/2012/2016-breaks = exactly
  arc-2014's "any gate that removes the 2018 pocket destroys good folds," now on the breadth axis.

## Diagnosis — fbr-2018 is breadth-unconditionable; 2018 ≠ a clustering event

Two findings, one decisive: (1) **The motivating mechanism is wrong** — 2018's fbr failures are not a
broad simultaneous-clustering phenomenon (2018 breadth ≈ normal). 2018 fails because the *individual*
reclaim doesn't HOLD over the following days in the strong-USD trend (a FORWARD failure, per arc 2020),
which an entry-time breadth count cannot see. (2) **The real secondary texture** (solo fbr fires beat
clustered ones, +0.52 vs −0.4) is a genuine "idiosyncratic-grab > USD-wide-move" effect but is (a) not
all-folds-positive, (b) thins the already-thin edge (arc 1025: thinner = worse fold resolution), (c)
carried by 2 pairs, and (d) its only 2018 lift is n=8. Selecting a breadth threshold to flip the 2018
fold would be the arc-2016 Soundness-lens trap (gate-painting below the noise floor).

This is the **5th independent confirmation that fbr's −2018 is mechanism-intrinsic and entry-time-
unconditionable** (2014 regime-gate, 2020 M1, 1025 depth, 3013 level, **3020 breadth**). Per arc 2017,
fbr-2018 is the route's one statistically-real fold obstacle; it is now closed on the breadth axis too.

## Verdict: KILL (cheap-kill at observation)

Breadth does not separate the 2018 regime (premise falsified), and the secondary solo-texture neither
reaches all-folds-positive nor survives robustness — no solo-PASS candidate, no improvement to fbr. No
engine/null/council spent: the breadth axis is decisive at observation (premise falsified + the only
fold-flipping filter breaks good folds and rests on n=8). §5f does not bite — this is a falsified
CONDITIONING of an existing component whose exit menu was already swept (arc 1013/1025), not a new
non-coin-flip entry; the fbr entry family + its frozen exits are UNCHANGED. arc-1013 component
UNCHANGED (still PORTFOLIO); no new `portfolio-candidates/` entry (would double-count). OOS never touched.

## Threads / lessons

1. **NEW (real secondary finding): clustered fbr fires are worse than idiosyncratic ones** — solo fires
   (no other USD major firing within ±3d) drift +0.52 vs clustered −0.4, monotone in breadth. The
   "USD-wide simultaneous failed-breakdown = less-reliable reclaim (it's a real dollar trend, not a
   per-pair liquidity grab)" texture is REAL. But it is NOT 2018-aligned and does NOT yield a solo-PASS.
   A possible (low-priority) future refinement of fbr's *quality*, not a fold-fixer.
2. **2018 is NOT a clustering/breadth event** — 2018's fbr breadth ≈ normal years. 2018 fails per-pair,
   forward (the reclaim doesn't hold over following days), which no entry-time count can filter. Confirms
   arc-2020's "forward-reversal" diagnosis from the breadth angle.
3. **fbr-2018 unconditionable across FIVE independent entry-time axes** (regime 2014 / M1 2020 / depth
   1025 / level 3013 / breadth 3020) — overwhelming evidence it is mechanism-intrinsic. The standalone-
   PASS-via-fbr and improve-the-best-edge lanes are closed at the entry side; arc-2017's option-B
   (thick fold-resolving standalone) is not reachable by refining fbr.
4. **Route status reinforced (converges with 2019):** the deployability lever is the operator gate-
   governance call on a sound, mean-positive, ~3-bet PORTFOLIO that fails only a per-year gate sitting
   below its noise floor — not a missing/improvable edge. Autonomous edge-hunting for the book is closed;
   I will NOT spend OOS or loosen the gate (conservative bias + measurement integrity).

## Tooling

No new BUILT tool — reused BUILT `FailedBreakdownReclaimLongSignal` (1013) + `observe_long_capture`
(post-1007). The cross-pair breadth count is a one-off scratch conditioner in the driver (not a reusable
filter: it's specific to this falsified hypothesis). Driver `_disco3_work/arc3020_fbr_breadth.py`
(reproducible from this doc).

## FLAGS (code not merged)

None. No canonical-core change. Carries the standing `A1Config.time_exit_bars`-unwired flag + the
4-way-book gate-governance OPERATOR FLAG (arc 2016/2017/2019/1023 — the per-year all-folds gate sits
below the corpus's noise floor; path A mean/pooled/regime-block gate vs path B denser components is the
operator's call).

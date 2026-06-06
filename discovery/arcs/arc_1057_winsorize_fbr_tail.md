# arc 1057 — TAIL-DEPENDENCE of the honest 2-leg deploy book mean (winsorize the fbr runner tail)

**Chat:** 1000s · **Range:** 1000–1999 · **Disposition:** **DIAGNOSTIC → KILL** (no new component;
components UNCHANGED, all 4 PORTFOLIO) · **Council:** none (a measurement resolving a measurement —
1056/1043/2016 precedent) · **OOS:** NEVER touched (book fails IS AFP; §5g firewall, IS-only).

> **Headline.** Arc 1056's owed thread — *cap the fat tail and see whether the deploy mean / significance
> survives* — resolves **decisively against robustness.** The honest **2-leg me_long+fbr** book's
> borderline significance is **fbr-runner-tail-dependent in the strongest sense:** removing the **single
> largest position** (the fbr 2015 runner, $2,513) drops the risk-parity book from **SIG+ (cluster
> t=2.29)** to **non-significant (t=1.52)**; **winsorizing fbr's positive tail at the 95th percentile**
> also kills it (t=2.29→1.89, ~0); removing the **top-5 of 245 positions** collapses the mean to
> **+0.07%/yr** (RP) / **−0.05%/yr** (equal). Per-leg attribution is unambiguous — winsorizing **only
> fbr** kills the t (2.29→1.90) while winsorizing **only me_long** leaves it intact (2.28). The fbr leg's
> **top-5 positions = 127% of its total net P&L** (the other 165 net-negative). So the deploy mean is
> not merely *borderline* (1056) but *tail-fragile*: it cannot be treated as a robust, sizeable
> expectancy — it is one un-repeatable convex fbr-runner regime away from zero. A THIRD, more fundamental
> limit on the path-A "significant mean" pillar, alongside the AFP-gate wall (1032) and the vehicle/Calmar
> wall (2033/2059). Components UNCHANGED; deployable-system count = 0.

---

## 1. Read + synthesize (step a)

Pulled main (clean; highest 1000s arc = 1056, highest 2000s = 2059). No `discovery/STOP`. Read the
protocol, TOOL_REGISTRY (CANONICAL + BUILT rows), LESSONS (Closed-ground prior), and the recent ledger
(1042–1056, 2040–2059).

**State.** The OHLC-only EDGE frontier is exhaustively mined out: the explore-now MENU (M1 1027/2023, L1
1054, O1 1029/1025/1055, Q1 1028, G1 2052) is closed; shorts revive neither structure (1014/2009/2011/
3010/3012/1035) nor relative-value (2003/2010/2018); the leg-hunt route is *provably* closed (3021: a
shared USD/risk factor ρ≈+0.12 floors P(AFP) below the gate at any N). Four PORTFOLIO components (gap 1006,
me_long 1011, fbr 1013, me_short 1019); the combined book FAILS all-folds-positive (combiner-invariant, co-sim
item E). §11 verification is COMPLETE (signal + outcome + cost all independently audited honest, 2034–2039/
1037–1039/2046). The sole live lever is the operator's **path-A gate-governance call**; the only remaining
within-charter value is **decision-support diagnostics on the characterized object** (chat-1000s handoff /
DISCOVERY_DIRECTION), NOT new edge obs-kills.

**The explicitly-owed thread (the resume pointer).** Arc 1056 resolved arc 1043's per-trade-bootstrap thread
and found the honest 2-leg deploy mean (+0.41%/yr) only **BORDERLINE** significant (cluster RP t=2.27 / equal
t=1.91), with within-year sampling sd (1.08% RP) **exceeding** across-year spread (0.55%) → the per-year ROIs
are **fat-tail-(fbr-runner)-dominated**, the significance "rests on a lucky tight clustering of noisy annual
numbers." The natural falsifier 1056 named (and the chat-1000s handoff pointer): **quantify the
tail-dependence — winsorize / leave-out the fbr runner tail and watch the mean & t.** This is the
highest-value remaining within-charter diagnostic on the single live lever, zero fabrication surface
(winsorization arithmetic on canonically-scored per-trade net P&L).

## 2. Idea (step b)

A deploy mean carried by a few un-repeatable fat-tailed convex-runner trades is not a robustly-deployable
edge — you cannot size to a mean that is one lucky +R fbr-runner regime away from zero. Test it directly:
**winsorize** (cap above a percentile) and **leave-top-N** (remove the largest positions) the honest 2-leg
book's per-position net P&L, recompute the cluster bootstrap (the correct SE-of-mean, 1056), and measure how
much of the +0.41%/yr mean and its borderline t survives. Per-leg attribution isolates whether the tail
dependence is the fbr runner (the hypothesis) or me_long. No council (a measurement resolving a measurement).

## 3. Method (steps c–g) — `_disco1_work/arc1057_winsorize_fbr_tail.py`

Reuses arc 1056's EXACT honest-§5f machinery (CALLS canonical; experiment side = winsorization + bootstrap
arithmetic only, never realizes P&L, never touches the gate):
- **Score** each leg's 18-cfg registry grid (6 exits × SL{1.5,2.0,2.5}, `trail_enabled=False`; me_long with
  the 2-bar time exit on the signal) over the 10 IS folds via `ArcFoldRunner`→`A1`→`MultiPairBacktester`,
  FundedNext, risk 0.005; capture **per-cell per-fold per-POSITION NET P&L** (gross − `apply_cost_model`
  cost, the canonical chokepoint; a partial+runner is ONE position).
- **Honest series** via BUILT `nested_exit_selection` (`afp_then_mean`): per fold-year pick the cfg best over
  STRICTLY-EARLIER folds → 8 evaluable no-lookahead folds 2013–2020. **Reproduces arc 1056 EXACTLY** —
  me_long **+0.203%** (5/8, all-folds `sl_only`/SL2.0), fbr **+1.001%** (5/8, `tp_3r`/`trailing_atr`/
  `partial_runner` SL1.5 selected per fold), IS-frozen RP weights **me_long 0.740 / fbr 0.260**, baseline RP
  book **+0.396%/yr** cluster **t=+2.29** (≡ 1056's 2.27), equal **+0.586%/yr t=+1.92** (≡ 1056's 1.91).
- **(2) Tail concentration:** pool each leg's positions (under its selected exits) over the 8 folds; report
  the top-1/3/5/1% share of the leg's total net P&L.
- **(3) Winsorize** both legs at percentile q∈{99,97.5,95,90} of each leg's own pooled positive net P&L;
  recompute per-fold ROI → recombine at the SAME IS-frozen weights → cluster bootstrap.
- **(5) Per-leg attribution:** winsorize ONLY fbr vs ONLY me_long at q95.
- **(4) Leave-top-N:** remove the N largest book positions (across both legs) for N∈{1,2,3,5}.

## 4. Results

**(2) Tail concentration — the whole book edge is ~5 trades:**

| leg | n_pos | Σnet | max | top-1 | top-3 | top-5 |
|---|---|---|---|---|---|---|
| me_long | 75 | $1,516 | $580 | 38.3% | 111.7% | 171.7% |
| **fbr** | 170 | $7,991 | $2,513 | 31.4% | **83.2%** | **127.0%** |

fbr's top-5 positions exceed its **entire** net P&L (127%) → the bottom 165 fbr positions are collectively
net-negative; the leg's edge IS its handful of convex runners.

**(3)/(4)/(5) Survival of the mean & significance (cluster bootstrap = correct SE-of-mean):**

| transform | RP mean %/yr | RP t | RP verdict | equal mean %/yr | equal t |
|---|---|---|---|---|---|
| **BASELINE** (1056 anchor) | **+0.396** | **+2.29** | **SIG+** | +0.586 | +1.92 |
| winsorize q99 | +0.386 | +2.20 | SIG+ | +0.568 | +1.88 |
| winsorize q97.5 | +0.377 | +2.12 | SIG+ | +0.551 | +1.85 |
| **winsorize q95** | +0.341 | **+1.89** | **~0** | +0.484 | +1.64 |
| winsorize q90 | +0.282 | +1.61 | ~0 | +0.405 | +1.40 |
| winsorize ONLY me_long q95 | +0.393 | **+2.28** | SIG+ | +0.584 | +1.91 |
| winsorize ONLY fbr q95 | +0.344 | **+1.90** | ~0 | +0.486 | +1.65 |
| **drop top-1** (fbr 2015 $2,513) | +0.315 | **+1.52** | ~0 | +0.430 | +1.42 |
| drop top-2 | +0.247 | +1.11 | ~0 | +0.300 | +0.97 |
| drop top-3 | +0.180 | +0.86 | ~0 | +0.172 | +0.54 |
| **drop top-5** | **+0.067** | +0.31 | ~0 | **−0.045** | −0.15 |

The five largest book positions are all fbr runners: 2015 $2,513, 2016 $2,112, 2019 $2,024, 2016 $1,864,
2020 $1,635.

## 5. Verdict + what it means

**DIAGNOSTIC → KILL** (no new component). Arc 1056's owed tail-dependence thread is resolved decisively:

1. **The deploy mean is tail-FRAGILE, not merely borderline.** The borderline SIG+ (RP t=2.29) survives only
   a *light* trim (q99/q97.5); a **standard q95 winsorization** or removal of the **single** largest trade
   takes it below significance, and removing the top-5 of 245 positions zeroes the mean (RP +0.07% / equal
   −0.05%). The within>across-year-sd tell arc 1056 flagged is now quantified: the per-year consistency is a
   tight realized clustering carried by ~5 un-repeatable fbr runners.
2. **It IS the fbr runner tail** (per-leg attribution decisive): winsorizing only fbr kills the t (2.29→1.90),
   winsorizing only me_long does not (2.28). fbr's top-5 = 127% of its net P&L.
3. **Not a mechanism KILL — a robustness/certifiability KILL of the MEAN.** A convex stop-run-reversal edge
   (many −1R stops + a few big reclaim-and-run winners) is *expected* to be tail-carried; that is fbr's
   genuine shape, independently verified honest (2034/2036/1037/1038). But at n=170 fbr positions over a
   decade with ~5 fat tails, the sample **cannot distinguish a real convex edge from tail luck** — the mean
   estimate is statistically un-certifiable and un-sizeable from this data, even though the mechanism is real.

**Net for the operator's path-A call.** The "significant mean-positive" pillar is materially downgraded: the
honest deploy mean is positive in expectation but **fat-tail-fragile** (one fbr-2015-runner deep), so it
should not be sized against as a robust expectancy. This is a **third, more fundamental limit** distinct from
the two already mapped — the AFP-gate wall (1032: holes persist through any meaningful block granularity) and
the vehicle/Calmar wall (2033/2059: Calmar 0.24–0.41, multi-year underwater, prop-firm-infeasible). All three
say the same thing from different angles: the corpus's edge is real but too thin/tail-carried to certify or
deploy. Components UNCHANGED (all 4 PORTFOLIO); deployable-system count = 0.

## 6. Threads / lessons

- **NEW reusable lesson (statistical/deployment):** for a thin reversion book whose per-year edge is carried
  by a few fat-tailed **convex-runner** trades, the deploy mean is not just borderline-significant but
  **tail-FRAGILE** — winsorizing the positive tail at q95, or removing the single largest position, erases the
  cluster-bootstrap significance, and removing the top-5 (of ~245) zeroes the mean. The arc-1056 "within-year
  sd > across-year sd" tell is the *signature* of this; this arc is its direct quantification. A convex edge
  being tail-carried is mechanistically honest, but at small n it makes the MEAN un-certifiable and
  un-sizeable — a robustness limit on the estimate, not the mechanism. Winsorize + leave-top-N are the
  standard diagnostics; run them before leaning on any thin book's mean for a deploy decision.
- **Sharpens arc 1056** (borderline → tail-fragile) and **arc 2059** (IS-side analog of its "doubly
  fat-tail-fragile" OOS finding: 2015/2016/2019/2020 fbr runners IS ≈ 2024 fbr / 2025 me_long OOS).
- **Reinforces the frontier state:** decision-support on the characterized object; no new edge ground. The
  three deployment walls (gate 1032 / vehicle 2033·2059 / mean-robustness this arc) are now all mapped — the
  autonomous edge-hunt has nothing further to add; the lever is the operator's path-A governance call.
- **Open (for the operator, not an autonomous chat):** whether a fundamentally THICKER edge class (more
  trades, less tail-carried) exists is the only thing that would change this — but the corpus has shown the
  surviving FX edges are intrinsically rare forced-flow/structural reversions (1025/2024/2026: thickening
  destroys the edge), so this is closed ground for the autonomous search.

**Tooling.** No new BUILT tool (reused `nested_exit_selection` 2040 + `combine_fold_roi` 2006 + the canonical
`apply_cost_model` / `build_fold_stats_from_run` reads — cf. arc 2016/1056 precedent: a measurement on
canonical outputs). Driver `discovery/_disco1_work/arc1057_winsorize_fbr_tail.py`. No canonical change, no
FLAG-as-code, no council, OOS untouched.

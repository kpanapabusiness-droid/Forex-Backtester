# arc 1056 — honest-§5f PER-TRADE bootstrap of the 2-leg deploy book (resolves arc-1043's owed thread)

**Chat:** 1000s · **Range:** 1000–1999 · **Disposition:** **DIAGNOSTIC → KILL** (no new component; book
UNCHANGED, all 4 PORTFOLIO) · **Council:** none (a measurement resolving a measurement, per the
1023/1043/2016 precedent) · **OOS:** NEVER touched (book fails IS AFP; §5g firewall).

> **Headline.** The per-trade bootstrap arc 1043 explicitly flagged as owed does **NOT robustly rescue
> the deploy book's significance.** Done on the RIGHT object (the honest **2-leg me_long+fbr** book — arc
> 2044 FLAG F1, not the 4-leg arc 1043 tested) and with the RIGHT estimator (the per-year **cluster**
> bootstrap is the SE-of-mean; the within-fold per-trade resample is **not**, and a two-level resample
> **double-counts**), the honest deploy mean **+0.41%/yr** is at best **BORDERLINE** significant: cluster
> bootstrap RP **t=2.27 (SIG+)** but equal-weight **t=1.91 (non-sig at 5%)** — the only SIG+ reading leans
> on IS-fit RP weights. The per-trade dimension's real contribution is a **fragility tell**: within-year
> sampling sd (**1.08%**) EXCEEDS the across-year spread (**0.55%**), so the apparent significance rests on
> a lucky tight clustering of **fat-tail-(fbr-runner)-driven** annual ROIs, not a thick consistent edge.
> Materially weaker than the committed book's advertised t=2.66; the deploy decision stays the operator's
> path-A gate-governance call. Deployable-system count = 0.

---

## 1. Read + synthesize (step a)

Pulled main (clean; highest 1000s arc = 1055, highest 2000s = 2054). No `discovery/STOP`. Read protocol,
TOOL_REGISTRY (CANONICAL + all BUILT rows), LESSONS (Closed-ground prior), DISCOVERY_DIRECTION (the
explore-now MENU), NEEDS_ENABLEMENT, and the recent ledger (1042–1055, 2040–2054).

**State.** The OHLC-only EDGE frontier is exhaustively mined out (chat-1000s handoff after 1054/1055; the
explore-now MENU M1/L1/O1/Q1/G1 all closed). Four PORTFOLIO components (gap 1006, me_long 1011, fbr 1013,
me_short 1019); the combined book FAILS all-folds-positive (co-sim item E: FUNDAMENTAL, not a combiner
artifact). The §11 verification programme is COMPLETE (signal + outcome + cost audits all honest). The
sole remaining lever is the operator's **path-A gate-governance call** — and the explicit guidance is that
the only remaining within-charter value is **decision-support diagnostics on the characterized object**,
NOT new edge obs-kills.

**The open, explicitly-OWED diagnostic thread.** Arc 1043 killed the path-A "significant mean-positive"
pillar (committed t=2.66 → honest §5f fold-bootstrap t≈1.3, every CI spans zero) but flagged: *"n=10 low
power, a per-TRADE bootstrap (arc-2016 method) MIGHT tighten the honest CI."* Arc 2016 ran a per-trade
bootstrap, but on the **committed** book; the **honest-§5f** per-trade bootstrap was never done. This is
the highest-value remaining within-charter diagnostic — it bears directly on the single live lever and has
zero fabrication surface (significance arithmetic on canonically-scored trades).

## 2. Idea (step b)

Resolve arc 1043's owed thread: does propagating within-fold (per-trade) sampling tighten the honest
deploy book's CI to significance? Object = the honest **2-leg me_long+fbr** book (arc 2044 FLAG F1: under
honest §5f, gap & me_short flip mean-NEGATIVE, so the honest book's best expression is the 2-leg — the
4-leg arc 1043 tested is the wrong, drag-loaded object). No council (a measurement resolving a
measurement).

## 3. Method (steps c–g) — `_disco1_work/arc1056_honest_pertrade_bootstrap.py`

Reuses arc 2044's EXACT honest-§5f machinery (CALLS canonical; experiment side = bootstrap arithmetic
only — no new BUILT tool, cf. arc 2016):
- **Score** each leg's 18-cfg registry grid (6 exits × SL{1.5,2.0,2.5}, `trail_enabled=False`; me_long
  with the 2-bar time exit on the signal) over the 10 IS folds via `ArcFoldRunner`→`A1`→
  `MultiPairBacktester`, FundedNext, risk 0.005.
- **Honest series** via BUILT `nested_exit_selection` (`afp_then_mean`, gate-aligned): per fold-year pick
  the cfg best over STRICTLY-EARLIER folds. Warmup folds 2011/2012 (<2 priors) excluded → **8 evaluable
  no-lookahead folds 2013–2020**. Reproduces arc 2044: me_long **+0.203%**, fbr **+1.001%**.
- **Bootstrap unit = per-POSITION net P&L.** Two Arc-10-relevant fidelity points the build pinned down
  (the first cut got both wrong; the reproduction anchor caught it):
  1. `ClosedTrade.pnl` is **GROSS**; the gate scores NET. The unit is per-position
     `net_pnl = gross − total_cost` from the canonical chokepoint `apply_cost_model(...).breakdown` (a
     partial+runner is ONE position; legs are not independent → resample positions, not legs).
  2. `build_fold_stats_from_run`'s `roi_pct` = the **net-equity curve sliced to the OOS window**,
     end/start − 1 — i.e. the equity GAIN **realized (position EXIT) inside the OOS year** over the
     OOS-start equity. Filtering positions by `final_exit_time ∈ OOS` and dividing by the OOS-start
     equity **reproduces `fs.roi_pct` to ≤0.02pp** for fast exits (max residual **0.89pp** on three
     slow `tp_3r`/low-trade cells where a position open at the year boundary carries unrealized MtM the
     uncertainty closed-trade bootstrap cannot see — disclosed; the point means use the gate-exact nested
     series, so the bootstrap-center bias is ~0.05–0.08pp at the book level and the verdict is invariant).
- **Three estimators**, IS-frozen RP + equal weights, N=10000, seed 42:
  - **(A) Cluster (fold-level) bootstrap** [= arc-1043 method] — resample the 8 yearly book ROIs. This is
    the **textbook-correct SE-of-mean** for per-year-clustered data (each year's realized ROI already
    embeds its own trade-sampling noise; the variance algebra gives cluster-Var = σ²_b/8 + σ̄²_ε/8 = the
    true Var(θ̂)).
  - **(B) Within-fold per-trade bootstrap** — resample each fold's positions (under its selected exit),
    hold fold membership. This is NOT the SE of the mean; it measures **per-year estimation uncertainty**.
  - **(C) Two-level (cluster + trade)** — resample years AND trades within. Reported only as an
    **over-conservative bound**: the algebra shows it **double-counts** σ̄²_ε (two-level-Var = true +
    σ̄²_ε/8), so it overstates uncertainty; not the verdict estimator.

## 4. Results

Honest 2-leg book, IS-frozen RP weights **(me_long 0.740, fbr 0.260)**, point mean **+0.4110%/yr**, **6/8**
positive, worst **−0.211% (2018)**, across-fold sd **0.5481%**:

| estimator | 95% CI | P(mean<0) | implied t | verdict |
|---|---|---|---|---|
| **(A) cluster / fold-level** (correct SE) | [+0.068%, +0.772%] | 0.008 | **+2.27** | **SIG+** |
| (B) within-fold per-trade (per-year uncertainty) | [−0.337%, +1.167%] | 0.156 | +1.07 | ~0 |
| (C) two-level (double-counts; upper bound) | [−0.406%, +1.261%] | 0.172 | +0.97 | ~0 |

Equal weights (no IS fit), point **+0.6021%/yr**, 6/8, worst −1.235% (2018), across-fold sd 0.9656%:
cluster t=**+1.91** (CI [−0.042%, +1.176%], P=0.035) → **~0 / non-sig at 5%**; within-fold t=+0.94;
two-level t=+0.84.

**GUARD (the per-trade dimension's real finding):** within-fold sampling sd (RMS over folds) = **1.08%**
(RP) / **1.81%** (equal) **EXCEEDS** the across-fold sd **0.55%** / **0.97%** → ratio across/within ≈
**0.5**. The realized year-to-year spread is HALF the within-year (fat-tail-driven) sampling noise.

## 5. Verdict + what it means

**DIAGNOSTIC → KILL.** Arc 1043's owed per-trade-bootstrap thread is RESOLVED with a nuanced, honest
answer:

1. **It does not cleanly tighten to significance.** The statistically-correct estimator (cluster
   bootstrap) on the honest 2-leg deploy object gives RP **t=2.27 (SIG+)** but equal-weight **t=1.91
   (non-sig at 5%)** → **BORDERLINE**, and the only SIG+ reading leans on **IS-fit RP weights** (an
   in-sample tightening that down-weights fbr's fat-tail leg to 0.26). The per-trade *within-fold*
   resample is NOT the SE of the mean (it's per-year estimation uncertainty), and the two-level resample
   double-counts — so the higher "power" of per-trade resampling that arc 1043 hoped for is **illusory
   for the SE of the mean**.
2. **It DID correct the object.** The honest **2-leg** deploy book (me_long+fbr) is materially stronger
   than the **4-leg** arc 1043 found non-significant — confirming arc 2044 PART A from the significance
   angle: dropping the honest-negative drag legs (gap, me_short) restores most of the lost t. So the
   deploy case is **not as dead as the 4-leg number implied** — but it is **borderline**, not the robust
   t=2.66 the committed book advertised.
3. **The per-trade dimension's genuine contribution is a FRAGILITY tell, not power.** within-year sampling
   sd > across-year spread (ratio ~0.5) means the per-year ROIs are dominated by fat-tail (fbr-runner)
   sampling noise; the borderline significance rests on a **lucky tight clustering** of those noisy annual
   numbers rather than a thick consistent edge. Re-drawing the trades, individual years swing more than
   the across-year spread.

**Net for the operator's path-A call:** the honest deploy object's mean is positive (+0.41%/yr) and
**borderline-but-not-robustly** significant (RP-weighting-dependent, fat-tail-fragile) — stronger than
arc 1043's 4-leg verdict, weaker than the committed advertisement. Components UNCHANGED (all 4 PORTFOLIO);
deployable-system count = 0. The gate-resolution question remains the operator's governance call.

## 6. Threads / lessons

- **NEW reusable lesson (statistical):** for a thin book whose per-year ROIs are carried by **a few
  fat-tailed trades**, the cluster (fold-level) bootstrap is the correct SE-of-mean, but its significance
  is **fragile when within-year sampling sd > across-year spread** — a tell that the per-year consistency
  is a tight realized clustering of noisy annual numbers, not a thick edge. The within-fold per-trade
  bootstrap is the diagnostic for this fragility; it is NOT a higher-power SE-of-mean (and a two-level
  cluster+trade resample double-counts within-year variance — do not read it as the SE). This refines arc
  1023's "across-fold sd ≈ within-fold sampling sd" (committed 4-leg) to "across < within" on the honest
  2-leg — i.e. the honest book is *even more* sampling-dominated.
- **Confirms arc 2044 PART A from the significance angle:** the honest 2-leg me_long+fbr is the
  deploy-relevant object and is materially stronger than the 4-leg; gap+me_short are honest drag.
- **Reinforces the frontier state:** this is decision-support on the characterized object, the only
  remaining within-charter value (chat-1000s handoff / DISCOVERY_DIRECTION). No new edge ground touched.

**Tooling.** No new BUILT tool (reused `nested_exit_selection` 2040 + `combine_fold_roi` 2006 + the
canonical `apply_cost_model` / `build_fold_stats_from_run` reads, cf. arc 2016's precedent). Driver
`discovery/_disco1_work/arc1056_honest_pertrade_bootstrap.py`. No canonical change, no FLAG-as-code, no
council, OOS untouched.

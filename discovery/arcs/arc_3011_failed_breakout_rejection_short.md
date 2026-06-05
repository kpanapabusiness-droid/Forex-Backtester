# arc 3011 — Failed-breakout REJECTION short (the short mirror of arc 1013)

**Chat:** 3000s · **Range:** 3000–3999 · **Date:** 2026-06-05 · **Verdict:** KILL
**Disposition:** KILL (no robust short edge — the pooled drift is a thin-tail/pair-mix artifact per the
concurrent independent arc 2011; and on the engine it nets ~0, 0/18 exit·SL cells all-folds-positive,
beats null only +0.021pp, NEGATIVE in 2018).

This is the **FIRST discovery short arc to reach the honest engine** (arcs 1014/2009/3010 all
cheap-killed at observation). The short path (PR #273) scored end-to-end correctly. **Independently
reproduced by arc 2011 (2000s, concurrent) — see Convergence; arc 2011's robustness analysis corrected my
initial "structure pass" over-claim, the Arc-10 independent-reproduction defense working as designed.**

---

## (a) Log read / synthesis (FRESH EYES, honest-era only)

Read DISCOVERY_PROTOCOL, full DISCOVERY_LOG (Tier-1 all 31 arcs + recent Tier-2), LESSONS, TOOL_REGISTRY.
State of the programme at resume (highest 3000s id = 3010 → I open 3011):

- **Directional prediction is comprehensively CLOSED, long AND short** — momentum/breakout/reversion/
  trend, H1/H4/D1, 28 pairs, capture AND drift lenses, every exit, stop removed (3004). Forward drift ≈
  cost everywhere. Regime conditioning anti-predictive **both** directions (3003 long, 3010 short).
- **Three long PORTFOLIO components exist**, all fade/reversion-flavored: gap-fill JPY-cross H4 (1006,
  +0.685%), month-end USD-major D1 (1011, +0.232%), failed-breakdown **reclaim** USD-major H4 (1013,
  +1.854% — the strongest, cleanest edge in the corpus; a structural stop-run reversal).
- **The portfolio route is blocked on ONE named fold** (2006/2008/3009): 2-way KILL (mutually-negative
  2015); 3-way KILL but converging — only **2015 and especially 2018** bleed (0/5151 convex weightings
  all-folds-positive; 2015-needs-fbr vs 2018-needs-month-end is a hard simplex infeasibility). The 4th
  component must be **net-positive on 2018** (a strong-USD trend year where all three long fades lose).
- **Structural shorts tested so far are dead:** 1014 (clean confirmed-breakdown short — swing-low sweep
  inert), 2009 (climax-sweep short — swing-low sweep ANTI-load-bearing). Both shorted a swept swing
  **LOW**, which 1013 proved is a *reversal-LONG* structure → no short mirror.
- **The open short frontier (dispatch):** flow-event asymmetries (up-gap weekend short = 1000s arc-1015
  lane; climax-sweep = 2009 dead) **and arc 1013's own flag: "the up-sweep is arguably the stronger
  leg."** That up-sweep — sweeping a swing **HIGH** + rejecting → SHORT — is the **untested clean mirror
  of the corpus's strongest edge**, and is unclaimed by any range. That is this arc.

## (b) Idea + because (mirror of the strongest corpus edge)

Arc 1013 proved that sweeping a swing **low** + reclaiming (failed breakdown) is the strongest directional
long: resting sell-stops below the level get run, the down-move is liquidity-driven not informational, and
the reclaim confirms the reversal *at entry*. By the **symmetric liquidity-grab mechanism**, sweeping a
swing **high** (running buy-stops above resistance) + failing to hold (close back below) → **short**. Arc
1013 itself flagged the up-sweep as "arguably the stronger leg." Now unblocked (PR #273).

**Why it could be the 2018 leg:** unlike the three long fades, this is a SHORT. In strong-USD 2018 the
falling XXXUSD majors (EUR/GBP/AUD/NZD) rally-then-fail repeatedly — a failed-upside-breakout short on
those is *trend-aligned*, so the construction is plausibly net-positive in exactly the year the portfolio
needs. (Tested below; the answer turned out NO — see per-year.)

## (c)/(d) Characterize + cheap-kill — OBSERVATION (decisive structure control)

`observe_long_capture(direction="short")` on 7 USD majors H4, IS 2010–2020 (drift>0 ⇔ price fell ⇔ good
for short). K=40 swing-high lookback, upper-shadow ≥1.25 ATR (mirror of 1013's defaults).

```
BASE (all bars, short)              : n=121432  cap 0.4849  drift -0.015 ATR   (coin-flip, per 1014/3010)
swept_high & failed (reject below)  : n=  5192  cap 0.4900  drift +0.096 ATR
CELL swept&failed&shadow>=1.25      : n=   241  cap 0.4730  drift +0.278 ATR  frac+ 0.494

STRUCTURE CONTROL (the 1014/2009 discriminator):
  big-reject AT swept swing-high    : n=   241  cap 0.4730  drift +0.278 ATR   <- structure HELPS the short
  big-reject ELSEWHERE (not at high): n=   570  cap 0.5053  drift -0.171 ATR   <- no structure: drift REVERSES
```

**Structure control APPEARS to pass on the pooled mean** (+0.278 AT vs −0.171 elsewhere) — but this is a
**thin-tail / pair-mix CONFOUND, not a robust edge** (correction credited to the concurrent independent
arc 2011, 2000s — see Convergence below; I initially over-claimed it as a genuine structure pass). My own
per-pair data exposes it: the +0.278 is carried by **2 thin outlier pairs** — AUDUSD (n=29, +1.58) and
USDJPY (n=31, +1.49); the other five average ≈ −0.04 (EURUSD +0.63, GBPUSD +0.20, NZDUSD +0.37, USDCAD
−0.47, USDCHF −0.95). Arc 2011 quantified the same: pooled mean +0.261 but **median −0.069**, and dropping
those two pairs takes the cell **negative** (−0.133). So the "structure pass" is the arc-2009 USD-quote
pair-mix beta tell, NOT a load-bearing structure. Shadow "monotonicity" (deep tail shadow≥2 cap 0.606 /
drift +0.65, n=33 ≈3/yr) is the same thin-tail artifact.

**BUT — the critical asymmetry vs 1013:** the +1R-before-SL **capture is 0.473 < 0.50** (and below the
0.485 base), where 1013's long captured **0.55–0.61**. The favorable drift is **tail-driven** (frac+
0.494 = coin-flip on direction; the +0.278 mean is a handful of big winners), not a reliable per-trade
edge. A non-coin-flip *drift* signal with sub-0.50 *capture* → §5f mandates the honest engine + exit menu
before any FAIL (do NOT cheap-kill).

**Per-year (the portfolio question):** 2015 drift **+1.09** (positive) but **2018 drift −0.131
(NEGATIVE)**. Per-pair: positive 5/7 (EUR/GBP/AUD/NZD/JPY) but USDCHF −0.95, USDCAD −0.47 (the USD-up
legs fight the 2018 uptrend) drag the basket. → it does **not** provide the 2018-positive leg.

## (e)/(f)/(g) Honest engine — §5f exit/SL menu (full IS, all-folds-positive gate)

Standard entry point: `build_arc_pool` (n=241, pool_sha ccb8ef27, pool capture 0.4730 ✓ matches obs,
**mean final_r +0.0102R ≈ 0**) → `ArcFoldRunner` (A1, direction=SHORT) → `run_config_over_folds` over the
10 IS folds → `judge_all_folds_positive`. FundedNext ON, SL-first, `trail_enabled=False`.

§5f nested hyperparameter — full registered exit menu × {1.5, 2.0, 3.0}·ATR SL, scored on the IS folds:

```
exit                              SL    mean      worst    pos
sl_plus_tp_3r                    2.0   +0.00%   -0.04%    7/10   <- most folds positive...
sl_plus_trailing_atr             2.0   +0.01%   -0.04%    6/10
sl_plus_trailing_atr             1.5   +0.00%   -0.03%    5/10
sl_partial_close_1r_runner_trail 2.0   -0.00%   -0.03%    4/10
sl_only                          2.0   -0.04%   -0.11%    3/10
... (all 18 cells: mean in [-0.04%, +0.01%], 0 all-folds-positive, best worst-fold -0.028%)
```

**NONE of the 18 exit/SL cells is all-folds-positive.** The structure-confirmed +0.278 ATR gross drift
collapses to mean ≈ 0% net: capture 0.473 means the 2·ATR stop fires *before* the favorable drift plays
out more than half the time, and FundedNext costs eat the thin remainder. IS not all-folds-positive →
**OOS deliberately NOT touched** (§4 holdout preservation).

## Null baseline (KILL vs PORTFOLIO, §11) — the entry's structure edge IS real

Fair same-exit random-entry null (BUILT `build_null_signal_evaluation`, matched fire-count, 3 seeds),
exit = `sl_plus_trailing_atr` SL2.0:

```
REAL short        : mean +0.005%  worst -0.037%  6/10 pos
NULL mean-of-means: mean -0.015%                  (seeds 42/7/123: -0.016/-0.006/-0.025)
REAL - NULL       : +0.021 pp   <- noise-floor margin, consistent with a thin-tail artifact (see below)
```

The real signal beats random by only **+0.021pp** — initially read as "the structure genuinely helps,"
but in light of arc 2011's robustness finding (the pooled drift is a 2-pair thin-tail artifact; median
negative) this margin is at the noise floor, NOT evidence of a robust edge. **+0.005% mean is config-
fragile** (most other exits marginally negative; ~24 trades/yr), two orders of magnitude below the
PORTFOLIO archetype (arc 1006 +0.69%, 1011 +0.23%), and **negative in its one intended fold (2018)**.
Beating the null is necessary, not sufficient (§11) → **KILL, not PORTFOLIO.**

## Verdict: KILL

**No robust short edge.** Two complementary, independently-reached reads converge on KILL: (1) the pooled
"structure pass" is a thin-tail / pair-mix artifact (median −0.069; drops negative excluding 2 thin pairs
— arc 2011 + my per-pair data); (2) even taking the pooled +0.278 ATR drift at face value, capture is
sub-0.50 so under SL-first + FundedNext it nets ~0 (mean final_r +0.0102R), is all-folds-positive on 0 of
18 exit/SL cells, beats the fair null by only +0.021pp (noise floor), and is negative in 2018. Either way
it is the canonical corpus signature ("looks like an edge on the mean, isn't one net") — and the
robustness read explains exactly why the engine net came out at ~0 and the null margin was negligible: the
"edge" was never robust.

## Diagnosis — the swing-low reclaim long has NO short mirror (3 constructions)

Across all three short constructions of the swing-sweep, arc 1013's reclaim-long has no tradeable mirror:
1014 (clean confirmed-breakdown — swing-low sweep inert), 2009 (climax — swing-low sweep anti-load-bearing,
shorts into the bounce), and 2011/3011 (swing-HIGH reject-reversal — drift not robust, capture coin-flip,
nets sub-cost). The unifying read: **the +1R capture is the wall.** 1013's long captures 0.55–0.61; every
short construction captures ~0.47–0.51 (coin-flip). A *candidate* mechanism for the capture asymmetry —
downside stop-runs (fear) snap back fast and cleanly (high capture) while upside stop-runs (greed) drift
slowly and re-test the high before paying (drift-but-low-capture) — is consistent with the data but
secondary to the decisive finding: there is simply no robust, capturable short here. The swing-low sweep is
a reversal-LONG structure with no symmetric short.

## Threads / lessons

1. **arc 1013's reclaim-long has NO tradeable short mirror — now confirmed across ALL THREE short
   constructions** (1014 clean-breakdown, 2009 climax, 2011/3011 reject-reversal). The swing-low sweep is an
   asymmetric reversal-LONG structure; the structural-short route to a 2018 leg is fully closed.
2. **Check the MEDIAN and pair-mix before believing a pooled structure-control "pass."** My pooled mean
   structure control (+0.278 AT vs −0.171 elsewhere) looked like the first real structural short, but it was
   2 thin outlier pairs (AUDUSD/USDJPY); median −0.069, negative excluding them (arc 2011 caught this; my
   per-pair data confirms). This is the **arc-2009 USD-quote pair-mix beta tell** recurring — demand a
   median + leave-2-pairs-out check on any short structure control, not just the pooled mean. (The most
   valuable methodology carry-forward here, and a self-correction credited to the independent reproduction.)
3. **Capture, not drift, is the wall for reversal shorts.** Every swing-sweep short captures ~0.47–0.51
   (coin-flip) vs 1013's long 0.55–0.61; even a pooled-positive drift nets ~0 under a 2·ATR SL-first stop
   because capture is sub-0.50 — the §5f 18-cell menu cannot lift it (extends arc 0/2000). The
   "buy-fear-reverts-fast / sell-greed-drifts-slow" mechanism is a plausible-but-secondary candidate for the
   capture asymmetry; the decisive fact is no robust capturable short exists.
4. **2018 is still the portfolio wall** — even a (would-be) trend-aligned short fade is negative in 2018
   (USDCHF/USDCAD legs fight the USD uptrend); reinforces 3009/3010 that the 2018-positive leg is NOT a
   directional construction of ANY kind (long, short, structural, or trend) → the arc-3004 escalation /
   relative-value-second-leg route stands.
5. **First discovery short reached the engine cleanly** — `direction=SHORT` flows end-to-end (signal state
   → A1 `directions` map → Order → MultiPairBacktester); pool capture reproduced the observation (0.4730),
   validating the merged short path (PR #273) on the discovery apparatus. My §5f engine pass + arc 2011's
   observation-robustness pass are complementary independent confirmations of the same KILL.

## Convergence with arc 2011 (chat 2000s — concurrent independent reproduction)

The 2000s chat ran the IDENTICAL idea (failed-breakout reject short at a swing high, the 1013 mirror) as
arc 2011 and pushed it to main while I was mid-arc (I read the log before it landed — true independent
reproduction, the Arc-10 defense, cf. 2008/3009). Same verdict (KILL), complementary methods:
- **Arc 2011 (observation + robustness):** caught the pooled structure-control "pass" is a thin-tail /
  pair-mix confound (mean +0.261 ≫ median −0.069; → −0.133 excluding AUDUSD/USDJPY; deeper cell inverts
  negative). Cheap-killed at observation on robustness grounds.
- **Arc 3011 (this arc — full §5f engine):** took the entry all the way to the honest engine (first short
  to do so) and showed that even granting the pooled drift it nets ~0 (mean final_r +0.0102R), 0/18
  exit·SL cells all-folds-positive, beats null only +0.021pp — the engine-side fingerprint of "never robust."
Together: arc 1013's reclaim-long has no short mirror in any of three constructions; the structural-short
route to the 2018 portfolio leg is closed both at observation (2011 robustness) and on the engine (3011 net).

## Tooling

**BUILT (new):** `FailedBreakoutRejectionShortSignal` → `discovery/tools/failed_breakout_signals.py` (the
short mirror of `FailedBreakdownReclaimLongSignal`; mask + ATR geometry only, `direction=Direction.SHORT`).
Registered at step (i). Reused: `observe_long_capture(direction="short")`, `build_null_signal_evaluation`,
and the canonical entry point (CALLED, never re-rolled).

## FLAGS (code not merged)

None requiring the canonical core. No core change (the short path was already merged, PR #273); this arc
only added an EXPERIMENT signal under `discovery/tools/`. Carries the standing FLAG-1 (the 2018-positive
4th portfolio leg is NOT a directional construction — narrowed further: not even a structure-confirmed
short fade) and the `A1Config.time_exit_bars`-unwired flag. Drivers: `_disco3_work/arc3011_*.py`
(reproducible: `PYTHONPATH=. py discovery/_disco3_work/arc3011_engine.py`,
`histdata_root=C:\Users\panap\histdata_backup`).

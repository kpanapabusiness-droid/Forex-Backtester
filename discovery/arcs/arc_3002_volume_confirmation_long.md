# Arc 3002 — Volume-Confirmation Long (the last untouched data column)

> **Arc id:** 3002 · **Chat:** 3000–3999 (continuous) · **Date:** 2026-06-04
> **Final verdict:** **FAIL (full IS WFO, 7/10 folds negative, mean −3.91%)** — volume-confirmation does
> not rescue the directional-long base. The 3-fold triage looked positive (+2.38% mean) but that was a
> lucky-year artifact (it sampled 2013 & 2016, the two most momentum-friendly years); the full IS is 7/10
> negative.
> **Idea-family:** the VOLUME axis — `volume` (HistData tick-count, an activity proxy) is the **one data
> column no arc (0,1000–1005,2000,3000,3001) has ever used.** Does volume-confirmation concentrate a
> cost-clearing forward drift where price/vol/structure (arc 3001) could not?

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). Engine/measurement **called, never
re-rolled**; the scan + signal are experiment tools.

---

## (a) Log read + synthesis — FRESH EYES

Pulled `origin/main`. Honest-era corpus = 10 arcs across 3 chats (0; 1000–1005; 2000; 3000–3001), **all
FAIL**. The consolidated picture: directional price-structure long is **closed under BOTH metrics** (capture,
arcs 0/1000/1001/3000; drift, arc 3001), across timeframes (H4/D1), the full 28-pair universe, and exit
engineering (1004). Calendar/flow turn-of-month is also sub-cost (1005). Trend convexity-harvest fails (2000).
The binding constraint is **EDGE < COST**, universal.

**What's left untested:** every arc conditioned on price / volatility / structure / timeframe / calendar.
**None used `volume`** — yet the panel carries a `volume` column (tick-count per bar, a real activity proxy).
Volume-confirmation of directional moves (price moving on high volume = informed = continuation; low-volume
moves = noise) is a classic, mechanistically-grounded lever, and it is the **last untouched data axis**.
(Pre-reset eliminated a "C7 volume gate" — but that's pre-reset, fresh-eyes re-explorable, and was a gate on
another signal, not a volume-as-primary-edge scan.) Distinct from the 1000s/2000s chats' current work.

## (b) Observation → idea (volume on the drift lens)

Re-used arc 3001's forward-drift lens (`fwd_drift_12 = (mid_close[s+12] − entry@s+1 ask)/(2·ATR[s])`, gross R)
with **volume conditionings** (causal `vol_ratio = volume[s] / SMA(volume,20)[s]`), all 28 pairs, IS 2010–2020.
Cost hurdle ≈ +0.05 to +0.10R.

**Group-level drift (R, gross):**

| group | hivol_up | hivol_brk | vol_spike(>2.5×) | lovol_pull | hivol_all | lovol_all |
|---|---|---|---|---|---|---|
| MAJOR | −0.028 | −0.033 | −0.056 | +0.002 | −0.032 | −0.019 |
| COUPLED | −0.076 | −0.109 | −0.042 | −0.041 | −0.063 | −0.107 |
| TREND_X | −0.017 | −0.014 | **+0.0475** | +0.001 | −0.016 | −0.019 |

**No group cell clears cost.** The single best group cell is TREND_X vol-spike **+0.0475R** (n=710, thin, at
the hurdle's lower edge). Per-pair flags (drift > +0.05R) are **scattered single-pair**: GBPJPY lovol_pull
+0.092 (n=1120), CHFJPY hivol_brk +0.088, EURNZD hivol_brk +0.065, GBPJPY hivol_brk +0.057 — GBPJPY-driven,
not group-consistent. A single-pair drift is regime-luck (the arc-1000 lesson), not a generalizable edge, so I
did NOT build a single-pair system. The only arguably-generalizable candidate is TREND_X vol-spike.

**Idea:** long the bar after a > 2.5× volume-spike on the 12 trending crosses, refractory 6, SL=2·ATR — the
best cross-pair volume cell. Mechanism (to test): a volume spike marks an information event that price
continues.

## (c) Characterize + (d) cheap kill (triage — NOT deeply negative → proceeded)

`build_arc_pool`, 12 trending crosses, H4 5ers_eet, IS 2010–2020, SL=2·ATR, hold 120.
`pool_sha256 dd2cb8d882b587d9…`. **839 IS trades**, capture **0.4779**, **gross mean final_r +0.0035** (≈ zero
gross edge). Pool floor PASS.

**3-fold triage** (A1, SL=2·ATR, 1% reset-floor, exposure 1/pair 2/ccy, `sl_partial_close_1r_runner_trail`,
FundedNext costs ON, SL-first; OOS 2013/2016/2019): 2013 **+11.43%** / 2016 **+11.97%** / 2019 **−16.25%** →
worst −16.25%, **mean +2.38%, 2/3 positive**. **NOT deeply negative → did NOT cheap-kill → proceeded** (per
§5d). This is the first non-deeply-negative triage in the 3000s range.

## (e)+(g) Diagnose + validate — full IS WFO unmasks the triage as lucky-year sampling

Before investing in a heavy diagnosis/council, ran the **full IS WFO** (all 10 folds, measurement not
optimization):

| OOS year | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 |
|---|---|---|---|---|---|---|---|---|---|---|
| ROI | −13.62 | −0.84 | **+11.43** | −11.50 | −3.61 | **+11.97** | −9.92 | −12.92 | −16.25 | **+6.13** |

**worst −16.25%, mean −3.91%, median −6.76%, 7/10 NEGATIVE → NOT all-folds-positive → FAIL.** The positive
folds are exactly the momentum-friendly years (2013, 2016, 2020); the 3-fold triage's +2.38% mean was an
artifact of having sampled 2013 **and** 2016. With near-zero gross edge (+0.0035R), the ROI swings are
exit/sizing-driven regime variance, not an edge.

**Diagnosis (clean, and unifying):** **volume predicts move MAGNITUDE, not DIRECTION.** A volume spike marks
an information event / activity surge → a *bigger* move is coming, but its *direction* is still the ~0.49
coin-flip. A long-only volume-spike entry therefore has higher variance (bigger wins AND losses) on the same
sub-0.50 directional base → regime-dependent, net sub-cost. This is **exactly arc 1001's finding** (volatility
contraction predicts expansion, not direction) re-derived on the volume axis: both volume and volatility are
MAGNITUDE predictors; DIRECTION is the binding constraint, and it remains a coin-flip.

No worthwhile/reachable ceiling (7/10 neg, ≈nil gross, regime-luck = the closed arc-0/1000 selection lever),
so the heavy diagnosis council (§5e, for ideas with reachable upside) is not triggered. OOS not measured (IS
failed all-folds-positive, per §5g).

## (h) Survivor stress-test — NOT REACHED

No candidate passed IS all-folds-positive. No `passed/` record.

## Final verdict — FAIL

Volume-confirmation does not rescue the directional-long base. The volume axis — the last untouched data
column — is closed: no group-level volume conditioning shows cost-clearing drift, and the best cross-pair
candidate (vol-spike on trending crosses) is 7/10 negative on full IS (regime-luck, near-zero gross). Volume
predicts magnitude, not direction.

## Lessons (candidate for LESSONS.md compression)

1. **Volume predicts move MAGNITUDE, not DIRECTION** — unifies with arc 1001 (volatility-contraction). Both
   volume and volatility tell you a *bigger* move is coming, not *which way*; a long-only bet on them is a
   higher-variance directional coin-flip → regime-dependent, net sub-cost. The binding constraint is direction,
   which stays ~0.49 regardless of the magnitude predictor. The **last untouched data column (volume) is now
   closed.**
2. **METHODOLOGICAL: the 3-fold triage (OOS 2013/2016/2019) over-samples momentum-friendly years.** 2013 and
   2016 are positive for almost every momentum-ish signal (cf. arc 1000's +11.98% 2013, arc 3001's +16.74%
   2013); a regime-dependent signal can show a *positive* triage mean (+2.38% here) yet be 7/10 negative on
   full IS. **A not-deeply-negative 3-fold triage is necessary but not sufficient — confirm with full IS
   before any diagnosis/council investment.** (Cheap mitigation: include a chop year like 2014/2017/2018 in
   the triage, or just run full IS once the triage is non-negative.)
3. **Single-pair drift flags are regime-luck, not edge** (GBPJPY +0.092R lovol_pull) — re-confirms the
   cross-pair-robustness discipline; do not build single-pair systems.

## Threads / what didn't help

- **Closed (this arc):** volume-confirmation as a directional-long edge (magnitude≠direction). The volume data
  axis is now exhausted alongside price/vol/structure/timeframe.
- **Surviving steer:** with the last price/volume/structure axis closed and calendar weakening (1005), the
  3000s side has now systematically closed the directional + data-conditioning space. **Arc 3003 is the right
  point for a LIGHT generative council** (§5b) — I am at a genuine idea-fork (no obvious concrete lever left)
  and should widen with the council before another lone guess, OR pivot to a meta-question (is the apparatus
  itself — long-only, single-instrument, these costs — capable of expressing any edge, and what would need to
  change?).
- **Premature, not closed:** portfolio/selection (no net-positive component exists yet).

## Flags (code NOT merged — human-gated, per protocol §9)

None requiring the canonical core. Scan + signal + drivers scratch `_disco3_work/`. **Methodological note (not
a code change):** the chat-convention 3-fold triage year-set {2013,2016,2019} is momentum-biased (lesson 2);
future arcs should confirm a non-negative triage against full IS. Not a canonical-core bug — the canonical
folds (`build_v3_folds`) are unaffected; this is about the cheap-kill triage sample choice.

## Reproduction

- **Data:** `histdata_root = C:\Users\panap\histdata_backup`, `cache_root = data/cache`,
  `boundary_convention = "5ers_eet"`, TF H4. **Scan:** all 28 pairs. **Triage/WFO pairs:** 12 trending crosses
  (EURJPY GBPJPY AUDJPY NZDJPY CADJPY CHFJPY EURAUD EURNZD EURCAD GBPAUD GBPNZD GBPCAD).
- **Drivers (scratch):** `observe4_volume.py` (b, volume drift scan), `arc3002_signal.py`
  (`VolumeSpikeLong(vol_mult=2.5, vol_win=20, refractory=6)`), `arc3002_triage.py` (3-fold),
  `arc3002_fullis.py` (full IS WFO). `pool_sha256 dd2cb8d882b587d9…`. Run `PYTHONPATH=. py _disco3_work/<script>.py`.
- **Engine:** `MultiPairBacktester` via `A1Architecture` + `ArcFoldRunner`; FundedNext costs at
  `build_fold_stats_from_run`; IS folds `build_v3_folds`; judge `judge_all_folds_positive`.

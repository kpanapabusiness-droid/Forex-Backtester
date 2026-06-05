# arc 1016 — up-gap weekend SHORT (the dispatch's #1 short lead; the gap-fill long's "stronger leg")

**Chat:** 1000s · **Date:** 2026-06-05 · **Disposition:** KILL (cheap-kill at observation) · **Council:**
none (falsified at observation; capture <0.50 + adverse/coin-flip short drift → no worthwhile-ceiling
fork; arc-1014/3010 discipline).

## Idea + why (log-seeded — the named 2018-leg candidate)

My arc 1015 (3-way portfolio combination) produced a razor-sharp 4th-component spec: a leg **positive in
BOTH 2015 & 2018** (the strong-USD/risk-off folds where every long flow-reversion + reclaim edge bleeds),
structurally a SHORT, NOT long-reversion. The dispatch's #1/#2 leads and arcs 1014/1015 all name the same
first target: the **up-gap weekend SHORT** — arcs 2001/2003 measured weekly-open UP gaps drifting **−0.57
ATR, frac+ 0.64**, i.e. up-gaps revert DOWN, "the gap-fill's STRONGER leg," structurally blocked by
long-only and now unblocked (PR #273). Unlike arc 1014's backward-confirming breakdown short, the up-gap
short is the **mirror of the arc-1006 gap-fill LONG** (PORTFOLIO) — a discrete weekend-illiquidity flow
event where the reversion might still be ahead of the i+1 entry. So it is the highest-EV short to test, and
its **2015 & 2018 sign is the acceptance test** for the portfolio route's missing leg.

## Method (CALLED canonical observation; §5b observe-first)

Direction-aware honest observation via the BUILT `observe_long_capture(direction="short")` (validated by
arc 1014) over H4, IS+OOS bars, restricted to weekly-open (Δt>36h) UP-gap bars (gap_atr ≥ +thr; mirrors
`WeekendGapFillLongSignal`'s ex-ante gap detection — open[i] & close[i-1] at bar i close, ATR shift1; the
honest i+1 short enters next bar `open_bid`, SL = close_bid + 2·ATR). For a SHORT: want **capture > 0.50
AND fwd_drift_atr > 0** (the tool flips the drift sign, so +drift = price FELL = good short). Universes: 5
JPY crosses (the gap-fill-long universe) + 7 USD majors (arc 2001's up-gap universe); thresholds 0.5/1.0/1.5
ATR. Driver scratch `_disco_work/arc1016_upgap_short_obs.py`.

## What happened — FALSIFIED at observation (no capturable i+1 short edge, either universe)

**JPY crosses** (base short cap 0.4348, drift −0.1816 — JPY-cross shorts already adverse, the Abenomics-era
up-drift):
- up-gap thr+0.5: **cap 0.4484 (<0.50), drift −0.093** (NEGATIVE for the short → the up-gap CONTINUES up, it
  does NOT revert at the i+1 bar), per-pair drift>0 only **1/5** (only EURJPY +0.39).
- thr+1.0: cap 0.469, drift −0.034 (still coin-flip/adverse), 3/5 pairs.
- thr+1.5: cap 0.420, drift −0.449, 1/5 pairs.

**USD majors** (base short cap 0.4531, drift −0.0276 — coin-flip):
- up-gap thr+0.5: **cap 0.4175, drift −0.065** (adverse for the short), per-pair 3/7.
- thr+1.0/1.5: cap 0.457/0.474, drift −0.184/−0.076 — coin-flip-or-adverse throughout.

**The 2015 & 2018 acceptance test (the whole point):**
- **USD majors** (arc 2001's universe): 2015 drift **−0.65**, 2018 drift **−0.21** — BOTH NEGATIVE for the
  short. Fails outright.
- **JPY crosses** thr+0.5: 2015 **+0.36** (cap 0.357!), 2018 **+1.99** (n=31) — *look* positive, but this is
  **regime-luck within a coin-flip**: the overall cell is cap<0.50 + drift −0.093, per-pair 1/5, tiny n
  (2015 n=14), and the per-year drift swings wildly (2014 −1.88, 2019 +2.15, 2020 −2.74) — exactly arc
  3010's finding that "2018-positivity is uncapturable regime-luck within a yearly coin-flip." A positive
  2015/2018 inside an overall-negative, sub-0.50, 1-of-5-pair signal is noise, not a capturable leg.

## Verdict

**KILL (cheap-kill at observation).** The honest i+1 up-gap short is coin-flip-or-adverse (cap 0.42–0.47 <
0.50, short drift ≤ 0) on BOTH universes; the JPY-cross 2015/2018-positive cells are regime-luck within a
coin-flip (overall-negative, per-pair 1/5, tiny n), and the USD-major 2015/2018 cells are outright negative.
Entry doesn't beat its base + no positive gross short drift → **§5f exit-sweep does NOT bite** (it is
reserved for entries that beat null or show +gross drift); a sub-0.50/≤0-drift gross signal is guaranteed
sub-cost on the honest engine (arc 3003) → **no pool/engine/null/council spent** (arc-1001/1002/3006/1010/1014
discipline).

## Diagnosis — why the "stronger leg" isn't tradeable (the mirror asymmetry, again)

arc 2001's "+0.57 ATR / frac+ 0.64 / stronger leg" was measured from the **untradeable gap-bar OPEN**
(hindsight framing — arc 2001 itself flagged this). At the honest i+1 entry the reversion is **already
spent**, and the up-gap then CONTINUES up — the same **backward-confirming** failure as arc 1014's
confirmed-breakdown short (the move already happened in the signal bar → the i+1 short enters the local high
→ price drifts back up against it). The arc-1006 gap-fill LONG mirror does NOT flip favorably because (1)
the down-gap-fill long catches a reversion that is *still ahead* of i+1 on JPY crosses (and rides the
Abenomics up-drift tailwind), whereas the up-gap short fights that same up-drift as a headwind; (2) the
reversion magnitude that survives to i+1 is below the coin-flip/cost line on both universes. **Forward-
confirming reversal structure is tradeable (arc 1013 reclaim long); backward-confirming flow continuation
(up-gap short, confirmed-breakdown short) is not** — the arc-1014 asymmetry re-confirmed on a flow event.

## Threads / lessons

1. **The up-gap weekend SHORT — the dispatch's #1 short lead and the named 2018-leg candidate — is a
   cheap-kill at the honest i+1 entry.** arc 2001's "stronger short leg" is a gap-bar-open hindsight
   artifact, not capturable at a tradeable entry (arc-2001's own caveat, now confirmed on the short side).
2. **The 2018-positive 4th component is NOT found in ANY short construction tried** — structural breakdown
   (1014), climax-sweep (2009), trend/sustained-USD (3010), and now flow-event up-gap (1016) all KILL. The
   portfolio route's 2018 wall (arc 1015) stands; shorts revive neither directional structure (1014) nor
   trend (3010) nor flow-continuation (1016). The value of shorts is NOT a symmetric flow-leg either.
3. **The arc-3010 "regime-luck within a coin-flip" screen is now load-bearing for the 2018 hunt:** a cell
   that is positive in 2015/2018 but sub-0.50 cap + per-pair-minority + tiny-n is noise, not the leg — do
   NOT promote it to the engine (would be fishing the 2 target folds, the arc-1012 trap). The acceptance
   test is "robustly short-positive in 2015 & 2018," not "the 2015/2018 cells happen to be positive."
4. **Surviving frontier for the 2018 leg:** the remaining open escalation unlocks are market-neutral
   relative-value (arc 2010 already KILLed it — doubled cost vs coin-flip) and a genuinely tighter-cost
   execution regime (operator decision, Arc-10 gate-mismatch risk). Within the long-only/shorts-enabled
   apparatus, the named short leads are now exhausted; the 3-way book (arc 1015) stands as the strongest
   non-deployable result and the standing arc-3004 escalation is reinforced from the short side.

## FLAGS (code not merged)

None. No canonical-core change; no new BUILT tool (reused the direction-aware `observe_long_capture`; the
up-gap mask is a one-off scratch conditioning helper, like arc 1014's short observer). Carries FLAG-1
(shorts unblocked PR #273, but the named flow/structure/trend short leads are now shown dead for the 2018
leg — FLAG-1's residual value narrows to a genuinely different unlock, e.g. tighter-cost regime) and the
standing `A1Config.time_exit_bars`-unwired flag. Driver scratch `_disco_work/arc1016_upgap_short_obs.py`
(reproducible from the arc doc).

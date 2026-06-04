# Arc 3001 — Drift-Lens Scan (is the directional-long death a capture-metric artifact?)

> **Arc id:** 3001 · **Chat:** 3000–3999 (continuous) · **Date:** 2026-06-04
> **Final verdict:** **FAIL (cheap-kill at observation, confirmed by triage)** — no instrument × condition
> shows cost-clearing forward drift; the one positive cell (post-up-spike continuation on trending crosses,
> +0.023R gross) is net −10.78% (2/3 folds neg) after costs.
> **Idea-family:** a *methodological* probe, not a new signal family — test whether the universal
> directional-long failure (arcs 0,1000–1004,3000) is partly an artifact of the **+1R-before-SL capture
> metric**, which arc 1004 flagged as **blind to small persistent drifts**. Re-scan with the correct **mean
> forward DRIFT** lens.

Scored solely by `MultiPairBacktester` (FundedNext costs ON, SL-first). Engine/measurement **called, never
re-rolled**; the scan + signal are experiment tools.

---

## (a) Log read + synthesis — FRESH EYES

Pulled `origin/main`. Honest-era corpus = 7 arcs (0; 1000–1004; 3000), **all FAIL**. LESSONS.md still
uncompressed. The consolidated picture: **six independent directional-long arcs fail identically** —
continuation family (pullback 0, XS-momentum 1000, vol-contraction-breakout 1001, D1 trend 1002, cross-trend
1003) + reversion family (oversold-revert 3000) — across **two timeframes** (H4, D1), the **full 28-pair
universe** (majors + crosses), and **two chats** (1000s + 3000s). Arc 1003 sharpened the constraint to
**EDGE < COST**; arc 1004 showed exit engineering can't fix it (it's an entry/cost problem).

**The specific thread this arc attacks.** Arc 1004's parting methodological note: *"calendar-flow tested with
the CORRECT metric (mean forward DRIFT; +1R-before-SL is blind to small drifts)."* Every prior arc judged the
entry with the **+1R-before-SL capture** metric (a symmetric ±R barrier race). That metric is **blind to a
small persistent drift**: a signal could have capture < 0.50 yet a positive mean forward return (positive
skew / slow drift). Indeed arc 1003 reported +0.10R *gross* on crosses despite sub-0.50 capture. **So before
declaring the directional-long space dead, the honest move is to re-scan it with the drift lens.** If a
cost-clearing drift exists somewhere the capture-scan missed → a real lead. If not → the death is
metric-robust and the directional-long space is definitively closed.

**My own arc-3000 note suggested "portfolio/selection" next — I reconsidered and rejected it as premature**
(documented in (b)): you cannot diversify net-negative components into a positive system; a portfolio needs a
net-positive component first, and none exists. The actionable gap is the metric re-scan. **Distinct from the
1000s chat:** they take the *calendar/time* drift axis (month-end); I take *price/volatility/structure*.

## (b) Observation → idea (the drift lens)

**Metric:** `fwd_drift_N(s) = (mid_close[s+N] − entry@(s+1) open_ask) / (2·ATR[s])` — N-bar forward return in
R units, realistic entry at ask, gross. **N = 12** H4 bars (~2 trading days). All 28 pairs, IS 2010–2020.
**Cost hurdle ≈ +0.05 to +0.10R** (1.5× spread + slippage×n_fills + commission, per round-turn).

**Forward 12-bar drift (R, gross), by group × conditioning (causal states at signal bar):**

| group | uncond | post-up-spike | post-dn-spike | uptrend (>SMA200) | mom_hi (top r24) | low-vol |
|---|---|---|---|---|---|---|
| MAJOR | −0.0188 | +0.0067 | −0.0723 | −0.0306 | −0.0319 | −0.0038 |
| COUPLED | −0.0825 | −0.1010 | −0.1383 | −0.0744 | −0.1011 | −0.1314 |
| TREND_X | −0.0120 | **+0.0228** | −0.0138 | −0.0117 | −0.0047 | −0.0211 |

**NO instrument × condition shows gross forward drift > +0.10R; none even clears +0.05R.** The single best
cell anywhere is **TREND_X post-up-spike at +0.0228R** — *below* the cost hurdle by construction. Three
findings:
1. **The directional-long death is METRIC-ROBUST.** The drift lens *agrees* with the capture lens — there is
   no hidden, cost-clearing forward drift in price/vol/structure on any instrument. The capture-metric
   blindness flagged by arc 1004 did NOT mask a price-structure edge.
2. **Coupled crosses drift NEGATIVE for a long** (uncond −0.0825R; oversold/low-vol *more* negative) — a
   completely independent re-confirmation of arc 3000 (reversion long on coupled crosses is bad): not only is
   capture < 0.50, the realised forward drift is negative (mean-reversion + spread).
3. **The only positive cells are post-up-spike momentum** (MAJOR +0.007, TREND_X +0.023) — i.e. the
   already-killed momentum-continuation family (arcs 1000/1002/1003), and sub-cost.

**Idea (confirmatory, not hopeful):** the one positive-drift cell — long the bar after a > 2·ATR up close on
the 12 trending crosses (post-up-spike continuation), refractory 6, SL=2·ATR. Predict: net-negative after
costs (gross drift +0.023R < cost). Run it on the honest engine to anchor the verdict-of-record with real P&L
(the engine realises drift — only the +1R-before-SL *label* was blind, not the engine).

## (c) Characterize + (d) cheap kill (confirmatory triage)

`build_arc_pool`, 12 trending crosses, H4 5ers_eet, IS 2010–2020, SL=2·ATR, hold 120.
`pool_sha256 bd5502811c1b9ed3…`. **1,375 IS trades**, honest +1R-before-SL capture **0.4982** (≈ coin-flip),
**gross mean final_r +0.0884** (small positive — the post-up-spike DOES carry a faint gross momentum, matching
the drift scan). Pool floor PASS. Oracle ceiling SKIPPED (best gross drift already < cost ⇒ no reachable
net upside).

**3-fold honest triage** (A1, SL=2·ATR, 1% reset-floor, exposure 1/pair 2/ccy,
`sl_partial_close_1r_runner_trail`, FundedNext costs ON, SL-first; OOS 2013/2016/2019):

| fold OOS | ROI | DD | n |
|---|---|---|---|
| 2013 | **+16.74%** | 7.75% | 159 |
| 2016 | **−28.07%** | 32.38% | 142 |
| 2019 | −21.01% | 24.08% | 157 |

worst **−28.07%**, mean **−10.78%**, **2/3 negative** → **KILL.** The faint gross momentum is **regime-fragile**
(huge in trending 2013, catastrophic in choppy 2016) and **net-negative** after costs — the identical
arc-1003/1004 momentum-on-crosses signature.

## (e)–(h) Diagnose / council / survivor — NOT REACHED

Cheap-killed; no reachable ceiling (the broad drift scan IS the diagnosis — drift is sub-cost everywhere). No
council (no diagnosis fork with reachable upside, no survivor). No `passed/` record.

## Final verdict — FAIL (metric-robust closure)

The directional-long failure is **not a capture-metric artifact**. Under the correct DRIFT lens (arc 1004's
flag), there is still no cost-clearing forward drift on any of 28 instruments under any of 6 price/vol/structure
conditionings; the lone positive cell is sub-cost and in the already-dead momentum family. The directional
price-structure long space is now closed under **both** metrics.

## Lessons (candidate for LESSONS.md compression)

1. **The directional-long death is METRIC-ROBUST (capture AND drift agree).** A 28-pair × 6-condition forward-
   drift scan finds no cell with gross drift > +0.023R (cost hurdle +0.05–0.10R). The +1R-before-SL metric's
   blindness to small drift (arc 1004) did NOT hide a price-structure edge — closing that loophole.
2. **Coupled crosses have NEGATIVE forward long-drift** (−0.08R/12 bars; more negative when oversold or
   low-vol) — independently re-confirms arc 3000 from the drift angle.
3. **Faint gross momentum on trending crosses is regime-fragile and net-negative** (post-up-spike: +16.74%
   trending-2013 vs −28.07% choppy-2016, mean −10.78%) — re-confirms arc 1003/1004; momentum-on-crosses is a
   regime bet, not an edge.
4. **"Portfolio/selection" is premature/empty right now.** Diversification cannot turn net-negative components
   positive; it needs a net-positive component, and the drift scan shows none exists in the price-structure
   directional space. The steer only activates once a net-positive (even if thin/fold-fragile) edge is found.

## Threads / what didn't help

- **Closed (this arc):** the hypothesis that a cost-clearing drift hides under the capture metric — it does
  not. The directional price-structure long space is closed under both metrics, all instruments, all standard
  conditionings.
- **Surviving steer:** the directional drift wall leaves only non-price-direction constructions. **Update
  (landed mid-arc):** arc 1005 tested the calendar/flow steer (turn-of-month USD-long) and it ALSO came back
  sub-cost and thin (worst −2.78%, n=128) — so even calendar/flow is weakening, not just price-structure.
  After two arcs (3000, 3001) I have closed the price-structure directional space from the 3000s side, and
  the obvious calendar effect is fading too. Arc 3002 needs a *generatively different* idea — a strong
  candidate for a LIGHT generative council at the idea-fork (protocol §5b) rather than another lone guess.
- **Premature, not closed:** portfolio/selection — revisit only after a net-positive component exists.

## Flags (code NOT merged — human-gated, per protocol §9)

None requiring the canonical core. Scan + signal + triage drivers are scratch (`_disco3_work/`, not committed;
reproducible below). No reusable experiment tool needed.

## Reproduction

- **Data:** `histdata_root = C:\Users\panap\histdata_backup`, `cache_root = data/cache`,
  `boundary_convention = "5ers_eet"`, TF H4. **Scan:** all 28 pairs. **Triage pairs:** the 12 trending crosses
  (EURJPY GBPJPY AUDJPY NZDJPY CADJPY CHFJPY EURAUD EURNZD EURCAD GBPAUD GBPNZD GBPCAD).
- **Drivers (scratch):** `observe3_drift.py` (b, drift scan), `arc3001_signal.py`
  (`PostUpSpikeContinuationLong(spike_atr=2.0, refractory=6)`), `arc3001_triage.py` (c/d).
  `pool_sha256 bd5502811c1b9ed3…`. Run `PYTHONPATH=. py _disco3_work/<script>.py`.
- **Engine:** `MultiPairBacktester` via `A1Architecture` + `ArcFoldRunner`; FundedNext costs at
  `build_fold_stats_from_run`; IS folds `build_v3_folds`; triage OOS 2013/2016/2019; judge
  `judge_all_folds_positive`.

# DISCOVERY_LOG

This is the machine-scannable index of every discovery arc. The **Tier-1** table below is one row per
arc with fixed fields; the operator's check-in is a single scan of the `passed` column for `Y`. Discovery
chats **APPEND only** — they never edit or compress this file (compression is operator-run, out-of-band;
see [`DISCOVERY_PROTOCOL.md`](./DISCOVERY_PROTOCOL.md) §6). The log is READ at arc step (a) and appended
at arc step (i).

## Tier 1 — Arc Ledger (strict schema)

| arc_id | chat | timestamp | hypothesis | IS_all_folds_pos | OOS_all_folds_pos | worst_fold_ROI_IS | worst_fold_ROI_OOS | worst_DD | n_trades | VERDICT | passed | disposition |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | trial | 2026-06-04 | Pullback-in-uptrend long (close>SMA50 & pierce prior-5-bar-low); best ver = excursion-banking exit | N | N | -14.2% | -14.3% | 17.1% | 4985 | FAIL | N | KILL |
| 1000 | 1000s | 2026-06-04 | Cross-sectional momentum long (top-quintile 24-bar ret, rising; partial/runner exit) | N | N | -21.20% | -15.36% | 24.50% | 7757 | FAIL | N | KILL |
| 1001 | 1000s | 2026-06-04 | Volatility-contraction breakout long (low-ATR coil + 10-bar-high break) — cheap-kill at triage | N | n/e | -9.41% | n/e | 12.30% | 3186 | FAIL (triage) | N | KILL |
| 1002 | 1000s | 2026-06-04 | D1 daily trend-following long (Donchian-20 breakout in uptrend) — TF lever; cheap-kill at triage | N | n/e | -5.72% | n/e | 6.61% | 831 | FAIL (triage) | N | KILL |
| 1003 | 1000s | 2026-06-04 | Cross trend-momentum long (Donchian-20 breakout in uptrend, 8 trending crosses) — universe lever; cheap-kill | N | n/e | -9.75% | n/e | 12.39% | 4078 | FAIL (triage) | N | KILL |
| 1004 | 1000s | 2026-06-04 | Cross-trend exit/cost engineering (let-it-run / 3R vs partial-runner) — EXIT lever; cheap-kill | N | n/e | -9.75% | n/e | n/e | 4078 | FAIL (triage) | N | KILL |
| 1005 | 1000s | 2026-06-04 | Turn-of-month USD-long calendar drift (USDCHF/USDXXX, 6-bar time exit) — non-directional mechanism; cheap-kill | N | n/e | -2.78% | n/e | 2.96% | 128 | FAIL (triage) | N | KILL |
| 1006 | 1000s | 2026-06-04 | Weekend gap-down-fill long, JPY crosses (gap<-0.5ATR, 24-bar time exit) — generative-council idea; REAL but fold-fragile (FIRST mean-positive IS edge +0.69%, beats random) | N | N | -6.79% | -4.13% | 8.57% | 396 | FAIL | N | PORTFOLIO |
| 1007 | 1000s | 2026-06-04 | Gap-fill exit-AT-target (TP at gap origin) — best-version test of arc 1006; edge is OVERSHOOT not fill, capping it is WORSE (IS mean -1.21%, loses to null) | N | N | -6.76% | -3.93% | 6.95% | 396 | FAIL | N | KILL |
| 2000 | 2000s | 2026-06-04 | Trend-following long via full-size convexity harvest (Donchian breakout + full-size trailing) — fat tail is generic not trend-selected; cheap-kill at triage | N | n/e | -13.97% | n/e | 15.8% | 1617 | FAIL (triage) | N | KILL |
| 3000 | 3000s | 2026-06-04 | Mean-reversion long on coupled crosses (RSI<25 oversold) — instrument-universe + reversion lever; cheap-kill | N | n/e | -20.22% | n/e | 22.54% | 1009 | FAIL (triage) | N | KILL |
| 3001 | 3000s | 2026-06-04 | Drift-lens scan (mean fwd drift, the metric +1R-before-SL is blind to) across 28 pairs × 6 conds — directional-long death is METRIC-ROBUST; best cell post-up-spike trending-cross net −10.78% | N | n/e | -28.07% | n/e | 32.38% | 1375 | FAIL (triage) | N | KILL |
| 3002 | 3000s | 2026-06-04 | Volume-confirmation long (vol-spike on trending crosses) — the last untouched data column; full IS 7/10 neg | N | n/e | -16.25% | n/e | 20.57% | 839 | FAIL (IS WFO) | N | KILL |
| 3003 | 3000s | 2026-06-04 | Regime detection for momentum (Kaufman Efficiency-Ratio): is the trending regime detectable in advance? REGIME INVERSION (strong trends revert); best-case band catastrophic | N | n/e | -63.11% | n/e | 66.51% | 9250 | FAIL (IS WFO) | N | KILL |
| 2001 | 2000s | 2026-06-04 | Weekend-gap-fill long, MAJORS (buy big weekly-open down-gap, reversion to prior close) — CONVERGES w/ arc 1006 (JPY crosses); REAL edge but uncapturable: majors mean-NEG (vs 1006 crosses mean-POS), honest i+1 entry lands in adverse continuation (MAE −1.1R), thin (~13-17/fold); HEAVY diagnosis-council KILL, OOS preserved pristine | N | n/e (OOS preserved) | -4.57% | n/e | n/e | 176 | FAIL (IS+council) | N | KILL |
| 2002 | 2000s | 2026-06-05 | H1 entry-resolution test of the weekend gap-fill (enter ~6h closer to the gap open) — does finer resolution capture the fill the H4 i+1 entry missed? NO: H1 capture 0.450 ≡ H4 0.45-0.47, MAE −1.2R ≡ H4; the down-gap's adverse continuation is TF-INVARIANT. Confirmatory cheap-kill; vindicates arc-2001 council (uncapturable, not an H4 artifact) | N | n/e | -2.02% | n/e | n/e | 151 | FAIL (triage) | N | KILL |
| 2003 | 2000s | 2026-06-05 | Cross-pair relative-value divergence catch-up (long the under-performing leg when the synthetic cross is extreme) — LIGHT generative-council idea; cheap-kill at obs: the laggard CONTINUES lagging (z<−2 fwd drift −0.05 to −0.11 ATR; per-group z<−1 fwd10 −0.20 EURvGBP / −0.15 AUDvNZD), NO reversion — relative perf is momentum/coin-flip (re-confirms 1000/3001). Council's strongest call = CONCEDE: long-only constraint is the binding blocker (FLAG-1) | N | n/e | n/e | n/e | n/e | n/e | FAIL (obs) | N | KILL |
| 3004 | 3000s | 2026-06-05 | Stop-confound test (council-driven): is the universal 2*ATR SL the wall? Pure time-exit on best +drift entry — stop removed lifts mean ~4-5pp but STILL not all-folds-pos → ESCALATION (apparatus structurally incapable; needs 2nd leg/shorts/cost regime) | N | n/e | -20.43% | n/e | n/e | 799 | FAIL → ESCALATION | N | KILL |
| 3005 | 3000s | 2026-06-05 | Cross-rate triangulation residual (synthetic vs quoted cross) — last non-directional in-apparatus thread; residual sub-spread, conv-corr ~0 at H4 | N | n/e | n/e | n/e | n/e | n/e | FAIL (obs cheap-kill) | N | KILL |
| 1008 | 1000s | 2026-06-05 | Gotobi-day Tokyo-morning USDJPY long (5/10/15/20/25+EOM, fix-window) — documented JP corp USD-settlement microstructure flow; FALSIFIED at obs: gotobi premium ≤+0.25bp gross (best window, frac+ 0.501 coin-flip) vs ~2bp cost, absent at H4 AND H1; only +era (2010-14) is generic Abenomics drift not gotobi-specific | N | n/e | n/e | n/e | n/e | n/e | FAIL (obs cheap-kill) | N | KILL |
| 2004 | 2000s | 2026-06-05 | Stop-tax A/B on the FILL/REVERSION family (weekend gap-fill majors) — completes arc-3004's escalation for the OTHER mechanism family. Removing the 2·ATR SL-first stop ~HALVES DD/loss-tail (maxDD 3.76→1.56%, worst −3.11→−1.12%; larger relative effect than 3004's momentum case, matches the −1.1R MAE mechanism) + lifts mean to ~0, but NO config/horizon is all-folds-positive → residual edge ≈ cost, regime-dependent. Stop is a real drag for BOTH families but NOT the wall; FLAG-2 answered (H4+FundedNext hostile to fill via EDGE≈COST, not a fill-specific stop tax). OOS preserved | N | n/e (OOS preserved) | -1.92% (best stop-removed N=24) | n/e | 1.56% (stop-removed N=6) | 176 | FAIL | N | KILL |
| 1009 | 1000s | 2026-06-05 | AUDIT of arc 1006 gap-fill (SAME component, not a new edge): reproduce + robustness. REPRODUCES +0.685% IS via registered tools; NOT single-pair (leave-one-out all +); but threshold-FRAGILE (lives at 0.5ATR, ~0 by 1.25) AND edge over a FAIR same-exit null is ~+0.36pp (~HALF the headline; null itself +0.327% from JPY-basket drift). Component HOLDS but thinner | N | n/e (not re-run) | -6.79% | n/e | n/e | 260 | PORTFOLIO (re-affirmed, thinner) | N | PORTFOLIO |
| 3006 | 3000s | 2026-06-05 | Multi-TF breakout-RETEST long (resistance→support flip; the last untested long-only lane, arc-1008 flag) — FALSIFIED at obs: retest-hold capture ≤ base in all 18 L×tol×hold cells, fwd drift NEGATIVE in 17/18 (−0.35..+0.00 ATR); catches FADING breakouts not defended supports; monotone-in-L only converges TO coin-flip, never above | N | n/e | n/e | n/e | n/e | 1200 | FAIL (obs cheap-kill) | N | KILL |
| 1010 | 1000s | 2026-06-05 | Round-number / order-cluster reversion long (Osler 2003: take-profit clusters at big-figure "00" levels → bounce off support) — novel structural microstructure mechanism, decorrelated portfolio candidate; FALSIFIED at obs: support-rejection capture only 0.4883 (+0.5pp, drift ~0) sub-0.50, per-pair lift 3/9 >0.50 = NOISE, and the H4 lift EVAPORATES at H1 (0.4746 < base; EURUSD 0.5155→0.4834). Round-number effect arbitraged at H4/H1 like gotobi | N | n/e | n/e | n/e | n/e | n/e | FAIL (obs cheap-kill) | N | KILL |
| 1013 | 1000s | 2026-06-05 | DEEP multi-factor: failed-breakdown RECLAIM (stop-run reversal) long, USD majors H4 — deep wick that pierces a 40-bar swing low (sweeps stops) + reclaims it + lower-shadow ≥1.25 ATR. **STRONGEST + cleanest directional edge in the corpus** → PORTFOLIO (3rd component). Structure control-proven (same-wick-AT-swing-low cap 0.52-0.61 vs SAME-wick-elsewhere coin-flip + NEG drift; generic wick-reversion dead). Engine IS mean +1.85% (9/10), every exit mean-pos; OOS +0.94% (3/6, mean-POS forward); beats a NEGATIVE same-exit null by +2.96pp (cleanest null in corpus); robust K∈{40,60}×shadow≥1.25 + LOO all+. NOT all-folds-pos (only neg fold = 2018 strong-USD regime; OOS neg 2022/25 same signature). POSITIVE in 2015/16/20 → the arc-2006 regime-orthogonal 3rd leg | N | N | -4.20% | -2.76% | 4.87% | 172 | FAIL → PORTFOLIO | N | PORTFOLIO |
| 3008 | 3000s | 2026-06-05 | Month-end London-4pm-fix flow dislocation-and-reversion long (buy abnormal DOWN-push into the 16:00 WM/Reuters fix on the last weekday of month, bet post-fix reversion) — arc-1010's last named discrete-flow lane; REAL & month-end-specific (corr −0.121 vs −0.035 non-ME; big-down-push +4.3bp/2h frac+ .576) and BEATS the fair null (−0.28% vs −0.73%, 9/10 neg) but SUB-COST: best of 8 exit/SL×threshold cells (stop-removed time3) is −0.22 to −0.28% IS-mean, none all-folds-positive; H1 cost-in-R (1R≈H4-ATR) ~2× taxes the +0.12R/3h drift; 4th flow effect real-but-uncapturable (gotobi/round-no/triangulation). OOS preserved | N | n/e (OOS preserved) | -0.22% (best, stop-removed N=3 thr0.25) | n/e | 1.41% (stop-removed) | 160 | FAIL → KILL | N | KILL |
| 3007 | 3000s | 2026-06-05 | Intraday-SPREAD cost-timing lever on EDGE<COST (restrict cross-trend +gross signal to tightest-spread hours to cut cost — the one untried cost lever, arc 1003/1004) — BACKFIRES: spread×edge entangled, tight-spread/liquid hours have WEAKEST (negative −0.02R) gross edge; only +gross bucket is mid-spread (+0.12R) which nets ≤0 SL-honest (triage mean −0.02%, not all-folds-pos on friendly yrs). Cannot cut cost w/o cutting edge | N | n/e | -0.13% (mid-spread triage) | n/e | 0.14% | 2955 | FAIL → KILL | N | KILL |
| 1011 | 1000s | 2026-06-05 | Month-end reversion long, USD majors (big DOWN move into month-end reverses — WMR-fix mechanical rebalancing over-extension; D1, ~2-bar time exit) — **2nd net-positive DECORRELATED long-only component**. MECHANISM-CONTROLLED: month-end vs random-day +0.249 ATR EXCESS (generic reversion NEGATIVE −0.063 → timing is causal). IS mean +0.23% (sl_only 2-bar), beats fair null +0.56pp, threshold-robust (0.75–1.5), leave-one-pair-out all+; but 7/10 folds → NOT all-folds-pos. Corr +0.117 vs arc 1006 → portfolio thread ACTIVE | N | n/e (OOS preserved) | -1.14% | n/e | n/e | 121 | FAIL → PORTFOLIO | N | PORTFOLIO |
| 2005 | 2000s | 2026-06-05 | Attack EDGE<COST from the COST side: restrict cross-trend +gross entry (Donchian-20+SMA200, 12 crosses) to its cheapest bars (spread/ATR trailing-rolling-quantile, BUILT make_low_cost_mask). MONOTONE IS lift −7.96%(all)→+1.72%(cheapest 15%), beats matched random-cheap null (≈−0.3%) IN-SAMPLE — but OOS −2.26% (4/6 neg) + q full-sample-swept → NOT durable. CONVERGES w/ arc 3007 (3000s, absolute spread-timing): cost↓ entangled with edge↓ (liquid hours = weakest gross edge) → both chats CLOSE the cost side. DIRECTION remains the wall (3004); cost+stop are secondary ~5–10pp drags | N | N | +1.42% (cheapest-15% IS) | -8.33% | 6.01% | 682 | FAIL | N | KILL |
| 1012 | 1000s | 2026-06-05 | Month-end reversion §5f best-version completion + survivor-ceiling test (same arc-1011 component, not a new edge): full registered exit/SL menu × time-cap, 24 configs → **0 all-folds-positive** (best 7/10 = baseline; te5 + all TP/trail DILUTE). Negative folds = contiguous **2014/2015/2016 USD-bull block** (SNB-unpeg/EUR-collapse); both pre-registered reasoned refinements FAIL to rescue — quarter-end-only 4/10 (thins to 0), trend-filter close>SMA100 4/10; post-hoc inverse close<SMA100 8/10 but still 2015/16 neg & fishing → not claimed. Survivor ceiling CLOSED; exits/filters can't lift it. OOS preserved | N | n/e (OOS preserved) | -1.14% | n/e | 1.43% | 121 | FAIL survivor → PORTFOLIO (ceiling confirmed) | N | PORTFOLIO |
| 2007 | 2000s | 2026-06-05 | **DEEP multi-factor** liquidity-sweep reversal long (sweep prior swing-low × immediate reclaim × fast-3-bar drop × SMA200 uptrend; developed freely on IS) — the open-frontier "can depth clear all-folds-positive?" test. Obs: conjunction lifts +1R-capture 0.486→0.52 but drift sub-cost (+0.03 ATR); the CLIMAX (forced) variant is a FALLING KNIFE (drift −0.33, the short leg). Honest engine: ALL 7 exits × 2 SLs mean-NEGATIVE, none all-folds-pos, and **LOSES TO THE FAIR NULL** (real −1.76% vs null −0.33% te6) → not even a real edge. Buying structural weakness = adverse-excursion = worse than random under take-the-loss; a chart pattern that resembles a grab is NOT a grab (no forced-flow control, cf. 1011). OOS preserved | N | n/e (OOS preserved) | -5.61% (PRIMARY L20 te6) | n/e | 5.61% (worst, te6) | 508–763 | FAIL → KILL (worse than null) | N | KILL |
| 1014 | 1000s | 2026-06-05 | **FIRST SHORT ARC.** Confirmed-breakdown continuation SHORT (deep swing-low pierce, NO reclaim, close-below — the short mirror of arc 1013's reclaim-long), H4 USD majors — FALSIFIED at obs: confirmed-breakdown short cap 0.4893 (<0.50) + drift −0.046 (REVERTS, bad for short); deeper close-below reverts HARDER (−0.21 @ 0.25-0.5 ATR); **structure control FAILS** (big-red AT swept-low 0.4998 ≈ big-red ELSEWHERE 0.4963 — swing-low NOT load-bearing for the short, unlike the long); per-pair 3/7 = noise. Diagnosis: reclaim is FORWARD-confirming (move hasn't started → 1013 long wins), confirmed-breakdown is BACKWARD-confirming (move already happened → i+1 short enters the local low → reverts). The 1013 long has NO tradeable short mirror; shorts don't revive directional STRUCTURE (coin-flip by symmetry). Live short frontier = FLOW-event shorts w/ acc>0.50 (up-gap weekend short, 2001/2003) | N | n/e | n/e | n/e | n/e | ~4500 obs | FAIL (obs cheap-kill) | N | KILL |
| 2006 | 2000s | 2026-06-05 | 2-way PORTFOLIO combination WFO — the route's gated next step (arc 1011 flag): co-simulate the ONLY two net-positive long-only components (gap-fill 1006 JPY-cross H4 + month-end 1011 USD-major D1) under equal + risk-parity weights, gate all-folds-positive on the COMBINED book. Both reproduce EXACTLY (gap +0.685%, me +0.232%, corr +0.117 ✓). Risk-parity cuts worst-fold −6.79%(gap solo)→−1.53% but STILL 4/10 neg — BLOCKED by a mutually-negative fold (2015, both lose): no convex/co-sim combo can pass (P&L additive). The two edges are flow-reversion → TAIL-CORRELATED (both bleed 2015/16/18/20) despite corr +0.117. ⇒ ≥3 components needed, SELECTED for fold-complementarity (regime-orthogonality), not avg-corr. Components UNCHANGED (still PORTFOLIO); 3rd-component spec = net-pos on 2015/16/18/20, esp 2015 | N | n/e (OOS preserved) | -1.53% (risk-parity combined) | n/e | n/e | 396+121 | FAIL → KILL (combined book; components unchanged) | N | KILL |
| 2008 | 2000s | 2026-06-05 | 3-way PORTFOLIO combination WFO (gap-fill 1006 + month-end 1011 + **failed-breakdown 1013**, the new regime-orthogonal 3rd component) — the route's gated next step now a 3rd exists (arc-1013 spec). All 3 reproduce EXACTLY (≤0.005pp: gap +0.685%, me +0.232%, fbr +1.854%; corr gap-me +0.117 ✓, gap-fbr +0.189, me-fbr −0.366). C did as arc 1013 predicted — CUT the 2-way book's 4 bleed folds to 2 (fixed 2016 & 2020 outright) — but **2015+2018 survive**: risk-parity worst −0.77% (2/10 neg), equal worst −3.36% (3/10 neg). **PROVABLY blocked — 0/5151 convex weightings all-folds-positive** (best max-min −0.222%, 2014/15/18 neg): 2018 needs heavy month-end (its only +leg), 2015 needs heavy fbr & is HURT by month-end → mutually exclusive. **2018 = binding fold: flow-reversion (gap −6.79) AND structural stop-run reversal (fbr −4.20) BOTH lose** in the strong-USD trend year (avg-corr hid the shared 2018 tail). All 3 edges are fade/short-trend → the 4th leg must be trend-continuation/short, dead long-only → escalation/FLAG-1, arc-2007 climax-sweep SHORT the named target. Components UNCHANGED (still PORTFOLIO) | N | n/e (OOS preserved) | -0.77% (risk-parity combined) | n/e | n/e | 396+121+172 | FAIL → KILL (combined 3-way book; components unchanged) | N | KILL |
| 3009 | 3000s | 2026-06-05 | **3-way PORTFOLIO combination WFO** (INDEPENDENT concurrent re-run of arc 2008 — same verdict, complementary findings; Arc-10 independent-reproduction). Combine gap-fill 1006 (JPY-cross H4) + month-end 1011 (USD-major D1) + failed-breakdown-reclaim 1013 (USD-major H4) under equal + risk-parity, gate all-folds-positive on the COMBINED book. **All 3 reproduce EXACTLY** (gap +0.685%, me +0.232%, fbr +1.854% — caught arc-1013's native-trail double-trail via reproduce-live, trail_enabled=True). **The 2-way's combination-invariant 2015 block is GONE — NO mutually-negative fold** (fbr +3.17 rescues 2015; me +0.90 rescues 2018). But still NOT all-folds-positive: risk-parity 2/10 neg {2015,2018}, equal 3/10. fbr rescued 2016+2020 → blockers narrowed {2015,16,18,20}→{2015,2018}, worst-fold −1.53→−0.77%. **2018 = the hard wall** (strong-USD trend yr: gap −6.79 & fbr −4.20 both deep-neg, only me +0.90) — arc 2008's convex-search proves 0/5151 weightings pass (2015-needs-fbr vs 2018-needs-me are mutually exclusive). 4th-component spec = net-pos on 2015 AND esp 2018 → a SHORT / trend / risk-off-positive leg (longs dead those yrs; shorts now OPEN PR#273; named targets = climax-sweep short 2007, up-gap short 2001/2003). Components UNCHANGED | N | n/e (OOS preserved) | -0.77% (risk-parity combined) | n/e | n/e | 260+98+208 | FAIL → KILL (combined book; components unchanged) | N | KILL |
| 2009 | 2000s | 2026-06-05 | **Climax-sweep SHORT** (down-trend continuation) — the named 4th-component target (arc 2008): the violent fast-3-bar-drop big-range bar that sweeps a swing low & closes below → continuation DOWN, the engine-unvalidated thread arc 1014 flagged (arc 2007's −0.33 "falling knife"). FALSIFIED at obs: cell coin-flip (cap 0.49–0.505) + short drift near-zero-to-REVERTING (−0.04 to −0.06; only thin range≥2 tail +0.009); arc-2007 −0.33 does NOT reproduce in a clean SHORT construction. Climax monotonicity INVERTS (small −0.155→climax −0.038; deep pierce reverts to only +0.006). **STRUCTURE CONTROL decisive + ANTI-load-bearing:** climax AT swept-low drift −0.043 (REVERTS) vs ELSEWHERE +0.128 (CONTINUES) — the swing-low sweep is a REVERSAL setup (arc-1013 long), shorting it shorts INTO the bounce. Per-pair 3/7 = noise, split by USD quote-convention (XXXUSD continue/USDXXX revert = directional USD-beta). Swing-low sweep has NO tradeable short mirror (confirmed both leans: 1014 clean + 2009 climax). 2018's trend-continuation leg is NOT a structural short → reinforces arc-3004 escalation | N | n/e | n/e | n/e | n/e | ~121k obs | KILL (obs cheap-kill) | N | KILL |

---

## Per-Arc Reasoning (free-form)

Each arc appends under its own `### arc_<id>` header — the why/because, the approach taken and its rationale, what was tried, what didn't help, and threads worth pursuing. Append freely; empty until the first arc lands.

### arc_0

**SUPERVISED TRIAL** (first run of the protocol; arc id 0, real arcs start at 1000). Full record:
[`arcs/arc_0_pullback_in_uptrend_long.md`](arcs/arc_0_pullback_in_uptrend_long.md). Council
transcript: [`results/arc_0_pullback_in_uptrend_long/council_transcript.md`](results/arc_0_pullback_in_uptrend_long/council_transcript.md).

**Idea + why.** Log was empty (cold start, fresh eyes — no pre-reset priors). Observed EURUSD H4
(IS 2010–2020): naive long breakouts FADE (p_win 0.477/0.471 < 0.492 unconditional) — momentum-
long rejected by the data; the best simple long trigger is buying a short-term dip in an uptrend
(p_win 0.505). Hypothesis: trend resumes after shallow pullbacks; breakouts revert at extremes.
Signal v0: long when close>SMA50 & bar pierces prior-5-bar-low, refractory 6, SL=2·ATR. 8 liquid
pairs, real HistData bid/ask (recovered from the 65 GB backup), H4 5ers_eet, honest engine, costs ON.

**What happened.** Pool 4,985 IS trades (floor PASS). Raw exit slightly negative (mean −0.045R)
but huge favorable excursion given back (≥1R MFE 0.50, ≥2R 0.33; honest +1R-before-SL 0.498).
Clustering (k=4) cleanly split ~50% "good" (clusters 0/1, median MFE 5.9R/2.0R) vs ~50% "bad"
(clusters 2/3, wrong-way-first 96–99.7%). Oracle-best-cluster ceiling worst-fold **+12.7%** (strong
→ PROCEED); raw triage worst −14%.

**The crux (why it died).** Diagnosis: the gap is SELECTION, but entry-time observables do NOT
separate good from bad clusters — every univariate AUC ≈0.5 (max atr_pct 0.527), no structural
filter lifts the 0.497 good-fraction. The oracle ceiling is **answer-key hindsight** (clusters
defined from realised path). Diagnosis council (heavy) recommended: don't run an ML entry-filter
(it would chase hindsight clusters + overfit at 5k samples); test an excursion-banking EXIT instead,
gated by a random-entry null baseline. CC committed. Full honest WFO: `partial_1r_runner_trail`,
`tp_2r`, `sl_only` ALL fail all-folds-positive (best worst-fold IS −14.2%, 7/10 folds negative; OOS
likewise). Null baseline: real signal (mean fold ROI −4.7%, 7/10 neg) beats random entry (−8.7%,
9–10/10 neg) → **a real but sub-cost edge**; not enough to clear FundedNext costs + SL-first.

**Verdict: FAIL.** Family dead for deployment.

**Threads / what didn't help.** Entry-selection on this signal is closed (the discriminator is
post-entry, not on the chart). Excursion-banking exits (partial/TP/trail) do not rescue a near-
coin-flip entry once costs + take-the-loss apply. WEAK open thread: the signal is decorrelated-ish
and sub-cost — possible portfolio value only if a *structurally different* entry refinement raises
the bankable +1R fraction; low priority, not standalone.

**Carry-forward lessons (candidate for LESSONS.md compression):** (1) a high oracle-best-cluster
ceiling proves nothing when clusters are realised-path-defined — always ask "can entry-time
observables reach it?" before believing it (Arc-10 lesson applied); (2) gross MFE-touch rates are
path-optimistic — the bankable fraction after costs + SL-first is materially lower; (3) for buy-
the-dip-in-uptrend H4 majors, the resume/fail split is not in entry-time price structure.

**FLAGS (code not merged):** council's preferred BE-after-+1R exit policy absent from
`core/sim/exit_policies/` (needs human-gated code to test); `build_ex_ante_bounded_population` named
in protocol/CLAUDE.md is `build_arc_pool` in-tree (doc nit); arc drivers kept in scratch `_arc0_work/`.

### arc_1000

**Cross-sectional momentum long** (chat 1000–1999, first continuous arc). Full record:
[`arcs/arc_1000_xsect_momentum_long.md`](arcs/arc_1000_xsect_momentum_long.md). Council transcript:
[`results/arc_1000_xsect_momentum_long/council_transcript.md`](results/arc_1000_xsect_momentum_long/council_transcript.md).

**Idea + why.** Arc 0 closed single-pair entry-time price structure for a long. So I probed the axes
ORTHOGONAL to it: WHEN (time-of-day/session, day-of-week, vol regime, prior move) and WHICH
(cross-sectional relative strength across the 8 majors). Observation (138,816 hypothetical honest longs,
IS 2010–2020, unconditional capture 0.4877): the WHEN axes are DRY (max lift +1.3pp, noise, DST-confounded
seasonality artifact). The WHICH axis showed a faint, monotone, mechanistic momentum tilt (strongest
cross-sectional quintile capture 0.495 vs weakest 0.483) — so the best reasoned version = long the
top-quintile 24-bar-return pair that is also rising (refractory 6, SL=2·ATR, partial/runner exit). Because:
cross-sectional FX momentum is a continuation effect; the absolute-positive filter keeps us long a real uptrend.

**What happened.** Pool 7,757 IS trades, honest capture 0.4939 (matches obs). Cluster structure MIRRORS
arc 0 (~50% resume / ~50% fail; cluster 0 mfe_p50 5.79R; clusters 2/3 wrong-way 0.97/1.0). Cheap kills:
oracle ceiling STRONG (worst-fold +22.4%), raw 3-fold triage NOT deeply negative (2013 +11.98%, 2016 −0.92%,
2019 −3.03%) → proceeded to diagnose (did not cheap-kill).

**The crux (why it died).** Diagnosis: the selection lever is CLOSED — every entry-time observable,
INCLUDING the cross-sectional ones that are the signal's whole novelty (rank/dispersion/board-drift/relative-
strength), separates good {0,1} from bad {2,3} at AUC ≈ 0.50–0.51; good-fraction by tercile flat 0.48–0.51.
The regime hypothesis is FALSIFIED: the highest cross-sectional-dispersion (clearest-trend) quintile is the
WORST (mean final_r −0.24R). The 2013 +11.98% is regime/luck, not separable edge. HEAVY council (5 lenses +
peer review + chairman): KILL — dispersion inversion = falsification not tuning; capture 0.4939 over 7,757
is indistinguishable from 0.50 (CI ±0.011); oracle ceiling is hindsight exit-variance; the lone "keep
testing" dissent (extension-ceiling sweep) was settled against by all 5 reviewers as a non-orthogonal
re-cut of the already-falsified dispersion axis ("ritual not rigor"). CC committed (no override); did NOT
run the sweep. Full WFO verdict-of-record: IS 7/10 folds neg (worst −21.20%) → NOT all-folds-positive; OOS
5/6 neg (worst −15.36%) → NOT all-folds-positive. Null baseline: REAL mean fold ROI −4.36% (7/10 neg) BEATS
random −9.19% (9–10/10 neg) → a real but SUB-COST edge, the SAME signature as arc 0.

**Verdict: FAIL.** Family dead for deployment.

**Threads / what didn't help.** Cross-sectional momentum as a TRADE-LEVEL directional long is closed (the
discriminator is post-entry, not in the cross-sectional structure). The extension-ceiling refinement is
closed (non-orthogonal to the falsified dispersion axis). Temporal/regime conditioning of a naive long is
closed. OPEN (future candidate arc, NOT a rescue of this one — council's reframe): use cross-sectional rank
to select WHICH pairs/universe to run a *different* entry on, or harvest portfolio diversification from a
decorrelated sub-cost edge — a DIFFERENT claim (selection/portfolio, not trade-level direction) needing its
own ex-ante population. Extends arc 0's portfolio thread.

**Carry-forward lessons (candidate for LESSONS.md):** (1) cross-sectional relative strength adds NO
separable entry-time edge on H4 majors (AUC≈0.50, same as arc 0's single-pair features) — the dry zone is
wider than single-pair structure; (2) a dispersion INVERSION (worst performance in the clearest-trend
regime) is a falsification, not a regime to filter — don't sweep to rescue it; (3) WHEN-context
(session/DoW/vol) does not condition a naive long-capture edge on H4 majors; (4) two independent long
families (arc 0 pullback, arc 1000 XS-momentum) now share the SAME real-but-sub-cost signature (beat
random, fail costs) — evidence the binding constraint on H4-major longs is the cost/SL-first hurdle against
a ~coin-flip directional base, not the specific entry construction.

**Tooling:** built + registered the random-entry NULL baseline (`discovery/tools/null_entry_baseline.py`,
TOOL_REGISTRY BUILT) — the protocol's first expected BUILT tool; mask randomization only, scoring stays
canonical (`ArcFoldRunner`). Reusable by all future arcs.

**FLAGS (code not merged):** none requiring the canonical core. Signal + drivers in scratch `_disco_work/`
(reproducible from the arc doc).

### arc_1001

**Volatility-contraction breakout long** (chat 1000–1999). Full record:
[`arcs/arc_1001_contraction_breakout_long.md`](arcs/arc_1001_contraction_breakout_long.md). No council
(cheap-killed at triage; no worthwhile-ceiling fork to evaluate).

**Idea + why.** Arc 0 + arc 1000 both bet on PRICE DIRECTION and failed (direction ≈ coin-flip on H4
majors). The ignored regularity: VOLATILITY clusters (autocorrelated). Idea: after a low-vol coil (ATR
percentile-in-trailing-100 < 0.33), enter long on a break above the prior 10-bar high — vol-clustering makes
expansion-timing predictable, and a squeeze-break triggers stops + breakout flow that generic breakouts
(which fade) lack. The coil is the differentiator.

**What happened.** Observation (honest +1R-before-SL capture, IS 2010–2020, unconditional 0.4877):
contraction-breakout **0.4880** (+0.0003) ≈ generic-breakout 0.4863 ≈ base. The coil adds ZERO directional
lift — the hypothesis is FALSIFIED at observation. (coil-only 0.4895, oversold-reversion 0.4844 — both dry.)
Formal cheap-kill: pool 3,186 IS trades (capture 0.4862, matches obs); 3-fold honest triage 2013 −9.41% /
2016 −1.15% / 2019 −5.43% — ALL negative, worst −9.41%, mean −5.33%, no lucky fold. → KILL at triage
(protocol §5d), before full WFO.

**Verdict: FAIL (cheap-kill).** The coil predicts vol EXPANSION but not DIRECTION; a long-only break is still
a directional coin-flip, inheriting the same sub-cost base as the prior two arcs.

**Threads / lessons.** (1) A volatility coil confers NO directional long edge on H4-major breakouts
(contraction-brk = generic-brk = unconditional ~0.487). (2) THIRD independent long family (pullback, XS
momentum, contraction-breakout) with the SAME ~coin-flip-base / sub-cost outcome — strong accumulating
evidence that the limiting factor for an H4-major LONG is the cost/SL-first hurdle against a ~0.49
directional base, NOT the entry construction. Direction-prediction entries (continuation, reversion,
momentum, breakout, squeeze) are systematically dry at this TF/RR. **Implication for future arcs:** try a
structurally different LEVER — payoff/RR asymmetry, portfolio/selection (the open XS-as-universe thread), a
different timeframe, or a non-directional construction — rather than another H4 directional-long entry
trigger. (Did NOT need the null baseline — an all-negative triage is decisive; the null is for distinguishing
a sub-cost edge that survives to a full WFO.)

**FLAGS (code not merged):** none. Signal + drivers scratch `_disco_work/` (reproducible from the arc doc).

### arc_1002

**D1 timeframe directional long (daily trend-following)** (chat 1000–1999). Full record:
[`arcs/arc_1002_d1_trend_following_long.md`](arcs/arc_1002_d1_trend_following_long.md). No council (cheap-kill).

**Idea + why.** 3 H4 directional longs failed identically (coin-flip base). Lever pivot = TIMEFRAME: daily
trends are classically more persistent. Test whether the coin-flip base is H4-specific or general.

**What happened.** D1 observation (honest +1R-before-SL LONG capture, IS 2010–2020): unconditional **0.4908**
(≈ H4 0.4877); uptrend +0.0000, momentum-breakout +0.0051, pullback −0.0013, oversold +0.0087 (0.4996) — NO
trigger lifts the D1 base above coin-flip, none cross-pair consistent. The directional base is TIMEFRAME-
INVARIANT. Apparatus check on the textbook case (D1 Donchian-20 breakout-in-uptrend, 831 trades, capture
0.4838): 3-fold triage 2013 −1.95% / 2016 −5.72% / 2019 −4.71% — all negative, mean −4.13% → cheap-kill.

**Verdict: FAIL (cheap-kill).** Changing TF does not escape the constraint.

**Threads / lessons.** (1) The FX-major directional-long coin-flip base is TIMEFRAME-INVARIANT (D1≈H4); even
textbook daily trend-following is sub-cost. (2) FOUR independent directional-long attempts now FAIL the same
way (pullback H4, XS-mom H4, contraction-brk H4, trend-following D1) — the approach is comprehensively dry
across entry construction AND timeframe. **STRONG STEER: stop testing directional-long entries on majors;
change a more fundamental lever** — instrument universe (less-efficient CROSSES; the backup has ~28 pairs),
portfolio/selection of decorrelated sub-cost signals (XS-as-universe thread), or a non-directional
construction. Another major-directional-long entry is very likely wasted compute.

**FLAGS (code not merged):** none. Signal + drivers scratch `_disco_work/`.

### arc_1003

**Cross trend-momentum long (instrument-universe lever)** (chat 1000–1999). Full record:
[`arcs/arc_1003_cross_trend_momentum_long.md`](arcs/arc_1003_cross_trend_momentum_long.md). No council (cheap-kill).

**Idea + why.** 4 major-directional-longs failed (coin-flip base). Lever = INSTRUMENT UNIVERSE: crosses
(JPY + commodity crosses) carry rate diffs / trend more than the hyper-efficient majors. Test if the cross
directional base clears coin-flip.

**What happened.** Cross observation (H4, honest +1R-before-SL LONG capture, IS 2010–2020): unconditional
cross base **0.4712** — WORSE than majors (0.4877); every cross < 0.50. Momentum-breakout lift +0.0270 (to
0.4982) — the run's LARGEST conditioning effect, but still coin-flip gross. Apparatus check (cross Donchian-20
breakout-in-uptrend, 4,078 trades, capture 0.4809, **mean final_r +0.1021 gross — crosses DO trend**): triage
2013 +1.43% / 2016 −8.15% / 2019 −9.75% — worst −9.75%, mean −5.49% → cheap-kill.

**Verdict: FAIL (cheap-kill).** Crosses trend (positive gross drift) but the WIDER cross spreads eat the thin
edge → net sub-cost.

**Threads / lessons.** (1) Crosses are NOT a better directional-long universe (base 0.4712 < majors 0.4877).
(2) **Binding constraint sharpened to EDGE < COST**: on crosses gross drift is POSITIVE (+0.10R/trade) yet
the wider spreads consume it; on majors gross edge ≈ 0. Both net sub-cost — the hurdle is finding an edge
LARGE enough to clear spread+commission+slippage+SL-first, not finding *a* directional edge. (3) FIFTH
directional-long FAIL across entry/TF/universe → **change the MECHANISM**: non-price-direction (calendar/flow
— turn-of-month rebalancing, NOT yet tested; arc 1000 covered hour/day-of-week but not day-of-month), or a
construction whose gross edge clears cost.

**FLAGS (code not merged):** none. Signal + drivers scratch `_disco_work/`.

### arc_1004

**Cross-trend exit/cost engineering (EXIT lever)** (chat 1000–1999). Full record:
[`arcs/arc_1004_cross_trend_exit_engineering.md`](arcs/arc_1004_cross_trend_exit_engineering.md). No council.

**Idea + why.** arc 1003 cross trend = +0.10R gross drift but sub-cost (EDGE<COST). Hypothesis: the
+1R-partial-runner CAPS the runner; a let-it-run exit captures more of the +drift per winner → higher
R-per-trade amortizes the per-trade cost. Test reasoned exit variants on the SAME cross trend signal.

**What happened.** Triage across exits (3 folds): baseline partial+runner worst −9.75%/mean −5.49%; wide-trail
let-it-run worst −9.75%/mean −4.07%; standard trail worst −11.45%/mean −5.16%; 3R target worst −11.71%/mean
−4.73%. Let-it-run/asymmetric IMPROVE the trending year (2013 +1.43%→+4.75–6.43%) but leave choppy folds
(2016/2019) deeply negative. No exit flips net-positive → cheap-kill.

**Verdict: FAIL (cheap-kill).** Exit engineering does NOT beat EDGE<COST.

**Threads / lessons.** (1) Exits can't rescue the cross trend — arc 0's "exits don't rescue a coin-flip entry"
extends to a POSITIVE-drift instrument: thin +drift + coin-flip split + per-trade spread = net-negative
regardless of harvest. (2) EDGE<COST is an ENTRY/COST problem, not an exit problem — the fix must raise
per-trade gross edge or cut per-trade cost (frequency), not redistribute the exit. (3) Cross-trend family now
CLOSED across entry + all exit structures. Next: change the MECHANISM — calendar-flow (turn-of-month) tested
with the CORRECT metric (mean forward DRIFT; +1R-before-SL is blind to small drifts) and correct long-only
USD-pair-structure handling.

**FLAGS (code not merged):** none. Driver scratch `_disco_work/arc1004_exits.py`.

### arc_3000

**Mean-reversion long on less-efficient / coupled crosses** (chat 3000–3999, FIRST continuous
arc of the 3000s range). Full record: [`arcs/arc_3000_reversion_long_crosses.md`](arcs/arc_3000_reversion_long_crosses.md).
No council (cheap-kill at triage; no reachable ceiling).

**Idea + why.** Fresh eyes; honest-era corpus = arcs 0 + 1000/1001/1002 (1003 landed on main mid-arc). Four
(then five) directional-long arcs share one signature: ~0.49 directional base, beats random, fails costs. Arc
1002's steer: change a fundamental lever — instrument universe (crosses), portfolio/selection, or a
non-directional construction. I attacked TWO untouched axes at once: (1) **instrument universe** — crosses are
less efficient than the hyper-arbitraged majors, and the tightly-coupled ones (EUR/GBP, EUR/CHF, AUD/NZD,
NZD/CAD…) behave like spreads between linked economies; (2) **mean-reversion** — the FIRST non-continuation
mechanism in the programme (all prior arcs bet on continuation/trend), and coupled crosses are where reversion
should be strongest. Long-only forced by the apparatus → "reversion" = buy the oversold/stretched-down side.

**What happened.** Observation #1 (honest +1R-before-SL LONG capture, all 28 pairs, IS 2010–2020): the base is
**instrument-invariant** — MAJOR 0.4859, COUPLED 0.4742, TREND_X 0.4719; nothing ≥ 0.50, crosses *worse* than
majors. z-decile buckets show NO monotone reversion lift as price stretches down on any group; the faint
high-capture cells sit at *positive* z (the already-dry continuation axis). Observation #2 (fail the BEST
reversion version: RSI/z/Bollinger/consec-down/big-drop × two barrier scales, on the 9 most-coupled crosses):
only **RSI<25** lifts non-trivially, and it peaks at **0.4907** (still < 0.50), thin (~30 fires/pair/yr), and
only 6/9 pairs lift. Characterize: pool 1,009 IS trades, capture **0.4936** (sanity-matches obs), mean final_r
**−0.14R**, 80% hard_sl. Cheap kills: pool floor PASS (1,009); oracle ceiling SKIPPED (capture<0.50 ⇒ no
reachable upside; a high oracle ceiling there is the Arc-0 hindsight trap — ritual not rigor); 3-fold honest
triage (A1, SL=2·ATR, 1% reset-floor, exposure 1/pair 2/ccy, `sl_partial_close_1r_runner_trail` = reversion's
best banking exit, FundedNext costs ON, SL-first; OOS 2013/2016/2019): **−20.22% / −5.17% / −12.89% → worst
−20.22%, mean −12.76%, 3/3 negative → KILL at triage.**

**Verdict: FAIL (cheap-kill).** Mean-reversion long on coupled crosses is not deployable. Reason reached
twice: the directional-long base is instrument-invariant (crosses ≤ majors, all sub-0.50), and the reversion
tilt — best version, best instruments — never crosses 0.50 (real but sub-cost). Buying oversold dips does NOT
catch the positive gross drift crosses have (mean final_r −0.14R), so reversion is gross-coin-flip *and*
cost-bled.

**Convergence with arc 1003 (landed mid-arc).** The 1000s chat independently hit crosses from the
*trend-momentum* angle and found the identical cross base **0.4712** < majors — two chats, two cross signals,
one conclusion: the instrument-universe lever is **closed** for a directional long. Arc 1003 sharpened the
constraint to **EDGE < COST** (crosses trend gross-positive +0.10R/trade but wider spreads eat it); my
reversion case is the complement (gross-coin-flip *and* cost-bled). Together arcs 1003 + 3000 close BOTH
directional mechanism families (continuation AND reversion) on crosses.

**Threads / lessons.** (1) The FX directional-long coin-flip base is **INSTRUMENT-INVARIANT** (28-pair scan;
crosses worse, not better) — the "less-efficient crosses" steer is closed, the way arc 1002 closed the
timeframe steer. Independently corroborated by arc 1003. (2) **Mean-reversion confers no deployable long
edge** — first non-continuation mechanism tested, lands sub-cost too (RSI<25 best, ~0.491 < 0.50). (3) **SIX
directional-long arcs now fail identically across BOTH mechanism families, two timeframes, the full 28-pair
universe, two chats** → overwhelming evidence the constraint is structural (gross edge too small × cost ×
SL-first), not the entry. (4) The excursion-banking partial/runner exit does not rescue a sub-0.50 entry
(re-confirmed). **Surviving steers (both away from trade-level price-direction; either is a fresh arc, not a
rescue):** (a) non-price-direction / calendar-flow (turn-of-month — arc 1003's steer, not yet tested), (b)
portfolio/selection of decorrelated sub-cost edges (mine; subsumes arc 0/1000's portfolio thread).
**Coordination:** arc 1003 named the calendar/flow steer, so the 3000s range should prefer
portfolio/selection to avoid collision.

**FLAGS (code not merged):** none requiring the canonical core. Signal + drivers scratch `_disco3_work/`
(reproducible from the arc doc). No reusable experiment tool needed (null baseline not required — an
all-negative triage is decisive).

### arc_2000

**Trend-following long via full-size convexity harvest** (chat 2000–2999, first continuous arc). Full
record: [`arcs/arc_2000_trend_convexity_harvest.md`](arcs/arc_2000_trend_convexity_harvest.md). No
council (cheap-kill; falsified at observation).

**Idea + why (fresh insight, NOT "arc 1002 again").** The four prior arcs all *screened* on +1R-before-SL
*capture* — a WIN-RATE statistic, structurally blind to a low-win-rate / fat-tailed (convex) payoff (the
time-series-momentum profile, edge in the right tail not the hit rate). And every engine triage used the
50%-partial exit, which CAPS the right tail at half size. So the prior failures might be a measurement
artifact. Untested question: does a trend entry have a fat harvestable right tail, and does a FULL-SIZE
tail-preserving trailing exit (`sl_plus_trailing_atr`: −1R floor, uncapped upside, 1R-from-peak trail)
bank enough of it to clear costs? Entry = Donchian breakout long (canonical TSMOM/trend entry).

**What happened.** OBSERVATION (canonical pool MFE distribution, H4 majors, IS 2010-2020, hold 360,
SL=2·ATR) — built a periodic (time-random) long base and Donchian breakouts (N=20/55/120 ±SMA200). The
**fat tail is real but the trend entry does NOT create it**: periodic base already P(mfe≥5R)=.147,
P(mfe≥8R)=.077, meanMFE 2.46R; Donchian-120 lifts only to .165/.088/2.69 — trivial. The right tail is a
generic property of being long a vol-clustering FX major at any time, not trend-selected. cap+1R≈.49
everywhere too → win-rate lens and convexity lens AGREE: trend entry adds ~no separable edge on either.
Hypothesis falsified at observation. Cheap-kill triage (3 IS folds 2013/2016/2019, honest engine, costs
ON) on the untested full-size trails: `sl_plus_trailing_atr` worst −13.97% / mean −9.50% / 3-of-3 neg
(WORSE than the 50%-partial's −7.14% — the per-winner 1R give-back + full-size whipsaw losses dominate);
`sl_plus_trailing_swing` mean −0.04% but 2/3 neg with a +20.81% thin (n=23) fold and a −19.86% / DD-20%
blowup (regime-luck, the FX-majors-only trend signature); periodic null + trailing_atr −11.00% (donchian
−9.50% barely beats it). All N on all-folds-positive.

**Verdict: FAIL (cheap-kill).** Trend-following-long-via-convexity-harvest is not deployable on H4 majors.

**Convergence with arc 1004 (landed on main mid-arc).** The 1000s chat independently engineered exits
(let-it-run / 3R / wide-trail) on its CROSS trend-momentum signal and found no exit flips net-positive —
"EDGE<COST is an ENTRY/COST problem, not an exit problem." My arc is the MAJORS + full-size-trailing
complement and reaches the identical verdict from the convexity angle. Two chats, two universes, two exit
families: **payoff/exit engineering cannot rescue a sub-cost trend.** And my MFE observation explains WHY —
the harvestable tail is generic (a random long has it too), so there is no trend-specific convexity for any
exit to harvest.

**Threads / lessons.** (1) The fat right tail in FX-major longs is generic, not trend-selected — both the
win-rate lens (+1R≈.49) and the convexity lens (MFE tail) say the trend entry adds nothing; the prior
arcs' win-rate screen was NOT hiding a convexity edge. (2) No full-size trailing harvest rescues a
~coin-flip long (atr trail worse than partial; swing trail is regime-luck/blowup, not all-folds edge) —
converges with arc 1004 on crosses. (3) With arcs 1003/1004/3000, the count is now SEVEN+ directional-long
FAILs sharing the EDGE<COST signature across entry construction, timeframe (H4/D1), the full 28-pair
universe (majors+crosses), BOTH mechanism families (continuation+reversion), AND now payoff/exit structure
— the constraint is structural. (4) Surviving steers (coordinated): calendar-flow/non-price-direction
(1000s), portfolio/selection of decorrelated sub-cost edges (3000s). For the 2000s range, the open lane is
a construction that **raises per-trade gross edge or cuts per-trade cost/frequency** enough to clear the
EDGE<COST hurdle (not another entry/exit re-cut), OR a genuinely different instrument-structure (e.g.
relative-value/spread between coupled pairs) — to be decided fresh at arc 2001's observation.

**Tooling:** built + registered `discovery/tools/trend_entry_signals.py` (`DonchianBreakoutLongSignal`,
`PeriodicLongSignal`) — reusable trend-entry + random-base experiment tools (mask + ATR geometry only;
scoring stays canonical). TOOL_REGISTRY BUILT updated.

**FLAGS (code not merged):** none requiring the canonical core. Drivers in scratch `_disco2000_work/`
(reproducible from the arc doc).

### arc_1005

**Turn-of-month USD-long (calendar-flow mechanism)** (chat 1000–1999). Full record:
[`arcs/arc_1005_turn_of_month_usd_long.md`](arcs/arc_1005_turn_of_month_usd_long.md). No council (cheap-kill).

**Idea + why.** 6 prior families fail on EDGE<COST (entry/cost problem, not exit). Steer: change the
MECHANISM → non-price-direction CALENDAR FLOW (month-end rebalancing). Tested with the CORRECT metric (mean
forward DRIFT in ATR; +1R-before-SL is blind to small drifts), PER PAIR (month-end USD flow pushes XXXUSD vs
USDXXX oppositely).

**What happened.** D1 turn-of-month (last 2 + first 3 trading days), mean forward-5d drift: a REAL USD-strength
signal — EURUSD −0.18 ATR, GBPUSD −0.26 (18% years pos), USDCHF **+0.177** (73% years pos), concentrated in
EUR/GBP/CHF. Long-only-exploitable via USDCHF. Built a signal-class TIME-EXIT predicate (A1Config.time_exit_bars
is NOT wired into the Order — FLAG); long at 3rd-to-last trading day, 6-bar time exit, SL=2ATR. Triage: USDCHF
2013 +0.68% / 2016 −2.78% / 2019 −0.27% (worst −2.78%, mean −0.79%, tiny DDs 1–3%, mean final_r ≈ break-even
gross); USDXXX basket worst −3.55%, mean −1.61%. Both sub-cost → cheap-kill. (USDCHF = 1-of-8 cherry-pick.)

**Verdict: FAIL (cheap-kill).** The drift is REAL but too small (~0.18 ATR/5d) to clear FundedNext cost +
the ~31% 2ATR stop-out. EDGE<COST.

**Threads / lessons.** (1) A real turn-of-month USD-strength drift exists (USDCHF +0.177 ATR/5d) but is
SUB-COST — the FIRST non-directional mechanism, and EDGE<COST holds for it too → the cost hurdle is
**mechanism-general**. (2) SIX families across all levers (entry/TF/universe/exit/mechanism) now say the
realizable gross edge of simple long-only FX signals (~0.1–0.2 ATR or coin-flip capture) is below the
FundedNext cost hurdle. A deployable long needs a MUCH larger per-trade gross edge or a different cost regime.
(3) month-of-year seasonality = 11-sample noise. (4) Built reusable `make_time_exit_predicate` (BUILT) +
`build_null_signal_evaluation` (arc 1000) — calendar/hold and soundness tooling now in place.

**FLAGS (code not merged):** `A1Config.time_exit_bars` defined but NOT wired into the Order by A1 (worked
around with the signal-class time-exit predicate; config-level time exit silently no-ops — human-gated fix).

### arc_3001

**Drift-lens scan — is the directional-long death a capture-metric artifact?** (chat 3000–3999). Full record:
[`arcs/arc_3001_drift_lens_scan.md`](arcs/arc_3001_drift_lens_scan.md). No council (cheap-kill).

**Idea + why.** A *methodological* probe, not a new family. Six directional-long arcs (0,1000–1003,3000) all
judged entries with **+1R-before-SL capture**, which arc 1004 flagged as **blind to small persistent drift**
(arc 1003 saw +0.10R gross under sub-0.50 capture). Before declaring the directional-long space dead, re-scan
it with the **correct mean forward-DRIFT lens**. Distinct from the 1000s chat: they take the calendar/time
drift axis (month-end); I take price/volatility/structure. (Reconsidered my own arc-3000 "portfolio/selection"
suggestion and rejected it as **premature** — you can't diversify net-negative components positive; need a
net-positive component first, and none exists.)

**What happened.** `fwd_drift_12 = (mid_close[s+12] − entry@s+1 ask)/(2·ATR[s])`, R units, gross, 28 pairs, IS
2010–2020. **NO instrument × condition clears the ~0.05–0.10R cost hurdle.** By group (uncond / up-spike /
dn-spike / uptrend / mom_hi / low-vol): MAJOR −0.019/+0.007/−0.072/−0.031/−0.032/−0.004; COUPLED
−0.083/−0.101/−0.138/−0.074/−0.101/−0.131; TREND_X −0.012/**+0.023**/−0.014/−0.012/−0.005/−0.021. Best cell
anywhere = TREND_X post-up-spike **+0.023R gross** (< cost). Confirmatory triage on that best cell
(post-up-spike continuation, 12 trending crosses; pool n=1375, cap 0.4982, gross mean final_r +0.088R): net
**2013 +16.74% / 2016 −28.07% / 2019 −21.01% → worst −28.07%, mean −10.78%, 2/3 neg → KILL.**

**Verdict: FAIL (metric-robust closure).** The directional-long failure is NOT a capture-metric artifact — the
drift lens agrees with the capture lens. No cost-clearing forward drift exists in price/vol/structure on any
instrument; the lone positive cell is sub-cost and regime-fragile (the arc-1003/1004 momentum-on-crosses
signature). Directional price-structure long is now closed under BOTH metrics.

**Threads / lessons.** (1) **Directional-long death is METRIC-ROBUST** (capture AND drift; 28×6 scan, best
gross drift +0.023R < cost) — closes the arc-1004 "blind-to-drift" loophole. (2) **Coupled crosses drift
NEGATIVE for longs** (−0.08R/12bars; worse oversold/low-vol) — independent re-confirm of arc 3000 from the
drift angle. (3) Faint trending-cross momentum is regime-fragile + net-negative — re-confirms 1003/1004. (4)
**Portfolio/selection is premature/empty** until a net-positive component exists (none does). **Surviving steer
(singular, non-price-structure):** flow/calendar (1000s chat's axis) or a genuinely novel construction — arc
3002 needs a *generatively different* idea (candidate for a LIGHT generative council at the idea-fork, §5b),
since the 3000s side has now closed the price-structure directional space.

**FLAGS (code not merged):** none. Scan + signal + drivers scratch `_disco3_work/`.

### arc_3002

**Volume-confirmation long — the last untouched data column** (chat 3000–3999). Full record:
[`arcs/arc_3002_volume_confirmation_long.md`](arcs/arc_3002_volume_confirmation_long.md). No council
(no reachable ceiling once full IS unmasked the lucky triage).

**Idea + why.** Every arc conditioned on price/vol/structure/timeframe/calendar; **none used `volume`** — yet
the panel carries a tick-count `volume` column (activity proxy). Volume-confirmation (price moving on high
volume = informed = continuation) is the last untouched data axis. Re-used arc 3001's forward-drift lens with
volume conditionings, 28 pairs, IS.

**What happened.** Group-level volume-conditioned drift (gross R): NO cell clears the ~0.05–0.10R cost hurdle;
best group cell = TREND_X vol-spike(>2.5×) **+0.0475R** (thin n=710). Per-pair flags were scattered single-pair
(GBPJPY lovol_pull +0.092 etc.) = regime-luck, not built. Triaged the best cross-pair cell (vol-spike long, 12
trending crosses; pool n=839, cap 0.478, **gross mean final_r +0.0035** ≈ nil): 3-fold (2013/2016/2019) =
**+11.43 / +11.97 / −16.25 → mean +2.38%, 2/3 pos → NOT deeply negative → proceeded** (first non-negative
triage in 3000s). **Full IS WFO (all 10 folds) unmasked it: 2011−13.6, 2012−0.8, 2013+11.4, 2014−11.5,
2015−3.6, 2016+12.0, 2017−9.9, 2018−12.9, 2019−16.3, 2020+6.1 → worst −16.25%, mean −3.91%, 7/10 NEGATIVE →
FAIL.** The triage was lucky (sampled 2013 & 2016, the two most momentum-friendly years).

**Verdict: FAIL.** Volume-confirmation does not rescue the directional base. **Diagnosis (unifying): volume
predicts move MAGNITUDE, not DIRECTION** — exactly arc 1001's volatility-contraction finding re-derived on the
volume axis. Both volume & volatility say a *bigger* move is coming, not *which way*; a long-only bet on them is
a higher-variance directional coin-flip → regime-dependent, net sub-cost. Direction (≈0.49) is the binding
constraint regardless of the magnitude predictor.

**Threads / lessons.** (1) **Volume = magnitude, not direction** (unifies arc 1001); the last untouched data
column is now closed. (2) **METHODOLOGICAL — the 3-fold triage {2013,2016,2019} over-samples momentum-friendly
years** (2013/2016 are positive for almost every momentum-ish signal — cf. arc 1000 +11.98% 2013, arc 3001
+16.74% 2013); a regime-dependent signal can show a *positive* triage mean yet be 7/10 neg on full IS. **A
non-negative 3-fold triage is necessary but not sufficient — confirm with full IS before any
diagnosis/council investment** (or include a chop year 2014/2017/2018 in the triage). (3) Single-pair drift
flags = regime-luck (GBPJPY +0.092R), not edge — keep the cross-pair-robustness discipline. **Surviving steer:**
price/volume/structure/timeframe directional space is now systematically closed from the 3000s side and
calendar is weakening (1005) → **arc 3003 is the right point for a LIGHT generative council (§5b)** — a genuine
idea-fork — or a meta-pivot (is this apparatus capable of any edge, and what must change?).

**FLAGS (code not merged):** none requiring the canonical core. Methodological note only: the chat-convention
triage year-set {2013,2016,2019} is momentum-biased (lesson 2); canonical `build_v3_folds` is unaffected.
Scan + signal + drivers scratch `_disco3_work/`.

### arc_3003

**Regime detection for momentum — is the trending regime detectable in advance?** (chat 3000–3999). Full
record: [`arcs/arc_3003_regime_detection_momentum.md`](arcs/arc_3003_regime_detection_momentum.md). No council.

**Idea + why.** The central unsolved question after 12 arcs: momentum drift is real but regime-dependent (good
trending years 2013/2016/2020, bad chop — arcs 1003/3001/3002). If the trending regime were detectable IN
ADVANCE, trade momentum only then. Tested with Kaufman Efficiency Ratio (causal trending-ness ∈[0,1]) — the
most direct "is this trending now" measure, not yet tried (arcs 1000/1001 tried dispersion/vol-level regimes).

**What happened.** Bucketed forward 12-bar drift by ER60 on trending crosses (momentum entries above SMA50):
chop→trend drift = −0.010 / −0.001 / +0.011 / +0.019 / **−0.061** (top ER bucket). **REGIME INVERSION** — the
strongest-trending regime has the MOST NEGATIVE forward drift (strong trends REVERT, don't persist); the naive
"trade momentum when trending" is actively wrong; the mid-ER band is mildly positive but sub-cost. Fail-the-
best-version: gated momentum to the mid-ER sweet spot [0.09,0.23] (best band). Pool n=9250, capture 0.478,
**gross mean final_r −0.0547** (the +0.019R *raw* drift evaporates to NEGATIVE SL-honest expectancy). Full IS
WFO (skipped the lucky 3-fold triage per arc 3002's lesson): 2011−63, 2012+11, 2013+32, 2014−61, 2015−19,
2016−31, 2017−4, 2018−30, 2019−45, 2020−43 → **worst −63.11%, mean −25.30%, 8/10 neg, DDs to 66% →
catastrophic FAIL.**

**Verdict: FAIL.** The trending regime is NOT detectable in advance as a cost-clearing momentum filter; the
strongest-trend regime inverts (reverts). Closes the regime-detection lever — the central remaining hope for
rescuing the faint momentum drift.

**Threads / lessons.** (1) **Trending regime NOT detectable for momentum** (ER inverts: strong trends revert
−0.061R). **Regime-conditioning has now failed across THREE measures — dispersion (1000), vol-level (1001),
trending-ness/ER (3003)** → no exploitable regime structure for a long-only FX directional bet; extremes
mean-revert, the middle is random. (2) **METHODOLOGICAL: raw forward drift OVERSTATES SL-honest expectancy**
(+0.019R raw → −0.055R SL-honest; take-the-loss kills slow drifts). The drift lens (arc 3001) is an OPTIMISTIC
cheap pre-filter; a sub-+0.05R drift cell is ~guaranteed SL-honest-negative. Only the engine verdict counts.
(3) Wide regime band fires too often (9250 trades) → frequency amplifies a negative edge (−25%/66%DD).
**Surviving:** the 3000s side has now closed directional entries (both metrics), instrument universe, volume,
and regime detection; combined with 1000s/2000s (timeframe, exits, calendar, convexity) the
price/volume/structure/regime directional space is exhausted. **Arc 3004 → LIGHT generative council (§5b):** I
am genuinely at a stuck-point/idea-fork; convene the council for generative perspectives / to pressure-test a
meta-pivot (is this apparatus capable of any edge, and what structural change is required?) before another
lone guess. Portfolio/selection still premature (no net-positive component).

**FLAGS (code not merged):** none. Scan + signal + drivers scratch `_disco3_work/`.

### arc_1006

**Weekend gap-down-fill long, JPY crosses (generative-council idea)** (chat 1000–1999). Full record:
[`arcs/arc_1006_weekend_gap_fill_long.md`](arcs/arc_1006_weekend_gap_fill_long.md). Council transcript (LIGHT
generative, §5b): [`results/arc_1006_weekend_gap_fill/council_transcript.md`](results/arc_1006_weekend_gap_fill/council_transcript.md).

**Idea + why.** Directional + calendar closed fleet-wide (arc 3001 flagged a §5b generative council). Convened
it (5 lenses); synthesized the WEEKEND GAP-FILL (untouched by arcs 0–3001; price-only, long-only): big weekend
gap-downs tend to FILL (weekend positioning/liquidity overhang reverts Monday). Rejected SL-geometry (1/W cancels
the sign), triangular divergence (dead at H4), deferred the Devil's null-confirmation (premature).

**What happened.** Observation: big gap-downs (<−0.5 ATR) fill (+0.20 ATR pooled fwd); **JPY crosses strongest**
(EURJPY +0.48 ATR, capture 0.54 — biggest gross edge of the run); moderate gap-downs CONTINUE down. Best version:
long a JPY cross on a >0.5-ATR weekend gap-down, 24-bar time exit, SL=2ATR. Pool 396 trades, **mean final_r
+0.1127 gross**. Triage 2013 −2.06% / 2016 +3.20% / 2019 +7.45% (2/3 POSITIVE, mean +2.86%, tiny DDs) → PROCEED.
Full WFO: IS mean **+0.69% (FIRST mean-positive IS of the run)**, beats random null (−0.60%), DDs 2–9% — BUT
all-folds-positive NO (5/10 IS neg worst −6.79%; OOS 4/6 neg worst −4.13%). One reasoned refinement (uptrend
filter close>SMA50): negatives 5/10→4/10 but still not all-folds-positive AND over-thinned (min 5 trades/fold).

**Verdict: FAIL** (not all-folds-positive) — but **categorically the most promising result of the run**: a REAL,
mean-positive, beats-random, low-DD edge that is merely FOLD-FRAGILE (JPY-cross weekend gaps are tail-event-
timing-dependent), not a coin-flip.

**Threads / lessons.** (1) Weekend gap-FILL is REAL on JPY crosses — first net-positive long edge; fold-fragile,
large-gap-only (moderate gap-downs continue down). (2) **RE-OPENS the portfolio thread**: arc 3001 noted you
can't diversify net-NEGATIVE components positive — but this is the FIRST net-POSITIVE (fold-fragile) component;
a decorrelated combination of ≥2 such edges could plausibly reach all-folds-positive (diversification cuts
fold-variance, preserves positive mean). **HIGH-VALUE fleet steer: hunt for a 2nd net-positive edge to combine.**
(3) Generative council earned its keep (steered off the exhausted rut to the first promising lead) and queued
untested ideas: spread-tier gating, vol/cost-ratio conditioning, Asia→London timing, Devil's null-confirmation.

**FLAGS (code not merged):** none. Reused BUILT `make_time_exit_predicate`; drivers scratch `_disco_work/`.

### arc_2001

**Weekend-gap-fill long, MAJORS** (chat 2000s). Full record:
[`arcs/arc_2001_weekend_gap_fill_long.md`](arcs/arc_2001_weekend_gap_fill_long.md). HEAVY diagnosis council:
[`results/arc_2001_weekend_gap_fill_long/council_transcript.md`](results/arc_2001_weekend_gap_fill_long/council_transcript.md).

**Idea + why.** EDGE<COST exhausts price-DIRECTION; the price-only corpus leaves price-STRUCTURE / time-
STRUCTURE. Untouched axis: weekend/weekly-open GAPs — discrete, sometimes-large moves with a documented
partial-fill tendency, attacking EDGE<COST from the move-size side. Observe first (drift metric).

**What happened.** OBSERVATION (H4 majors, IS 2010-2020): a clean, monotone, SYMMETRIC gap-fill — DOWN
>1·ATR weekly-open gaps drift +0.45 ATR/~1day (frac+ 0.59); UP >1·ATR gaps drift −0.57 ATR (frac+ 0.36).
The ONLY conditioning variable in the programme to push the directional hit-rate clearly >0.50 with sizable
drift. Long-only tradeable side = buy big down-gaps. Characterize (threshold×SL grid): the down-gap's
ADVERSE continuation is large (median MAE −1.1 to −1.3R, P(MAE≤−0.5R)=0.77-0.89) — the honest i+1 entry
(after the gap bar) lands INSIDE the continuation, so honest +1R-before-SL capture only 0.45-0.47 and gross
meanFinalR is positive but small (+0.07..+0.18R). 3-fold triage looked near-break-even (favorable-fold
LUCK), but the FULL 10-fold IS WFO is mean −1 to −2.5%, 6-7/10 folds negative for every threshold/SL/exit.

**Diagnosis + HEAVY council.** Diagnosis: real edge, but the i+1 entry sits in the adverse continuation and
the small surviving edge is cost-eaten. Council (5 lenses + 3 reviewers + chairman) → **KILL**: (1) the
delayed/confirmation-entry refinement is a contamination trap (invented+tuned on IS, validatable only by
spending pristine OOS); (2) the sample can't support the all-folds-positive judge even if real (~6-8
trades/OOS-year filtered → coin-flip); (3) the exciting 0.59/monotone drift is measured from the
UNTRADEABLE gap-bar open (hindsight framing); the tradeable i+1 edge is the small one costs eat. "The
mechanism survives; the trade does not." CC committed (no override). **OOS deliberately NOT touched** (no
IS-clearing system + preserve the holdout).

**Verdict: FAIL** (real edge, uncapturable) — the 9th EDGE<COST result, but the first whose binding
constraint is CAPTURABILITY/frequency/long-only-access, not absence of edge.

**Convergence with arc 1006 (chat 1000s, landed mid-arc).** 1000s independently found weekend gap-down-fill
(generative council) on JPY CROSSES — the run's MOST promising result (FIRST mean-POSITIVE IS edge +0.69%,
beats random) but fold-fragile FAIL. **Two chats, two universes, converged: weekend gap-fill is the realest
edge found.** CROSS-FINDING: mean-POSITIVE on JPY crosses, mean-NEGATIVE on majors → **JPY crosses are the
better gap universe; majors too efficient.** My distinct adds: (1) majors are the weaker gap universe (don't
pursue gap-fill there); (2) **FLAG-1 the long-only apparatus blocks the STRONGER short side** (UP-gaps:
−0.57 ATR drift, 0.64 accuracy — the better edge, structurally untradeable; short support = human-gated
code, NOT self-merged); (3) the OOS-pristine + n-too-thin-for-the-judge discipline.

**Threads / lessons.** (1) Weekend gap-fill = the first real, monotone, mechanistically-clean edge (weekend-
illiquidity repricing toward prior close), fails on CAPTURABILITY not edge. (2) A 3-fold triage can be
favorable-fold LUCK — run the full IS WFO before believing it (the same 2013-regime-luck that burned
1000/1003). (3) OOS-pristine discipline: when the only rescue is an IS-tuned refinement, FAIL-on-IS +
preserve-OOS, don't burn the holdout (HEAVY council). (4) **HIGH-VALUE open thread (shared w/ 1006's steer):
gap-fill on JPY crosses is net-POSITIVE-but-fold-fragile; a decorrelated COMBINATION of ≥2 net-positive
gap-type edges may reach all-folds-positive — but the single most promising gap signal (the UP-gap short) is
blocked by long-only.** The 2000s lane next should AVOID the majors gap-fill (done) and the 1000s
gap-portfolio thread (theirs); a lower-timeframe gap test (more events) or the FLAG-2 upstream diagnostic
(is H4+FundedNext generically hostile to fill/reversion?) are open.

**Tooling:** built + registered `discovery/tools/gap_signals.py` (`WeekendGapFillLongSignal`). TOOL_REGISTRY
BUILT updated.

**FLAGS (code not merged):** **FLAG-1** long-only apparatus blocks the stronger UP-gap short side (canonical-
core change, operator-gated, NOT merged); **FLAG-2** future-arc seed: is H4+FundedNext cost structure
generically hostile to fill/mean-reversion entries? Drivers scratch `_disco2000_work/`.

### arc_1007

**Gap-fill best-version test: exit AT the fill target** (chat 1000–1999). Full record:
[`arcs/arc_1007_gap_fill_exit_at_target.md`](arcs/arc_1007_gap_fill_exit_at_target.md). No council.

**Idea + why.** Develop the run's only net-positive lead (arc 1006 gap-fill, 24-bar exit was arbitrary).
Mechanism-aligned hypothesis: exit AT the gap origin (prior-week close) to take the reversion cleanly. Built
`make_price_target_exit_predicate` (BUILT). One reasoned version, measured IS then OOS.

**What happened.** EXIT-AT-TARGET IS WORSE: IS mean +0.69% (24-bar) → **−1.21%** (exit-at-target), now LOSES
to the random null (−0.60%); not all-folds-positive (6/10 IS neg, OOS 4/6 neg). **Diagnosis:** the gap-fill
edge is the OVERSHOOT — the JPY-cross snapback runs PAST the gap origin and continues; capping at the origin
cut the big winners while keeping full −1R losers → asymmetry flipped negative. The arbitrary 24-bar "let it
run" exit was capturing the overshoot.

**Verdict: FAIL.** Arc 1006's 24-bar version remains the best version of the gap-fill; it is NOT
all-folds-positive in any tested exit.

**Threads / lessons.** (1) The JPY-cross weekend gap-down edge is OVERSHOOT/continuation, NOT fill-to-origin
(re-frames the mechanism: big gap-down → snapback past the prior close). (2) Confirms arc 1004 ON THE POSITIVE
LEAD: exit engineering can't fix fold-fragility (it's tail-event-timing-driven, not exit-driven). (3) The BEST
version of a lead can be the naive one — the "mechanism-aligned" refinement was a worse hypothesis (mechanism
mis-stated). (4) gap-fill stays the portfolio-component candidate (arc 1006 24-bar version); needs a 2nd
net-positive edge. Queued/untested: spread-tier gating, vol/cost conditioning, Asia→London, Devil's
null-confirmation (the decisive closure test as leads narrow).

**FLAGS (code not merged):** none. Built + registered `make_price_target_exit_predicate` (works; just not the
right exit for an overshoot edge). Drivers scratch `_disco_work/`.

### arc_2002

**H1 entry-resolution test of the weekend gap-fill** (chat 2000s). Full record:
[`arcs/arc_2002_h1_gap_resolution.md`](arcs/arc_2002_h1_gap_resolution.md). No council (confirmatory
cheap-kill); reused BUILT `WeekendGapFillLongSignal` on H1.

**Idea + why.** The arc-2001 HEAVY council ruled the gap-fill "uncapturable at H4 RESOLUTION" — the honest
i+1 entry fills 8h after the gap open, deep in the adverse continuation. Entry resolution is the one
UNBLOCKED, non-curve-fit, non-colliding lever (distinct from 1000s' cross-gap-portfolio): on **H1** the i+1
entry is ~2h after the open (~6h earlier), so it should capture the fill the H4 entry misses. Mechanistic
change (finer bars), not a tuned parameter — so the council's contamination/thinness objection doesn't block
it. Is "at H4 resolution" load-bearing, or uncapturable at any resolution?

**What happened.** H1 majors, IS 2010–2020 (H1 loads ~1s/pair from m1 cache). Characterize: H1 capture
**0.450–0.454 ≡ H4 0.45–0.47**; meanMAE **−1.2 to −1.3R ≡ H4**; stop 0.76–0.80. Entering 6h closer changes
NOTHING — because the down-gap continues adversely for hours-to-days, so shaving 6h off a multi-hour
continuation doesn't escape it. Triage (3 folds, comparable thr2.0 SL2.0 partial): +2.43/−2.02/−1.04, mean
−0.21%, 2/3 neg — the same near-break-even-to-sub-cost as H4. → confirmatory cheap-kill.

**Verdict: FAIL (confirmatory).** Entry resolution is NOT the binding constraint. The gap-fill's adverse
continuation is **TIMEFRAME-INVARIANT** (H4 ≡ H1); the long-only fill entry is structurally late at any bar
size. Empirically vindicates the arc-2001 council ("the mechanism survives; the trade does not") — and
complements arc 1007 (1000s: the gap edge is OVERSHOOT not fill, capping at the target is worse) and arc
3001 (intraday down-spikes CONTINUE not revert). The long-side gap-fill is now closed across resolutions
AND exit structures.

**Threads / lessons.** (1) The weekend gap-fill's adverse continuation is TF-INVARIANT (H4≡H1) — entry
resolution was never the lever. (2) Convergent closure of the long-side gap-fill from 4 angles across 2
chats: universe (1006 crosses>majors), exit-at-target (1007 overshoot-not-fill), entry-resolution (2002
TF-invariant), and intraday-analog (3001 spikes continue). (3) The only live gap paths remain the two named
in arc 2001 — neither a long-only price-signal: the cross-universe net-positive PORTFOLIO (1000s' active
thread) and the blocked UP-gap SHORT (FLAG-1, stronger side, human-gated code). (4) **The price-only
long-only H4/H1/D1 space is now comprehensively mapped across 15 arcs / 3 chats with no capturable
EDGE>COST edge found.** Next 2000s arc should weigh a LIGHT generative council (§5b, flagged by arc 3001)
for a genuinely non-price-structure construction, vs documenting the structural blockers as the standing
result.

**FLAGS (code not merged):** none new (carries FLAG-1 long-only-blocks-short-side, FLAG-2 H4-cost-vs-
reversion from arc 2001). No new tool (reused `WeekendGapFillLongSignal(primary_tf="H1")`). Driver scratch
`_disco2000_work/`.

### arc_2003

**Cross-pair relative-value divergence (laggard catch-up long)** (chat 2000s). Full record:
[`arcs/arc_2003_crosspair_divergence.md`](arcs/arc_2003_crosspair_divergence.md). LIGHT generative council
(§5b) summarized inline in the arc doc (no separate transcript — light/generative).

**Idea + why.** 15 arcs closed the obvious price-structure space → genuine dry-log fork → LIGHT generative
council (3 lenses). Two of three converged on **cross-pair relative-value divergence**: long the
under-performing leg of a correlated XXX/USD pair when the synthetic cross is at an extreme (relative-value,
NOT single-pair direction — distinct from arc 3000's absolute RSI-oversold-on-the-cross). The Devil lens'
strongest self-judged call was **CONCEDE** (the binding blocker is the long-only constraint; the gap-fill's
strong side is short). CC synthesized: test the convergent divergence idea; defer the M1 spread-spike idea
(sub-pip reversion vs ~3-pip cost → EDGE<COST).

**What happened.** Drift observation (D1, ex-ante, correlated groups EURUSD↔GBPUSD, AUDUSD↔NZDUSD): the
laggard does NOT catch up. At z<−2 (A under-performed) fwd drift of A = −0.109/−0.046/−0.007 ATR (5/10/20
bars), frac+ ~0.49–0.53; per-group z<−1 fwd10 −0.202 EURvGBP, −0.150 AUDvNZD (laggard keeps lagging). The
catch-up hypothesis is FALSIFIED — relative performance of correlated majors is weak momentum/coin-flip, not
mean-reverting. → cheap-kill at observation.

**Verdict: FAIL (cheap-kill).** No relative-value reversion in FX majors (3rd confirm with 1000/3001).

**Threads / lessons.** (1) No relative-value catch-up in correlated majors — the laggard continues lagging
(pairwise-divergence angle confirms the cross-sectional 1000 + drift 3001 findings). (2) **The generative
council INDEPENDENTLY reached arc 2001's FLAG-1 via its CONCEDE lens**: across 16 arcs/3 chats the one real
edge (weekend gap-fill) is mechanism+direction+magnitude-confirmed and its STRONGER leg is the UP-gap SHORT,
which long-only structurally forbids — **the binding blocker is the constraint, not the search space.** (3) A
generative council on a heavily-mapped space mostly re-surfaces the known answer-space; its value here was
the independent CONCEDE corroboration + cheap in/out on the divergence idea. (4) Low-odds untested seed: M1
bid-ask spread-spike mean-reversion (microstructure liquidity dislocation) — almost certainly EDGE<COST.

**FLAG-1 (REINFORced — most actionable):** add short support (operator/human-gated canonical-core change,
NOT self-merged) to unlock the gap-fill's dominant UP-gap short leg — the single highest-leverage operator
decision the programme has surfaced. **FLAGS (code not merged):** none new. Driver scratch `_disco2000_work/`.

### arc_3004

**The stop confound (council-driven) → clean ESCALATION** (chat 3000–3999). Full record:
[`arcs/arc_3004_stop_confound_and_escalation.md`](arcs/arc_3004_stop_confound_and_escalation.md). Council
transcript: [`results/arc_3004_stop_confound/council_transcript.md`](results/arc_3004_stop_confound/council_transcript.md).
**Operator-facing finding:** [`ESCALATION_apparatus_capability.md`](ESCALATION_apparatus_capability.md).

**Idea + why.** Genuine stuck-point after 3000–3003 closed the price/volume/structure/regime directional space.
Per §5b/§7 convened the LIGHT generative council (`/llm-council-discovery`). The council surfaced a confound I
(and ALL 13 prior arcs) had missed: **every arc used a 2·ATR SL-first stop.** The take-the-loss tax converts
positive gross drift → negative honest expectancy (+0.019R raw → −0.055R, arc 3003), so "directional space is
closed" was confounded with "a tight stop truncates a slow drift." Chairman (commit): investigate-X-first —
re-score the strongest +gross-drift entry with the stop REMOVED (pure time-exit) before escalating; read TWO
numbers (stop effect, cost effect); stop-removed+all-folds-positive → stop was the wall, else → escalate clean.

**What happened.** Clean A/B on the post-up-spike trending-cross entry (arc 3001 pool, gross +0.088R), full IS
WFO, FundedNext costs ON, scored only by MultiPairBacktester. Stop removed via the BUILT
`make_time_exit_predicate` (arc 1005 — reused, not rebuilt; geometry only) + wide 6·ATR sizing-only SL.
BASELINE (2·ATR partial/runner): mean −4.52%, 6/10 neg. STOP-REMOVED time-exit: N=6 mean −2.13%/8neg; N=12
−3.46%/8neg; N=24 −2.77%/7neg; N=48 **+0.62%/5neg, worst −20.4%**. **Removing the stop lifts mean ~4–5pp (the
take-the-loss tax is REAL) but NO horizon is all-folds-positive** — the residual ≈ cost and is regime-dependent
(chop years stay negative regardless of stop). The chairman's "stop-removed but still sub-cost" branch.

**Verdict: FAIL — and the directional space is now CLEANLY closed.** The stop was a contributing ~4–5pp drag,
NOT the binding wall. → **ESCALATION raised** (operator decision, NOT a halt; loop continues): the fixed
apparatus (long-only, single-instrument, per-trade SL/TP, price+volume, FundedNext costs) cannot express a
deployable FX edge — it can only bet DIRECTION, which liquid FX prices to a coin-flip; durable FX edges are
relative-value/carry needing a structural unlock. Ranked unlocks: (1) second simultaneous leg (market-neutral /
cost-mutualising — the only lever not requiring beating 0.50 per-trade), (2) shorting, (3) a real tighter-cost
execution regime (NB must match an obtainable broker or it re-introduces Arc-10 gate-mismatch).

**Threads / lessons.** (1) **The 2·ATR SL was a real ~4–5pp drag but NOT the wall** — removing it (pure
time-exit) lifts mean −4.52%→~0% yet still 5–8/10 folds negative; the deeper wall is gross-drift≈cost +
regime-dependence. Retires "the stop was the wall" with a clean unconfounded test. (2) **Directional space is
closed structurally, not by tuning** → escalation. (3) **TOOLING DISCIPLINE: check TOOL_REGISTRY BUILT FIRST
before writing** — I started to rebuild the time-exit predicate; the Write failed (it existed, arc 1005); I
called the BUILT tool instead. Cross-chat tools compound (reused arc 1005's time-exit). (4) The council earns
its keep at a real stuck-point — it found a confound 4 arcs missed and its measure-then-decide prevented
escalating on confounded data (Arc-10 lesson in research-conclusion space). **Remaining thin in-apparatus
thread:** cross-rate triangulation lag (Mechanism lens) — likely M1/sub-cost, a future cheap-kill candidate.

**FLAGS (code not merged):** **`A1Config.time_exit_bars` defined but UNWIRED** (never consumed by A1; a time
exit must be an `ExitPredicate`). Caused a silent no-op in an early version of this arc's test. Independently
flagged by arc 1005; re-flagged. Fix human-gated (wire it, or remove the dead param). Scratch drivers
`_disco3_work/`.

### arc_3005

**Cross-rate triangulation residual — closing the last in-apparatus thread** (chat 3000–3999). Full record:
[`arcs/arc_3005_triangulation_residual.md`](arcs/arc_3005_triangulation_residual.md). No council (cheap-kill at
observation).

**Idea + why.** The arc-3004 council named ONE remaining concrete in-apparatus idea that is genuinely
*non-directional*: cross-rate triangulation lag (synthetic cross from its two USD legs vs the quoted cross).
Low conviction (Mechanism lens: "likely M1, dislocation may be < the cross spread") but closing it completes the
escalation's coverage. Tested at H4: residual = (quoted_mid − synthetic)/synthetic, bps, contemporaneous closes.

**What happened.** Across 6 triples (EURGBP, EURJPY, GBPJPY, AUDJPY, EURAUD, AUDCAD): residual median ≈ 0
(±0.05 bps); |residual| exceeds the cross spread only 1.3–6.1% of bars (within spread-noise); **forward-6-bar
convergence correlation ≈ 0.00–0.02** (no predictive power). Triangular arb pins quoted=synthetic at H4 closes;
any lag is sub-second/tick — out of apparatus scope and below the cost floor. Nothing to trade.

**Verdict: FAIL (cheap-kill at observation).** No H4 triangulation dislocation. The last concrete
non-directional in-apparatus thread is closed.

**Threads / lessons.** (1) No H4 triangular-arb residual (median ≈0, within spread, conv-corr ≈0). (2) **The
3000s chat's in-apparatus space is now comprehensively exhausted** — directional (both metrics; all
instruments/timeframes/regimes/exits/stop-geometry), volume, calendar, convexity, and triangulation all closed.
The operative finding is the **arc-3004 escalation**: a deployable FX edge needs a structural unlock (second
leg / shorts / real tighter-cost regime) = the operator's decision. No grounded in-apparatus arc remains;
further long-only single-instrument directional variants would grind a cleanly-closed space. **3000s chat has
reached its in-apparatus terminus; escalation pending operator.**

**FLAGS (code not merged):** none. Driver scratch `_disco3_work/observe6_triangulation.py`.

### arc_1008

**Gotobi-day Tokyo-morning USDJPY long (microstructure settlement flow)** (chat 1000–1999). Full record:
[`arcs/arc_1008_gotobi_usdjpy_tokyo_fix.md`](arcs/arc_1008_gotobi_usdjpy_tokyo_fix.md). No council (falsified
at observation; no worthwhile-ceiling fork).

**Idea + why.** Fresh eyes, honest-era only. 16 arcs closed shallow directional prediction; the ONE net-positive
lead (weekend gap-fill, arc 1006, PORTFOLIO) is a discrete-FLOW-event reversion, not continuous prediction. My
pre-shorts lane (dispatch) = hunt a **decorrelated** 2nd net-positive long-only component; refining the gap-fill
gives a *correlated* one (already mapped 4 ways). The most-documented, untested, decorrelated long-only FLOW is
the **gotobi effect**: on days divisible by 5 (+ month-end), Japanese corporates settle invoices → banks buy
USD/JPY → documented upward USDJPY drift into the 9:55 JST (=00:55 UTC) fixing. Long-only-tradeable (long
USDJPY, a major), event-timed, decorrelated from weekend gaps. Attacks frontier item 3 (execution/
microstructure, available NOW pre-shorts). Documented to have decayed post-2015 → **observe before believing.**

**What happened.** The panel's EET-00:00 H4 bar (starts 22 UTC winter / 21 UTC summer, ends 01/02 UTC) fully
contains the Tokyo run-up + the 00:55 UTC fix. OBSERVATION (gross mid bp, gotobi tagged 5/10/15/20/25 + last
trading day, prior-trading-day shift; EURUSD placebo): gotobi Tokyo-morning USDJPY return **−0.06 bp** vs
non-gotobi **+0.13 bp** — gotobi is WORSE (frac+ 0.490). By era: 2010–2014 gotobi +2.00bp ≈ non +1.98bp (NOT
gotobi-specific — generic Abenomics Tokyo-morning USDJPY drift, identical on non-gotobi days, gone by 2015);
2015–2020 −0.98 vs −0.88; 2021+ −0.97 vs −0.49 (gotobi worse). The gotobi *premium* ≈ −0.2 bp gross vs ~2 bp
FundedNext round-trip. H1 fix-window check (rule out a run-into-fix the H4 bar nets against the post-fix hour):
largest gotobi premium anywhere **+0.25 bp** (fix-hour-only, frac+ 0.501 = coin flip), negative in the other
four windows — confirms the H4 finding at finer resolution (arc 2002 discipline). **No edge at any resolution.**

**Verdict: FAIL (cheap-kill at observation) → KILL.** The documented gotobi mechanism produces no capturable
H4/H1 USDJPY bias in 2010–2026 — fully arbitraged (one of the most-published FX seasonalities) or never of a
surviving magnitude. Premium ≤ +0.25 bp gross (coin-flip) ≪ ~2 bp cost; the honest engine (costs + SL-first)
can only make a no-edge gross signal worse (arc 3003) → engine compute not spent. No pool/council/null needed.

**Threads / lessons.** (1) **Gotobi is absent/arbitraged in 2010–2026** (gotobi ≤ non-gotobi every era; the
only +era is generic regime drift). A real *because* (the flow genuinely existed) does NOT imply a still-
capturable edge — the famous seasonality dissolves on honest data. (2) **A documented flow is not a free pass
past observation** — arc 1005 (turn-of-month) found a real-but-sub-cost flow, arc 1008 a famous-but-~zero one;
calendar/microstructure-flow magnitude (not existence) is what's measured against the cost floor. The
microstructure-settlement-flow frontier item is now tested and dry for gotobi. (3) **Mean is additive across
sub-bars** — the cheap H1 run-into-fix vs post-fix split is the right way to rule out a netted-away intraday
run (quantifies arc 2002's resolution discipline). (4) **Surviving pre-shorts long-only lanes** narrow further:
calendar/flow now weakly mapped (TOM sub-cost 1005, gotobi absent 1008); remaining untested = multi-timeframe
STRUCTURE as a setup (not a regime filter), and the standing escalation note — after 17 arcs the only net-
positive long-only edge is the weekend gap-fill (PORTFOLIO); no 2nd decorrelated net-positive long-only
component has been found, reinforcing arc 3004's escalation + the 2001/2003 FLAG-1 shorts/second-leg unlock as
the highest-leverage operator move.

**FLAGS (code not merged):** none. No canonical-core change; no new BUILT tool (gotobi-date tagging is a
one-off calendar helper kept in scratch, like arc 3005's triangulation observer). Drivers scratch
`_disco_work/arc1008_observe.py`, `arc1008_h1_fixwindow.py` (reproducible from the arc doc).

### arc_2004

**The stop-tax on the FILL/REVERSION family — completing the arc-3004 escalation** (chat 2000s). Full record:
[`arcs/arc_2004_stop_tax_reversion.md`](arcs/arc_2004_stop_tax_reversion.md). No council (confirmatory A/B
diagnostic on an already-characterized signal; not a stuck-point fork, not a survivor — cf. 2002/3005).

**Idea + why (log-seeded).** Arc 3004 lifted the one confound shared by all 13 prior arcs — the universal
2·ATR SL-first stop — but on a SINGLE entry: the post-up-spike **momentum/continuation** cell (it found the
stop is a real ~4–5pp drag, NOT the wall). It never tested a **reversion/fill** entry, and that is exactly
where the take-the-loss tax should bite HARDEST: a fill entry buys into weakness → large adverse excursion
(arc 2001 measured the majors gap-fill MAE at −1.1 to −1.3R, P(MAE≤−0.5R)=0.77–0.89), so a 2·ATR stop converts
a large fraction of eventual-reverters into −1R losers. So "the stop isn't the wall" was only proven for one
mechanism family. This also directly answers arc 2001's **FLAG-2** (is H4+FundedNext generically hostile to
fill/reversion?). Universe = **majors** (the 2000s-owned arc-2001 result; deliberately NOT the 1000s JPY-cross
portfolio component — a family diagnostic, not a portfolio build). Reused BUILT `WeekendGapFillLongSignal` +
`make_time_exit_predicate` (geometry/timing only; scoring canonical).

**What happened.** Clean A/B, full 10-fold IS WFO, FundedNext costs ON. STOP-PRESENT vs STOP-REMOVED share the
identical N-bar time exit, isolating the stop (sl_only sl2.0 vs sl_only sl6.0-sizing-only, N∈{6,12,24,48}); plus
the partial/runner baseline. Pool n=176 (matches arc 2001 ✓). **Removing the SL-first stop ~HALVES DD and the
loss tail** — maxDD 3.76→1.56% (N=6), 5.13→1.94% (N=12), 3.94→2.22% (N=24); worst fold −3.11→−1.12% (N=6),
−4.90→−1.35% (N=12) — and lifts mean toward zero (−0.50→−0.07% N=6; −1.08→−0.16% N=12). This is a **LARGER
relative DD/tail effect than arc 3004 saw on momentum**, exactly as the −1.1R-MAE mechanism predicts. **BUT no
config — present or removed, any horizon — is all-folds-positive** (best stop-removed N=24: worst −1.92%, 4/10
neg, mean −0.04% ≈ break-even). The only mean-positive readings (N=48 +0.13/+0.29%) are the cherry-picked best
of the sweep — claiming them = exit-fishing (§5f); the robust read is break-even + regime-dependent (+2.8%
trending folds vs −2 to −3% chop folds, stop or no stop). IS not all-folds-positive → **OOS preserved pristine**
(§4 + arc-2001 council; ~6–8 trades/OOS-yr too thin for the judge anyway).

**Verdict: FAIL → KILL.** The robust read is ≈ the null line; §11 → KILL, not PORTFOLIO (break-even,
all-folds-failing is not a portfolio component). Arc 2001 already KILLED majors gap-fill; this confirms no
exit/SL (incl. the stop-removed extreme) rescues it.

**Threads / lessons.** (1) **The 2·ATR SL-first stop is a real drag for BOTH mechanism families but NOT the
binding wall for either** — 3004 (momentum: ~4–5pp mean drag) + 2004 (reversion: ~HALVES DD, larger relative
effect). For both: the stop fattens the loss tail (removal fixes that), it does not manufacture the EDGE≈COST
deficit. The stop-geometry confound is now lifted on both families; the escalation stands on both. (2)
**FLAG-2 answered:** H4+FundedNext IS hostile to fill/reversion entries — but via the same EDGE≈COST wall, NOT
a fill-specific stop tax (the honest i+1 entry's residual edge after cost ≈ 0; the stop only added variance).
(3) **A reversion entry's relative sensitivity to the SL-first tax (≈halved DD) is bigger than a momentum
entry's** — useful if a future net-positive reversion edge ever appears: prefer a stop-removed/time-exit
construction there, but only the gross edge clearing cost makes it deployable. (4) Driver-convention note (not
a code flag): `FoldStats.roi_pct`/`max_dd_pct` are FRACTIONS (×100 for %) — a first-pass print looked 100×
too small; corrected before reading the verdict.

**FLAGS (code not merged):** none new. Carries FLAG-1 (long-only blocks the stronger UP-gap short side,
operator/human-gated) + the `A1Config.time_exit_bars`-unwired flag (arcs 1005/3004; worked around via the
BUILT `ExitPredicate`). Driver scratch `_disco2000_work/arc2004_gap_stoptax.py`.

### arc_1009

**Weekend gap-fill robustness/firmness audit (the one PORTFOLIO component)** (chat 1000–1999). Full record:
[`arcs/arc_1009_gapfill_robustness_audit.md`](arcs/arc_1009_gapfill_robustness_audit.md). PORTFOLIO record
updated additively: [`portfolio-candidates/arc_1006_weekend_gap_fill_long/robustness_audit_arc1009.md`](portfolio-candidates/arc_1006_weekend_gap_fill_long/robustness_audit_arc1009.md).
No council (not a survivor; an audit, not a new signal).

**Idea + why.** After arc 1008 (gotobi absent) the novel-long-only well is near-dry, and a generative council
on a heavily-mapped space re-surfaces FLAG-1 (arc 2003). The highest-value CHEAP arc that doesn't grind dead
directional ground = **firm the one asset.** The gap-fill record (arc 1006) was TRANSCRIBED, not re-run (no
committed turnkey repro), and arc 1006 picked ONE config (thr 0.5) with a null that wasn't a same-exit
apples-to-apples — so two artifact risks were unchecked (single-pair dependence; threshold-fragility). If the
programme's only asset is an artifact, the PORTFOLIO route is empty — decision-relevant. Reused 3 BUILT tools
(`WeekendGapFillLongSignal`, `make_time_exit_predicate`, `build_null_signal_evaluation`); canonical engine; IS only.

**What happened.** (1) **REPRODUCED** arc 1006 exactly via the registered tools: thr=0.5 → IS mean **+0.685%**
(≈ +0.69%), worst −6.79%, 5/10 neg, 260 IS trades (the record's 396 is the pool count). First committed-tool
reproduction — record verified. (2) **Threshold-FRAGILE:** 0.5→+0.685%, 0.75→+0.191%, 1.0→+0.244%,
1.25→+0.021% — the edge is concentrated at 0.5 ATR and ~vanishes by 1.25; it lives in the many small-to-
moderate gaps, NOT the rare huge ones (consistent with arc 1007's overshoot reframe). (3) **NOT a single-pair
artifact:** leave-one-out at 0.5 stays positive everywhere (+0.27% to +1.34%) — EURJPY strongest contributor,
CADJPY/GBPJPY are drags. (4) **De-inflation (the key finding):** a FAIR random null (matched fire-rate, SAME
24-bar exit + 2·ATR SL + same 5 pairs, only entry timing randomized) is itself IS-mean-**positive +0.327%**;
the real +0.685% beats it by only **~+0.36pp — about HALF the headline.** A random long JPY-cross basket earns
~+0.33pp in-sample from Abenomics-era JPY weakness + the exit/SL geometry; the gap-SPECIFIC excess is thin.
Arc 1006's reported null (−0.60%) was not a same-exit apples-to-apples and inflated the apparent edge ~3.6×.

**Verdict: PORTFOLIO (re-affirmed, thinner).** The component HOLDS — reproduced, beats a fair null,
mean-positive net of costs, not single-pair — but is materially thinner (gap-specific excess ~+0.36pp) and
threshold-fragile. Disposition unchanged (PORTFOLIO, not all-folds-positive). This is the SAME 1006 edge,
audited — NOT a 2nd component (the operator's PORTFOLIO scan should read 1006 + 1009 as one asset).

**Threads / lessons.** (1) **Always null with IDENTICAL exit/SL/universe** — a null that differs in exit (arc
1006's −0.60%) inflates the apparent edge several-fold; the honest gap-specific excess is ~+0.36pp, ~half the
+0.69% headline (the rest is JPY-basket drift + 24-bar/2·ATR geometry random entries share). Arc-10 lesson in
null-construction space. (2) **A random long basket can be IS-mean-positive purely from in-sample regime
drift** (random JPY-cross longs +0.327%, 2010–2020) — IS-mean-positive ALONE is weak evidence; the fair null
+ OOS separate edge from drift. (3) **The gap-fill edge is threshold-fragile** (lives at 0.5 ATR) and pair-
weighted (EUR/CHF/AUD-JPY carry it; CAD/GBP-JPY drag) — a future combination arc should weight accordingly.
(4) **17 arcs, still ONE (thin) net-positive long-only component** — even the best long-only edge is marginal
under FundedNext costs → the shorts/second-leg unlock (arc-3004 escalation, FLAG-1) remains highest-leverage.

**FLAGS (code not merged):** none. No canonical-core change; no new BUILT tool (reused 3 existing). Driver
scratch `_disco_work/arc1009_gapfill_robustness.py` (reproducible from the arc doc + the addendum).

### arc_3006

**Multi-TF breakout-retest (resistance→support flip), long-only** (chat 3000–3999). Full record:
[`arcs/arc_3006_breakout_retest_structure.md`](arcs/arc_3006_breakout_retest_structure.md). No council (falsified
at observation; no worthwhile-ceiling fork).

**Idea + why.** Fresh eyes, honest-era only (20th arc). Arc 1008 flagged the ONE long-only lane still untested:
**multi-TF STRUCTURE as a setup (not a regime filter)** — also the arc-3004 frontier item "structural mechanisms
available NOW (no shorts)." Hypothesis with a real *because*: when price clears a well-established prior swing-high
(resistance) and then **retests and HOLDS** it, the broken level flips to **support** (trapped breakout-sellers
cover + fresh buyers defend = order-flow MEMORY at a real level) → a long on the holding retest should beat the
0.4877 coin-flip base. A **two-stage** structural setup (break THEN retest-hold), distinct from arc-0 SMA-pullback
(no level memory) and the generic breakout (fades). Observe before believing.

**What happened.** 7 USD majors, H4, IS 2010–2020, canonical `observe_long_capture` (honest +1R-before-SL capture
+ 24-bar gross drift) joined with a retest-hold conditioning mask. Primary (L=120, K=20, tol=0.25·ATR): retest-hold
capture **0.4733 < base 0.4860** (lift −0.0127), forward drift **−0.2353 ATR vs base −0.0474** (5× more negative),
on 6/7 pairs (only USDJPY capture 0.535 but drift −0.161 = hindsight-noise cherry). 18-cell robustness sweep
(L∈{60,120,250} × tol∈{0.15,0.25,0.40} × loose/strict hold): capture lift ∈[−0.025,+0.001] (≤ base everywhere bar
one noise cell), drift NEGATIVE in 17/18 cells (−0.35..+0.00). **Monotone in L:** bigger/more-significant levels
give *less negative* drift (−0.30 at L=60 → ~−0.03 at L=250) — i.e. significant structural levels only converge
the retest **TO** the coin-flip base, never above it.

**Verdict: FAIL (cheap-kill at observation) → KILL.** The retest-hold catches **fading breakouts** (the level is
retested *because* the breakout stalled, and stalled H4-major breakouts drift down), NOT defended supports. The
"former resistance flips to support" textbook mechanism is falsified, parameter-robustly, on liquid majors. Entry
is coin-flip-or-worse → §5f exit-sweep does not bite (reserved for entries that beat null or show +gross drift);
no pool/engine/council spent (like arcs 1001/1002/3005/1008).

**Threads / lessons.** (1) **Structural-level memory is arbitraged on liquid H4 majors** — retest-and-hold of a
broken prior-high gives no long edge (capture ≤ base, drift negative in every cell; selects fading breakouts). (2)
**The monotone-in-L signature is the tell** — more-significant levels don't help, they merely stop hurting; a
directional structural bet asymptotes to ~0.49 however "important" the level. 20th arc to land on the directional
coin-flip, now from a 2-stage structural angle. (3) **The last untested long-only lane (multi-TF structure, the
arc-1008 flag) is now CLOSED** → pre-shorts, only portfolio construction (a 2nd net-positive long-only component
to combine with the arc-1006 gap-fill — still not found across 20 arcs/3 chats) and the standing arc-3004
escalation (shorts / second-leg / tighter-cost unlock = operator decision) remain. **Reinforces the escalation:**
every grounded in-apparatus long-only direction the corpus can name is now closed.

**FLAGS (code not merged):** none. No canonical-core change; no new BUILT tool (retest mask is a one-off
conditioning helper kept in scratch). Reused BUILT `observe_long_capture`. Drivers scratch
`_disco3_work/arc3006_observe.py`, `arc3006_sweep.py` (reproducible from the arc doc).

### arc_1010

**Round-number / order-cluster reversion long (Osler 2003 microstructure)** (chat 1000–1999). Full record:
[`arcs/arc_1010_round_number_reversion_long.md`](arcs/arc_1010_round_number_reversion_long.md). No council
(falsified at observation; no worthwhile-ceiling fork).

**Idea + why.** Fresh eyes, honest-era only (21-arc corpus). My pre-shorts lane = a 2nd *decorrelated*
net-positive long-only component, or a novel structural mechanism with a *because*. The one win (weekend
gap-fill, PORTFOLIO) is a discrete-flow-event reversion. The cleanest untested novel mechanism with a strong
documented *because* is **round-number / order-cluster reversion** (Osler 2003, J. Finance): take-profit
orders cluster at big-figure "00" levels → price reverses UP off a round-number support (shorts covering =
buying). An ORDER-FLOW microstructure mechanism, structurally distinct from indicator-reversion (RSI/z, arcs
3000/3001), decorrelated from the weekend gap (fires intraweek on any pair) → a candidate 2nd portfolio
component. Long-tradeable = bar dips INTO a big-figure support from above and bounces. Observe-first (§5b).

**What happened.** `observe_long_capture` (honest +1R-before-SL, gross, IS 2010–2020, 9 pairs = 7 USD majors
+ EURJPY/GBPJPY), 156,130 bars, base capture 0.4831. (A) raw proximity to nearest round = NO structure
(nearest-bucket 0.4765 < base). (B) **support-rejection** (low touches round-below within 0.1·ATR, close
holds) capture **0.4883** vs 0.4817, drift +0.0048 vs −0.0525 — a tiny +0.5pp lift, drift only to ~zero,
sub-0.50. (C) best dist-to-support bucket 0.4948 (sub-0.50); the sharper pierce-and-recover "spring" bucket
0.4873 — NOT better. (D) **per-pair the tell:** EURUSD 0.5155 / USDJPY 0.5003 / GBPUSD 0.5013 lift but
AUDUSD 0.4822 / USDCAD 0.4464 / USDCHF / EURJPY / GBPJPY do not — **3/9 >0.50 = noise-consistent, NOT robust.**
Finer-resolution **H1 confirm** (arc-2002/1008 discipline, 3 best pairs, 204,807 bars): the lift **evaporates**
— touched 0.4746 *below* base 0.4755; EURUSD 0.5155→0.4834, USDJPY 0.5003→0.4638 (its +0.118 drift sits under
sub-0.47 capture = thin-tail artifact). The H4 "3/9 >0.50" does not replicate → confirmed noise.

**Verdict: FAIL (cheap-kill at observation) → KILL.** No robust capturable round-number long edge at H4
(sub-0.50, drift ~0, noise cross-section) and the marginal lift evaporates at H1. One of the most-published
FX microstructure effects is fully arbitraged at H4/H1 in 2010–2020 — the gotobi (1008) / triangulation
(3005) dissolution again. Engine compute not spent (a sub-0.50/~0-drift gross signal is guaranteed sub-cost,
arc 3003).

**Threads / lessons.** (1) **Round-number big-figure order-cluster reversal (Osler) is absent/arbitraged at
H4 & H1** — the 2nd documented-microstructure mechanism (after gotobi) to dissolve on honest data; a famous
*because* ≠ a surviving edge. The "structural price LEVEL as a setup" sub-lane is now dry for round numbers.
(2) **A 3/9-pairs-above-0.50 cross-section is the NOISE signature; a finer-TF re-test is the cheap
discriminator** (arc 2002/1008) — the H4 lift did not survive to H1. Re-usable: when a marginal capture lift
is carried by a minority of pairs, confirm at a finer TF before believing. (3) **Positive drift under sub-0.50
capture = thin-tail artifact, not edge** (re-confirms arc 3003; the drift lens alone misleads — capture +
cross-pair robustness + finer-TF confirm together make the KILL airtight). (4) **21 arcs, still ONE thin
net-positive long-only component** — the novel-structural-mechanism well keeps coming up dry (gotobi,
round-numbers, triangulation); arc-3004 escalation + FLAG-1 shorts/second-leg unlock stays highest-leverage.
**Surviving pre-shorts untested sub-lanes:** multi-timeframe STRUCTURE as a setup, and other discrete
liquidity/flow EVENTS beyond the weekend gap (e.g. month-end fixing-flow *dislocation-and-reversion*, distinct
from arc 1005's turn-of-month *drift*).

**FLAGS (code not merged):** none. No canonical-core change; no new BUILT tool (round-number grid is a one-off
conditioning helper kept in scratch, like 3005's triangulation observer / 1008's gotobi tagger). Reused BUILT
`observe_long_capture`. Drivers scratch `_disco_work/arc1010_observe_roundnumber.py`, `arc1010_h1_confirm.py`.

### arc_3007

**Intraday-spread cost-timing lever on EDGE<COST** (chat 3000–3999). Full record:
[`arcs/arc_3007_spread_cost_timing.md`](arcs/arc_3007_spread_cost_timing.md). No council (cheap-kill — the lever
backfires at observation; engine-confirmed the lone positive bucket).

**Idea + why.** The corpus-wide binding constraint is **EDGE<COST** (arcs 1003/1004: the fix is "raise gross
edge OR cut per-trade cost"). Every cost lever tried = frequency-via-exits (1004, didn't flip it). The ONE
untried cost lever = **intraday SPREAD TIMING**: FX spreads vary 3–5× across the day, the honest engine charges
1.5× the *actual bar spread*, so restricting a positive-gross-drift signal to the tightest-spread window cuts
realized cost. Tested on arc 1003's cross-trend long (Donchian-20 + SMA200-uptrend, 12 crosses — the textbook
+0.10R-gross-but-net-negative EDGE<COST case).

**What happened.** Intraday spread profile: median spread_R 0.0256 (tight) → 0.0764 (wide), **2.99× range** —
real cost range. But splitting the cross-trend pool (n=4539, capture 0.4807, gross +0.0218R) by entry-spread
tercile shows spread and gross-edge are **entangled the wrong way**: **tight** spread_R 0.0138 → gross
**−0.0199R** (NEGATIVE); **mid** 0.0328 → **+0.1207R**; **wide** 0.0858 → −0.0354R. The tightest-6-UTC-hour
window is gross −0.0158R too. The cost-saving direction (tight spread = liquid London/NY hours) has the
WEAKEST gross edge — high efficiency arbitrages the breakout. The lever **backfires**: timing to low spread =
timing to low edge. Confirm (don't hand-wave the +0.12R mid bucket — Arc-10 discipline): mid-spread-band
[0.02,0.05] cross-trend (2955 fires), 3-fold honest triage 2013/2016/2019 = +0.08% / +0.01% / −0.13% → **mean
−0.02%, worst −0.13%, not all-folds-pos, net-NEGATIVE** on the momentum-friendly years (so full-IS ≤ this, arc
3002 lesson). The +0.12R gross collapses under take-the-loss + cost (arc-3003 signature).

**Verdict: FAIL → KILL.** Spread-timing cannot flip the cross-trend net-positive: the cost-saving direction has
negative gross edge, and the lone positive-gross bucket nets ≤0 SL-honest. Spread and edge are entangled (both
liquidity-driven) — you cannot cut cost without cutting edge.

**Threads / lessons.** (1) **Cost-timing lever closed — spread×edge entangled.** Tight-spread/liquid hours
(cheapest) carry the weakest gross edge; the one +gross bucket (mid-spread) is non-monotone and dies SL-honest.
Arcs 1003/1004's "cut per-trade cost" steer is now tested and dead. (2) **Non-monotone gross-vs-spread is the
tell it isn't a lever** — a monotone relation would be exploitable; a bulging middle bucket is regime/selection
noise (engine-confirmed). (3) **EDGE<COST is now closed from BOTH sides** — edge-side (raise gross: every
entry/TF/universe/regime, arcs 0–3006) and cost-side (cut cost: exits/frequency 1004, spread-timing 3007).
**Cements the arc-3004 escalation:** the apparatus cannot express a deployable edge; a structural unlock
(shorts / second leg / genuinely tighter-cost broker regime) is the operator's decision. No 2nd net-positive
long-only component found (still only the arc-1006 gap-fill).

**FLAGS (code not merged):** none. No canonical-core change; no new BUILT tool (mid-spread filter is a one-off
wrapper in scratch). Reused BUILT `observe_long_capture` + `DonchianBreakoutLongSignal`. Drivers scratch
`_disco3_work/arc3007_observe.py`, `arc3007_triage.py`.

### arc_1011

**Month-end reversion long (mechanical rebalancing-flow over-extension)** (chat 1000–1999). Full record:
[`arcs/arc_1011_month_end_reversion_long.md`](arcs/arc_1011_month_end_reversion_long.md). PORTFOLIO record:
[`portfolio-candidates/arc_1011_month_end_reversion_long/`](portfolio-candidates/arc_1011_month_end_reversion_long/).
No council (council is mandatory only for a PASS survivor; this is PORTFOLIO).

**Idea + why.** Fresh eyes, honest-era only (22-arc corpus; my prior arc 1010 round-numbers KILL, concurrent
3006 breakout-retest KILL). My pre-shorts lane = a 2nd *decorrelated* net-positive long-only component. The
ONLY mechanism family that has produced net-positive is the discrete liquidity/flow-EVENT reversion (weekend
gap-fill, arc 1006). So I applied that WINNING TEMPLATE to a structurally different event/universe:
**month-end mechanical rebalancing** (WMR 4pm London fix, last business day) — large inelastic flows that
over-extend a currency into the fix and then REVERSE. Distinct from arc 1005 (which measured turn-of-month
*drift over the window*, sub-cost); here the object is the discrete spike-into-month-end → reversion.
Long-tradeable = a big DOWN move into month-end → buy the reversion UP.

**What happened.** Observation (D1, 7 USD majors, IS): the reversion concentrates in EXTREME down moves —
≤ −1 ATR into month-end → fwd2 **+0.186 ATR, 61.6% pos** (comparable to the gap-fill). **CRITICAL CONTROL
(the decisive test, since generic reversion is dead — 3000/3001):** the SAME ≤ −1 ATR 2-day down move,
month-end vs RANDOM day → month-end +0.186 vs random **−0.063** (random big-down CONTINUES down, all 7 pairs)
→ **+0.249 ATR month-end EXCESS** = the *timing* is the mechanism, not generic reversion. Canonical engine
(`MultiPairBacktester`, FundedNext costs): pool n=121 (floor PASS), gross +0.0635; full IS WFO sl_only + 2-bar
time exit **mean +0.23%, 7/10 folds positive** (3-bar +0.50%/6-10; longer holds + TP/trail dilute); **beats
the fair same-exit null (random entry, matched fire-rate) by +0.56pp** (real +0.23% vs −0.33%). Robustness
(arc-1009 discipline): threshold-robust (0.75–1.5 all +0.18 to +0.27%), per-fold broad-based, leave-one-pair-out
all positive. **NOT all-folds-positive (7/10) → OOS deliberately NOT touched** (holdout preserved).

**Verdict: FAIL the sole judge → PORTFOLIO.** Mean-positive net of costs, mechanism-controlled, robust,
decorrelated — but fold-fragile. The **2nd net-positive decorrelated long-only component**, found by applying
the winning discrete-flow template to a new event/universe (deliberate, not luck). Fold-ROI corr vs arc 1006
gap-fill = **+0.117** (near-zero; structurally disjoint events + universes) → **the portfolio thread is now
ACTIVE.** A naive 2-way fold-add is still 6/10 (the gap-fill's ±8% swings dominate my ±1%) → the combined-book
all-folds-positive WFO (co-simulated, risk-weighted, likely a 3rd component) is the gated next step, NOT claimed.

**Threads / lessons.** (1) **Month-end mechanical-rebalancing reversion is a REAL, controlled, decorrelated
net-positive long-only edge** — 2nd PORTFOLIO component. (2) **The random-day control is the decisive test for
ANY reversion claim** (generic reversion is dead): a reversion edge must beat its own same-magnitude
random-day control, not just the base — month-end's +0.249 ATR excess passes. Re-usable discipline. (3) **The
winning template GENERALIZES across events** — "discrete flow-event over-extension → reversion" worked for both
the weekend gap (1006) and the month-end fix (1011); a 3rd decorrelated flow event is the natural next hunt.
(4) **More threshold-robust than the gap-fill** (0.75–1.5 vs the gap-fill's single 0.5), though thinner in mean
(+0.23% vs +0.69%). (5) **Portfolio thread ACTIVE** (2 components, corr +0.12) but 2 are not yet enough for
all-folds-positive (gap-fill variance dominates) → wants ≥3 and/or risk-weighting; that combination is its own
gated arc. The arc-3004 escalation (shorts/second-leg) still dominates for a *solo* deployable.

**Tooling:** built + registered `discovery/tools/month_end_signals.py :: MonthEndReversionLongSignal`
(EXPERIMENT signal — mask + ATR geometry only; scoring canonical). TOOL_REGISTRY BUILT updated.

**FLAGS (code not merged):** none requiring the canonical core. Carries the standing `A1Config.time_exit_bars`
-unwired flag (arcs 1005/3004; worked around via the BUILT `make_time_exit_predicate`). Drivers scratch
`_disco_work/arc1011_observe_monthend_reversion.py`, `arc1011_control_monthend_vs_random.py`, `arc1011_wfo.py`,
`arc1011_robustness.py` (reproducible from the arc doc).

### arc_2005

**Attacking EDGE<COST from the COST side (cost-regime conditioning)** (chat 2000s). Full record:
[`arcs/arc_2005_cost_side_attack.md`](arcs/arc_2005_cost_side_attack.md). No council (the decisive control —
random-on-cheap-bars null — settled the soundness question in-arc; cf. 2002/3005). Built + registered
`make_low_cost_mask` (BUILT).

**Idea + why (log-seeded).** The programme's binding constraint is EDGE<COST; every arc attacked the EDGE side.
Arc 3004's peer review flagged "the cost is a wall INDEPENDENT of the stop." FundedNext cost = 1.5×spread+…, and
spread varies a lot by bar. Arc 1003's cross trend-momentum has +0.10R GROSS but nets sub-cost because crosses'
WIDE spreads eat it. So the never-run experiment: restrict that +gross entry to its cheapest bars (spread/ATR in
a trailing-rolling LOW quantile) — does cutting cost flip the net sign? Tests WHICH half of EDGE<COST binds.

**What happened.** Donchian-20+SMA200 on 12 trending crosses, full IS WFO, costs ON. **MONOTONE cost effect:**
mean −7.96%(all bars) → −3.38%(cheapest 50%) → −0.38%(30%) → **+1.72%(15%)**; neg folds 8→7→7→2; maxDD
27.55→6.01%. The cheapest-15% (+1.72%, 2/10 neg) is the best long-only IS mean in the programme. **Decisive
control — random entry on the SAME cheap bars** (3 seeds): mean ≈ −0.3%, ~5/10 neg → the real cross-trend
selection beats random-cheap IN-SAMPLE (so it's not merely "cheap bars cost less"; a zero-edge long stays
break-even even when cheap). **BUT OOS (2021-26) = mean −2.26%, 4/6 neg** → the edge does NOT survive; combined
with the IS positivity being concentrated ONLY at the most aggressive q (q full-sample-swept), the IS +1.72% is
q-overfit + favorable-regime.

**Verdict: FAIL → KILL (conservative, §8/§11).** Genuine PORTFOLIO-vs-KILL fork (IS mean-positive + beats null =
arc-1006 profile), but OOS-negative ⇒ net-negative forward ⇒ KILL (can't diversify net-negative positive), and
the q-sweep means it isn't the clean un-swept IS-positive arc 1006 had. A proper nested cost-quantile WFO +
HEAVY council is the flagged path that could in principle re-qualify a milder version — current evidence does
not support PORTFOLIO.

**Convergence with arc 3007 (chat 3000s, landed mid-arc) — reconciling a contradiction.** 3007 INDEPENDENTLY
attacked the cost side on the SAME signal via **intraday spread-timing** and found it **backfires**: the
tight-spread tercile (= liquid London/NY hours) has the WEAKEST/NEGATIVE gross edge (tight −0.0199R, mid +0.12R,
wide −0.035R), "you cannot cut cost without cutting edge — spread×edge entangled." Reconciliation: 3007 used
**absolute** spread/hours (→ liquid hours, low gross edge); I used **trailing-relative** cheapness (→ different
bars, cheap vs a pair's own recent norm), and the Donchian selection beats random-cheap IN-SAMPLE on those bars
— but my OOS-negative confirms it doesn't generalize, and 3007 supplies the mechanism (cost↓ entangled with
edge↓). **Both chats reach the same KILL; together they CLOSE the cost side of EDGE<COST from two independent
angles.**

**Threads / lessons.** (1) **The COST side of EDGE<COST is closed (two chats, two cheapness definitions).** Cost
is NOT a free lever on a directional FX entry — the cost-saving direction (liquid/tight-spread) is entangled
with the weakest gross edge (3007), and a trailing-relative-cheapness IS-positive does not survive OOS (2005).
(2) **Completes the "stop + cost" closure of the arc-3004 escalation:** removing the STOP (3004) lifts mean but
not to all-folds-positive; cutting COST lifts IS mean (2005) or backfires (3007) but yields no durable edge.
Both secondary ~5–10pp drags are now lifted across both chats and **DIRECTION is still the binding wall** — the
real unlock is the 2nd leg / shorts (relative-value), not cheaper execution (a raw-spread ECN helps margins, not
the directional edge). (3) **Methodological:** a MONOTONE in-sample lift + beats-random can STILL be
non-durable (OOS-negative) when the conditioning variable is entangled with the edge and the cut depth is
full-sample-swept — the OOS + a sister-chat's mechanism analysis (3007) caught it. (4) `make_low_cost_mask`
(BUILT) is reusable, but apply it ONLY to a signal whose gross edge is demonstrably decoupled from the spread
regime (none found yet).

**FLAGS (code not merged):** none new. Carries FLAG-1 (long-only blocks the UP-gap short side) + the
`A1Config.time_exit_bars`-unwired flag. Drivers scratch `_disco2000_work/arc2005_cost_side.py`,
`arc2005_null_control.py`, `arc2005_oos.py`.

### arc_1012

**Month-End Reversion: best-version completion (§5f) + survivor-ceiling test** (chat 1000–1999).
Full record: [`arcs/arc_1012_month_end_best_version.md`](arcs/arc_1012_month_end_best_version.md).
Additive PORTFOLIO note:
[`portfolio-candidates/arc_1011_month_end_reversion_long/best_version_audit_arc1012.md`](portfolio-candidates/arc_1011_month_end_reversion_long/best_version_audit_arc1012.md).
No council (a protocol-mandated best-version completion on an existing PORTFOLIO component, cf. 1009).

**Idea + why.** Honest-era corpus = 23 arcs. Two PORTFOLIO components exist; the gap-fill (1006) was
thoroughly worked across 6 arcs and is firmly fold-fragile, but the month-end reversion (1011) was only
3 days old and **under-explored** — 1011 sampled exits informally and never diagnosed its 3 negative
folds. §5f MANDATES the best HONEST exit be tested before a FAIL on a NON-coin-flip entry (month-end
beats a fair null +0.56pp, mechanism-controlled +0.249 ATR excess), so completing it is required, not
optional. It is also the one asset with genuine **survivor** upside (all-folds-positive → PASS). Decisive
either way. (Chose this over a 3rd-component hunt because closing whether either existing component can be
a *solo* survivor is logically prior to combining them — a solo survivor beats any portfolio.)

**What happened.** Reproduced arc 1011 byte-for-byte via the registered tools (sl_only/te2/sl2.0 → mean
+0.232%, 7/10; per-fold vector identical). **§5f nested exit/SL sweep (24 configs)** — full registered
menu × SL{1.5,2.0,2.5} × time-cap{2,3,5}, anti-fishing (same config must be all-folds-positive, no
per-fold cherry-pick): **0/24 all-folds-positive.** Best fold count 7/10 (the baseline); the higher-mean
te3/sl2.0 (+0.505%) is WORSE on folds (6/10); te5 and ALL tp/trail policies dilute (mean ≤0). The
reversion is a tight ~2-3 day event — holding past it gives the move back. No config cleared IS → **OOS
never touched.** **Negative-fold diagnosis:** fold→year maps the 3 negatives (folds 5/6/7) to a
CONTIGUOUS **2014/2015/2016** block = the strong-USD-bull regime (EUR 1.39→1.05, SNB unpeg Jan-2015, oil
crash) — a regime drag, not random fold-fragility. Two pre-registered, mechanism-reasoned refinements,
both tested without tuning to the bad folds: **(A) quarter-end-only** (larger flow) → FALSIFIED (4/10,
thins to min-trades 0); **(B) trend filter close>SMA100** (informed trend overwhelms flow) → FALSIFIED
(4/10; SMA50/200 worse). The **inverse** (close<SMA100) is marginally better (8/10, +0.302% — a coherent
"rebalancing buys the already-weak leg" story) but is **post-hoc/fishing AND still leaves 2015/2016
negative** — the SNB-unpeg/USD-peak folds survive every honest cut. Not claimed.

**Verdict: FAIL the survivor judge → PORTFOLIO (re-affirmed, survivor ceiling CLOSED).** The month-end
reversion's best HONEST version is not all-folds-positive; its 2014-2016 regime block is irreducible by
exit/SL or pre-registered filter. It remains a valid PORTFOLIO component (mean-positive, decorrelated,
threshold-robust) — it just cannot stand alone. The month-end analogue of arc 1009's gap-fill audit.

**Threads / lessons.** (1) **Both PORTFOLIO components are now best-version-closed as non-survivors**
(gap-fill 1006/1007/1009/2002/2004; month-end 1011/1012) — neither exits nor reasoned filters lift either
to all-folds-positive; both are regime/tail-timing-fragile. PORTFOLIO-combination + the arc-3004
escalation (shorts/second-leg) are the only live paths. (2) **§5f exit menu is exhausted on the month-end
signal** (0/24) — a future arc need not re-sweep exits on it. (3) **A contiguous-in-time negative-fold
block is a REGIME drag, not exit-fixable** — diagnose the regime but never let "which filter flips those
folds" drive the choice (the Arc-10/§5f fishing trap; here both reasoned refinements FAILED and the
post-hoc inverse still fell short, so the verdict rests on NO fished number). (4) **Thread (fresh-arc,
pre-register first):** month-end reversion may be stronger for below-trend/oversold currencies
("rebalancing buys the underweight") — needs its own ex-ante population + fair null, not a post-hoc
contrast. (5) **Next in this range:** a 3rd decorrelated discrete-flow-event component (arc 1011's named
hunt), as both existing components' solo ceilings are now closed and a naive 2-way combination is
gap-fill-variance dominated (6/10).

**FLAGS (code not merged):** none requiring the canonical core. No new BUILT tool (reused
`MonthEndReversionLongSignal` + `make_time_exit_predicate`; quarter-end / SMA masks are one-off
conditioning helpers in scratch). Carries the standing `A1Config.time_exit_bars`-unwired flag (arcs
1005/3004). Drivers scratch `_disco_work/arc1012_*.py` (reproducible from the arc doc).

### arc_2006

**2-way PORTFOLIO combination WFO (gap-fill 1006 + month-end 1011) — the route's gated next step**
(chat 2000s). Full record: [`arcs/arc_2006_portfolio_combination_2way.md`](arcs/arc_2006_portfolio_combination_2way.md).
No council (not a PASS survivor, not an idea-fork/diagnosis — a direct quantitative combination test).

**Idea + why (log-seeded).** After 23 arcs the directional space (both metrics, all instruments/TFs/
regimes/exits/stop, both EDGE and COST sides) is comprehensively closed, leaving exactly TWO net-positive
long-only components — both PORTFOLIO, both *discrete flow-event reversion* edges: weekend gap-fill on
JPY crosses (1006/1009) and month-end reversion on USD majors (1011), corr ≈ +0.117. Arc 1011 flagged the
**combined-book all-folds-positive WFO (co-simulated, risk-weighted) as the gated next step, NOT claimed**
— and nobody had actually RUN it (the "naive 2-way add is 6/10" was a back-of-envelope fold-ROI add). The
PORTFOLIO route's entire premise is that decorrelated mean-positive components combine to all-folds-positive
(§6/§11). With exactly the 2 components the route was built on, the single most decision-relevant unrun
experiment is the honest risk-parity combination — decisive either way (PASS candidate, or a rigorous "2
is not enough, here is the 3rd's exact spec").

**What happened.** Reproduced each component via its REGISTERED signal over the SAME canonical IS folds
(`build_v3_folds`, 10 expanding folds, OOS years 2011–2020), scored solely by `MultiPairBacktester`,
FundedNext ON, SL-first. **Reproduction is EXACT** (Arc-10 discipline — don't trust transcription):
gap-fill IS mean **+0.685%**, month-end **+0.232%**, corr **+0.117** — matching the 1009/1011 records to
the basis point and confirming fold-alignment (`build_v3_folds` date windows are TF/universe-agnostic, so
fold *j* is the same calendar year for the H4 and D1 signals). Combined via the BUILT
`combine_fold_roi` under equal + risk-parity (inverse-fold-vol) weights, fit-on-IS-then-frozen:
- **equal** (0.5/0.5): mean +0.459%, **4/10 neg**, NOT all-folds-positive.
- **risk-parity** (gap 0.128 / me 0.872): mean +0.290%, **4/10 neg**, NOT all-folds-positive; worst fold
  −1.53%. Risk-parity does its job on *variance* (gap-fill solo worst −6.79% → equal-weight −2.94% →
  risk-parity −1.53%) but cannot pass the gate.
- **The blocker is a mutually-negative fold:** fold 6 = **OOS year 2015** is negative for BOTH components
  (gap −4.19%, me −1.14%). **No convex combination — and no honest single-engine co-simulation — can make
  2015 positive** (book P&L is additive: two books that both lose over a window sum to a loss over it). The
  conclusion is **combination-method-invariant**; it does not rest on the linear-combination approximation.
- IS not all-folds-positive → **OOS deliberately NOT touched** (§4 + holdout preservation).

**Verdict: FAIL the sole judge (combined 2-way book) → KILL.** Not deployable, and *provably* blocked by
≥1 mutually-negative fold. **The two components are UNCHANGED** (reproduced exactly, not re-tested/weakened)
— they retain PORTFOLIO status; no new `portfolio-candidates/` entry (would double-count 1006/1011). The
portfolio **thread stays ACTIVE**: a 3rd component is now a hard requirement with a precise spec.

**Threads / lessons.** (1) **Two near-zero-correlated mean-positive components are NOT enough** for an
all-folds-positive book when they share even one mutually-negative fold (2015) — the route needs **≥3**,
*selected for fold-complementarity*, not just low average correlation. (2) **Average-correlation
decorrelation ≠ fold-complementarity — TAIL-correlation is the real test.** corr +0.117 looked great, but
both edges are flow-reversion and bleed in the *same* risk-off years (2015 CHF-depeg, 2016, 2018, 2020);
the risk-parity 2-way is negative in exactly those years. **Re-usable selection criterion: rank a candidate
3rd by its ROI on the existing book's NEGATIVE folds, not by standalone mean or average correlation.** A 3rd
*reversion-flavored* flow edge would likely inherit the same 2015 tail and not fix it — the diversifying leg
must be a *different mechanism family / regime* (positive when reversion bleeds). (3) **Risk-parity is the
right weighting (cut worst-fold −6.79%→−1.53%) but is a variance lever, not an edge lever** — the gate is
robust to re-weighting; only a complementary component moves it. (4) **The mutually-negative-fold test is a
cheap, combination-method-invariant pre-screen** for any portfolio-combination arc: if components share a
fold where all are negative, stop — no weighting/co-sim passes; you need a component positive on that fold
first. (5) **Reinforces the arc-3004 escalation:** the regime-orthogonal leg the long-only portfolio route
needs (positive in risk-off) is most naturally a short / second-leg / relative-value construction — the
directional/trend long menu that would win in those years is comprehensively dead — so shorts/second-leg
unlock (FLAG-1) stays the highest-leverage operator move. **Surviving pre-shorts lane:** a 3rd long-only
flow edge with a *different regime signature* than weekend-gap / month-end reversion (positive in 2015/16/
18/20), if one exists.

**Tooling:** built + registered `discovery/tools/combine_fold_roi.py` (`fit_weights`, `combine_fold_rois`,
`rois_from_fold_stats`, `CombinedBook`) — EXPERIMENT tool: equal / inverse-vol-risk-parity linear
combination of canonical per-fold `FoldStats` ROI into a combined-book series for the all-folds-positive
judge (never realizes P&L; the engine did, upstream). Documents its first-order-linear-vs-co-simulation
LIMITATION (faithful here: disjoint universes + disjoint event timing). Reusable by every future
portfolio-combination arc.

**FLAGS (code not merged):** none new. Carries FLAG-1 (long-only blocks the short/regime-orthogonal leg
the portfolio route needs) + the `A1Config.time_exit_bars`-unwired flag (worked around via the BUILT
`make_time_exit_predicate`). Driver scratch `_disco2000_work/arc2006_combo.py` (reproducible:
`PYTHONPATH=. py _disco2000_work/arc2006_combo.py`).

### arc_3008

**Month-end London-4pm-fix flow dislocation-and-reversion (long)** (chat 3000–3999). Full record:
[`arcs/arc_3008_month_end_fix_reversion.md`](arcs/arc_3008_month_end_fix_reversion.md). No council (a
documented-mechanism observation that beat its placebo+null but died at the cost floor — confirmatory, not
a stuck-point fork nor a survivor).

**Idea + why (log-seeded).** Fresh eyes, honest-era only (22-arc corpus). Pre-shorts lane = a 2nd
*decorrelated* net-positive long-only PORTFOLIO component (the one win, weekend gap-fill arc 1006, is a
discrete-flow-event reversion). Arc 1010 named the last untested discrete-flow lane verbatim: **month-end
fixing-flow dislocation-and-reversion**, distinct from arc 1005's turn-of-month *drift*. The London 4pm
WM/Reuters fix is the most-documented FX flow event (Melvin & Prins 2015; Evans 2018): month-end equity
hedging rebalances at the 16:00 London benchmark; the mechanical, predictable flow pushes price INTO the
fix and partially REVERSES after. Long-only side = buy an abnormal DOWN push into the fix → bet post-fix
reversion. The gap-fill archetype on a decorrelated, calendar-timed *intraday* event → a candidate 2nd
portfolio component.

**What happened.** H1, 7 USD majors, IS 2010–2020. Fix-hour bar = the 15:00–16:00 London bar (DST-robust
via Europe/London tz), ending at the 16:00 fix. **OBSERVATION (bp):** a real, month-end-SPECIFIC reversion
— corr(push, +2h) **−0.121 on month-end** vs −0.035 non-ME; big down-push into the month-end fix reverts
**+4.31bp/2h, frac+ 0.576** (n=59) vs only +1.44bp non-ME. **Honest ATR/R lens (BUILT `observe_long_capture`,
next-bar-open entry):** gross **+0.11–0.12R/3h drift** at every threshold — but capture 0.24–0.26
(uninformative: +1R=2·ATR over 6 H1 bars is far; drift is the lens) and the cross-section is the arc-1010
NOISE signature (only USDCAD/USDCHF robustly positive across thresholds; GBP/NZD/JPY negative). Because it
beat the placebo AND showed +gross drift (non-coin-flip), §5f required the honest engine before a FAIL —
with the reversion-appropriate exit (short time-exit; arc 2004). **Full 10-fold IS WFO** (BUILT
`MonthEndFixReversionLongSignal`, ArcFoldRunner→MultiPairBacktester, FundedNext ON), best-version sweep
(2 thresholds × 4 exits): **every cell net-NEGATIVE, none all-folds-positive.** Best = stop-removed time3
(−0.22 to −0.28% IS-mean, 5/10 neg, maxDD 1.53%). **Fair null (same exit/SL/universe):** REAL −0.28% vs
NULL −0.73% (9/10 neg) → **REAL but SUB-COST edge** (arc-0/1000 signature). OOS preserved (IS never cleared;
~3 trades/pair/yr too thin for the judge anyway — arc-2001/2003 discipline).

**Verdict: FAIL → KILL** (§11 beats-null-but-net-negative; NOT PORTFOLIO, which needs mean-POSITIVE net
of cost). The fix reversion genuinely exists but its +0.12R/3h gross drift does not clear FundedNext cost +
take-the-loss on the H1 construction.

**Convergence with arc 1011/1012 (chat 1000s, landed concurrently).** The 1000s chat independently tested
**month-end reversion long on USD majors** and reached **PORTFOLIO** (the 2nd net-positive component): arc
1011 is a **D1 multi-bar over-extension INTO month-end** (mean +0.23% IS, mechanism-controlled +0.249 ATR
excess vs random day, beats fair null +0.56pp; 7/10 folds → PORTFOLIO). Mine is the **H1 intraday
16:00-fix-WINDOW** dislocation of the same broad idea — sub-cost, KILL. **Two chats converged: month-end
reversion is REAL on USD majors.** The disposition split is explained by my lesson #2: the D1 R (2·D1-ATR)
is large so cost-in-R is low and the drift clears; my H1 R (2·H1-ATR≈H4-ATR) is small so the same spread
costs ~2× in R and eats it. **The capturable expression is the D1 over-extension (arc-1011 PORTFOLIO), NOT
the intraday fix window (my KILL).** My arc CORROBORATES theirs and adds the negative boundary on resolution.

**The arc-2004/3004 reversion A/B reproduces on a NEW family.** Removing the SL-first 2·ATR stop halves DD
(5.97→1.53%) and lifts mean from −1.3% toward zero (−0.28%) — buy-into-weakness = large adverse excursion =
the take-the-loss tax bites hardest — but never manufactures the edge. Two independent reversion families
(weekend gap-fill majors, month-end fix) now share this exact signature: the stop is a real drag, not the
wall; residual edge ≈ cost.

**Threads / lessons.** (1) **Month-end fix reversion is REAL (placebo- AND null-beating) but sub-cost on H1
majors** — the 4th documented microstructure/flow effect to be real-but-uncapturable on honest data (gotobi
1008, round-numbers 1010, triangulation 3005, now the fix). A famous *because* with a measurable footprint
STILL dies at the cost floor. (2) **The H1 cost-in-R penalty is a structural tax on intraday reversion:**
1R = 2·H1-ATR ≈ H4-ATR, so the same spread costs ~2× in R — an intraday reversion needs a *larger* gross
drift than an H4 signal; +0.12R/3h is not enough. (3) **The discrete-liquidity-EVENT lane (arc 1010's last
named long-only thread) is now mapped:** month-end **D1** over-extension reversion is mean-POSITIVE →
**PORTFOLIO (arc 1011, the 2nd decorrelated long-only component)**; the month-end **H1 fix-window** version
(this arc) is sub-cost KILL. There are now **TWO net-positive long-only PORTFOLIO components** (arc-1006
gap-fill + arc-1011 D1 month-end reversion). **Arc 2006 already combined those two → KILL** (mutually-negative
2015 fold + tail-corr; pick a 3rd by its ROI on the book's NEGATIVE folds). My net-negative fix-window variant
is NOT a usable 3rd. Reinforces the arc-3004 escalation + FLAG-1 shorts/second-leg unlock as highest-leverage. (4) **The fix's STRONGER leg is the UP-push SHORT** (the
up-push into the fix reverts DOWN, the better side — FLAG-1 again), structurally blocked by long-only.

**FLAGS (code not merged):** none new. Carries FLAG-1 (long-only blocks the stronger UP-push-into-fix short
leg — operator/human-gated) + the `A1Config.time_exit_bars`-unwired flag (arcs 1005/3004; worked around via
the BUILT `ExitPredicate`). BUILT + registered `MonthEndFixReversionLongSignal`
(`discovery/tools/fix_flow_signals.py`). Drivers scratch `_disco3_work/arc3008_observe_fix.py`,
`arc3008_capture.py`, `arc3008_wfo.py`.

### arc_2007

**DEEP multi-factor liquidity-sweep reversal long** (chat 2000s — the range's DEEP-arc entry). Full
record: [`arcs/arc_2007_liquidity_sweep_reversal_long.md`](arcs/arc_2007_liquidity_sweep_reversal_long.md).
No council (the fair-null + honest engine settled it in-arc decisively — loses to random; not a
stuck-point fork nor a survivor, cf. 2004/3008).

**Idea + why (the dispatch's open frontier).** The corpus closed *shallow single-condition* directional
prediction; the DEEP, multi-factor structural lane was untested (§5(a) carve-out: a novel conjunction
with a documented *because* earns a fresh test). I built the genuinely multi-condition entry the
programme never ran: a **liquidity-grab / Wyckoff-spring** long — a fast DOWN move that pierces a prior
swing low (runs resting sell-stops) and **immediately reclaims** it (a large buyer used the stop-cascade
liquidity to fill) → revert UP. The DEEP claim: each single condition is a corpus-proven coin-flip, but
the *intersection* (sweep × reclaim × fast-3-bar-drop × SMA200 uptrend), developed FREELY on IS, isolates
the genuine forced-sweep cell. Forced/triggered flow is the same family as the only two wins (gap-fill
1006, month-end 1011).

**What happened.** OBSERVATION (H4 majors, IS, `observe_long_capture`, n reported at every cut): base
capture 0.486. The nested conjunction lifts +1R-capture to **0.52** (L30 sweep&reclaim&uptrend&fast3,
n=484) — but forward drift only **+0.034 ATR** (sub-cost; cost hurdle ~0.05–0.10R). **Key finding: the
CLIMAX (big-range, forced) sweep is a FALLING KNIFE** — capture ticks up but drift goes strongly NEGATIVE,
*worse the bigger the climax* (−0.124 → −0.280 → −0.333) → my *because*'s strongest prediction (violent
forced sweep = exhaustion = reversion) is **falsified**; the violent leg continues DOWN (the **SHORT**
edge, FLAG-1). Round-2 refinement (depth/reclaim-strength/trend/horizon): "strong reclaim" DROPS capture
to 0.45 (buys after the bounce ran); >0.55 cells are tiny-n noise (n=4–57). Best robust long version =
sweep & reclaim & close>SMA200 & drop3<−1·ATR → marginally non-coin-flip → §5f honest best-version test.
**Honest WFO** (10 IS folds, registered exit menu × SL × time-caps, anti-fishing single-config): **every
config mean-NEGATIVE, none all-folds-positive** (PRIMARY L20: −1.76% to −5.26%, 1–5/10 pos; L30: −0.28%
to −1.79%, 3–4/10). **Decisive — the entry LOSES TO THE FAIR NULL** (random entry, matched fire-rate,
identical exit/SL/universe): real −1.76/−2.39/−1.96% vs null −0.33/−0.66/−0.59% at te6/12/24 → worse than
random at every horizon (~1.4–1.7pp). OOS NOT touched (IS never cleared).

**Verdict: FAIL → KILL** (§11: loses to the null ⇒ net-negative, below the baseline ⇒ KILL, not
PORTFOLIO).

**Diagnosis (why it loses to random).** The capture lift (0.486→0.52) is a **thin-tail artifact** — a
quick +1R bounce fires often enough to tick capture above 0.50, but drift is ~0 and the entry **buys into
recent weakness** (a fast down move), sitting in large adverse excursion → take-the-loss converts a
disproportionate share into −1R losers, *more* than a random entry suffers. The small capture edge is
more than offset by the fat adverse tail → underperforms random. This is the **arc-2001/2004 reversion
stop-tax** mechanism reproduced on a *structural chart* setup. It fails where month-end (1011) succeeded
because 1011 has a **forced-flow control** (month-end vs random-day +0.249 ATR EXCESS — timing is causal);
the sweep-reclaim *looks* like a grab but is **not tied to any measurable forced participant** — a chart
pattern, arbitrated to coin-flip like round-numbers (1010) and breakout-retest (3006).

**Threads / lessons.** (1) **DEEP multi-factor depth does NOT escape the directional coin-flip** — the
first genuinely multi-condition structural long, developed freely toward all-folds-positive, lands at
cap 0.52 and *below the fair null* on the honest engine. Reinforces the arc-3004 escalation from a deeper
angle: the wall is DIRECTION, not the shallowness of prior cuts. The clean directional falsification the
programme never ran on a multi-factor entry. (2) **A climactic forced sweep below a swing low is a
FALLING KNIFE** (drift −0.33, worse the bigger) — "violent = exhaustion = reversion" is wrong on liquid
H4 majors; the violent leg is the **SHORT** edge → a concrete, well-motivated first target the moment
shorts unlock (FLAG-1). (3) **Buying structural weakness is worse than random under take-the-loss**
unless a forced-flow control proves the timing is causal (the arc-1011 random-day-control discipline);
a chart pattern that *resembles* a grab is not a grab — edge needs a measurable forced participant.
(4) **Capture-lens lift with ~0 drift is a thin-tail artifact the engine reverses for a buy-weakness
entry** — distrust a capture-only lift; require beats-fair-null on the engine (re-usable screen). (5)
**Surviving lanes unchanged:** the two PORTFOLIO components still need a 3rd regime-orthogonal component
positive in 2015/16/18/20 (arc-2006 spec), and the long menu that would win those risk-off years is dead
→ shorts/second-leg unlock (FLAG-1) stays highest-leverage; the climax-sweep SHORT is the named first
shorts target.

**Tooling:** built + registered `discovery/tools/sweep_signals.py :: SweepReclaimReversalLongSignal`
(EXPERIMENT `SignalModule`; mask + ATR geometry only). Reused BUILT `observe_long_capture`,
`make_time_exit_predicate`, `build_null_signal_evaluation`. TOOL_REGISTRY BUILT updated.

**FLAGS (code not merged):** none new requiring the canonical core. Carries FLAG-1 (long-only blocks the
stronger leg — here the climax-sweep SHORT) + the standing `A1Config.time_exit_bars`-unwired flag (arcs
1005/3004; worked around via the BUILT `make_time_exit_predicate`). Drivers scratch
`_disco2000_work/arc2007_observe.py`, `arc2007_observe2.py`, `arc2007_wfo.py` (reproducible from the arc doc).

### arc_1013

**DEEP multi-factor directional — failed-breakdown RECLAIM (stop-run reversal) long** (chat 1000s; the
DEEP-arc dispatch). Full record: [`arcs/arc_1013_failed_breakdown_reclaim_long.md`](arcs/arc_1013_failed_breakdown_reclaim_long.md).
PORTFOLIO record: [`portfolio-candidates/arc_1013_failed_breakdown_reclaim_long/`](portfolio-candidates/arc_1013_failed_breakdown_reclaim_long/).
No council (PORTFOLIO, not a PASS survivor — arc-1011 precedent).

**Idea + why (ideation shown, per dispatch).** The shallow single-trigger directional slice is closed
(25 arcs: direction ≈ coin-flip, instrument/TF/metric-invariant). The OPEN frontier (§5a carve-out) is
DEEP, multi-factor directional. Structural *because*: resting sell-stops cluster below visible swing lows;
a bar that PIERCES a K-bar swing low (sweeps the stops) and CLOSES BACK ABOVE it (failed breakdown) with a
large rejection wick is a **stop-run / liquidity grab** — the down-move was liquidity-driven, not
informational, and the **reclaim confirms the adverse excursion is OVER *at entry*** (the mechanistic fix to
the capturability wall that killed every prior reversion — gap-fill/RSI/month-end-fix all entered INTO the
adverse move, arcs 2001/2002/3000/3001/3008). Distinct from the closed corpus: the **inverse of arc 3006**
(break-and-HOLD continuation → break-and-FAIL reversal), NOT indicator-reversion (3000/3001), and unlike
arc 0's bare pierce-the-dip the **reclaim + conjunction** is the whole point.

**What happened — the data led, and inverted my priors.** Observation (7 USD majors, H4, IS; base cap
0.4860): the bare reclaim ≈ coin-flip. My hypothesised session + D1-uptrend conjunctions FAILED/INVERTED
(reclaim is BETTER when D1 ≤ SMA50, not in an uptrend; session = noise). But DEPTH separated, inverted from
my guess: a wick that spikes >1 ATR below a 40-bar swing low and still reclaims captures **0.55–0.61**, broad
across 6–7/7 pairs, both lenses agreeing — the strongest directional capture in the corpus. **DECISIVE
CONTROL** (arc-1011 discipline): a same-magnitude rejection wick AT a swept swing low captures 0.52→0.61,
the SAME wick elsewhere stays a coin-flip (~0.49–0.51) with NEGATIVE drift, and the excess GROWS with
rejection size → the **structure is the mechanism, not the generic wick** (generic deep-down reversion is
dead). Honest engine (committed IS config K=40, shadow≥1.25, sl_plus_trailing_atr, SL=2.0): **IS mean
+1.854%, 9/10 folds positive** (only 2018 neg), and **EVERY exit is mean-positive** (+0.83% to +1.85% —
unprecedented for a directional long). Soundness: fair same-exit NULL is firmly NEGATIVE (−1.107% avg, 3
seeds) → real **beats null by +2.96pp** (cleanest null in the corpus); robust across K∈{40,60}×shadow≥1.25
(shadow=1.0 dilutes — the rejection gate is load-bearing); leave-one-pair-out positive dropping any pair.
**One-shot frozen OOS (2021+): mean +0.936%, 3/6 positive — mean-POSITIVE forward** (the edge persists,
where arc 2005 died OOS). NOT all-folds-positive (IS 2018; OOS 2022/25) — the negatives cluster in
persistent strong-USD/risk-off years where breakdowns are real, not swept (arc-1012-class regime drag, not
fished).

**Verdict: FAIL the sole judge → PORTFOLIO.** The strongest, cleanest, most-robust directional long in the
corpus (IS +1.85% vs gap-fill +0.69% / month-end +0.23%; the only one to beat a NEGATIVE null and survive
OOS mean-positive), but not all-folds-positive → the **3rd PORTFOLIO component**, of a NEW mechanism family
(structural stop-run reversal vs the two flow-event reversions).

**Convergence with arc 2007 (chat 2000s, landed concurrently) — the decisive mechanistic split.** The
2000s chat independently built the SAME family (DEEP liquidity-sweep reversal long) and reached **KILL**
(loses to the fair null) — its construction conditioned on a **fast 3-bar drop + climax**, i.e. it ENTERED
AFTER a multi-bar plunge (buying into weakness, sitting in the adverse excursion → take-the-loss converts
the fat MAE tail into −1R losers worse than random; the arc-2001/2004 reversion stop-tax). Mine reaches
**PORTFOLIO** because the **same-bar rejection wick** means the reversal ALREADY happened within the signal
bar — the i+1 entry lands AFTER the low, not inside the drop — and the **shadow≥1.25 gate + structure
control** prove the edge is the reclaim of a swept level, not "buy a fast drop." **Two chats, two
constructions of the same idea, opposite verdicts: the load-bearing distinction is WHEN you enter relative
to the adverse excursion** (same-bar confirmation = PORTFOLIO; after-the-drop = worse-than-null KILL).
Arc 2007 also found the *climax* sweep is a falling-knife SHORT edge (FLAG-1) — consistent with my dropping
the deep-but-non-reclaiming bars.

**Threads / lessons.** (1) **DEPTH was the missing axis — the dispatch's thesis confirmed.** The shallow
slice hid a deep-tail structural edge; the multi-factor conjunction (swing-low structure × pierce-reclaim
sequence × deep-rejection magnitude) reaches cap 0.55–0.61 where every shallow single-trigger cut was 0.49.
"Deep, not broad." (2) **The RECLAIM fixes the capturability wall** — the first reversion whose honest i+1
entry is mean-positive net of cost, because the confirmation lands AFTER the adverse move (the structural
answer to arcs 2001/2002/3008). (3) **My session/MTF priors were wrong and the data inverted them** —
develop-on-IS-let-data-lead earned its keep; DEPTH, which I'd guessed was "informational/down-weighted,"
was the real conditioner. (4) **Cleanest null in the corpus** (real beats a −1.1% null by +2.96pp) — a
genuine directional structural edge, not the exit/drift artifacts that thinned the flow components (cf. arc
1009's gap-fill null was *positive*). (5) **The high-value 3rd component (arc-2006 spec):** positive in
2015/16/20 — three of the four years the 2-way book bled — sharing only 2018. A 3-way combination (the
2000s' arc-2006 machinery) should cut the book's mutually-negative folds from four to ≈one (2018). **HIGH-VALUE
fleet steer for the 2000s range: run the 3-way combination WFO** (gap-fill 1006 + month-end 1011 + this),
ranking by ROI on the 2-way book's negative folds, gated all-folds-positive on the combined book. (6) **The
named regime weakness** (persistent strong-USD trend years 2018/2022/2025 where breakdowns are real) is the
honest open thread: a pre-registered, causal regime filter (NOT fished to flip 2018) could in principle lift
it toward all-folds-positive — a fresh arc, the arc-1012 trap explicitly avoided.

**Tooling:** built + registered `discovery/tools/failed_breakdown_signals.py :: FailedBreakdownReclaimLongSignal`
(EXPERIMENT signal — mask + ATR geometry only; scoring canonical). TOOL_REGISTRY BUILT updated.

**FLAGS (code not merged):** none requiring the canonical core. Carries the standing `A1Config.time_exit_bars`
-unwired flag (arcs 1005/3004; worked around via the BUILT `make_time_exit_predicate`). Drivers scratch
`_disco_work/arc1013_*.py` (reproducible from the arc doc; obs cached `_disco_work/obs_h4_usd.parquet`).

### arc_1014

**Confirmed-breakdown continuation SHORT — the short mirror of arc 1013 (the programme's FIRST short arc).**
Full record: [`arcs/arc_1014_confirmed_breakdown_short.md`](arcs/arc_1014_confirmed_breakdown_short.md). No
council (falsified at observation; no worthwhile-ceiling fork).

**Idea + why.** Shorts are now open (PR #273; engine short-symmetric, `PerPairSignalState.direction` flows
into the pool bar-walk + Step-5 emission; `observe_long_capture(direction="short")` mirrors entry/SL/label/
drift). The dispatch's lead priority is the short asymmetries the corpus flagged, and my range's own arc 1013
(strongest directional long in the corpus — deep failed-breakdown RECLAIM, PORTFOLIO 3rd) is NEGATIVE in the
strong-USD trend years 2018/22/25 *because* there "breakdowns are real, not swept." The arc-2006 portfolio
spec needs a component positive in exactly those risk-off years. Hypothesis (§5a DEEP carve-out, extend the
1013 template incl. its short mirror): the SHORT mirror — a deep pierce of a swing low that does NOT reclaim
(closes BELOW with down-momentum) = a CONFIRMED breakdown → continuation DOWN, positive when the longs bleed.
Corpus corroboration: arc 2007's climax sweep is a falling knife (drift −0.124→−0.333, worse the bigger,
FLAG-1); arc 1013 flagged the non-reclaim leg as the short. NOT a shallow breakout short (dead by symmetry) —
the multi-factor cell (depth × swing-low structure × close-below × momentum).

**What happened — FALSIFIED at observation; the data inverted the corpus prior.** H4 USD majors, IS, short
lens. **Confirmed breakdown (pierce, no-reclaim, close-below): cap 0.4893 (<0.50), drift −0.046** (price
drifts UP after = the breakdown REVERTS, bad for the short). **Deeper close-below reverts HARDER** (drift
−0.212 at 0.25-0.5 ATR below), opposite of the continuation hypothesis; only the thin extreme tail (>1 ATR,
n=623) faintly continues (+0.077, cap 0.504 = coin-flip). Down-momentum (climax) buckets are non-monotone
noise (the arc-3007 "not a lever" tell) — does NOT reproduce arc 2007's −0.33 in this clean construction.
**STRUCTURE CONTROL decisive (the 1013 discipline mirrored):** big-red bar AT a swept swing-low (cap 0.4998,
drift +0.018) ≈ big-red ELSEWHERE (cap 0.4963, drift +0.055) — the swing-low structure is **NOT load-bearing
for the short** (contrast 1013's long: AT-swept-low 0.55-0.61 vs elsewhere ~0.49). Per-pair deep cell 3/7
positive (EUR/GBP/AUD continue; JPY/CHF/CAD revert −0.4..−0.67) = the arc-1010 noise signature.

**Verdict: KILL (cheap-kill at observation).** Coin-flip-or-worse short that reverts; structure control fails,
momentum non-monotone, cross-section noise. Entry doesn't beat base + no positive gross short drift → §5f
exit-sweep doesn't bite; no pool/engine/council spent (arc-1001/1002/3006/1010 discipline).

**Diagnosis — the load-bearing asymmetry (why the 1013 long has no tradeable short mirror).** arc 1013's
reclaim-LONG wins because the reclaim is **FORWARD-confirming**: the wick swept stops and closed back above,
so the i+1 long enters AFTER the adverse low — the move it bets on hasn't started. The confirmed-breakdown
SHORT is the mirror image but the asymmetry does NOT flip: a close BELOW the swept level is
**BACKWARD-confirming** — the down-move already happened in the signal bar, so the honest i+1 short enters at
the local low, and on liquid majors (down-moves revert, arcs 3000/3001/3008) it reverts UP against the short.
Forward-confirming reversal structure (reclaim) is tradeable; backward-confirming continuation structure
(confirmed breakdown) is shorting realized weakness = the arc-2001/2004/2007 sell-into-the-move stop-tax. The
−0.33 "falling knife" (2007) was inside a fast-3-bar-drop sweep-reclaim construction (a momentum/breakout
short = closed ground), never engine-validated; the clean confirmed-breakdown doesn't carry it.

**Threads / lessons.** (1) **The arc-1013 reclaim-long does NOT mirror to a tradeable short** — forward-
confirming reversal structure is capturable; backward-confirming continuation structure is a coin-flip that
reverts on the honest i+1 short. **Shorts do not revive directional STRUCTURE** (the structural short base
0.485 ≈ the long base 0.488 by symmetry; the swing-low structure that was load-bearing for the long is inert
for the short — control-proven). (2) **The regime-orthogonal short the portfolio needs (positive 2018/22/25)
is NOT a confirmed-breakdown continuation short.** (3) **The live short frontier is the FLOW-EVENT shorts
where the corpus measured directional accuracy >0.50, NOT structural-continuation shorts** (those inherit the
coin-flip). Named candidate, the natural next arc (1015): the **up-gap weekend SHORT** (arcs 2001/2003:
up-gaps drift −0.57 ATR, frac+ 0.64 — the gap-fill's STRONGER leg, direction NOT a coin-flip, was blocked by
long-only, now unblocked). Distinct mechanism family (flow event, not structural continuation). (4) **The
arc-1013 structure control, mirrored on the short, was the decisive one-line discriminator** — "big-red AT
swept-low ≈ big-red elsewhere" killed the structural claim immediately; the right first test for any
structural short. (5) **First live short measurement in the programme** — the PR-#273 short path works
end-to-end in observation (base short cap ≈ 0.485 reproduces the long-base mirror).

**FLAGS (code not merged):** none. No canonical-core change; no new BUILT tool (reused the direction-aware
`observe_long_capture`; conditioning is a one-off scratch observer). Driver scratch
`_disco_work/arc1014_observe_short.py` (reproducible from the arc doc).

### arc_2008

**3-way PORTFOLIO combination WFO (gap-fill 1006 + month-end 1011 + failed-breakdown 1013)** (chat 2000s).
Full record: [`arcs/arc_2008_portfolio_combination_3way.md`](arcs/arc_2008_portfolio_combination_3way.md).
No council (not a PASS survivor, not an idea-fork/diagnosis — a direct quantitative combination test with
a decisive, provable result).

**Idea + why (log-seeded — the route's gated next step).** arc 2006 ran the 2-way (gap-fill + month-end)
→ KILL (mutually-negative 2015 fold + tail-correlation) and specced exactly what a 3rd component must do:
net-positive on the 2-way book's bleed folds (2015/16/18/20), selected for fold-complementarity not
avg-corr. arc 1013 then DELIVERED a 3rd — the failed-breakdown reclaim (stop-run reversal) long, a NEW
mechanism family (structural, not flow-reversion), the strongest/cleanest directional long in the corpus
(IS +1.854%, beats a NEGATIVE null +2.96pp, OOS mean-positive), positive in 2015/16/20 (three of the four
bleed years). arc 1013's explicit steer for the 2000s range: run the 3-way combination WFO. Decisive
either way — a PASS candidate (the first deployable), or "3 is still not enough + the 4th's exact spec."

**What happened.** Reproduced each component LIVE via its REGISTERED signal over the SAME canonical IS
folds (`build_v3_folds`, 10 folds, OOS 2011–2020), scored solely by `MultiPairBacktester`, FundedNext ON.
**All three reproduce EXACTLY (≤0.005pp/fold):** gap +0.685%, month-end +0.232%, failed-breakdown
+1.854%; corr(gap,me)=+0.117 (matches arc 2006 → fold-alignment re-confirmed), corr(gap,fbr)=+0.189,
corr(me,fbr)=−0.366. Combined via the BUILT `combine_fold_roi`, weights fit-on-IS-frozen:
- **equal** (1/3 each): mean +0.924%, **3/10 neg** (2013 −0.07, 2015 −0.72, 2018 −3.36), NOT
  all-folds-positive.
- **risk-parity** (gap 0.106 / me 0.726 / fbr 0.168): mean +0.552%, **2/10 neg** (2015 −0.74, 2018
  −0.77), NOT all-folds-positive.
- **C did exactly what arc 1013 predicted — and it is not enough.** It CUT the 2-way book's four bleed
  folds to TWO: fixed **2016** (C +2.55) and **2020** (C +4.03) outright, but **2015 and 2018 survive**.
- **2018 is the binding fold.** gap −6.79, month-end +0.90, **failed-breakdown −4.20** — BOTH the
  flow-reversion AND the new structural stop-run reversal lose; only the small month-end is positive. 2018
  is a persistent strong-USD trend year (arc 1013 named it as C's single weak fold: "breakdowns are real,
  not swept"). C SHARES the 2018 weakness rather than fixing it.
- **PROVABLY blocked — combination-method-invariant.** Gridded the full convex simplex (5,151 weightings):
  **0 are all-folds-positive.** The best-possible (max-min) weighting gap 0.00 / me 0.78 / fbr 0.22 STILL
  leaves min-fold −0.222% (2014/15/18 neg). Hard 2015-vs-2018 conflict: 2018 needs heavy month-end weight
  (its only positive leg ⇒ at w_A=0, w_B > 4.67 w_C); 2015 needs heavy failed-breakdown weight and is
  *hurt* by month-end (⇒ w_B < 2.78 w_C). The interval is empty — no weighting (and, P&L being additive,
  no honest co-simulation) passes both. The KILL does not rest on the linear approximation.
- IS not all-folds-positive → **OOS deliberately NOT touched** (§4 + holdout preservation).

**Verdict: FAIL the sole judge (combined 3-way book) → KILL.** Not deployable, *provably* blocked by a
2015-vs-2018 simplex infeasibility. **The three components are UNCHANGED** (reproduced exactly, not
re-tested/weakened) — they retain PORTFOLIO status; no new `portfolio-candidates/` entry (would
double-count 1006/1011/1013). The portfolio **thread stays ACTIVE**: a 4th component is a hard requirement
with a sharply narrowed spec.

**The 4th-component spec (the actionable deliverable).** Must be net-positive on the surviving negative
folds **2015 and 2018**, critically **2018** — the binding fold where BOTH a flow-reversion (gap-fill) AND
the new structural stop-run-reversal family (failed-breakdown) lose. The deeper requirement (sharper than
arc 2006): **all three current edges are fade/reversion-flavored = implicitly SHORT-trend, and tail-correlate
in a strong-trend year (2018) regardless of surface mechanism.** Average correlation hid it (gap-fbr +0.189,
me-fbr −0.366 look decorrelated) but the two large components share the 2018 tail. The 4th leg must be a
**trend-continuation / momentum** edge positive when trends persist — and the long-only trend menu that
would win 2018 is comprehensively dead (arcs 0–3006). → sharpens the arc-3004 escalation with a NAMED
binding fold (2018): the regime-orthogonal leg is structurally a short / second-leg / trend-continuation
construction (operator shorts/second-leg unlock, FLAG-1); **arc 2007's climax-sweep SHORT is the concrete
named target** (it is the short side of the very 2018 breakdowns that beat the long failed-breakdown).

**Threads / lessons.** (1) **Three near-decorrelated mean-positive components — incl. a NEW mechanism
family — are STILL not enough** when two share one un-rescuable tail (2018); "≥3 components" (arc 2006) was
necessary not sufficient — the real bar is fold-complementarity on the BINDING fold. (2) **Avg-correlation
decorrelation STILL ≠ fold-complementarity, even across mechanism families** — all long-only fade/reversion
edges are implicitly short-trend and tail-correlate in a strong-trend year; the 4th must be long-trend, not
another fade. (3) **The simplex-feasibility grid is the rigorous combination-invariant test** (extends arc
2006's mutually-negative-fold pre-screen): when no SINGLE fold is mutually-negative but the book still
fails, grid the convex hull — 0/N pass + max-min ≤ 0 proves the KILL without touching OOS or trusting the
linear approximation. Re-usable. (4) **`sl_only` is not trail-free** — `A1Config.trail_enabled` defaults
True and silently added a trailing stop to the 24-bar gap-fill (3.8pp/fold error at identical 260 entries);
set `trail_enabled=False` for a pure `sl_only`+time-exit reproduction (config-usage note, not a code flag;
B/C unaffected). (5) **Reinforces + sharpens the arc-3004 escalation:** the portfolio route is now blocked
on a single named regime fold (2018) needing a long-trend/short/second-leg leg the long-only apparatus
cannot express → shorts/second-leg unlock (FLAG-1) is the highest-leverage operator move.

**Tooling:** no new tool. Reused BUILT `combine_fold_roi` (arc 2006) + the three REGISTERED signals +
BUILT `make_time_exit_predicate`. Simplex grid is a one-off arithmetic check over the reproduced vectors.

**FLAGS (code not merged):** none new. Carries FLAG-1 (long-only blocks the trend-continuation/short/
second-leg leg the portfolio route demonstrably needs for fold 2018) + the standing
`A1Config.time_exit_bars`-unwired flag. Drivers scratch `_disco2000_work/arc2008_combo3.py`,
`arc2008_feasibility.py` (reproducible from the arc doc).

### arc_3009

**3-way PORTFOLIO combination WFO (gap-fill 1006 + month-end 1011 + failed-breakdown 1013)** (chat 3000s).
Full record: [`arcs/arc_3009_portfolio_combination_3way.md`](arcs/arc_3009_portfolio_combination_3way.md).
No council (not a PASS survivor, not an idea-fork/diagnosis — a direct quantitative combination test with a
decisive arithmetic result; arc-2006 precedent).

**Idea + why (log-seeded — the route's explicit gated next step).** After 29 arcs the directional space is
comprehensively closed, leaving THREE live net-positive long-only PORTFOLIO components: (A) weekend gap-fill
JPY-cross H4 (+0.685%), (B) month-end reversion USD-major D1 (+0.232%), (C) failed-breakdown RECLAIM
USD-major H4 (+1.854%, the strongest, a NEW structural-stop-run mechanism family). Arc 2006 combined A+B →
KILL, *provably* blocked by a mutually-negative 2015 fold; its spec called for a 3rd positive on the bleed
years. Arc 1013 landed C (positive 2015/16/20 — 3 of the 2-way's 4 bleed years) and named the 3-way
combination as the HIGH-VALUE next step. I ran it: the single most decision-relevant unrun experiment, the
deployable gate for the entire long-only PORTFOLIO thesis.

**Method (measurement CALLED; combination is the one BUILT tool).** Reproduced each component LIVE via its
REGISTERED signal over the SAME `build_v3_folds` IS folds (Arc-10 discipline — don't trust transcription),
scored solely by `MultiPairBacktester` (FundedNext ON, SL-first), then linear-combined per-fold ROI via the
BUILT `combine_fold_roi` under equal + risk-parity (fit-on-IS-frozen), judged all-folds-positive on the
combined book. **All three reproduce EXACTLY** (gap +0.685%, me +0.232%, fbr +1.854%, per-fold byte-identical)
— and the live re-run CAUGHT a config detail transcription would have hidden: the faithful arc-1013 component
needs **trail_enabled=True** (native A1 trail ON TOP OF the `sl_plus_trailing_atr` exit policy — a
double-trail; with trail off the same signal scores +2.084%/2-neg). A probe pinned the published series to
trail=True at L1=0.03.

**What happened.** Per-fold IS (folds 2–11 → 2011–2020):
```
gap (A): -0.07,+8.23,-2.06,+2.94,-4.19,+3.20,+0.53,-6.79,+7.45,-2.39   +0.685%
me  (B): +0.40,+0.29,+0.96,-0.23,-1.14,-0.51,+0.34,+0.90,+1.16,+0.15   +0.232%
fbr (C): +7.55,+3.05,+0.91,+0.19,+3.17,+2.55,+1.23,-4.20,+0.05,+4.03   +1.854%
```
- **Mutually-negative-fold pre-screen (arc-2006 lesson #4): PASSES.** Every fold has ≥1 positive component —
  the 2-way's *provable* 2015 block is GONE (fbr +3.17 rescues 2015; me +0.90 rescues 2018). The 3-way is no
  longer combination-invariant-blocked.
- **equal** (1/3 each): +2.63,+3.86,−0.07,+0.97,−0.72,+1.75,+0.70,−3.36,+2.89,+0.60 → mean +0.924%, **3/10
  neg** (2013,2015,2018), NOT all-folds-positive.
- **risk-parity** (gap 0.106/me 0.726/fbr 0.168): +1.55,+1.60,+0.63,+0.18,−0.74,+0.40,+0.51,−0.77,+1.64,+0.53
  → mean +0.552%, **2/10 neg** (2015,2018), worst −0.77%, NOT all-folds-positive.
- IS not all-folds-positive → **OOS deliberately NOT touched** (§4 + holdout preservation).

**Verdict: FAIL the sole judge (combined 3-way book) → KILL.** Components UNCHANGED (reproduced exactly,
retain PORTFOLIO); no new `portfolio-candidates/` entry. But this is the CLOSEST the route has come and a
*weaker* failure than the 2-way: not a mutually-negative fold (gone), just a weighting/magnitude shortfall
on 2 folds.

**Threads / lessons.** (1) **Three mean-positive components STILL fall short — but the route is converging,
not dead.** Adding the regime-different 3rd (structural stop-run reversal) removed 2 of the 2-way's 4 blocker
folds (2016, 2020), halved the worst fold (−1.53→−0.77 risk-parity), and dissolved the combination-invariant
2015 block. Fold-complementarity works; the book is **one component short.** (2) **2018 is the corpus's
recurring portfolio wall** — a persistent strong-USD *trend* year where flow-reversion (gap-fill, month-end)
AND structural-reclaim-long (failed-breakdown) all bleed together (gap −6.79 & fbr −4.20 deep-neg, only me
+0.90). Sharpens arc-2006's tail-correlation lesson: even a *different long mechanism family* shares the
strong-trend tail, because **all three are LONG and none profits from a sustained directional risk-off move**
— the diversifier must change SIGN-exposure (short/trend), not just mechanism family. (3) **The
mutually-negative-fold screen distinguishes "impossible" from "needs a 4th component"** — the 3-way passes it
where the 2-way failed, correctly flagging the book is worth pursuing. (4) **Risk-parity helps variance, not
the gate** (re-confirmed on 3 components). (5) **Reproduce-live caught the arc-1013 double-trail** —
Arc-10 discipline working; verdict invariant to it. **The 4th-component spec (razor-sharp): net-positive on
2015 AND especially 2018** (deficits −0.74%/−0.77% risk-parity) → structurally a **SHORT / trend /
risk-off-positive** leg (the long menu for those years is dead). **Shorts are now OPEN (PR #273)** — the
named, well-motivated first targets are the **climax-sweep SHORT** (arc 2007: violent forced sweep is a
falling knife, drift −0.33) and the **up-gap weekend SHORT** (arcs 2001/2003) — either is a high-value next
arc whose explicit job is to be positive in 2018/2015. **HIGH-VALUE fleet steer: the next arc should be a
short construction targeted at the strong-USD/risk-off regime** (the 4th portfolio leg), not another
long-only flow edge (which would inherit the same 2018 tail).

**Tooling:** no new BUILT tool — reused `combine_fold_roi` + the three registered signals +
`make_time_exit_predicate` (all CALLED; scoring canonical). The combiner's first-order-linear LIMITATION is
slightly less clean than arc 2006 (month-end + failed-breakdown share the USD-majors universe, though
disjoint TF/event-timing) — FLAGGED; the KILL is robust to it (co-sim shares risk budget, doesn't ADD return;
2018's two-deep-negatives block is additive-P&L invariant). If a future 4-way clears IS linearly, verify
under co-simulation before any `passed/` claim.

**FLAGS (code not merged):** none requiring the canonical core. Repro/doc note for the operator's deep-dive:
the committed arc-1013 `sl_plus_trailing_atr` config double-trails (native A1 trail + exit-policy trail), so
its exit-menu rows are "exit_policy + native trail," not the policy alone — a labeling clarification, not a
disposition change. Carries FLAG-1 (the regime-orthogonal 4th leg is a short/relative-value construction —
now UNBLOCKED, PR #273) + the standing `A1Config.time_exit_bars`-unwired flag. Drivers scratch
`_disco3_work/arc3009_combo3.py` + `arc3009_fbr_probe.py` (reproducible:
`PYTHONPATH=. py _disco3_work/arc3009_combo3.py`, `histdata_root=C:\Users\panap\histdata_backup`).

**Convergence with arc 2008 (chat 2000s, ran concurrently — independent reproduction).** The 2000s chat
built and ran the identical 3-way combination in parallel and reached the SAME verdict (KILL, combined book
not all-folds-positive, blockers 2015+2018) — two independent chats, two drivers, one conclusion (the
Arc-10 independent-reproduction defense, here for free). The findings are complementary: arc 2008 ran an
EXHAUSTIVE convex-weight grid (0/5151 weightings all-folds-positive; best max-min −0.222%) and proved the
block is convex-hull-wide, with the sharp mechanism that **2015-needs-fbr and 2018-needs-month-end are
mutually exclusive** (heavy month-end fixes 2018 but worsens 2015; heavy fbr the reverse). My run adds the
exact-reproduction discipline (caught arc-1013's native-trail double-trail, trail_enabled=True) and the
framing that the 2-way's *combination-invariant* 2015 block is GONE (no mutually-negative fold) — so the
3-way failure is a weighting/magnitude shortfall, not an additive-P&L impossibility. Both reduce to the same
4th-component spec and the same steer: the 4th leg must be SHORT / trend / risk-off-positive (positive in
2018), now unblocked by PR #273.

### arc_2009

**Climax-sweep SHORT (down-trend continuation) — the named 4th-PORTFOLIO-component target** (chat
2000s). Full record: [`arcs/arc_2009_climax_sweep_short.md`](arcs/arc_2009_climax_sweep_short.md). No
council (falsified at observation by three independent cheap discriminators — arc-1014/1010/3006
precedent).

**Idea + why (log-seeded — the route's named next target).** My range's portfolio route is BLOCKED on
one named fold: arcs 2006/2008 gridded the convex simplex (0/5151 weightings all-folds-positive); the
binding fold is **2018** (strong-USD trend year) where all three PORTFOLIO components (gap-fill 1006,
month-end 1011, failed-breakdown 1013) — all fade/reversion-flavored, implicitly short-trend — lose.
Arc 2008's 4th-component spec named the concrete target: **arc 2007's climax-sweep SHORT** (the
trend-continuation leg, positive in 2018). Arc 2007 (a LONG arc) had measured the CLIMAX sweep below a
swing low as a FALLING KNIFE (drift −0.124→−0.333, worse the bigger = the short leg); arc 1014 tested
the CLEAN confirmed-breakdown short → KILL but **explicitly left the climax/fast-violent-drop variant
engine-unvalidated** ("the −0.33 was inside a fast-3-bar-drop construction; the clean confirmed-
breakdown doesn't carry it"). This arc fills that exact gap, testing whether the climax-sweep SHORT is
a real, STRUCTURE-LOAD-BEARING trend-continuation short via arc 1014's structure control mirrored.

**What happened — FALSIFIED at observation (three discriminators).** 7 USD majors, H4, IS, short lens
(direction-aware `observe_long_capture`; short drift>0 ⇔ price fell ⇔ good for short; base cap 0.4849,
drift −0.0151). (Q1) The cell (swept & close_below & fast-drop & big-range) is coin-flip (cap
0.49–0.505, lift ≤+0.02) and short drift is NEGATIVE/reverting (−0.04 to −0.06; only the thin range≥2
tail +0.009) → **arc 2007's −0.33 does NOT reproduce in a clean SHORT construction.** (Q2) Climax
monotonicity INVERTS — small-range −0.155 → climax −0.038, deep pierce reverts only to +0.006 (the arc
1014 "deeper reverts harder, thin tail faintly continues," on the climax variant). (Q3) **STRUCTURE
CONTROL decisive AND anti-load-bearing:** climax AT a swept swing-low drift **−0.043 (REVERTS UP)** vs
the SAME climax ELSEWHERE **+0.128 (CONTINUES DOWN)** — the swing-low sweep is the *wrong side*:
sweeping a swing low is a REVERSAL setup (arc 1013's reclaim-long, AT-swept-low cap 0.55–0.61), so the
climax-sweep short shorts INTO the bounce. (Q4) Per-pair 3/7 positive = noise, and the split is by
quote convention (EUR/GBP/AUD-USD continue +0.28–0.31; USD-CAD/JPY/CHF revert −0.30 to −0.51) =
directional USD-beta, not structure.

**Verdict: KILL (cheap-kill at observation).** Coin-flip-or-reverting short whose only "edge" is
USD-beta quote-convention noise; entry doesn't beat base + no robust positive gross short drift → §5f
exit-sweep doesn't bite; no pool/engine/council spent. The named 4th-component target (arc-2008
climax-sweep SHORT) is dead.

**Diagnosis (why the sweep has no tradeable short, from BOTH leans).** arc 1013 proved the swing-low
sweep is a REVERSAL structure (deep pierce-and-reclaim LONG, AT-swept-low load-bearing). A reversal
structure has no symmetric short: the clean confirmed-breakdown (1014, backward-confirming, enters the
local low → reverts) and the climax/violent variant (2009, the swept low is *where the bounce starts* →
shorting it shorts into the reversal) BOTH fail at exactly the swept-low structure. arc 2007's −0.33
was a within-sweep-reclaim-long sub-bucket artifact, NOT a robust short edge — it does not survive a
clean dedicated SHORT construction with a structure control. The "elsewhere climax continues +0.128" is
shallow momentum (a big fast drop, not at any level, continues) = closed ground (momentum shorts dead
by symmetry); stripping the anti-load-bearing structure leaves a dead shallow short → not the 4th
component, no engine compute.

**Convergence with arc 1014 (chat 1000s).** 1014 found the CLEAN confirmed-breakdown short reverts and
the swing-low is INERT for the short; 2009 closes the climax/fast-drop variant 1014 flagged as
unvalidated and goes further — for the climax the structure is ANTI-load-bearing (the swept low is
where the short loses, drift −0.043 vs elsewhere +0.128). Two chats, two short constructions of the
swing-low sweep, one conclusion: **the sweep is a reversal structure with NO tradeable short mirror.**

**Threads / lessons.** (1) **The swing-low sweep is a REVERSAL structure with no tradeable short —
confirmed from both leans** (1013 reclaim-long edge, 1014 clean-breakdown short reverts/inert, 2009
climax-breakdown short reverts/anti-load-bearing). arc 2007's −0.33 falling knife was a long-construction
sub-bucket artifact. The named 4th-component climax-sweep SHORT (arc 2008) is closed. (2) **Structure
control is the one-line decisive discriminator for any structural short** (mirrors 1014) — "climax AT
swept-low −0.043 reverts vs elsewhere +0.128 continues" killed it in one read; a structural short whose
structure helps the OPPOSITE direction is dead. (3) **The 2018 binding fold's trend-continuation leg is
NOT a structural-continuation short** — both shorts the corpus can name from structure inherit the
reversal/coin-flip; the genuine trend-positive leg would be long-trend (dead, arcs 0–3006) or a
non-structural short → reinforces the arc-3004 escalation (route blocked on a leg the apparatus + now
structural shorts can't express). (4) **Per-pair USD-quote-convention split is a short-side noise tell**
— positive on XXXUSD + negative on USDXXX (or vice-versa) = directional USD beta, demand the structure
control first. (5) **Surviving short frontier (unchanged):** the FLOW-event up-gap weekend SHORT (arcs
2001/2003, accuracy 0.64) is the one short where direction is not a coin-flip — but it is a fade edge
(would inherit the 2018 fade-book tail) and is the 1000s chat's named arc-1015 lane; the regime-
orthogonal trend-continuation 4th leg remains unfound.

**FLAGS (code not merged):** none. No canonical-core change; no new BUILT tool (reused direction-aware
`observe_long_capture`; conditioning is a one-off scratch observer). Driver scratch
`_disco2000_work/arc2009_observe_climax_short.py` (reproducible from the arc doc).

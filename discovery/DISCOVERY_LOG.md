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
| 3010 | 3000s | 2026-06-05 | **Sustained-USD-strength SHORT** — can a trend-following / established-downtrend-conditioned short on USD majors provide the 2018-positive 4th PORTFOLIO leg the 3-way book (3009/2008) needs? FALSIFIED at obs (H4 USD majors, IS, direction-aware observe): short base cap 0.485 / drift −0.015 (coin-flip, mirrors the long base). **The established-downtrend regime filter INVERTS** — drift −0.127 (vs +0.067 non-DN), ALL 7 pairs negative (−0.02..−0.32): the **symmetric completion of arc 3003** (strong uptrends revert/long ↔ strong downtrends revert/short). 2018-positivity is **uncapturable regime-luck within a yearly coin-flip** (unconditional +0.152 in 2018 / +0.125 in 2015 but negative 5/10 yrs, mean −0.015; the regime filter KILLS 2015 to −0.107). The 2018 leg is NOT trend-buildable → the route's 2018 wall stands, arc-3004 escalation reinforced. Confirms arc 1014 from the trend angle: shorts revive neither directional STRUCTURE (1014) nor TREND (3010); value of shorts = genuine flow-event asymmetries (up-gap 1015, climax-sweep 2000s), not symmetric direction/trend. No engine/council spent (3003 lesson #2) | n/e | n/e | n/e | n/e | n/e | ~121k obs | FAIL (obs cheap-kill) | N | KILL |
| 3009 | 3000s | 2026-06-05 | **3-way PORTFOLIO combination WFO** (INDEPENDENT concurrent re-run of arc 2008 — same verdict, complementary findings; Arc-10 independent-reproduction). Combine gap-fill 1006 (JPY-cross H4) + month-end 1011 (USD-major D1) + failed-breakdown-reclaim 1013 (USD-major H4) under equal + risk-parity, gate all-folds-positive on the COMBINED book. **All 3 reproduce EXACTLY** (gap +0.685%, me +0.232%, fbr +1.854% — caught arc-1013's native-trail double-trail via reproduce-live, trail_enabled=True). **The 2-way's combination-invariant 2015 block is GONE — NO mutually-negative fold** (fbr +3.17 rescues 2015; me +0.90 rescues 2018). But still NOT all-folds-positive: risk-parity 2/10 neg {2015,2018}, equal 3/10. fbr rescued 2016+2020 → blockers narrowed {2015,16,18,20}→{2015,2018}, worst-fold −1.53→−0.77%. **2018 = the hard wall** (strong-USD trend yr: gap −6.79 & fbr −4.20 both deep-neg, only me +0.90) — arc 2008's convex-search proves 0/5151 weightings pass (2015-needs-fbr vs 2018-needs-me are mutually exclusive). 4th-component spec = net-pos on 2015 AND esp 2018 → a SHORT / trend / risk-off-positive leg (longs dead those yrs; shorts now OPEN PR#273; named targets = climax-sweep short 2007, up-gap short 2001/2003). Components UNCHANGED | N | n/e (OOS preserved) | -0.77% (risk-parity combined) | n/e | n/e | 260+98+208 | FAIL → KILL (combined book; components unchanged) | N | KILL |
| 2009 | 2000s | 2026-06-05 | **Climax-sweep SHORT** (down-trend continuation) — the named 4th-component target (arc 2008): the violent fast-3-bar-drop big-range bar that sweeps a swing low & closes below → continuation DOWN, the engine-unvalidated thread arc 1014 flagged (arc 2007's −0.33 "falling knife"). FALSIFIED at obs: cell coin-flip (cap 0.49–0.505) + short drift near-zero-to-REVERTING (−0.04 to −0.06; only thin range≥2 tail +0.009); arc-2007 −0.33 does NOT reproduce in a clean SHORT construction. Climax monotonicity INVERTS (small −0.155→climax −0.038; deep pierce reverts to only +0.006). **STRUCTURE CONTROL decisive + ANTI-load-bearing:** climax AT swept-low drift −0.043 (REVERTS) vs ELSEWHERE +0.128 (CONTINUES) — the swing-low sweep is a REVERSAL setup (arc-1013 long), shorting it shorts INTO the bounce. Per-pair 3/7 = noise, split by USD quote-convention (XXXUSD continue/USDXXX revert = directional USD-beta). Swing-low sweep has NO tradeable short mirror (confirmed both leans: 1014 clean + 2009 climax). 2018's trend-continuation leg is NOT a structural short → reinforces arc-3004 escalation | N | n/e | n/e | n/e | n/e | ~121k obs | KILL (obs cheap-kill) | N | KILL |
| 3011 | 3000s | 2026-06-05 | **Failed-breakout REJECTION short — the short mirror of arc 1013** (sweep a swing HIGH + reject below → SHORT; arc 1013's flagged "up-sweep is arguably the stronger leg"). **FIRST discovery short to reach the engine** (validates merged short path PR #273 end-to-end). **INDEPENDENT REPRODUCTION of concurrent arc 2011 (2000s) — same KILL, complementary method; 2011's robustness analysis corrected my initial over-claim (Arc-10 defense).** Pooled structure control LOOKED like a pass (AT-swept drift +0.278 vs elsewhere −0.171) but is a **thin-tail/pair-mix CONFOUND** (median −0.069; carried by AUDUSD/USDJPY outliers; negative excluding them — arc 2011 + my per-pair data) = arc-2009 USD-quote-beta tell. Capture **0.473<0.50** (vs 1013 long 0.55–0.61). Honest engine §5f (even granting the pooled drift): pool mean final_r **+0.0102R≈0**, all 18 exit/SL cells mean∈[−0.04%,+0.01%], **0 all-folds-positive** (best 7/10); beats fair null only **+0.021pp** (noise floor) and **NEGATIVE in 2018**. ⇒ arc 1013's reclaim-long has NO short mirror across all 3 constructions (1014/2009/2011·3011); capture (not drift) is the wall; structural-short route to the 2018 leg CLOSED. OOS preserved | N | n/e (OOS preserved) | -0.04% (best-exit IS worst-fold) | n/e | ~0.1% | 241 | FAIL (no robust edge) → KILL | N | KILL |
| 2010 | 2000s | 2026-06-05 | **Market-neutral RELATIVE-VALUE** (the arc-3004 escalation's #1 unlock — "the only lever that doesn't need to beat 0.50"), shorts now open (PR #273): long-outperformer/short-underperformer of a cointegrated pair (EUR-GBP, AUD-NZD, EUR-AUD), relative-MOMENTUM bet (arc-2003 *because*: laggard keeps lagging), candidate 2018-positive 4th leg. FALSIFIED at obs (H4+D1, the COST REALITY decides): relative-strength has ≈ZERO predictive power (corr(relstr,fwd_rel) −0.048..+0.006), frac+ coin-flip (0.43–0.55). The 2nd leg DOUBLES FundedNext cost (~6–14 bp) without doubling edge (relative move ~2–10 bp) → every H4 cell net-NEG; the only +D1 cells (EUR-GBP \|z\|≥2 +4.5bp) are THIN (~19/yr), NON-MONOTONE in \|z\| (arc-3007 not-a-lever tell), and net-NEG once full 2-leg slippage+commission counted. **CLOSES arc-2003's CONCEDE with the real reason:** relative-value's blocker was NOT long-only (now lifted) but doubled-cost-vs-coin-flip; shorts do NOT revive it. Escalation #1 unlock is itself EDGE<COST on majors | N | n/e | n/e | n/e | n/e | ~17k obs/grp | KILL (obs cheap-kill) | N | KILL |
| 1015 | 1000s | 2026-06-05 | **3-way PORTFOLIO combination WFO** — the route's deployable gate (arc 1013/2006 flag): combine ALL THREE net-positive long-only components (gap-fill 1006 JPY-cross H4 + month-end 1011 USD-major D1 + failed-breakdown-reclaim 1013 USD-major H4) under equal + risk-parity, gate all-folds-positive on the COMBINED book. All three reproduce **BYTE-EXACT** (gap +0.685%, me +0.232%, fbr +1.854% 9/10 — fbr needs `trail_enabled=True`, the double-trail quirk in 1013's committed config). The 3rd component WORKS as 1013 predicted (positive 2015/16/20) → cuts arc-2006's 2-way risk-parity 4/10-neg(worst −1.53%) to **2/10 neg (worst −0.77%, mean +0.55%)** — strongest book in the corpus — but STILL not all-folds-positive → KILL. **Provably blocked, combination-method-invariant, by 2015 & 2018:** 2015 positive ONLY in fbr (+3.17; gap/me both neg), 2018 positive ONLY in me (+0.90; gap −6.79 & fbr −4.20 both neg) → no convex weighting passes both (2015 wants fbr, 2018 wants me; opposite directions, neither solo is AFP). CONVERGES w/ arcs 2008 (2000s) + 3009 (3000s) — triple-independent same verdict/same fold; the 2008/3009 follow-ups (2009 climax-short, 3010 trend-short) already confirm the 2018 leg is NOT a structural/trend short. Components UNCHANGED (still PORTFOLIO). **Precise 4th-component spec: positive in BOTH 2015 & 2018 (strong-USD/risk-off) → structurally a SHORT/USD-trend leg, NOT long-reversion → FLAG-1 shorts/2nd-leg unlock the named blocker w/ a 2-fold acceptance test (now the live frontier = flow-event short asymmetries, e.g. up-gap weekend short).** Route is exactly ONE regime-orthogonal component from deployable | N | n/e (OOS preserved) | -0.77% (risk-parity combined) | n/e | n/e | 396+121+172 | FAIL → KILL (combined book; components unchanged) | N | KILL |
| 2011 | 2000s | 2026-06-05 | **Failed-breakout REJECT SHORT @ swing HIGH** — the TRUE forward-confirming mirror of arc 1013's reclaim-long (1014 tested the wrong/continuation mirror, 2009 the climax continuation; NONE tested the reject-reversal): price sweeps buy-stops above a 40-bar swing high then REJECTS back below with a big upper wick (bull-trap → reversal DOWN), candidate 2018-positive 4th leg. POOLED obs LOOKED like the first passing short (drift +0.171→+0.463 monotone in shadow, structure-control AT-swept +0.261 vs elsewhere −0.213) — but capture COIN-FLIP (0.47–0.51, not 1013's 0.55–0.61). **ROBUSTNESS KILLS it:** mean≫median (s≥1.25 mean +0.261 vs median −0.069 = thin-tail), excluding the 2 thin outlier pairs (AUDUSD n25 +1.26, USDJPY n31 +1.49) collapses it (s≥1.0 +0.085, s≥1.25 **−0.133 NEG**); deeper cell INVERTS negative (opposite of 1013). Structure-control "pass" was a PAIR-MIX confound; USD-quote split (arc-2009 USD-beta). arc-1013 reclaim-long has **NO tradeable short mirror in ANY of 3 constructions** (1014/2009/2011) → asymmetric long-only edge; structural-short route to 2018 fully closed | N | n/e | n/e | n/e | n/e | ~121k obs | KILL (obs cheap-kill, robustness-confirmed) | N | KILL |
| 1016 | 1000s | 2026-06-05 | **Up-gap weekend SHORT** — the dispatch's #1 short lead + arc-1015's named 2018-leg candidate: the mirror of the arc-1006 gap-fill LONG (PORTFOLIO), a weekly-open UP gap reverting DOWN (arc 2001/2003: up-gaps drift −0.57 ATR / frac+ 0.64 = the "stronger leg"), checked specifically for its 2015 & 2018 sign. FALSIFIED at obs (direction-aware observe, H4, JPY crosses + USD majors): the honest i+1 short is **coin-flip-or-adverse** — JPY up-gap cap 0.448 (<0.50) drift −0.093 (the up-gap CONTINUES up at i+1, doesn't revert), per-pair 1/5; USD majors cap 0.418 drift −0.065, 3/7. **2015 & 2018 acceptance test FAILS as a real edge:** USD majors 2015 −0.65 / 2018 −0.21 both NEG; JPY 2015 +0.36 / 2018 +1.99 LOOK positive but = **regime-luck within a coin-flip** (overall cap<0.50, drift −0.09, per-pair 1/5, tiny n, wild per-year swings — arc-3010 lesson). Diagnosis: arc-2001's −0.57 ATR was the untradeable gap-bar-OPEN (hindsight); by i+1 the reversion is spent → **backward-confirming** like arc 1014; JPY-cross carry up-drift is a short headwind. §5f doesn't bite (sub-0.50/≤0 drift); no engine/null/council spent. **The 2018-positive 4th leg is NOT in any short construction (1014 structure / 2009 climax / 2011 reject / 3010 trend / 1016 flow) — the route's 2018 wall stands** | N | n/e | n/e | n/e | n/e | ~309k obs | KILL (obs cheap-kill) | N | KILL |
| 3014 | 3000s | 2026-06-05 | **WEEKLY (W1) trend-following CONVEXITY** — the LAST untested timeframe (CTA classic; closed ground = H1/H4/D1 only) × arc-2000's convexity lens × the "2018 = clean weekly USD trend → trend-following wins" intuition, both directions (shorts open). FALSIFIED at obs (7 USD majors, D1→W-FRI resample, Donchian-8w break, 8w hold). LONG breakout: mean drift +0.07 but **median −0.027** (positive mean = thin fat-tail artifact); tail mildly trend-SELECTED (P(mfe≥5) 0.015→0.044 ~3×, fatter than H4's "generic") but **median-negative & single-pair-carried** (per-pair 2/7 +, USDJPY +1.09 / USDCAD +0.38 only). SHORT: mean +0.005/**median −0.126**. **−2018 BOTH directions** (long −0.26, short −0.07): 2018's USD strength is mean-reverting bursts → weekly breakouts WHIPSAW (re-confirms 3012 capitulation at weekly scale). ⇒ directional/trend base **timeframe-invariant THROUGH W1** (extends 1002 H4≈D1, closes the TF lever); convexity real but sub-cost at every FX-major TF (arc-2000 confirmed one horizon up); **8th dead route to the 2018 leg** — the one trend-following most predicted. No engine/null/council | n/e | n/e | n/e | n/e | n/e | 1207 breakout obs | KILL (obs cheap-kill) | N | KILL |
| 3013 | 3000s | 2026-06-05 | **Failed-breakdown RECLAIM long at session-liquidity levels (prior-DAY / prior-WEEK low)** — improve the corpus's BEST edge (arc 1013, 9/10 IS, one fold from PASS): are ICT-style time-based liquidity pools (prior-day/week extreme, where stops cluster densely) a cleaner/stronger grab than 1013's rolling-40-bar SWING low? FALSIFIED at obs (H4, 7 USD majors, deep-grab shadow≥1.25ATR). swing40 reproduces 1013 (cap **0.5815**, drift +0.191, **7/7 pairs >0.50**); **prior_day WORSE** (cap 0.552, drift **−0.062 NEG**, 5/7); **prior_week WORSE** (cap 0.562, 5/7, thin n242). **All 3 levels remain −2018** (swing40 −1.28, pd −0.72, pw −2.30). Diagnosis: the 40-bar swing low is a STRUCTURAL PIVOT (reversal point); time-based pools are swept routinely WITHOUT being reversals → dilute. **1013's level vindicated as load-bearing** (swept liquidity ≠ grab without structural support); 2018 weakness is **mechanism-intrinsic** (reversal-long fails in strong-USD whichever low is swept — re-confirms 3012). 1013 UNCHANGED (still PORTFOLIO); standalone-PASS-via-1013 route closed. No engine/null/council spent | n/e | n/e | n/e | n/e | 1185 obs (356 at 1013-level) | KILL (obs cheap-kill; 1013 level confirmed best) | N | KILL |
| 3012 | 3000s | 2026-06-05 | **Risk-off vol-EXPANSION SHORT on high-risk-beta majors (AUD/NZD)** — the candidate 2015 **&** 2018-positive 4th PORTFOLIO leg, attacked from the one un-tested conditioner: a realized-vol-EXPANSION (risk-off) trigger (dollar-smile/flight-to-quality *because*; distinct from price-trend 3010 & vol-LEVEL closed-ground). FALSIFIED at obs (H4, AUDUSD/NZDUSD/AUDJPY, direction-aware short): pooled cap **0.5025** (coin-flip) + drift mean **+0.264 / median −0.004** (thin-tail artifact, arc-2011 tell). **Acceptance test FAILS on the BINDING fold:** 2015 strong + (cap 0.654/+0.748) but **2018 robustly NEGATIVE — cap 0.369/−1.17, all 3 pairs agree** (AUDUSD −1.24, NZDUSD −1.09, AUDJPY −0.30). Diagnosis: 2018's vol spikes are **capitulation lows** → short bounces (backward-confirming, arc-2009/1016 mode); 2015's are mid-trend → continues. Vol-EXPANSION confers no directional edge either side (controls cap ~0.50) — symmetric completion of arc-1001 vol-contraction + closed-ground vol-LEVEL: vol = magnitude not direction. 6th dead route to the 2018 leg (structure 1014/2009/2011, trend 3010, flow 1016, continuation-long 2012, now vol-state short); arc-3004 escalation reinforced. No engine/null/council spent | n/e | n/e | n/e | n/e | n/e | ~57k obs | KILL (obs cheap-kill) | N | KILL |
| 2012 | 2000s | 2026-06-05 | **DEEP multi-factor trend-CONTINUATION long** (1013's forward-confirm property applied to continuation, the one untested face of dispatch item d) — the portfolio's missing 4th leg must be trend/2018-POSITIVE (a continuation long: trend-positive, a LONG so no short-mirror death). Construction: uptrend (close>SMA200 & SMA50>SMA200) × shallow pullback holding the 40-bar higher-low × strong prior impulse (SMA50 slope) × FORWARD-confirm resume (close>prior-high). FALSIFIED at obs (H4 USD majors, IS): cell cap 0.479–0.486 (≤ base 0.486, all <0.50); impulse gate ANTI-predictive (stronger trend → worse, 3003 echo). **Q2 forward-confirm does NOT rescue continuation** (resume −0.105 vs into-dip −0.156, both neg) — mechanism: forward-confirming a REVERSAL (1013) enters BEFORE the move starts (works); a CONTINUATION enters AFTER the bounce (late) → reverts. **Q3 STRUCTURE CONTROL INVERTS (decisive):** FULL conjunction 0.4787/−0.105 WORSE than generic resume-in-uptrend 0.4956/+0.007 — deep trend structure is ANTI-load-bearing (mirror-image of 1013 where structure HELPS). Per-pair 1/7 pos (USDJPY only); **2018 drift −0.233 NEG** → not the trend leg. Closes the deep-continuation-long face: 1013's forward-confirm property is REVERSAL-SPECIFIC; no structural directional bet (long OR short, shallow OR deep) yields the 2018 leg → arc-3004 escalation reinforced | N | n/e | n/e | n/e | n/e | ~121k obs | KILL (obs cheap-kill) | N | KILL |
| 1017 | 1000s | 2026-06-05 | **Carry-unwind cascade SHORT on JPY crosses** (vol-ignition conditioned) — a genuinely novel risk-off FLOW mechanism for the 2018 leg: leveraged carry force-unwinds in risk-off → JPY bought back → persistent down-cascade; short the vol-expansion big-red IGNITION fired from a built-up carry uptrend (forward-confirming, enters at the START of the unwind, unlike trend-short 3010). REAL structural effect (**structure control PASSES**: in-carry-uptrend drift +0.134 ATR / median +0.068 = continues DOWN vs same big-red bar NOT-in-uptrend median −0.178 = reverts UP) **but sub-cost**: capture coin-flip 0.50, drift on the JPY-cross cost line, **deeper cell INVERTS** (2011/3011 tell). Honest engine §5f (exit menu swept, entry shows gross drift): 5-pair best exit sl_only mean **+0.013%, 4/10 folds pos, NOT all-folds-positive**; beats fair null by only +0.044pp (noise floor, ~10× below 1006's +0.36pp). First-read "2015&2018 both +0.19%" was a **JPY currency-exposure-cap artifact** — all-JPY-quote crosses fire simultaneously in one risk-off cascade, the 2-per-ccy cap guts the clustered fires (uncapped pool 92–109/yr → capped engine 0–3/yr in 2015/16/19). Single-pair (cap removed) best AUDJPY +0.011% 6/10, beats null +0.022pp; NZDJPY dead (+0.0001pp); none all-folds-positive. **NEW lesson: a correlated-cascade signal is structurally un-scalable into a portfolio leg.** 2018 IS genuinely engine-positive (AUDJPY +0.07–0.10%, pool +1.04R — the only short positive in 2018) but tiny/un-scalable + 2015 marginal → KILL. The 2018 leg is now unfound across EVERY route (structure 1014/2009/2011, trend 3010, up-gap flow 1016, rel-value 2010, deep-continuation 2012, carry-unwind flow 1017). OOS preserved | N | n/e (OOS preserved) | +0.011% (AUDJPY best-exit, not AFP) | n/e | ~0.05% | 1204 | FAIL → KILL (sub-cost, un-scalable) | N | KILL |
| 2014 | 2000s | 2026-06-05 | **Failed-breakdown reclaim long × downtrend-regime GATE** — improve the corpus's strongest edge (arc 1013, IS 9/10, ONLY 2018 neg) toward a SOLO PASS by conditioning OUT the strong-USD regime with a PRE-REGISTERED downtrend-STRENGTH/PERSISTENCE gate (the one conditioner obs#3's binary D1≤SMA50 + arc-3013's level-variants never tried; arc-1013 thread #5's open refinement). FALSIFIED at obs (no engine/null/council). **Hypothesis INVERTED:** strong-SMA200-down-slope is the BEST bucket (+0.218, n103, reclaim bounces sharpest when stretched down), mild-down NEG (−0.245); edge strongest in BALANCED context (near-SMA200 +0.397) — a mean-reversion-when-not-trending-up texture, not a 2018 separator. **Decisive: 2018 is NOT regime-separable — it's a near-total wipeout** (18/19 trades −1R; **6/7 pairs 100% −1R**; both "best-context" 2018 trades −1.00; gate drops 58% of 2018 fires yet kept trades still −0.69). Only non-loser = USDJPY (+0.24, n2 — the lone USD-aligned pair) but ~2 trades/yr = un-scalable (arc-1017 mode). Any gate removing the neg pockets (consec_below200>100 −0.30; mild-down) DESTROYS good folds (2012 +1.14→−0.09, 2013→−0.31, 2019→−0.47, 2020→+0.12). **NEW lesson: 1013's 2018 drag is mechanism-intrinsic & entry-time-UNCONDITIONABLE** (in risk-off the failed breakdown becomes a real breakdown, reclaim doesn't hold) — independently re-confirms 1013/3013 via a new conditioner; closes the "improve 1013 toward solo-PASS" lane. 1013 UNCHANGED (still PORTFOLIO); route's 2018 wall stands (arc-3004 escalation reinforced) | N | n/e | n/e | n/e | n/e | 237 (obs) | FAIL (obs cheap-kill) | N | KILL |
| 2015 | 2000s | 2026-06-05 | **Month-end reversion SHORT — INDEPENDENT REPRODUCTION of concurrent arc 1019 (1000s) + the 4-WAY portfolio combination** (the deployable-gate test 1019 deferred to arc 1020). Same component (short the big UP-move into month-end, the direction-mirror of the proven long me 1011, shorts now open PR#273): my obs/engine numbers CONVERGE with 1019 (short cap 0.551, 2018 cap 0.818; §5f best partial-runner mean **+0.683%**, beats fair SHORT null by **+0.92pp** [1019: +0.80pp]; per-fold **2015 +0.40 / 2018 +0.86** positive in BOTH binding folds). Robustness (Arc-10/arc-2013 defense): **2018 ROBUST** (survives all LOO incl −GBPUSD +1.14, all thr; EURUSD/AUDUSD-broad) — the corpus's first SCALABLE +2018 leg; **2015 = THIN REGIME-LUCK** (the +0.40 fold = a single +8.3R GBPUSD trade; flips neg dropping GBPUSD or thr≥1.25), overall mean GBPUSD-leaning (−GBPUSD → −0.016%). **4-WAY COMBINATION (NEW, the headline):** reproduced gap/me_long/fbr headlines EXACTLY (+0.685/+0.232/+1.854) then combined; **0/1771 convex weightings all-folds-positive (still blocked)** BUT me_short is the **FIRST component to MOVE the convex-search wall** — best achievable worst-fold **−0.222% (3-way, arc 2008) → −0.124% (4-way)**, fixing 2018 (−0.77→−0.12) and nudging 2015 to −0.05. Residual block = **2015** (me_short's 2015 is luck; only fbr robustly carries 2015) → precise 5th-leg spec: a **robust +2015 mirror** of me_short (2015 = acute-event-driven, SNB/EUR-collapse, likely a different mechanism than month-end). Closest the corpus has come to deployable; components UNCHANGED (me_short PORTFOLIO via 1019) | N | n/e (OOS preserved) | -0.124% (4-way best convex worst-fold) | n/e | n/e | 145 (me_short IS) | FAIL → KILL (4-way book; me_short PORTFOLIO per 1019) | N | KILL |
| 1018 | 1000s | 2026-06-05 | **Weekend gap-FILL long on USD-NEUTRAL crosses** (EURGBP/AUDNZD/GBPCHF/AUDCAD/NZDCAD) — target the (+2015,+2018) 4th leg via a NEW principle: **decorrelation from the USD regime** (apply the proven gap-fill mechanism 1006 to crosses whose gaps are intra-bloc-news-driven, not USD-driven) rather than predicting the regime (9 routes failed). FALSIFIED at obs (H4, IS, down-gap≥0.5ATR, i+1 fill): **no robust fill edge** — cap **0.4881 (coin-flip ≈ base)**, fill drift mean +0.078 but **median 0.000** (thin-tail artifact, 2011 tell); EURGBP (cleanest USD-neutral) drift −0.014≈0. **Acceptance test FAILS: 2018 drift −0.085 (cap 0.500, NEG); 2015 +0.390 (cap 0.525 barely >coin-flip)** — USD-neutrality gave noise, not +2018. **Lesson: the gap-fill edge is JPY-cross-SPECIFIC (carry/USD-related), NOT a universal weekend-gap property — USD-neutral gaps are efficient/random-walk; removing the USD/carry exposure that makes 2015/2018 negative ALSO removes the edge (edge & tail are the SAME exposure → net-zero, can't diversify a book, §11).** 10th dead route to the 2018 leg; closes the "USD-neutral net-positive fade" sub-route. No engine/null/council. OOS untouched | N | n/e | n/e | n/e | n/e | 463 obs | KILL (obs cheap-kill) | N | KILL |
| 2013 | 2000s | 2026-06-05 | **Weekend UP-gap weekend SHORT, JPY crosses** (the gap-fill's "stronger leg", arc 2001/2003 FLAG-1; 1015's named 2018-leg candidate) — the up-gap reverting DOWN, mirror of the 1006 down-gap LONG. **CONVERGES with independent 1000s arc 1016 → KILL.** **FIRST end-to-end SHORT engine run** — validates the merged PR #273 short path (pool builds sign-correctly, costs net symmetrically, no canonical change). Obs: honest i+1 short capture clears 0.50 only in a single fragile band (≥1.0 cap 0.518/drift +0.08/median +0.146) — but **≥0.5 median −0.115, ≥1.5 INVERTS −0.22, drop-AUDJPY → −0.107** (1016 independently got cap 0.448/drift −0.093). §5f honest engine (non-coin-flip at ≥1.0, so engine-run required): mean-positive under overshoot exits (trailing_atr **+0.745%** IS, beats random-ANY-bar null +1.74%) BUT **NOT all-folds-positive (5/10 neg)** and the positive mean is **THIN REGIME-LUCK — excluding 2018(+5.08,n8)+2019(+5.92,n10) the other 8 folds avg −0.44%** (net-neg); fails the 2015 half of the spec (−0.57). Diagnosis: arc-2001's −0.57 ATR was the untradeable gap-bar-OPEN (hindsight); by i+1 reversion is spent (backward-confirming, arc 1014) + JPY-basket up-drift taxes the short. Real-but-regime-luck = KILL not PORTFOLIO (Arc-10 defense: independent reproduction + fold-decomposition caught the thin-fold over-claim). 2018-leg unfound in EVERY short (1014/2009/2011/3011 structure, 3010 trend, 1016/2013 flow, 3012 vol, 2012 cont-long). OOS preserved | N | n/e (OOS preserved) | -3.18% (best-exit IS worst-fold) | n/e | n/e | 137 | FAIL → KILL (regime-luck, converges 1016) | N | KILL |
| 1019 | 1000s | 2026-06-05 | **Month-end reversion SHORT** (the untested direction-mirror of the proven long `me`, arc 1011) — a big UP move into month-end REVERTS DOWN (same WMR-fix inelastic-rebalancing flow, short side), targeting the strong-USD binding folds 2015 & 2018. **The 4th net-positive component AND the FIRST one robustly positive in 2018** — the wall that blocked the 3-way book (1015/2008/3009) and killed ~12 prior routes. Obs: month-end up-move ≥+1 ATR short drift +0.075 (median +0.096 = NOT thin-tail), **month-end excess +0.089 ATR** vs random-day control, honest short **capture 0.5508 >0.50 (first corpus short to clear it)**, 2018 capture 0.818. Engine §5f: **mean-positive net of costs under EVERY exit** (sl_only-2bar +0.043% → partial-runner **+0.683%, 7/10**), **beats fair null +0.80pp** (5/10 vs 2/10). **2018 robustly positive** (every exit-except-tp2r, every thr 0.75–1.5, every leave-one-pair-out drop +0.59..+1.99); **2015 fragile** (thr ≤1.0, GBPUSD-leaning). corr +0.157 vs me-long, regime-complementary (short 2015 +0.40 where me-long −1.14). NOT all-folds-positive (7/10) → PORTFOLIO. 4-way combination = gated next arc 1020. OOS preserved | N | n/e (OOS preserved) | -0.91% (best-exit partial-runner) | n/e | n/e | 116 | FAIL → PORTFOLIO | N | PORTFOLIO |
| 1020 | 1000s | 2026-06-05 | **4-way PORTFOLIO combination WFO** (gap 1006 + me-long 1011 + fbr 1013 + **month-end-SHORT 1019**, the new robustly-2018-positive 4th leg) — the gated next step now a 2018-positive component exists (arc 1015/1019 spec). All 4 reproduce (me-long/fbr/me-short BYTE-EXACT; gap per-fold ~1-2pp off recorded — config nit, verdict invariant under the audited recorded gap vector). **THE 2018 WALL IS BREACHED IN COMBINATION:** 2018 was the unsolvable 3-way binding fold (positive in only weak me-long +0.90; 12+ standalone routes died on it) — me-short gives it a 2nd robust contributor (+0.86) and at the book optimum **2018 is now POSITIVE (+0.08)**; worst fold improved **7× (−0.77% 3-way → −0.115% 4-way)** = strongest book in the corpus. But **STILL not all-folds-positive → KILL: 0/all convex weightings** (robust under both gap vectors AND every me-short exit), now blocked by **2015 (−0.08) & 2016 (−0.12), both MARGINAL.** 2016 = both month-end legs negative under every exit (relies on gap+fbr, which 2018 rejects); 2015 = me-long −1.14 drag + me-short's fragile 2015 leg (+0.40). Components UNCHANGED (retain PORTFOLIO). **5th-component spec: positive in 2015 AND 2016 without dragging 2018, and NON-reversion (the gap/month-end family is 2016-saturated).** Route ~0.11% from deployable. OOS preserved | N | n/e (OOS preserved) | -0.115% (best max-min combined) | n/e | n/e | 116+121+172+260 | FAIL → KILL (combined 4-way book; components unchanged) | N | KILL |
| 1021 | 1000s | 2026-06-05 | **Broad-universe month-end-SHORT (§5f best-version of 1019) + 4-way re-combination** — does broadening me-short beyond 7 USD majors (add 5 JPY crosses + 3 EUR/GBP crosses = 15 pairs) give a robust 2015/2016 to unblock the 4-way book (arc 1020)? Obs: flow exists in all 3 groups (each beats its random-day control; JPY carries 2016 +0.327, XCROSS adds independent 2015 +0.405). **NEAR-MISS: under the partial-runner exit it LOOKED like the corpus's FIRST all-folds-positive book** (broad me-short 9/10 +1.998%; 4-way 25 convex weightings AFP, best worst-fold +0.069%). **DISQUALIFIED by robustness (Arc-10 false-positive caught):** the AFP rests on TWO in-sample selections — (1) the partial-runner FAT TAIL (under honest sl_only broad me-short is 4/6/7-of-10, 2018 NEG under te2/te3, median often <0) and (2) IS-OPTIMIZED max-min weights (risk-parity gives 8/10 worst −0.193%, equal 7/10; 4-way is 0-AFP under EVERY sl_only exit). **2016 robustly NEG in me-short under every exit (−1.18 to −1.96)** — broad universe does NOT fix 2016. Real carry: broad universe robustifies me-short's 2015 (multi-pair +0.92..+3.33 under sl_only, no longer single-GBPUSD) but at the cost of exit-fragile 2018 — a trade-off, not a clean win. KILL the book claim; components UNCHANGED; narrow 4-way (1020, worst −0.11%) remains the honest frontier. **Re-usable rule: require AFP under sl_only AND risk-parity, not just partial-runner+optimized weights.** OOS NEVER touched (disqualified on IS robustness) | N | n/e (OOS preserved) | -0.193% (risk-parity 4-way, broad, partial-runner) | n/e | n/e | 409 fires | FAIL → KILL (book = exit×weight overfit; Arc-10 near-miss) | N | KILL |
| 3015 | 3000s | 2026-06-05 | **End-of-week (Friday) position-squaring reversion** — is there a WEEKLY analog of the month-end leg (`me`, arc 1011, the one demonstrably 2018-positive mechanical-flow reversion)? Does mechanical reversion concentrate at the end-of-week squaring boundary, positive in 2015 & 2018? FALSIFIED at obs (D1, 7 USD majors, big-down-move→buy by day-of-week). **NO day-of-week concentrates reversion** — Friday cap **0.502 / drift −0.045** (vs the month-end subset of the SAME setup cap 0.511 / drift +0.193, reproducing `me`). **2015/2018 acceptance FAILS** (2018 cap 0.421/−0.346 robustly neg; 2015 cap 0.417/med −0.51); per-pair = USD-quote-beta split (4/7 = noise, arc-2009/3012 tell); best-version sweep thins below pool floor before turning + (thr1.5 "positive" rests on 2&5 samples = thin-tail). Diagnosis: `me` survives 2018 via the **hard inelastic WMR/index rebalancing mandate unique to month-end** — the weekly boundary has no forced-rebalancing driver → the calendar-flow reversion edge does NOT generalize down the calendar hierarchy (month-end ≠ week-end). **9th dead route to the 2018 leg**; the mechanical-flow well is dry at the weekly scale. No engine/null/council spent | n/e | n/e | n/e | n/e | n/e | 4225 down-move obs (261 Fri) | KILL (obs cheap-kill) | N | KILL |
| 3017 | 3000s | 2026-06-05 | **Month-End Reversion SHORT — INDEPENDENT REPRODUCTION of arc 1019** (concurrent same-idea convergence; Arc-10 defense). Sell a big UP move into month-end, bet post-fix rebalancing reversion DOWN — the unharvested SHORT side of `me` (1011), aimed at the 2015 & 2018-positive 4th-leg spec. **Signal REPRODUCES 1019 exactly** (pool n=116, gross +0.1713, capture **0.5508 >0.50** first corpus short, month-end structure-control excess **+0.0996 ATR**, 2015 drift +0.437/2018 cap 0.818; linear-regime partial-runner 7/10, 2015 +, 2018 +, beats null). **My initial KILL was an ERROR — RETRACTED → PORTFOLIO** (deferring to 1019's folder; not duplicated). The error: I passed `A1Config.risk_pct=0.005` (the `ArcPoolConfig` FRACTION) but **`A1Config.risk_pct` is in PERCENT (0.5=0.5%)** → every per-fold ROI 100×-compressed; I then misjudged the compressed real-vs-null margin (+0.012pp) against the corpus's uncompressed benchmarks (+0.36pp) as "noise floor." The **scale-INVARIANT** judgments (fold-sign pattern, real>null) agreed with 1019 all along. **TWO FLAGs (code human-gated, not patched): (1)** the `risk_pct` PERCENT-vs-FRACTION unit split between A1Config & ArcPoolConfig (A1Config default 0.005 = 0.005% silently 100×-low; the registry entry-point omits risk_pct → copies inherit the trap) — pin the canonical discovery risk_pct; **(2)** the daily-DD cap makes per-fold ROI NONLINEAR in risk_pct — at risk 0.5 the partial-runner fold-SIGNS flip (2015/2018 NEGATIVE, blow through the 5% cap) vs low-risk 7/10/2015+/2018+ → **the all-folds-positive / 2015-2018 verdict for this thin, USD-concurrency-clustered short is risk-convention-DEPENDENT**; the gated 4-way combo (arc 1020) must report risk-sensitivity, not a single-risk number. **Lesson: independent reproduction caught a 100× apparatus discrepancy → traced to a measurement convention, not the signal; scale-invariant metrics are the trustworthy cross-config judges.** OOS preserved. | N | n/e (OOS preserved) | +0.004% (linear-regime partial-runner 2015; risk-convention-dependent) | n/e | n/e | 116 | PORTFOLIO (independent reproduction confirms arc 1019; KILL retracted; risk_pct FLAG) | N | PORTFOLIO |
| 3016 | 3000s | 2026-06-05 | **Intraday session structure: Asian-range / London-open break** (continue OR fade) — the single genuinely-UNMAPPED data modality (every prior arc is H4/D1/W1 daily-scale or calendar-flow; arc 1000 only did H4 hour-CONDITIONING, not a session-RANGE construction). DST-clean (Europe/London) Asian range × first London-open break, H1 majors. FALSIFIED at obs: **the break does NOT continue** (long-cont cap **0.375**/drift −0.116, short-cont 0.396/−0.039) AND **does NOT cleanly fade** — the naive `1−cont_cap≈0.62` is an asymmetric-SL artifact; the fade's OWN honest capture is **0.386** (short up-break) / 0.383 (long down-break): cont 0.375 + fade 0.386 = 0.76 < 1 = the break entry is a **high-vol London-open WHIPSAW point where a 2·ATR SL is hit BOTH ways**. Net fade drift after entry spread (~0.08 ATR) is **~+0.02R ≈ 0**, ~10× below the ~0.125R H1 cost (~2× H4). Only positive cell EURUSD +0.19 = **1/4 pairs (single-pair noise tell)**; 2015 (+0.174)/2018 (+0.251) drift-positive but inside cap 0.385/0.433 (<0.50) → won't survive engine (arc-3004/3012 "+drift inside sub-0.50 capture collapses SL-honest"). **Intraday session structure is sub-cost — extends the H1 cost wall (1008/3008/1010) from discrete-flow events to session-RANGE; the last unmapped modality mapped dead.** Reusable caution: `1−cont_cap ≠ fade capture` (asymmetric SL). No engine/null/council spent | n/e | n/e | n/e | n/e | n/e | 7983 IS break obs | KILL (obs cheap-kill) | N | KILL |
| 1022 | 1000s | 2026-06-05 | **Failed-breakdown RECLAIM long across UNIVERSES** (JPY crosses + non-USD crosses) — does the corpus's strongest edge (fbr, arc 1013) extend off USD majors to give a decorrelated +2018 (& +2016) 5th PORTFOLIO leg, since fbr's 2018-wipeout (arc 2014) is plausibly USD-major-specific & crosses' 2018 risk-off mean-reverts? FALSIFIED at obs: **fbr is USD-major-SPECIFIC.** MAJORS reproduce 1013 (cap **0.5865**, structure excess **+0.108**, 7/7 pairs); **JPY-crosses cap 0.481<0.50, structure excess −0.001 (swing-low NOT load-bearing), 2018 cap 0.389 WORSE**; **non-USD-crosses cap 0.495≈coin-flip, structure excess only +0.035** (2018 cap 0.515 = regime-luck inside an overall sub-0.50/weak-structure universe; 4/9 pairs >0.50 = noise). Diagnosis: the stop-run-reclaim needs the **deep, coordinated stop book only USD majors carry** — extends arc-3013's "clustered stops ≠ grab without a structural pivot" to the UNIVERSE axis. **Heuristic: FLOW edges port across universes (gap-fill majors→crosses, 1006), STRUCTURAL stop-liquidity edges do NOT.** fbr-on-crosses route to the 2018/2016 leg CLOSED; 1013 stays 9/10 IS, 2018 immovable in the reversal-long family. No engine/null/council spent (coin-flip cap + non-load-bearing structure → §5d, like 1014/2009/3010/3013). OOS never touched | n/e | n/e | n/e | n/e | n/e | 237 maj / 212 JPY / 309 xUSD obs | KILL (obs cheap-kill) | N | KILL |
| 2016 | 2000s | 2026-06-05 | **DIAGNOSTIC (HEAVY council redirected the arc): is the 4-way book's residual block REAL signal or measurement-floor NOISE?** Planned a 19th +2015-leg hunt; first obs killed vol-shock fade (extreme 1-bar move CONTINUES, cap 0.20–0.35); the `/llm-council-discovery` (unanimous: Soundness lens strongest, gate-loosening the worst blind-spot) **redirected to measuring whether the block is real BEFORE hunting**. Reproduced the 4 components EXACTLY (gap +0.685/me_long +0.232/fbr +1.854/me_short +0.683) at IS-best convex w={gap 0,me_long .65,fbr .2,me_short .15} (worst −0.124%, mean +0.624%). **(1) 2015 decomp: single-component-driven** (me_long −0.742% contrib nearly cancels fbr +0.635%, net book2015 −0.047%), NOT a 4-way co-drawdown. **(2) NOISE FLOOR (decisive): bootstrap of honest per-trade P&L → book 2015 ROI −0.047% ± sd 0.71%, 95% CI [−1.22%,+1.55%], P(neg)=0.42; |worst-fold|/sd = 0.067σ** (2018 = 0.176σ); 6/10 folds within ±1σ of zero, only 2011/12/19/20 clear the floor. **(3) corr CI: ALL 6 pairwise component fold-ROI corr 95% CIs SPAN ZERO at n=10** (me_long·fbr −0.366 [−0.81,+0.34]) → the "2015↔2018 anti-correlation" motivating ~18 leg-hunts is an n=10 artifact. ⇒ **hunting a 5th leg to flip a 0.07σ fold = chasing noise / fold-painting**; the leg-hunt-to-paint-folds route hits its MEASUREMENT FLOOR. Components UNCHANGED (PORTFOLIO); book strict-gate FAIL (did NOT loosen the gate — the rejected lens). **OPERATOR FLAG: the all-folds-positive calendar-year gate on thin-component books is applied BELOW its noise floor — governance call (reconsider fold/gate resolution OR pursue components whose per-fold ROI clears the floor).** No OOS spent (book fails IS AFP → unearned). NEW lesson: quantify a marginal fold's bootstrap CI before hunting a leg to flip it | N | n/e (not earned) | -0.047% (book 2015, =0.067σ) | n/e | 0.71% (book per-fold sd) | 396+121+172+145 | DIAGNOSTIC → KILL (residual block = noise; leg-hunt route hits measurement floor) | N | KILL |
| 3018 | 3000s | 2026-06-05 | **Failed-breakdown-reclaim LONG on CROSSES** (2018-neutral decorrelated 5th-leg candidate) — the 4-way book (arc 1020) is ~0.11% from deployable, blocked only by MARGINAL 2015(−0.08)&2016(−0.12); apply the corpus's STRONGEST structural edge (`fbr` 1013, USD majors) to a DIFFERENT universe (crosses) for a decorrelated, plausibly-2018-NEUTRAL component to lift 2015/2016 (documented spec-override: arc-1020 wanted NON-reversion but 2015/2016's only tradeable edge IS reversion; the new lever is 2018-NEUTRALITY-via-universe, not non-reversion). FALSIFIED at obs (H4, JPY crosses cached + non-JPY crosses warmed). **STRUCTURE CONTROL INVERTS on BOTH universes — opposite of arc 1013:** swept-low reclaim cap **JPY 0.4633 / non-JPY 0.4248 (both <0.50)**, WORSE than a generic deep down-wick (0.5463 / 0.4620) → the swing-low pierce/reclaim REMOVES edge on crosses (anti-load-bearing). **Target-year sign WRONG:** 2016 drift JPY +0.067/med −0.708, non-JPY **−0.952**; 2018 neg both (−0.455/−0.154); 2015 either strongly neg (JPY −0.997) or a THIN-TAIL artifact (non-JPY mean +1.25/median −0.295, n=9). Diagnosis: the `fbr` edge is **USD-major-SPECIFIC** — USD-major swing-lows at dense stop-clusters are defended reversal PIVOTS (1013 control-proven, 3013 level-confirmed), but cross swing-lows are swept routinely WITHOUT being supports (thinner/momentum/carry-driven) → reclaim enters mid-cascade, a falling knife (worst on carry JPY crosses in risk-off 2015/2018, arc 1017). **Generalizes arc 1018 (cross gaps efficient) from FLOW to STRUCTURE: corpus edges are universe-specific, don't transfer off USD majors.** The 2015/2016 leg is mechanistically hard — its only edge IS reversion, and the reversion that works (USD-major `fbr`) is the one that's −2018 → 2015/2016-lift & 2018-lift want OPPOSITE universes/directions. §5f doesn't bite (sub-0.50, anti-structure, wrong sign). 1013 UNCHANGED. No engine/null/council spent | n/e | n/e | n/e | n/e | n/e | ~330 IS fbr obs (177 JPY + 153 non-JPY) | KILL (obs cheap-kill) | N | KILL |
| 3019 | 3000s | 2026-06-05 | **Forward-confirmed extreme-shock CONTINUATION** (the candidate 5th portfolio leg for the +2015/+2016 residual, arc 1020/1021 spec) — an EXTREME ≥3-ATR single-bar shock (forced flow: liquidations/stop-cascades/CB) → multi-day continuation, entered FORWARD-CONFIRMED (bar after shock breaks its extreme; the fix to the i+1-bounce death of 1016/2009/3012). Built `ShockContinuationSignal` (BUILT, kept). **Corpus's STRONGEST continuation IS result + most promising 5th-leg candidate ever:** magnitude-MONOTONE capture 0.498@1ATR (coin-flip=generic momentum dead, closed ground) → **0.589@3ATR** (edge ONLY in the extreme tail, NOT shallow-breakout); forward-confirm load-bearing (0.589>0.543 unconf>0.488 base); all 7 USD majors >0.50; honest §5f LINEAR regime EVERY exit mean-positive & beats fair null (tp_3r **9/10 +0.034, beats null +0.048 [9 vs 3]**; tp_2r 9/10 +0.051 [9 vs 1]); **+2015 AND +2016 under EVERY exit** (the exact 4-way blockers); the EXACT MIRROR of me-short (me-short +2018/weak-15-16, this +15/+16/weak-18). **HEAVY council** at the disposition fork (all 5 reviewers ranked Soundness strongest): verdict = do NOT record on IS (selection-suspect: 15 routes died vs same 2015/16/18 target; thin n=287≈26/yr; 3.0-ATR a post-hoc peak), run the cheap event-removal test then OOS first; strongest dissent = EPOCH-dependence (post-2020 shocks may mean-revert). **Test 1 event-removal:** REFUTES naive coincidence (drop SNB/CNY/Brexit/election windows → 2015 +0.188, 2016 +1.204 both still pos; big days are VARIED macro shocks: FOMC/oil-panic/NFP) but CONFIRMS thinness (2015 flips −0.049 dropping top-2/22 days; 2016 robust +0.528). **Test 2 one-shot frozen-exit OOS (tp_3r@3ATR, §5f-faithful select-on-IS-freeze): FAILS — 2/6 folds, mean −0.0022, LOSES to null (−0.005pp)** (2021/22 + [Fed-hiking/LDI shocks continued], 2023-25 neg [post-shock mean-reverted]). IS 9/10-beats-null → OOS 2/6-below-null = **EPOCH-DEPENDENT**: 2010-2020 shocks trended, 2021+ shocks revert. ALSO un-scalable (shocks cluster across pairs on macro days → FundedNext 5%-daily-DD cap breaches 0→7-84/yr from risk 0.005→0.5; daily-DD analog of arc-1017 currency-cap). **A textbook Arc-10 SAVE** — recording PORTFOLIO on the IS evidence (my initial lean) would have been a fabricated route-unblocking component; council + one-shot OOS caught it. NEW lesson: a continuation/tail edge can be epoch-specific even with a clean mechanism + clean IS null-beat → IS-mean-positive+beats-null is NECESSARY-not-SUFFICIENT; OOS before PORTFOLIO recording for epoch-vulnerable edges. OOS spent (frozen, no re-selection). Components UNCHANGED; 5th leg still unfound | N | N | +0.034 (IS tp_3r best non-fat-tail) | -0.0022 (OOS combined, loses to null) | n/e | 287 IS / 143 OOS | FAIL (OOS, loses to null) → KILL | N | KILL |
| 1023 | 1000s | 2026-06-05 | **INDEPENDENT verification of arc 2016's portfolio NOISE-FLOOR** (different method + chat; Arc-10 norm) — re-ran all 4 components through the canonical apparatus, **reproduced every headline EXACTLY** (gap +0.685% / me_long +0.232% / fbr +1.854% 9/10 / me_short +0.683%) at `risk_pct=0.005` (the low-risk LINEAR regime; at the intended 0.5 the daily-DD cap blows through → ROIs in the 100s of %, flipped signs — arc-3017 FLAG-1 reproduced), book at arc-2016 frozen weights reproduces to the decimal (mean +0.624%, worst −0.124%@2018, 2015 −0.047%, 8/10). **CONFIRMED via an ACROSS-fold lens** (vs 2016's within-fold per-trade bootstrap): worst fold **−0.167 across-fold-sd from zero** ≈ 2016's 0.176σ; **across-fold sd (0.741%) ≈ 2016's within-fold sampling sd (0.703%)** ⇒ fold-to-fold variation is essentially ALL sampling noise (decisive method-independent statement); P(≥1 neg fold in 10) = 0.887. **NEW: the book MEAN is significantly POSITIVE — t=2.66 (p≈0.026), fold-bootstrap CI [+0.22%,+1.09%], P(mean≤0)=0.** ⇒ the 4-way book is a genuine positive-expectancy edge that fails ONLY the per-year all-folds gate, and that gate sits below its noise floor. Decision-support for the operator's path A (mean/pooled/regime-block gate) vs path B (denser components); **strict all-folds gate STAYS FAIL — quantifies why, does NOT loosen it.** Components UNCHANGED (all 4 PORTFOLIO); combined-book OOS NOT spent (§5g); no council (a measurement resolving a measurement). FLAGs: arc-3017 risk-convention (load-bearing) + the gate-resolution governance question (operator's call) | N | n/e (OOS preserved) | -0.124% (book worst, =−0.167σ, stat. zero) | n/e | n/e | 260+98+208+145 | KILL (no new component; book stays strict-gate FAIL) | N | KILL |
| 2017 | 2000s | 2026-06-05 | **DIAGNOSTIC: per-component SOLO noise-floor — is any single edge "all-folds-positive up to noise," and where is the ONE real fold-level obstacle?** (follow-through on arc 2016). Bootstrap (honest per-trade P&L, seed 42) EVERY IS fold's 95% CI for each of the 4 components; classify POS(CI>0)/NEG(CI<0)/~0(spans 0). **The components split: (a) TOO THIN TO RESOLVE — gap 10/10, me_long 9/10, me_short 8/10 folds within-noise** (even gap's +8.23%/−6.79% folds span 0) → strict all-folds-positive is **VACUOUS** for them ("AFP-up-to-noise"=True but means *can't reject anything*, NOT a pass); **(b) ONE edge resolves folds — fbr** (strongest, +1.854%/yr) carries the **corpus's ONLY statistically-real negative fold: 2018 (−4.20%, n18, CI strictly <0)**; me_long has a real POSITIVE 2018 (+0.90%, CI>0). ⇒ **the route's single genuine fold-level obstacle is fbr's −2018** (NOT a missing +2015/+2016 leg — those are within-noise for every component); the book's −0.124% 2018 is me_long(+real) vs fbr(−real) near-cancellation = the arc-2016 noise residual; arc 2014 already proved fbr-2018 mechanism-intrinsic/unconditionable. **Explains the concurrent arc-1021 Arc-10 near-miss** (an AFP book that disqualified on robustness) — sub-noise folds let exit×weight optimization PAINT a pass; statistical grounding for 1021's "require AFP under sl_only AND risk-parity" rule. Key ratios (σ-units, corr CIs, CI/point) are scale-invariant → robust to the arc-3017 risk_pct convention flag; headlines reproduce committed values exactly. **OPERATOR FLAG reinforced: the AFP-calendar-year gate is structurally mismatched to this corpus — vacuous on thin components, trips on the one thick component's intrinsic single-year hole; cannot return a confident PASS.** Components UNCHANGED (PORTFOLIO). No OOS. NEW lesson: "AFP up to noise" is only meaningful where folds RESOLVE; report per-fold CI resolvability, not just sign | N | n/e (not earned) | fbr 2018 −4.20% (CI<0, only real neg fold) | n/e | per-comp per-fold CI ±2.6%/±0.7% | 396+121+172+145 | DIAGNOSTIC → KILL (no new component; route's one real obstacle = fbr-2018, mechanism-intrinsic) | N | KILL |
| 1024 | 1000s | 2026-06-05 | **DIAGNOSTIC: arc-3017 FLAG-1 (`risk_pct` "unit split") RESOLVED — it is a FRACTION; headlines are at 0.5% deployable risk.** While reproducing components for arc 1023 I hit an empirical contradiction with FLAG-1 (which claims `A1Config.risk_pct` is PERCENT, so 0.005=0.005% "100× too low", verdict "risk-convention-dependent"). Code: `live_balance.py` sizes `risk_amount = balance × risk_pct`, default `0.01 = "1% per trade"` (FRACTION, no ×100; `A1Architecture` passes it straight through). Empirical risk-sweep of committed fbr (sl_plus_trailing_atr): **PERFECTLY LINEAR across the deployable band** — 0.25%/0.5%/1% → +0.927%/+1.854%/+3.705% (per-unit-risk const ~0.927%), **fold signs STABLE 9/10**; DD-cap only bites at ~2% (8/10), 5% (7/10), catastrophic at the absurd 50% (2/10, −129%). ⇒ `risk_pct` is a FRACTION; the +1.854% headline is at **0.005 = 0.5% per trade, a normal deployable risk** (NOT 0.005%); 3017's "risk 0.5 flips signs" is just **50% risk** destroying the account, NOT a convention ambiguity. **arc-3017 FLAG-1 WITHDRAWN; my arc-1023 restatement of it as "load-bearing" CORRECTED.** Net: the whole portfolio characterization (component headlines, 4-way book, arc-2016/2017/1023 noise-floor + book-mean-positive t=2.66) is at a sound 0.5% deployable risk with stable fold signs across 0.25–1% — no risk-convention asterisk remains; deployability blocked only by the per-year-gate noise floor (operator path A/B). Process lesson: a flag contradicting a direct code-read + linear-scaling check gets diagnosed, not propagated. Code correct; no canonical change. OOS untouched; no council | n/e | n/e | n/e | n/e | n/e | fbr risk-sweep 6 levels × 10 folds | KILL (diagnostic; FLAG-1 resolved/withdrawn) | N | KILL |
| 2018 | 2000s | 2026-06-05 | **Cross-sectional month-end rebalancing reversion, USD-NEUTRAL** — attack arc-2017's ONLY surviving spec (option B: a component THICK enough that folds RESOLVE) via the one proven-AND-thickenable mechanism: rank the 7 USD majors as currencies-vs-USD by month-to-date return, long laggard/short leader AT month-end. because: `me` (1011/1019, WMR forced rebalancing) is per-pair ABSOLUTE → USD-beta-exposed (why 2015/2018 bind); a cross-sectional rank is USD-NEUTRAL by construction (common USD move cancels) → should strip the regime exposure, AND thicker (~50/yr both sides). Genuinely untested (2003/2010 rel-value NOT month-end-timed; `me` per-pair absolute not cross-sectional). FALSIFIED at obs (drift lens, gross, IS, 7 ccy vs USD, random-day control). **THESIS FAILS on the binding fold: USD-neutrality does NOT relieve 2018 — robustly NEG cross-sectionally every horizon (FWD1/2/3/5 = −14.8/−25.1/+1.1/−20.5bp).** Reason (ties arc 2017): the book's 2018 obstacle is `fbr`-2018, and `me_long` carries a REAL POSITIVE 2018 (+0.90%, directional WMR) → stripping USD beta REMOVES the +2018 help, doesn't add it (strictly worse expression of `me`). **Coin-flip + sub-cost (arc-2010 re-confirmed WITH month-end timing):** month-end timing IS load-bearing (FWD2 top1 ME-excess **+14.6bp** vs random-day ≈0 → WMR reversion real cross-sectionally) but the spread is ~coin-flip (frac+ 0.51–0.55) & small (median ~2–10bp); market-neutral needs ≥2 legs → 4-leg(top2) net-NEG every horizon (−10bp), 2-leg(top1) clears only an OPTIMISTIC 6bp cost on a single KNIFE-EDGE horizon (FWD2; FWD1/3 net-neg), net-neg at realistic ~10bp 2-leg RT. The only net-marginal cell (single-leg short-the-leader, top fwd −9bp) re-introduces USD beta = just `me_short` (already PORTFOLIO). NEW lesson: stripping a regime exposure strips the edge that lives ON it (generalizes 1018's "edge & tail are the same exposure" from universe to the cross-sectional/market-neutral axis); arc-2010 closed WITH event-timing — rel-value on majors is doubled-cost-vs-coin-flip even timed to the forced-flow event. §5d cheap-kill (coin-flip + structural multi-leg cost; §5f doesn't bite). Components UNCHANGED. OOS never touched | n/e | n/e | n/e | n/e | n/e | 130 ME obs (×7 ccy) | KILL (obs cheap-kill) | N | KILL |
| 1025 | 1000s | 2026-06-05 | **Does the corpus's ONLY fold-RESOLVING edge (`fbr` 1013) THICKEN? — arc-2017 option B via the depth lever.** Arc 2017 named the one productive spec (a component thick enough that per-year folds RESOLVE & clear zero); arc 2018 closed it for `me` (cross-sectional, multi-leg cost). Untried route: thicken `fbr` ITSELF by relaxing its deep/long trigger (shadow & swing-`K`) to fire more often. FALSIFIED. **Obs surface (16 cells K×shadow, H4 USD majors, IS): the edge degrades MONOTONICALLY with density** — capture 0.5865→~0.50, structure-excess +0.108→~0, drift +0.237→~0 as triggers shallow/shorten; the committed K40/s1.25 is the cleanest cell (deepest grab). §5f honest engine on the one non-coin-flip denser candidate (K40/s1.00, n=507 ~2× ref): best exit +0.576% **6/10** (vs ref +1.854% 9/10, headline reproduced EXACTLY), **across-fold SD RISES 3.1→4.5%** while mean falls to ⅓ → **fold-resolution proxy mean/SD craters 5× (0.604→0.127)**; beats fair null +0.870pp (REAL but diluted edge, dominated by committed deep `fbr`). **HYP-B confirmed:** 2018 coin-flip-or-wipeout at EVERY cell (obs cap<0.52 all 16; engine fold-9/2018 = −4.118%, denser ADDS 2014/2019 negatives) → `fbr`-2018 is density-INVARIANT, mechanism-intrinsic (4th confirm w/ 1013/2014/3013). **NEW lesson: for a forced-flow edge, fold-resolution & edge-strength are COUPLED through trigger DEPTH — can't trade depth for trade-count without trading edge for whipsaw variance (denser cell = lower mean AND higher SD).** Closes option-B for the BEST edge (complement to 2018's `me` closure); strengthens the unified theory (real FX edges = intrinsically-rare forced-flow reversions → thin → per-year gate structurally unsatisfiable). `fbr` UNCHANGED (still PORTFOLIO). OOS never touched | N | n/e (OOS preserved) | +0.576% denser best-exit (mean, 6/10) | n/e | 4.5% across-fold SD (denser) | 507 denser / 237 ref | KILL (fbr does not thicken; option-B closed via depth lever) | N | KILL |
| 2019 | 2000s | 2026-06-05 | **DIAGNOSTIC (council-driven): how many INDEPENDENT bets is the 4-component book? — resolve the generative-vs-governance fork.** Convened `/llm-council-discovery` at the route-fork (generative); the 5 lenses split (refine fbr-2018 via M1 reclaim-confirm / test fbr-alone-on-OOS warning of "laundering a noise-mined 100%-reversion book" / hunt novel event-anchored liquidity) but the PEER-REVIEW converged (3/5 reviewers independently) on the one quantity ALL lenses MISSED: the EFFECTIVE NUMBER OF INDEPENDENT BETS in the EXISTING book. Measured it (reused arc-2015 configs, canonical apparatus; headlines reproduce EXACTLY gap +0.685/me_long +0.232/fbr +1.854/me_short +0.683). **The council's rank-1/shared-tail/fake-mean fears are ALL REFUTED:** (1) per-fold ROI corr LOW (−0.366..+0.406, not rank-1); (2) **ENB = 3.32/4** (top eigenvalue only 38.9% of variance) → ~3 independent bets, style-homogeneous (all reversion) but statistically decorrelated via disjoint event-timing; (3) **bootstrap ENB 95% CI [2.02,3.33], P(ENB<2)=0.021** → diversification ROBUST to n=10 noise; (4) **TAIL co-movement NEGATIVE** — mean pairwise corr in the book's 5 WORST folds **−0.201** (vs +0.123 best), co-negativity NEVER >2/4, **2018 = 2-down(gap,fbr)/2-up(me_long+0.90,me_short+0.86) near-cancellation NOT a co-crash** (refutes arc-2008 "shared 2018 tail"); (5) book MEAN robustly + (risk-parity +0.589%, 95% CI [+0.120%,+1.088%], **P(mean<0)=0.004**). **⇒ the book is a genuine ~3-independent-bet, negative-tail, mean-positive PORTFOLIO; its all-folds-positive FAILURE is PURELY the arcs-2016/2017 thinness expressed at book level** — the every-calendar-year gate trips because in any year ONE thin leg dips within-noise-neg (every worst fold is a single-leg noise dip, never a co-drawdown), NOT a diversification/tail/mean problem. **RESOLVES the fork: a 5th decorrelated REVERSION leg CANNOT make the book AFP** (no diversification deficit, no tail to hedge; an added thin leg just adds another every-year trip-chance); only a THICK fold-resolving standalone (arc-2017 option B, none known = closed ground) or the operator gate-resolution call remains → **edge-hunting for the book is closed; the lever is definitively the operator gate-governance call.** Council's best NOVEL candidate (option-expiry/gamma-pin) unanimously flagged NOT OHLC-constructible (no strike/options data → collapses to round-number, dead arc 1010). Components UNCHANGED. No OOS (book fails IS AFP). NEW lesson: style-homogeneity ≠ factor-homogeneity (disjoint events → independent fold-ROIs); a book can be a sound PORTFOLIO yet uncertifiable by a calendar-year gate below its legs' noise floor | N | n/e (not earned) | book worst-fold = single-thin-leg noise dip (no co-drawdown) | n/e | risk-parity mean CI [+0.12%,+1.09%] | 396+121+172+145 | DIAGNOSTIC → KILL (book is sound ~3-bet PORTFOLIO; AFP failure = gate-resolution artifact; lever = operator call) | N | KILL |
| 1026 | 1000s | 2026-06-05 | **Session/overnight inventory-REVERSAL** — the equity overnight-intraday-reversal analog in FX (illiquid Asian-session net move REVERTS when London/NY liquidity arrives; dealer-inventory mean-reversion at the liquidity transition), a **THICK-by-design ~580/yr** candidate for arc-2017 **option B** (the one open route: a fold-resolving standalone; a 5th reversion leg can't make the book AFP, arc 2019). FALSIFIED at obs (H1, 7 USD majors, daily-ATR-normalized session returns): **corr(asian_ret, day_ret) = +0.0006 pooled** (per-pair −0.025..+0.021, MIXED) = random walk across the Tokyo→London boundary; conditional "reversal" drift NEGATIVE/coin-flip (|asian|≥0.5 mean **−0.038 ATR**, frac+ 0.463 = slight CONTINUATION not reversion; the +0.10 at |asian|≥1.0 is a thin-tail artifact, median −0.02/frac+ 0.49). The only faint reversion is the **early-London 07-09 DEEP tail** (|asian|≥1.0 +0.054 ATR, frac+ 0.552) but **17/yr (thin)** AND **~5× sub-cost** (~0.027R vs ~0.125R H1 RT). **2015 (−0.121) & 2018 (−0.045) both NEG** (wrong sign). Diagnosis: 24h FX never closes → no warehoused-inventory price concession to mean-revert (the closed-market/retail friction the equity effect needs) → the session "transition" is a liquidity gradient, efficient (corr≈0). **The one thick-by-design option-B candidate has ZERO edge at tradeable thickness; edge appears only in the thin deep tail (sub-cost) → REINFORCES the unified theory (real FX edges are intrinsically thin), doesn't break it.** Complements arc 3016 (which killed the session BREAKOUT) → intraday session structure dead in BOTH break & net-return-reversal constructions. Option B now closed on 3 constructions (fbr-thicken 1025, me-cross-sectional 2018, thick-by-design 1026). No engine/null/council (§5d coin-flip + sub-cost). OOS never touched | n/e | n/e | n/e | n/e | n/e | ~19.8k day obs | KILL (obs cheap-kill) | N | KILL |
| 2020 | 2000s | 2026-06-05 | **fbr toward solo-PASS: does M1 reclaim-QUALITY confirmation separate the strong-USD-year (2018) failures?** — the council's (arc 2019) UNRESOLVED "refine fbr-2018 via M1 reclaim-confirm" thread; a genuinely-untested ENTRY-QUALITY lever on the corpus crown jewel (`fbr` 1013, 9/10 IS, ONE fold from a deployable solo PASS), distinct from what closed fbr-2018 before (arc 2014 daily regime-gates / arc 2017 per-fold CI). Reproduced fbr exactly (K40/shadow1.25, 7 USD majors, **237 fires** = corpus standard) and measured M1 reclaim-quality WITHIN each H4 reclaim bar (no-lookahead: all M1 ≤ signal-bar close; entry t+1 open): pierce_count, last_pierce_frac, hold_min_margin, time_above, close_margin, frac_above. **FALSIFIED at obs — BOTH questions NO.** **(Q1) no entry-time M1 tell in 2018:** 2018 reclaim-quality vs other-yrs MIXED & SMALL (hold_min −0.095, close_margin −0.105 but **frac_mid_above +0.104 HIGHER** — 2018 spent MORE of the bar above the swept level; last_pierce/time_above ≈identical); the one big delta pierce_count 39 vs 66 is M1-density/vol-confounded with WRONG sign vs outcome (corr −0.091). The 2018 reclaims are if anything MORE decisive at M1 and STILL get sold. **(Q2) M1-quality does NOT predict outcome:** every metric |corr|≤0.10 w/ capture, ≤0.074 w/ drift (≈0); terciles flat/non-monotone. **No filter rescues 2018:** requiring HIGH decisiveness makes 2018 capture WORSE (close_margin/hold/frac q.50 → 2018 cap 0.27–0.30); the only "lift" (pierce hi q.50 → 0.833) thins to **n=6** (un-scalable regime-luck, arc-1017/3010 tell) while gutting good folds to 7/11 — reproduces arc 2014's "any gate removing the neg pockets DESTROYS good folds," now at M1 resolution. Diagnosis: fbr's 2018 hole is a **FORWARD-reversal** (reclaim structurally real at entry, doesn't HOLD over following days in the strong-USD trend) → finer ENTRY resolution can't see/filter a forward failure. **Triangulates fbr-2018 = mechanism-intrinsic/entry-unconditionable across THREE independent levers (2014 regime-gate, 2017 CI, 2020 M1-microstructure); definitively CLOSES the council's M1-reclaim-confirm thread.** §5f doesn't bite (refinement falsified at obs, no best-version filter to run); fbr UNCHANGED (PORTFOLIO). Strengthens arc 2019: solo-PASS-via-fbr route closed at the entry side; deployability lever = operator gate-governance call. No engine/null/council. OOS never touched | n/e | n/e | n/e | n/e | n/e | 237 fbr fires (M1-windowed) | KILL (obs cheap-kill; council M1-confirm thread closed) | N | KILL |
| 2023 | 2000s | 2026-06-05 | **M1 (DISCOVERY_DIRECTION menu): driver-shock-CONDITIONAL cross-timeframe triangulation residual** — attacks arc 3005's "residual ≈ 0", measured UNCONDITIONALLY and at H4 ONLY. `EURJPY ≡ EURUSD×USDJPY`; a big DRIVER-leg shock forces the cross — if the quoted cross re-prices with a LAG, the next-bar identity residual continues in the shock direction. EX-ANTE (all H4, 6-bar driver-return shock proxy at t-1 predicting t; 3 clean XXXUSD triangles EUR/GBP/AUD). FALSIFIED at obs: **conditional corr(driver_shock, next-bar identity residual) ≈ 0** (EURJPY −0.009 / GBPJPY −0.025 / AUDJPY +0.001; pooled −0.012, n=6849 shocks) — indistinguishable from 3005's unconditional ≈0.01, well below the 0.05 falsifier; directional continuation ~−0.01 bp (~100–200× below the 1.2–2.0 bp cross spread); the menu-literal "not-explained-by-USDJPY" residual also ≈0 (+0.007/+0.005/−0.033 = no driver-momentum leak, closed-ground momentum dead). The cross re-prices the identity WITHIN the bar even after a 1-day driver shock → **extends arc 3005's first-moment closure from unconditional to driver-shock-CONDITIONAL** (the exact untested angle). Triangulation FIRST MOMENT now fully closed; only menu-L1 (residual SECOND-moment/OU amplitude at M1→H1, into the H1 cost wall) remains. Operative frontier unchanged (2019/2022 + menu §0.2: leg-hunt closed, operator gate-call is the lever). No engine/null/council (§5d). OOS untouched | n/e | n/e | n/e | n/e | n/e | 6849 shock obs (3 triangles) | KILL (obs cheap-kill; 3005 conditional closure) | N | KILL |
| 2022 | 2000s | 2026-06-05 | **DIAGNOSTIC: can ANY 5th leg make the 4-way book all-folds-positive? — tested with the strongest REAL targeted candidate (arc-3019 shock-continuation), the one never combined before it was OOS-killed.** Questions arc-2019's qualitative terminal claim ("a 5th decorrelated REVERSION leg can't make AFP"). Reproduced the 4 components **BYTE-EXACT** (gap +0.685/me_long +0.232/fbr +1.854/me_short +0.683) + the 3019 leg (verified LINEAR scale: 0 daily-DD breaches @0.005, ROI scales 10.1×; the ~100× gap vs 3019's doc is a reporting-unit diff, sign-pattern matches: 9/10, +2015 +2016 +2018). The shock leg is the **textbook perfect 5th leg** — + in BOTH blocker folds (2015 +5.67, 2018 +2.26) AND the most DECORRELATED component in the corpus (corr −0.60 me_short, −0.42 me_long, −0.34 fbr). **DECISIVE SPLIT: it makes the 5-way AFP ONLY under IS-OPTIMIZED convex weights (1383/10626, leaning 50% on shock = arc-1021 weight-painting); under HONEST weights it is STILL NOT AFP** — risk-parity worst −0.084% (1 fold: 2018; narrows the 4-way −0.124%), equal −1.393%. **NEW mechanism — the WEIGHTING DILEMMA:** the only leg that lifts a deep blocker fold is a TAIL/event edge → high per-fold variance (sd 3.78% vs me_long 0.71%) → risk-parity throttles it to ~9% weight → its +2018 contributes only +0.20%, too little; the weight that WOULD clear AFP is overfit. **Even a perfectly-targeted, maximally-decorrelated leg fails honest weighting.** Feasibility frontier: an added leg needs ~+0.70%/blocker @w5≈0.15 (≈ noise floor) at MODERATE variance — but +both-blocker candidates are either high-variance (shock, throttled) or thin regime-luck (me_short 2015). **CONFIRMS+SHARPENS+EXTENDS arc 2019 (reversion→non-reversion); leg-hunt route now closed STRUCTURALLY, not just by induction.** 3019 itself stays dead (OOS-epoch + un-scalable). Components UNCHANGED. **OPERATOR FLAG reinforced (structural reason): the lever is the gate-governance call, not more leg-hunting; the book's OOS-under-path-A remains unspent (a §5g firewall decision for the operator, not an autonomous chat).** No OOS, no council | N | n/e (not earned) | 5-way risk-parity worst −0.084% (2018); 4-way −0.124% | n/e | 3.78% shock per-fold sd | 396+121+172+145+232 | DIAGNOSTIC → KILL (no new component; leg-hunt route structurally closed) | N | KILL |
| 3020 | 3000s | 2026-06-05 | **fbr × cross-pair BREADTH gate — close the LAST untried entry-time axis on the route's one real obstacle (`fbr`'s −2018, arc 2017).** because: in USD-wide risk-off MANY majors should print a failed-breakdown-reclaim on the same days (broad clustering) and those broad reclaims FAIL (real dollar trend, not a per-pair grab); idiosyncratic fires hold → a LOW-BREADTH gate might drop 2018's clustered false-reclaims & give fbr its first solo-PASS. Distinct from 2014 (per-pair downtrend) / 2020 (M1) / 1025 (depth) / 3013 (level). FALSIFIED at obs (no engine/null/council). Reproduced fbr EXACTLY (237 fires, cap **0.5865**). **PREMISE FALSE — 2018 is NOT more clustered:** 2018 breadth(±3d) 0.737 ≈ non-2018 0.560, share-high(≥2) 0.158 ≈ 0.156 → breadth cannot select the 2018 regime. **Real SECONDARY texture (genuine, in the hypothesized direction):** solo fires (0 others ±3d, n140) drift **+0.519** vs low(1,n60) −0.354 vs high(≥2,n37) −0.452 — clustered fbr = USD-wide-move = less-reliable reclaim. **But no solo-PASS:** SOLO-only is 8/11 yrs drift>0 — makes 2018 + (+0.59) but BREAKS 2010/2012/2016, the 2018 fix is **n=8 regime-luck**, solo edge carried by 2 pairs (EURUSD +2.20/USDCAD +1.17; USDJPY n35≈0) = exactly arc-2014's "any gate that fixes 2018 breaks good folds," now on the breadth axis. Diagnosis: 2018 fails per-pair FORWARD (reclaim doesn't HOLD over following days, arc 2020), not by clustering → entry-time breadth can't see it. **5th independent confirmation fbr-2018 is mechanism-intrinsic/entry-unconditionable** (regime 2014 / M1 2020 / depth 1025 / level 3013 / breadth 3020); improve-the-best-edge & solo-PASS-via-fbr lanes closed at the entry side. Route lever = operator gate-governance call (arc 2016/2017/2019/1023). fbr UNCHANGED (PORTFOLIO). OOS never touched | n/e | n/e | n/e | n/e | n/e | 237 fbr fires (breadth-windowed) | KILL (obs cheap-kill; breadth axis closed) | N | KILL |
| 2021 | 2000s | 2026-06-05 | **ADVERSARIAL temporal-stability stress of the 4-component book's mean-positive edge** — arcs 2016/2017/1023/2019 characterized the book's NOISE but never its behavior in TIME; before anyone leans on the +0.589% mean (arc 2019) for a deploy call, the first quant question: is the edge STABLE across the IS decade or FRONT-LOADED in 2011–2015 and decayed since? Conservative bias = try to BREAK the mean. Reproduced the 4 components EXACTLY (arc-2019 frozen configs; gap +0.685/me_long +0.232/fbr +1.854/me_short +0.683 ✓), split per-year (2011–2020) risk-parity book into EARLY(11–15)/LATE(16–20), bootstrapped half-means + decay CI. **The adversarial hypothesis FAILS — the edge is temporally ROBUST:** (1) LATE half still mean-positive **risk-parity +0.404% (P(<0)=0.060), 1/5 neg (2018 only)** — does NOT die after 2015; (2) decay EARLY +0.773 vs LATE +0.404 = +0.365% but **95% CI [−0.636,+1.326] spans zero, P(decay>0)=0.76** — cannot reject late≥early; (3) components decay HETEROGENEOUSLY — fbr front-loaded (early +2.975→late +0.733) but **me_long STRENGTHENS late (+0.055→+0.409)**, gap +0.97→+0.40, me_short +1.11→+0.26 → the book's time-stability is the SAME ~3-bet decorrelation (arc 2019) now on the TIME axis (me_long backfills as fbr fades), not a single front-loaded factor. **Honest caveats (conservative):** n=5/half = low power (so "no decay" is weak not strong); late robustness leans on risk-parity (equal-weight late +0.449% but P(<0)=0.249); fbr (strongest leg) IS materially weaker recently. ⇒ the book's mean is NOT a decayed 2011–2015 artifact — **REMOVES one objection to the operator's path-A and modestly strengthens the deploy case**, while changing NOTHING about deployability (still fails strict AFP; no gate altered; characterization not verdict). Completes the book map on the time axis. Components UNCHANGED (PORTFOLIO). No OOS, no council | n/e | n/e (not earned) | LATE-half book +0.404% (P(<0)=0.060) | n/e | n/e | 396+121+172+145 | DIAGNOSTIC → KILL (no new component; book temporally robust but stays strict-gate FAIL; informs operator gate call) | N | KILL |
| 1029 | 1000s | 2026-06-05 | **O1: inelasticity-STATE conditioning via the calendar-flow-DENSITY proxy** (strategist MENU #2-EV — the one item attacking the book's actual blocker, per-year fold resolution; tested via its single council-rated-LIVE proxy) — concentrate the proven month-end reversion onto its highest forced-flow-density windows (quarter-end Jun/Sep, fiscal-YE Mar/Dec), betting reversion edge-per-trade rises MONOTONICALLY with density. FALSIFIED at obs (pooled me_long+me_short reversion trades, D1 USD majors, IS, n=248). **Capture NOT monotone — small-sample artifact:** tier capture 0.521/0.543/0.600 looks monotone but T3=0.600 is **March(n=10,cap 1.000=fluke) diluted by December(n=25,cap 0.440=WORST, below coin-flip)**; the 2 fiscal-YE months DISAGREE; by-month is noise (ordinary January highest 0.769, quarter-ends Jun/Sep 0.56/0.50). Drift non-monotone (T2 0.187 > T3 0.076) = arc-3007 not-a-lever tell. **Decisive — concentration WORSENS fold resolution: hi-density(tier≥2) 7/11 neg-years vs full pool 2/11** (the council's predicted death #1: thinner sample → higher per-year variance → more gate trips; full denser pool is MORE fold-stable). Calendar density does NOT concentrate the edge; re-confirms 2016/17 (too thin to resolve) + 1025 (can't thicken). O1's live proxy dead; spread-z (cost-trap-flagged) the only untested O1 sub-thread. No engine/null/council; OOS untouched | n/e | n/e | n/e | n/e | n/e | 248 me reversion trades | KILL (obs cheap-kill) | N | KILL |
| 1028 | 1000s | 2026-06-05 | **Q1: central-bank PEG / boundary-defense persistence** (strategist MENU long-shot; the divergent council seat's "new forced actor" — a persistent non-reverting flow you JOIN, vs the corpus's reversions) — fade touches of a defended boundary for a THICK high-win-rate edge (the thickness the book lacks). Tested on the corpus's one clean defended band: **SNB EURCHF 1.20 floor (2011-09..2015-01), in-IS, H4 cached.** FALSIFIED at obs: in-regime near-floor-touch honest capture **0.526 ≈ out-of-regime 0.525 (coin-flip)**, in-regime fwd-drift **median −0.237 ATR** (the +1.19 mean is thin-tail). A hard floor **PINS** price at 1.20 (absorbs selling, no upward bounce) → truncated distribution, NOT a fadeable mean-reversion. The causal vol-collapse "lift" (cap 0.598) is **generic low-vol reversion** (detector fires EVERY year, not 2012-14) = closed ground. **Un-gateable + un-hedgeable even if it worked:** single pair-epoch (~3 yrs, can't populate per-year folds, absence-FAIL cf. 3019) ending in a **−18.8% / ~182-ATR gap-through-stop break** (Jan 15 2015) the −1R invariant flatters in backtest but cannot hedge live. Cost not the killer (in-regime ~0.048 R). The join-a-persistent-forced-actor inverse-of-reversion idea is dead at obs. No engine/null/council; OOS untouched | n/e | n/e | n/e | n/e | n/e | 346 in-regime near-floor bars | KILL (obs cheap-kill) | N | KILL |
| 1027 | 1000s | 2026-06-05 | **M1: driver-shock-CONDITIONAL cross-rate triangulation residual** (strategist MENU #1-EV `explore-now`) — attack arc 3005's UNCONDITIONAL/H4-only closure: condition the triangular residual `r=log(XXXJPY)−log(XXXUSD)−log(USDJPY)` on a large D1 driver-leg shock (`|Δclose/ATR|>1.5`), measure the next-H4-bar residual; if the quoted cross re-prices with a lag, `corr(shock, next-bar resid)>0.05`. FALSIFIED at obs (4 triangles EUR/GBP/AUD/NZD-vs-JPY, IS): pooled corr **0.0135** (within-triangle-standardized 0.0144) ≈ 3005's unconditional ~0.01, below the 0.05 falsifier; **per-triangle corrs sign-INCONSISTENT (+0.112/−0.196/+0.029/+0.152)** — a real shared lag would be same-signed across the 4 crosses (shared USDJPY leg) → mixed signs within ~2SE = noise; residual stays **sub-spread** (median 0.168 bp, >0.8 bp on only 16.5% of shock bars). Apparatus reproduces 3005 (unconditional conv-corr 0.00–0.04). A shock bumps the residual VARIANCE (std 1.1–1.3 vs 0.85–1.1 bp) not its directional MEAN → that's L1's second-moment thread, not M1. Triangulation now closed BOTH unconditionally (3005) AND driver-shock-conditionally (1027); only L1 (OU amplitude at finer res) untouched but cost-walled. No engine/null/council; OOS untouched | n/e | n/e | n/e | n/e | n/e | 423 shock bars (4 tri) | KILL (obs cheap-kill) | N | KILL |
| 3021 | 3000s | 2026-06-05 | **PATH-B TRACTABILITY DIAGNOSTIC — how many decorrelated legs would the per-year AFP gate need?** The corpus's sole deployability lever is the operator's path-A (change the gate) vs path-B (denser book) call, left QUALITATIVE by every prior arc. Portfolio math (never checked): averaging N decorrelated positive-mean legs shrinks book per-fold variance → P(AFP)=Φ(per-leg-Sharpe·√N)¹⁰ rises with N → AFP reachable at SOME N even with thin legs; the real question = at what N, and is it tractable? Reproduced the 4 components EXACTLY (gap +0.685/me_long +0.232/fbr +1.854/me_short +0.683); calibrated per-leg per-fold **Sharpe 0.14–0.60 (median 0.42)**, mean pairwise corr **+0.115**. Gate needs **book** per-fold Sharpe ≥**2.31** for P(AFP)≥0.9. **RESULT — path-B closed: (1)** decorrelated (ρ=0) needs **N≈30** median-quality legs (15 fbr-class, 22 @0.50) — at ~1 PORTFOLIO leg/14 arcs + frontier closed = hundreds of arcs, intractable; **(2) DECISIVE: at realistic residual correlation ρ=0.1 (BELOW empirical +0.115) P(AFP) PLATEAUS ~0.33 and NEVER reaches 0.9 at ANY N** (ρ=0.2→0.14) — a shared dollar/risk-regime factor (arc 2008's 2015/2018 tails) floors book variance at ρσ², so book Sharpe ceilings at (μ/σ)/√ρ < gate. Densification CANNOT satisfy the per-year gate. **Sharpens arc 2019's "5th leg can't help" from fold-painting into a portfolio-math PROOF** (independent method: variance-sim vs ENB, same verdict); converts the operator flag into a number: **path-A (gate-resolution) is the ONLY route to deployability** (with arc 2021 temporal-robustness + arc 1023 t=2.66 as support). Gate NOT loosened; components UNCHANGED (all 4 PORTFOLIO); no OOS, no council. Caveat: equicorr/normal idealization, but the variance-flooring conclusion is distribution-robust + optimistic about the tail that trips the gate | n/e | n/e | n/e | n/e | n/e | 4 legs ×10 folds + MC sim | DIAGNOSTIC → KILL (path-B quantitatively closed; lever = operator path-A) | N | KILL |
| 2024 | 2000s | 2026-06-05 | **Full-month-conditioned, dual-direction month-end reversion — option-B thick-standalone test** (the one autonomous route left: a thick fold-resolving standalone, arc-2017 option B, via the ONLY mechanism that carries the binding 2018 fold = `me`). because: the WMR month-end fix rebalancing flow is documented to scale with the MONTH'S currency appreciation (equity-hedge rebalancing), so the full-month (~20 D1-bar) return should be a cleaner trigger than `me`'s 2-day `into_bars=2`; a UNIFIED dual-direction fade (long if the currency fell over the window, short if it rose) is ~2× thicker → could RESOLVE folds while inheriting +2015/+2018. FALSIFIED at obs (7 USD majors, D1, IS; canonical `_month_end_into_move` + `observe_long_capture`; into_bars∈{2,5,10,20}×thr∈{1.0,0.5} vs random-day same-move control). **Month-end timing LOAD-BEARING at every window** (month-end EXCESS drift +0.13..+0.32 ATR vs the +0.01..+0.04 random-day control, n=5–18k — reproduces arc-1011 control dual-directionally). **Q1 REFUTED: the full-month trigger is NOT cleaner** — capture non-monotone, peaks ~into5 (0.5548), into2/20 both ~0.53-0.55; `me`'s into_bars=2 is vindicated as ~optimal. **Q2 DECISIVE — no cell is option-B; clean window×2018 mechanism:** 2018 drift decreases MONOTONICALLY with window length, ROBUST under both thresholds (thr1.0 into2 **+0.045**→5 −0.029→10 −0.084→20 **−0.170**; thr0.5 +0.083→+0.021→−0.285→−0.288). 2018 was a strong-USD TREND year → a 2-day move into month-end isolates the fix over-extension, a 20-day move IS the trend → fading it = fading a real trend → loses precisely in 2018. **Q3:** thickening does NOT collapse to coin-flip (cap 0.51-0.55 from n=240→783, unlike fbr arc 1025) but buys NOTHING toward AFP (fold-res stays 7-9/11, never all-folds-pos; the only +2018 cells are the thin 2-day = existing `me`, sub-cost +0.045 ATR « ~0.05-0.10R hurdle). No new best-version entry → §5f doesn't bite (the only non-coin-flip cell IS the engine-tested `me` PORTFOLIO); §5d cheap-kill (mirrors arc 2014/2020). **Closes: the full-month-trigger refinement of `me`; single-leg option-B for `me` via the window lever** (4th option-B construction closed, w/ 2018 cross-sectional + 1025 fbr-depth + 1026 thick-by-design). **NEW lesson: the month-end reversion is a SHORT-horizon (2-5 bar) fix-flow over-extension, NOT a monthly-appreciation rebalancing effect — lengthening the trigger window contaminates with trend and destroys the strong-USD 2018 fold; the "edge scales with the monthly move" economic model is the WRONG model for the tradeable signal.** Operative frontier unchanged (operator path-A gate-call is the lever, arc 2019/3021). Components UNCHANGED (me legs PORTFOLIO). No engine/null/council. OOS untouched | n/e | n/e | n/e | n/e | n/e | 240–783 unified-fade obs (×8 cells) | KILL (obs cheap-kill; full-month trigger + option-B-via-window closed) | N | KILL |
| 3022 | 3000s | 2026-06-05 | **COST-ROBUSTNESS (cost cushion) of the 4-component book — the last unmapped axis after noise (2016/2017/1023), time (2021), densification (3021).** Every "sub-cost" kill AND the book's "mean-positive **net of costs**" pass is measured at ONE cost point (FundedNext); the deploy conclusion's robustness to that single assumption was never checked, and the book's *net* edge is small vs gross → cushion could be thin. Re-net each component's gross per-fold `RunResult` at a κ-multiplier of the FundedNext cost VECTOR (BUILT `scaled_fundednext(κ)`; cost MATH canonical via `build_fold_stats_from_run`, no engine re-run — A1 trade set is gross-geometry-invariant). All 4 components reproduce **BYTE-EXACT** at κ=1 (gap +0.685/me_long +0.232/fbr +1.854/me_short +0.683; risk-parity book **+0.589% = arc 2019 exact**) — caught that default per-currency=2 cap is load-bearing (None-override inflated gap to +1.23%). **RESULT — the book's mean edge is COST-ROBUST, NOT knife-edge: break-even κ = 3.32** (book mean→0 only at 3.3× FundedNext cost; cost eats **30% of gross, 70% survives**; positive out to ~3× FN). **AND the AFP failure is NOT a cost artifact — the book is 2/10 NEG even COST-FREE (κ=0)** → cost is not the wall, the per-year gate vs thin components is (independent confirmation of 2016/2017 from the cost angle; n_neg climbs 2→7 across κ 0→5 but the failure pre-exists at zero cost). Per-component cushion ordering is mechanism-meaningful: gap **2.10** (cost-FRAGILE, 47.6% drag, H4/frequent/JPY-cross — the corpus's most cost-exposed leg, ties arc 1009 fragility) < me_long 2.38 < fbr 3.84 < me_short **5.84** (17% drag, D1/infrequent/overshoot); risk-parity down-weights the fragile gap (0.077) so the BOOK cushion (3.3) > its weakest leg — variance-optimal weighting is also cost-optimal here. **Net for path-A: the deploy case now has 3 robustness legs — mean-positive (t=2.66, 1023) + temporally-stable (2021) + cost-robust (this arc) — and AFP-failure is confirmed cost-independent; path-A unblocked of the cost objection, path-B stays closed (3021).** NEW lesson: separate cost-robustness-of-the-MEAN from gate-satisfiability (a book can survive 3× cost yet its per-year gate be cost-SENSITIVE — moot once the gate fails cost-free). Components UNCHANGED (all 4 PORTFOLIO); gate NOT loosened; no OOS, no council | n/e | n/e (not earned) | book κ=0 −0.265% (2/10 neg even cost-free) | n/e | book break-even κ=3.32 | 396+121+172+145 | DIAGNOSTIC → KILL (no new component; book cost-robust but stays strict-gate FAIL; informs path-A) | N | KILL |
| 2025 | 2000s | 2026-06-05 | **GOTOBI (五十日) Tokyo-fix settlement flow** — a documented, calendar-anchored, regime-ORTHOGONAL intra-month forced flow distinct from arc 3008's month-end London WMR fix: on days÷5 (5/10/15/20/25+month-end) Japanese corporates settle in USD → USD-buying into the 09:55 JST (00:55 UTC) Tokyo fix → USD-vs-JPY drifts up the Tokyo morning. A thick (~6/mo), calendar-driven candidate component if > cost. FALSIFIED at obs (4 JPY pairs USDJPY/EUR/GBP/AUD-JPY, H1, IS; the 00:00-UTC fix-spanning bar open→close vs non-gotobi). **The effect is REAL & mechanism-consistent:** gotobi-day fix-bar mean POSITIVE on ALL 4 pairs (+0.10..+0.28p) while non-gotobi Tokyo mornings are NEGATIVE (−0.13..−0.97p) — normal Tokyo morning drifts down, gotobi flips it up; gotobi EXCESS +0.24..+1.14p (strongest on JPY crosses; EURJPY even +2018 +1.05p, 8/11 folds). **DECISIVE: ~10–60× SUB-COST** — drift +0.1..+0.28p « ~3–6p H1 RT cost on every pair; frac+ ≈0.50-0.52 (tiny mean, not a directional bet). Resolution-invariant (M1 captures only the pre-fix leg, still « cost; cf. arc-2002 TF-invariance). §5f doesn't bite (no above-cost entry). **Closes the Gotobi/intra-month Tokyo-fix flow, complementing arc 3008 (month-end WMR fix) → the FIX-FLOW FAMILY is real-but-sub-cost (2 independent fixes, same verdict).** NEW lesson: pure calendar-fix flows are sub-pip-to-~1-pip = an order of magnitude below FX spread+commission → un-capturable net; the corpus's surviving edges (gap/me/fbr) clear cost only because they condition on a LARGE (≥1 ATR) displacement. Operative frontier unchanged (operator path-A). Components UNCHANGED. No engine/null/council. OOS untouched | n/e | n/e | n/e | n/e | n/e | 561 gotobi / 2287 non-gotobi bars ×4 pairs | KILL (obs cheap-kill; fix-flow family sub-cost) | N | KILL |
| 2026 | 2000s | 2026-06-05 | **Japanese FISCAL-YEAR-END (31 March) repatriation** — a multi-WEEK, LARGE-displacement, calendar-anchored forced flow (distinct from the sub-cost intraday fix flows 2025/3008, and from price-triggered `me`): Japanese institutions close books 31 March → repatriate foreign assets / settle hedges → JPY-BUYING into the fiscal year-end → JPY-pairs should FALL (JPY strengthens) into 31 March. The arc-2024/2025 hypothesis (a LARGE-displacement forced flow CAN clear cost) tested on a new instance; candidate risk-off-ish decorrelated component. FALSIFIED — **hypothesis INVERTS at obs** (4 JPY pairs, D1, IS, last-10-bar JPY-strength drift, March vs other month-ends). **JPY WEAKENS into March-end, the OPPOSITE of repatriation-strength:** March jpy_strength NEG on all 4 (−0.10..−0.59 ATR), March EXCESS NEG on all 4 (−0.10..−0.62) — robust across the JPY leg; **AND coin-flip per-year** (March>0 only 4-6/11; frac+ 0.36-0.55; n=11/pair thin). Mechanism: the well-known scheduled repatriation is FRONT-LOADED into Feb/early-March → by the actual year-end the JPY-buying is done and the late-March approach sees squaring/reversal → JPY weakness = a textbook "the flow everyone knows is in the price" (2015 is the lone +JPY-strength March on all 4, an SNB/EUR-collapse idiosyncratic risk-off, not a March tell; 2018 mixed-neg). Wrong-sign + coin-flip + thin → no above-cost entry, §5f doesn't bite. **NEW lesson — the TWO failure modes of known-calendar forced flows: (i) SUB-COST (displacement too small — the fix-flow family 3008/2025) or (ii) PRICED-IN (displacement large enough but the flow is anticipated → front-run → realized move at the date is coin-flip-to-reversed — this arc).** The corpus's surviving edges (gap/me/fbr) avoid BOTH by keying off a SURPRISE displacement (weekend gap / ≥1-ATR move into the fix / deep stop-sweep), NOT a known calendar date → tradeability needs LARGE displacement AND not-fully-anticipated; a known date supplies at most one. Sharpens the arc-2024/2025 unifying read. Operative frontier unchanged (operator path-A). Components UNCHANGED. No engine/null/council. OOS untouched | n/e | n/e | n/e | n/e | n/e | 11 March / 121 other month-ends ×4 pairs | KILL (obs cheap-kill; hypothesis inverted / priced-in) | N | KILL |

---

## Per-Arc Reasoning (free-form)

Each arc appends under its own `### arc_<id>` header — the why/because, the approach taken and its rationale, what was tried, what didn't help, and threads worth pursuing. Append freely; empty until the first arc lands.

### arc_3022 — cost-robustness (cost cushion) of the 4-component book
**Why this arc.** After 3021 (path-B proven closed) the corpus is fully converged and the sole
deployability lever is the operator's path-A (gate-governance). Before anyone leans on the book's +0.589%
mean for a path-A deploy call, one assumption has been held fixed and never swept: **the cost model.**
Every "sub-cost" KILL in the corpus and the book's "mean-positive **net of costs**" pass is measured at
FundedNext default. The book's net edge is small relative to gross, so the cost cushion could be thin —
in which case path-A is moot (a book that dies at 1.1× cost isn't deployable under any gate). This is the
last unmapped book axis (noise → 2016/2017/1023, time → 2021, densification → 3021, **cost → here**), and
it is an adversarial stress in arc-2021's spirit (try to BREAK the deploy case on a new axis).

**Method that made it cheap + faithful.** The engine is gross; cost is netted post-hoc. In A1 the
entry/SL/exit geometry is gross-price-based → the trade SET is invariant to cost. So I run each component's
gross fold ONCE and re-net the same `RunResult` at a κ-sweep of the FundedNext cost vector
(`scaled_fundednext(κ)`, BUILT). One engine pass, free re-netting at any cost. The cost-passive break-even
is a conservative LOWER bound (a live trader who skipped marginal trades at high cost does strictly better).

**Reproduction discipline paid off.** First run overrode `max_concurrent_per_currency=None` and gap came
out +1.23% (vs +0.685%) — the default per-currency=2 cap is load-bearing (the JPY crosses all share JPY;
without the cap the clustered weekend-gap fires over-size the book). Reverting to A1Config DEFAULTS gave all
4 components BYTE-EXACT (per-fold series identical to arc 3009; me_short's 2018 fold +0.86 / 2015 +0.40 =
arc 2015) and the risk-parity book +0.5889% = arc 2019's +0.589%. Reproduce-don't-transcribe caught it,
just like 3009's fbr double-trail.

**What it found.** (1) **Cost-robust, not knife-edge:** book break-even κ = 3.32 (risk-parity; 3.17
equal-weight). FundedNext eats 30% of gross; 70% survives; positive out to ~3× FN. It would take 3.3× the
*already-conservative* FundedNext cost to erase the edge — far outside any plausible live-vs-model gap.
The deploy case is not a cost-modeling artifact. (2) **AFP failure is cost-INDEPENDENT:** even cost-free
(κ=0) the book is 2/10 negative → the per-year-gate failure is structural (thin-component noise floor +
fbr's −2018), not about costs. Raising cost makes the gate harder (n_neg 2→7) but the failure pre-exists
at zero cost. This confirms 2016/2017 from a new angle: **cost is not the wall; the per-year gate vs thin
components is.** (3) **Per-component cushion is mechanism-meaningful:** gap (H4, frequent, JPY-cross,
thin-per-trade) is the cost-FRAGILE leg (break-even 2.10, 47.6% drag); me_short (D1, infrequent, overshoot)
the most robust (5.84, 17%); fbr 3.84; me_long 2.38. Risk-parity down-weights the fragile gap (0.077) →
the principled weighting is also the cost-robust one.

**Threads.** The book now has THREE independent robustness legs for the operator's path-A call —
mean-positive (t=2.66, 1023), temporally stable (2021), cost-robust (this) — and the AFP failure is
confirmed cost-independent. There is nothing left to characterize about the existing book that would change
the path-A vs nothing decision; the lever is squarely the operator's gate-governance call (every diagnostic
since 2016 converges here). If a future arc revisits the gap-fill, note it is the corpus's most
cost-exposed component (break-even only 2.1×). NEW reusable lesson: cost-robustness-of-the-mean and
gate-satisfiability are DIFFERENT questions — a book can survive 3× cost while its per-year gate is
cost-sensitive; report both, never conflate "survives cost" with "passes the gate." No edge found; KILL.

### arc_2023 — M1 driver-shock-conditional triangulation residual (DISCOVERY_DIRECTION menu, freshest prior)
**Why this arc.** The strategist MENU (`discovery/DISCOVERY_DIRECTION.md`) landed mid-session and independently corroborated my arc 2022 (leg-hunt for the book is closed). Its #1 explore-now item, M1, attacks a *named unconditional* corpus result: arc 3005 killed the triangulation residual but only unconditionally and only at H4. The untested angle: condition on a large DRIVER-leg shock and look for a next-bar lag dislocation in the quoted cross. A genuine fresh-frontier item, one-leg expressible, cheap — exactly the kind the menu says to prefer, and a high-value closure if (as the honest prior expects) it dies.

**What I did + the one design choice that matters.** Kept everything at H4 (one timeframe) and proxied the "D1 driver shock" by a 6-H4-bar driver return at t-1 → predicts bar t: this is ex-ante AND sidesteps D1/H4 boundary-alignment ambiguity (cleaner than mixing timeframes). The decisive measure is the PURE identity residual `idr = r_cross − r_driver − r_other` (≈0 by identity; arc 3005's residual in return space) — NOT the menu-literal "residual not explained by contemporaneous USDJPY," which I computed too but flagged conflates driver MOMENTUM (closed ground). The clean test is whether `idr` at the next bar correlates with the prior driver shock (a lag), not whether the cross tracks driver momentum.

**What happened.** Across all 3 clean triangles (EUR/GBP/AUD-JPY, n=6849 shocks), the conditional corr(driver_shock, next-bar idr) is ≈0 (−0.009/−0.025/+0.001, pooled −0.012) — indistinguishable from 3005's unconditional ≈0.01, far below the 0.05 falsifier. Directional continuation ~−0.01 bp, ~100–200× below the 1.2–2.0 bp cross spread. The "not-explained-by-USDJPY" residual is also ≈0 → no driver-momentum leak either. The cross re-prices the identity WITHIN the bar even after a 1-day driver shock.

**What this means.** KILL at the cheap-obs stage = a high-value closure: extends arc 3005's first-moment "residual ≈ 0" from unconditional to driver-shock-CONDITIONAL — the exact angle 3005 never tested. The triangulation first moment is now fully closed (level + conditional). The only triangulation door left is menu-L1 (the residual SECOND moment / OU amplitude at M1→H1), which walks into the H1 cost wall — left for a chat with budget. No engine/null/council (corr≈0 + sub-spread → §5d cheap-kill, like 3005/3016). Reinforces the standing conclusion: the lever is the operator gate-call, not new H4 edges.
### arc_1029 — O1: inelasticity-state via the calendar-flow-density proxy (the council's #2)
**Why this arc.** After M1 (1027) and Q1 (1028), the MENU's remaining highest-value item is **O1** — the
only one attacking the book's ACTUAL blocker (per-year fold resolution). Its thesis: an inelasticity-STATE
variable concentrates risk into high-edge bars, raising edge-per-trade without adding coin-flips. The council
named 3 collinear proxies and rated **calendar-flow density** the single LIVE one (spread-z = cost trap;
trigger-shallowing = arc-1025-confirmed). So I tested whether the proven month-end reversion gets STRONGER at
higher-rebalancing windows (quarter-end Jun/Sep; fiscal-YE Mar/Dec).

**What I did.** Pooled BOTH proven legs as reversion trades (me_long fades a big DOWN move into ME; me_short
fades a big UP move into ME — direction-aware honest +1R-before-SL capture IS the per-trade reversion edge),
D1, 7 USD majors, IS, threshold 1.0/into 2 (the 1011/1019 config), n=248; tagged each by calendar-density
tier and grouped.

**Result — FALSIFIED at obs.** (1) Tier capture 0.521/0.543/0.600 LOOKS monotone but decomposes to a
small-sample artifact: T3 = March(n=10, capture 1.000 — a fluke) diluted by December(n=25, capture 0.440 —
the WORST month, below coin-flip). The two fiscal-YE months disagree; by-month capture is noise (ordinary
January highest at 0.769; quarter-ends ≈ coin-flip). Drift is non-monotone (T2 0.187 > T3 0.076) — the
arc-3007 not-a-lever tell. (2) DECISIVE: concentrating onto high-density (tier≥2) WORSENS fold resolution —
2/11 → 7/11 negative years (the full, denser pool is more fold-stable). That is the council's predicted death
#1 made concrete: sub-sampling a thin edge raises per-year variance → more per-year-gate trips.

**Mechanistic read.** The month-end reversion's thinness is UNIFORM, not concentrable — its edge does not
pile onto the highest-rebalancing windows (December, the biggest, is the worst). You cannot buy
fold-resolution by sub-sampling; re-confirms arcs 2016/17 (too thin to resolve) and 1025 (fbr won't thicken)
on a new (calendar) axis.

**Threads.** O1's live proxy closed. The only untested O1 sub-thread is spread-z, pre-flagged a COST TRAP
(must charge the realized wide spread at entry). The `explore-now` frontier is now largely exhausted from
this chat (M1/Q1/O1 all KILL); remaining: L1 (triangulation 2nd moment, cost-walled). Components UNCHANGED;
lever = operator gate-governance (path-A). Re-usable tell: decompose a "monotone tier" by its members — a
clean T1<T2<T3 trend collapsed once T3 split into a March fluke vs a December worst-case.

### arc_1028 — Q1: central-bank peg / boundary-defense persistence (the long-shot)
**Why this arc.** After M1 (1027), the strategist MENU's highest genuine-UPSIDE cheap swing is **Q1** — the
one item that targets the book's actual deficit (a THICK component) via a genuinely NEW mechanism: a
price-insensitive actor defending a band, a persistent forced flow you JOIN (the inverse of the corpus's
reversions-of-overextension). The OHLC-detectable instance is the SNB EURCHF 1.20 floor (2011-09..2015-01),
in-IS, H4 cached — the textbook peg. (Chose Q1 over L1, the triangulation second moment, because L1 needs a
heavy multi-pair H1/M1 backup load for an expected-FAIL variance probe walking into the H1 cost wall; Q1 is
one cached pair and a genuinely different, higher-upside mechanism.)

**What I did.** EURCHF H4. Established floor = causal `rolling-min(mid_low,250).shift(1)`; near-floor touch =
a bar dipping within 0.5 ATR of the floor. Honest +1R-before-SL capture + 24-bar fwd drift via the canonical
`observe_long_capture` restricted to near-floor bars; split by hindsight regime label and by a causal
bottom-decile-realized-vol detector. Also characterized the Jan-2015 break tail.

**Result — FALSIFIED at obs.** (1) No asymmetry: in-regime near-floor capture 0.526 ≈ out-of-regime 0.525 ≈
coin-flip. (2) In-regime fwd-drift MEDIAN −0.237 ATR (the +1.19 mean is thin-tail) — the floor **pinned**
price at 1.20 (absorbed selling without pushing it up) → a truncated distribution, not a fadeable reversion.
(3) The causal vol-collapse "lift" (cap 0.598) is generic low-vol reversion: the detector fires in EVERY year
(2010–2026), not the 2012–2014 regime → closed ground, not the peg. (4) Decisive regardless: the regime is a
single pair-epoch (~3 yrs, can't populate per-year folds — absence-FAIL, cf. 3019) ending in a −18.8% /
~182-ATR gap-through-stop break (Jan 15 2015) that the −1R invariant flatters in backtest but cannot hedge
live. Cost was NOT the killer (in-regime ~0.048 R) — no-edge + the tail are.

**Mechanistic read.** A defended boundary PINS, it does not BOUNCE: the SNB absorbed all selling at 1.20 so
price sat glued to the floor in a tiny band; there is no reversion-to-a-higher-level to fade. The council's
"join a persistent forced actor" inverse-of-reversion idea has no tradeable instance in the corpus's pairs.

**Threads.** Q1 closed. The `explore-now` frontier is now nearly exhausted (M1 1027 + Q1 1028 both KILL);
remaining MENU items are L1 (triangulation second moment — cost-walled, low-EV) and O1 (inelasticity-state —
fold-resolution attack, but path-B closed, 3021). Components UNCHANGED (all 4 PORTFOLIO); operative lever
unchanged (operator gate-governance, path-A). Re-usable tells reinforced: median-vs-mean catches thin-tail;
a "regime detector" that fires every year is generic conditioning (closed ground), not the named regime.

### arc_1027 — M1: driver-shock-conditional triangulation residual (closing 3005's open angle)
**Why this arc.** The corpus is mature (path-B closed, 3021; lever = operator gate-governance call). The
strategist `DISCOVERY_DIRECTION.md` MENU ranks **M1** as the #1-EV `explore-now` candidate: it attacks a
NAMED unconditional corpus result — arc 3005 killed the cross-rate triangulation residual but *only
unconditionally and only at H4*. M1's twist: condition on a large D1 driver-leg shock; if the dependent
quoted cross re-prices with a lag, the next H4 bar carries a transient residual signed by the shock. One-leg
expressible, cheap-obs-first.

**What I did.** Built `discovery/tools/triangulation_residual.py` (residual-bp + D1 shock + no-lookahead
next-day-bar re-keying), ran 4 product-form triangles (EUR/GBP/AUD/NZD × USDJPY → XXXJPY) on IS 2010-2020.
Shock = `|Δclose_mid/Wilder-ATR(14).shift1| > 1.5` on the driver leg; residual measured at the first H4 bar
after the D1 close.

**Result — FALSIFIED at obs.** Pooled `corr(shock, next-bar resid) = 0.0135` (within-triangle-standardized
0.0144) ≈ 3005's unconditional ~0.01, below the 0.05 falsifier. Decisive secondary tell: the per-triangle
corrs are **sign-inconsistent** (+0.112 / −0.196 / +0.029 / +0.152) and each within ~2 SE of zero (n≈100,
SE≈0.10). A real asynchronous-repricing lag must be SAME-signed across all four XXXJPY crosses (they share
the USDJPY leg and the lag structure); mixed signs = noise, and pooling them (raw or standardized) washes to
≈0. Residual stays sub-spread even at shock bars (median 0.168 bp; >0.8 bp on 16.5%). Apparatus reproduces
3005 (unconditional conv-corr 0.00–0.04 vs its 0.009–0.016) — measurement validated.

**The one real (but wrong-thread) effect.** A shock modestly raises the residual's VARIANCE (std 1.08–1.31 bp
vs unconditional 0.85–1.11) without raising its directional MEAN. That amplitude bump is L1's domain (the OU
second moment), not M1's first-moment lag — and L1 needs finer resolution, walking into the H1 cost wall that
killed the whole microstructure cluster. So I did not open L1; flagged it as a low-EV remaining door.

**Threads.** Triangulation is now directionally closed at H4 both unconditionally (3005) and
driver-shock-conditionally (1027). L1 (residual amplitude at H1→M1) is the only untouched sub-thread, but
cost-walled. Components UNCHANGED (all 4 PORTFOLIO); operative lever unchanged (operator gate-governance,
path-A). Re-usable method note: when a mechanism predicts a COMMON sign across structurally-parallel
instruments, test the sign agreement — it kills a noise mechanism faster and harder than the pooled magnitude.

### arc_2022 — can ANY 5th leg make the 4-way book AFP? (test with the strongest real targeted candidate, 3019)
**Why this arc.** The corpus is near-terminal: the 4-way book is a sound, mean-positive, decorrelated, temporally-robust PORTFOLIO that fails only the strict AFP gate, and arc 2019 concluded "a 5th decorrelated REVERSION leg can't make it AFP → operator gate-call is the lever." But that claim was QUALITATIVE and REVERSION-only, while arc 3019 had just found the strongest +2015/+2016 candidate ever (shock-continuation, a CONTINUATION leg) and killed it on its OWN OOS *before ever combining it with the book*. So the literal feasibility question — does a real, targeted leg of that profile make the 5-way book AFP? — was open. PRIME DIRECTIVE: question the terminal claim before the run converges on it. (Six straight 2000s diagnostics 2016–2021, but a decisive test of the *terminal claim itself* beats a 7th futile thin-edge hunt or a redundant book re-measurement.)

**What I did.** Reproduced the 4 components byte-exact (Arc-10 discipline — verified gap/me_long/fbr/me_short headlines before trusting any combination), reproduced the 3019 shock leg (long+short @3.0 ATR tp_3r, summed per-fold ROI), then 4-way vs 5-way convex search + 5-way honest weights (risk-parity & equal, the arc-1021 rule) + an idealized-leg feasibility frontier. All canonical apparatus, 0.5% risk (arc-1024 FRACTION).

**Resolved the magnitude scare honestly.** My shock leg ROI (+4.319% mean) is ~100× 3019's reported +0.034 — I did NOT propagate this as either a bug or a 3019 error. Checked: 0 daily-DD breaches at 0.005 (linear regime, not cap-distorted) AND ROI scales 10.1× from 0.0005→0.005 (perfectly linear). So +4.319% is the honest LINEAR ROI at 0.5% risk; the gap vs 3019 is a doc reporting-unit difference (3019's low-risk/per-trade table). The trustworthy scale-invariant pattern (9/10, +2015/+2016/+2018, decorrelation) matches 3019 — exactly arc-3017's "scale-invariant metrics are the trustworthy cross-config judges." (Minor: 232 fires vs 3019's 287 — same sign pattern; a universe/threshold nuance, not load-bearing.)

**What happened.** The shock leg is the textbook perfect 5th leg: + in both blocker folds (2015 +5.67, 2018 +2.26, also 2016 +11.18, 9/10) and the most decorrelated component in the corpus (corr −0.60 me_short / −0.42 me_long / −0.34 fbr). **And it STILL doesn't make the book AFP honestly.** 5-way AFP appears under IS-optimized convex weights (1383/10626, w≈50% shock) — but that is arc-1021 weight-painting. Under honest risk-parity it's −0.084% (1 fold short: 2018; narrows 4-way's −0.124%); equal −1.393%. The mechanism: shock is a tail edge → per-fold sd 3.78% (vs me_long 0.71%) → risk-parity gives it only 0.089 weight → its +2.26% 2018 contributes +0.20%, too little to clear gap/fbr's deep −2018. The weight that WOULD clear AFP is overfit.

**What this means (the contribution).** Confirms + sharpens + extends arc 2019: even a perfectly-targeted, maximally-decorrelated NON-reversion leg fails the honest gate. The blocker is not "no good leg exists" (this leg is decorrelated AND targeted) — it's a structural WEIGHTING DILEMMA: the legs that can lift a deep blocker fold are high-variance tail edges that honest weighting throttles, and the weight that suffices is overfit. The leg-hunt route to an AFP book is now closed STRUCTURALLY, not just by induction over ~18 dead arcs. The operative conclusion (operator gate-governance is the lever) is strengthened, with a reason.

**Threads.**
1. **Weighting-dilemma rule (reusable):** judge any future 5th-leg combination under risk-parity, NEVER IS-optimized convex weights — the convex-AFP is a mirage (generalizes arc-1021 from exits to weights).
2. **Sharpened residual spec:** a MODERATE-variance leg robustly + in BOTH 2015 AND 2018 (~+0.7% each). 2015 (acute EUR/SNB) and 2018 (USD grind) are different-mechanism regimes; +both at deployable variance may not exist in FX-OHLC (consistent with 3016/3019 "well nearly dry"). This is a harder, more precise spec than the log's prior "+2015/+2016 non-(−2018)."
3. **The last decision-relevant open question is the book's OOS** under a candidate path-A gate — but spending the 4-way holdout breaches the §5g OOS-firewall and presupposes the operator's gate choice, so it is the operator's call, not an autonomous chat's. Flagged, not taken (conservative bias, §8).
### arc_3021 — path-B tractability (how many decorrelated legs would the AFP gate need?)
**Why this arc.** After 3020 the corpus is fully converged: the edge frontier is closed on every
documented lever, and arcs 2016/2017/1023/2019/2021 establish the book as a sound ~3-bet, mean-positive,
temporally-robust PORTFOLIO failing only a per-year gate below its noise floor. Every one of those arcs
flagged the SAME remaining lever — the operator's path-A (change the gate) vs path-B (denser book) call —
and left it QUALITATIVE. The single highest-value unmade contribution was to QUANTIFY path-B, which also
lets me TEST (not inherit) arc 2019's "a 5th reversion leg can't help" (PRIME DIRECTIVE).

**The insight arc 2019 missed.** 2019's claim is a fold-painting argument. But portfolio math says
averaging N decorrelated positive-mean legs shrinks book per-fold variance, so P(AFP)=Φ(book-Sharpe)¹⁰
RISES with N and AFP is reachable at SOME N — the real question is at what N and whether that's tractable.
I simulated it, calibrated from the real 4 legs (canonical reproduction; per-leg per-fold Sharpe 0.14–0.60,
median 0.42; mean corr +0.115).

**What I found.** Two results. (1) Even decorrelated (ρ=0) the gate needs ~15–34 corpus-quality legs —
intractable at ~1 leg/14 arcs with the frontier closed. (2) The decisive one: at the corpus's OWN
realistic residual correlation (ρ≈+0.12), P(AFP) PLATEAUS ~0.30 and is unreachable at ANY N — a shared
dollar/risk factor floors book variance, so densification can NEVER satisfy the per-year gate. This turns
2019's qualitative claim into a portfolio-math proof, by an independent method (variance-sim vs ENB),
same verdict. ⇒ path-A (the operator's gate-governance call) is the ONLY deployability route; path-B is off the table.

**Threads.** (a) The operator now has a quantified path-A/B decision: path-B closed; path-A supported by
arc 2021 (temporal robustness) + arc 1023 (t=2.66 mean-positive) + this (why densification fails). (b)
The per-year AFP gate demands book Sharpe ~2.3 vs the corpus's ~0.4 thin-leg reality + shared factor —
the portfolio-level statement of arc 1025 (edges intrinsically thin) + arc 2008 (shared tail). (c) With
the edge frontier closed (3020 et al.) AND path-B closed (here), the autonomous discovery levers are
genuinely exhausted short of a brand-new OHLC-constructible mechanism (none known) or the operator's gate
call. Future chats: a genuinely novel mechanism with a *because* (rare) is the only remaining edge lane;
otherwise the corpus is in a steady state pending the operator. I did NOT loosen the gate or spend OOS.

### arc_3020 — fbr × cross-pair breadth gate (the last untried entry-time axis on fbr-2018)
**Why this arc.** Resumed 3000s at 3020 (highest in-range 3019; no STOP). The honest-era corpus has
converged hard: 4 net-positive PORTFOLIO components, a sound mean-positive ~3-bet book (ENB=3.32,
t=2.66) that fails the per-year all-folds gate ONLY below its noise floor (arcs 2016/2017/1023/2019),
and arc 2017's verdict that the route's single statistically-real fold obstacle is `fbr`'s −2018 (the
only CI<0 fold in the corpus). arc 2017 names the only productive remaining edge-spec as option B (a
THICK fold-resolving standalone). The highest-leverage in-apparatus target is therefore to improve the
corpus crown jewel `fbr` (1013, 9/10 IS, ONE fold from the first standalone PASS) toward solo-PASS. Its
2018 hole has been attacked on per-pair regime (2014), M1 microstructure (2020), depth (1025), and level
(3013) axes — all mechanism-intrinsic. **Cross-pair BREADTH was the one untried entry-time axis**, with a
clean *because*: a strong-USD risk-off move should make many majors fire fbr simultaneously and those
broad reclaims should fail (real dollar trend), so a low-breadth gate might drop 2018's clustered
false-reclaims while keeping idiosyncratic good-fold winners. Cheap obs test; if it separated 2018 it
would be the corpus's first solo-PASS — a high payoff on a cheap kill.

**What I did.** Reproduced fbr (K40/sh1.25, 7 USD majors, H4) via the BUILT signal + observe harness
(237 fires, cap 0.5865 — exact reproduction, Arc-10 discipline). For each fire counted OTHER majors
firing within ±{0.17,1,3,7}d; primary ±3d. Binned solo/low/high, compared 2018 vs non-2018, ran per-year
drift under breadth filters.

**What happened.** (1) Breadth DOES separate drift — solo +0.52 vs low −0.35 vs high −0.45 — a real
"USD-wide cluster = less-reliable reclaim" texture, in the hypothesized direction. (2) **But the premise
is false:** 2018 breadth (0.74) ≈ normal years (0.56); 2018 is not a clustering event. (3) **No
solo-PASS:** solo-only is 8/11 yrs drift>0; it flips 2018 positive (+0.59) but breaks 2010/2012/2016,
the 2018 fix is n=8, and the solo edge is carried by EURUSD/USDCAD (3/7 pairs flat-negative) — arc-2014's
exact "gate that fixes 2018 breaks good folds" failure mode, on a new axis.

**Why it failed / diagnosis.** 2018 fbr trades fail per-pair and FORWARD (the reclaim is structurally
real at entry but doesn't HOLD over the following days in the strong-USD trend, arc 2020) — there is no
entry-time signature (regime, M1 quality, depth, level, or now breadth) that distinguishes the 2018
losers from the good-fold winners, because the failure happens AFTER entry. This is the 5th independent
confirmation; fbr-2018 is overwhelmingly mechanism-intrinsic.

**Threads.** (a) Genuinely new, low-priority: solo (idiosyncratic) fbr fires are a higher-QUALITY subset
(+0.52 vs +0.146 baseline drift) — a possible quality refinement, NOT a fold-fixer (thins the edge,
2-pair-carried). (b) The improve-fbr / solo-PASS lane is now closed at the entry side across 5 axes; arc
2017's option-B thick edge is not reachable by refining fbr. (c) Route status converges with 2019: the
lever is the operator gate-governance call on a sound mean-positive book, not a missing/improvable edge.
I am not spending OOS or loosening the gate (conservative bias). The new-component frontier and the
improve-the-best-edge lane both read closed; the next chats' highest-EV move is a genuinely novel
mechanism with a *because* (rare) or to surface the gate-governance call — not another fbr-2018 axis.

### arc_2020 — fbr M1 reclaim-quality confirmation (the council's last open fbr-2018 thread)
**Why this arc.** My range (2000s) ended arc 2019 with a definitive council-backed conclusion: the 4-component
book is a sound ~3-independent-bet, mean-positive PORTFOLIO whose all-folds-positive failure is a *gate-resolution
artifact*, so edge-hunting **for the book** is closed and the lever is the operator's gate call. But arc 2019's
council had raised one UNRESOLVED generative thread aimed at the programme's actual goal (a deployable *solo*
PASS): *refine fbr-2018 via M1 reclaim-confirm.* `fbr` (1013) is the crown jewel — 9/10 IS, the single fold
(2018, strong-USD) between it and a solo all-folds-positive system. Worth one decisive test, because the value
of success is the whole programme and a clean negative closes the last open thread on the best edge.

**Why it's not grinding.** fbr-2018 was attacked twice and declared mechanism-intrinsic — but by DIFFERENT levers:
arc 2014 = DAILY regime-gating (downtrend strength/persistence), arc 2017 = per-fold CI diagnostic (no
intervention). Neither asked the entry-QUALITY question: are the 2018 "reclaims" weak/false at M1 resolution
(distinguishable AT ENTRY), such that a uniform M1-confirm filter lifts the worst fold? That's a genuinely new,
falsifiable lever the council itself flagged. Observation-first and cheap (M1 cached for all 7 majors), decisive
either way.

**What I found.** 237 fbr fires (the corpus-standard count). For each I measured M1 reclaim-quality within the
H4 reclaim bar (strictly ≤ signal-bar close; entry is next-bar open, so no-lookahead). The 2018 fold is the
failing one (capture 0.474 vs 0.596; drift −1.28 vs +0.27) — but its M1 reclaim-quality is NOT worse. Deltas are
mixed and tiny, and the cleanest "decisiveness" metric (frac of M1 mid above the swept level) is actually HIGHER
in 2018. M1-quality has ~zero correlation with the per-trade outcome (all |corr| ≤ 0.10). And no keep-rule on any
metric rescues 2018: high-decisiveness filters make 2018 *worse*; the only one that "helps" thins 2018 to 6
trades (regime-luck) and destroys the good folds (7/11). 

**The mechanism (why M1 can't help).** The 2018 reclaim BARS are real and, if anything, more decisive — price
sweeps the swing low and convincingly closes back above it. The failure is FORWARD: over the following days, in
the strong-USD trend, the "failed" breakdown becomes a real breakdown and the reclaim gets sold again. An
entry-time signal — at ANY resolution — cannot see a multi-day-forward reversal. This is exactly arc 2014's
"near-total 2018 wipeout, entry-time-unconditionable," now confirmed at M1 micro-structure, a third independent
lens. fbr-2018 is mechanism-intrinsic; finer entry resolution is the wrong axis.

**What it means for the route.** The solo-PASS-via-fbr-2018-fix route is now closed at the entry side (this was
the last untested entry-resolution lever). Triangulated across regime (2014), statistics (2017), and
micro-structure (2020), fbr-2018 will not yield to entry-side work. This strengthens — does not change — arc
2019: deployability is now an operator gate-governance decision (does the every-calendar-year AFP gate, which
sits below the legs' noise floor, remain the bar?) or arc-2017 option B (a THICK fold-resolving standalone, none
known, and the data is FX-only). No new component; fbr stays PORTFOLIO; nothing on the engine/null/council was
spent because the refinement died honestly at observation.

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

### arc_3010

**Sustained-USD-strength SHORT — can a trend-following short provide the 2018-positive 4th portfolio leg?**
(chat 3000s). Full record: [`arcs/arc_3010_sustained_trend_short.md`](arcs/arc_3010_sustained_trend_short.md).
No council (falsified at observation; regime-inversion is a falsification, not a tuning fork — arc-1014/3003
discipline).

**Idea + why.** My own arc 3009 (and the 2000s' concurrent arc 2008) left the portfolio route blocked on
exactly **2 folds — 2015 and especially 2018** (strong-USD trend years), with a sharp spec: the 4th leg must
be net-positive there → a SHORT / trend / risk-off-positive construction. arc 1014 (1000s) had just shown
*structural* shorts are coin-flip and CLAIMED the up-gap weekend short (1015); the 2000s lane is the
climax-sweep short. The unclaimed, highest-EV open question for my range was the **deployment blocker
itself**: can a TREND-following short — conditioned on an established downtrend (the regime the portfolio
needs), held through a trend year — provide the 2018 leg? This is the symmetric mirror of arc 3003 (same
question for momentum *longs* via Kaufman ER → regime INVERSION). Observation-grounded, decision-relevant,
cheap.

**What happened — FALSIFIED at observation (no engine needed; arc-3003 lesson #2).** Reused the BUILT
direction-aware `observe_long_capture(direction="short")` on 7 USD majors H4, IS, joined an established
-downtrend regime (close<SMA200 & SMA50<SMA200) + year. **Short base: cap 0.4849, drift −0.0151 ATR** (a
coin-flip mirroring the long base, per arc 1014). **The downtrend regime filter INVERTS the short:**
downtrend drift **−0.127** vs non-downtrend +0.067, and negative on **all 7 pairs** (−0.02 to −0.32) — an
established downtrend REVERTS UP against the trend-aligned short, exactly as arc 3003 found strong uptrends
revert against the momentum long. **The 2018/2015 portfolio question: answered negatively.** The
*unconditional* short is faintly positive in 2018 (+0.152) and 2015 (+0.125) — but only as regime-luck
within a yearly coin-flip (positive 5/10 years, mean drift −0.015, below the 0.50 line); the regime filter
that should *isolate* the favorable trend instead KILLS 2015 (downtrend −0.107) and is negative on every
pair. So the 2018-positivity is **uncapturable** — it cannot be conditioned on or separated from the
coin-flip.

**Verdict: KILL (cheap-kill at observation).** A trend-following short does not provide the 2018-positive
leg; the short is a coin-flip, the regime filter is anti-predictive (inverts), and the 2018 positivity is
uncapturable regime-luck.

**Threads / lessons.** (1) **Regime-conditioning on trend strength is anti-predictive in BOTH directions —
the symmetric completion of arc 3003.** Strong uptrends revert (3003 momentum long); strong downtrends
revert (3010 trend short). "The trend is established" makes a directional bet WORSE on liquid FX H4, both
sides — a now-two-sided robust closure of regime detection as a directional lever. (2) **The portfolio's
2018 wall is not a price-trend problem** — no conditionable trend short isolates the 2018 positivity, so the
4th leg (if it exists) is NOT a trend-following short. Sharpens the arc-3009/2008 spec into a
near-impossibility within the price-direction apparatus → arc-3004 escalation reinforced (the 2018-positive
leg needs genuinely non-directional structure / relative-value / a second leg; carry is OFF on FundedNext).
(3) **Confirms arc 1014 from the trend angle:** shorts revive neither directional STRUCTURE (1014) nor TREND
(3010) — the short base ≈ the long base by symmetry, both failing identically. The value of shorts is
confined to genuine ASYMMETRIES (flow events with measured accuracy >0.50), not symmetric direction/trend.
(4) **What remains open (not closed by this arc):** the FLOW-EVENT short asymmetries — up-gap weekend short
(1000s arc 1015) and climax-sweep short (2000s) — are not trend-direction bets and stay the live short
frontier; but they are reversions/bursts, so whether either is specifically *2018-trend-positive* (the
portfolio's need) is itself doubtful (reversion edges bleed in trend years — the very reason 2018 is the
wall). **Fleet steer:** the portfolio's 2018 leg likely needs a genuinely non-price-direction construction
(relative-value/second-leg) or the route is escalation-bound; another directional short (any flavor) is
unlikely to be the 2018 leg.

**Tooling:** no new BUILT tool — reused the BUILT direction-aware `observe_long_capture(direction="short")`.

**FLAGS (code not merged):** none. No canonical-core change. Carries FLAG-1 (shorts unblocked PR #273, but
trend-direction shorts now shown dead, so FLAG-1's value narrows to flow-event asymmetries + the standing
escalation) and the `A1Config.time_exit_bars`-unwired flag. Driver scratch
`_disco3_work/arc3010_trend_short_obs.py` (reproducible from the arc doc).

### arc_3011

**Failed-breakout REJECTION short — the short mirror of arc 1013** (chat 3000s). Full record:
[`arcs/arc_3011_failed_breakout_rejection_short.md`](arcs/arc_3011_failed_breakout_rejection_short.md).
No council (a decisive observation + engine result, not an idea-fork / diagnosis / survivor). **The FIRST
discovery short arc to reach the honest engine** (1014/2009/3010 all cheap-killed at observation).

**Idea + why (log-seeded — the unclaimed clean mirror of the corpus's strongest edge).** Arc 1013 (the
strongest, cleanest directional edge in the corpus) is a failed-breakdown RECLAIM long: sweep a swing
**low** (run sell-stops), reclaim above it (failed breakdown), go long — structure-load-bearing. Arc 1013
explicitly flagged "the up-sweep is arguably the stronger leg." The symmetric mechanism: sweep a swing
**HIGH** (run buy-stops above resistance), reject back below (failed breakout), go **short**. This is the
untested clean mirror of the best edge, unclaimed by any range (the up-GAP short is the 1000s arc-1015
lane; the swing-LOW climax short is 2009-dead; the confirmed-breakdown short is 1014-dead). Now unblocked
(PR #273). Hypothesis it could be the 2018 portfolio leg: it's a SHORT, so on falling XXXUSD majors in
strong-USD 2018 a failed-upside-breakout short is trend-aligned (the three long fades are not).

**What happened — OBSERVATION (structure control, the decisive 1014/2009 discriminator).** 7 USD majors
H4, IS, `observe_long_capture(direction="short")` (drift>0 ⇔ price fell ⇔ good for short). Base short cap
0.4849 / drift −0.015 (coin-flip mirror, per 1014/3010). The CELL (swept swing-high & reject-below &
upper-shadow≥1.25 ATR, n=241): cap **0.4730**, drift **+0.278 ATR**, frac+ 0.494. The pooled structure
control LOOKED like a pass (big-reject AT swept swing-high +0.278 vs the SAME wick ELSEWHERE −0.171) — and
I initially over-claimed it as the first load-bearing structural short. **It is NOT robust** (correction
credited to the concurrent independent arc 2011): the +0.278 is a **thin-tail / pair-mix artifact** —
carried by 2 thin outlier pairs (AUDUSD n=29 +1.58, USDJPY n=31 +1.49); the other five average ≈ −0.04
(EURUSD +0.63, GBPUSD +0.20, NZDUSD +0.37, USDCAD −0.47, USDCHF −0.95). Arc 2011 quantified the same:
pooled mean +0.261 but **median −0.069**, and dropping those 2 pairs takes the cell **−0.133 negative**
(deeper cell inverts) — the arc-2009 USD-quote pair-mix beta tell. Capture is **0.473 < 0.50** (1013 long
0.55–0.61) → coin-flip, the favorable pooled drift TAIL-driven not reliable per-trade. Non-coin-flip
pooled *drift* + sub-0.50 *capture* ⇒ §5f mandates the engine + exit menu before a FAIL (no cheap-kill).
Per-year: 2015 drift +1.09 but **2018 −0.131 (NEGATIVE)**; per-pair positive 5/7 but USDCHF −0.95 / USDCAD
−0.47 (USD-up legs fight the 2018 uptrend) → does NOT provide the 2018 leg.

**What happened — HONEST ENGINE (§5f).** Standard entry point: pool n=241 (sha ccb8ef27), pool capture
0.4730 ✓ matches obs, **mean final_r +0.0102R ≈ 0**. Full registered exit menu × {1.5,2.0,3.0}·ATR SL
(18 cells) over the 10 IS folds, `trail_enabled=False`, FundedNext ON, SL-first: **every cell mean ∈
[−0.04%, +0.01%], 0 all-folds-positive** (best `sl_plus_tp_3r` SL2.0 = 7/10 pos but worst-fold −0.04%; best
worst-fold −0.028%). The structure-confirmed +0.278 ATR gross drift collapses to ~0 net — capture 0.473
means the 2·ATR stop fires before the slow drift pays, and FundedNext costs eat the remainder. IS not
all-folds-positive → **OOS NOT touched** (§4). Fair same-exit null (`build_null_signal_evaluation`, 3
seeds, exit `sl_plus_trailing_atr` SL2.0): REAL mean +0.005% vs NULL −0.015% → **REAL beats null +0.021pp**
(the structure edge is genuinely real) but at the noise floor and config-fragile.

**Verdict: KILL (no robust short edge).** Two independent reads converge: the pooled structure pass is a
thin-tail/pair-mix artifact (median −0.069; negative excluding 2 pairs — 2011 + my per-pair data); and on
the engine, even granting the pooled drift, it nets ~0 (mean +0.005% best exit, beats null only +0.021pp =
noise floor), 0/18 exit·SL cells all-folds-positive, NEGATIVE in 2018. §11: no usable mean-positive edge ⇒
KILL, not PORTFOLIO.

**Diagnosis — arc 1013's reclaim-long has NO short mirror (3 constructions).** 1014 (clean
confirmed-breakdown — swing-low sweep inert), 2009 (climax — anti-load-bearing, shorts into the bounce),
2011/3011 (swing-HIGH reject-reversal — drift not robust, capture coin-flip, nets sub-cost). The unifying
fact: **capture is the wall** — 1013's long captures 0.55–0.61; every short construction ~0.47–0.51. A
candidate mechanism (downside stop-runs/fear snap back fast = high capture; upside stop-runs/greed drift
slow = low capture) is plausible but SECONDARY to the decisive "no robust capturable short exists."

**Threads / lessons.** (1) **arc 1013's reclaim-long has NO tradeable short mirror across all 3 short
constructions** (1014/2009/2011·3011) — the swing-low sweep is an asymmetric reversal-LONG structure; the
structural-short route to a 2018 leg is closed. (2) **Check the MEDIAN + pair-mix before believing a pooled
structure-control "pass"** — my pooled +0.278 was 2 thin pairs (AUDUSD/USDJPY); median −0.069, negative
excluding them (arc 2011 caught it, my per-pair confirms) = arc-2009 USD-quote-beta tell recurring. Demand
a median + leave-2-pairs-out check on any short structure control. **The most valuable carry-forward here,
and a self-correction credited to the independent reproduction.** (3) **Capture, not drift, is the wall for
reversal shorts** — even a pooled-positive drift nets ~0 under a 2·ATR SL-first stop when capture is
sub-0.50; the §5f 18-cell menu cannot lift it (extends arc 0/2000). (4) **2018 is still the portfolio
wall** — a would-be trend-aligned short fade is negative in 2018 (USDCHF/USDCAD fight the USD uptrend) →
reinforces 3009/3010 that the 2018-positive leg is NOT a directional construction of ANY kind; the
relative-value / second-leg / arc-3004 escalation route stands. (5) **First discovery short reached the
engine cleanly** — `direction=SHORT` flows end-to-end (signal state → A1 `directions` map → Order →
MultiPairBacktester); pool capture reproduced the observation (0.4730), validating PR #273 on the discovery
apparatus.

**Convergence with arc 2011 (chat 2000s — concurrent independent reproduction).** The 2000s chat ran the
identical idea as arc 2011 and pushed to main while I was mid-arc (I read the log before it landed — true
independent reproduction, the Arc-10 defense, cf. 2008/3009). Same KILL, complementary methods: arc 2011
cheap-killed at observation by catching the thin-tail/pair-mix confound (median −0.069 ≪ mean; −0.133
excluding AUDUSD/USDJPY); arc 3011 took it to the honest engine (first short to do so) and showed it nets
~0 / 0-of-18-cells / beats-null-only-+0.021pp — the engine-side fingerprint of "never robust." Its
robustness analysis corrected my initial "structure pass" over-claim.

**Tooling.** BUILT (new): `FailedBreakoutRejectionShortSignal` → `discovery/tools/failed_breakout_signals.py`
(short mirror of `FailedBreakdownReclaimLongSignal`; mask + ATR geometry only, `direction=Direction.SHORT`)
— registered in TOOL_REGISTRY. Reused `observe_long_capture(direction="short")`,
`build_null_signal_evaluation`, and the canonical entry point (CALLED, never re-rolled).

**FLAGS (code not merged).** None requiring the canonical core (the short path was already merged, PR #273;
this arc added only an EXPERIMENT signal under `discovery/tools/`). Carries the standing FLAG-1 (the
2018-positive 4th portfolio leg is not a directional construction — narrowed further: not even a
structure-confirmed short fade) + the `A1Config.time_exit_bars`-unwired flag. Drivers scratch
`_disco3_work/arc3011_observe_breakout_short.py`, `arc3011_engine.py`, `arc3011_null.py` (reproducible:
`PYTHONPATH=. py discovery/_disco3_work/arc3011_engine.py`, `histdata_root=C:\Users\panap\histdata_backup`).
### arc_2010

**Market-neutral relative-value (the arc-3004 escalation's #1 unlock), shorts-enabled** (chat 2000s).
Full record: [`arcs/arc_2010_relative_value_market_neutral.md`](arcs/arc_2010_relative_value_market_neutral.md).
No council (coin-flip + non-monotone-noise observation across 18 cells; arc-1014/2009/3007 precedent).

**Idea + why (the escalation's #1 lever, now testable).** The arc-3004 escalation ranked
market-neutral/relative-value as unlock #1 — "a second simultaneous leg; the only lever that does NOT
require beating 0.50 per trade." Arc 2003 tried it long-only and could only CONCEDE (its Devil lens
judged the long-only constraint the binding blocker — couldn't short the leader of a divergent pair).
Shorts are now open (PR #273), so the concession is testable: a genuinely market-neutral pair trade
(long one leg / short the other of a cointegrated pair). Arc 2003's own finding gives the *because* for
the side — relative performance is NOT mean-reverting ("the laggard keeps lagging" = relative MOMENTUM)
— so the bet is long-outperformer/short-underperformer, betting the ordering persists; positive when
orderings persist = strong-trend years (2018) → a candidate 2018-positive 4th component. The decisive
question is THE COST REALITY: a 2-leg trade pays FundedNext cost on BOTH legs (~2× the single-leg hurdle
the whole corpus couldn't clear); the relative drift must clear the DOUBLED cost.

**What happened — FALSIFIED at observation.** 3 cointegrated groups (EUR-GBP, AUD-NZD, EUR-AUD; all
XXX/USD so the common USD factor cancels), H4 and D1, IS 2010–2020; relative-strength = logret_A(LB) −
logret_B(LB), z-scored causally; forward relative move signed by the momentum bet. **Relative strength
has ≈ZERO predictive power** — corr(relstr, fwd_rel) ∈ [−0.048, +0.006] across all 6 group×TF cells;
frac+ coin-flip 0.43–0.55. **No cell clears the doubled cost on an honest read:** every H4 cell is
net-negative; the only faintly-positive cells are D1 EUR-GBP |z|≥2 (+4.53 bp net) and EUR-AUD |z|≥1/≥2 —
but THIN (n=153–210, ~14–19/yr), **NON-MONOTONE in |z|** (EUR-GBP is −14.49 bp net at |z|≥1, +4.53 at
|z|≥2 = the arc-3007 not-a-lever tell), frac+ ≈ 0.51–0.55, and net-positive ONLY on the spread-only cost
that **understates** the true 2-leg FundedNext cost (adds ~0.5pip slippage + $5/lot RT per leg → flips
net-negative).

**Verdict: KILL (cheap-kill at observation).** Market-neutral relative-value on correlated FX majors
does not clear the 2-leg cost at H4 or D1; coin-flip relative move (corr ≈ 0), the few +cells are thin
non-monotone noise on understated cost. Entry doesn't beat coin-flip + no robust +gross relative drift →
§5f doesn't bite; no pool/engine/council spent.

**Diagnosis — the second leg doubles cost faster than it adds edge.** The escalation's hope was that a
second leg sidesteps the "beat 0.50" wall — it does, but replaces it with a DOUBLED cost wall, and the
relative drift (~2–10 bp) is smaller than the doubled FundedNext cost (~6–14 bp), not 2× larger. The
deeper kill is corr ≈ 0 (no relative predictability to scale, hedge-ratio aside). This **directly closes
arc 2003's CONCEDE:** relative-value's blocker was NOT (only) the long-only block — with shorts enabled
it STILL fails, on the doubled-cost EDGE<COST wall against a coin-flip relative move. Removing direction
risk did not buy an edge; it bought a second spread.

**Threads / lessons.** (1) **The escalation's #1 unlock (market-neutral/relative-value) is itself
EDGE<COST on liquid FX majors** — tested honestly now shorts are open; the 2nd leg doubles cost without
doubling edge, relative move is a coin-flip (corr ≈ 0) sub-2×-cost at H4 AND D1. (2) **Closes arc 2003's
CONCEDE with the real reason** — the blocker was the doubled cost vs a coin-flip relative move, not the
long-only constraint; the FLAG-1 shorts unlock does NOT revive relative-value (it was never the binding
constraint there). (3) **Non-monotone-in-|z| net-positive cells = the arc-3007 noise signature**,
re-confirmed on a relative-value construction; the strongest-|z|-only positive cell with deeply-negative
milder cells is multiple-comparison selection noise, not an edge — no engine compute. (4)
**Cost-favorable-TF discipline (arc-3008 lesson):** 2-leg cost is spread-dominated (≈constant bp across
TF) while a longer hold captures a bigger relative move, so D1 is the cost-favorable resolution — checked
explicitly before the KILL; D1 did not rescue it. (5) **The 2018-positive 4th component remains unfound
and the relative-value route to it is now closed** — of the escalation's three ranked unlocks,
relative-value (#1) is cost-bound on majors, short-side asymmetries (#2) are mapped (the up-gap weekend
SHORT, 1000s' arc-1015 lane, is a fade that inherits the 2018 tail), leaving the genuinely-tighter-cost
execution regime (#3) as the operator lever the apparatus cannot self-supply.

**FLAGS (code not merged):** none. No canonical-core change; no new BUILT tool (pre-pool spread/drift
screen is a one-off scratch observer, like arc 3005/1008). A two-leg book, if an edge ever appeared,
would be scored as two canonical single-pair legs combined via BUILT `combine_fold_roi` (additive;
sidesteps any multi-leg apparatus gap) — so relative-value scoring needs no engine change; the kill is
the absence of edge, not an apparatus block. Driver scratch `_disco2000_work/arc2010_observe_relmom.py`
(reproducible from the arc doc).
### arc_1015

**3-way PORTFOLIO combination WFO (gap-fill 1006 + month-end 1011 + failed-breakdown 1013) — the
route's deployable gate** (chat 1000s). Full record:
[`arcs/arc_1015_portfolio_combination_3way.md`](arcs/arc_1015_portfolio_combination_3way.md). No
council (a direct quantitative combination test — not a PASS survivor, not an idea-fork/diagnosis;
exactly arc 2006's situation). **Note:** the 2000s + 3000s ranges independently ran the same 3-way as
arcs 2008 + 3009 concurrently (triple-independent reproduction, identical verdict + identical blocking
fold) — this arc 1015 is the 1000s-range record, landed after on the shared log.

**Idea + why (log-seeded — the gated next step).** After 26 arcs the directional space is closed and
the programme has exactly THREE net-positive long-only PORTFOLIO components. Arc 2006 ran the 2-way
(gap-fill + month-end) → KILL, blocked by a mutually-negative 2015 fold, and specced a ≥3rd component
selected for fold-complementarity (positive on the 2-way's negative folds 2015/16/18/20). **Arc 1013
IS that 3rd component** (positive 2015/16/20, negative only 2018) and flagged "run the 3-way." Item #5
on the dispatch frontier, unblocked for the first time — the most decision-relevant unrun experiment,
decisive either way (first PASS candidate, or a precise spec for the missing piece).

**Method (CALLED, never re-rolled).** Each component reproduced via its REGISTERED signal over the
same canonical `build_v3_folds` IS folds (10, fold 6=2015, fold 9=2018), scored solely by
`MultiPairBacktester` (`ArcFoldRunner`/`run_config_over_folds`, FundedNext ON, SL-first); per-fold ROI
combined with the BUILT `combine_fold_roi` (equal + risk-parity, weights fit on IS and FROZEN); the
all-folds-positive judge applied to the COMBINED book. Arc-10 discipline: all three reproduce
**byte-exact** — gap +0.685%, month-end +0.232%, failed-breakdown +1.854% (9/10, per-fold vector
identical; the fbr match required `trail_enabled=True`, the A1Config default left ON in 1013's
committed config alongside `exit_policy="sl_plus_trailing_atr"` — a double-trail quirk;
`trail_enabled=False` gives +2.084%/8-of-10, caught by reproducing the per-fold vector not just the mean).

**What happened.** The 3rd component worked **exactly as arc 1013 predicted** (positive 2015/16/20),
cutting the 2-way risk-parity book (arc 2006: 4/10 neg, worst −1.53%) to **2/10 neg, worst −0.770%,
mean +0.552%** (risk-parity gap 0.106 / me 0.726 / fbr 0.168) — the strongest portfolio book the
corpus has produced. Equal-weight: 3/10 neg (2013/2015/2018), worst −3.36%, mean +0.92%. But **NOT
all-folds-positive → KILL the combined book** (§11). IS not all-folds-positive → **OOS deliberately
NOT touched** (§4 + holdout preservation).

**Why it's blocked — provable, combination-method-invariant.** No single convex weighting can pass,
and this does NOT rest on the linear-combination approximation: **2018** is positive in only ONE
component (month-end +0.90; gap −6.79 AND fbr −4.20 both deeply negative) and **2015** in only ONE
(failed-breakdown +3.17; gap −4.19 AND month-end −1.14 both negative). **2015 wants weight on fbr,
2018 wants weight on me — opposite directions — and neither fbr-solo nor me-solo is all-folds-positive
(both fail the OTHER fold).** No component is positive in BOTH 2015 and 2018, so no weighting clears
both. (Arc 2008's brute-force convex search confirms 0/5151 weightings pass.) The two surviving
blockers are persistent strong-USD / risk-off years where every long flow-reversion +
structural-reclaim edge bleeds simultaneously (the take-the-loss tail of the whole long-reversion family).

**Verdict: FAIL the sole judge (combined book) → KILL.** The three components are UNCHANGED
(reproduced exactly), retain PORTFOLIO; **no new `portfolio-candidates/` entry** (would triple-count).
The portfolio thread stays ACTIVE with a razor-sharp 4th-component spec.

**Threads / lessons.** (1) **The 3-way is the strongest book the corpus has produced** (risk-parity
2/10 neg, worst −0.77%, mean +0.55%) yet still not deployable — two near-zero-corr mean-positive
components + a genuinely regime-orthogonal third HALVE the 2-way's negative folds but cannot clear the
gate, because the complementary leg (fbr, covers 2015/16/20) does not also cover 2018. (2) **The block
is now pinned to two specific folds (2015 & 2018) with a structural cause and a precise, falsifiable
spec** — far tighter than arc 2006's "need a 3rd." (3) **A 4th LONG reversion edge would re-inherit the
2015/18 tail and not help** (arc 2006's lesson confirmed by adding the most complementary long available
and STILL failing). (4) **Selection criterion refined (extends arc 2006 #2):** rank a candidate not by
aggregate ROI on the book's negative folds but by whether ONE component covers EACH residual
mutually-negative fold; the next must cover the *intersection the current set leaves open* (2015 ∩
2018) — positive where the existing complementary legs disagree. (5) **The escalation is at its sharpest
form:** a deployable book is exactly ONE component away, structurally a **short / USD-trend** leg
(positive in strong-USD/risk-off 2015 & 2018), which the long-only reversion menu cannot provide →
**FLAG-1 shorts/second-leg unlock is the specific named blocker with a 2-fold acceptance test.** The
concurrent 2000s/3000s follow-ups already tested the obvious structural/trend shorts for the 2018 leg —
climax-sweep SHORT (arc 2009) and sustained-USD-strength/trend SHORT (arc 3010), **both KILL** (the
swing-low sweep is a reversal not a continuation; strong downtrends revert, the symmetric completion of
arc 3003) — so the 2018 leg is NOT a directional/trend short. **Next live frontier for the 4th leg:**
genuine FLOW-EVENT short asymmetries where measured accuracy >0.50 (the up-gap weekend SHORT, arcs
2001/2003: drift −0.57 ATR, frac+ 0.64), checked first for its 2015 & 2018 sign — not symmetric
direction/trend (dead by symmetry, 1014/2009/3010).

**Tooling:** no new BUILT tool. Reused BUILT `combine_fold_roi` (2006), `WeekendGapFillLongSignal`
(2001), `MonthEndReversionLongSignal` (1011), `FailedBreakdownReclaimLongSignal` (1013),
`make_time_exit_predicate` (1005) — all CALLED, scoring stayed canonical.

**FLAGS (code not merged):** none requiring the canonical core. Carries the standing
`A1Config.time_exit_bars`-unwired flag (1005/3004) + FLAG-1 (long-only blocks the regime-orthogonal
short/trend leg the route now specifically needs for 2015 & 2018). Config-quirk note: arc 1013's
committed config runs double-trailing (`trail_enabled=True` default + `exit_policy="sl_plus_trailing_atr"`)
— it is the committed component definition (reproduces +1.854% exactly), combined as-is. Linear-combination
LIMITATION FLAGGED but immaterial (the 2015/2018 ROI signs are large + unambiguous; cross-TF
co-simulation is impossible in one A1 run — H4 vs D1). Driver scratch `_disco_work/arc1015_combo3.py`.

### arc_2011

**Failed-breakout REJECT SHORT at a swing HIGH — the true forward-confirming mirror of arc 1013** (chat
2000s). Full record: [`arcs/arc_2011_failed_breakout_reject_short.md`](arcs/arc_2011_failed_breakout_reject_short.md).
No council (a +gross-drift entry proven a 2-pair thin-tail artifact; arc-1010/3002/3003 cheap-kill).

**Idea + why (the mirror nobody tested).** arc 1013's failed-breakdown RECLAIM long is the strongest
corpus edge *because* the reclaim is **forward-confirming** (pierces a swing LOW, closes back ABOVE → the
i+1 long enters AFTER the adverse low). The two prior "short mirrors" both KILLed for being
**backward-confirming**: arc 1014 (swing-LOW pierce + close BELOW = confirmed-breakdown *continuation*)
and arc 2009 (climax *continuation*). The TRUE forward-confirming short mirror was never tested: a
**failed-breakOUT REJECT** at a swing HIGH — sweep buy-stops above a 40-bar swing high, then REJECT back
below with a big upper wick (bull-trap → reversal DOWN); the i+1 short enters AFTER the adverse high. On
a USD major a failed-rally reject → short → continues down in strong-USD 2018 = a candidate 2018-positive
4th leg. Exact mirror of BUILT `FailedBreakdownReclaimLongSignal` (low→high, lower→upper shadow,
reclaim→reject), K=40, shadow≥1.25.

**What happened — promising pooled signal, FALSIFIED by robustness.** 7 USD majors, H4, IS, short lens
(base cap 0.4849, drift −0.0151). The POOLED observation looked like the first passing short: drift
POSITIVE and **monotone in the rejection wick** (s≥0.5 +0.171 → s≥1.5 +0.463), and the **structure
control APPEARED to pass** — AT-swept-high +0.261 (continues down) vs ELSEWHERE −0.213 (reverses up), the
opposite of arc 2009's anti-load-bearing result. BUT capture stayed **coin-flip (0.47–0.51)** vs 1013's
0.55–0.61. **Robustness destroyed it:** (a) mean ≫ median (s≥1.25 mean +0.261 vs median −0.069 = thin
tail); (b) **excluding the 2 thin outlier pairs (AUDUSD n=25 +1.26, USDJPY n=31 +1.49) collapses it** —
s≥1.0 +0.242→+0.085 (cap 0.489), s≥1.25 +0.261→**−0.133 NEGATIVE** (cap 0.45); (c) the deeper/cleaner
cell INVERTS negative — the **opposite** of arc 1013's deepens-the-edge signature; (d) the single-pair
LOO "passed" only because dropping one outlier left the other inflating the pool (a 2-outlier artifact
hides from a 1-out LOO); (e) per-pair is the arc-2009 USD-quote split (XXXUSD + / USDXXX −).

**Verdict: KILL (cheap-kill at observation, robustness-confirmed).** A 2-pair thin-tail artifact, not a
structural edge: robust drift coin-flip-to-negative, median ~0-to-negative, capture coin-flip, deeper
cell inverts. The structure-control "pass" was a PAIR-MIX confound (the AT-swept cell is
disproportionately the two thin-tail pairs). No engine compute — a coin-flip-capture/median-negative/
outlier-carried drift collapses SL-honest (arc 3003) and there is no robust edge for a §5f exit to
rescue (arc 0/1004); the LOO+outlier+median robustness IS the airtight §5f-grade justification.

**Diagnosis — arc 1013's reclaim-long has NO tradeable short mirror, in ANY construction.** Three short
constructions of the swing-sweep idea are now dead: confirmed-breakdown continuation (1014), climax
continuation (2009), failed-breakout reject reversal (2011). The forward-confirming property that made
1013 robust does NOT transfer to the high: 1013's swing-LOW reclaim is robustly load-bearing (LOO-all-
positive, cap 0.55–0.61, deeper-grows-the-edge); the swing-HIGH reject's edge is pair-mix/thin-tail and
inverts on the robust population. **The strongest corpus edge is an asymmetric LONG-ONLY structural
edge** — the swing-sweep short well is exhausted.

**Threads / lessons.** (1) **arc 1013's reclaim-long has no tradeable short mirror in any of 3
constructions** (1014/2009/2011, continuation AND reversal) — an asymmetric long-only edge; forward-
confirming is necessary but not sufficient (must also be robust, not a 2-pair thin tail). (2) **A
2-outlier thin-tail artifact hides from a 1-out LOO** — the decisive cuts are exclude-ALL-outliers,
mean-vs-median, and deeper-cell-direction (sharpens arc-1010's "positive drift under coin-flip capture =
artifact"; re-usable for any +drift/coin-flip-capture cell). (3) **A structure-control pass can be a
pair-mix confound** — re-run it within the outlier-excluded universe (or per-pair) before believing it;
the one gap in the 1013/1014/2009 structure-control discipline, now patched. (4) **USD-quote split
(XXXUSD + / USDXXX −) is the short-side USD-beta tell** (re-confirms 2009), here co-occurring with the
thin-tail (the + pairs are the thin-n outliers). (5) **The 2018-positive 4th component is unfound and the
structural-short route to it is fully closed** (1014/2009/2011); with relative-value cost-bound (2010)
and trend-shorts dead (3010), the portfolio route's 2018 wall stands — the only un-mapped >0.50-accuracy
short is the FLOW-event up-gap weekend SHORT (1000s' arc-1015 lane, a fade that inherits the 2018 tail).
Reinforces the arc-3004 escalation.

**FLAGS (code not merged):** none. No canonical-core change; no new BUILT tool (swing-high reject
conditioning + LOO are one-off scratch observers). The first-ever engine short-run was NOT needed
(resolved at observation by robustness), so the canonical short pool/engine path stays observation-
verified only (arc 1014) — a future short clearing observation will be the first to exercise it
end-to-end. Drivers scratch `_disco2000_work/arc2011_observe_failed_breakout_short.py`,
`arc2011_loo.py` (reproducible from the arc doc).

### arc_1016

**Up-gap weekend SHORT — the dispatch's #1 short lead & arc-1015's named 2018-leg candidate** (chat
1000s). Full record: [`arcs/arc_1016_upgap_weekend_short.md`](arcs/arc_1016_upgap_weekend_short.md). No
council (falsified at observation; arc-1014/3010 discipline).

**Idea + why.** My arc 1015 specced the missing 4th portfolio component: positive in BOTH 2015 & 2018,
structurally a SHORT. The dispatch's #1/#2 leads + arcs 1014/1015 all name the same first target — the
**up-gap weekend SHORT**, the mirror of the arc-1006 gap-fill LONG (PORTFOLIO): arcs 2001/2003 measured
weekly-open UP gaps drifting −0.57 ATR / frac+ 0.64 ("the stronger leg," long-only-blocked, now open
PR #273). Unlike arc 1014's backward-confirming breakdown short, this is a discrete weekend-illiquidity
flow event where the reversion might still be ahead of the i+1 entry. Its **2015 & 2018 sign is the
acceptance test.**

**What happened — FALSIFIED at observation.** Direction-aware honest observe (`observe_long_capture(
direction="short")`, H4, weekly-open UP-gap bars ≥ thr·ATR, i+1 short). **No capturable i+1 short edge
on either universe:** JPY crosses up-gap thr+0.5 cap 0.4484 (<0.50), drift −0.093 (NEGATIVE for the
short → the up-gap CONTINUES up at i+1, doesn't revert), per-pair 1/5; USD majors cap 0.4175, drift
−0.065, 3/7; coin-flip-or-adverse across thresholds 0.5/1.0/1.5 on both. **2015 & 2018 test:** USD
majors 2015 −0.65 / 2018 −0.21 both NEGATIVE; JPY crosses 2015 +0.36 (cap 0.357) / 2018 +1.99 (n=31)
*look* positive but are **regime-luck within a coin-flip** (overall cap<0.50 + drift −0.09, per-pair
1/5, tiny n, per-year drift swinging +2.15/−2.74/−1.88) — arc-3010's exact lesson.

**Verdict: KILL (cheap-kill at observation).** Cap<0.50 + ≤0 short drift → §5f exit-sweep doesn't bite
(reserved for beats-null / +gross-drift entries); a sub-0.50/adverse gross signal is guaranteed sub-cost
on the honest engine (arc 3003) → no pool/engine/null/council spent.

**Diagnosis.** arc-2001's "+0.57 ATR / stronger leg" was measured from the **untradeable gap-bar OPEN**
(hindsight, arc-2001's own caveat). At the honest i+1 entry the reversion is **already spent** and the
up-gap continues up — the **backward-confirming** failure of arc 1014, re-confirmed on a flow event. The
gap-fill-long mirror does NOT flip favorably: the down-gap-fill long catches a reversion still ahead of
i+1 AND rides the Abenomics JPY-cross up-drift tailwind, whereas the up-gap short fights that same
up-drift as a headwind and the surviving i+1 magnitude is below the coin-flip/cost line. Forward-
confirming reversal structure is tradeable (1013); backward-confirming flow continuation is not.

**Threads / lessons.** (1) **The up-gap weekend SHORT — the top-named short lead and the named 2018-leg
candidate — is a cheap-kill at the honest i+1 entry;** arc-2001's "stronger short leg" is a gap-bar-open
hindsight artifact. (2) **The 2018-positive 4th component is NOT in ANY short construction tried** —
structure (1014), climax (2009), reject (2011), trend (3010), now flow (1016) all KILL; shorts revive
neither directional structure nor trend nor a symmetric flow-leg. The arc-1015 3-way book's 2018 wall
stands. (3) **The arc-3010 "regime-luck within a coin-flip" screen is load-bearing for the 2018 hunt:** a
cell positive in 2015/2018 but sub-0.50 + per-pair-minority + tiny-n is noise, not the leg — promoting it
to the engine would be fishing the 2 target folds (arc-1012 trap); the test is "robustly short-positive
in 2015 & 2018," not "the 2015/2018 cells happen to be positive." (4) **Surviving frontier for the 2018
leg:** market-neutral relative-value (arc 2010 KILLed — doubled cost vs coin-flip) and a genuinely
tighter-cost execution regime (operator decision) — the in-apparatus short leads are now exhausted; the
3-way book (1015) stands as the strongest non-deployable result, arc-3004 escalation reinforced.

**Tooling:** no new BUILT tool (reused the direction-aware `observe_long_capture`; up-gap mask is a
one-off scratch helper). **FLAGS (code not merged):** none. Carries FLAG-1 (shorts open but the named
flow/structure/trend short leads now dead for the 2018 leg) + the `A1Config.time_exit_bars`-unwired flag.
Driver scratch `_disco_work/arc1016_upgap_short_obs.py`.
### arc_2012

**Deep multi-factor trend-CONTINUATION long** (chat 2000s). Full record:
[`arcs/arc_2012_deep_trend_continuation_long.md`](arcs/arc_2012_deep_trend_continuation_long.md). No
council (a structure-control INVERSION is a falsification, not a tuning fork — arc-3003/1014 discipline;
capture<0.50 ⇒ no reachable ceiling, the oracle-best-cluster there is the Arc-0 hindsight trap).

**Idea + why.** The portfolio route (2006/2008/3009/1015) is provably blocked on **2018** (and 2015): all
three PORTFOLIO components are fade/reversion → tail-correlated, all bleed trend years; 0/5151 convex
weightings clear all-folds-positive. The missing 4th leg must be **trend/2018-POSITIVE**. The short and
relative-value routes to that leg are exhausted (structural shorts 1014/2009/2011, trend-short 3010,
relative-value 2010). The one untested face (dispatch item d) is a **deep multi-factor directional
LONG** in the arc-1013 template. Every prior trend cut was SHALLOW (Donchian/SMA → coin-flip; 3003:
strong trends revert), but 1013 proved a DEEP conjunction extracts an edge where the shallow version is
dead — via being **forward-confirming** (the reclaim enters AFTER the adverse low, before the up-move).
Nobody applied that property to trend CONTINUATION — intrinsically trend-positive, a LONG (no
short-mirror death). Hypothesis: uptrend × shallow pullback holding the higher-low × strong impulse ×
forward-confirm resume (close>prior-high) extracts the trend edge and is +2018.

**What happened — FALSIFIED at observation (decisive, no engine).** H4 USD majors, IS 2010-2020, honest
+1R-before-SL long capture + i+1 drift via direction-aware `observe_long_capture`. Base cap 0.4860,
drift −0.0474. (Q1) the deep continuation cell cap 0.4787–0.4865 (≤ base, all <0.50) and a STRONGER
impulse gate makes it WORSE (impulse≥0.5 → 0.4787/−0.105) — the more "established" the trend, the worse
(trade-level echo of 3003's regime inversion). (Q2) the forward-confirm RESUME does NOT rescue
continuation: resume −0.105 vs into-the-dip −0.156, **both negative, both sub-0.50**. (Q3) STRUCTURE
CONTROL **inverts** — FULL deep conjunction 0.4787/−0.105 is WORSE than the generic resume-in-uptrend
0.4956/+0.0070; the deep trend structure pulls a near-coin-flip bounce NEGATIVE (the exact opposite of
1013, where AT-structure ≫ elsewhere). (Q4) 1/7 pairs positive (USDJPY +0.094, the carry pair); 6/7
negative — below the noise floor. (Q5) **2018 drift −0.233 (NEGATIVE)**; per-year is a coin-flip (4/11
positive, range −1.33[2019]..+0.81[2020]). Not the trend leg.

**Verdict: KILL (cheap-kill at observation).** Sub-0.50, forward-confirm reversal-specific, structure
anti-load-bearing, 2018 negative.

**Threads / lessons.** (1) **arc-1013's forward-confirm property is REVERSAL-SPECIFIC, not a general
edge.** It works only because the reclaim enters BEFORE the bet's move begins; applied to continuation the
same trigger enters AFTER the bounce (late) and reverts. Forward-confirming is necessary-not-sufficient —
it helps only when the confirmed event is the START of the move, not its tail. This is the mechanistic
reason 1013 has no continuation sibling (and complements the 1014/2009 finding that it has no short
mirror). (2) **Deep multi-factor structure does NOT rescue trend continuation** — the trade-level
completion of 3003: the cleaner/stronger the trend structure (impulse↑, full vs generic), the MORE
negative. Structure HELPS reversals (1013), HURTS continuation. (3) **The portfolio's 2018-positive leg is
not a structural-continuation long either** (2018 −0.233), adding to the closed structural-short
(1014/2009/2011) and trend-short (3010) routes — every directional FACE is now mapped: shallow long/short
dead, deep reversal long = the 3 existing fades, deep continuation long dead here, structural/trend short
dead. The trend/2018 flavor is not expressible as a price-structure directional bet in either direction →
**arc-3004 escalation reinforced from the long-continuation angle**: the 4th leg needs a genuinely
non-price-direction construction (or the route is escalation-bound). (4) **Surviving frontier
(unchanged):** the FLOW-event up-gap weekend SHORT (arcs 2001/2003, acc 0.64 — the one place direction
beats 0.50) is STILL unrun (1000s used arc 1015 for the portfolio combination, not the up-gap short) and
is the next live target — descends from arc 2001 (2000s lineage). It is a FADE so likely inherits the
2018 tail (not the portfolio's 2018 leg), but is the single highest-acc directional edge in the corpus
and a candidate 4th decorrelated PORTFOLIO component / standalone test.

**Tooling:** no new BUILT tool — reused the BUILT direction-aware `observe_long_capture(direction="long")`.

**FLAGS (code not merged):** none. No canonical-core change. Conditioning is a one-off scratch observer.
Driver `_disco2000_work/arc2012_observe_trend_continuation.py` (reproducible from the arc doc).

### arc_1017

**Carry-unwind cascade SHORT on JPY crosses** (chat 1000s). Full record:
[`arcs/arc_1017_carry_unwind_cascade_short.md`](arcs/arc_1017_carry_unwind_cascade_short.md). No council
(a sub-cost / not-all-folds-positive engine result is a clean quantitative KILL, not an idea-fork /
diagnosis / survivor — arc-1015/2011 no-council convention).

**Idea + why.** The deployable gate is an all-folds-positive COMBINED book; the strongest (3-way 1006+1011+1013)
is provably blocked by 2015 & 2018 (strong-USD/risk-off), and the missing 4th leg must be net-positive AND
positive in BOTH. Every prior 2018-leg attempt died (structural shorts 1014/2009/2011·3011, trend-short 3010,
up-gap flow short 1016, relative-value 2010, deep-continuation long 2012). The one documented mechanism
intrinsically aligned with BOTH risk-off years and NOT a symmetric directional coin-flip, still UNMAPPED: the
**carry-unwind cascade** — leveraged carry (long risk-cross/short JPY) force-unwinds in risk-off, JPY bought
back, persistent one-way cascade ("up the stairs, down the elevator"). Short the vol-expansion big-red IGNITION
fired from a built-up carry uptrend = enter at the START of the unwind (forward-confirming, arc-1013), distinct
from arc 3010's established-downtrend short (which reverts). Uses the merged short path (PR #273) non-trivially.

**What happened.** OBSERVATION (direction-aware short lens, H4 carry crosses AUD/NZD/EUR/GBP/CAD-JPY, IS): the
ignition cell (carry-uptrend & vol≥1.5 & big-red≥1.0) is **capture 0.50 (coin-flip)** with **positive forward
short drift +0.134 ATR** (median +0.068). The **structure control PASSES** — the strongest point for the
mechanism: in-carry-uptrend median +0.068 (continues DOWN) vs the SAME big-red+vol bar NOT-in-uptrend median
−0.178 (reverts UP); the carry context genuinely flips a reverting big-red bar into a weakly-continuing one. BUT
the **deeper cell INVERTS** (vol2.0/down1.5 → drift −0.017; the 2011/3011 artifact tell, opposite of 1013) and
per-year drift is noise (−1.26..+1.21). Per §5f (entry shows gross structure-controlled drift → sweep the exit
menu on the engine before FAIL): 5-pair IS WFO, **best exit sl_only mean +0.013%, 4/10 folds positive, NOT
all-folds-positive** (every other exit mean ≤ 0). Fair 5-seed null: real +0.013% vs null −0.031% → beats random
by **+0.044pp = noise floor** (~10× below arc 1006's +0.36pp-over-null).

**The exposure-cap discovery (the load-bearing finding).** My first engine read showed 2015 & 2018 OOS folds
both **+0.19%** — the exact regime-orthogonal profile the route needs — and I nearly proceeded to a 4-way
combination. The per-fold **n_trades killed it: 2015 n=0, 2016 n=0, 2019 n=0, 2018 n=3** on the engine, vs
**92/103/109** in the uncapped pool. Cause: all 5 pairs share **JPY quote**, and carry-unwind ignitions fire
*simultaneously* across them (ONE correlated risk-off cascade), so the FundedNext **2-per-currency exposure cap**
guts exactly the clustered cascade trades — the positive "2015/2018 folds" were near-zero-trade accounting
artifacts. Removing the confound (single-pair AUDJPY): best mean **+0.011%, 6/10, beats null +0.022pp**, none
all-folds-positive; NZDJPY dead (+0.0001pp); EURJPY 2015 flips negative across exits. Real but noise-floor.

**Verdict: KILL.** Real-but-sub-cost (arc-0/1000 signature): beats random by a noise-floor margin, not
all-folds-positive under any exit/pair, capture coin-flip, drift on the cost line and inverting deeper. Mean
60× below the PORTFOLIO archetype and within noise → §11 + conservative bias forbid a portfolio leg on
+0.02pp-over-null noise (cannot diversify a noise-floor component positive, arcs 3000/3001). OOS 2021+ NOT
touched (IS not all-folds-positive, §4).

**Threads / lessons.** (1) **The carry-unwind cascade short is a REAL structural effect but sub-cost** —
structure control passes, but magnitude on the cost line + coin-flip capture → net noise, like every directional
construction (forward-confirming is necessary-not-sufficient, re-confirms 1013/2011/2012). (2) **NEW: a
correlated-cascade signal is structurally UN-SCALABLE into a portfolio leg.** A single-quote-currency cross-basket
fires together in the target regime, so the currency-exposure cap guts the clustered trades that carry the edge
(uncapped 92–109/yr → capped 0–3/yr); the book is internally one redundant bet and capped too thin to size. This
is a general portfolio-construction prior: a risk-off-positive flow edge concentrated in one currency cannot
become the magnitude-meaningful 2018 leg. (3) **2018 is genuinely engine-positive here (AUDJPY +0.07–0.10%, pool
+1.04R — the ONLY short construction positive in 2018 on the honest engine), but tiny + un-scalable + 2015
marginal/inconsistent** → "directionally right, magnitude-dead"; the 3-way's 2018 deficit (−0.77% risk-parity)
dwarfs a +0.07% contribution. (4) **The 2018-positive 4th leg is now unfound across EVERY mapped route** —
structure / climax / reject (1014/2009/2011·3011), trend (3010), up-gap flow (1016), relative-value (2010),
deep-continuation long (2012), carry-unwind flow short (1017, the last named risk-off flow idea) — reinforcing
the arc-3004 escalation: the apparatus's only net-positive edges are calm-year fades; risk-off-positive
constructions are noise-floor / un-scalable. The 3-way book (1015) stands as the strongest non-deployable result.

**Tooling:** BUILT + registered `CarryUnwindCascadeShortSignal` (`discovery/tools/carry_unwind_signals.py`).
Reused BUILT `observe_long_capture` (short lens) + `build_null_signal_evaluation`; CALLED `build_arc_pool`,
`ArcFoldRunner`, `run_config_over_folds`, `judge_all_folds_positive`, `build_v3_folds` — scoring stayed canonical.

**FLAGS (code not merged):** none requiring the canonical core. The exposure-cap effect is a realistic gate
constraint correctly applied (not a defect). Carries the standing `A1Config.time_exit_bars`-unwired flag +
FLAG-1. Drivers scratch `_disco_work/arc1017_*.py` (reproducible from the arc doc).
### arc_3012

**Risk-off vol-EXPANSION SHORT on high-risk-beta majors (AUD/NZD)** (chat 3000s). Full record:
[`arcs/arc_3012_riskoff_volexpansion_short.md`](arcs/arc_3012_riskoff_volexpansion_short.md). No
council (capture-coin-flip + thin-tail + target-fold-negative — a falsification, not a tuning fork;
arc-3003/2011/1016 discipline).

**Idea + why.** The portfolio route (1015/2008/3009) is provably blocked by **2015 & 2018**, both
**risk-off / vol-spike years** where all three long-reversion fades bleed. Every short tried for the
2018 leg conditioned on price STRUCTURE (1014/2009/2011), price TREND (3010), or a FLOW event (1016).
The one untested conditioner is a **risk-STATE** proxy: a realized-**vol-EXPANSION** trigger on the
highest risk-beta majors (AUDUSD/NZDUSD + AUDJPY risk-cross), SHORT. *Because:* dollar-smile /
flight-to-quality — risk-FX falls vs USD in risk-off deleveraging, a flow that **persists** (so the
i+1 short still has move ahead = forward-confirming). Distinct from 3010 (price trend, inverted) and
closed-ground's vol-LEVEL generic-long filter (this is vol-EXPANSION × risk-beta-SELECTION × SHORT,
now expressible PR #273). Pre-registered acceptance test: robustly short-positive in 2015 **AND** 2018.

**What happened — FALSIFIED at observation, on the binding fold.** Direction-aware honest short
(`observe_long_capture(direction="short")`, H4) restricted to vol-expansion bars (ATR.shift1 ÷ trailing-
100 rolling median ≥ 1.3). Pooled RISK basket (AUDUSD+NZDUSD): cap **0.5025** (coin-flip, vs 1013 long
0.55–0.61); drift **mean +0.264 / median −0.004** = thin-tail artifact (arc-2011 mean≫median tell). By
year: 2015 cap 0.654/+0.748 and 2016 0.578/+0.789 POSITIVE — **but 2018 cap 0.369 / drift −1.170,
robustly NEGATIVE across all three pairs** (AUDUSD 0.328/−1.24, NZDUSD 0.415/−1.09, AUDJPY 0.545/−0.30).
The leg LOSES in the exact fold it was built to cover. Controls (EURUSD/GBPUSD) vol-expansion capture
~0.50 too → no directional edge either side.

**Diagnosis — the *because* inverts in 2018.** 2018's vol spikes were **capitulation/washout lows**
(Feb & Q4 risk-off flushes that snapped back), so shorting the spike shorts **into the bounce** —
the backward-confirming failure of arcs 2009 (climax-at-swept-low) and 1016 (up-gap reversion already
spent). 2015's spikes came mid-decline (continuation → short wins). The trigger is **regime-dependent**:
it captures the easy fold (2015, already covered by fbr) and fails the binding one (2018). Rescuing 2018
via a threshold sweep would be fishing the target fold (arc-1012 trap); the mechanism is structural and
all three pairs agree → 2018-negative is honest.

**Verdict: KILL (cheap-kill at observation).** Coin-flip capture + thin-tail median≈0 drift +
2018-robustly-negative → §5f doesn't bite (a coin-flip/median-≈0/target-fold-negative cell collapses
SL-honest, arc 3003). No pool/engine/null/council spent.

**Threads / lessons.** (1) **The risk-off vol-EXPANSION short does NOT supply the 2018 leg** (+2015 /
−2018) — a SIXTH dead route to the 2018 leg (structure 1014/2009/2011, trend 3010, flow 1016,
continuation-long 2012, now vol-state short 3012). (2) **Vol-EXPANSION confers no directional edge**
(cap ~0.50 both sides; mean drift is a thin-tail artifact, median ≈0) — symmetric completion of arc-1001
(vol-contraction) + closed-ground (vol-LEVEL): volatility = magnitude not direction at H4, in either vol
direction. (3) **The 2018 wall is mechanism-deep, not construction-shallow** — 2018's tradeable price
events are mean-reverting capitulations, so any *fade* bleeds (the 3 longs) AND any *vol/structure-
triggered short* shorts into the bounce; 2018 is positive only for a slow trend-follower already short
before the spike, a coin-flip on the +1R lens (3010). The 2018 leg is not a price/vol-conditioned
directional bet in either direction → **arc-3004 escalation reinforced**. (4) **Surviving frontier
(unchanged):** in-apparatus directional/flow/vol short leads for the 2018 leg are exhausted; what
remains is operator-gated (tighter-cost execution regime) or a non-price-direction construction not yet
conceived.

**Tooling:** no new BUILT tool — reused BUILT `observe_long_capture(direction="short")`; the
vol-expansion ratio is a one-off scratch conditioner. **FLAGS (code not merged):** none. No
canonical-core change. Driver `_disco3_work/arc3012_observe_riskoff_short.py` (reproducible from the
arc doc).

### arc_3013

**Failed-breakdown RECLAIM long at session-liquidity levels (prior-DAY / prior-WEEK low)** (chat
3000s). Full record: [`arcs/arc_3013_liquidity_level_grab.md`](arcs/arc_3013_liquidity_level_grab.md).
No council (an improvement hypothesis decisively falsified by a lower-capture/negative-drift/
fewer-pairs comparison — a falsification, not a tuning fork).

**Idea + why.** Rather than a 7th strained 2018-leg short, attack the higher-value target: **arc 1013
(the corpus's strongest edge) is 9/10 IS — one fold from a standalone PASS.** 1013's liquidity pool is
a rolling-40-bar SWING low. A documented *because* never tested: stops cluster most densely at
**session-defined extremes — the prior-DAY and prior-WEEK low** (ICT/Wyckoff liquidity pools),
arguably cleaner grab targets. Hypothesis: sweeping+reclaiming those is a stronger/cleaner (maybe
all-folds-positive, or +2018-robust) version of 1013.

**What happened — FALSIFIED; the rolling swing is best.** H4, 7 USD majors, deep-grab shadow≥1.25 ATR,
honest long capture+drift via `observe_long_capture(restrict=<level mask>)`. **swing40 (1013
reproduction):** cap 0.5815, drift +0.191 (med +0.141), **7/7 pairs >0.50** — reproduces 1013's
quality. **prior_day:** cap 0.552 but drift **mean −0.062 (NEGATIVE)**, 5/7 pairs (more fires n=587,
diluted). **prior_week:** cap 0.562, drift +0.103, 5/7 (AUDUSD 0.41), thin n=242. **All three levels
remain −2018** (swing40 −1.28, prior_day −0.72, prior_week −2.30; prior_week worst there).

**Diagnosis — structural pivot > time-based pool.** The 40-bar swing low is a *structurally
significant pivot* (a low that held 40 bars = real support → sweep-and-reclaim is a genuine
failed-breakdown with high reversal probability). The prior-day/prior-week low is a *time-based* level
swept routinely in normal trend/range WITHOUT being a reversal point → the reclaim filter catches many
non-reversals → dilution. **1013's swing-pivot level is load-bearing**, not arbitrary; "swept
liquidity" without "structural support" is not a grab. The **2018 weakness is mechanism-intrinsic** —
a bullish-reversal long fails in strong-USD 2018 whichever low is swept (the breakdown succeeds), so
re-leveling cannot make 1013 +2018 (re-confirms 3012). (OOS years appeared in the by-year print but
were only glanced, never optimized — the level ranking is an aggregate IS-dominated structural-quality
comparison; §4 intact.)

**Verdict: KILL (cheap-kill at observation).** No cleaner/stronger grab, no PASS candidate, no
improvement to 1013; component UNCHANGED (still PORTFOLIO), no new portfolio entry. The alternative
levels have lower capture, negative/weaker drift, fewer pairs positive → nothing to escalate.

**Threads / lessons.** (1) **arc-1013's rolling-40-bar swing low is confirmed the BEST liquidity
level** — a structural reversal pivot beats time-based ICT session pools (prior-day reclaim has
*negative* mean drift); the grab edge needs a *reversal pivot*, not merely *clustered stops*. (2) **The
2018 weakness is intrinsic to the reversal-long mechanism, not the level** (all 3 levels −2018) → 1013
cannot be re-leveled to +2018; the portfolio's 2018 wall is not addressable by re-leveling the best
edge. (3) **1013 stays 9/10 IS, one fold from PASS, with that fold (2018) structurally immovable** in
the reversal-long family → the standalone-PASS-via-1013 route is closed; 1013's value is the 3-way
book's regime-orthogonal (2015/16/20) leg. (4) **Surviving frontier (unchanged):** operator-gated
tighter-cost execution regime, or a genuinely non-price-direction construction — the in-apparatus
directional/structural leads (incl. re-leveling the best edge) are exhausted.

**Tooling:** no new BUILT tool — reused BUILT `observe_long_capture(direction="long")`; per-level
reclaim masks are one-off scratch (the BUILT `FailedBreakdownReclaimLongSignal` already parameterizes
the swing-low version; no value generalizing to the worse time-based levels). **FLAGS (code not
merged):** none. No canonical-core change. Driver `_disco3_work/arc3013_observe_liquidity_levels.py`
(reproducible from the arc doc).

### arc_3014

**WEEKLY (W1) trend-following CONVEXITY — the last untested timeframe lever** (chat 3000s). Full
record: [`arcs/arc_3014_weekly_trend_convexity.md`](arcs/arc_3014_weekly_trend_convexity.md). No
council (median-negative / single-pair / −2018-both-directions is a falsification, not a tuning fork).

**Idea + why.** Two corpus facts leave one un-probed cell: closed ground covers **H1/H4/D1 only** —
the **WEEKLY** horizon (the CTA trend-following classic, strongest trend persistence) was never tested
(1002's "timeframe-invariant" was H4-vs-D1); and arc 2000's lens (trend edge = CONVEXITY/fat tail, not
win-rate) was tested only at H4 (tail *generic* there). W1 is where trend convexity *should* live, and
**2018 was a clean weekly USD trend** → weekly trend-following is the natural 2018-positive candidate,
both directions (shorts open). Convexity-lens observation (forward weekly MFE distribution + drift).

**What happened — FALSIFIED, both lenses.** D1→W-FRI resample, 7 USD majors, Donchian-8w breakout, 8w
hold. LONG breakout: mean drift +0.07 but **median −0.027**; the tail IS mildly trend-selected
(P(mfe≥5) 0.015→0.044, ~3× base — fatter than H4's generic) but median-negative → the positive mean is
a thin fat-tail artifact. **Per-pair 2/7 positive** (USDJPY +1.09, USDCAD +0.38 only; GBP/AUD/NZD/CHF
all negative) = single-pair (USDJPY carry) artifact. SHORT breakout: mean +0.005 / **median −0.126**
(worse). **−2018 BOTH directions** (long −0.26, short −0.07).

**Diagnosis.** The "2018 = clean weekly USD trend → trend-following wins" intuition is FALSE: 2018's USD
strength came in **mean-reverting bursts** (3012's capitulation finding at the weekly scale), so weekly
Donchian breakouts whipsawed and lose both ways. The mild W1 convexity is real but median-negative +
single-pair-carried → sub-cost after FundedNext + SL-first (arc-2000's conclusion, one timeframe up).
The directional/trend base is **timeframe-invariant THROUGH W1**.

**Verdict: KILL (cheap-kill at observation).** Median-negative both directions, single-pair-carried,
−2018 both directions → §5f doesn't bite (collapses SL-honest, arc 3003/2000). No engine/null/council.

**Threads / lessons.** (1) **Directional/trend base is timeframe-invariant THROUGH the weekly horizon**
— W1 (the last untested TF, the CTA classic) behaves like H4/D1; closes the timeframe lever entirely
(H1/H4/D1/W1 mapped). (2) **Weekly trend-following does NOT supply the 2018 leg** (−2018 both
directions) — the **8th** dead route (structure 1014/2009/2011, trend-short 3010, flow 1016,
continuation-long 2012, vol-state short 3012, re-leveling 3013, now weekly-trend 3014), and the most
decisive since trend-following is what the 2018 intuition most predicted. (3) **Convexity is mildly
horizon-dependent** — a refinement of arc 2000's "generic tail": the tail gets *somewhat* trend-selected
as the horizon lengthens (W1 fatter than H4) but never enough to flip the median positive or beat cost.
(4) **Meta-signal (three consecutive cheap-kills 3012/3013/3014):** the in-apparatus directional /
structural / trend / vol / timeframe frontier is comprehensively exhausted for the 2018 leg AND for a
standalone edge — future chats should weight away from re-grinding directional/trend/vol cuts (all
TFs/directions/levels now mapped) toward the operator-gated tighter-cost execution regime or a genuinely
non-price-direction construction (the only un-closed frontier).

**Tooling:** no new BUILT tool — weekly resample + Donchian convexity is a one-off scratch observer
(like arc 2000's H4 MFE probe). **FLAGS (code not merged):** none. No canonical-core change. Driver
`_disco3_work/arc3014_observe_weekly_trend.py` (reproducible from the arc doc).

### arc_1018

**Weekend gap-FILL long on USD-NEUTRAL crosses** (chat 1000s). Full record:
[`arcs/arc_1018_usdneutral_gapfill.md`](arcs/arc_1018_usdneutral_gapfill.md). No council (a coin-flip-capture /
median-zero-drift / 2018-negative cell is a decisive observation cheap-kill — arc 1010/1016/2011 discipline).

**Idea + why.** Nine routes that tried to *predict* the strong-USD/risk-off 2015/2018 regime for the missing
4th portfolio leg all failed. A genuinely different principle: reach the (+2015,+2018) fold profile by
**decorrelation FROM the USD regime** instead — apply the PROVEN net-positive gap-fill flow mechanism (arc
1006, JPY crosses) to **USD-NEUTRAL crosses** (EURGBP, AUDNZD, GBPCHF, AUDCAD, NZDCAD) whose weekend gaps are
intra-bloc-news-driven, not USD-driven. If the fill edge survives there (a flow fade, not a directional bet)
its fold profile should be orthogonal to the USD regime that kills 1006/1011/1013 in 2015/2018. Not closed
ground (gap-fill on USD-neutral crosses is untested; arc-3000's closure was a directional reversion *capture*
cut, not the weekend-gap flow event).

**What happened — FALSIFIED at observation.** Weekend down-gap (≥0.5·ATR) fill long, i+1 entry, 24-bar fill
drift, H4 IS, 5 USD-neutral crosses (BUILT `observe_long_capture` + the 1006 gap mask). n=463 (thin,
~8/pair/yr). **No robust fill edge:** cap **0.4881 (coin-flip ≈ the 0.49 base)**, fill drift mean +0.078 ATR
but **median 0.000** → the positive mean is a thin-tail artifact (the 2011/1010 tell). EURGBP — the cleanest
USD-neutral, where the decorrelation thesis is strongest — drift **−0.014 ≈ 0** (cap 0.512); per-pair mixed
(AUDCAD −0.276, NZDCAD +0.152, AUDNZD +0.154, GBPCHF +0.456), median-zero pooled. **Acceptance test FAILS on
the binding folds:** 2015 +0.390 (cap 0.525, barely >coin-flip) but **2018 −0.085 (cap 0.500, NEGATIVE)** —
USD-neutrality gave noise, not +2018. Per-year pure noise (2012 +0.92 .. 2019 −1.34).

**Verdict: KILL (observation cheap-kill, §5d).** Cap ≈ base + median fill-drift 0.000 + 2018-negative → no
robust edge; §5f doesn't bite (sub-0.50/median-zero, no gross drift to bank); no pool/engine/null/council
spent. OOS untouched.

**Threads / lessons.** (1) **The weekend gap-fill edge is JPY-cross-SPECIFIC (carry/USD-related), NOT a
universal weekend-gap property.** On USD-neutral crosses the fill drift is coin-flip / median-zero — those
gaps are efficient/random-walk with no systematic reversion. Sharpens arc 1006/1009: 1006's edge AND its
null's +0.33% JPY-basket drift both ride JPY-cross carry/USD structure; remove that exposure and the fill
edge vanishes with it. (2) **Decorrelation-from-USD is NOT a free path to the (+2015,+2018) leg** — removing
the USD/carry exposure that makes 2015/2018 negative *also* removes the edge; the edge and the tail are the
SAME exposure. A USD-neutral instrument is decorrelated from the blockers but net-zero (can't diversify a
book positive, §11 / arcs 3000/3001). Closes the "USD-neutral net-positive fade" sub-route: net-positive FX
fades are USD/JPY/carry-driven (hence 2015/2018-exposed), so a USD-neutral one is net-zero. **10th dead route
to the 2018 leg.** (3) The portfolio's 4th-leg spec (net-positive AND +2015 AND +2018) is now empirically
unreachable by USD-prediction (9 routes) AND USD-decorrelation (this arc) — converging hard on the arc-3004
escalation: the apparatus's net-positive edges are USD/carry-exposed calm-year fades, structurally
2015/2018-tailed. The 3-way book (1015) remains the strongest non-deployable result.

**Tooling:** no new BUILT tool — reused BUILT `observe_long_capture` (long lens, `restrict` mask) + the
arc-1006 weekly-gap mask logic (one-off scratch). CALLED canonical `Panel.from_pairs`.

**FLAGS (code not merged):** none. Driver `_disco_work/arc1018_observe_usdneutral_gapfill.py` (reproducible).
### arc_2013

**Weekend UP-gap weekend SHORT, JPY crosses** (chat 2000s). Full record:
[`arcs/arc_2013_weekend_upgap_short_jpy.md`](arcs/arc_2013_weekend_upgap_short_jpy.md). **KILL —
converges with the independent 1000s arc 1016.** No council (convergent reproduction + fold-decomposition
decisive; arc-3010/1016 regime-luck discipline).

**Idea + why.** arc 2001 found the weekly-open gap-fill symmetric on H4 — UP gaps drift −0.57 ATR (frac
DOWN 0.64), flagged the "stronger leg" (FLAG-1); arc 1006 found the DOWN-gap fill LONG mean-positive on JPY
CROSSES (the one PORTFOLIO long). Shorts merged (PR #273). The up-gap SHORT on JPY crosses — the mirror of
1006's strong leg and 1015's named candidate 2018-leg — was the highest-acc still-untested short. (The
1000s chat tested the same idea concurrently as arc 1016; we converge to KILL.)

**What happened.** OBSERVATION (direction-aware observe, H4): JPY up-gap≥1.0 honest i+1 capture 0.518 /
drift +0.08 / median +0.146 LOOKED promising, but it is a single fragile band — ≥0.5 median −0.115
(negative typical), ≥1.5 INVERTS to −0.22, leave-one-pair-out drop-AUDJPY → −0.107. Majors drift −0.355
(continues up, dead). The independent 1000s arc 1016 got the entry coin-flip-to-adverse (JPY cap 0.448 /
drift −0.093). Because the ≥1.0 band clears 0.50 and beats the fair weekly-open null (+0.27 ATR), §5f
required the honest engine before a FAIL — so I ran it (the FIRST end-to-end short engine run).

**Honest engine IS WFO (§5f exit menu).** The merged short path is GREEN — pool builds sign-correctly
(mean final_r +0.137 at thr 1.0), WFO runs, FundedNext costs net symmetrically, no FLAG, no canonical
change. thr ≥1.0 is mean-positive under the overshoot-exit family (sl_plus_trailing_atr +0.745%, tp_3r
+0.510%, partial +0.324%, tp_2r +0.015%; non-overshoot exits negative), beats a random-ANY-bar short null
(−0.99%) by +1.74% — but **NOT all-folds-positive (5/10 neg)**.

**The crux (why KILL not PORTFOLIO).** The +0.745% IS mean is **thin REGIME-LUCK**: per-fold (trailing_atr)
2011 +1.91(n5), 2012 +2.21(n14), 2013 −2.24, 2014 −3.14, 2015 −0.57(n2), 2016 −3.18, 2017 +1.97, **2018
+5.08(n8)**, **2019 +5.92(n10)**, 2020 −0.51(n1) — **excluding 2018+2019 the other 8 folds average −0.44%
(net-negative)**. The entire positivity is 2 thin lucky years, exactly the "2018-positivity is uncapturable
regime-luck within a yearly coin-flip" the 1000s (1016) + arc 3010 diagnosed; and it fails the 2015 half of
1015's spec (−0.57). The engine null is random-ANY-bar (so +1.7% partly reflects up-gaps firing at price
extremes); the FAIR weekly-open null gives only +0.27 ATR. A real-but-regime-luck signal is KILL (§11) —
you cannot diversify thin regime-luck into a robust book.

**Verdict: KILL** (converges with 1016).

**Threads / lessons.** (1) **The up-gap weekend SHORT is dead — convergent KILL across two independent
chats** (2000s engine + 1000s observation). arc-2001's −0.57 ATR "stronger leg" was the untradeable
gap-bar-OPEN (hindsight); the honest i+1 short is coin-flip-to-adverse (backward-confirming like arc 1014;
JPY-basket up-drift taxes it). The gap-fill's only tradeable edge is the DOWN-gap LONG (1006). (2) **The
Arc-10 defense worked again** — independent reproduction (1016) + my own per-fold decomposition caught a
thin-regime-luck "engine positive" before it became a PORTFOLIO record; trusting the +0.745% mean would
have been the single-engine-trust trap. **New lesson: for a thin signal, decompose the mean by fold/year
before any PORTFOLIO claim — a +mean carried by 1–2 thin folds is regime-luck, not a component.** (3)
**First end-to-end SHORT engine run is GREEN** — PR #273's short path (Step-1 pool + architecture +
MultiPairBacktester + cost netting) builds sign-correctly and nets costs symmetrically on a real arc; now
exercised, not just observation-verified (closes arc 2011's note). Reusable: `WeekendUpGapShortSignal`
(BUILT), direction-aware null. (4) **The 2018-positive 4th leg is unfound in EVERY short construction**
(1014/2009/2011/3011 structure, 3010 trend, 1016/2013 flow, 3012 vol, 2012 continuation-long) — the
portfolio route's 2018 wall stands; arc-3004 escalation reinforced.

**Tooling:** BUILT `WeekendUpGapShortSignal` (discovery/tools/gap_signals.py — kept, valid reusable short
signal); made `build_null_signal_evaluation` direction-aware (carries `direction`; additive, longs
byte-identical). Both registered.

**FLAGS (code not merged):** none requiring the canonical core. Two EXPERIMENT-tool additions in
discovery/tools/ (flow freely §9). Drivers scratch `_disco2000_work/arc2013_*.py` (reproducible).

### arc_2014

**Failed-breakdown reclaim long × downtrend-regime GATE** (chat 2000s). Full record:
[`arcs/arc_2014_fbr_downtrend_regime_gate.md`](arcs/arc_2014_fbr_downtrend_regime_gate.md). No council
(falsified at observation; no worthwhile-ceiling fork).

**Idea + why.** Every 2018-leg attempt (16 arcs) hunted a *separate* 4th portfolio component — all dead.
The untried highest-EV lever: improve arc **1013 itself** (the strongest edge — IS 9/10 folds positive,
the ONLY negative fold 2018) past its single 2018 fold → a *solo* PASS, the first survivor. arc 1013
thread #5 left this open: condition OUT the strong-USD regime with a PRE-REGISTERED causal measure (NOT
fished to flip 2018 — the arc-1012 trap). The reclaim is a liquidity-GRAB reversal; *because*: in an
extreme/established downtrend (2018 strong-USD) a swept low is an INFORMATIONAL breakdown and the reclaim
is a dead-cat pause, so a TAIL gate on downtrend STRENGTH/PERSISTENCE should drop the falling-knives and
keep the 9 good folds. obs#3 only tested a BINARY D1≤SMA50 split (found mild-down FINE); arc 3013 tested
liquidity LEVELS — neither tested downtrend strength as a continuous gate.

**What happened — the data led AWAY from the hypothesis.** Reproduced the 1013 pool (K=40, shadow≥1.25,
7 USD majors, H4, IS; n=237, mean final_r +0.137 ✓). Edge by downtrend-strength buckets:
- **Hypothesis INVERTED:** strong-SMA200-down-slope is the *strongest* bucket (+0.218, n=103) — reclaim
  bounces are sharpest when stretched down; mild-down is the negative slice (−0.245, n=27). Edge peaks in
  BALANCED context (near-SMA200 +0.397) and is weak in established UPtrend (dist>1 +0.041, win 0.10). A
  mean-reversion-works-when-not-trending-up texture — NOT a 2018 separator.
- **Decisive 2018 decomposition — not regime-separable, a near-total wipeout:** 18/19 trades hit the full
  −1R stop; **6 of 7 pairs went 100% to −1R** (AUDUSD/EURUSD/GBPUSD/NZDUSD/USDCAD/USDCHF). Both 2018 trades
  in the "best context" (near-SMA200) are −1.00. The candidate gate drops 58% of 2018 fires yet the *kept*
  trades still average −0.69. The lone non-loser is USDJPY (+0.24, 2 trades — the one pair trending *up
  with* USD in 2018), but at ~2 trades/yr that is un-scalable (arc-1017 mode; arc 3010 already killed the
  USD-trend-alignment angle).
- Any gate that removes the negative pockets (consec_below200>100: −0.30; mild-down) **destroys the good
  folds** (2012 +1.14→−0.09, 2013 +0.36→−0.31, 2019 −0.08→−0.47, 2020 +0.40→+0.12). No free lunch.

**Verdict: KILL (cheap-kill at obs; no engine/null/council — like 3013/3010/1014/1016).** 1013 UNCHANGED
(still PORTFOLIO).

**Threads / lessons.** (1) **NEW:** 1013's 2018 drag is not a tail of bad-context trades but a near-uniform
−1R wipeout across all 7 pairs and ALL trend-contexts (incl. the "best") → the strong-USD-regime failure of
the reclaim-long is **mechanism-intrinsic and entry-time-UNCONDITIONABLE** (in risk-off the failed breakdown
becomes a real breakdown; the reclaim doesn't hold). Independently re-confirms 1013/3013 via a new
conditioner; **closes the "improve 1013 toward a solo PASS" lane.** (2) The reclaim's entry edge is real and
textured (best in balanced/strong-down context, weak in uptrend) but that texture does not isolate 2018 —
building an all-folds gate on it is the arc-1012 fishing trap. (3) The 2018 portfolio leg is now unfound
across ALL routes INCLUDING improving the best existing component — the route's 2018 wall stands
(arc-3004 escalation reinforced). The portfolio route remains exactly one regime-orthogonal +2018&+2015
component short of deployable, and no construction (separate-component OR component-improvement) has
produced it.

**FLAGS (code not merged):** none. No new BUILT tool (the downtrend features are one-off diagnostic, not a
reusable signal). Driver scratch `_disco2000_work/arc2014_observe_downtrend_gate.py` (reproducible from the
arc doc).
### arc_3015

**End-of-week (Friday) position-squaring reversion — a weekly analog of month-end?** (chat 3000s).
Full record: [`arcs/arc_3015_end_of_week_reversion.md`](arcs/arc_3015_end_of_week_reversion.md). No
council (cheap-kill at observation; no reachable ceiling — coin-flip capture).

**Idea + why.** The 2018-leg hunt (now 8 prior dead routes) has been almost entirely **price-direction**
bets, all failing the same way (2018 USD strength = mean-reverting bursts → fades bleed, vol/structure
shorts short into the bounce, arc-3012 diagnosis). The ONE construction that is **+2018** is `me`
(month-end reversion, +0.90 in the 2018 fold), and it works **because it is a MECHANICAL calendar
flow** (WMR/index rebalancing reverts the month's over-extension regardless of trend) — the
"non-price-direction construction" arc-3012 thread #4 named as the surviving frontier. Open question:
is `me` a SINGULAR calendar anchor or one instance of a generic "calendar-boundary squaring" effect?
Tested the natural untested sibling — the **end-of-week (Friday) position-squaring** boundary (soft
de-risking into reduced Friday/weekend liquidity should mechanically revert intra-week over-extensions,
direction-agnostically, ~4× more frequent than month-end, distinct from both `me` (monthly) and the
weekend gap-fill 1006 (Monday-OPEN gap discontinuity, not the Friday-CLOSE move)).

**What happened — falsified at observation.** D1, 7 USD majors (`me`'s universe), BUILT
`observe_long_capture` + a `me`-style big-down-move conditioner (`into=(close[i]−close[i−2])/ATR≤−1.0`,
buy the sold side), grouped by **day-of-week**. (1) Generic big-down-move→buy is dead pooled (cap 0.489,
drift −0.03 — re-confirms 3000/3001). (2) **NO day-of-week concentrates the reversion** — Mon..Fri caps
0.46–0.50, Friday cap **0.502 / drift −0.045**; the whole hypothesis dies here. (3) The **month-end
subset of the SAME setup is cap 0.511 / drift +0.193 / med +0.247** — POSITIVE, reproducing `me`:
month-end concentrates reversion, the weekly boundary does not. (4) **2015/2018 acceptance FAILS** —
2018 cap 0.421/drift −0.346 (robustly neg), 2015 cap 0.417/med −0.51; excluding month-end Fridays
(orthogonal to `me`) is the same coin-flip (cap 0.496/−0.064). (5) Per-pair Friday is a **USD-quote-beta
split** (EUR/AUD/NZD/JPY-USD mildly +, USDCAD/USDCHF −; 4/7 = noise, the arc-2009/3012 tell). (6)
Best-version sweep (thr {1.0,1.5,2.0} × drift {2,5}) thins below the pool floor before turning positive —
the only "positive" cells (drift5·thr1.5) rest on **2 samples in 2015 (cap 0.0) + 5 in 2018** = thin-tail
noise (arc-2011/3012 mean≫median tell); thr2.0 has 2015 n=1, 2018 n=0.

**The crux (why it died).** `me` survives 2018 because month-end rebalancing is a **hard, quantified,
inelastic, date-certain** flow (index/WMR-fix rebalancing forced by mandate). The **weekly boundary has
no equivalent forced-rebalancing driver** — Friday squaring is soft/discretionary, small relative to the
week's noise, and swamped by the same directional dynamics as any other day. So the calendar-flow
reversion edge does NOT generalize down the calendar hierarchy: **month-end ≠ week-end.** §5f does not
bite (coin-flip capture, negative/median-≈0 drift, target-fold-negative → collapses SL-honest, arcs
3003/3012). No engine/null/council spent (arc-3010/3012/3014 efficiency discipline).

**Verdict: KILL (cheap-kill at observation).**

**Threads / lessons.** (1) **The mechanical-flow reversion edge is MONTH-END-SPECIFIC, not a generic
calendar-boundary effect** — `me` has no weekly twin (Friday cap 0.50/neg drift vs month-end +0.193). The
load-bearing ingredient is the **hard inelastic rebalancing mandate**, not the calendar boundary per se;
the mechanical-flow well is now **dry at the weekly scale**, narrowing the arc-3012-thread-#4
non-price-direction frontier. (2) **9th dead route to the 2018 leg** (structure 1014/2009/2011/3011,
trend 3010, flow-short 1016, continuation-long 2012, vol-state 3012, weekly-convexity 3014,
session-levels 3013, now end-of-week reversion 3015). (3) Generic D1 reversion re-confirmed dead
independent of the day conditioner. (4) **Surviving frontier (narrowed):** the in-apparatus calendar-flow
reversion leads are exhausted at BOTH monthly (taken: `me`) and weekly (dead: 3015) scales; what remains
for the 2018 leg is operator-gated (tighter-cost execution regime, arc-3004 escalation #3) or a new
non-price, non-calendar-boundary construction not yet conceived. The portfolio route stands exactly one
regime-orthogonal component from deployable, and that component increasingly looks like it must come from
**outside the H4/D1-FX-major in-apparatus search space**.

**Tooling.** No new BUILT tool — reused BUILT direction-aware `observe_long_capture`; day-of-week /
into-move / month-end are one-off scratch conditioners. Drivers
`_disco3_work/arc3015_observe_dow_reversion.py` + `arc3015_friday_bestversion.py` (reproducible).

**FLAGS (code not merged):** none. No canonical-core change. Carries standing FLAG-1 + the
`A1Config.time_exit_bars`-unwired flag.

### arc_1019

**Month-End Reversion SHORT** (chat 1000–1999). Full record:
[`arcs/arc_1019_month_end_reversion_short.md`](arcs/arc_1019_month_end_reversion_short.md); PORTFOLIO
candidate dossier [`portfolio-candidates/arc_1019_month_end_reversion_short/`](portfolio-candidates/arc_1019_month_end_reversion_short/).

**Idea + because.** The route is blocked at 2018 (and 2015) — the 3-way book (gap 1006 + me-long 1011 +
fbr 1013) is provably un-passable (0/5151 convex weightings; 2018 positive only in the weak me-long, 2015
only in fbr, mutually exclusive). ~12 routes to a 2018-positive leg died. The ONE survivor of the 2018 hunt
is `me` (month-end mechanical/inelastic-rebalancing reversion), 2018-positive via the WMR/index rebalancing
mandate. **The gap nobody had tested: `me` (arc 1011) tested only the long/DOWN side and is 2015-negative;
the SHORT side of the same flow was never run.** Because the WMR-fix rebalancing reverts BOTH directions, the
short (fade a big UP move into month-end) should be positive precisely in strong-USD years — there a big up
move into month-end on a USD major is a counter-trend bounce that mechanical reversion + trend-resumption
pushes back down. NOT a structural coin-flip short (which died by symmetry, 1014/2009/2011): the flow is real
and directional-agnostic, and `me` already proved its timing load-bearing.

**What happened.** Built + registered `MonthEndReversionShortSignal` (direction-mirror of the 1011 long).
Observation (D1, 7 USD majors, IS): month-end up-move ≥+1 ATR → short drift +0.075 (median +0.096 → NOT a
thin-tail artifact, the tell that killed 2011/3012/3014); **month-end excess +0.089 ATR** over the random-day
control (timing load-bearing); honest short **capture 0.5508 — the FIRST corpus short to clear 0.50** (every
prior short sub-0.50: 1014 .489, 2011/3011 .473, 1016 .448). **2018 capture 0.818.** Both binding folds
gross-positive (2015 +0.464, 2018 +0.365, both frac .727). Honest engine §5f (exit menu swept): mean-positive
net of FundedNext costs under EVERY exit (sl_only-2bar +0.043% → sl_partial_close_1r_runner_trail **+0.683%,
7/10, worst −0.91%**); **beats the fair same-exit null by +0.80pp** (5/10 vs 2/10). NOT all-folds-positive →
**PORTFOLIO**.

**Robustness / the honest split.** **2018 is the robust leg** — positive under every exit (except tp_2r),
every threshold 0.75–1.5 (+0.68 to +1.72), and every leave-one-pair-out drop (+0.59 to +1.99). **2015 is
fragile** — positive only at threshold ≤1.0 and collapses when GBPUSD is dropped (drop-GBPUSD: mean −0.016%,
2015 −0.16). For the portfolio this is acceptable: 2015 is already strongly handled by fbr (+3.17), so this
component's unique value is the **robust 2018 contributor that only the weak me-long previously provided.**

**Decorrelation.** corr +0.157 vs me-long (near-zero); cannot co-fire (up vs down move); opposite pairs
(short = EUR/GBP/JPY/CAD-driven, long = AUD/NZD-driven); **regime-complementary on 2015** (short +0.40 where
me-long −1.14), both 2018-positive. This is the regime-orthogonality the 3-way book lacked.

**Threads / lessons.** (1) **The proven long mechanical-flow edge `me` has a real, mechanism-controlled,
null-beating SHORT mirror** — found by applying the winning calendar-flow template to the now-open short side,
not by predicting the regime (the 10 failed routes' approach). (2) **A mechanical/inelastic flow is the only
construction that has produced a robustly 2018-positive leg** — both directions of month-end rebalancing
work; the failed 2018 routes were all directional/structural/trend bets (coin-flip in a strong trend),
whereas the flow is regime-agnostic. (3) **The directional asymmetry of `me` (long 2015-neg, short
2015/2018-pos) is a quote-convention + trend-interaction effect:** in strong-USD years the long's "big down
move into month-end" IS the trend (continues), while the short's "big up move" is the counter-trend bounce
(reverts) — so the two sides are naturally regime-complementary, not redundant. (4) **The 4-way combination
is now the gated next arc (1020):** four net-positive components, and for the first time 2018 has a ROBUST
contributor (not just the weak me-long +0.90). Whether the convex search now clears all-folds-positive is the
open question.

**Tooling.** Built + registered `discovery/tools/month_end_signals.py :: MonthEndReversionShortSignal`
(EXPERIMENT tool — mask + ATR geometry + `Direction.SHORT` only; scoring canonical). Reused BUILT
`make_time_exit_predicate`, `build_null_signal_evaluation` (direction-aware), `observe_long_capture`
(direction="short"). Drivers `_disco_work/arc1019_observe_monthend_short.py`, `arc1019_wfo.py`,
`arc1019_robust.py` (reproducible from the arc doc).

**FLAGS (code not merged):** none requiring the canonical core. Carries the standing
`A1Config.time_exit_bars`-unwired flag (arcs 1005/1011; worked around via `make_time_exit_predicate`).
OOS untouched.
### arc_3016

**Intraday session structure: Asian-range / London-open break (continue OR fade)** (chat 3000s). Full
record: [`arcs/arc_3016_session_break_fade.md`](arcs/arc_3016_session_break_fade.md). No council
(cheap-kill at observation; honest capture far below 0.50 both directions).

**Idea + why.** With the flow/short/portfolio space saturated by the 1000s/2000s chats (now 10 dead
routes to the 2018 leg, all one USD/carry/risk-reversion-fade exposure — arc 1018's lesson: edge & tail
are the same exposure, can't diversify) and my own 3015 closing the weekly-calendar route, I targeted the
ONE genuinely-unmapped data modality for collision-freedom + information: **intraday session structure.**
Every prior arc is H4/D1/W1 daily-scale or calendar-flow; arc 1000's dry hour-of-day result was capture
CONDITIONING, never a session-RANGE construction. The classic, documented London-open break of the Asian
range — either a liquidity-ignition CONTINUATION or a stop-run FADE — is a genuinely different
(intraday-liquidity) exposure, decorrelated from every daily-fade edge, so a positive would be
portfolio-relevant regardless of the 2018 wall. Stated the H1 cost reality up front (1R=2·ATR≈24pip,
FundedNext RT≈3pip≈**~0.125R ≈ 2× the H4 cost-in-R**, arc-3008) → the edge must clear ~2×, not be marginal.

**What happened — falsified at observation.** DST-clean (Europe/London) Asian range (London-hours 0–7)
× first London-open break (hours 8–11), H1 EURUSD/GBPUSD/USDJPY/AUDUSD, honest +1R-before-SL capture both
directions. (1) **No continuation:** long-cont cap 0.375/drift −0.116, short-cont 0.396/−0.039 — the break
reverses, no momentum-ignition. (2) **No clean fade either — and a measurement trap caught:** the naive
`1−cont_cap ≈ 0.62` is NOT the fade's capture (the SLs are asymmetric). Measured as its own direction, the
fade's honest capture is **0.386** (short the up-break) / 0.383 (long the down-break). cont 0.375 + fade
0.386 = **0.76 < 1** → the break entry sits at a **high-volatility London-open WHIPSAW point where a 2·ATR
SL is hit BOTH ways** before +1R. Net fade drift after the entry spread (~0.08 ATR on H1) is **~+0.02R ≈ 0**,
~10× below the ~0.125R cost. (3) **No robustness:** only EURUSD short-up-break fade is positive (+0.19,
1/4 pairs = single-pair noise tell, arcs 1010/2011/3012); the long-down-break fade is 4/4 negative. The
fade IS drift-positive in 2015 (+0.174) and 2018 (+0.251) — but inside capture 0.385/0.433 (<0.50): the
gross drift does NOT survive the take-the-loss 2·ATR SL.

**The crux (why it died).** Entering immediately after a London-open break is entering ON a volatility
spike: the next bars extend (stopping the fade) or snap back (stopping the continuation) with near-equal
frequency, so ANY directional bet with a 2·ATR SL there loses to the stop (cap ~0.38 both ways). The
residual fade drift (~0.02R) is the ceiling on any exit and is ~10× below cost → no exit/SL sweep rescues
it, §5f does not bite (no non-coin-flip entry). This extends the corpus's **H1 cost wall** (gotobi 1008,
month-end-fix 3008, round-number-H1 1010) from discrete-flow events to **session-RANGE structure**.

**Verdict: KILL (cheap-kill at observation).** No pool/engine/null/council spent.

**Threads / lessons.** (1) **Intraday session structure is sub-cost — the last unmapped data modality is
now mapped dead.** The London-open break is a whipsaw entry (honest cap ~0.38 BOTH directions = a 2·ATR SL
hit both ways), net drift ≈0 after spread; the H1 cost wall extends to session-RANGE constructions. (2)
**`1−continuation_capture` ≠ the fade's capture** (asymmetric SL) — a reusable measurement caution; always
measure the contrarian leg in its own direction (the naive flip suggested a 0.62 "fade edge" the honest
0.38 refuted). (3) **A gross-drift cell inside sub-0.50 honest capture is dead even when 2015/2018-positive**
(re-confirms 3004/3012; the 2·ATR take-the-loss SL is the arbiter) → this fade's +2015/+2018 drift is NOT
a 2018-leg lead. (4) **Surviving frontier (further narrowed):** with intraday session structure dead, the
in-apparatus (H1/H4/D1/W1 FX-major OHLCV) search is comprehensively exhausted — direction, flow, calendar
(monthly taken / weekly+intraday dead), structure, vol, regime, relative-value, session structure. The
portfolio route stands one regime-orthogonal 2018-leg component from deployable, and that component must
come from a lever the in-apparatus search cannot supply (operator-gated tighter-cost execution regime —
arc-3004 escalation #3 — or a non-OHLCV data source). Honest read: the H4/D1-FX-major idea well is nearly
dry; remaining arcs are low-EV map-completion until the operator opens a new lever.

**Tooling.** No new BUILT tool — scratch observers `_disco3_work/arc3016_observe_session_break.py` +
`arc3016_observe_fade.py` (canonical `Panel.from_pairs` H1 + `reached_1r_before_sl`; reproducible).

**FLAGS (code not merged):** none. No canonical-core change. Carries standing FLAG-1 + the
`A1Config.time_exit_bars`-unwired flag.

### arc_3017

**Month-End Reversion SHORT — INDEPENDENT REPRODUCTION of arc 1019 (Arc-10 defense).** Full record:
[`arcs/arc_3017_month_end_reversion_short.md`](arcs/arc_3017_month_end_reversion_short.md). I formed this idea
independently (the unharvested SHORT side of `me`/1011, for the 2015 & 2018-positive 4th leg) and ran it
concurrently with chat 1000s' arc 1019 — genuine convergence, the protocol's cross-check working as designed.

**Idea + because.** The 3-way book (1006+1011+1013) is provably blocked by {2015, 2018} — 2015 positive ONLY
via fbr, 2018 ONLY via me — so the 4th leg must be positive in BOTH. `me` (1011) is the ONE demonstrably
2018-positive mechanism, but only its LONG side was harvested. Month-end WMR/index rebalancing is INELASTIC and
**direction-symmetric**; the LONG side bleeds 2014/15/16 (EURUSD down-moves continue), so the unharvested SHORT
side (sell big UP moves into month-end → revert DOWN) should fire on USDXXX over-extensions in those years.

**What happened — signal CONFIRMED, then a 100× engine discrepancy.** Observation reproduced 1019 exactly: the
FIRST short in the corpus to clear **>0.50 capture (0.5508)** with a **PASSING structure control** (month-end
excess +0.0996 ATR vs 1019's +0.089; generic big-UP continues up −0.041), median drift positive, obs 2015
drift +0.437 / 2018 capture 0.818. Every prior short died on SIGN/capture (<0.50) or a failed control; this one
gets the sign right (the month-end mechanism is genuinely direction-symmetric — strengthens 1011). **But my
first engine WFO read ROI ~100× SMALLER than 1019 and the whole corpus, and I (wrongly) concluded KILL.**

**The error (RETRACTED) + the Arc-10 lesson it forced.** Diagnosis: I passed `A1Config.risk_pct=0.005` — the
value `ArcPoolConfig` takes as a FRACTION (0.005 = 0.5%) — but **`A1Config.risk_pct` is in PERCENT (0.5 =
0.5%)**; its default 0.005 means 0.005%. So every per-fold ROI was 100×-compressed. I then compounded it by
judging the compressed real-vs-null margin (+0.012pp) against the corpus's UNCOMPRESSED benchmarks (1006
+0.36pp) and calling it "noise floor." **The scale-INVARIANT judgments agreed with 1019 the whole time:** at my
(linear, low-risk) scale the partial-runner was 7/10 with **2015 +0.004% and 2018 +0.009% (both positive) and
real > null** — the same fold-sign pattern 1019 reports (7/10, 2015+, 2018+, beats null +0.80pp). Confirming
the 1011-long through my harness (+0.0023% vs 1011's reported +0.23%, exact 100× with matching 7/10 sign
pattern) nailed it as a harness convention, not the signal.

**A second, deeper FLAG.** The per-fold ROI is not merely scaled but RESHAPED by risk_pct through the daily-DD
cap: at risk 0.5 the partial-runner fold-SIGNS flip (3/10; 2015 −1.07%, 2018 −1.08% — the high-vol strong-USD
folds blow through the 5% daily cap and truncate), the OPPOSITE of the low-risk 7/10/2015+/2018+. So the
all-folds-positive / 2015-2018-sign verdict for this thin (n≈116, ~10/yr), USD-concurrency-clustered short is
**risk_pct-convention-DEPENDENT.** The signal edge is robustly real; the engine *disposition* is only as firm
as the pinned risk convention.

**Verdict: PORTFOLIO — independent reproduction CONFIRMS arc 1019; my KILL is RETRACTED.** The component lives
in 1019's `portfolio-candidates/arc_1019_month_end_reversion_short/`; this arc does not duplicate it. OOS
preserved (§4).

**FLAGS (code human-gated, NOT patched — §9).** (1) **`A1Config.risk_pct` PERCENT (0.5) vs `ArcPoolConfig`
FRACTION (0.005); A1Config default 0.005 = 0.005% silently 100×-compresses ROI; the registry standard-entry
example omits `risk_pct` from A1Config so copies inherit the trap.** Recommend pinning the canonical discovery
`A1Config.risk_pct` in TOOL_REGISTRY's entry point (or unifying the two configs' units). (2) **The daily-DD cap
makes per-fold ROI / all-folds-positive / 2015-2018-sign nonlinear in risk_pct** for thin concurrency-clustered
books — the gated 4-way combination (arc 1020) MUST report risk-sensitivity, not a single-risk number, or it
repeats the Arc-10 single-config trust.

**Threads / lessons.** (1) **Independent reproduction works exactly as intended** — two chats, same idea, a
100× engine gap → traced to a measurement convention, not the signal; the signal is confirmed and a real
apparatus trap is surfaced. (2) **Scale-invariant metrics (fold-sign pattern, real-vs-null in the linear
regime) are the trustworthy cross-config judges; absolute ROI magnitude is config-fragile** — never judge a
verdict on an absolute pp-margin compared across configs. (3) The month-end short is the corpus's first
2018-positive PORTFOLIO short, but flag-don't-celebrate: its engine disposition rests on the risk convention.
Built/confirmed `discovery/tools/month_end_signals.py :: MonthEndReversionShortSignal` (registry credits arc
1019 + 3017). Drivers scratch `_disco_work/arc3017_*.py`.
### arc_1020

**4-way PORTFOLIO combination WFO** (chat 1000–1999). Full record:
[`arcs/arc_1020_portfolio_combination_4way.md`](arcs/arc_1020_portfolio_combination_4way.md).

**Idea + because.** Arc 1015 (this chat) proved the 3-way book (gap + me-long + fbr) is blocked by 2015 &
2018 (0/5151 convex weightings; 2018 positive only in weak me-long, 2015 only in fbr — mutually exclusive),
and specced the 4th leg: positive in BOTH 2015 & 2018. Arc 1019 (this chat) found it — the month-end
reversion SHORT, robustly 2018-positive (+0.86) with a fragile 2015 tilt (+0.40). This arc runs the gated
4-way combination: does the new leg clear an all-folds-positive book?

**What happened.** Reproduced all 4 components live (me-long/fbr/me-short byte-exact; gap ~1-2pp per-fold off
the recorded — a config nit, FLAGGED; verdict re-confirmed under the audited recorded gap vector). Combined
via `combine_fold_roi` + a full 4-simplex convex grid search. **The 2018 wall is BREACHED in combination:**
2018 was the unsolvable 3-way binding fold (12+ standalone routes died on it); me-short gives it a 2nd robust
contributor, and at the book optimum 2018 is now POSITIVE (+0.08). Worst fold improved 7× (−0.77% → −0.115%)
— the strongest book the corpus has produced. But **STILL 0/all convex weightings all-folds-positive → KILL**
(robust under both gap vectors AND every me-short exit from its 1019 menu). The block moved to **2015 (−0.08)
& 2016 (−0.12), both marginal.**

**Why the constraint moved (the key structural insight).** fbr is wanted by 2015/2016/2020 but rejected by
2018 (deeply negative there). In the 3-way, 2018 had only the weak me-long → unsolvable. me-short is a
*non-fbr* 2018 contributor, so it relieved the 2018 constraint — which let the search lift 2018 positive and
exposed the NEXT-marginal fold. **2016 is the new knot:** both month-end legs (long −0.51 AND short −0.91)
are negative there under every exit, so 2016 relies on gap (+3.20) & fbr (+2.55) — but weighting those up
breaks 2018. **2015** is dragged by me-long (−1.14, weighted heavy for 2018) plus me-short's fragile 2015
leg (+0.40, the GBPUSD/low-threshold fragility flagged in 1019). So the route advances fold-by-fold: each
regime-orthogonal leg neutralizes one wall and exposes the next.

**Verdict: KILL the combined 4-way book** (not all-folds-positive). Components UNCHANGED (retain PORTFOLIO;
no new portfolio-candidate entry — would quadruple-count). OOS preserved.

**Threads / lessons.** (1) **A mechanical-flow SHORT breached the 2018 wall the entire directional/structural
menu could not** — the portfolio gate rewards regime-complementarity, not solo strength (a 2018-positive,
not-all-folds-positive component was exactly what was needed). (2) **The residual block is the same fbr-vs-2018
tension shifted one fold over (to 2016).** (3) **5th-component spec: positive in 2015 AND 2016 without
dragging 2018, and NON-reversion** — the gap/month-end reversion family is 2016-saturated (all three are
2016-negative or rely on the same risk-off tail). 2016 = Brexit/US-election; 2015 = CHF-depeg/USD-bull. (4)
**me-short's 2015 fragility is now load-bearing at the book level** — a non-fishing IS-fold improvement of its
2015 robustness (a real refinement, NOT exit-selection-on-the-gate) is a concrete sub-thread that could lift
2015 above zero. (5) The route is **~0.11% and one non-reversion, 2015&2016-positive component from a
deployable book** — far better than arc 1015's "blocked at 2018 by −0.77%."

**Tooling.** No new BUILT tool — all CALLED (`combine_fold_roi`, the four registered signals,
`make_time_exit_predicate`; scoring canonical via `ArcFoldRunner` → `MultiPairBacktester`). Driver
`_disco_work/arc1020_combo4.py` (reproducible from the arc doc).

**FLAGS (code not merged):** none requiring the canonical core. gap reproduction nit + linear-combination
limitation FLAGGED in the arc doc (both immaterial to the verdict). Carries the standing
`A1Config.time_exit_bars`-unwired flag.
### arc_2015

**Month-end reversion SHORT — independent reproduction of arc 1019 + the 4-way combination** (chat 2000s).
Full record: [`arcs/arc_2015_month_end_reversion_short.md`](arcs/arc_2015_month_end_reversion_short.md).
Component record (co-discovered): [`portfolio-candidates/arc_1019_month_end_reversion_short/`](portfolio-candidates/arc_1019_month_end_reversion_short/).

**Idea + why.** After arc 2014 (KILL) closed the "improve 1013 toward solo-PASS" lane, I re-read the log:
arc 3015's diagnosis flagged that month-end (1011) is +2018 via the *inelastic WMR/index-rebalancing
mandate* — a flow that reverts the month's move REGARDLESS of sign. 1011/1012/3008 only tested the LONG
side (shorts were disabled then; PR #273 merged today). The SHORT side (sell a big UP-move into month-end)
fires exactly in strong-USD months (2015/2018) where the long-only book under-covers → a clean +2015&+2018
thesis using a PROVEN mechanism + the new short capability. **Chat 1000s formed the identical idea
concurrently (arc 1019)** — independent co-discovery; the convergence is itself the Arc-10 reproduction
defense (like 2008/3009 or 2011/3011).

**What happened — the component (confirms 1019).** Observation (direction-aware): short cap 0.551, and
POSITIVE in both binding folds (2015 drift +0.437; 2018 cap 0.818 / drift +0.293) where the long side is
negative. Honest §5f: best exit `sl_partial_close_1r_runner_trail` (the reversion OVERSHOOTS in strong-USD
→ runner exit, not 1011's 2-bar; sl_only NEGATIVE) mean **+0.683%**, beats fair SHORT null −0.234% by
**+0.92pp** (1019 got +0.80pp — matching). Per-fold 2015 +0.40 / 2018 +0.86. **Robustness (the decisive
Arc-10/arc-2013 check):** 2018 is ROBUST (survives every LOO incl −GBPUSD +1.14, every threshold;
EURUSD/AUDUSD-broad) → the corpus's first SCALABLE +2018 leg. **2015 is THIN REGIME-LUCK** — the +0.40 fold
rests on a single GBPUSD trade that ran +8.3R; everything else in 2015 is −1R/negative; flips negative
dropping GBPUSD or raising threshold to 1.25. Overall mean leans on GBPUSD (−GBPUSD → −0.016%). So the
component's value is the robust +2018, NOT the GBPUSD-leaning headline or the luck-2015. → PORTFOLIO (via
1019; I did not duplicate the component record).

**What happened — the 4-WAY COMBINATION (this arc's additive result; 1019 deferred it to arc 1020).**
Reproduced gap/me_long/fbr at their EXACT committed configs and VERIFIED the headlines (+0.685 / +0.232 /
+1.854, matching 2015/2018 signs) before trusting any combination (Arc-10). Per-fold ROI correlation:
me_short vs gap +0.188, me_long +0.157, fbr +0.406. Convex search (step 0.05, 1771 weightings, mirrors arc
2008): **0/1771 all-folds-positive (still blocked)** — BUT me_short is the **FIRST new component to MOVE the
convex-search wall**: best achievable worst-fold **−0.222% (3-way, arc 2008) → −0.124% (4-way)**, at
w={gap:0, me_long:0.65, fbr:0.2, me_short:0.15}, with both blockers now marginal (2015 −0.05, 2018 −0.12).
me_short's robust +2018 nearly closed the 2018 wall (−0.77 → −0.12). The book is the CLOSEST the corpus has
come to deployable, but still blocked — now by **2015**, since me_short's 2015 is luck and only fbr robustly
carries 2015.

**Verdict.** 4-way combined book = **KILL** (0/1771 all-folds-positive, like 2006/2008/1015). me_short
component = **PORTFOLIO** (confirmed, via 1019). Components UNCHANGED.

**Threads / lessons.**
1. **The 2018 wall is now breached robustly; the 2015 wall is the new binding constraint.** The route's
   precise need flips: it was "any +2015&+2018 leg" (18 dead arcs); it is now "a **robust +2015** leg" —
   the MIRROR of me_short's robust +2018. 2015's strong-USD was ACUTE-event-driven (SNB unpeg Jan-2015, EUR
   collapse, China deval Aug-2015) vs 2018's grind → the +2015 leg is likely a different mechanism (event /
   vol-shock reversion), not month-end. **This is the sharpest, most-actionable open spec in the corpus.**
2. **The short side of a proven LONG flow-mechanism is a productive, previously-impossible lane.** me_short
   (mirror of 1011) WORKS; the gap-fill short mirror (1016/2013) did NOT — the lane is mechanism-specific:
   it works where the forced flow reverts regardless of sign (month-end), not where the long edge was
   direction-specific (gap-fill's JPY-carry up-drift). A heuristic for future short arcs.
3. **me_short's portfolio value is +2018 only** — honest weighting must not lean on its GBPUSD-mean or
   luck-2015. The 4-way convex result already reflects this (the best weighting puts me_short at 0.15,
   using it as a 2018-patch, with me_long 0.65 / fbr 0.2 carrying the rest).

**FLAGS (code not merged):** none requiring the canonical core. The me_short signal tool was built +
registered by arc 1019 (`MonthEndReversionShortSignal`); I removed my duplicate registry row + duplicate
component dir on merge (kept 1019's canonical record). Drivers scratch `_disco2000_work/arc2015_*.py`
(reproducible; the 4-way reproduces the 3 committed component headlines EXACTLY before combining).

### arc_2016

**DIAGNOSTIC: is the 4-way book's residual block REAL or measurement-floor NOISE?** (chat 2000s). Full
record: [`arcs/arc_2016_portfolio_2015_noise_floor.md`](arcs/arc_2016_portfolio_2015_noise_floor.md).

**Why this arc exists (the council redirected it).** I resumed at the arc-2015 thread-1 spec — "the
sharpest open spec is a robust +2015 leg" — intending a 19th leg-hunt. First obs (vol-shock fade, the
acute-2015-event candidate) died cheaply: fading an extreme 1-bar move (≥1.5–2.5 ATR) on D1 USD majors has
capture 0.20–0.35 and NEGATIVE drift — the extreme move CONTINUES (closed-ground momentum, not reversion);
2015 not positive. I then convened `/llm-council-discovery` at the idea-fork. **Peer review was unusually
unanimous: all 5 reviewers ranked the Soundness lens strongest and all 5 named the Alternative-framing
(loosen the gate) lens the biggest blind-spot.** The chairman's directive: do NOT open a 19th leg-hunt;
first compute three numbers — (1) is worst-fold −0.124% distinguishable from zero at ~10–15 trades/fold?
(2) the 2015↔2018 correlation CI at n≈10; (3) per-component 2015 decomposition — and *"if inside noise OR
single-component-driven, skip the 5th leg entirely."* Strongest dissent: there is **no out-of-sample 2015**
(a leg engineered to clear a fixed year is in-sample by construction; the real test is forward survival,
unearned while the book fails IS AFP). I committed (heavy weight) and converted the arc to that diagnostic,
IS-only (did not spend the combined-book OOS — §5g: book fails IS AFP).

**What the diagnostic found (`arc2016_diagnostic.py`, canonical apparatus + honest per-trade P&L from
`StrategyResult.closed_trades`).** Reproduced all 4 components EXACTLY (gap +0.685 / me_long +0.232 / fbr
+1.854 / me_short +0.683) at IS-best convex w={gap 0, me_long .65, fbr .2, me_short .15} (worst −0.124%,
mean +0.624%, book per-fold std 0.703%). The two negative folds are 2015 (−0.047%) and 2018 (−0.124%).
- **(1) 2015 is single-component-driven, not co-drawdown:** me_long −0.742% contribution (w.65 × −1.141%,
  n=10) nearly cancels fbr +0.635% (w.2 × +3.174%); gap at weight 0. A me_long-vs-fbr near-cancellation.
- **(2) NOISE FLOOR (decisive):** bootstrap (10k, seed 42) of each component's 2015 honest per-trade P&L,
  combined at frozen weights → **book 2015 ROI −0.047% ± sd 0.71%, 95% CI [−1.22%, +1.55%], P(neg)=0.42**.
  **|worst-fold|/sd = 0.067σ** (2018 = 0.176σ). Per-component 2015 sampling sd is enormous (gap ±2.64%,
  fbr ±2.62%, me_short ±1.27%, me_long ±0.68%) — these are 10–28-trade folds. 6/10 book folds sit within
  ±1σ of zero; only 2011/12/19/20 clear the floor.
- **(3) corr CI:** all 6 pairwise component fold-ROI correlations' Fisher-z 95% CIs SPAN ZERO at n=10
  (me_long·fbr −0.366 [−0.81,+0.34]; fbr·me_short +0.406 [−0.30,+0.83]; etc).

**Verdict + meaning. KILL** (no new component) — but a **programme-redirecting** result. The residual
negative folds are **statistically zero**; a 5th leg that flips a 0.07σ fold would be **fold-painting**
(selecting against the test statistic — the council's unanimous concern). The convex-search worst-fold
"improvements" across the ~18-arc leg-hunt (−0.222→−0.124) were largely **moving inside the noise floor**,
not closing a real gap (me_short's +2018 was real *as a component*, but the book-level wall it "breached"
was already sub-noise). **I did NOT loosen the gate** (the rejected lens); the book stays a strict-gate
FAIL, components UNCHANGED (PORTFOLIO).

**OPERATOR FLAG (governance — theirs, not mine).** The all-folds-positive calendar-year gate, applied to
books of thin decorrelated components, is evaluated **below its own noise floor** (±0.7% per-fold here).
Two honest forward paths, the operator's call: **(A)** reconsider fold/gate resolution for thin-component
books (pooled-trade or regime-block gating with explicit SE); **(B)** redirect discovery toward components
whose *per-fold* ROI clears the noise floor (more trades/fold and/or larger per-trade edge), not
ever-thinner legs whose yearly ROIs are sampling-error-dominated. The genuine forward test (4-way 2021+ OOS)
stays deferred until a book earns the OOS look.

**Threads / lessons.**
1. **NEW high-value lesson: the portfolio-book gate has a NOISE FLOOR set by component trade-counts.** A
   negative fold smaller than ~1 book-σ is statistically zero; "fix it with a 5th leg" chases noise and
   invites fold-painting. **Quantify a marginal fold's bootstrap CI BEFORE hunting a component to flip it.**
2. **The 2015↔2018 anti-correlation is an n=10 story, not a structural constraint** (all corr CIs span 0).
   Treat per-fold correlations at n≈10 as uninformative in future combination arcs.
3. The council process caught a sunk-cost treadmill (a 19th noise-painting arc) and dissolved the question
   with a measurement — generative council at a genuine fork earned its cost.

**FLAGS (code not merged):** none touching canonical core; no new BUILT tool (one-off arithmetic on
canonical outputs; `closed_trades` P&L is a read of `StrategyResult`). Drivers scratch
`_disco2000_work/arc2016_*.py` (reproduce the 4 headlines EXACTLY before any bootstrap).
### arc_3018

**Failed-breakdown-reclaim LONG on CROSSES** — a fresh, well-motivated 5th-component hypothesis aimed at
the live frontier (arc 1020: the 4-way book is ~0.11% from deployable, blocked only by marginal 2015/2016).
Full record: [`arcs/arc_3018_fbr_reclaim_crosses.md`](arcs/arc_3018_fbr_reclaim_crosses.md).

**Idea + because.** The only tradeable thing in 2015/2016 is washout-bounces (reversion off SNB/China-deval
2015, oil/Brexit 2016) — which is exactly why the USD-major `fbr` (1013) is +2015/+2016. But USD-major `fbr`
is −2018 (failed breakdown → real breakdown in the sustained USD trend; arc 2014 proved un-conditionable), so
the optimizer can't up-weight it to fix 2015/2016 without re-breaking 2018. Hypothesis: run the SAME reclaim
mechanism on a DIFFERENT universe (crosses) → decorrelated from USD-major `fbr`, plausibly +2015/+2016, and
crucially maybe 2018-NEUTRAL (crosses out of the USD trend) → satisfies the real 5th-leg constraint. Documented
spec-override (CC commits): arc 1020 asked for NON-reversion, but 2015/2016's only edge IS reversion; the
genuinely new lever is 2018-neutrality-via-universe, not non-reversion.

**What happened — FALSIFIED at observation, both universes.** Structure control INVERTS vs arc 1013: the
swept-low reclaim captures 0.4633 (JPY crosses) / 0.4248 (non-JPY), BOTH below 0.50 AND below a generic deep
down-wick (0.5463 / 0.4620) — the swing-low pierce/reclaim REMOVES edge on crosses (the opposite of USD majors,
where reclaim 0.55–0.61 ≫ generic-wick coin-flip). Target-year sign wrong: 2016 drift −0.95 (non-JPY); 2018
negative both; 2015 strongly negative (JPY −0.997) or a thin-tail artifact (non-JPY mean +1.25/median −0.295,
n=9). No robust per-pair edge.

**Diagnosis + lesson.** The `fbr` edge is **USD-major-SPECIFIC**. USD-major swing-lows sit at the densest
resting-stop clusters and are defended reversal pivots (1013 control-proven, 3013 level-confirmed); cross
swing-lows are swept routinely WITHOUT being supports (thinner liquidity, momentum/carry-driven), so the
reclaim enters mid-cascade — a falling knife, worst on carry JPY crosses in risk-off (2015/2018, arc-1017
mechanism). **This generalizes arc 1018 (cross gaps are efficient/random-walk) from FLOW to STRUCTURE: the
corpus's edges are universe-specific — a USD-major structural edge does NOT port to crosses.**

**Threads.** (1) The 2015/2016 leg is a genuine structural bind: its only edge IS reversion, and the reversion
that works (USD-major `fbr`) is the one that's −2018 → 2015/2016-lift and 2018-lift want OPPOSITE
universes/directions; a single static-weight book can't have both (re-confirms arc 2008's convex-search
impossibility from the mechanism side). (2) Remaining 5th-leg candidate spaces: a `me_short` sibling robustly
+2015 (the SNB/EUR-collapse 2015 is acute-event-driven — likely a different mechanism than month-end, per arc
2015's spec); or operator-gated levers (tighter-cost execution regime — arc-3004 escalation #3 — or non-OHLCV
data). The honest read (arc 3016 thread 4, reaffirmed): the in-apparatus FX-major OHLCV well is nearly dry for
this specific +2015/+2016/non-(−2018) bi-fold spec. (3) `fbr` 1013 UNCHANGED (still PORTFOLIO); the
standalone-via-cross-port route is closed.

**FLAGS (code not merged):** none requiring the canonical core. No new BUILT tool (reused 1013's
`FailedBreakdownReclaimLongSignal` + canonical `observe_long_capture`). Driver scratch
`_disco3_work/arc3018_observe_fbr_crosses.py` (reproducible). Carries standing flags (arc-3017 `risk_pct`
unit split; arc-1005 `time_exit_bars` unwired).
### arc_1021

**Broad-universe month-end-SHORT (§5f best-version) + 4-way re-combination** (chat 1000–1999). Full record:
[`arcs/arc_1021_month_end_short_broad_universe.md`](arcs/arc_1021_month_end_short_broad_universe.md).

**Idea + because.** The 4-way book (arc 1020 / independently 2000s arc 2015) is blocked at 2015 & 2016, and
me-short's 2015 (+0.40) is a single GBPUSD trade. arc 1019 never established the month-end-short flow's BEST
universe. Question: does broadening to 15 pairs (7 USD majors + 5 JPY crosses + EURGBP/EURAUD/GBPAUD) give a
robust multi-pair 2015 AND 2016 (the binding folds)? Same proven mechanism, broader universe = legitimate
best-version development (all on IS; OOS untouched). Obs supported it: each group beats its random-day control,
JPY crosses carry 2016 (+0.327) where USD majors don't, EUR/GBP crosses add an independent 2015 (+0.405).

**The near-miss (and the catch).** Under the partial-runner exit it LOOKED like the corpus's first deployable
book: broad me-short 9/10 / +1.998%, and the 4-way (gap+me-long+fbr+broad-me-short) had **25 convex weightings
all-folds-positive** (best worst-fold +0.069%, 0/10 neg). **I did NOT take it at face value (Arc-10 paranoia).**
The robustness check disqualified it as a two-layer in-sample-selection artifact: (1) the 9/10 rests on the
**partial-runner's fat right tail** — under honest **sl_only** broad me-short is only 4/6/7-of-10 with 2018
NEGATIVE under te2/te3 and median often <0; (2) even under partial-runner the all-folds-positive book exists
only at **IS-fold-optimized max-min weights** — risk-parity gives 8/10 (worst −0.193%), equal 7/10; and the
4-way is **0-AFP under every sl_only exit**. **2016 is robustly negative in me-short under every exit** (−1.18
to −1.96) — the broad universe does not fix it. → **KILL the book claim.** OOS NEVER touched (disqualified on
IS robustness — the holdout was not spent on a false positive).

**Verdict: KILL.** The broad-universe "all-folds-positive book" is partial-runner-fat-tail × IS-weight-overfit.
Components UNCHANGED; the narrow 4-way (arc 1020, worst −0.11%, blocked 2015/2016) remains the honest frontier.

**Threads / lessons.** (1) **Arc-10 near-miss caught by discipline** — a combined book all-folds-positive on
IS can rest entirely on in-sample selection (exit choice + weight optimization). **Re-usable rule: before
promoting any combined book, require all-folds-positive under a NON-runner exit (sl_only) AND under risk-parity
(not just the IS-optimal max-min weighting); razor-thin binding folds (+0.07%) demand maximal scrutiny.** This
would have been a fabricated "first deployable system" if the partial-runner+max-min number were trusted. (2)
**Genuine carry-forward:** the broad universe DOES robustify me-short's 2015 (multi-pair +0.92..+3.33 under
honest sl_only, no longer single-GBPUSD) — but trades it for exit-fragile 2018 and does not fix 2016, so it is
not a strict component improvement. (3) **2016 is now the precise hardest residual fold** — negative in every
reversion leg (gap, me-long, me-short narrow AND broad, every exit); positive only in fbr & gap, which 2018
caps. The 5th-component spec sharpens to: **positive in 2016 without dragging 2018, structurally NON-reversion**
(2016 = Brexit / US-election). (4) **A combination must freeze me-short on an honest non-runner exit** — the
runner harvests fat-tail variance that inflates IS.

**Tooling.** No new BUILT tool — all CALLED (`MonthEndReversionShortSignal`, `make_time_exit_predicate`,
`build_null_signal_evaluation`, `combine_fold_roi`). Drivers `_disco_work/arc1021_obs_universe.py`,
`arc1021_wfo_broad.py`, `arc1021_robust_check.py` (reproducible from the arc doc).

**FLAGS (code not merged):** none requiring the canonical core. Carries the standing
`A1Config.time_exit_bars`-unwired flag + the arc-1020 gap-reproduction nit (immaterial — gap 0-weight).

### arc_3019

**Forward-confirmed extreme-shock CONTINUATION** — the corpus's strongest continuation IS result and most
promising 5th-leg candidate ever, KILLED by a one-shot OOS. Full record:
[`arcs/arc_3019_shock_continuation.md`](arcs/arc_3019_shock_continuation.md). Council transcript summarized in
the arc doc (not separately saved; the verdict drove the two decisive tests).

**Idea + because.** 2015 (SNB/China-deval) and 2016 (oil/Brexit/Trump) were extreme-EVENT years where the
book's reversion legs bled because the big moves CONTINUED. The missing +2015/+2016 leg must be a
CONTINUATION. Generic momentum is closed ground (coin-flip) and the i+1-entry-after-a-big-move death
(backward-confirming → catches the bounce) killed every prior shock trade (1016/2009/3012). The novel fix:
an EXTREME ≥3-ATR shock (forced flow — liquidations/stop-cascades/CB) creates multi-day pressure; harvest it
only when FORWARD-CONFIRMED (the bar after the shock breaks its extreme), entering on the resumption (the
1013-reclaim property applied to continuation).

**IS — the strongest continuation result in the corpus.** Magnitude-MONOTONE capture 0.498@1ATR (coin-flip,
= why shallow momentum is closed ground) → 0.589@3ATR (the edge is ONLY in the extreme tail). Forward-confirm
load-bearing (0.589 > 0.543 unconf). Honest §5f, linear regime: EVERY exit mean-positive & beats the fair
null; tp_3r 9/10 +0.034 (beats null +0.048, 9 vs 3); 2015 AND 2016 positive under EVERY exit — the exact
4-way blockers; the exact mirror of me-short. This looked like THE route-unblocking find.

**Why I didn't record it on IS — council + two tests.** I convened the HEAVY council at the
PORTFOLIO-vs-KILL fork. All 5 reviewers ranked the Soundness lens strongest; the chairman verdict: do NOT
record on IS (the "+ exactly where the book needed it" is selection-suspect — 15 routes died vs the same
target; n thin; 3.0-ATR a post-hoc peak), run the cheap event-removal test then OOS first; strongest dissent
= epoch-dependence (post-2020 shocks may mean-revert). I committed (heavy weight; the reasoning beat my
PORTFOLIO lean). **Test 1 (event-removal):** refuted the naive "just SNB+Brexit+Trump" coincidence (both
years stay positive without the mega-event windows; the big days are varied macro shocks — FOMC/oil/NFP) but
confirmed 2015 is thin (flips negative dropping its top-2 of 22 days; 2016 robust). **Test 2 (one-shot frozen
OOS, tp_3r@3ATR, select-on-IS-freeze):** FAILS — 2/6 folds, mean −0.0022, **loses to the null** (−0.005pp).
2021/2022 positive (Fed-hiking/LDI shocks continued); 2023-2025 negative (post-shock mean-reverted).

**Diagnosis + lesson.** A real but EPOCH-SPECIFIC tail edge: the mechanism captured the 2010-2020 shock
regime (shocks trended for days, esp. 2016) but does NOT generalize to 2021+ (shocks mean-revert faster).
IS 9/10-beats-null → OOS 2/6-below-null. **NEW lesson: a continuation/tail edge can be epoch-dependent even
with a clean mechanism, a clean magnitude-monotone signature, AND a clean IS null-beat — so IS-mean-positive
+ beats-null is NECESSARY but NOT SUFFICIENT; an epoch-vulnerable edge must clear OOS before any PORTFOLIO
recording.** This is a higher-value negative than the prior continuation kills (those were coin-flip on IS;
this PASSED IS and died OOS). Independently, it is un-scalable: shocks cluster across pairs on macro-event
days → the FundedNext 5%-daily-DD cap breaches 0→7-84/yr from risk 0.005→0.5 (the daily-DD analog of
arc-1017's currency-cap un-scalability) — a second, independent disqualifier.

**Arc-10 SAVE.** Recording PORTFOLIO on the IS evidence (my initial lean) would have fabricated the corpus's
headline "route-unblocking 5th leg." The HEAVY council's refusal-to-record-on-IS + the one-shot OOS caught
it. The event-removal test (council's idea) is a cheap, reusable soundness instrument for event-concentrated
edges.

**Threads.** (1) The +2015/+2016/non-(−2018) 5th leg remains unfound; the strongest candidate yet died OOS.
(2) A future +2015/+2016 mechanism should be NON-epoch-vulnerable (calendar/flow-mechanical like me/me-short,
not regime/event-driven like this — mechanical flows recur across epochs; shock-directionality doesn't).
(3) Tail/shock strategies are structurally cap-hostile on FundedNext (cap-clustering) — a standing prior for
any future shock/event idea. (4) `ShockContinuationSignal` BUILT + registered, kept (real IS mechanism,
epoch-bound).

**FLAGS (code not merged):** none requiring the canonical core. New BUILT tool `ShockContinuationSignal`
(experiment tool, committed). Carries the arc-3017 `risk_pct` FLAG (judged in the linear regime + reported
risk-sensitivity, which is severe here and REINFORCES the daily-DD-cap-nonlinearity flag) and the arc-1005
`time_exit_bars`-unwired flag. Scratch drivers `_disco3_work/arc3019_*.py` (reproducible).
### arc_2017

**Per-component SOLO noise-floor assessment** (chat 2000s). Full record: [`arcs/arc_2017_solo_noise_floor.md`](arcs/arc_2017_solo_noise_floor.md). Direct follow-through on arc 2016 (same session): the 4-way book's residual negative folds are measurement noise — so, component by component, **which single edge is closest to all-folds-positive once per-fold sampling noise is accounted for?** A painted blend hides the components' own fold structure; a SOLO edge whose only negatives are within the noise floor is a cleaner, harder-to-fool candidate than a convex blend.

**Method (`arc2017_solo_noise.py`).** Reused arc 2016's machinery: each component at its committed config via the canonical apparatus, capture every IS fold's honest per-trade P&L (`StrategyResult.closed_trades[].pnl`), bootstrap (10k, seed 42) each fold's ROI 95% CI, classify POS(CI>0)/NEG(CI<0, a *real* negative)/~0(spans 0, within noise).

**Result.** gap mean +0.685% — **10/10 folds within noise** (even +8.23%/−6.79% span 0; n~28/yr, huge dispersion). me_long +0.232% — 9/10 within noise, only 2018 (+0.90%) resolves, a **real POSITIVE** (the WMR-rebalancing mechanism). fbr +1.854% — 9/10 within noise, but **2018 (−4.20%, n18) is the corpus's ONLY strictly-negative fold (CI<0)** = a real loss (arc 2014: 18/19 −1R, mechanism-intrinsic). me_short +0.683% — 2011/2012 real-POS, no real negatives, rest within noise. "AFP-up-to-noise" True for gap/me_long/me_short — but **VACUOUSLY** (can't reject anything), False only for fbr (the one edge whose folds resolve).

**Read + verdict. KILL** (diagnostic; components UNCHANGED). The strict AFP gate is in a structural bind for this corpus: a component is either **thin** (folds unresolvable → gate vacuous; "passes up to noise" is not a pass) or **thick enough to resolve** (only fbr) → its real holes show, and fbr's is a genuine mechanism-intrinsic −2018. The route's single real fold-level obstacle is **fbr-2018** — not a missing +2015/+2016 leg (within-noise for every component). In 2018 the components carry REAL opposing signals (me_long +0.90 real-POS vs fbr −4.20 real-NEG); the book's −0.124% is their near-cancellation = arc-2016's noise residual. arc 2014 already closed the only real obstacle (fbr-2018 entry-time-unconditionable).

**Convergence with concurrent chats.** This explains arc 1021's (1000s) Arc-10 near-miss — an AFP book disqualified on robustness: sub-noise folds let exit×weight optimization PAINT a pass, which is exactly 1021's empirical finding ("require AFP under sl_only AND risk-parity, not just partial-runner+optimized weights"). My arc gives the statistical foundation for WHY those near-misses recur. Re the arc-3017 risk_pct PERCENT-vs-FRACTION flag: my key results are scale-invariant (σ-ratios, correlation CIs, CI-width-vs-point) and my headlines reproduce the committed portfolio numbers exactly (same convention as all prior portfolio arcs), so the convention flag does not affect the conclusion.

**OPERATOR FLAG (reinforced, governance).** The AFP-calendar-year gate on thin-component FX books cannot return a confident PASS — vacuous on thin components, trips on the one thick component's mechanism-intrinsic single-year hole. Honest options remain the operator's: (A) a noise-aware gate (pooled-trade or CI-aware per-fold) asking "is any fold *resolvably* negative?" — under which fbr-solo's only failure is its real 2018, and gap/me_long/me_short are *unfalsified*; (B) seek components thick enough that folds resolve AND clear zero (closed ground makes high-trade-count directional edges coin-flips, so genuinely hard). 4-way 2021+ OOS forward test stays deferred (book fails IS AFP).

**Threads / lessons.** (1) NEW: a thin-component portfolio book faces a gate DILEMMA — thin ⇒ vacuous (folds unresolvable), thick ⇒ real holes surface; "AFP up to noise" is only meaningful where folds resolve — report per-fold CI resolvability, not just sign. (2) fbr's −2018 is the corpus's ONLY statistically-real negative fold; the ~18-arc 2015/2018-leg hunt chased fold-level signals that (bar fbr-2018 and me_long-2018) don't exist above noise. (3) gap's per-fold CIs are widest (±2.6%+) — pure low-correlation mean contributor, never a per-fold reliable edge (the convex search already drops it to 0).

**FLAGS (code not merged):** none touching canonical core; no new BUILT tool (bootstrap arithmetic on canonical per-trade P&L). Driver scratch `_disco2000_work/arc2017_solo_noise.py` (reproduces the 4 committed headlines exactly before any bootstrap).
### arc_1022

**Failed-breakdown RECLAIM long across UNIVERSES** (chat 1000–1999). Full record:
[`arcs/arc_1022_failed_breakdown_reclaim_universe.md`](arcs/arc_1022_failed_breakdown_reclaim_universe.md).

**Idea + because.** The 4-way book (arc 1020) is blocked at 2015 & 2016. Chasing a 2016-event-specific
patch (Brexit/election) is overfit (no Brexit in the 2021+ holdout); the honest need is a 5th leg with a
GENERALIZABLE mechanism whose fold profile relieves 2016/2018 structurally. Key fact (arc 1020): 2016 is
strongly +fbr (+2.55) & +gap (+3.20); the block is the CONVEX tension (weighting them up for 2016 breaks
2018, where fbr is wiped). So the ideal leg is +2018 AND +2016. fbr is +2016 −2018 on USD majors because
in strong-USD 2018 the failed breakdown becomes a real one (arc 2014: 18/19 −1R). **Hypothesis: that
2018-wipeout is USD-major-specific; on JPY / non-USD crosses (no persistent 2018 USD trend, risk-off
mean-reverts) the stop-run-reclaim could survive 2018 → a decorrelated fbr-class +2018 leg.** The
universe lever is proven productive (gap-fill ports majors→JPY crosses, 1006).

**What happened — FALSIFIED at obs; fbr is USD-major-specific.** Observed fbr capture/drift + the
arc-1013 load-bearing STRUCTURE CONTROL (deep wick AT swept swing-low vs ELSEWHERE) on three H4
universes, IS only. **MAJORS reproduce 1013** (cap 0.5865, structure excess +0.108, 7/7 pairs >0.50 —
apparatus + edge reproduce). **JPY-crosses: the swing-low is NOT load-bearing** (cap 0.481<0.50,
structure excess −0.001, 2018 cap 0.389 WORSE than majors). **non-USD-crosses: coin-flip, weak
structure** (cap 0.495, excess only +0.035; 2018 cap 0.515 is regime-luck inside an overall sub-0.50
universe with negative drift — the 3010/3012/2013 tell; 4/9 pairs >0.50 = noise, and carving them is
forbidden outcome-aware filtering). The crosses-avoid-2018 hypothesis is moot: crosses carry no fbr edge
in ANY year.

**Diagnosis.** The stop-run-reclaim needs **deep, coordinated resting-stop liquidity below a structural
swing low** — USD majors carry the densest stop book (most participation, clearest levels), so the
sweep-and-reclaim is a genuine reversal there (excess +0.108); crosses' thinner stop liquidity makes the
swing-low location non-load-bearing → coin-flip. This **extends arc 3013's lesson ("clustered stops ≠
grab without a structural pivot") to the UNIVERSE axis.**

**Verdict: KILL (cheap-kill at observation).** No +2018/+2016 fbr-class leg off USD majors. Components
UNCHANGED (1013 still PORTFOLIO). No pool/engine/null/council spent (coin-flip capture + non-load-bearing
structure on both cross universes; §5f doesn't bite on a coin-flip entry). OOS never touched.

**Threads / lessons.** (1) **fbr's edge is USD-major-SPECIFIC, tied to stop-liquidity depth, not a
universal stop-run** — load-bearing only on majors, coin-flip on crosses. (2) **NEW heuristic: FLOW edges
port across universes (gap-fill majors→JPY crosses, 1006); STRUCTURAL stop-liquidity edges do NOT** —
flow follows the instrument, stop-structure follows the most-participated instrument. Useful prior for
future universe-lever arcs. (3) **fbr-on-crosses route to the +2018 leg CLOSED** — adds to the dead-2018
list (structure 1014/2009/2011/3011, trend 3010, flow 1016/2013, vol 3012, rel-value 2010, cont-long
2012, carry 1017, re-leveling 3013, downtrend-gate 2014, now re-universing). The book's residual
2015/2016 block is not addressable by re-universing the existing best edge. (4) Surviving frontier
unchanged: a genuinely novel non-price-direction mechanism, or the operator-gated tighter-cost regime.

**Tooling.** No new BUILT tool — CALLED `observe_long_capture(direction="long")` + the BUILT
`FailedBreakdownReclaimLongSignal` (1013) fire logic (replicated inline as a restrict mask) + a
structure-control deep-wick-elsewhere conditioner (scratch). Driver `_disco_work/arc1022_obs_universe.py`.

**FLAGS (code not merged):** none. Carries the standing `A1Config.time_exit_bars`-unwired flag.
### arc_2018

**Cross-sectional month-end rebalancing reversion, USD-neutral** (chat 2000s). Full record:
[`arcs/arc_2018_xsec_monthend_rebalancing.md`](arcs/arc_2018_xsec_monthend_rebalancing.md).

**Idea + because.** Resuming behind arcs 2016/2017 (the two diagnostics that closed the thin-leg-hunt
route as noise-floor-capped + operator-gated). Arc 2017 named the ONLY surviving productive spec —
**option (B): a component THICK enough that its folds RESOLVE and clear zero.** Leg-hunting a 5th thin
fold-painting component is explicitly flagged as chasing noise, so I did NOT. Instead I attacked option
(B) with the one mechanism that is both *proven* and *extendable to thickness*: rank the 7 USD majors as
currencies-vs-USD by month-to-date return; long laggard / short leader AT month-end. **because:** `me`
(1011 long / 1019 short, the WMR-4pm-fix forced inelastic rebalancing) is a per-pair **absolute** move →
USD-beta-exposed (exactly why 2015/2018 are its binding folds). A **cross-sectional** rank is **USD-neutral
by construction** (the common USD move cancels across long+short legs) → should strip the regime exposure,
AND is **thicker** (fires on the extremes every month, both sides, ~50/yr vs `me`'s ~11/yr). Genuinely
untested: 2003/2010 rel-value were NOT month-end-timed; `me` 1011/1019 are per-pair absolute, not
cross-sectional.

**What happened — FALSIFIED at obs.** Drift-lens screen (gross, no engine/cost), IS 2010-2020, 7 majors
as 7 ccy vs USD (XXXUSD = +ret, USDXXX = −ret). At each month-end: rank by month-to-date vs-USD return;
market-neutral spread = fwd(bottom-k) − fwd(top-k); random-day control. Driver
`_disco2000_work/arc2018_xsec_monthend.py`.

Best cell FWD2 top1 (2-leg, the `me` horizon): ME spread mean +15.27bp / **median +9.82bp**, **frac+
0.546**, **ME-excess +14.60bp** over random-day (≈0). 4-leg (top2): net-NEGATIVE every horizon.

Three decisive reads:
1. **Thesis FAILS on the binding fold.** USD-neutrality did NOT relieve 2018 — robustly NEGATIVE
   cross-sectionally at every horizon (−14.8 / −25.1 / +1.1 / −20.5 bp). Why (ties arc 2017): the book's
   2018 obstacle is `fbr`-2018, and `me_long` carries a **real POSITIVE** 2018 (+0.90%, the directional
   WMR rebalancing). Stripping USD beta REMOVES `me_long`'s +2018 help rather than adding a +2018 leg —
   the cross-sectional reframe is a strictly *worse* expression of `me` for the book.
2. **Coin-flip + sub-cost (arc-2010 re-confirmed WITH month-end timing).** The month-end timing IS
   load-bearing (FWD2 ME-excess +14.6bp vs random-day ≈0 → the WMR reversion is real cross-sectionally,
   consistent with `me`), but the gross spread is ~coin-flip (frac+ 0.51–0.55) and small (~2–10bp
   median). Market-neutral needs ≥2 legs: 4-leg (top2) net-NEG every horizon (−10bp); 2-leg (top1)
   clears only an *optimistic* 6bp cost on a single **knife-edge horizon** (FWD2 only — FWD1/FWD3
   net-neg), net-negative at a realistic ~10bp 2-leg FundedNext RT. Arc 2010's "doubled-cost vs a
   coin-flip" persists even when the relative move is timed to the forced-flow event.
3. **No new component either way.** The only net-marginal cell (single-leg short-the-leader, top fwd
   −9bp FWD2) re-introduces USD beta and is just `me_short` with cross-sectional selection — already a
   PORTFOLIO component (1019). The novel, decorrelated contribution requires the 2-leg USD-neutral form,
   which is sub-cost.

**Verdict: KILL (cheap-kill at observation).** No pool/engine/null/council spent — §5d (coin-flip
capture-frequency entry + a structural multi-leg cost problem already mapped by 2010; §5f exit-sweep
doesn't bite). OOS never touched. Components UNCHANGED.

**Threads / lessons.** (1) The month-end WMR reversion is **real cross-sectionally** (FWD2 ME-excess
+14.6bp over random ≈0) — re-confirms the `me` mechanism from a new angle — but it is the SAME edge, not
a new decorrelated component; the cross-sectional form adds doubled cost and removes the directional 2018
benefit. (2) **USD-neutrality is NOT a lever to relieve the route's 2018 obstacle:** `fbr`-2018 is
mechanism-intrinsic (arc 2017) and the `me` family is 2018-*helped* by its directional leg; removing the
beta removes the help. NEW lesson: **stripping a regime exposure also strips the edge that lives ON it**
(generalizes arc 1018's "edge & tail are the same exposure" from the universe axis to the
cross-sectional/market-neutral axis). (3) **arc 2010 extended/closed:** relative-value on liquid FX
majors is doubled-cost-vs-coin-flip *even when timed to the month-end forced-flow event* — the
"but what if it were event-timed?" sub-question is now closed. (4) arc-2017 option (B) attacked once,
dead via the obvious thicken-a-proven-edge route (cross-sectional expansion of `me` is killed by
multi-leg cost); a genuinely thick fold-resolving component remains unfound — the route stays
operator-gated (the AFP/gate-resolution governance call from arcs 2016/2017 stands).

**Tooling.** No new BUILT tool — drift-lens arithmetic on the canonical `Panel` (vs-USD log returns +
month-to-date cumsum + forward sum) + the month-end mask convention (arc 1005/1011). No canonical core
touched. Driver `_disco2000_work/arc2018_xsec_monthend.py`.

**FLAGS (code not merged):** none.

### arc_1023

**INDEPENDENT verification of arc 2016's portfolio NOISE-FLOOR** (chat 1000–1999). Full record:
[`arcs/arc_1023_portfolio_noise_floor_independent.md`](arcs/arc_1023_portfolio_noise_floor_independent.md).

**Idea + because.** Arc 2016 (2000s) made a programme-redirecting claim — the 4-way book's residual
negative folds (2015 −0.047%, 2018 −0.124%) are statistically zero, so the ~18-arc 5th-leg hunt was
fold-painting inside the noise floor. My own arc 1022 (and 3018) were exactly that deprecated thin-leg
hunt. The Arc-10 norm for a consequential claim is independent reproduction by a DIFFERENT method
(2008/3009, 1019/3017, 1022/3018). arc 2016 used a within-fold per-trade bootstrap of the combined book;
I use an ACROSS-fold lens (10 IS fold ROIs as 10 annual samples) + a fold-resample bootstrap. NOT
gate-loosening (2016's council rejected that): the strict all-folds gate stays FAIL; I quantify why.

**What happened — EXACT reproduction + confirmation + a new finding.** Re-ran all 4 components through the
canonical apparatus at their committed configs. **Risk-convention calibration (arc-3017 FLAG-1):** at
`A1Config.risk_pct=0.5` (intended PERCENT) the daily-5%-DD cap BLOWS THROUGH (ROIs in the 100s of %,
flipped signs — 3017's nonlinear warning reproduced); the headlines live at the low-risk LINEAR
`risk_pct=0.005`, where I reproduce all four **EXACTLY** (gap +0.685 / me_long +0.232 / fbr +1.854 9/10 /
me_short +0.683) and the book at arc-2016 frozen weights to the decimal (mean +0.624%, worst −0.124%@2018,
2015 −0.047%, 8/10, across-fold sd 0.741%). Noise floor: **worst fold = −0.167 across-fold-sd from zero**
≈ 2016's 0.176σ; **across-fold sd (0.741%) ≈ 2016's within-fold sampling sd (0.703%)** — the decisive
method-independent statement (fold-to-fold variation is essentially ALL sampling noise; the year-folds
carry almost no real signal beyond the pooled mean). P(≥1 neg fold in 10) = 0.887. **NEW: the book MEAN is
significantly POSITIVE** — t=2.66 (df=9, p≈0.026), fold-bootstrap CI [+0.22%,+1.09%], P(mean≤0)=0.

**Diagnosis + meaning.** The per-year all-folds-positive gate demands a 10/10 positive-year streak from a
process whose per-year ROI sampling sd (~0.7%) EXCEEDS its per-year mean (~0.6%) — statistically
near-impossible regardless of edge quality. The book is a real, significant positive edge that fails ONLY
this too-fine gate. Scopes the operator's flagged paths: **A** (a mean/pooled/regime-block gate with
explicit SE fits a thin-component book; the per-year gate does not) — decision-support only, NOT adopted;
**B** (denser components to shrink per-fold SE below the per-fold mean) — hard given closed ground
(dense/shallow = coin-flip), but the only path keeping the strict gate.

**Verdict: KILL** (no new component; strict gate stays FAIL, 8/10). Confirms arc 2016 by an independent
method + chat and adds the book-mean-significance number. Components UNCHANGED (all 4 PORTFOLIO);
combined-book OOS NOT spent (§5g); no council (a measurement resolving a measurement question).

**Threads / lessons.** (1) **CONFIRMED (independent): the per-year all-folds gate has a NOISE FLOOR set by
component trade-counts** — across-fold sd ≈ within-fold sampling sd ⇒ year-folds are sampling-dominated; a
5th leg to flip a 0.17σ fold is fold-painting. Twice-confirmed reusable rule: quantify a marginal fold's
CI before hunting a leg; for a thin book judge the MEAN with SE, not a 10/10 streak. (2) **NEW: the 4-way
book mean is statistically positive (t=2.66, CI [+0.22%,+1.09%])** — a real edge failing a too-fine gate
(decision-support for path A; the strict gate stays sovereign-FAIL). (3) **arc-3017 risk-convention FLAG-1
is load-bearing** — the whole portfolio picture lives at `risk_pct=0.005`; at deployable 0.5 the DD-cap
nonlinearity flips signs, so deployable-risk evaluation must re-derive the floor (linear-regime
σ-distances don't carry through the cap). (4) **Programme status: the in-apparatus 5th-leg route is
noise-floor-blocked**; live levers are operator-gated (path-A gate-resolution decision, or escalation #3
tighter-cost/non-OHLCV). Autonomous discovery has mapped the book route to its honest floor.

**Tooling.** No new BUILT tool — all CALLED (`Panel.from_pairs`, the 4 registered signals,
`make_time_exit_predicate`, `A1Architecture`/`ArcFoldRunner`, `run_config_over_folds`; scoring canonical).
Noise-floor stats = one-off arithmetic on canonical `FoldStats.roi_pct`. Driver
`_disco_work/arc1023_noise_floor.py` (reproduces the 4 headlines EXACTLY at `risk_pct=0.005`).

**FLAGS (code not merged):** arc-3017 risk-convention FLAG-1 (restated, load-bearing) + the gate-resolution
governance question (operator's call — the per-year gate is below its noise floor for thin books; whether
to adopt a mean/pooled/regime-block gate is an operator/protocol decision, FLAGGED not taken). Carries the
standing `A1Config.time_exit_bars`-unwired flag. OOS never touched.
**[CORRECTION, arc 1024]** the "risk-convention FLAG-1 (load-bearing)" restatement above is WITHDRAWN —
arc 1024 proves `risk_pct` is a FRACTION (0.005 = 0.5% deployable risk), so the noise-floor finding holds
at deployable risk with stable fold signs; there is no risk-convention asterisk. See arc 1024.

### arc_1024

**DIAGNOSTIC — arc-3017 FLAG-1 (`risk_pct` "unit split") RESOLVED** (chat 1000–1999). Full record:
[`arcs/arc_1024_risk_convention_flag_resolved.md`](arcs/arc_1024_risk_convention_flag_resolved.md).

**Idea + because.** While reproducing the 4 components at deployable risk for arc 1023 I hit an empirical
contradiction with arc-3017 FLAG-1, which claims `A1Config.risk_pct` is PERCENT (0.5=0.5%) so the
components were run at `0.005 = 0.005%` ("100× too low") and the verdict is "risk-convention-dependent."
Two facts contradicted it: at 0.005 I reproduce the headlines EXACTLY (not 100× smaller), and at 0.5 ROIs
EXPLODE to the 100s of % (50%-risk ruin, not a "cap"). The code settles it: `live_balance.py` →
`risk_amount = balance × risk_pct`, default `0.01 = "1% per trade"` — a FRACTION, no ×100, passed straight
through by `A1Architecture`. Per the framework-bug-vs-methodology discipline, a flag that contradicts a
direct code read gets a HALT + diagnostic, not propagation.

**What happened.** Risk-sweep of the committed fbr config over the 10 IS folds at risk_pct ∈ {0.0025,
0.005, 0.01, 0.02, 0.05, 0.5}: **perfectly LINEAR across the deployable band** — 0.25%/0.5%/1% →
+0.927%/+1.854%/+3.705% (per-unit-risk constant ~0.927%), **fold signs stable at 9/10**. The daily-5%-DD
cap only begins biting at ~2% (8/10), bites at 5% (7/10), and is catastrophic at the absurd 50% (2/10,
−129%). ⇒ `risk_pct` is a FRACTION; the +1.854% headline is at **0.5% per trade, a normal deployable
risk**; 3017's "risk 0.5 flips signs" is the **50%-risk** regime (account-destroying), not a deployable
convention ambiguity.

**Verdict: FLAG-1 RESOLVED/WITHDRAWN** (KILL — diagnostic, no component). The entire portfolio
characterization (component headlines, the 4-way book, arc-2016/2017/1023 noise-floor + book-mean-positive
t=2.66) is at a sound 0.5% deployable risk, with fold signs stable across the 0.25–1% deployable band — no
risk-convention asterisk remains. My arc-1023 restatement of FLAG-1 as "load-bearing" is corrected.
Code is correct; no canonical change. OOS untouched; no council.

**Threads / lessons.** (1) **`risk_pct` is a FRACTION (0.005 = 0.5%); committed discovery risk = 0.5%/trade,
deployable.** No hidden 100× scaling. (2) **Fold signs stable across 0.25–1%;** DD-cap non-binding for
these thin signals until ~2%, catastrophic only at absurd risk → the noise-floor (1023/2016) and
me_short-2018-positivity (1019/3017) results carry to deployable risk unchanged. (3) **Process: a prior
chat's flag contradicting a code-read + linear-scaling check is diagnosed + corrected, not propagated** —
a wrong flag mislabels the whole corpus's numbers as untrustworthy. (4) Net: deployability picture is
clean — the book is a genuine positive-mean edge at 0.5% risk, blocked only by the per-year-gate noise
floor (operator path A/B). The real standing flag is `A1Config.time_exit_bars`-unwired (arc 1005), not
risk-convention.

**Tooling.** No new BUILT tool — CALLED `Panel.from_pairs`, `FailedBreakdownReclaimLongSignal`,
`A1Architecture`/`ArcFoldRunner`, `run_config_over_folds`; one-off risk-sweep arithmetic on canonical
`FoldStats.roi_pct`. Driver `_disco_work/arc1024_risk_convention.py`.

**FLAGS (code not merged):** arc-3017 FLAG-1 **WITHDRAWN** (resolved — `risk_pct` is a fraction, code
correct). Standing real flag: `A1Config.time_exit_bars`-unwired (arc 1005). OOS never touched.

### arc_1025

**Does the corpus's ONLY fold-RESOLVING edge (`fbr`) THICKEN? — arc-2017 option B via the depth lever**
(chat 1000s). Full record: [`arcs/arc_1025_fbr_density_sweep.md`](arcs/arc_1025_fbr_density_sweep.md).

**Idea + because.** The diagnostic chain (2016/2017/1023/1024, three chats) closed the thin-5th-leg hunt
as noise-floor-blocked and named the ONLY surviving productive spec — arc-2017 **option (B): a component
THICK enough that its per-year folds RESOLVE and clear zero.** arc 2018 attacked (B) by thickening `me`
cross-sectionally → killed by multi-leg cost. The un-tried (B) route: thicken the corpus's single
fold-resolving edge ITSELF. `fbr` (1013) fires only ~17–21/yr **because** it needs a DEEP (shadow ≥ 1.25
ATR) reclaim of a LONG (`K`=40) swing low — rare, deep forced-flow stop-runs. The same stop-run mechanism
fires more often at shorter `K` / shallower shadow. **HYP-A:** does a denser cell keep a fold-resolving
edge? **HYP-B:** is `fbr`'s 2018 wipeout (arc 2014) density-invariant, or do thinner triggers catch
tradeable 2018? USD majors only (arc 1022: fbr is USD-major-specific). All IS → OOS untouched.

**What happened — FALSIFIED.** (1) **Obs surface** (`arc1025_fbr_density_sweep.py`, 16 cells K∈{10,20,40,60}
× shadow∈{0.5,0.75,1.0,1.25}, `observe_long_capture` + the arc-1013 load-bearing structure control): the
edge degrades **MONOTONICALLY** with density along BOTH levers — capture 0.5865→~0.50, structure-excess
+0.108→~0, drift +0.237→~0 as the trigger shallows/shortens; the committed **K40/s1.25 is the cleanest
cell** (the deepest grab). One genuine non-coin-flip denser candidate: **K40/s1.00** (n=507, ~2× ref; cap
0.5227, struct +0.044, drift +0.098). (2) **§5f honest engine** (`arc1025_fbr_density_engine.py`, 7-exit
menu + fair null, risk 0.005): ref K40/s1.25 reproduces the committed **+1.854% 9/10 EXACTLY** (apparatus
validated, fold-9/2018 the lone neg, mean/SD 0.604). **DENSER K40/s1.00:** best exit only **+0.576% 6/10**;
**across-fold SD RISES 3.07→4.53%** while the mean falls to ⅓ → **fold-resolution proxy mean/SD craters
5× (0.604→0.127)**; 2018 (fold 9) still **−4.118%** and it ADDS 2014/2019 negatives; beats fair null
+0.870pp (a REAL but diluted edge, strictly dominated by the committed deep `fbr`).

**Diagnosis.** Reclaim DEPTH *is* the forced-flow signal, and depth is exactly what makes the event rare.
A deep wick that pierces a swing low and reclaims = a genuine stop-run liquidation reversal (dense
USD-major stop book); a shallow sweep is a routine probe with no liquidation behind it → no edge (cap →
0.50) AND it whipsaws (adds across-fold variance). Thickening fails on BOTH terms at once: lower mean AND
higher SD. **Fold-resolution and trade-count are NOT independent levers for a forced-flow edge — depth
couples them.** HYP-B confirmed (obs cap<0.52 every cell; engine 2018 neg every thickened exit):
`fbr`-2018 is density-invariant, mechanism-intrinsic (4th confirm w/ 1013/2014/3013).

**Verdict: KILL** (no new component; `fbr` UNCHANGED, PORTFOLIO at the committed deep config). arc-2017
**option (B) is closed for the BEST edge via the depth lever** — the complement to arc 2018's
cross-sectional-`me` closure; B is now dead from both its obvious levers (universe/cross-section + depth).

**Threads / lessons.** (1) **NEW (the escape hatch closes):** for a forced-flow edge, fold-resolution
(thickness) and edge-strength are COUPLED through trigger DEPTH — you cannot trade depth for trade-count
without trading edge for whipsaw variance; measure mean/SD (fold-resolution), not just trade count, before
claiming a denser variant "thickens." (2) **Strengthens the unified theory:** every real edge here is an
intrinsically-rare forced-flow reversion → all real edges are thin → the per-year all-folds gate is
structurally unsatisfiable for this corpus; option B is dead via both levers → the route stays
operator-gated (path A gate-resolution, or escalation #3 tighter-cost/non-OHLCV). (3) **`fbr`-2018 is
density-invariant** — a 4th independent confirmation the one real fold-obstacle is mechanism-intrinsic, not
a missing leg. (4) Surviving frontier unchanged: a genuinely-novel non-price-direction mechanism, or the
operator-gated levers.

**Tooling.** No new BUILT tool — CALLED `Panel.from_pairs`, the BUILT `FailedBreakdownReclaimLongSignal`
at swept `(swing_lookback, min_shadow_atr)`, `observe_long_capture(direction="long")`, the BUILT
`build_null_signal_evaluation`, and canonical `A1Architecture`/`ArcFoldRunner`/`run_config_over_folds`.
Drivers: `_disco_work/arc1025_fbr_density_sweep.py` + `_disco_work/arc1025_fbr_density_engine.py`
(reproduces the committed +1.854% 9/10 fbr headline EXACTLY).

**FLAGS (code not merged):** none new. Carries the standing `A1Config.time_exit_bars`-unwired flag
(arc 1005). OOS never touched.
### arc_2019

**DIAGNOSTIC: how many INDEPENDENT bets is the 4-component book? (council-driven)** (chat 2000s). Full
record: [`arcs/arc_2019_effective_bets_diagnostic.md`](arcs/arc_2019_effective_bets_diagnostic.md).

**Idea + because.** At the genuine strategic fork behind arcs 2016/2017/2018 (route noise-floor-capped,
operator-flagged; my arc 2018 closed the cross-sectional thicken-`me` lane) I convened
`/llm-council-discovery` (generative juncture, light weight). The 5 lenses split — refine fbr-2018 via an
M1 reclaim-confirmation filter (Refinement); test fbr ALONE on OOS, warning of "laundering a noise-mined
100%-reversion book into a deployable" (Soundness); hunt a novel event-anchored liquidity mechanism
(Mechanism/Steelman) — but the anonymous PEER-REVIEW round **converged, 3 of 5 reviewers independently,
on the one quantity all five lenses MISSED:** the effective number of INDEPENDENT bets in the EXISTING
book. That gates every branch (rank-1 ⇒ no leg/reframe helps, governance call; multi-bet-but-shared-tail
⇒ no reversion leg helps; genuinely diversified ⇒ different conclusion). CC committed to measuring it.

**Method.** Reproduced the 4 committed components EXACTLY (arc-2015 configs, canonical apparatus), then on
the 10 IS folds: per-fold ROI correlation; eigenvalue spectrum + ENB = (Σλ)²/Σλ²; bootstrap ENB CI (5000×,
seed 42); tail co-movement (co-negativity, corr in worst vs best folds); book pooled MEAN CI (frozen
equal + risk-parity). Driver `_disco2000_work/arc2019_effective_bets.py`; headlines reproduce committed
values exactly.

**What happened — all three council "laundering" fears REFUTED; the gate-thinness diagnosis CONFIRMED.**
(1) corr low (−0.366..+0.406), not rank-1. (2) **ENB = 3.32/4**, top eigenvalue only 38.9% of variance →
~3 independent bets (style-homogeneous, all reversion, but statistically decorrelated via DISJOINT event
timing — weekends vs month-ends vs stop-runs). (3) **bootstrap ENB 95% CI [2.02, 3.33], P(ENB<2)=0.021** →
diversification robust to n=10 noise (the cross-component complement to 2016/2017: per-component folds
don't resolve, but the cross-component INDEPENDENCE does). (4) **TAIL co-movement NEGATIVE** — pairwise
corr in the book's 5 worst folds **−0.201** (vs +0.123 best); co-negativity NEVER >2/4; **2018 = 2-down
(gap −6.79, fbr −4.20) / 2-up (me_long +0.90, me_short +0.86) near-cancellation, NOT a book-wide
drawdown** (refutes arc-2008's "shared 2018 tail"). (5) book MEAN robustly + (risk-parity +0.589%,
95% CI [+0.120%, +1.088%], P(mean<0)=0.004; equal +0.863%, CI [−0.150%, +1.863%], P=0.046).

**Read + verdict — KILL (diagnostic).** The book is NOT rank-1, NOT tail-correlated, and its mean is NOT
fake — it's a genuine ~3-independent-bet, negative-tail, robustly-mean-positive reversion PORTFOLIO. So
the all-folds-positive FAILURE is **purely the arcs-2016/2017 thinness at book level**: the every-
calendar-year gate trips because in any year ONE thin leg dips within-noise-negative (every worst fold is
a single-thin-leg noise dip, never a co-drawdown), not a diversification/tail/mean problem. **This
resolves the generative-vs-governance fork: a 5th decorrelated REVERSION leg CANNOT make the book AFP** —
no diversification deficit, no tail to hedge; an added thin leg just adds another every-year trip-chance.
Only a THICK fold-resolving standalone (arc-2017 option B — which would be AFP solo, a deployable on its
own; none known, closed ground) or the operator gate-resolution call remains. **Edge-hunting FOR THE BOOK
via the all-folds-positive route is closed; the lever is definitively the operator gate-governance call.**
Components UNCHANGED (still PORTFOLIO). No OOS (book fails IS AFP). The council's best NOVEL candidate
(option-expiry/gamma-pin, Mechanism lens) was unanimously flagged by reviewers as NOT OHLC-constructible
(no strike/options data → collapses to round-number behavior, dead arc 1010) — recorded so it is not
re-proposed.

**Threads / lessons.** (1) **NEW (decisive): the book is genuinely diversified — ENB 3.32, robust CI
[2.02, 3.33] — with NEGATIVE tail co-movement (worst-fold corr −0.201).** Style-homogeneity (all
reversion) does NOT imply factor-homogeneity: disjoint event-timing → ~3 independent fold-ROI bets. The
"one reversion bet sliced four ways" intuition is empirically wrong. (2) **NEW: every worst fold is a
single-thin-leg within-noise dip, not a co-drawdown** (2018 = 2-up/2-down) → the AFP failure is a
gate-RESOLUTION artifact, not a portfolio-construction failure → upgrades the operator FLAG from
"per-component vacuous" (2016/2017) to "the book is a SOUND PORTFOLIO the calendar-year gate cannot
certify." (3) **a 5th decorrelated reversion component is LOW-EV for the book** — it can't fix a gate that
trips on single-leg noise dips, and the book needs no more diversification; only a thick fold-resolving
standalone (none known) or the gate call remains. (4) **council process worked as the Arc-10 insurance
intends:** generative lenses each proposed a different action; anonymous peer-review surfaced the decisive
unmeasured quantity ALL of them missed; measuring it refuted the strongest dissent (the laundering charge)
with data, not rhetoric. (5) option-expiry/gamma-pin is NOT OHLC-constructible — do not re-propose
(reduces to dead round-number, arc 1010).

**Tooling.** No new BUILT tool — reused the canonical apparatus + BUILT `combine_fold_roi` (arc 2006) +
the four committed component signals; the diagnostic is `numpy.linalg.eigvalsh` + bootstrap resampling on
canonical per-fold ROI. No canonical core touched. Driver `_disco2000_work/arc2019_effective_bets.py`.

**FLAGS (code not merged):** none new. **OPERATOR FLAG reinforced (3rd time, cross-component angle):** the
4-component book is a genuine ~3-independent-bet, negative-tail, mean-positive PORTFOLIO; the all-folds-
positive calendar-year gate cannot certify it because the gate resolution sits below the thin legs'
per-fold noise floor. Honest operator options unchanged from arc 2017: (A) a noise-aware / pooled-trade /
coarser-window gate; (B) a thick fold-resolving standalone (none known — closed ground). Discovery has
mapped the accessible reversion frontier; the lever is the gate-resolution governance call.

### arc_1026

Full record: [`arcs/arc_1026_session_overnight_reversal.md`](arcs/arc_1026_session_overnight_reversal.md).

**Why this arc.** Resumed 1000s fresh; highest id in range = 1025 → 1026. The corpus has converged
(arc 2019): the 4-component book is a sound ~3-bet mean-positive PORTFOLIO whose AFP failure is a
gate-resolution artifact, and the only open *edge* route is **arc-2017 option B — a THICK fold-resolving
standalone** (a 5th reversion leg can't fix the book). Option B had two closed attempts (fbr-thicken
1025, me-cross-sectional 2018). The "unified theory" (real FX edges are intrinsically-thin forced-flow
reversions) is the seductive search-ending conclusion §2 says to question. The instrument universe is
pure FX (no metals/indices in the backup) so closed-ground applies everywhere. So I attacked the theory
with a **thick-by-design** mechanism: the **equity overnight-intraday reversal ported to FX sessions** —
does the illiquid Asian net move REVERT when London liquidity arrives (dealer inventory mean-reversion)?
It fires daily (~580/yr), the one property a fold-resolver needs.

**Distinct from arc 3016** (which I read carefully to avoid repeating): 3016 killed the London-open
**BREAKOUT** of the Asian range (whipsaw entry, cap 0.38 both ways). This is the **net-session-return
REVERSAL** — enter at London open OPPOSITE the overnight net move, not at a break point — a different
construction tradeable at ~H4 cost.

**What I did.** Observation-first (§5b), SL-agnostic, because a tight 2·ATR_H1 SL over a day-hold would
just reproduce 3016's whipsaw and tell us nothing about the mechanism. The mechanism's signature is
negative autocorrelation between the Asian net move and the London+NY day move. Measured corr, the
conditional reversal drift in daily-ATR units (vs the ~0.06–0.125R cost hurdle), per-pair, by year.
H1, 7 USD majors, IS 2010–2020, `Panel.from_pairs` canonical loader.

**Result — clean falsification.** corr(asian, day) = **+0.0006** pooled (per-pair −0.025..+0.021,
mixed) = random walk across the boundary. Conditional reversal drift is NEGATIVE/coin-flip at tradeable
thickness (|asian|≥0.5: −0.038 ATR, frac+ 0.463 — a slight *continuation*); the +0.10 at |asian|≥1.0 is
a thin-tail artifact (median −0.02). The only faint reversion is the early-London 07-09 deep tail
(|asian|≥1.0 +0.054 ATR, frac+ 0.552) but 17/yr and ~5× sub-cost. 2015 (−0.121) and 2018 (−0.045) both
negative — wrong sign.

**What it means.** 24h FX never closes, so there is no warehoused-inventory price concession to
mean-revert — the equity overnight-reversal's friction is absent. The near-zero autocorrelation is the
signature of an efficient continuous market. Crucially for option B: the one *thick-by-design* candidate
has ZERO edge where it is thick (~580/yr), and edge appears only in the thin deep tail (17/yr, sub-cost)
— so it **reinforces** the unified theory (real FX edges are intrinsically rare) rather than breaking it.

**Threads.** (1) Intraday session structure now dead in BOTH constructions (3016 break + 1026 reversal)
— no obvious third. (2) Option B closed on three constructions (1025/2018/1026). (3) The lever remains
the operator gate-governance call (arcs 2016/2017/2019/1023) — which I cannot make and must not pre-empt;
OOS frozen. (4) A genuinely novel thick mechanism with a *because* still earns a fresh test (§5a), but
none is currently visible; the in-apparatus FX-major OHLCV well is, on the honest read, dry.

**Tooling/FLAGS.** No new BUILT tool (focused scratch observers, pattern reused from 3012/3014/3015/3016).
No canonical change. Environment note (not a code flag): the working-tree `forex_backtester` env lacks
pyarrow — run discovery scripts via `conda run -n base python` (base has numpy/pandas/pyarrow). Drivers
reproducible from the arc doc.
### arc_2021 — adversarial temporal-stability stress of the book's mean-positive edge
**Why.** Arc 2019 (mine) closed the book's edge-frontier and concluded the deployability lever is the
operator's gate-governance call, resting on one load-bearing fact: the book is *mean-positive* (risk-parity
+0.589%, P(mean<0)=0.004). But arcs 2016/2017/1023/2019 all characterized the book's NOISE — none asked the
single most basic question a quant asks before trusting a backtest mean: *is it stable in time, or
front-loaded in an early regime that decayed?* If the +0.589% is a 2011–2015 artifact, path-A is hopeless
and the book should be downgraded. I attacked the mean adversarially (conservative bias: try to break it).

**What I did.** Reproduced the 4 components exactly off arc 2019's frozen configs (headlines verify to the
decimal), combined to the per-year (2011–2020) risk-parity book, split EARLY(2011–15)/LATE(2016–20),
bootstrapped each half's mean and the early−late decay. Canonical apparatus only; no OOS; no gate touched.

**What I found — the adversarial test fails to break it.** The late half (2016–2020) is still mean-positive
(+0.404%, P(<0)=0.060, 1/5 neg = 2018), the decay (+0.365%) is statistically insignificant (CI spans zero,
P(decay>0)=0.76), and — most informatively — the components decay HETEROGENEOUSLY: fbr is front-loaded
(+2.975→+0.733) but me_long *strengthens* in the late half (+0.055→+0.409). So the book's time-stability is
the SAME ~3-independent-bet decorrelation arc 2019 found in the fold-correlation matrix, now expressed on the
TIME axis — me_long backfills as fbr fades. The book is not one decayed front-loaded factor.

**Honest about what it doesn't prove.** n=5/half is low power, so "no significant decay" is weak evidence of
stability rather than proof; the late robustness leans on risk-parity (equal-weight late mean is positive but
P(<0)=0.249); and fbr — the strongest leg — *is* materially weaker recently. This is IS-only; the OOS holdout
stays frozen (the book hasn't earned it), and late-IS persistence is only a proxy for 2021+.

**Meaning.** The book's mean edge is not a decayed early-era artifact — it persists into the late IS half with
no significant decay. This REMOVES one adversarial objection to the operator's path-A and modestly strengthens
the deploy case, while changing nothing about deployability (the book still fails strict all-folds-positive; I
loosened no gate and claimed no pass — this is characterization). With this, the book is now mapped on both
axes the operator's gate call needs: noise (2016/2017/1023/2019) and time (2021). The frontier verdict is
unchanged — edge-hunting for the book is closed; the lever is the operator's gate-governance call — but that
call now has the temporal-robustness input it was missing. Components UNCHANGED (PORTFOLIO); diagnostic → KILL.


### arc_2024

**Full-month-conditioned, dual-direction month-end reversion — option-B thick-standalone test** (chat
2000s). Full record: [`arcs/arc_2024_fullmonth_month_end_reversion.md`](arcs/arc_2024_fullmonth_month_end_reversion.md).
No council (falsified at observation; §5d cheap-kill).

**Idea + because.** The portfolio/leg-hunt route is proven closed (arc 2019/2022/3021), so the one
autonomous route to a deployable is arc-2017 **option B**: a component THICK enough that per-year folds
RESOLVE and clear zero. The `me` family (1011/1019) is the ONLY mechanism that carries the binding 2018
fold (me_long +0.90, me_short +0.86). `me` fires on a 2-day move (`into_bars=2`); the documented WMR
month-end fix rebalancing flow scales with the MONTH'S currency appreciation (equity-hedge rebalancing),
so the economically-correct trigger is the full-month (~20 D1-bar) return. A **unified dual-direction** fade
(long if the currency fell over the window, short if it rose) is ~2× thicker than each one-sided `me` leg →
a shot at a thick standalone that keeps +2015/+2018. (arc 2018's option-B closure was the CROSS-SECTIONAL
2-leg USD-neutral version — a different, doubled-cost construction; single-leg per-pair via the window
lever was untested.)

**What happened — the data inverted the because.** Obs (7 USD majors, D1, IS 2010-2020; canonical
`_month_end_into_move` + direction-aware `observe_long_capture`; into_bars∈{2,5,10,20}×thr∈{1.0,0.5}, vs a
random-day same-move control). (1) **Month-end timing is load-bearing at every window** — month-end EXCESS
drift +0.13..+0.32 ATR vs the random-day control's +0.01..+0.04 (n=5-18k) — the dual-direction
construction reproduces arc-1011's control. (2) **Q1 REFUTED:** capture is non-monotone in window length,
peaking ~into5 (0.5548), into2/20 both ~0.53-0.55 → the full-month return is NOT a cleaner trigger; `me`'s
`into_bars=2` is ~optimal. (3) **Q2/option-B DECISIVE — no cell is all-folds-positive, and the binding fold
has a clean mechanism:** 2018 drift decreases MONOTONICALLY with window length, robust under BOTH
thresholds (thr1.0: into2 +0.045 → 5 -0.029 → 10 -0.084 → 20 -0.170; thr0.5: +0.083 → +0.021 → -0.285 →
-0.288). 2018 was a strong-USD TREND year — a 2-day move into month-end isolates the mechanical fix
over-extension, but a 20-day move IS the trend, so fading it at month-end = fading a real trend → loses
precisely in 2018 (the same "in a strong-USD regime the move IS the trend" failure that makes me_long
2015-negative, now for the full-month window in 2018). (4) **Q3:** thickening does NOT collapse to coin-flip
(cap stays 0.51-0.55 from n=240→783, unlike fbr arc 1025's depth dilution) but buys nothing toward AFP
(fold-resolution stays 7-9/11; the only +2018 cells are the thin 2-day = the existing `me`, gross +0.045 ATR
« the ~0.05-0.10R cost hurdle = sub-cost).

**Verdict: KILL (obs cheap-kill, §5d).** The refinement is falsified at observation; the only non-coin-flip
cell is the already-engine-tested `me` PORTFOLIO config, so §5f does not bite (no new best-version entry to
put on the engine) — mirrors arc 2014/2020 discipline. OOS untouched.

**What this closes + lesson.** (a) Closes the "full-month trigger improves `me`" refinement — `into_bars=2`
is ~optimal, the full-month return is strictly worse on 2018. (b) Closes single-leg **option B for `me` via
the trigger-window lever** — the 4th independent option-B construction to fail (with 2018 cross-sectional me,
1025 fbr-depth, 1026 thick-by-design session). (c) **NEW lesson:** the month-end reversion is a SHORT-horizon
(2-5 bar) fix-flow over-extension, NOT a monthly-appreciation rebalancing effect — lengthening the trigger
window contaminates the signal with trend and specifically destroys the strong-USD 2018 fold (2018 drift
monotone-decreasing in window length). The intuitive "edge scales with the monthly move" economic model is
the WRONG model for the tradeable signal; the edge lives in the acute short-horizon over-extension, exactly
where the pool is thin (generalizes arc 2017). Operative frontier unchanged — the deployability lever is the
operator path-A gate-governance call (arc 2019/3021). Components UNCHANGED (me legs PORTFOLIO).

**Tooling:** no new tool (reused canonical `_month_end_into_move` + BUILT `observe_long_capture`); no
TOOL_REGISTRY append. **FLAGS:** none. Driver: `_disco2_work/arc_2024_fullmonth_me_obs.py`.


### arc_2025

**Gotobi (五十日) Tokyo-fix settlement flow** (chat 2000s). Full record:
[`arcs/arc_2025_gotobi_tokyo_fix.md`](arcs/arc_2025_gotobi_tokyo_fix.md). No council (falsified at obs; §5d).

**Idea + because.** A genuinely novel, documented forced flow distinct from the corpus's fix work (arc
3008 was the month-end *London* WMR fix): the **gotobi** effect — on days divisible by 5 (5/10/15/20/25 +
month-end), Japanese corporates settle trade invoices in USD, creating USD-buying demand into the **09:55
JST Tokyo fix** → USD-vs-JPY drifts UP through the Tokyo morning. Attractive on paper: calendar-driven (so
orthogonal to the price-move risk-off factor that ties every existing component together) and thick (~6
events/month). Timing: 09:00 JST = 00:00 UTC, 09:55 JST = 00:55 UTC → the H1 bar timestamped 00:00 UTC
(09:00–10:00 JST) spans the fix; its open→close return is the cheap pre-fix-drift proxy.

**What happened.** Obs on 4 JPY pairs (USDJPY + EUR/GBP/AUD-JPY), H1, IS 2010-2020, gotobi vs non-gotobi.
**The effect is REAL and mechanism-consistent:** on ALL 4 pairs the gotobi-day fix-bar mean is POSITIVE
(+0.10..+0.28p) while the non-gotobi Tokyo-morning mean is NEGATIVE (−0.13..−0.97p) — the normal Tokyo
morning drifts down, but the documented gotobi USD-settlement demand flips it up. The gotobi EXCESS is
positive on all 4 (+0.24..+1.14p), strongest on the JPY crosses (the USD-buy → JPY-sell also lifts the
cross); EURJPY is even +2018 (+1.05p) and 8/11 folds positive. **DECISIVE: ~10–60× SUB-COST** — the
gotobi drift (+0.1..+0.28p) is an order of magnitude or two below the ~3–6p H1 round-turn cost on every
pair; frac+ ≈ 0.50–0.52 confirms a tiny mean drift, not a reliable directional bet. **Resolution-invariant:**
the excess sits in a mostly-pre-fix window, so even a perfect M1 entry capturing only 09:00→09:55 stays far
below cost (cf. arc 2002, gap continuation TF-invariant).

**Verdict: KILL (obs cheap-kill, §5d).** Real but sub-cost; no above-cost entry exists so §5f does not bite.
OOS untouched.

**What this closes + lesson.** Closes the gotobi / intra-month Tokyo-fix flow, directly complementing arc
3008 (month-end London WMR fix, also real-but-sub-cost). **Generalized: the FIX-FLOW FAMILY is
real-but-sub-cost** — two independent fix mechanisms (month-end WMR London, intra-month Tokyo gotobi), same
verdict. Pure calendar-fix flows are sub-pip-to-~1-pip = an order of magnitude below FX spread+commission,
un-capturable net on a retail-cost broker. This sharpens WHY the corpus's surviving edges (gap/me/fbr)
clear cost: they all condition on a LARGE (≥1 ATR) price displacement; pure calendar timing without a large
displacement is too small to trade. Operative frontier unchanged (operator path-A gate call); components
UNCHANGED.

**Tooling:** no new tool (self-contained obs). No TOOL_REGISTRY append. **FLAGS:** none. Driver:
`_disco2_work/arc_2025_gotobi_obs.py`.


### arc_2026

**Japanese fiscal-year-end (31 March) repatriation** (chat 2000s). Full record:
[`arcs/arc_2026_jpy_fiscal_yearend_repatriation.md`](arcs/arc_2026_jpy_fiscal_yearend_repatriation.md). No
council (falsified at obs; §5d).

**Idea + because.** The arcs 2024/2025 unifying read — surviving edges need a LARGE (>=1 ATR) displacement +
forced flow; pure calendar-fix flows are sub-cost — left one clean test: is there a LARGE-displacement
(multi-week) calendar flow that DOES clear cost? The canonical candidate is the **Japanese fiscal-year-end
(31 March) repatriation**: institutions close books 31 March -> repatriate foreign assets / settle hedges
-> JPY-buying into year-end -> JPY-pairs should FALL (JPY strengthens) into 31 March. Multi-week => multi-ATR
displacement => clears cost IF directionally reliable. Distinct from `me` (price-triggered reversion at every
month-end); this is a calendar directional bias SPECIFIC to March.

**What happened — the hypothesis INVERTS.** Obs (4 JPY pairs, D1, IS; JPY-strength drift over the last 10
D1 bars into each month-end, March vs other months). **JPY WEAKENS into March-end** — March jpy_strength is
NEGATIVE on all 4 pairs (USDJPY -0.46, EURJPY -0.10, GBPJPY -0.59, AUDJPY -0.45 ATR) and the March EXCESS is
NEGATIVE on all 4 (-0.10..-0.62) — the OPPOSITE of repatriation-driven JPY strength, robust across the JPY
leg. **And coin-flip per-year** (March>0 only 4-6/11; frac+ 0.36-0.55), n=11/pair (thin). The lone exception
is 2015 (+JPY-strength on all 4: +0.47..+2.36) — an SNB/EUR-collapse-era idiosyncratic risk-off, not a
March-specific tell; 2018 is mixed-to-negative.

**Diagnosis.** The fiscal-year-end repatriation is well-known and SCHEDULED -> the hedging/repatriation is
front-loaded into Feb/early-March, so by the actual 31-March deadline the JPY-buying is largely done and the
late-March approach sees position-squaring / reversal -> JPY weakness. A textbook "the flow everyone knows
about is already in the price." The flow may be real, but it is NOT capturable as a late-March JPY-strength
directional bet.

**Verdict: KILL (obs cheap-kill, §5d).** Falsified twice — wrong sign (negative March excess on all 4) and
coin-flip per-year on thin n=11/pair. No reliable above-cost entry; §5f does not bite. OOS untouched.

**What this closes + lesson.** Closes the Japanese fiscal-year-end repatriation as a tradeable late-March
flow. **NEW lesson — the TWO failure modes of known-calendar forced flows:** (i) **sub-cost** (displacement
too small: the fix-flow family, WMR 3008 + gotobi 2025) or (ii) **priced-in** (displacement large enough but
the flow is anticipated -> front-run -> the realized move at the calendar date is coin-flip-to-reversed: this
arc). The corpus's surviving edges (gap/me/fbr) avoid BOTH because they key off a **SURPRISE price
displacement** (a weekend gap, a >=1-ATR move into the fix, a deep stop-sweep) — an unscheduled, large,
not-fully-anticipated event — NOT a known calendar date. Tradeability needs LARGE displacement AND
not-fully-anticipated; a known calendar date supplies at most one of the two. This sharpens the arc-2024/2025
unifying read and helps explain WHY ~80 arcs of calendar/seasonal cuts (closed ground) are dead. Operative
frontier unchanged (operator path-A gate call); components UNCHANGED.

**Tooling:** no new tool (self-contained obs). No TOOL_REGISTRY append. **FLAGS:** none. Driver:
`_disco2_work/arc_2026_jpy_fye_repat_obs.py`.

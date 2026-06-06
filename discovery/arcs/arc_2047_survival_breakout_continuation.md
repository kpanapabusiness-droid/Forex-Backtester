# arc_2047 — Survival-filtered swing-HIGH breakout CONTINUATION long (does fbr's load-bearing filter work on the continuation side?)

**Chat:** 2000s · **Date:** 2026-06-06 · **Verdict:** KILL (obs cheap-kill, §5d) · **Disposition:** KILL

## Log synthesis (step a — fresh eyes, honest-era only)
Pulled main; read DISCOVERY_PROTOCOL (followed it), the Tier-1 ledger (arcs 0→2046) + recent Tier-2 (the honest-exit deploy thread 1042→1046 / 2040→2046, and the exploratory cheap-kills 2027/2028/2029), LESSONS, TOOL_REGISTRY. State of the corpus:
- **The honest-exit deploy thread is CLOSED on every axis (1042→1046, 2040→2046).** Under honest §5f nested-WFO exit selection, the 4-component reversion book (gap 1006 · me_long 1011 · fbr 1013 · me_short 1019) loses ~half its advertised deploy mean, all its significance (t 2.16→~1.3), and ~half its effective bets (ENB 3.32→~1.8). On the FROZEN holdout the whole "portfolio" collapses to **me_long-SOLO** (~+0.4%/yr, Sharpe ~0.5, borderline-significant, NOT all-folds-positive, vehicle-infeasible per 2033/2045). fbr's IS-diversification dies OOS (mean-negative on the holdout). The book's signal+outcome+cost layers are §11-verified honest end-to-end.
- **The binding wall is the 2014 + 2015↔2018 strong-USD regime** — leg-count-invariant (2-way and 4-way both fail it). A **2014/2018-positive regime-orthogonal component** is the *only* path to an AFP book and remains unfound across ~18 routes.
- **Closed/mapped:** all shallow single-condition directional (long+short, every TF/pair/lens, stop-removed 3004); relative-value/market-neutral (2010: 2-leg cost wall + corr≈0); microstructure (spread-z 2027, triangulation level/1st/2nd-moment 3005/1027/2028, round-number 1010, gotobi 1008/2025, fix 3008); regime detection (inverts, 3003); session-break fade (3016); breakout-RETEST (3006); the whole DISCOVERY_DIRECTION menu; FX instrument universe (pure FX, no metals — 1045).
- **fbr is the corpus's ONE clean directional structural win.** Arc 2029 (my prior chat's last exploratory arc) established the fresh insight that **swing-SURVIVAL significance is fbr's load-bearing feature** — a calendar-refreshed level (PDL/PWL) dilutes it to a coin-flip. That insight has only ever been applied to the *reversal* (failed-breakdown reclaim) side.

## Idea (step b — log-seeded, with a mechanism)
The corpus's breakouts (Donchian 1001/1002/1003) used **raw N-bar highs** — they never carried the swing-survival significance filter that 2029 just proved load-bearing. The never-tested continuation analog of fbr: **a decisive-momentum close ABOVE a long-*surviving* swing high** — a level that survived K bars = genuine resistance with concentrated breakout-buy-stops above it.

*Because:* if a confirmed break of a *significant* (survived) resistance triggers the resting stop cluster and signals genuine demand absorbing the level, the move should **continue**; and in sustained strong-USD trends (2014/2015/2018) breaks of significant resistance in the trend direction should persist → **positive exactly where the reversion book bleeds** → a candidate 2018-orthogonal leg. It also extends the one directional win (the fbr deep-structural template) to its untested continuation mirror, with fbr's own proven filter.

**Falsifiable prediction:** survival-filtered-breakout honest +1R capture must **EXCEED** both the unconditional baseline (~0.4877) AND a same-momentum-bar-NOT-at-a-surviving-high control — and ideally be positive in 2014/2018. **Falsifier:** breakout ≈ control (survival not load-bearing) and/or coin-flip capture / negative drift.

## What I did (steps c–d — cheap-kill observation)
Population = H4, **7 USD majors** (AUDUSD/EURUSD/GBPUSD/USDJPY/NZDUSD/USDCAD/USDCHF — fbr's universe), IS 2010–2020 + OOS measured (not tuned). Per bar, ex-ante (no-lookahead):
- **surviving swing high** = `high_ask.shift(1).rolling(K).max()` (strictly prior K bars), K ∈ {20, 40, 60}.
- **breakout fire** = decisive up-momentum body `(close_mid−open_mid)/ATR ≥ 1.0` **AND** `high_ask > prior_high` (pierced) **AND** `close_mid > prior_high` (confirmed close above). ATR = Wilder(14) MID shift1.
- **control** = the SAME big-up-body bar that is NOT at a surviving high (`~pierced`) — the matched same-magnitude baseline (the fbr-style structure control).
Scored honest +1R-before-SL capture + 24-bar drift via canonical `observe_long_capture` (gross, characterization only — NOT a gate).

## What happened — FALSIFIED four ways
Pooled (7 majors, full span; baseline unconditional capture ≈ 0.4877):

| K | population | n | capture | 24-bar drift (ATR) |
|---|---|---|---|---|
| 20 | breakout | 4070 | 0.4953 | −0.034 |
| 20 | control | 5641 | 0.4960 | −0.043 |
| 40 | breakout | 2944 | 0.4976 | −0.057 |
| 40 | control | 6918 | 0.4958 | −0.034 |
| 60 | breakout | 2424 | 0.4889 | −0.136 |
| 60 | control | 7515 | 0.4983 | −0.009 |

1. **Coin-flip capture.** Survival-filtered breakout capture ~0.49–0.50 at every K — at, not above, the 0.4877 baseline. No edge.
2. **Breakout ≈ control — the decisive control failure.** 0.4976 vs 0.4958 (K=40); 0.4953 vs 0.4960 (K=20). The swing-survival structure is **NOT load-bearing for the continuation** — completely unlike fbr's reversal, where AT-swing-low (0.52–0.61) ≫ same-wick-elsewhere (coin-flip). The very filter that makes fbr work adds nothing here.
3. **Forward drift NEGATIVE, and deeper-survival is WORSE.** The decisive breakout bar mildly *reverts* (−0.03 → −0.14 ATR); the most-significant levels (K=60) revert the most (drift −0.136, cap 0.4889 — below baseline). The signal runs backwards at the extreme.
4. **NOT 2014/2018-positive — its raison d'être fails.** breakout K=40 per-year: **2014 cap 0.4398** (among the worst — the key blocker year), 2016 cap 0.4368 / drift −1.15 (catastrophic), 2018 cap 0.4821 / drift +0.17 (mild). Per-pair, 4 of 7 majors sub-0.50 (GBPUSD 0.452, NZDUSD 0.467, USDCAD 0.475, AUDUSD 0.499); the one tail (USDJPY 0.556) is the arc-2011/3011 pair-mix signature, not an edge.

## Why it fails (the mechanism)
**Swing-survival significance is load-bearing only for the FORWARD-confirming reversal, not the BACKWARD-confirming continuation** — the breakOUT-side completion of arc 1014's reclaim-vs-breakdown finding. fbr works because the reclaim is *forward-confirming*: at entry the liquidity-driven down-move is over and the up-move *hasn't started*, so i+1 enters before it. A confirmed break of a surviving high is *backward-confirming*: the move that closes above the level **IS** the stop-run — the resting breakout-stops are already triggered by the breakout bar itself, so there is nothing left to fuel continuation, and the i+1 entry buys AFTER the displacement, landing at the local high that mildly reverts (the same i+1-enters-the-extreme problem as arc 1014's confirmed-breakdown short and arc 2001's gap entry). A surviving level is a dense *resting-liquidity* pool whose **grab-and-reclaim reverts** (fbr); a **break** of it spends that liquidity rather than inheriting a directional edge. So the survival filter does not transfer from reversal to continuation — it is specific to the forward-confirming grab.

## Verdict + what this closes
**KILL (obs cheap-kill, §5d).** Capture is a coin-flip (~0.49–0.50, ≈ control, negative drift) → **§5f does not bite** (no non-coin-flip entry to put on the engine); no engine / null / council spent (matches 2016/2027/2028/2029/1014). Closes the most natural continuation extension of the corpus's one directional win: **survival-filtered breakout continuation is not a continuation-side fbr.** STRENGTHENS the unified theory — it sharpens 2029's "swing-survival is load-bearing" to its exact scope (*forward-confirming reversal only*) and re-confirms arc 1014's forward-vs-backward-confirming distinction on the breakout side. The 2014/2018-positive regime-orthogonal component remains unfound; structural trend-continuation at H4 is closed-ground (coin-flip), survival-filter included. Components UNCHANGED (all 4 PORTFOLIO; deploy core = me_long-solo per 1046). Deployable-system count = 0.

**Threads.** None on the breakout-continuation / survival-filter axis (the filter is reversal-specific). The 2018/2014 component, if it exists, is not a directional-structure leg — every structural-direction lane (reversal + continuation, long + short) is now mapped to coin-flip. The operative deployability lever remains operator **path-A (gate-governance)** on the existing mean-positive book.

**Tooling:** no new BUILT tool — canonical `Panel.from_pairs`, `observe_long_capture`, BUILT `_atr_shift1_mid` only; no TOOL_REGISTRY append (observation was inline scratch, matching 2027/2028/2029). **FLAGS:** none (no canonical change). Driver: `_disco_work/arc2047_survival_breakout_obs.py` (scratch, not committed).

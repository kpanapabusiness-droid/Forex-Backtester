# Best-version audit (arc 1012) — month-end reversion long

> **Additive note to the arc-1011 PORTFOLIO record.** Arc 1012 completed the §5f
> best-version test that arc 1011 left informal (the exit/SL menu was only
> sampled, not swept; the negative folds were not diagnosed). **Outcome: the
> survivor ceiling is closed — the component's best HONEST version is NOT
> all-folds-positive. Disposition unchanged: PORTFOLIO.** OOS never touched (no
> IS-clearing config; holdout preserved).

## Reproduction (registered tools, canonical engine)
`MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2)` (BUILT) → D1, 7 USD
majors, `ArcFoldRunner` → `MultiPairBacktester`, FundedNext costs ON, `build_v3_folds`
IS folds, `make_time_exit_predicate` (BUILT) for the hold. Baseline reproduced
byte-for-byte: sl_only / time-exit 2-bar / sl 2.0 → per-fold
`+0.40 +0.29 +0.96 −0.23 −1.14 −0.51 +0.34 +0.90 +1.16 +0.15`, mean **+0.232%**,
7/10 positive (matches arc 1011 exactly).

## §5f nested exit/SL sweep (24 configs) — 0 all-folds-positive
Full registered menu × SL ∈ {1.5, 2.0, 2.5}; time-exit caps {2,3,5}. Anti-fishing:
a config qualifies ONLY if the SAME config is all-folds-positive (no per-fold
cherry-pick). Result: **0/24 all-folds-positive.** Best fold count 7/10
(sl_only/te2/sl2.0 = the baseline; sl_only/te3/sl1.5). Longer holds (te5) and
TP/trail policies DILUTE (the reversion is a tight ~2-3 day event; holding longer
gives it back and exposes to continuation) — re-confirms arc 1011's informal read
and the corpus-wide "exits don't rescue fold-fragility" (arcs 0/1004/1007/2004).

| construction | best SL | mean% | worst% | folds+ |
|---|---|---|---|---|
| sl_only / te2 | 2.0 | +0.232 | −1.14 | 7/10 |
| sl_only / te3 | 1.5 | +0.397 | −1.49 | 7/10 |
| sl_only / te3 | 2.0 | +0.505 | −0.79 | 6/10 |
| sl_only / te5, all tp/trail / te5 | — | ≤0 | ≤−1.2 | ≤6/10 |

## Negative-fold diagnosis — a 2014-2016 USD-bull regime block
fold_id→OOS year: {2:2011, 3:2012, 4:2013, 5:2014, 6:2015, 7:2016, 8:2017, 9:2018,
10:2019, 11:2020}. The 3 negative folds are **2014, 2015, 2016** — a CONTIGUOUS
block = the strong-USD-bull regime (EUR 1.39→1.05, SNB unpeg Jan-2015, oil crash).
A regime signature, not random fold-fragility. Two **pre-registered, mechanism-reasoned**
refinements, both tested WITHOUT tuning to the bad folds:

- **(A) Quarter-end only (Mar/Jun/Sep/Dec)** — *because* quarter-end rebalancing
  flows are materially larger/cleaner. **FALSIFIED:** 4/10, mean +0.058%, min-trades 0
  (thins catastrophically). The larger-flow hypothesis is wrong.
- **(B) Trend filter close>SMA100 (fire only when NOT in a downtrend)** — *because*
  an informed prevailing trend should overwhelm the mechanical flow. **FALSIFIED:**
  4/10 (SMA50 3/10, SMA200 3/10) — requiring "not a downtrend" HURTS.

The **inverse** (close<SMA100, fire only in a downtrend) is marginally better — 8/10,
mean +0.302% — i.e. the reversion is *stronger* for already-weak (below-trend)
currencies (a coherent "rebalancing buys the underweight/oversold leg" story). But
this is **post-hoc** (filter-vs-inverse selected after the fact = fishing) AND it
**still leaves 2015 & 2016 negative** — the SNB-unpeg / USD-peak folds survive every
cut. It does not cross the all-folds-positive line, so it cannot change the
disposition and is NOT claimed. Logged as a low-priority thread only (see below).

## Verdict
**Survivor ceiling CLOSED for the month-end reversion.** Best HONEST version (full
§5f exit/SL menu + both reasoned diagnosis-driven refinements) is NOT
all-folds-positive; the 2014-2016 USD-bull block is irreducible by any honest exit
or pre-registered filter. **Disposition: PORTFOLIO (re-affirmed, survivor-ceiling
confirmed)** — the component remains valid (mean-positive net of costs, decorrelated,
threshold-robust), it simply cannot stand alone as a survivor. OOS preserved.

**Thread (not claimed, for a possible future fresh arc):** month-end reversion may be
stronger for below-trend / oversold currencies ("rebalancing buys the underweight")
— pre-register that *as the mechanism* with its own ex-ante population + fair null
before believing it; here it was a post-hoc contrast and still sub-survivor.

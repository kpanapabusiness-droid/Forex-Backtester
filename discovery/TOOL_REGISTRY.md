# TOOL_REGISTRY — what discovery arcs CALL vs BUILD

> Two tiers of code, two trust rules. This registry is the single place an arc checks before
> writing a line of measurement or experiment code. **[`DISCOVERY_PROTOCOL.md`](./DISCOVERY_PROTOCOL.md)
> is authoritative**; this file is the operational lookup it points at.

## The load-bearing distinction (governs everything below)

- **MEASUREMENT tools** = what *does* the testing: the WFO runners, cost application, scoring /
  `FoldStats`, the pool/population build, the fold structures, the discovery judge, and the
  SL-honest engine. **A bug here is invisible to the gate — the gate IS this code** (exactly the
  Arc-10 failure class). These are **CANONICAL / LOCKED: arcs CALL them, never reimplement.** If
  one looks wrong, an arc FLAGS it in its arc doc (code is human-gated, protocol §9) — it does not
  patch it mid-run.
- **EXPERIMENT tools** = what is *being* tested: filters, clustering methods, transforms, exit
  policies, signal logic, feature constructions, soundness controls (e.g. a random-entry null).
  **A bug here fails the WFO and dies loudly.** CC builds these FREELY, registers them in the BUILT
  section, and reuses them across arcs.

The rule is NOT "CC can't code." It is: **CC builds experiment tools freely; CC calls measurement
tools always** (never re-rolls the apparatus).

## Why this registry exists (the Arc-0 survey)

Arc 0 (the supervised trial) was surveyed against the in-tree core. The finding:

- **Arc 0 CALLED canonical measurement for essentially everything** — `build_arc_pool`,
  `run_step_2` / `run_step_3`, `ArcFoldRunner`, `OracleFoldRunner`, `A1Architecture` / `A1Config`,
  `build_v3_folds`, `Panel.from_pairs`. It did **not** re-roll the engine / WFO / cost / scoring
  core. Good.
- **What it re-rolled in scratch (`_arc0_work/`) was avoidable:** (1) per-stage *driver
  boilerplate* (load → build → loop → summarize), (2) **per-year OOS folds** (2021-present) — hand
  built because `build_v3_folds` exposes 2021+ only as one locked holdout, (3) the **all-folds-
  positive discovery judge** — distinct from the L_PROTOCOL dual-tier gate, so it had no canonical
  home, and (4) a **random-entry NULL baseline** — written *twice* (in `wfo_validate.py` and
  `null_compare.py`), with no canonical or reusable equivalent (the only prior one is retired-era,
  archived under `attic/`).
- **Fix landed by this registry's PR:** (2) and (3) are MEASUREMENT and are now canonical
  (`core/wfo/discovery_measure.py`); (1) is removed by the **standard entry point** below; (4) is an
  EXPERIMENT tool → the first expected **BUILT** entry (built by the first arc that needs it, under
  `discovery/tools/`, then registered here).

---

## The standard measurement entry point (call this — do not re-roll a driver)

A discovery arc runs its measurement by CALLING the canonical pieces in this sequence. Copy/adapt;
do not rewrite the apparatus. (`PAIRS`, the signal module, and config are the arc's own; everything
imported below is LOCKED.)

```python
from datetime import date

# (1) load the panel — canonical loader, real bid/ask, EET sessions
from core.sim.panel import Panel
panel = Panel.from_pairs(
    PAIRS, tf="H4",
    histdata_root=r"C:\Users\panap\histdata_backup",  # the recovered 65 GB corpus (see README)
    cache_root="data/cache", boundary_convention="5ers_eet",
)

# (2) ex-ante population  (protocol alias: "build_ex_ante_bounded_population" → build_arc_pool)
from core.arc.arc_pool_builder import ArcPoolConfig, build_arc_pool
pool = build_arc_pool(my_signal, {"H4": panel}, ArcPoolConfig(
    arc_name="arc_<id>_<slug>", sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005,
    window_start=date(2010, 1, 1), window_end=date(2020, 12, 31),
))

# (3) characterize — cluster + capturability
from core.steps.step_2_clustering import run_step_2
from core.steps.step_3_capturability import run_step_3
s2 = run_step_2(pool.trades, pool.paths)
s3 = run_step_3(pool.trades, pool.paths, s2.cluster_assignments, cluster_centroids=s2.centroids)

# (4) cheap kills — oracle-best-cluster ceiling + raw triage on representative folds
from core.architectures.a1_system_level_filter import A1Architecture, A1Config
from core.runners.arc_fold_runner import ArcFoldRunner
from core.runners.oracle_fold_runner import OracleFoldRunner
from core.wfo.folds import build_v3_folds
sig_eval = my_signal.evaluate({"H4": panel})
cfg = A1Config(config_id="arc_<id>", exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0)
raw_runner = ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=sig_eval, panels={"H4": panel})
oracle = OracleFoldRunner(signal_evaluation=sig_eval, panels={"H4": panel},
                          cluster_assignments=s2.cluster_assignments,
                          candidate_cluster_id=best_cid, trades=pool.trades)

# (5) validate — full honest WFO over IS folds + per-year OOS, then the discovery judge
from core.wfo.discovery_measure import (
    build_oos_year_folds, judge_all_folds_positive, run_config_over_folds,
)
is_folds = [f for f in build_v3_folds().folds if f.is_days >= 365]   # 2010-2020 search folds
oos_folds = build_oos_year_folds(start_year=2021)                    # per-year holdout 2021-present
is_verdict = judge_all_folds_positive(run_config_over_folds(raw_runner, is_folds, cfg))
oos_verdict = judge_all_folds_positive(run_config_over_folds(raw_runner, oos_folds, cfg))
# PASS the discovery judge  ⇔  is_verdict.all_folds_positive AND oos_verdict.all_folds_positive
```

Costs (FundedNext: 1.5× spread, 0.5 pip/fill slippage, $5/lot RT, no swaps), the SL-first
take-the-loss invariant and EET daily-DD bucketing are all applied *inside* the runner — the arc
never re-derives them.

---

## CANONICAL — LOCKED (call, never reimplement)

> Measurement apparatus. The gate IS this code. Calling only; FLAG, never patch, mid-run.

| Tool | What it does | Path | How to call |
|---|---|---|---|
| `Panel.from_pairs` | Load multi-pair OHLCV panel (real bid/ask, EET sessions, parquet cache) | `core/sim/panel.py` | `Panel.from_pairs(PAIRS, tf="H4", histdata_root=..., cache_root="data/cache", boundary_convention="5ers_eet")` |
| `build_arc_pool` + `ArcPoolConfig` | Ex-ante population build (the protocol's "ex-ante bounded population"); returns `ArcPool(.trades, .paths, .signal_evaluation, .pool_sha256, ...)` | `core/arc/arc_pool_builder.py` | `build_arc_pool(signal_module, {"H4": panel}, ArcPoolConfig(arc_name=..., sl_atr_mult=2.0, hold_bars=120, risk_pct=0.005, window_start=..., window_end=...))` |
| `run_step_2` | KMeans path-shape clustering (K∈2..6, silhouette pick); returns `Step2Result(.cluster_assignments, .centroids, .k_selected, ...)` | `core/steps/step_2_clustering.py` | `run_step_2(pool.trades, pool.paths)` |
| `run_step_3` | Per-cluster capturability + `is_candidate` flag; returns `Step3Result(.per_cluster, ...)` | `core/steps/step_3_capturability.py` | `run_step_3(pool.trades, pool.paths, s2.cluster_assignments, cluster_centroids=s2.centroids)` |
| `ArcFoldRunner` | Per-fold IS/OOS run → `FoldStats`; routes through the architecture → `MultiPairBacktester` → cost netting | `core/runners/arc_fold_runner.py` | `ArcFoldRunner(architecture=A1Architecture(), signal_evaluation=sig_eval, panels={"H4": panel})(fold, cfg)` |
| `OracleFoldRunner` | Oracle-best-cluster CEILING (perfect-hindsight cluster membership) → `FoldStats`; diagnostic upper bound, NOT deployable | `core/runners/oracle_fold_runner.py` | `OracleFoldRunner(signal_evaluation=sig_eval, panels={"H4": panel}, cluster_assignments=..., candidate_cluster_id=..., trades=pool.trades)(fold, cfg)` |
| `A1Architecture` + `A1Config` | System-level rule-filter architecture (no ML training). Other wired: A2 classifier, A3/A4 pipeline, A6 meta-labeling | `core/architectures/` | `A1Config(config_id=..., exit_policy="sl_partial_close_1r_runner_trail", sl_atr_mult=2.0, risk_pct=0.005, ...)` |
| `build_v3_folds` + `Fold` | 11 expanding-IS search folds (1-yr OOS each, 2010-2020) + 1 locked holdout | `core/wfo/folds.py` | `[f for f in build_v3_folds().folds if f.is_days >= 365]` |
| `build_oos_year_folds` | **Per-year OOS folds (2021-present)** — IS pinned to the 2010-2020 dev window, one fold per holdout year (the discovery OOS judge set) | `core/wfo/discovery_measure.py` | `build_oos_year_folds(start_year=2021)` |
| `judge_all_folds_positive` | **The DISCOVERY judge** — all-folds-positive (every fold ROI > 0); returns `DiscoveryVerdict`. SEPARATE from the L_PROTOCOL dual-tier gate | `core/wfo/discovery_measure.py` | `judge_all_folds_positive(fold_stats_seq)` |
| `run_config_over_folds` | Thin loop: run one config across folds via the canonical runner → `tuple[FoldStats]` | `core/wfo/discovery_measure.py` | `run_config_over_folds(runner, folds, cfg)` |
| `run_search` | Multi-config / multi-fold search orchestrator (wraps a per-fold runner; ranks via the L_PROTOCOL gate) — use when sweeping configs | `core/wfo/orchestrator.py` | `run_search(structure, candidates, fold_runner, min_is_days=365, top_k=3)` |
| `build_fold_stats_from_run` + `FoldStats` | **Cost chokepoint** — nets FundedNext costs (`apply_cost_model`) and emits per-fold `FoldStats(roi_pct, max_dd_pct, n_trades, days_breaching_daily_5pct, roi_dd_ratio, fold_id)` | `core/runners/_fold_stats_helpers.py` (+ `core/wfo/gates.py`) | called *inside* the runner; arcs read the returned `FoldStats` |
| `CostModel.fundednext()` | The gate-default broker profile (1.5× spread, $5/lot RT, 0.5 pip/fill, swaps off). `CostModel.zero()` only for explicit diagnostics | `core/sim/costs/model.py` | default — do not pass `zero()` for a gate |
| `MultiPairBacktester` | **The sole SL-honest gate engine** (bar-by-bar, SL-first take-the-loss). Reached via the architecture; arcs do not instantiate it directly | `core/sim/multipair_backtester.py` | (via `A1Architecture.run` inside the runner) |
| `build_exit_policy` (registry) | SL-honest exit-policy registry. Names: `sl_only`, `sl_plus_tp_2r`, `sl_plus_tp_3r`, `sl_plus_trailing_atr`, `sl_plus_trailing_swing`, `sl_partial_close_1r_runner_trail` | `core/sim/exit_policies/_registry.py` | pass the name as `A1Config(exit_policy="...")` |
| `SignalModule` / `SignalEvaluation` / `PerPairSignalState` | The signal contract every arc signal conforms to (the arc's signal is an EXPERIMENT tool, but it must implement this LOCKED Protocol) | `core/arc/signal_protocol.py` | implement `signal_name`, `primary_tf`, `causal_lineage`, `evaluate(panels) -> SignalEvaluation` |

---

## BUILT — CC-created reusable experiment tools (add + reuse freely)

> Filters, clusterers, transforms, exits, signals, soundness controls. CC builds these freely; a
> bug fails the WFO loudly. **No human gate to build one** — but it must be committed and registered
> so it compounds.

**Usage rule (do this every arc):**
1. **Before** building any filter / clusterer / transform / exit / signal / null-baseline, **check
   this section.** If it exists, **call it** (the row points at the script).
2. If it does not exist, **build it under `discovery/tools/`** (committed — persists and is reusable,
   NOT in per-arc scratch), use it in the arc, and **APPEND a row here at arc end** with: name |
   what it does | path | how to call | which arc created it.

This compounds reusable tooling the same way the log + LESSONS compound knowledge.

| Tool | What it does | Path | How to call | Created by |
|---|---|---|---|---|
| `build_null_signal_evaluation` | Random-entry NULL baseline (soundness control). Builds a random `SignalEvaluation` with the SAME per-pair fire-count as a real eval, placed at random eligible bars (≥ warmup, next bar exists), deterministic seed; reuses the real ATR so only entry TIMING is randomized. **Direction-aware (arc 2013): carries `direction` through, so a SHORT signal's null is a SHORT random entry — a fair same-side baseline; longs byte-identical (`direction` defaults LONG).** EXPERIMENT part = mask randomization ONLY; scoring stays canonical (run the returned eval through `ArcFoldRunner`). Never realizes P&L itself. | `discovery/tools/null_entry_baseline.py` | `build_null_signal_evaluation(real_eval, seed=42, warmup=100)` → run through `ArcFoldRunner` + `run_config_over_folds` | arc 1000 (dir-aware: arc 2013) |
| `DonchianBreakoutLongSignal`, `PeriodicLongSignal` | Trend-entry `SignalModule`s (mask + ATR geometry ONLY, never realize P&L). `DonchianBreakoutLongSignal(lookback, spacing_bars, sma_filter)` = long on a fresh N-bar-high break (canonical TSMOM/trend entry; ex-ante Donchian shift1, crossing-bar-only + spacing). `PeriodicLongSignal(period, warmup)` = time-random unconditional long base for MFE-distribution / convexity comparison. Both conform to the LOCKED `SignalModule` Protocol; feed to `build_arc_pool` / `ArcFoldRunner` (scoring stays canonical). | `discovery/tools/trend_entry_signals.py` | `DonchianBreakoutLongSignal(lookback=120).evaluate({"H4": panel})`; `PeriodicLongSignal(period=30)` | arc 2000 |
| `make_time_exit_predicate` | N-bar TIME-EXIT signal-class `ExitPredicate` for calendar/hold-based signals (exit by time, not +1R). Closes a position `n_bars` after entry at the exit bar's bid(long)/ask(short), dispatching by `position.pair`. GEOMETRY/TIMING ONLY — the canonical engine realizes P&L under take-the-loss (SL checked intra-bar BEFORE this close-of-bar predicate). Set on each pair's `PerPairSignalState.exit_predicate`. Exists because `A1Config.time_exit_bars` is NOT wired into the Order by A1 (FLAG, arc 1005). | `discovery/tools/time_exit_predicate.py` | `make_time_exit_predicate({p: panel.pair_dfs[p] for p in pairs}, n_bars=6)` → set as `PerPairSignalState(exit_predicate=...)` | arc 1005 |
| `WeekendGapFillLongSignal` | Long a significant weekly-open DOWN gap, betting on reversion toward the prior close (mask + ATR geometry ONLY, never realizes P&L). Flags the weekly-open bar via a > `gap_hours` index gap; fires when `(open_mid - prior_close_mid)/ATR <= -threshold_atr` (ex-ante: open[i] & close[i-1] known at bar i close, ATR shift1; entry next bar). Conforms to the LOCKED `SignalModule` Protocol; feed to `build_arc_pool` / `ArcFoldRunner`. NOTE (arc 1006+2001 convergence): gap-fill is mean-POSITIVE on JPY crosses, mean-NEGATIVE on majors — prefer crosses. | `discovery/tools/gap_signals.py` | `WeekendGapFillLongSignal(threshold_atr=1.0, gap_hours=20).evaluate({"H4": panel})` | arc 2001 |
| `WeekendUpGapShortSignal` | SHORT a significant weekly-open UP gap, betting on reversion (fill) DOWN toward the prior close — the direction-mirror of `WeekendGapFillLongSignal` (mask + ATR geometry + `Direction.SHORT` ONLY; never realizes P&L). Same ex-ante weekly-open detection/ATR; fires when `(open_mid - prior_close_mid)/ATR >= +threshold_atr`; declares `Direction.SHORT` on the per-pair state + eval so the canonical Step-1 pool + architecture emit a short (entry next bar, SL ABOVE entry, `final_r` short-signed). Conforms to the LOCKED `SignalModule` Protocol. **arc 2013 finding: KILL (converges w/ independent 1000s arc 1016). First end-to-end SHORT engine run — the merged short path builds sign-correctly, no canonical change. The engine mean-positive (+0.745% IS thr 1.0 trailing_atr) is THIN REGIME-LUCK: excluding 2018(+5.08,n8)+2019(+5.92,n10) the other 8 folds avg −0.44%; entry coin-flip-to-adverse (1016: cap 0.448/drift −0.093), ≥0.5 median-neg, ≥1.5 inverts. The up-gap "stronger leg" (arc 2001) was hindsight gap-bar-OPEN; honest i+1 short is dead. Tool kept (valid reusable short signal).** Intended TF=H4, JPY crosses. | `discovery/tools/gap_signals.py` | `WeekendUpGapShortSignal(threshold_atr=1.0, gap_hours=20).evaluate({"H4": panel})` | arc 2013 |
| `make_price_target_exit_predicate` | Structural price-TARGET exit `ExitPredicate` (+ N-bar time fallback) for "revert-to-a-level" signals (gap-fill, mean-reversion-to-band). LONG exits AT a per-trade target price when `high_bid >= target` (and `target > entry`), else at the time fallback. GEOMETRY ONLY; engine realizes P&L SL-first (only fires on stop-surviving bars → target fill is stop-free). Signal supplies a per-pair `{entry_ts → target_price}` map. NOTE (arc 1007): capping at a reversion target can KILL an overshoot edge — verify the edge is reversion-to-level, not continuation, before using. | `discovery/tools/price_target_exit_predicate.py` | `make_price_target_exit_predicate(pair_frames, targets_by_pair, n_bars_max=24)` → set as `PerPairSignalState(exit_predicate=...)` | arc 1007 |
| `MonthEndReversionLongSignal` | Long a big DOWN move into month-end, betting on the post-fix mechanical-rebalancing reversion UP (mask + ATR geometry ONLY, never realizes P&L). Fires at the last trading day of the month (bar i+1 in a new calendar month — ex-ante calendar knowledge, arc-1005 convention) when the move into it `(close_mid[i]-close_mid[i-into_bars])/ATR <= -threshold_atr`; entry next bar (first trading day of next month). ATR Wilder(14) mid shift1. Conforms to the LOCKED `SignalModule` Protocol; feed to `build_arc_pool` / `ArcFoldRunner`. NOTE (arc 1011): mechanism-CONTROLLED (month-end vs random-day +0.249 ATR excess; generic reversion is dead) → PORTFOLIO (mean-positive, not all-folds-positive); intended TF=D1, USD majors, ~2-bar time exit. | `discovery/tools/month_end_signals.py` | `MonthEndReversionLongSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": panel})` | arc 1011 |
| `make_low_cost_mask` | LOW-COST-regime entry mask — attacks EDGE<COST from the **cost side**. Returns a per-pair boolean Series, True where the bar's spread/ATR (cost in ~R units: a spread S costs ~S/(sl_mult·ATR) in R) is ≤ its own TRAILING rolling `quantile` (causal: spread_close known at the signal bar, ATR shift1, rolling-quantile threshold shift1 so a bar never enters its own threshold). AND it into a `SignalModule`'s per-pair `signal_mask` to restrict an entry to its cheapest bars; the engine still applies the REAL (now lower) per-bar cost. EXPERIMENT (mask/observation only; never realizes P&L). **arc 2005 finding: cutting cost is a real ~10pp net lever on a +gross-drift entry (beats a matched random-cheap null) but does NOT rescue a directional edge OOS — apply to a signal with a DURABLE gross edge.** | `discovery/tools/cost_regime_mask.py` | `make_low_cost_mask(panel, pairs, quantile=0.30, window=250)` → AND into `PerPairSignalState.signal_mask` | arc 2005 |
| `combine_fold_roi` (`fit_weights`, `combine_fold_rois`, `rois_from_fold_stats`, `CombinedBook`) | PORTFOLIO fold-ROI combiner — linear-combines the per-fold ROI of ≥2 ALREADY-SCORED components (each a tuple of canonical `FoldStats` from `ArcFoldRunner`/`run_config_over_folds`) into one combined-book per-fold ROI series, so `judge_all_folds_positive` can gate the COMBINED book (§6/§11). Modes: `"equal"` (naive capital weight) / `"risk_parity"` (inverse-fold-vol `w_k∝1/σ_k`). **No-lookahead: fit weights ONCE on IS (`fit_weights`), FREEZE for OOS — never recompute weights on OOS.** EXPERIMENT tool (linear combination of canonical numbers; never realizes P&L). **LIMITATION:** per-fold linear combination, not a single co-simulated equity curve — faithful first-order only for DISJOINT-universe / disjoint-event-timing components (else FLAG + prefer co-sim). **arc 2006 finding: 2 near-zero-corr (corr +0.117) mean-positive flow-reversion components could NOT make an all-folds-positive book — blocked by a mutually-negative fold (2015) + tail-correlation; select a 3rd by its ROI on the existing book's NEGATIVE folds, not avg-corr.** | `discovery/tools/combine_fold_roi.py` | `w = fit_weights([A_is_rois, B_is_rois], "risk_parity")`; `combine_fold_rois([A_rois, B_rois], w).combined_roi` → check `all(r>0)` | arc 2006 |
| `MonthEndFixReversionLongSignal` | Long an abnormal DOWN push into the **month-end 16:00 London WM/Reuters fix**, betting on post-fix reversion (mask + ATR geometry ONLY, never realizes P&L). Fires at the 15:00–16:00 London bar (DST-robust via Europe/London tz) on the last weekday of the month when `(close_mid−open_mid)/ATR <= −threshold_atr`; ATR=Wilder(14) MID shift1; engine enters next (post-fix) bar. Conforms to the LOCKED `SignalModule` Protocol; feed to `build_arc_pool`/`ArcFoldRunner`. NOTE (arc 3008): the fix reversion is REAL (beats placebo+null) but SUB-COST on H1 majors — KILL, not a portfolio edge. | `discovery/tools/fix_flow_signals.py` | `MonthEndFixReversionLongSignal(threshold_atr=0.5).evaluate({"H1": panel})` | arc 3008 |
| `SweepReclaimReversalLongSignal` | Long a fast sweep-and-reclaim of a prior swing low in an uptrend (liquidity-grab / Wyckoff-spring; mask + ATR geometry ONLY, never realizes P&L). Fires when MID low pierces the prior `swing_lookback`-bar low (shift1) AND close reclaims above it AND a fast 3-bar drop in (`drop3 <= -drop3_atr`) AND (optional) close>SMA(`sma_period`); ATR=Wilder(14) MID shift1; engine enters next bar open. Conforms to the LOCKED `SignalModule` Protocol. NOTE (arc 2007): the LONG is KILL — cap lifts to 0.52 but loses to the fair null (buys adverse-excursion weakness, worse than random under take-the-loss); the CLIMAX (big-range) variant is a falling-knife → the SHORT edge (FLAG-1, shorts-gated). Reusable for sweep/spring structural tests incl. the shorts-enabled climax-sweep short. | `discovery/tools/sweep_signals.py` | `SweepReclaimReversalLongSignal(swing_lookback=20, drop3_atr=1.0, sma_period=200).evaluate({"H4": panel})` | arc 2007 |
| `FailedBreakdownReclaimLongSignal` | Long a deep failed-breakdown RECLAIM (stop-run reversal) at a K-bar swing low (mask + ATR geometry ONLY, never realizes P&L). Fires at bar i when low_bid pierces the prior K-bar min low (`low_bid.shift(1).rolling(K).min()` — swept stops), close_mid reclaims above it (failed breakdown), AND the lower rejection shadow `(min(open_mid,close_mid)-low_bid)/ATR >= min_shadow_atr` (deep grab). ATR Wilder(14) MID shift1; engine enters next bar. Conforms to the LOCKED `SignalModule` Protocol. **arc 1013 finding: the STRONGEST + cleanest directional edge in the corpus → PORTFOLIO (3rd component): IS mean +1.85% (9/10), OOS +0.94%, beats a NEGATIVE same-exit null by +2.96pp; structure control-proven load-bearing (same-wick-elsewhere = coin-flip); robust K∈{40,60}×shadow≥1.25, LOO all+. NOT all-folds-positive (2018 strong-USD regime). Positive in 2015/16/20 → the arc-2006 regime-orthogonal 3rd leg.** Intended TF=H4, USD majors. | `discovery/tools/failed_breakdown_signals.py` | `FailedBreakdownReclaimLongSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": panel})` | arc 1013 |
| `FailedBreakoutRejectionShortSignal` | SHORT a deep failed-breakout rejection (stop-run reversal) at a K-bar swing HIGH — the short MIRROR of `FailedBreakdownReclaimLongSignal` (mask + ATR geometry ONLY, `direction=Direction.SHORT`, never realizes P&L). Fires at bar i when high_ask pierces the prior K-bar max high (`high_ask.shift(1).rolling(K).max()` — swept buy-stops above), close_mid rejects back below it (failed breakout), AND upper rejection shadow `(high_ask − max(open_mid,close_mid))/ATR >= min_shadow_atr`. ATR Wilder(14) MID shift1; engine enters next bar (sell open_bid). Conforms to the LOCKED `SignalModule` Protocol. **arc 3011 + 2011 (independent reproductions) finding: KILL — no robust short edge.** The pooled structure control LOOKS like a pass (AT-swept drift +0.278 vs −0.171 elsewhere) but is a thin-tail/pair-mix CONFOUND (median −0.069; carried by AUDUSD/USDJPY; negative excluding them — arc-2009 USD-quote-beta tell). Capture 0.473<0.50 (vs 1013 long 0.55–0.61). On the engine (first short to reach it): mean final_r +0.0102R≈0, 0/18 exit·SL cells all-folds-positive, beats null only +0.021pp, NEGATIVE in 2018. arc 1013's reclaim-long has NO short mirror across all 3 constructions (1014/2009/2011·3011); capture is the wall. Intended TF=H4, USD majors. | `discovery/tools/failed_breakout_signals.py` | `FailedBreakoutRejectionShortSignal(swing_lookback=40, min_shadow_atr=1.25).evaluate({"H4": panel})` | arc 3011 |
| `CarryUnwindCascadeShortSignal` | SHORT the vol-expansion big-red IGNITION bar of a carry-unwind cascade on a JPY carry cross (mask + ATR geometry ONLY, `direction=Direction.SHORT`, never realizes P&L). Fires when `close_mid > SMA(sma_period)` AND `SMA` rising/20 (carry built up) AND `TR/ATR ≥ vol_ignition` (vol-expansion) AND `(open_mid−close_mid)/ATR ≥ down_atr` (big-red body). ATR Wilder(14) MID shift1; engine enters next bar (sell open_bid). Conforms to the LOCKED `SignalModule` Protocol. **arc 1017 finding: KILL (real-but-sub-cost + UN-SCALABLE).** The mechanism is real (structure control passes: in-carry-uptrend drift +0.134 ATR/median +0.068 = continues down vs same big-red bar elsewhere median −0.178 = reverts up) but capture is coin-flip 0.50, drift on the JPY-cross cost line, deeper cell inverts (2011/3011 tell); honest engine §5f best exit sl_only mean +0.013% / 4/10 / not all-folds-positive, beats null only +0.044pp (noise floor). KEY: all-JPY-quote crosses fire simultaneously in one risk-off cascade → the 2-per-currency exposure cap guts the clustered fires (uncapped 92–109/yr → capped 0–3/yr 2015/16/19) → a correlated-cascade signal is structurally un-scalable into a portfolio leg. 2018 genuinely engine-positive (AUDJPY +0.07–0.10%) but tiny/un-scalable. Intended TF=H4, JPY carry crosses. | `discovery/tools/carry_unwind_signals.py` | `CarryUnwindCascadeShortSignal(sma_period=100, vol_ignition=1.5, down_atr=1.0).evaluate({"H4": panel})` | arc 1017 |
| `MonthEndReversionShortSignal` | SHORT a big UP move into month-end, betting on the post-fix rebalancing reversion DOWN — the direction-MIRROR of `MonthEndReversionLongSignal` (mask + ATR geometry + `Direction.SHORT` ONLY; never realizes P&L). Same ex-ante month-end detection / into-move / ATR (last-trading-day flagged by next bar being a new month; ATR Wilder(14) MID shift1); fires when `(close_mid[i]−close_mid[i−into_bars])/atr >= +threshold_atr`; declares `Direction.SHORT` so the canonical pool + architecture emit a short (entry next bar, SL ABOVE, `final_r` short-signed). Conforms to the LOCKED `SignalModule` Protocol. **arc 3017 finding: KILL — real-but-sub-cost + un-scalable.** FIRST short in the corpus to clear >0.50 capture (0.5508) WITH a passing structure control (+0.0996 ATR month-end excess; generic big-up continues up) and lean 2015/2018-positive (confirms the month-end mechanism is direction-symmetric, strengthens 1011) — but engine §5f 0/9 exits all-folds-positive (best 7/10), mean +0.007%, beats fair null only +0.012pp (noise floor); drop-top-2-pairs collapses capture to 0.4605 (arc-2011 tell); gross +0.171R but net≈0 because winning fires cluster on simultaneous USDXXX month-end up-extensions in strong-USD years → the 2-per-USD exposure cap erodes them (arc-1017 mode). Intended TF=D1, USD majors. | `discovery/tools/month_end_signals.py` | `MonthEndReversionShortSignal(threshold_atr=1.0, into_bars=2).evaluate({"D1": panel})` | arc 3017 |
| `observe_long_capture` | Step-(b) per-bar hypothetical-LONG OBSERVATION harness (re-rolled by arcs 1000–1006 + chat 3001's drift lens): honest +1R-before-SL CAPTURE (take-the-loss `reached_1r_before_sl`) + forward-drift in ATR, for every bar (or a `restrict` mask). Now **direction-aware** — `direction="long"` (default; byte-identical to the original long lens) or `"short"` (mirrors entry/SL/label/drift-sign). Matches `build_arc_pool`'s entry/SL convention so the unconditional capture predicts the pool (verified: H4-majors 0.4877). Returns tidy `DataFrame[pair, signal_time, capture, fwd_drift_atr, atr]`; the arc joins its conditioning cols + groups. **CHARACTERIZATION ONLY — gross, NOT cost-aware, NOT a gate** (the gate is `MultiPairBacktester` via `ArcFoldRunner`); a pre-pool screen, never a verdict. | `discovery/tools/observe_long_capture.py` | `observe_long_capture(panel, sl_mult=2.0, hold=120, drift_bars=24, direction="long")` → join conditioning → groupby | post-1007 refactor |

**First expected entry — the random-entry NULL baseline.** The council mandates a random-entry
soundness control (does the real signal beat random entry under the same exit?). No canonical or
reusable one exists (Arc 0 hand-rolled it twice in scratch; the only prior is retired-era under
`attic/`). The first arc that needs it builds it under `discovery/tools/` — it constructs a random
`SignalEvaluation` (random masks at a matched fire-rate, deterministic seed) and runs it through the
**canonical** `ArcFoldRunner` (the apparatus stays canonical; only the random-mask generation is the
experiment part) — then registers it in the table above.

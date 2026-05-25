# Signal spec — liquidity sweep + reclaim long (v0.1)

> **Family:** level-defined reversal (long only)
> **Primary TF:** 4H
> **Provenance:** copied from `docs/archive/arc_results/ARC_7_RESULT.md` Trigger
> section (v2.1.2 closure, 2026-05-17). v3.0.1 lands the spec at the
> conventional location for forward-arc consumption.
> **Arc:** Arc 7 v3.0.1 (`results/l_arc_7/`)
> **Implementation:** `core/strategies/liquidity_sweep_reclaim_long/signal_module.py`

## Trigger (bar-N close, ex-ante)

Let `swing_low_N = min(low_bid[t-N..t-1])` with `N=20`, and `ATR(14)[t]` =
Wilder ATR(14) on bid-side OHLC at bar t. The signal fires at bar t's close
when ALL of the following hold:

1. `low_bid[t]  < swing_low_N`                              — sweep
2. `close_bid[t] > swing_low_N`                              — reclaim
3. `swing_low_N − low_bid[t] ≥ 0.25 × ATR(14)[t]`            — magnitude threshold
4. `close_bid[t] > open_bid[t]`                              — bullish reclaim bar
5. `(close_bid[t] − swing_low_N) / (swing_low_N − low_bid[t]) ≥ 0.5` — reclaim strength
6. `gap_since_last_signal_on_pair ≥ 20 bars`                 — per-pair refractory

Entry executes at bar t+1's `open_ask` (next-bar open, long fill primitive
from `core.sim.fill.long_entry_fill_price`).

## Causal lineage

`clean` — every input at bar t is known by t's close. Producer-level trace:

- `swing_low_N` — `low_bid.rolling(N).min().shift(1).iloc[t]`. pandas
  semantics: `rolling(N).min()` at index t returns min over [t-N+1, t];
  `.shift(1)` shifts the entire series forward by one index, so the value
  AT index t becomes the value previously at index t-1. Net effect:
  `swing_low.iloc[t] = min(low_bid.iloc[t-N], ..., low_bid.iloc[t-1])`.
  `low_bid.iloc[t]` is NEVER in the window. No centred-window pattern (no
  Arc 9 lookahead). No right-edge confirmation lag.
- `ATR(14)[t]` — Wilder EWM on shifted true range. TR at bar t uses
  `high[t]`, `low[t]`, `close[t-1]` — all known by bar t's close.
- `low_bid[t]`, `close_bid[t]`, `open_bid[t]` — current bar; known at bar
  t's close.
- `gap_since_last_signal_on_pair` — stateful walk over prior signal
  indices on the same pair only; strictly historical.

No cross-pair, multi-TF, or future-bar dependency.

Step 1 integrity report's lookahead spot-check is elevated to 10 random
trades (vs default 5) for Arc 7 v3.0.1 with explicit causal-lineage trace
per trade. Step 6 §6.1 lookahead audit (Amendment 4) exercises this
producer via byte-compare from raw OHLC on a random trade sample.

## Default arc-config (Step 1 simulation)

- SL: `2.0 × ATR(14)[t]` measured from the signal-bar `close_ask` (v3
  convention via `core.arc.arc_pool_builder._simulate_pair_pool`)
- Forward window: 240 bars (time-exit cap)
- Direction: long only
- Bar boundary: `5ers_eet` per PR #189 (was UTC pre-amendment)
- Step 1 features: 27-feature catalogue with mid-price computation per PR #189
  (signal evaluation stays bid-side per KH-24 convention; mid-price applies
  to FEATURES only per `docs/PROTOCOL_RUNTIME.md` §15.1)

## Default architecture-search ranges (Step 5)

Step 5 searches per dispatch §"Step 5":
- SL multiplier: Step-3-selected SL ± 1 step from {1.5, 2.0, 2.5, 3.0, 3.5, 4.0}
- Exposure cap: `max_concurrent_per_currency` ∈ {2, unlimited}
- Trail: {enabled (act=2.0, dist=1.5), disabled} → `sl_plus_trailing_atr` / `sl_only`
- A3 only: `n_defer` ∈ {3, 5}
- A4 only: `exit_threshold` ∈ {0.3, 0.4, 0.5}; trail-first precedence per PR #186
- A6 only: `(lower, upper)` ∈ {(0.3,0.5), (0.4,0.6), (0.5,0.7)}

Architecture set per archetype follows the dispatch §"Step 5" mapping:
- V-shape recovery → A1, A3, A6
- Stepwise climber → A1, A2, A4
- Bimodal → A1, A4
- Monotonic up → A1, A2, A6
- Choppy → skip
- A5 if ≥ 2 candidate clusters survive Step 3

## ML mechanics (Amendment 2)

- A2: reuses Step 4's best classifier + AUC-best threshold per
  `core.steps.classifier_persistence.build_a2_config_from_step4`. NO retrain.
- A3: NEW classifier per fold via
  `core.steps.path_classifier_per_fold.build_path_classifier_fits_per_fold`.
  Features = path-so-far at bar N. RF defaults from L_PROTOCOL Appendix A.
- A4: NEW classifier per fold. Target = final R > 0. Features = path-so-far
  each bar. Initial SL still applies. Trail takes precedence over classifier
  exit per PR #186 (locked in `core/sim/multipair_backtester.py`).
- A6: reuses Step 4's best classifier; confidence → size mapping.
- All ML: causal lineage tags binding. `random_state=42`, `n_jobs=1`.

## Historical reference

- **Arc 7 v2.1.2** (2026-05-17, archive): CLEAN-NULL at Step 4. Step 3
  PASSED (3 V-shape units survived §2 conjunctively); Step 4 best AUC 0.536
  (vs 0.65 gate). Pre-CC_07 engine, no Step 4 classifier persistence, no
  Amendment 3 risk-normalised gates, no Step 6 framework.
- **Arc 7 v3.0** (2026-05-22, never merged): FAIL realised + ORACLE
  PASS-DEPLOYABLE both candidate clusters (c0 worst-fold ratio 6.85,
  c1 6.34) + c0 Step 4 AUC 0.6758 cracked the 0.65 bar. A2/A6 unwireable
  (Step 4 classifier persistence missing); Amendment 3 risk-normalised
  gates unimplemented. Engine gaps fixed in PRs #185, #186, #188, #189
  — v3.0.1 is the canonical re-run.

Per the v3.0.1 dispatch: v2.x and v3.0-first-attempt numbers are NOT
targets. v3.0.1 evaluates the signal through L_PROTOCOL v3.0 +
Amendments 1-4 with no preconception.

## v0.1 immutability

This spec is locked at v0.1. Parameter tweaks within an arc are forbidden per
L_PROTOCOL §1 "within-arc thresholds do not move." Cross-arc spec evolution
would land here as v0.2 with a calibration-review event.

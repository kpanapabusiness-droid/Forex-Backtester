# PR-E Scope Review — KH-24 Strategy Prerequisite Gap

**Status:** HALT before any code lands. Surfacing a prerequisite mismatch in the PR-E dispatch for chat review.

This is **not** an anchor-reproduction failure (the dispatch's documented HALT path) — the v3 engine cannot produce KH-24 numbers yet because the KH-24 strategy logic itself does not exist in the v3 stack. Running the dispatch as written would require implementing ~1700-2000 LOC of new code first.

The honest read of the dispatch is: it assumes KH-24 is already wired up in v3 and just needs to be run on two WFO modes. That assumption doesn't hold.

---

## What v3 has (PR-A through PR-D, all merged)

| Layer | Module | Provides |
|---|---|---|
| Data | `core.data.histdata_loader` + `aggregator` | M1 bid+ask + parquet-cached aggregations |
| Spread | `core.spread.real_spread` | per-bar spread, tradability mask, DQ summary |
| Fill | `core.sim.fill` | 8 bar-level primitives (long/short × entry/market-exit/SL/TP) |
| Sim | `core.sim.account.Account` | balance, equity, max-DD, exposure caps (per-pair/per-currency/total) |
| Sim | `core.sim.multipair_backtester.MultiPairBacktester` | bar-by-bar driver, intra-bar SL/TP exits, deferred next-bar entries |
| Sim | `core.sim.panel.Panel` | multi-pair OHLC panel |
| WFO | `core.wfo.{folds,gates,orchestrator}` | both 7-fold KH-24-anchor + 11-fold v3 fold builders; §3 gates; top-K + holdout one-shot |
| Features | `core.features.*` | 27 features across 7 classes with causal lineage |
| Parallel | `core.parallel` | per-pair `multiprocessing.Pool`, deterministic aggregation |
| Determ | `core.determinism` | `RANDOM_STATE=42`, `seed_everything`, LF text writes |

The `MultiPairBacktester` driver accepts a strategy callable with this signature:

```python
StrategyFn = Callable[[pd.Timestamp, dict[str, pd.Series | None], Account], list[Order]]
```

Orders carry pair, direction, size, and **fixed** `sl_price` / `tp_price`. No trailing-stop state, no time-exit, no signal-driven exits beyond fixed SL/TP.

---

## What KH-24 actually requires

Per the dispatch's exact spec:

```yaml
signal: kb_exhaustion_bar (c1-c6, c8, c9; c7 disabled)
direction: long
timeframe_primary: H4
filters:
  - d1_regime_filter (one-day lag — each H4 bar sees prior calendar day's D1 close)
  - exposure_cap: 2 concurrent open positions
  - h1_cir: T = 0.28
pairs: 28 FX
entry: bar N+1 open after signal on bar N close
stop_loss: entry - 2.0 * ATR(14)
trailing_stop:
  activation: close >= entry + 2.0 * ATR
  trail_distance: 1.5 * ATR behind highest close
  update_frequency: bar-close only
exits: trailing_stop | kijun_d1 | stoploss
risk_per_trade: 1.0% of reset floor balance
spread: real per-bar bid/ask from HistData
```

Mapping each line to the v3 stack:

| KH-24 requirement | v3 has? | Gap (LOC est.) |
|---|---|---|
| Signal `kb_exhaustion_bar` c1-c6, c8, c9 evaluator on bid+ask schema | **Partial** — `signals/kb_exhaustion_bar.py` covers c1-c3 only; c4-c6, c8, c9 live in `scripts/arc_kh24_v2/step1/_signal.py` and both use MT5 single-OHLC schema, not v3's bid+ask | port + extend, ~250 LOC + tests |
| H4 primary TF aggregation | ✅ PR-A | 0 |
| D1 + H1 panels for filters | ✅ PR-A (just call `aggregate(..., 'D1')` / `'H1'`) | 0 |
| D1 regime filter (one-day lag) — boolean gate on D1 close vs Kijun(D1, 26) | ❌ — `core.features.multi_tf` has `d1_close_slope_*` but no regime-filter gate function | new module, ~50 LOC |
| Exposure cap of 2 concurrent positions | ✅ PR-B — `ExposureRules(max_concurrent_total=2)` | 0 |
| H1 CIR filter at T=0.28 (Choppiness Index Range) | ❌ — CIR is not in v3's feature set | new feature + filter, ~120 LOC |
| Bar-N+1 open entry with correct bid/ask fill | ✅ PR-B — driver does this via `_pending` + `long_entry_fill_price` | 0 |
| Hard SL = `entry - 2.0 * ATR(14)` at entry price | ✅ — caller computes the SL price, driver enforces | 0 |
| Trailing stop: activates at `close >= entry + 2.0 * ATR`, then 1.5 × ATR behind highest close, bar-close updates only | ❌ — `Position` has fixed `sl_price`; no trailing-stop state machine | engine extension: `Position` mutable trail state, `MultiPairBacktester._check_exits` trail update, new tests. ~200 LOC |
| `kijun_d1` exit (close trade when D1 close crosses Kijun_D1) | ❌ — no signal-driven exit hook in driver | extension to `MultiPairBacktester` for caller-supplied exit predicate. ~100 LOC |
| Risk sizing: 1% of reset floor balance | ❌ — `Account.open` takes raw `size`; no 5ers reset-floor accounting | new sizing helper + reset-floor model. ~80 LOC |
| KH-24 strategy assembly tying signal + 3 filters + trailing + kijun_d1 + risk + multi-TF panels into one `StrategyFn` callable | ❌ — none of this exists | ~250 LOC + tests |
| `fold_runner` that wraps the strategy + computes per-fold `FoldStats` for the WFO orchestrator | ❌ — `core.wfo.orchestrator.run_search` takes `fold_runner` from caller; KH-24's isn't written | ~150 LOC |
| YAML config for the locked KH-24 spec + 7-fold and 11-fold anchor runs | ❌ | ~50 LOC + 2 configs |
| Anchor reproduction script + per-fold table writer | ❌ | ~150 LOC |

**Total new code estimate: ~1700-2000 LOC**, plus ~600 LOC of tests. The trailing-stop engine extension alone is a meaningful change with its own ordering / state-management subtleties (activation lag, multiple updates per bar, intra-bar vs bar-close logic).

---

## What the "+1.28% real-spread band" actually is

`ARC_HISTORY.md` documents this number, and the dispatch references it as an acceptable result. Reading carefully:

> Arc 4's HistData spread audit triggered a real-spread reconciliation against KH-24. KH-24 doesn't load `spread_floors_5ers.yaml` — it uses raw MT5 per-bar spreads. Audit-window overlap (2024-01 → 2026-01): 69 of 553 trades evaluated, mean under_R 0.02187, total under_pct_equity 1.509%. Fold 7 published +1.92% → ~+1.28% under real-spread reconciliation.

This was an **analytical reconciliation** — per-trade spread delta applied to 69 of 553 trades in the audit window overlap. It was **not** a v3 backtester run. The +1.28% is an extrapolation from delta-on-trades, not an actual end-to-end run with the new engine.

Implication: even with KH-24 fully wired up in v3, the v3 run will not exactly reproduce +1.28% either — that number is also an estimate. The "documented band" is +1.28%-+1.92% only as a rough acceptance window.

---

## Three paths forward

### (A) Split PR-E into PR-E.1 + PR-E.2 — **recommended**

- **PR-E.1: KH-24 strategy implementation in v3.** Port `kb_exhaustion_bar` to bid+ask, add D1 regime filter, H1 CIR filter, trailing stop (engine extension), kijun_d1 exit, risk sizing. ~1500-1700 LOC + tests. Self-contained, reviewable in isolation. CI green.
- **PR-E.2: Anchor reproduction + final docs.** Mechanical: run KH-24 through both WFO modes, compare to published, write `BACKTESTER_ARCHITECTURE.md` results section, finalise `features_reference.md` + `DATA_FOUNDATION.md` + `README.md`. ~200-400 LOC of orchestration + the actual compute runs. This becomes the gate originally specified.

Cost: 1 extra PR in the cadence (5 → 6). Benefit: each PR is reviewable; if PR-E.1 has a bug the cause is contained to engine wiring, not "anchor reproduction failure" with ambiguous source.

### (B) Single mega-PR with everything

Implement strategy + anchor + docs all in PR-E as originally scoped. ~2000-2500 LOC including tests. Hard to review; bug surface is wider.

If anchor doesn't match within tolerance, the diagnostic burden is high — "is the engine wiring wrong, or did real-spread bite us harder than expected?"

### (C) Lower the bar — stub KH-24 + document deviations

Implement only the c1-c3 signal already in `signals/kb_exhaustion_bar.py` (no c4-c9, no filters, no trail, no kijun_d1, no risk sizing — just signal + fixed SL/TP). Run that against the v3 engine. Document the deviations from published KH-24.

This produces SOMETHING but it's not KH-24 — call it "KH-24 stub" or "signal-only KH-24." It will not match published numbers because the deployed system's filters and exits are load-bearing. Cheap but not informative; mainly proves the v3 stack end-to-end runs without crashing.

---

## My recommendation

**(A) — PR-E.1 + PR-E.2 split.** Reasons:

1. The dispatch's "discipline rules" say "NO scope creep" — but implementing KH-24 strategy IS the scope of a separate PR, not scope creep. Calling it out keeps the actual PR-E mechanical-and-reproducible as originally intended.
2. PR-E.1's tests would assert that the strategy implementation matches the published `signals/kb_exhaustion_bar.py` c1-c3 logic byte-identically on overlap, and that the new c4-c9 logic matches `scripts/arc_kh24_v2/step1/_signal.py` (which is already in active scripts/). That's a clean diff-against-reference test, separable from anchor reproduction.
3. PR-E.2 becomes a pure compute + documentation PR — exactly the mechanical anchor-reproduction step the dispatch intended.

I'm not going to start coding PR-E.1 unilaterally — that's a chat call on whether to split or to push everything into PR-E.

---

## What I have NOT done

- No new code lands in this branch yet
- No protocol changes
- No changes to KH-24's deployed configuration
- `infra/backtester-v3-pr-e` branch contains only this single doc file

If chat picks (A), I'll start PR-E.1 with KH-24 strategy implementation. If (B), I'll bundle everything into the current PR-E. If (C), I'll stub and document.

End turn. Awaiting chat direction.

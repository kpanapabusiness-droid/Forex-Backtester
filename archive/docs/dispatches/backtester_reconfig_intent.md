# Backtester v3.0 Reconfiguration — Intent

Dispatch source: `CC_06_BACKTESTER_RECONFIG.md` (read from `~/Downloads/`).
Branch: `claude/agitated-hellman-c92fb6` (worktree), targeting `main` via PR per DoD #11.
Prereqs verified: HistData M1 layer in place (52 GB tick, 18 GB M1, integrity 0 mismatches per [DATA_FOUNDATION.md](docs/DATA_FOUNDATION.md)); CC_05 v3.0 landing merged (`612d3d7`); [L_PROTOCOL.md](L_PROTOCOL.md) v3.0 locked.

---

## Read-first findings (current state)

### 1. Backtester entry points

| Lineage | Path | LOC | Role |
|---|---|---:|---|
| KH-era engine | [core/backtester.py](core/backtester.py) | 2,011 | indicator-cache + WFO runner. Bound to MT5 OHLCV schema. |
| KH-era helpers | [core/backtester_helpers.py](core/backtester_helpers.py) | 388 | trade row finalisation |
| Path-D1 pipeline | [core/d1_pipeline.py](core/d1_pipeline.py) | 590 | path-so-far hook (Arc 4/5/6) |
| Exit policies | [core/exit_policies.py](core/exit_policies.py) | 186 | trail / kijun / stop |
| Signal logic | [core/signal_logic.py](core/signal_logic.py) | 630 | KH-era continuation logic |
| KH-24 v2.0 step1 | [scripts/arc_kh24_v2/step1/run_step1.py](scripts/arc_kh24_v2/step1/run_step1.py) | — | newer L_ARC v2.x Step-1 runner (bare KH-24 signal, MT5 schema) |
| KH-24 WFO runner | [scripts/phase_kgl_v2_4h_wfo.py](scripts/phase_kgl_v2_4h_wfo.py) | — | full KH-24 WFO with spread floor (MT5 schema) |
| L_char atlas | [scripts/lchar/run_layer4.py](scripts/lchar/run_layer4.py) (+ 1/2/3/5) | — | descriptive atlas (separate lineage) |

### 2. Data loader

[core/utils.py:259](core/utils.py:259) — `load_pair_csv(pair, data_dir) → pd.DataFrame`. Reads a single CSV from one MT5-style directory (`data/4hr/`, `data/1hr/`, `data/daily/`). Single-side OHLC (no bid/ask separation).

[scripts/arc_kh24_v2/step1/run_step1.py:43](scripts/arc_kh24_v2/step1/run_step1.py:43) — `_load_pair_csv(p)` local variant; same MT5 layout.

**Critical finding:** `data/4hr/`, `data/1hr/`, `data/daily/` no longer exist under `data/` — only `data/histdata/` and a tiny `data/test/` fixture remain. The existing backtester cannot run against any current arc data without this dispatch.

### 3. Spread-handling logic

[core/spread_floor.py](core/spread_floor.py) — full `spread_floors_5ers.yaml` floor mechanism: load → sha256-verify against `expected_body_sha256` → cap-floor `apply_spread_floor_to_pips()`. Already feature-flagged via cfg.

Integration site: [scripts/phase_kgl_v2_4h_wfo.py:60–66](scripts/phase_kgl_v2_4h_wfo.py:60) imports and threads `STATE_CFG_KEY` through the runner. `core/backtester.py` uses a different `resolve_spread_pips` path; spread floor flows in through cfg state.

### 4. WFO orchestration

[analytics/wfo.py](analytics/wfo.py) — generic `generate_folds(from, to, train_months, test_months, step_months)` returning `Fold(fold_id, train_start, train_end, test_start, test_end)`. Rolling windows, train_end < test_start enforced. Used by `scripts/phase_kgl_v2_4h_wfo.py` for the published KH-24 7-fold WFO (Oct 2020 → Jan 2026). Generic enough to be reused; the 11-fold expanding-IS pattern needs a new helper (anchored `train_start=2010-01-01`, 1-year OOS slices).

### 5. Feature engineering

No unified feature pipeline exists. Per-arc step-1 scripts compute their own feature columns inline:
- [scripts/arc_kh24_v2_patch/emit_step1_base8_features.py](scripts/arc_kh24_v2_patch/emit_step1_base8_features.py) — base-8 set for the v2.0 self-test
- [scripts/arc_kh24_v2/step2/_features.py](scripts/arc_kh24_v2/step2/_features.py) — path features
- [scripts/lchar/run_layer*.py](scripts/lchar/) — L_char atlas features
- [core/features_path_so_far.py](core/features_path_so_far.py) — Pipeline D1 path-so-far computer

L_PROTOCOL §2 Step 1 lists the broader feature space (session, cross-pair, multi-TF, vol regime, distance, spread regime, causal lineage tag) as the v3.0 default. None of that currently exists as a shared module.

### 6. L_PROTOCOL.md anchors confirmed

§1 non-negotiables — no lookahead, ex-ante pool, D1 one-bar lag, real bid/ask, determinism, config-driven, anchor preservation.

§2 Step 1 — feature space (price geometry, time, cross-pair, multi-TF, vol regime, cross-asset, distance, spread regime).

§2 Step 5 — WFO structure: training/search 2010-01-01 → 2020-12-31, 11-fold 1-year folds; holdout 2021-01-01 → present one-shot; top-K (K=3) on holdout.

Appendix B — architecture search space (filter sets, exposure rules, SL × ATR(14) ∈ {1.5..4.0}, exit policies, pipeline_de_n ∈ {1,3,5,8}, A1..A6 enabled).

### 7. DATA_FOUNDATION.md — actual HistData layout

**Dispatch path** (Task 1 prescription): `data/histdata/<pair>/{bid,ask}/<year>/DAT_ASCII_<pair>_M1_<YYYYMM>.csv`.

**Actual on-disk layout** (per [docs/DATA_FOUNDATION.md](docs/DATA_FOUNDATION.md) §Location and verified):
```
data/histdata/<PAIR>/
  tick/<YYYY>/DAT_ASCII_<PAIR>_T_<YYYYMM>.zip          # raw, canonical
  m1/bid/<YYYY>/<PAIR>_M1_BID_<YYYYMM>.csv             # derived
  m1/ask/<YYYY>/<PAIR>_M1_ASK_<YYYYMM>.csv             # derived
data/histdata/manifest.json                            # tick zip sha256
data/histdata/m1_manifest.json                         # derived M1 sha256
```

I will **use the actual layout** (the dispatch path string is wrong but the intent is clear). Cache invalidation will key off `m1_manifest.json` sha256 (the derived layer the loader actually reads), not the tick `manifest.json` — derivation is reproducible from tick anyway, but cache freshness is bounded by the M1 layer.

Volume column: HistData tick volume is 0 by source; the M1 layer substitutes per-minute tick count (per [DATA_FOUNDATION.md](docs/DATA_FOUNDATION.md) §Aggregation). I'll preserve this.

### 8. KH-24 anchor lineage (from [ARC_HISTORY.md](ARC_HISTORY.md))

| Metric | Published (5ers MT5) | Real-spread reconciled (HistData expected band) | Tolerance (§8) |
|---|---:|---:|---|
| Worst-fold ROI | +1.92% (F7) | ~+1.28% | ±0.5pp |
| Worst-fold DD | 6.37% (F1) | (not specifically reported; expected near 6.37%) | ±1pp |
| Folds positive | 7/7 | unchanged at sign level | — |

Per dispatch Task 8 and L_PROTOCOL §8: either within ±0.5pp/±1pp of the published 5ers numbers, OR within the documented +1.28% real-spread band, counts as anchor preserved. The KH-24 published WFO uses **7 folds** (Oct 2020 → Jan 2026, all post-2020). The v3.0 framework uses **11 folds 2010-2020 + 1-shot 2021-2025 holdout** — there is no clean 1:1 fold mapping. The natural reproduction is: run KH-24 on the 2021-2025 holdout window with the same 7-fold rolling structure as published, and compare. I will document this explicitly in Task 8's report and not silently re-map.

---

## Files CC will touch

### New modules

| Path | Purpose | Source task |
|---|---|---|
| `core/data/__init__.py` | package marker | infra |
| `core/data/histdata_loader.py` | M1 bid+ask reader + parquet cache (`data/cache/m1/<pair>.parquet`) | Task 1 |
| `core/data/aggregator.py` | deterministic M1→{M5,M15,M30,H1,H4,D1,W1} + per-TF parquet cache (`data/cache/<tf>/<pair>.parquet`) | Task 2 |
| `core/data/cache_keys.py` | sha256 cache-key + invalidation logic (tied to `m1_manifest.json`) | Tasks 1, 2, 9b |
| `core/spread/real_spread.py` | per-bar `ask.close − bid.close`, quality flag for bid==ask==0/NaN | Task 3 |
| `core/sim/fill.py` | bar-level fill — entry at next-bar open, SL/TP intra-bar against bid/ask, no idealised fills | Task 3 |
| `core/sim/multipair.py` | single account state across 28 pairs concurrently; exposure rules at account level | Task 6 |
| `core/wfo/v3.py` | 11-fold anchored-IS WFO 2010-2020 + holdout 2021-present one-shot | Task 4 |
| `core/features/v3.py` | broader feature space (entry point) | Task 5 |
| `core/features/session.py` | London/NY/Tokyo/overlap/dead, hour-of-day, day-of-week | Task 5 |
| `core/features/cross_pair.py` | signal density, dollar-bloc, currency strength ranks | Task 5 |
| `core/features/multi_tf.py` | D1 directional state, D1 ATR percentile, W1 state | Task 5 |
| `core/features/vol_regime.py` | 4H ATR vs trailing-100, percentile rank | Task 5 |
| `core/features/distance.py` | from prior session HL, round numbers (.0050, .0100, .0500) | Task 5 |
| `core/features/spread_regime.py` | current spread vs trailing 100-bar avg | Task 5 |
| `core/features/lineage.py` | `causal_lineage ∈ {clean, suspect, unverified}` tagging | Task 5 |
| `core/features/cache.py` | feature-matrix cache keyed by (signal_def + pool_sha + feature_set_version) | Task 9b |
| `core/architectures/__init__.py` | architecture registry | Task 8 (interface, no impl yet) |
| `core/architectures/base.py` | A1..A6 protocol — `run(config, pool, panel) → wfo_result` | Task 8 |
| `core/parallel.py` | `multiprocessing.Pool` wrapper, default `min(28, cpu_count()-1)`, deterministic aggregation | Task 9c, 9d |
| `core/manifest.py` | sha256 manifest writer per artefact | Task 7 |
| `configs/wfo_v3.yaml` | locked WFO config (window dates, fold counts, holdout) | Task 4 |
| `configs/features_v3.yaml` | feature set version + per-feature toggles | Task 5 |
| `configs/parallel.yaml` | pool size override knob | Task 9c |
| `configs/kh24_anchor.yaml` | KH-24 config for Task 8 reproduction (signal/SL/trail/filters/exposure) | Task 8 |

### Modified modules

| Path | Change |
|---|---|
| [core/backtester.py](core/backtester.py) | Route OHLCV access through `core/data/histdata_loader` + `aggregator`; route spread through `core/spread/real_spread`; deprecate MT5 schema branch (gated behind explicit `data_source: mt5_legacy` cfg flag — default `histdata`). |
| [core/spread_floor.py](core/spread_floor.py) | Marked deprecated; kept loadable only when `data_source: mt5_legacy` (no removal yet — preserves the 5ers/spread-floors path used to publish KH-24 for reference / spot-check). |
| [analytics/wfo.py](analytics/wfo.py) | Generic helper retained; `core/wfo/v3.py` is the new v3.0 orchestrator that calls into it. |
| [docs/DATA_FOUNDATION.md](docs/DATA_FOUNDATION.md) | Reflect parquet cache layout + cache invalidation rules. |

### New tests

| Path | Purpose |
|---|---|
| `tests/test_histdata_loader.py` | layout parse, bid+ask alignment, missing-month tolerance, parquet roundtrip, ≥10× speedup vs CSV reparse |
| `tests/test_aggregator.py` | OHLC rules per-side, UTC boundaries, two-run byte-identical, parquet roundtrip |
| `tests/test_real_spread.py` | bid==ask==0 flag path, NaN handling, no spread-floor reference |
| `tests/test_fill.py` | next-bar open entry, intra-bar SL/TP priority on bid/ask, long vs short symmetry |
| `tests/test_multipair_sim.py` | single equity curve, exposure caps, cross-pair feature access at bar t |
| `tests/test_wfo_v3.py` | 11 folds, anchored IS, holdout one-shot, top-K=3 |
| `tests/test_features_v3.py` (or per-class) | lookahead-invariant spot-checks on 5 random trades per feature class; causal_lineage column present |
| `tests/test_determinism_parallel.py` | two-run sha256 with pool=1 vs pool=8 byte-identical |
| `tests/test_kh24_anchor.py` | wires Task 8 reproduction into CI as an opt-in (slow) marker; tolerance check |

### New / updated docs

| Path | Action |
|---|---|
| `docs/BACKTESTER_ARCHITECTURE.md` | **CREATE** — data source, TF aggregation, spread/fill, WFO, features, multi-pair, determinism, anchor results, cache layout |
| `docs/features_reference.md` | **CREATE** — every Step-1 feature: definition, computation, causal lineage tag, inputs |
| `docs/dispatches/backtester_reconfig_intent.md` | this file |
| `docs/dispatches/backtester_reconfig_log.md` | created during execution — running log per task |

---

## Interpretive calls (flag for chat before any code lands)

The dispatch is mostly literal, but four points need a chat call. **Stopping turn after this intent doc lands so these can be resolved.**

1. **Scope of this PR.** Tasks 1–10 cover a multi-day to multi-week implementation surface (~25 new modules + 9 tests + 2 new docs + the KH-24 anchor compute run, which is itself many hours wall-clock on 16 years of M1 data × 28 pairs). The DoD asks for **one PR**. Two viable interpretations:
   - **(a) One bundled PR.** Implement all 10 tasks, run the anchor, open the single PR. Long-lived branch.
   - **(b) Staged PRs under the `infra/backtester-v3-reconfig` umbrella.** e.g. PR-1: data loader + aggregator + parquet cache + tests. PR-2: spread + fill + multi-pair sim. PR-3: WFO + features. PR-4: parallelism + determinism harness. PR-5: KH-24 anchor reproduction + docs. Each PR is reviewable in isolation; the final PR is the gate (anchor must pass).
   - **Recommendation: (b).** The KH-24 anchor reproduction (Task 8) is the gate, and it depends on every prior task working — a single 10-task PR would be unreviewable and would conflate "infra builds" with "infra works." Staging surfaces problems at each layer.

2. **MT5 legacy code path.** Existing arcs (KH-24 v2.0 self-test, Arc 4 rerun, Arc 10) ran against MT5 data that is now deleted from the repo. The reconfig defaults to HistData, but the existing `core/backtester.py` references and the `spread_floors_5ers.yaml` reconciliation logic are tightly coupled to the MT5 schema. Two options:
   - **Hard-cut MT5.** Delete `core/spread_floor.py`, the MT5 branch in `core/backtester.py`, and the `data/4hr/` / `data/daily/` / `data/1hr/` references entirely. Cleaner but irreversible without a revert.
   - **Soft-deprecate.** Keep MT5 paths gated behind `data_source: mt5_legacy` cfg flag (default `histdata`). Anyone re-running an old arc for forensics still can.
   - **Recommendation: soft-deprecate**, since the dispatch says "no 5ers MT5 data path" in Task 1 but ARC_HISTORY's real-spread reconciliation table is one of the most cited cross-arc artefacts and re-running it for KH-25 / KH-27 forensics is plausible.

3. **KH-24 anchor reproduction window (Task 8).** The published KH-24 numbers come from a **7-fold rolling WFO** over Oct 2020 → Jan 2026. The new v3.0 framework uses **11-fold anchored IS 2010-2020 + 1-shot holdout 2021-2025**. There is no fold-by-fold remapping. Three options:
   - **(a) Run KH-24 through the v3.0 11-fold + holdout framework.** Worst-fold and DD across the new fold structure are compared to the published numbers within tolerance. Cleanest forward-looking — what every future arc will do.
   - **(b) Run KH-24 with its original 7-fold rolling WFO** using the new HistData loader + real spreads. Most directly comparable to the published numbers, but it bakes the old WFO into the new engine.
   - **(c) Both.** Worst-fold ratio on the published 7-fold window for anchor preservation; full 11-fold + holdout reported for v3.0 record.
   - **Recommendation: (c).** Anchor preservation needs the apples-to-apples comparison (b); the v3.0 framework needs to be exercised end-to-end on a known signal (a). Reporting both is one extra config and one extra report — cheap.

4. **`spread_floors_5ers.yaml` deprecation timing.** Dispatch Task 3 says "no `spread_floors_5ers.yaml` reference in the new code path" — clear. But also dispatch §"Required outputs" #3 says "If a bar has bid==ask==0 or NaN, the bar is flagged as a data quality issue in a separate log; trades are not simulated through such bars." That replaces the floor's purpose entirely. **DATA_FOUNDATION.md** still says `spread_floors_5ers.yaml` becomes a "fallback for bars with no quote activity." Two readings:
   - **Strict dispatch reading**: zero-spread bars are dropped from simulation, no fallback ever, file is dead.
   - **DATA_FOUNDATION reading**: zero-spread bars are filled from the floor file as a documented fallback.
   - **Recommendation: strict dispatch reading.** Drop the bar, log it, no silent fallback. This matches L_PROTOCOL §1 non-negotiable on "Real bid/ask spreads. No fallback mechanism." If chat wants the DATA_FOUNDATION fallback restored, it's a one-line cfg toggle but L_PROTOCOL would need an amendment.

---

## Pre-commitments (no chat call needed — captured here for transparency)

- **Parquet only.** Every tabular artefact downstream of HistData CSVs writes parquet. Manifest = JSON, closure docs = markdown, logs = text. (Dispatch Task 9a.)
- **Determinism baseline.** `random_state=42`, `n_jobs=1` for any in-worker compute that affects results, `lineterminator='\n'` on every text/JSON/MD write, sha256 manifest per artefact. (Dispatch Task 7 + L_PROTOCOL §1.)
- **Parallelism via `multiprocessing.Pool`**, not threading. Default pool = `min(28, cpu_count()-1)`. (Dispatch Task 9c.)
- **Feature matrix cache key** = sha256 of `(signal_def_text + pool_sha256 + feature_set_version_string)`. Cache files at `data/cache/features/<arc_id>/<feature_set_hash>.parquet`. (Dispatch Task 9b.)
- **No protocol logic invented.** This dispatch implements what L_PROTOCOL v3.0 specifies; it does not extend the protocol. (Dispatch §Discipline rules.)

---

## End of intent

End turn. Awaiting chat on the four interpretive calls above (scope staging, MT5 legacy, anchor reproduction window, spread-floor deprecation) before any code or test file lands.

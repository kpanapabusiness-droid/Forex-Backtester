# arc_10_intent — D1 swing-low rejection long, v3.0 re-run

> **Dispatch:** `arc_10_dispatch.md` (Arc 10 v3.0)
> **Protocol:** L_PROTOCOL v3.0 (Amendments 1 + 2)
> **Sub-protocol:** vanilla
> **Branch:** `claude/optimistic-golick-abf13a` (see flag 1)
> **Worktree:** `.claude/worktrees/optimistic-golick-abf13a`
> **Status:** intent. Plumbing only. End turn after writing per dispatch.

Plumbing only. No interpretive judgement on cohort viability. Prior v2.x verdict at `docs/archive/arc_results/ARC_10_RESULT.md` is historical record, not a target.

---

## 1. Resolved signal spec path

[signals/lchar_dlr_long.py](signals/lchar_dlr_long.py) — locked module docstring at lines 3-48.

The referenced `docs/archive/signal_specs/signal_spec_d1_swing_low_rejection_long_v0.1.md` does **not** exist on disk in this worktree (verified via Glob `**/*swing_low_rejection*` — only the Python producer matches). The dispatch's fallback `docs/archive/arc_results/ARC_10_RESULT.md` is a closure narrative; it does not restate the signal spec verbatim. The producer module's docstring is the only verbatim spec available and matches the conditions referenced by the prior Arc 10 v2.3 closure. Proceeding with the producer docstring as the locked spec. See flag 2.

## 2. Concise signal definition

**D1 anchor (one-day-lag enforced):**
1. D1 swing-low at day d: `low[d] < min(low[d-3..d-1])` AND `low[d] < min(low[d+1..d+3])`
2. Right-edge constraint: most recent identifiable L_1 at most `D1[d_t − 4]` where `d_t` is the D1 bar containing 4H bar t
3. Prior swing-low L_0 = next-most-recent before L_1
4. HL structure: both L_1, L_0 within last 30 D1 bars at 4H bar t AND `L_1 > L_0` (strict)
5. L_1 freshness: L_1 not older than 20 D1 bars

**4H test / reject (signal bar t):**
6. Test (proximity): `low[t] <= L_1 + 0.25 * ATR(14)_4H[t]`
7. Reject (close above): `close[t] > L_1 + 0.10 * ATR(14)_4H[t]`
8. Trigger-bar geometry: `close[t] > open[t]` AND `(close[t] − low[t]) / (high[t] − low[t]) >= 0.6`

**Spacing & entry:**
9. ≥ 20 4H bars since last full signal on this pair
10. Entry: bar t+1 open

**Confirmed in spec** (per dispatch §3 item 2): D1 swing-low detection method, rejection condition (cond 7), 4H entry trigger (cond 10). Detection method is bilateral in form (±3 bars) but applied with a confirmation lag (right-edge offset 4); see §3.

## 3. Producer-level causal trace of D1 swing-low detection

**Function / code path:**
- Detector: [signals/lchar_dlr_long.py:95-119](signals/lchar_dlr_long.py:95) `compute_d1_swing_low_flags`
- Consumer: [signals/lchar_dlr_long.py:139-313](signals/lchar_dlr_long.py:139) `compute_signal`, lookup at lines 234-239 via `searchsorted` on pre-computed swing indices restricted to `idx <= d_search_max = d_t − 4`.
- D1 → 4H index map: [signals/lchar_dlr_long.py:122-136](signals/lchar_dlr_long.py:122) `_date_to_d1_index`, `searchsorted(d1_norm, bar_norm, side="right") − 1` = largest D1 idx with normalized D1 date ≤ normalized 4H bar date.

**Detector mechanics.** For D1 bar d, flag swing-low iff `low[d] < min(low[d−k..d−1])` AND `low[d] < min(low[d+1..d+k])`, with k=3. Bilateral form.

**Bilateral in form — but causal in application.** At signal time on 4H bar t:
- d_t = D1 bar containing t (largest D1 date ≤ t's calendar date)
- Candidate L_1 search restricted to `d_search_max = d_t − 4`
- With k=3, confirming a swing-low at index d uses future D1 bars d+1..d+3
- Latest confirmable d under this constraint is `d_t − 4`, whose future-confirmation window spans d_t−3 .. d_t−1 — **strictly before** the D1 bar containing the signal-bar open.

**Comparison to Arc 9 failure mode.** Arc 9 (per `ARC_HISTORY.md` Arc 9 row + INCIDENT note): `d1_bars_since_swing_low` and `d1_bars_since_swing_high` used a ±10-bar centred detector applied at signal time (k=10, no right-edge offset). Join-level audit GREEN; producer-level audit RED. Causal patch (10-day forward confirmation) dropped AUC 0.7508 → 0.5190 (LGBM) / 0.5551 (RF).

Arc 10's DLR detector is structurally the confirmation-lag form Arc 9's patch produced: k=3, applied with right-edge offset 4 ≥ k+1. The two-step structure (bilateral detector + right-edge offset ≥ k+1) makes this confirmation-lag, **not** centred-at-signal. The data read for swing-low geometry is strictly D1 bars d ≤ d_t − 4 (i.e. ≥ 4 calendar days before the signal bar's date), comfortably satisfying L_PROTOCOL §1's D1 one-bar lag rule.

**Module-asserted invariant.** [signals/lchar_dlr_long.py:39-43](signals/lchar_dlr_long.py:39) asserts NaN-perturbation invariance: NaN-ing D1[d_t] leaves Arc 10 signal output unchanged for any bar t, because the signal references only D1 bars d ≤ d_t − 4 for swing-low identification (and the d_t lookup itself uses only D1 dates strictly < the bar-t-open). Step 1 will verify this on 5 random trades (dispatch elevates from default 3 to 5 given D1-based signal).

**Verdict: producer-level causal trace PASSES at intent stage.** Arc 9 failure mode does NOT apply. No HALT triggered. Step 1 §"Lookahead spot-check, 10 random trades" + §"D1-lag NaN-perturbation, 5 random trades" will reconfirm at compute time. Step 6 will repeat the audit if Step 5 produces a PASS candidate; swing-detection features carry priority audit per dispatch §"Step 6".

## 4. Files CC will touch or create

**Create (absolute paths relative to repo root):**

Arc artefacts:
- `results/l_arc_10/ARC_OPEN.md` — protocol-required arc registration per L_PROTOCOL §6
- `results/l_arc_10/step_1/pool.parquet`
- `results/l_arc_10/step_1/integrity_report.md`
- `results/l_arc_10/step_1/manifest.json`
- `results/l_arc_10/step_2/cluster_assignments.parquet`
- `results/l_arc_10/step_2/cluster_summary.md`
- `results/l_arc_10/step_2/manifest.json`
- `results/l_arc_10/step_3/capturability.csv`
- `results/l_arc_10/step_3/capturability_summary.md`
- `results/l_arc_10/step_3/manifest.json`
- `results/l_arc_10/step_4/extraction_metrics.csv`
- `results/l_arc_10/step_4/feature_importance.csv`
- `results/l_arc_10/step_4/extraction_summary.md`
- `results/l_arc_10/step_4/manifest.json`
- `results/l_arc_10/step_5/wfo_results.csv`
- `results/l_arc_10/step_5/wfo_oracle.csv`
- `results/l_arc_10/step_5/architectures_ranked.md`
- `results/l_arc_10/step_5/best_candidate.md`
- `results/l_arc_10/step_5/manifest.json`
- `results/l_arc_10/step_6/causal_audit_report.md` — only if invoked
- `results/l_arc_10/step_6/manifest.json` — only if invoked
- `results/l_arc_10/ARC_CLOSURE.md`

Dispatch artefacts:
- `docs/dispatches/arc_10_log.md` — final, per WORKFLOW §2
- `docs/dispatches/arc_10_diagnostic.md` — only if HALT triggered per dispatch §"HALT triggers"

Code + config:
- `scripts/l_arc_10_v3/step_1.py` — v3-engine Step 1 driver. The prior `attic/scripts/l_arc_10/step1_plumbing.py` is v2 (custom `_simulate_pair`, MT5 spread file via `core.spread_floor`, no `data/histdata` layer) and is NOT re-usable on v3.
- `scripts/l_arc_10_v3/step_2.py`, `step_3.py`, `step_4.py`, `step_5.py` — per-step drivers
- `scripts/l_arc_10_v3/step_6.py` — only if invoked
- `configs/l_arc_10_v3/arc_open.yaml` — window, pairs, risk, signal params (mirrored from `lchar_dlr_long.py` module constants), SL/exit/exposure at Step 1
- `configs/l_arc_10_v3/step_3.yaml`, `step_4.yaml`, `step_5.yaml` — per-step parameters

**Touch (reuse, no edits at intent stage):**
- [signals/lchar_dlr_long.py](signals/lchar_dlr_long.py) — verified causal, invoked as-is
- [core/data/](core/data/) — `histdata_loader.load_m1`, `aggregator.aggregate`
- [core/spread/](core/spread/) — `real_spread` (HistData bid+ask, no floor file per L_PROTOCOL §1)
- [core/features/pipeline.py](core/features/pipeline.py) + 27-feature registry (`docs/features_reference.md`)
- [core/sim/](core/sim/) — `multipair_backtester.MultiPairBacktester`, `account.Account`, `fill.py`
- [core/wfo/](core/wfo/) — `folds.build_v3_folds`, `orchestrator.run_search`, `orchestrator.run_holdout`, `gates`
- [core/determinism.py](core/determinism.py) — `seed_everything`, line-terminator contract
- [core/features/cache.py](core/features/cache.py) — `get_or_compute` parquet caching
- [core/parallel.py](core/parallel.py) — `parallel_pair_map`, `build_panel_parallel`

**Do NOT touch (per dispatch §"What you do NOT do"):**
- `L_PROTOCOL.md`, `WORKFLOW.md`, sub-protocols, engine code under `core/`
- KH-24 anchor code, EA, deployed system, `reference/kh24_ea/`
- `ARC_TRACKER.md` (auto-updates from closure per L_PROTOCOL §6)
- `results/ARC_QUEUE.md`
- Anything under other Wave 1 arc folders (`results/l_arc_8/`, `l_arc_9/`, `l_arc_11/`)

## 5. Confirmed parameters

| Field | Value | Source |
|---|---|---|
| `signal_class` | D1 swing-low rejection long (DLR) | dispatch |
| `signal_definition` | [signals/lchar_dlr_long.py](signals/lchar_dlr_long.py) module docstring | §1 above |
| `tf_mode` | locked | dispatch |
| `tf` | 4H | dispatch |
| `sub_protocol` | vanilla | dispatch |
| `pair_set` | 28 FX (`configs/data_v3.yaml:39-67`) | dispatch + config |
| `window_start` | 2010-01-01 | dispatch |
| `window_end` | 2026-04-30 (April 2026 is the most recent complete month relative to today 2026-05-22; will narrow to HistData coverage end if earlier) | dispatch + flag 4 |
| `risk_per_trade` | 0.5% | dispatch |
| `step1_sl` | 2.0 × Wilder ATR(14)_4H at signal-bar close | dispatch §"Step 1 — Plumbing" |
| `step1_entry` | next-bar open (long fill = open_ask) | L_PROTOCOL §2 Step 1 |
| `step1_exit` | signal's natural time-exit horizon (240 4H bars per prior spec) + 2.0×ATR SL | dispatch §"Step 1 — Plumbing" |
| `step1_exposure_cap` | unrestricted | dispatch §"Step 1 — Plumbing" |
| `step3_sl_sweep` | SL ∈ {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} × ATR(14) per cluster | dispatch §"Step 3" |
| `wfo_search` | 11-fold anchored 2010-01-01 → 2020-12-31 (`build_v3_folds`) | dispatch §"Step 5" |
| `holdout` | 2021-01-01 → window_end, one-shot per top-3 candidates | dispatch §"Step 5" |
| `oracle_wfo` | per cluster, true-cluster-membership label | dispatch §"Step 5" |
| `hypothesis` | Re-run signal through L_PROTOCOL v3.0. No preconception; prior v2.x findings are historical, not targets. | dispatch line 11 |
| `expected_failure_modes` | (a) HistData M1 coverage shorter than 2010-2026, (b) per-pair pool size < 30 leading to thin clusters, (c) Step 4 AUC ceiling around V-shape archetype as cross-arc, (d) Step 5 admit-only vs full-pool gap per ARC_HISTORY cross-arc lesson | informational; not a target |

## 6. Step 1 feature space

L_PROTOCOL §2 Step 1 default feature space — 27 features registered in `core.features.pipeline.compute_feature_matrix`, catalogued in `docs/features_reference.md`. Used as-is per dispatch §3 item 6.

| Class | Features (count) |
|---|---|
| price_geometry | atr_14, kijun_26_distance, range_close_ratio, swing_high_distance_14, swing_low_distance_14 (5) |
| session | day_of_week, hour_of_day, session_dead, session_london, session_ldn_ny_overlap, session_ny, session_tokyo (7) |
| vol_regime | atr_percentile_100, atr_vs_trailing_100 (2) |
| distance | distance_to_round_number, prior_session_high_distance, prior_session_low_distance (3) |
| spread_regime | spread_percentile_100, spread_vs_trailing_100 (2) |
| multi_tf | d1_atr_percentile_100, d1_close_slope_magnitude, d1_close_slope_sign, w1_close_slope_sign (4) |
| cross_pair | dollar_bloc_state, eur_strength_index, signal_density_28, usd_strength_index (4) — all `suspect` lineage pending Step 6 |

**Per-arc signal-specific feature columns** carried from the DLR producer at the signal bar (added to the pool, tagged `clean` because the producer is causally verified at intent stage and reads only D1 bars d ≤ d_t − 4):

- `L1_value`, `L0_value` — D1 swing-low prices
- `L1_age_d1_bars`, `L0_age_d1_bars` — bar-counts since L_1, L_0
- `L1_to_atr_proximity` — `(low[t] − L_1) / ATR(14)_4H`
- `reject_buffer_atr` — `(close[t] − L_1) / ATR(14)_4H`
- `upper_fraction` — `(close − low) / (high − low)`

These columns participate in Step 4 extraction. If any cross_pair (`suspect`) feature appears in any Step 4 top-10, Step 4's special audit (dispatch §"Step 4 — Extraction") spot-checks the producer at that point, with full Step 6 audit deferred unless Step 5 produces a PASS candidate.

## 7. Determinism plan

- `random_state = 42` everywhere — model fits, KMeans/HDBSCAN, TimeSeriesSplit, any shuffling
- `n_jobs = 1` inside any per-row computation; parallelism only **between** work units (per-pair) via `core.parallel.parallel_pair_map` / `build_panel_parallel`
- `lineterminator = "\n"` for every CSV (PR-D determinism contract per `core/determinism.py`)
- Parquet caching enabled — `core.features.cache.get_or_compute` keyed on `sha256(signal_def + pool_sha256 + feature_set_version)`; sidecar `.meta.json` records inputs in plaintext for audit reconstruction
- Two-run sha256 reproducibility at Step 1 (per dispatch §"Determinism: sha256 of pool reproduces across two seeded runs") — Step 1 driver invocation will support `--verify-determinism` (mirrors the v2 attic plumbing convention) producing both-runs sha256 in the manifest
- Per-step manifest records: input config sha256, signal-module sha256, output artefact sha256s, environment versions (Python, pandas, numpy, sklearn, lightgbm), `core/determinism.py:RANDOM_STATE` and `N_JOBS`
- `seed_everything(42)` called at the top of every step driver

---

## Flags for chat before Step 1

1. **Branch name discrepancy.** Dispatch specifies `arc/l_arc_10` cut from main. Current worktree branch is `claude/optimistic-golick-abf13a` cut from main. PR title will be `[ARC 10 v3.0]...` either way. Options: (a) rename branch in this worktree to `arc/l_arc_10` before any commit, (b) accept the auto-generated branch name. No code impact; only matters for L_PROTOCOL §7 auto-cleanup hook conventions and consistency with parallel arcs.

2. **Signal-spec .md file missing.** `docs/archive/signal_specs/signal_spec_d1_swing_low_rejection_long_v0.1.md` is referenced by both the dispatch and the prior v2.3 closure but does not exist in this worktree. The verbatim spec is recovered from the Python producer docstring; the prior v2.3 Arc 10 ran against the same producer (sha256 in v2.3 manifest at `attic/scripts/l_arc_10/step1_plumbing.py:_sha256_file(... lchar_dlr_long.py)`). No action needed unless chat wants the canonical `.md` spec restored before Step 1.

3. **HistData M1 data not visible in this worktree.** `data/histdata/` contains manifests + reports but `ls` shows no per-pair `<PAIR>/m1/` subdirs. M1 manifest covers 28 pairs (verified — sample shows AUDCAD M1 ask CSVs Jan-Jun 2010). Most likely the per-pair tick + M1 CSVs (~52 GB tick + 18 GB M1) are gitignored and present on disk, with `ls` output filtered or the worktree's directory listing returning incomplete results for very-large dirs. Step 1 will verify at compute time by attempting `core.data.histdata_loader.load_m1("EURUSD", ...)`; if data is genuinely absent in this worktree, Step 1 HALTs with a Phase-0 data-availability diagnostic before doing anything else.

4. **Most-recent-complete-month interpretation.** Resolved as **2026-04-30** (April 2026 is the most recent calendar month complete relative to today 2026-05-22). Will narrow to HistData's actual coverage end if it falls earlier — chat will see the resolved window-end in the Step 1 manifest.

---

End of intent. Plumbing only. No interpretive judgement on cohort viability. Ending turn for chat review per dispatch §"Intent doc (mandatory, end turn after writing)".

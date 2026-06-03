# Arc 11 — Intent Doc (v3.0)

> **Dispatch:** `arc_11_dispatch.md` (CC Dispatch — Arc 11, v3.0)
> **Worktree:** `C:/Users/panap/Documents/Forex-Backtester/.claude/worktrees/strange-bassi-712010`
> **Current branch:** `claude/strange-bassi-712010` (cut from `main` at `f3bc078`)
> **Protocol:** `L_PROTOCOL.md` v3.0 (incl. Amendments 1 + 2, 2026-05-22), sub-protocol `vanilla`
> **Status:** plumbing-only intent. No code, no compute. End-turn after this doc for chat review.

This doc covers the seven items in dispatch §"Intent doc". Two items require chat decision before Step 1 can begin and are flagged inline: **(A)** signal spec file missing at all three dispatch-listed paths; **(B)** current branch is not `arc/l_arc_11` as the dispatch specifies.

---

## 1. Resolved signal spec path

**Flag (A): the canonical signal spec file is missing.**

Dispatch §"Read-first" item 6 specifies the primary path and two fallbacks:

| Path | Exists? |
|---|---:|
| `docs/archive/signal_specs/signal_swing_high_breakout_trend_long_v0.1.md` | NO |
| `docs/archive/arc_results/ARC_11_RESULT.md` | NO |
| `results/l_arc_11/ARC_11_CLOSURE.md` | NO |

None of the three exist on this v3.0 worktree (`docs/archive/arc_results/` itself has no `ARC_11_*`; `docs/archive/signal_specs/` contains nine other v2-era specs but not SHB; `results/l_arc_11/` does not exist at all).

**Producer code authoritative source.** [signals/lchar_swing_high_breakout_trend.py](signals/lchar_swing_high_breakout_trend.py) is committed under v3.0 (file present in `main` at `f3bc078`). Its module docstring (lines 1-37) reproduces the signal spec verbatim with the comment "locked verbatim from `docs/signal_spec_swing_high_breakout_trend_long_v0.1.md` / Downloads spec" — the spec file itself is missing from the repository but its contents survive inside the producer's docstring. The v2-era attic config [attic/configs/wfo_l_arc_11.yaml](attic/configs/wfo_l_arc_11.yaml) (lines 27-44) also asserts the same parameters against this producer module via runtime assertions in [attic/scripts/l_arc_11/step1_backtest.py](attic/scripts/l_arc_11/step1_backtest.py) (lines 519-526).

**Proposal:** treat the producer-module docstring + attic config as the authoritative definition for Arc 11 v3.0; reconstruct the missing spec file at `docs/archive/signal_specs/signal_swing_high_breakout_trend_long_v0.1.md` from the docstring as the first Step 1 file artefact, sha-pinned in `step_1/manifest.json`. Awaiting chat confirmation before doing so.

---

## 2. Signal definition (confirms swing-high method, break trigger, trend qualifier)

Reproduced from [signals/lchar_swing_high_breakout_trend.py:1-37](signals/lchar_swing_high_breakout_trend.py:1) and verified against [attic/configs/wfo_l_arc_11.yaml:29-44](attic/configs/wfo_l_arc_11.yaml:29):

**Swing definitions (3-bar local extreme — causal with confirmation lag):**
- swing-high at bar `k` iff `high[k] > max(high[k-3..k-1])` AND `high[k] > max(high[k+1..k+3])`
- swing-low at bar `k` iff `low[k]  < min(low[k-3..k-1])`  AND `low[k]  < min(low[k+1..k+3])`
- **right-edge constraint:** at signal-bar `t`, only swings with `k <= t - 4` are identifiable (`RIGHT_EDGE_OFFSET=4`). Right-side detection bars `k+1..k+3 ≤ t-1` → all data closed strictly before bar `t` open.

**Trend filter (structural, no MA — Arc 9 convention):**
1. Identify swing-lows in window `[t-30, t-4]` (`TREND_FILTER_LOOKBACK=30`).
2. Require ≥ 1 such swing-low exists.
3. Require `close[t-1] > min(swing-low values in window)`.

**Reference swing-high (`H_ref`):**
- Identify swing-highs in window `[t-20, t-4]` (`H_REF_LOOKBACK=20`).
- `H_ref` = most-recent identifiable swing-high in window.
- Required to exist (else no signal).

**Break trigger at bar `t` (long):**
- `close[t] > H_ref + 0.10 × ATR(14)[t]` (decisive break with buffer; `BREAK_BUFFER_ATR=0.10`)
- `close[t] > open[t]` (bullish close)
- `(close[t] - low[t]) / (high[t] - low[t]) ≥ 0.5` (`CLOSE_UPPER_HALF_MIN=0.5`)

**Spacing & entry:**
- `t - last_signal_t ≥ 20` (refractory, `REFRACTORY_BARS=20`)
- Entry: bar `t+1` open

**ATR(14):** Wilder smoothed on 4H bars, causal (uses TR values for bars `≤ t` only; signal evaluated at bar `t` close).

**Causality summary of swing detection method:** confirmation-lag variant (one-sided in the sense the protocol requires). NOT centred smoother, NOT ZigZag, NOT ±N-bar centred pivot at signal time. See §3 for the full producer-level trace per dispatch.

---

## 3. Producer-level causal trace of the swing-high detection

> Dispatch §"Intent doc" item 3 + §"Read-first" CLAUDE.md producer-level causal audit standard (Arc 9 lesson) — required to be explicit before any compute.

**Function/code path:**

1. **Swing detection function.** [signals/lchar_swing_high_breakout_trend.py:83-128](signals/lchar_swing_high_breakout_trend.py:83) — `_detect_swings(high, low, k=3) -> (sh_mask, sl_mask)`.
   - **Mechanism:** at each bar position `i`, compares `high[i]` against `max(high[i-3..i-1])` (left side, `pandas.rolling(3).max().shift(1)`) AND `max(high[i+1..i+3])` (right side, computed by reversing the series, applying the same rolling-max + shift, then reversing back).
   - **This function uses 3-bar forward-data within the detection window.** That is the mathematical swing definition, and on its own it is non-causal.

2. **Causal envelope enforcement.** [signals/lchar_swing_high_breakout_trend.py:196-260](signals/lchar_swing_high_breakout_trend.py:196) — `compute_signal` loop body at trigger time `t`.
   - Line 209: `right_edge = t - RIGHT_EDGE_OFFSET` with `RIGHT_EDGE_OFFSET=4`.
   - Lines 216, 230: window filters explicitly require `sl_positions ≤ right_edge` and `sh_positions ≤ right_edge` — only swing positions `k ≤ t-4` are consumed by trigger evaluation.
   - **Implication:** the latest swing position the trigger can consume is `k = t-4`, whose right-side detection bars are `k+1..k+3 = t-3..t-1`. All right-side bars used in detection of any consumed swing are at positions `≤ t-1`, i.e. closed strictly before bar `t` open (and trivially before the bar `t+1` open entry).
   - **Conclusion:** swing detection is mathematically ±N-bar centred (which is non-causal *if used at the signal bar*) but the right-edge constraint at consumption time makes the full pipeline confirmation-lag causal. **Producer-level audit: PASS.**

3. **Trend-filter swing-low.** Same mechanism as above, on the swing-low side. Lines 213-226 enforce identical right-edge ≤ t-4 constraint for swing-lows used in the trend filter. **PASS.**

4. **ATR(14) at trigger.** [signals/lchar_swing_high_breakout_trend.py:55-80](signals/lchar_swing_high_breakout_trend.py:55) — `_wilder_atr(df, period=14)` is the canonical Wilder recursive form `atr[i] = (atr[i-1]*(period-1) + tr[i]) / period`. Uses TR values for bars `0..t` only. Trigger uses `atr[t]` at bar `t` close (line 205, line 241). **PASS — no forward dependency.**

5. **Trigger conditions** (lines 238-258): all reference `close[t]`, `open[t]`, `high[t]`, `low[t]` at bar `t` close (signal evaluation time). No `t+1` or later references in trigger evaluation. **PASS.**

**Independent corroboration — the attic Step 1 backtester has a right-edge audit built in.** [attic/scripts/l_arc_11/step1_backtest.py:622-660](attic/scripts/l_arc_11/step1_backtest.py:622) asserts that `min(h_ref_bar_offset) ≥ 4` over every emitted trade in the pool and counts trades with `h_ref_bar_offset < 4` (must be 0). This audit is structurally redundant with the producer-side constraint but provides a runtime tripwire and will be carried into the v3.0 Step 1 runner (dispatch §"Integrity checks" elevates this to a 10-trade manual verification, vs the default 5).

**No producer-level non-causality detected. Step 1 cleared to proceed.**

The dispatch's HALT trigger "Swing-detection producer code uses non-causal definition (centred smoother, ±N-bar pivot, ZigZag, or any future-data dependency)" is **not** triggered: the detection function uses ±N-bar geometry by construction, but the trigger envelope (`right_edge = t-4`) means no future-data flows into the trade-time decision. This is exactly the "confirmation-lag variant" the dispatch text whitelists.

Arc 9 lesson rigour applied: the audit ran at producer-level scope (not join-level, not end-to-end probability), it explicitly checked which positions the trigger consumes (not just whether `merge_asof` had `direction='backward'`), and the result is being recorded in this intent doc *before* Step 1 compute runs (not after a model has been trained).

---

## 4. Files CC will touch or create

**To create (Step 1 prerequisites + artefacts):**

| Path | Purpose |
|---|---|
| `docs/archive/signal_specs/signal_swing_high_breakout_trend_long_v0.1.md` | Reconstruct the missing spec from the producer docstring (see §1 Flag A). |
| `results/l_arc_11/ARC_OPEN.md` | Per L_PROTOCOL §6 required arc-open fields. |
| `configs/wfo_l_arc_11.yaml` | v3.0-vintage config (distinct from `attic/configs/wfo_l_arc_11.yaml` which is v2). |
| `scripts/l_arc_11/__init__.py` | Module marker. |
| `scripts/l_arc_11/step_1_plumbing.py` | v3 Step 1 runner: HistData→aggregator→signal application→trade pool→v3 27-feature matrix→integrity checks. |
| `results/l_arc_11/step_1/pool.parquet` | Trade pool with full feature matrix. |
| `results/l_arc_11/step_1/integrity_report.md` | All 7 integrity checks (incl. 10-trade lookahead spot-check with explicit swing-detection verification per dispatch). |
| `results/l_arc_11/step_1/manifest.json` | sha256 per artefact. |

**To touch (likely, conditional on chat acceptance):**

| Path | Reason |
|---|---|
| `ARC_TRACKER.md` | Auto-update on arc open per L_PROTOCOL §6. |
| `docs/dispatches/arc_11_intent.md` | This doc. |
| `docs/dispatches/arc_11_log.md` | WORKFLOW §2 log doc (post-execution, before PR). |

**To create later (Steps 2-5, contingent on each step's chat-review checkpoint):**

`results/l_arc_11/step_{2,3,4,5}/` artefact tree per L_PROTOCOL §2 / dispatch §"Step N — ..." sections.

**Will NOT touch** (dispatch §"What you do NOT do"): `L_PROTOCOL.md`, `WORKFLOW.md`, sub-protocols, engine code under `core/`, KH-24 config or strategy, other arcs' result trees.

---

## 5. Window, pair set, risk, TF (confirmed)

| Setting | Value | Source |
|---|---|---|
| Window start | `2010-01-01` | Dispatch + L_PROTOCOL Appendix B |
| Window end | `2026-04-30` | Dispatch ("most-recent-complete-month"); today = 2026-05-22; April 2026 confirmed present in `data/histdata/m1_manifest.json` for all 28 pairs (BID + ASK) |
| WFO training window (Step 5) | `2010-01-01 → 2020-12-31` (11 folds) | L_PROTOCOL §2 Step 5 |
| Holdout (Step 5, one-shot) | `2021-01-01 → 2026-04-30` | L_PROTOCOL §2 Step 5 + Amendment 1 |
| Pair set | 28 FX (same as KH-24) | Dispatch; full list mirrors [attic/configs/wfo_l_arc_11.yaml:55-83](attic/configs/wfo_l_arc_11.yaml:55) |
| Signal TF | 4H (locked) | Dispatch |
| Risk per trade | 0.5% of reset-floor balance | Dispatch + L_PROTOCOL §6 ARC_OPEN default |
| SL at Step 1 | 2.0 × Wilder ATR(14) at entry | Dispatch §"Step 1 — Plumbing" |
| Entry | Next-bar open (long fill = `open_ask`) | Dispatch + v3 PR-B fill primitives |
| Exit (Step 1) | Natural exit: 2.0×ATR SL + 240-bar (40-day) time exit | Producer + attic config |
| Exposure (Step 1) | `max_concurrent_per_pair = 1`, no currency/total cap | Dispatch §"Step 1 — Plumbing": "unrestricted" |

---

## 6. L_PROTOCOL Step 1 default feature space

**Used as-is per dispatch §"Intent doc" item 6.**

The v3.0 default feature matrix is the 27-feature catalog at [docs/features_reference.md](docs/features_reference.md) computed by [core/features/pipeline.py](core/features/pipeline.py) `compute_feature_matrix(pair, pair_df, panel)`:

- **price_geometry (5):** `atr_14`, `kijun_26_distance`, `range_close_ratio`, `swing_high_distance_14`, `swing_low_distance_14`
- **session (7):** `day_of_week`, `hour_of_day`, `session_dead`, `session_ldn_ny_overlap`, `session_london`, `session_ny`, `session_tokyo`
- **vol_regime (2):** `atr_percentile_100`, `atr_vs_trailing_100`
- **distance (3):** `distance_to_round_number`, `prior_session_high_distance`, `prior_session_low_distance`
- **spread_regime (2):** `spread_percentile_100`, `spread_vs_trailing_100`
- **multi_tf (4):** `d1_atr_percentile_100`, `d1_close_slope_magnitude`, `d1_close_slope_sign`, `w1_close_slope_sign`
- **cross_pair (4, lineage `suspect`):** `dollar_bloc_state`, `eur_strength_index`, `signal_density_28`, `usd_strength_index`

23 features lineage = `clean`; 4 `suspect` (all cross_pair). No `unverified` features. Lineage tags propagate into Step 4 / Step 6 audits per the pipeline contract.

**Per-arc signal-specific features added** (L_PROTOCOL §2 Step 1 final paragraph: "Per-arc signal-specific features are added"): the SHB producer already surfaces five at signal-bar close — `h_ref`, `h_ref_bar_offset`, `break_magnitude_atr`, `close_position`, `trend_filter_swing_low`. These will be joined onto the pool as Arc 11-specific features alongside the 27 defaults, each lineage-tagged `clean` (producer-level audit in §3 above).

**No deviation from default.** Sub-protocol = `vanilla`. No `heavy_ml_probe`, no `signal_discovery_probe`. Step 1 feature matrix = 27 default + 5 SHB-specific = 32 columns.

---

## 7. Determinism plan

| Item | Setting |
|---|---|
| `random_state` | `42` everywhere (L_PROTOCOL §1 + Appendix A) |
| `n_jobs` (per work unit) | `1` (per L_PROTOCOL Amendment 2 + `core.determinism`) |
| Multiprocessing (parallel pair iteration) | `core.parallel.parallel_pair_map`, pool size default `min(28, cpu_count()-1)`. v3 contract: output is byte-identical across pool sizes — asserted by `tests/test_determinism.py`. Per-pair work is single-threaded (`n_jobs=1`); parallelism is across pairs only. |
| Parquet caching | Enabled. Cold-cache: `data/cache/{m1,H4,D1,W1}/<PAIR>.parquet` will be built on first Step 1 run from `data/histdata/m1_manifest.json` (which is present and current to 2026-04). Feature-matrix cache: `data/cache/features/l_arc_11/<feature_set_hash>.parquet` per [BACKTESTER_ARCHITECTURE.md §Feature matrix cache](docs/BACKTESTER_ARCHITECTURE.md). Cache key includes signal_def, pool_sha, feature_set_version — any change invalidates. |
| `lineterminator` | `'\n'` (L_PROTOCOL §1 + `core.determinism.LINE_TERMINATOR`) |
| sha256 manifests | Per L_PROTOCOL §6 — emitted on every artefact under `step_1/manifest.json` and every subsequent step. |
| Two-run reproducibility | Step 1 runner will support `--verify-determinism` mode (re-run, compare sha256, audit file). Mirrors the attic Step 1's audit pattern at [attic/scripts/l_arc_11/step1_backtest.py:752-815](attic/scripts/l_arc_11/step1_backtest.py:752). |

**Spread source.** Per L_PROTOCOL §1 v3.0 non-negotiable, real bid/ask from HistData M1 is canonical. NO `configs/spread_floors_5ers.yaml` lookup at the v3 layer (that was a v2-era fallback). Long entry fills at `open_ask`, long exits at `close_bid` / `low_bid` (SL) / `high_bid` (TP) — per `core.sim.fill`.

---

## Items for chat decision before Step 1 starts

**Flag (A) — signal spec reconstruction.** Recommend: yes, reconstruct from producer docstring + attic config as the first Step 1 artefact. Spec content is fully recoverable; absence of the canonical file is a docs-tree gap, not a signal-definition gap.

**Flag (B) — branch.** Dispatch §"Branch" specifies `arc/l_arc_11` (cut from main, dedicated worktree). Current worktree is on `claude/strange-bassi-712010` (cut from main at `f3bc078`, working tree clean). Options:
1. Rename current branch to `arc/l_arc_11` locally; first push creates the remote `arc/l_arc_11`. Worktree path stays as-is (`strange-bassi-712010`).
2. Keep current branch; final PR opens from `claude/strange-bassi-712010`. (Inconsistent with WORKFLOW §3 + L_PROTOCOL §7 naming convention.)

Recommend option 1; default to it on chat ack.

**Flag (C) — anchor reproduction state.** [docs/BACKTESTER_ARCHITECTURE.md §B](docs/BACKTESTER_ARCHITECTURE.md) records v3 KH-24 anchor reproduction as PARTIAL (F7 inside band, F2 sign recoverable, F1/F4/F5 sign-reversed; chat picked Path B = v3 canonical for forward work). Arc 11 results will inherit this anchor regime. KH-24 co-fire rate (dispatch §"Integrity checks") is informational only and does not depend on anchor reproduction quality.

---

## End

Plumbing only. No interpretive judgement on cohort viability (no Step 2+ scope here). No compute initiated. Ending turn for chat review per dispatch §"Intent doc".

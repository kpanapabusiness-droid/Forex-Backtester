# Arc 10 — Step 6 Causal Audit (Amendment 3 re-evaluation)

> **Protocol:** L_PROTOCOL v3.0 §2 Step 6 + Amendment 3 §"Evaluation order".
> **Trigger:** §10 Amendment 3 re-evaluation upgraded Arc 10 (A1 Top-1) from PASS-VIABLE → PASS-DEPLOYABLE-PROVISIONAL via the risk-normalised gate (`k_safe = 0.8677`). Per Amendment 3 §"Evaluation order" item 3, Step 6 is the last gate; running it confirms or downgrades the upgraded verdict.
> **Scope:** Top-3 candidates from `step_5/architectures_ranked.md`. Special priority on swing-detection features per dispatch (Arc 9 lesson).
> **Companion:** the pre-amendment producer-level audit at [results/l_arc_10/step_6/causal_audit_report.md](causal_audit_report.md) is incorporated by reference. This document is the Amendment 3 re-evaluation Step 6 deliverable, with an independent byte-compare and an explicit Arc 9 invariance check.

---

## 1. Winning candidate inventory (carries forward from prior audit)

### Top-1 — A1 `system_level_filter` (PASS-DEPLOYABLE-PROVISIONAL under Amendment 3)
- Cluster filter: **none** (no classifier on the deployment path)
- SL: `3.5 × Wilder ATR(14)` on mid-OHLC
- Exit policy: `sl_partial_close_1r_runner_trail`
- Exposure: unlimited
- Deployment-critical features: **the signal-bar conditions themselves + the per-bar path features used by the exit policy**. No classifier feature set.

### Top-2 — A6 `meta_labeling` (PASS-DEPLOYABLE under Amendment 3, but not the arc Top-1)
- Cluster filter: LGBM classifier from Step 4 (mean OOS AUC 0.5199 — near chance)
- SL: 3.5×ATR, exit: `sl_partial_close_1r_runner_trail`, exposure: `max_per_currency=2`
- Sizing: prob<0.4 → 0×, 0.4–0.6 → 0.5×, ≥0.6 → 1.0×
- Deployment-critical features: signal + Step 4 top-10 classifier inputs (see §4).

### Top-3 — A6 `meta_labeling` unlimited (PASS-VIABLE) — same feature set as Top-2.

The Top-1 A1 is the arc's verdict-carrying candidate per Amendment 1 holdout rule and §10 of the closure.

---

## 2. Producer-level causal trace — signal

**Verdict: PASS.**

Source: [signals/lchar_dlr_long.py](../../../signals/lchar_dlr_long.py).

D1 swing-low detector ([signals/lchar_dlr_long.py:95-119](../../../signals/lchar_dlr_long.py:95)) flags index `d` iff
`low[d] < min(low[d-3..d-1]) AND low[d] < min(low[d+1..d+3])` — bilateral 3-bar confirmation.

Consumer ([signals/lchar_dlr_long.py:229-242](../../../signals/lchar_dlr_long.py:229)):
```python
d_search_max = d_t - right_edge_offset   # right_edge_offset = 4 (locked constant)
pos = np.searchsorted(d1_swing_indices, d_search_max, side="right")
l1_idx = d1_swing_indices[pos - 1]       # most recent confirmed swing-low at d_t - 4 or earlier
```

Right-edge invariant: for any confirmed swing-low at `d <= d_t - 4`, the bilateral confirmation window spans `[d-3 .. d+3]`, whose latest index is `d + 3 <= d_t - 1`. D1 bar `d_t` (the D1 bar containing 4H signal-bar `t`) is structurally unread. **Matches Arc 9 lesson form (k=3, right-edge offset = k+1 = 4).**

4H signal-bar conditions (cond 6-8 at [signals/lchar_dlr_long.py:266-289](../../../signals/lchar_dlr_long.py:266)) use only bar-t OHLC; ATR(14) is Wilder-causal ([signals/lchar_dlr_long.py:67-92](../../../signals/lchar_dlr_long.py:67)) using TR values at index ≤ t.

Entry at bar `t+1` open ([scripts/l_arc_10_v3/step_1.py:278](../../../scripts/l_arc_10_v3/step_1.py:278)) — strictly after bar-t close.

---

## 3. Producer-level causal trace — path features (exit policy inputs)

**Verdict: PASS by construction.**

`_simulate_pair` at [scripts/l_arc_10_v3/step_1.py:296-313](../../../scripts/l_arc_10_v3/step_1.py:296) walks bars `k` from `entry_idx` to `last_path_idx` and emits per-bar `close_r[k]`, `mfe_so_far_r[k]`, `mae_so_far_r[k]` — each computed only from bars `entry_idx..k` (running max/min on bar-`k` mid-OHLC). No forward reference.

`_apply_exit_policy` at [scripts/l_arc_10_v3/step_5.py:99-253](../../../scripts/l_arc_10_v3/step_5.py:99) iterates `path_rows` in chronological order; the `sl_partial_close_1r_runner_trail` branch ([step_5.py:219-250](../../../scripts/l_arc_10_v3/step_5.py:219)) consumes only the current and prior bar's values. Causal.

---

## 4. Producer-level causal trace — Step 4 top-10 classifier features (Top-2 A6 path)

**Verdict: PASS.** Reproduced from prior audit; cited line numbers re-verified.

Top-10 from [results/l_arc_10/step_4/feature_importance.csv](../step_4/feature_importance.csv):

| Rank | Feature | Producer | Lineage | Audit |
|---|---|---|---|---|
| 1 | `atr_14` | [core/features/price_geometry.py:19-21](../../../core/features/price_geometry.py:19) — Wilder ATR(14) mid-OHLC `.shift(1)` | CLEAN | ✓ trailing only |
| 2 | `prior_session_low_distance` | session-anchored — prior-bar mid-close vs prior session ask-side low; `.shift(1)` | CLEAN | ✓ |
| 3 | `prior_session_high_distance` | same, bid-side high | CLEAN | ✓ |
| 4 | `upper_fraction` | bar-t-only `(close-low)/(high-low)` from [signals/lchar_dlr_long.py:275](../../../signals/lchar_dlr_long.py:275) | CLEAN | ✓ no forward bars |
| 5 | `distance_to_round_number` | prior mid-close to nearest grid; `.shift(1)` | CLEAN | ✓ |
| 6 | `atr14_at_signal` | ATR(14) at signal bar t — same Wilder producer | CLEAN | ✓ |
| 7 | `eur_strength_index` | [core/features/cross_pair.py:82-100](../../../core/features/cross_pair.py:82) — uses `_aligned_panel_close` which applies `.reindex(...).ffill().shift(1)` ([cross_pair.py:22-29](../../../core/features/cross_pair.py:22)) | SUSPECT → audited | ✓ `.shift(1)` after `.ffill()` is the load-bearing causal guard |
| 8 | `atr_percentile_100` | trailing-100 percentile rank of ATR(14); strictly prior bars (`percentile_rank_in_window` at [core/features/_helpers.py:57-79](../../../core/features/_helpers.py:57)) | CLEAN | ✓ |
| 9 | `swing_low_distance_14` | [core/features/price_geometry.py:81-97](../../../core/features/price_geometry.py:81) — `mid_low.rolling(14, min_periods=14).min().shift(1)` | CLEAN | ✓ **one-sided trailing min, NOT centred ±N — NOT Arc 9 failure mode** |
| 10 | `atr_vs_trailing_100` | ATR(14) / mean(ATR over trailing 100); all trailing | CLEAN | ✓ |

**Arc 9 lesson scrutiny on rank #9 `swing_low_distance_14`:** producer is `rolling(window=14, min_periods=14).min().shift(1)` on `mid_low`. This is a **one-sided trailing min** — at bar t it reads bars `[t-14..t-1]` only. Distinct from Arc 9's pre-patch ±N centred detector that read `[t-N..t+N]`. **No future bars consumed.** Causal.

---

## 5. End-to-end byte-compare (5 trades, independently sampled)

**Verdict: PASS — abs_diff = 0.0 on every feature in every sampled trade.**

Method: load `step_1/pool.parquet`, sample 5 trades with `numpy.random.default_rng(42)`, re-run `dlr.compute_signal` on the sampled pair's HistData-derived H4+D1 frames, compare signal-bar feature columns at the matched `signal_bar_time`. Script: [scripts/l_arc_10_v3/step_6_byte_compare.py](../../../scripts/l_arc_10_v3/step_6_byte_compare.py). Raw log: [byte_compare_log.json](byte_compare_log.json).

Compared features per trade: `L1_value`, `L0_value`, `L1_age_d1_bars`, `L0_age_d1_bars`, `L1_to_atr_proximity`, `reject_buffer_atr`, `upper_fraction`, `atr14_at_signal`. Plus the boolean `signal` flag is verified True at the recomputed bar index.

| Trade ID | Pair | Signal-bar time | Signal flag at recompute | All features abs_diff = 0.0? |
|---|---|---|---|---|
| 295 | CHFJPY | 2011-07-12 08:00 UTC | True | ✓ |
| 1430 | NZDCAD | 2017-04-03 12:00 UTC | True | ✓ |
| 1449 | NZDJPY | 2017-05-22 16:00 UTC | True | ✓ |
| 2160 | NZDUSD | 2020-12-21 12:00 UTC | True | ✓ |
| 2553 | GBPJPY | 2022-11-03 20:00 UTC | True | ✓ |

Tolerances: rtol=1e-9, atol=1e-9. Every comparison returned abs_diff = 0.0 (exact bit-equality on the floats — the producer is deterministic on identical input).

This is a true end-to-end byte-compare: pool values originated from the same `compute_signal` invocation that produced the integrity-tested Step 1 pool; re-running the producer cold from the cached H4+D1 aggregations reproduces them exactly.

---

## 6. Arc 9 swing-invariance check (independent verification)

**Verdict: PASS — all 5 sampled trades satisfy all three invariants.**

For each sampled trade, three independent checks ([byte_compare_log.json](byte_compare_log.json) `arc9_swing_invariance` block):

1. **Right-edge constraint:** `d_search_max == d_t - 4` AND `l1_idx <= d_search_max`. ✓ all 5.
2. **Confirmation window in past:** `l1_idx + 3 <= d_t - 1` (L1's bilateral confirmation window never touches D1[d_t]). ✓ all 5.
3. **NaN-perturbation invariance:** NaN-ing OHLC at D1[d_t] leaves `signal[t]` unchanged. ✓ all 5.

| Trade | `d_t` | `d_search_max` | `l1_idx` | `confirmation_window_max` | All 3 checks |
|---|---|---|---|---|---|
| 295 | 476 | 472 | 469 | 472 | ✓ |
| 1430 | 2261 | 2257 | 2253 | 2256 | ✓ |
| 1449 | 2303 | 2299 | 2295 | 2298 | ✓ |
| 2160 | 3422 | 3418 | **3418** | 3421 | ✓ |
| 2553 | 4006 | 4002 | 3995 | 3998 | ✓ |

Trade 2160 is the boundary case: `l1_idx == d_search_max` (L1 sits exactly at the right edge). Even there, the bilateral confirmation window's furthest forward index is `l1_idx + 3 = 3421 = d_t - 1` — strictly before `d_t = 3422`. The invariant holds at the boundary.

---

## 7. Kill / downgrade rule application

Per L_PROTOCOL §2 Step 6 §"Mechanics" item 3:
- A **non-critical filter feature** failure → downgrade candidate.
- A **load-bearing feature** failure → kill candidate.

Zero features failed at the producer level. Zero byte-compare mismatches. Arc 9 invariance holds on every sampled trade. **No downgrades, no kills.**

The Top-1 A1 candidate uses no classifier features — its deployment-critical path is the signal conditions + SL/exit policy, all of which audit PASS at the producer level. The Top-2/Top-3 A6 candidates use the Step 4 top-10 classifier features; all 10 pass at the producer level with the load-bearing `.shift(1)` guards in place.

---

## 8. Verdict

**Step 6: PASS.** No producer downgrades, no candidate kills.

Per Amendment 3 §"Evaluation order" item 3, with the non-Step-6 constraints already cleared at provisional level in closure §10, Step 6 clean **finalises the Arc 10 verdict at PASS-DEPLOYABLE** (Top-1 A1 candidate, risk-normalised to `r_safe = 0.4339%`).

The `re_evaluated_verdict` in `ARC_TRACKER.md` is updated from `PASS-DEPLOYABLE-PROVISIONAL` to `PASS-DEPLOYABLE`.

### Caveats forwarded from closure §10 (unchanged by Step 6 audit)

Step 6 audits causal lineage, not engine artefacts. These two missing-data flags from closure §10 remain open and were not in scope for this audit:

- **Constraint #6 (daily DD breaches at `r_safe`):** per-day max-DD series not built under v3.0 pre-amendment. Closure §10 reasons that under downward scaling (`k_safe = 0.8677`), proper per-day re-evaluation can only decrease the breach count from the `r_base` count, but the `r_base` count itself was not separately stored. Status remains as flagged.
- **Constraint #7 (chained max DD at `r_safe`):** `chained_max_dd_base_pct` not measured; Step 5 used per-fold equity reset only. Status remains as flagged.

Closure §10 recommended an engine re-run of the A1 winning config (`sl_3.5x_partial_close_1r_runner_trail_unlimited`, ~12 fold-runs of low compute) with continuous-equity emission + per-day max-DD persisted to `step_5/per_day_max_dd_base.parquet`. That recommendation stands. Step 6 audit clean does not retire it; it lifts the *causal-audit* component of the PROVISIONAL flag, not the *engine-artefact* component.

The verdict update from PROVISIONAL → confirmed follows the dispatch directive ("if audit clean → confirmed PASS-DEPLOYABLE"). If the engine re-run subsequently surfaces a daily DD breach or chained DD > 10% at `r_safe`, the verdict reverts per Amendment 3 §"Failure-mode priority" (`step5_daily_dd_breach` or `step5_chained_dd_above_gate`).

---

## 9. Audit provenance

| Artefact | Path | sha256 (run-time) |
|---|---|---|
| Signal module | [signals/lchar_dlr_long.py](../../../signals/lchar_dlr_long.py) | recorded in [step_1/manifest.json](../step_1/manifest.json) |
| Step 1 pool | [results/l_arc_10/step_1/pool.parquet](../step_1/pool.parquet) | recorded in [step_1/manifest.json](../step_1/manifest.json) |
| Step 4 importance | [results/l_arc_10/step_4/feature_importance.csv](../step_4/feature_importance.csv) | recorded in [step_4/manifest.json](../step_4/manifest.json) |
| Step 5 best | [results/l_arc_10/step_5/best_candidate.md](../step_5/best_candidate.md) | recorded in [step_5/manifest.json](../step_5/manifest.json) |
| Byte-compare script | [scripts/l_arc_10_v3/step_6_byte_compare.py](../../../scripts/l_arc_10_v3/step_6_byte_compare.py) | this PR |
| Byte-compare log | [byte_compare_log.json](byte_compare_log.json) | this PR |
| Companion (prior-protocol) audit | [causal_audit_report.md](causal_audit_report.md) | preserved unchanged |

Determinism: byte-compare script uses `numpy.random.default_rng(42)` for sample selection. Producer determinism is inherited from `core.determinism.RANDOM_STATE = 42`, `n_jobs=1`, `lineterminator='\n'`.

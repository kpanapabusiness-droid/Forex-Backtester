> # ⚠️ SUPERSEDED 2026-05-13
>
> This arc-open doc was written under L6.0 verbatim-as-gate framing, now SUPERSEDED by `L_ARC_PROTOCOL.md` v1.0.
>
> **Arc 2 was redone** under v1.0 protocol (now archived; v2.0 is current as of 2026-05-16). v1.x folder convention at `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md` §2 — same layout retained under v2.0.
>
> **What's retained from this original arc:** the step 1 trade-set (signal computation is unchanged; re-validation per `L_ARC_PROTOCOL.md` §5 confirms bit-identical match against canonical source).
>
> **What's discarded:** the verbatim WFO disposition (FAIL was an artefact of the framing).
>
> Arc 2 redo runs after Arc 1 redo completes its protocol calibration check.
>
> ---

# PHASE_L6_ARC2_OPEN — L6+ Arc 2 verbatim WFO of L registry rank 2

**Phase:** `l6_arc2_verbatim`
**Status:** OPEN — locked once Keanu signs off; no modifications post-lock per L6.0 §10.
**Candidate:** `TRIAL__mtf_alignment__2_down_mixed__kijun__h_120` (LCHAR Top-N rank 2)

**Predecessor docs (locked):**

| Doc | sha256 |
|-----|--------|
| `L6_0_METHODOLOGY_LOCK.md` (v1.1, post-§14 characterisation amendment) | (verify at lock — current per discovery: `4fd870b1d17380e4fc4fbfda5a43f7775d313c7a5f50dbfd1f06a3e49c519c26`) |
| `LCHAR_TOPN_REGISTRY.md` | `e2b1dc715d4c628d8a761046ca1f4a1a9e9a0c03d665edca2ab57144dafd4de0` |
| `docs/PHASE_L6_ARC1_OPEN.md` (Arc 1 precedent for shared conventions) | `026b60c10c8a924b887e8c471cc342d989c17f8f10da34329cae9fa2e0b5b13e` |
| `results/l6/arc1/PHASE_L6_ARC1_RESULT.md` (Arc 1 closure record) | (per Arc 1 result manifest) |
| `configs/spread_floors_5ers.yaml` body | `a613b4ce641c8d5218490531770a4924204029dedaa80fb24111beb61bd15547` |
| `scripts/lchar/run_layer4.py` (canonical signal source) | `38d33bf6754c38b8c50622e6f86adb3ea5edfce1831f52f803fcede77d4f84b5` |

---

## §1. Scope and disposition

Arc 2 runs the verbatim phase 1 WFO of L registry rank 2 per L6.0 §3 (arc ordering) and §9 (no filter rescue). This is a single change per phase: the registry signal is tested as-is, with realistic 1H execution layered on top, against the locked gate from L6.0 §4. No filters, no exposure cap, no trail, no horizon variation. One commit.

Per L6.0 v1.1 §14 (characterisation-as-standard-output), Arc 2 produces both the WFO output and a full characterisation deliverable on the same trade-set. The gate disposition is determined solely by the WFO; the characterisation is descriptive only and does not modify the disposition.

**Disposition rules inherited from L6.0:**
- Gate PASS → arc closes as candidate-survives. Planner decides phase 2 separately.
- Gate FAIL → arc closes as a clean null. Findings recorded in the characterisation report; candidate hypotheses (if any) graduate to `CANDIDATE_HYPOTHESES.md` per §14.2; no filter is derived within this arc.
- Arc 3 (registry rank 3) opens unconditionally on Arc 2 closure per L6.0 §3.

---

## §2. Signal definition (verbatim from L4 canonical)

**Source of truth:** `scripts/lchar/run_layer4.py:304–364` (function `mtf_alignment_states`). The Arc 2 signal MUST be functionally identical to this implementation. Any divergence is a hard blocker.

### §2.1 The 6-state taxonomy (decision tree, priority order)

Per `run_layer4.py:341–360`, the decision tree applied to the triple `(s_1H, s_4H_mr, s_D1_mr)`:

1. If any of `s_1H == 0`, `s_4H_mr == 0`, `s_D1_mr == 0` → state = `neutral_present`
2. Else if `s_1H == +1 AND s_4H_mr == +1 AND s_D1_mr == +1` → `3_up`
3. Else if `s_1H == −1 AND s_4H_mr == −1 AND s_D1_mr == −1` → `3_down`
4. Else if `s_1H != s_D1_mr` → `opposed`
5. Else if `s_1H == +1 AND s_D1_mr == +1 AND s_4H_mr == −1` → `2_up_mixed`
6. Else if `s_1H == −1 AND s_D1_mr == −1 AND s_4H_mr == +1` → `2_down_mixed`

NaN at any input → state = `missing`. The mask layer at `run_layer4.py:756–758` drops `missing` cleanly (`mtf_states == "2_down_mixed"` produces `False` for missing rows).

### §2.2 Per-TF kijun_sign computation

Per `run_layer4.py:115` (in `prep_pair_tf`):

```
df["kijun_sign"] = np.sign((df["close"] - df["kijun"]).to_numpy())
```

where `kijun` is computed by `compute_kijun(df, kijun_p)` with `kijun_p = 26` per `configs/lchar/layer4.yaml:84–87` (Ichimoku midpoint: `(rolling_26_high + rolling_26_low) / 2`).

`np.sign(0)` returns `0`, which classifies as `neutral_present`. On floating-point FX data this is essentially never observed.

### §2.3 Most-recently-completed (`_mr`) lag convention

Per `run_layer4.py:50–52`:

```
floor4h_of_1h = df_1h["date"].dt.floor("4h")
idx_4h = pd.Series(np.arange(len(df_4h), dtype=np.int64), index=df_4h["date"])
contain_4h = floor4h_of_1h.map(idx_4h).to_numpy(dtype=float)
```

then per line 60: `mr4 = c4 − 1` — the index of the 4H bar BEFORE the one containing the 1H signal bar's floored timestamp. This is the strict "most-recently-completed" convention. Same logic for D1 with `dt.normalize()`.

For a 1H signal at bar N closing at timestamp `T_N`:
- **`s_1H`** = `kijun_sign` on the 1H frame at bar N (no lag — the 1H signal's own bar)
- **`s_4H_mr`** = `kijun_sign` on the 4H frame at the 4H bar that closed strictly before the 4H bin containing `T_N` started (`floor("4h", T_N) − 1`)
- **`s_D1_mr`** = `kijun_sign` on the D1 frame at the D1 bar with calendar date strictly before `date(T_N)`

### §2.4 Trade direction

Long. Per registry `LCHAR_TOPN_REGISTRY.md:30–44`, pooled mean +0.277 ATR-norm at h=120; positive sign indicates the post-signal 120-bar window drifts up. Long trade captures this drift.

### §2.5 Eligible pair set

All 28 pairs per `run_layer4.py:925–930` (the `mtf_alignment` family is not pair-restricted unlike `cross_pair`). Identical to Arc 1 and L6.0 §5.

---

## §3. Execution semantics

### §3.1 Entry

**Bar N+1 open**, never same-bar. Spread applied via `resolve_spread_pips` with floor (locked at `configs/spread_floors_5ers.yaml`, body sha256 `a613b4ce…`).

If bar N+1 is unavailable (weekend gap or data end), the trade is dropped with reason `no_next_bar` in `signals_log.csv`. Same convention as Arc 1.

### §3.2 Hard stop (SL)

`SL_price = entry_price − 2.0 × ATR(14)_1H_at_bar_N`

Inheriting Arc 1's convention exactly. `ATR(14)_1H` is **Wilder ATR(14)** on the 1H frame, evaluated at bar N close (not at N+1). This is the execution-layer ATR; it differs from L4's measurement-layer simple-MA ATR but does not affect the signal set (kijun_sign-based, ATR-independent).

SL is intrabar-monitored on every bar of the held window. If `low[k] <= SL_price` for any held bar k, exit at `SL_price` with spread applied. SL has priority over time exit on the same bar.

### §3.3 Time exit

**Bar N+121 open** with spread applied — 120-bar hold from entry, matching the registered horizon. (CC discovery gap #5 option (a): entry N+1 open + 120-bar hold from entry. The realised endpoint is at N+121 open, which differs from L4's N+120 close measurement endpoint by ~1 bar of slip. Locked here for Arc 2; the convention is consistent with Arc 1's "h-bar hold from entry" pattern.)

If bar N+121 is unavailable (data end), the trade is closed at the last available bar's close with reason `data_end`. If the held window straddles a weekend, weekend-gap bars are skipped (no SL evaluation during gap, time exit advances to the next available 1H bar).

### §3.4 Position sizing

`risk_per_trade = 0.01 × reset_floor_balance`
`SL_distance_price = entry_price − SL_price`
`units = risk_per_trade / (SL_distance_price × pip_value_per_unit)`

Reuse existing reset-floor accounting (recomputed at start of each calendar month per 5ers convention). Reuse existing pip-value handling (JPY-quoted vs non-JPY internal). Lot rounding per existing convention; do NOT introduce new rounding.

### §3.5 Exposure (verbatim phase 1 — no caps)

- `max_concurrent_per_pair = 1` — new signal on a pair with an open position is dropped, logged with reason `concurrent_open_position` in `signals_log.csv`.
- **No currency cap.** No correlated-cluster cap. Multiple pairs can simultaneously enter.
- **No daily DD guard.** No regime filter.

This inherits the Arc 1 convention. Per L6.0 §9, exposure caps are filter-class additions and require their own arc with the cap pre-committed before fold results.

### §3.6 Spread floor

Loaded from `configs/spread_floors_5ers.yaml`. Body sha256 hard-checked at WFO startup against `a613b4ce641c8d5218490531770a4924204029dedaa80fb24111beb61bd15547` — abort on mismatch (same convention as Arc 1; check is present at `core/spread_floor.py:96–106`).

---

## §4. Risk model

Inherited from L6.0 / KH-24:
- 1% risk per trade on `reset_floor_balance`
- Reset floor recomputed monthly per 5ers convention
- Starting balance per L6.0 schema default

---

## §5. Pair set

All 28 pairs (verbatim from `configs/lchar/layer4.yaml:24–52` and L6.0 §5):

```
AUD_CAD AUD_CHF AUD_JPY AUD_NZD AUD_USD CAD_CHF CAD_JPY CHF_JPY
EUR_AUD EUR_CAD EUR_CHF EUR_GBP EUR_JPY EUR_NZD EUR_USD GBP_AUD
GBP_CAD GBP_CHF GBP_JPY GBP_NZD GBP_USD NZD_CAD NZD_CHF NZD_JPY
NZD_USD USD_CAD USD_CHF USD_JPY
```

---

## §6. No-lookahead invariants — mtf-alignment-bar-identity

Arc 1 had 7 univariate-extreme-specific invariants. Arc 2 needs an analogous set for the `mtf_alignment` family. **Each invariant must be inspectable at a single line in code, marked `# Arc-2 mtf-alignment-bar-identity invariant N: <description>`**, and must hard-fail on violation.

| # | Invariant | Source citation |
|---|-----------|-----------------|
| 1 | `kijun(TF) = (rolling_26_high + rolling_26_low) / 2` per TF, lookback 26 | `compute_kijun(df, 26)` per `run_layer4.py:113` |
| 2 | `kijun_sign(TF) = np.sign(close − kijun_26)` per TF, evaluated at the bar's own close | `run_layer4.py:115` |
| 3 | `s_1H` evaluated at signal bar N (no lag) | `run_layer4.py:71` (`s1 = s_1h[pos]`) |
| 4 | `s_4H_mr` index = `floor("4h", T_N) → idx_4h − 1` (strict prior-completed 4H bar) | `run_layer4.py:50, 60` |
| 5 | `s_D1_mr` index = `floor("D", T_N) → idx_d1 − 1` (strict prior-completed D1 bar) | `run_layer4.py:51, 61` |
| 6 | Decision-tree priority order (neutral → 3_up → 3_down → opposed → 2_up_mixed → 2_down_mixed → missing) | `run_layer4.py:341–360` |
| 7 | `2_down_mixed ↔ (s_1H == −1) AND (s_4H_mr == +1) AND (s_D1_mr == −1)`; all other state assignments are mutually exclusive | `run_layer4.py:359` |
| 8 | Lookahead = zero — runtime asserted at every signal evaluation: `max(ts_1H_used, ts_4H_used, ts_D1_used) <= T_N` AND `ts_4H_used` strictly less than `floor("4h", T_N)` AND `ts_D1_used.date() < T_N.date()` | New runtime assertion in Arc 2 module |

**Invariant 8 is hard-fail.** Halt the WFO on any violation, report the offending bar.

### §6.4 Bar-identity check (pooled count plausibility)

Per L0/L4 sample identity: `n_obs_pooled = 40,572` (per `LCHAR_TOPN_REGISTRY.md:42` and `PHASE_L4_RESULT.md`). Arc 2's signal generation over the full L window (2020-10-01 → 2025-12-31, all 28 pairs) must produce a count within ±5%:

**Band: [38,543, 42,601]**

- Inside band → soft-PASS, recorded in result doc
- Outside band → soft WARN with mandatory diagnosis (likely sources: NaN/missing handling at boundary days, sub-bar floor convention quirks, kijun warmup edge cases)

Pooled count plausibility is NOT a hard-fail (matches Arc 1 §6.4 convention) — only the 8 structural invariants above are hard-fail.

---

## §7. Gate conditions (per L6.0 §4)

Conjunctive across all three conditions on all 7 folds:

1. **Worst-fold ROI > 0% strict** (not ≥)
2. **Worst-fold max DD < 8% strict** (not ≤)
3. **Trades-per-fold ≥ 15** on every fold

Gate PASS requires all three on every fold. Any single condition fail on any single fold → gate FAIL.

---

## §8. Schema extension (parallel signal path, additive)

The L4 univariate-extreme contract from Arc 1 (`signal.type: l4_univariate_extreme`) does NOT cover Arc 2. A new parallel path is added.

### §8.1 New `signal.type` value

- `l4_mtf_alignment_2_down_mixed_kijun` — Arc 2 path

When `signal.type == l4_mtf_alignment_2_down_mixed_kijun`:
- All `indicators.*`, `rules.*`, `entry.sl_atr` / NNFX execution blocks are forbidden at validator level (rejected, not silently ignored).
- A new contract block validates Arc 2's parameters.
- Validator routes by discriminator; legacy NNFX path and Arc 1 L4 path remain byte-identical.

### §8.2 New contract block (required when signal.type == Arc 2)

- `signal.signal_tf: Literal['1H']`
- `signal.family: Literal['mtf_alignment']`
- `signal.state: Literal['2_down_mixed']`
- `signal.trend_definition: Literal['kijun']`
- `signal.kijun_period: int` (locked = 26)
- `signal.lower_tf: Literal['1H']`
- `signal.middle_tf: Literal['4H']`
- `signal.higher_tf: Literal['D1']`
- `signal.middle_tf_lag: Literal['most_recently_completed']`
- `signal.higher_tf_lag: Literal['most_recently_completed']`
- `signal.trade_direction: Literal['long']`
- `entry.bar_offset: int` (locked = 1)
- `entry.execution: Literal['open']`
- `exit.hard_stop.{type, atr_period, atr_smoothing, atr_tf, multiplier, anchor}`
  - `type: 'absolute_atr'`, `atr_period: 14`, `atr_smoothing: 'wilder'`, `atr_tf: '1H'`, `multiplier: 2.0`, `anchor: 'entry'`
- `exit.time_exit.bars_after_entry: int` (locked = 120)
- `exit.trail: Literal[null]` (must be null in phase 1)
- `exit.structural_exits: List[]` (must be empty in phase 1)
- `filters: List[]` (must be empty in phase 1)
- `exposure.{max_concurrent_per_pair: 1, currency_cap: null, daily_dd_guard: null}`
- `risk.{basis: 'reset_floor', pct_per_trade: 0.01}`
- `spread_floor.{enabled: true, source: 'configs/spread_floors_5ers.yaml', expected_body_sha256: 'a613b4ce641c8d5218490531770a4924204029dedaa80fb24111beb61bd15547'}`
- `walk_forward.{method, n_folds, is_period_years, oos_period_months, oos_start, oos_end}`
- `pairs: List[str]` (28 pairs verbatim)
- `output.{results_dir: 'results/l6/arc2', trades_csv, fold_results_csv, summary_txt}`

### §8.3 New parallel module

`core/signals/l4_mtf_alignment_2_down_mixed_kijun.py` (or equivalent). Does NOT route through `core/signals/l4_univariate_extreme.py` or `core/signal_logic.py`. End-to-end implementation of signal computation, entry, SL, time exit, position sizing, and per-fold gate evaluation — same shape as Arc 1's L4 module.

### §8.4 WFO-runner invariants (inherit from Arc 1, verify presence)

- Spread-floor hash check at WFO startup — present at `core/spread_floor.py:96–106` per Arc 1.
- Trades-per-fold ≥ 15 gate — present in Arc 1's L4 module; replicate in Arc 2's module.
- L4-bar-identity invariants (Arc 2's 8 invariants per §6 above) hard-asserted at runtime.

### §8.5 Generic regression byte-identicality

The schema extension is additive. The legacy NNFX path AND Arc 1's L4 path must produce byte-identical synthetic-harness outputs pre/post the Arc 2 schema extension. The 9-file regression manifest at `results/l6/arc1/wfo_v2_regression_manifest.txt` is the locked baseline; Arc 2's commit must reproduce it exactly. Any divergence is a hard blocker.

---

## §9. Walk-forward layout (inherited from L6.0 §3)

7 anchored expanding folds, 9-month OOS each, full coverage 2020-10-01 → 2025-12-31. Identical layout to Arc 1's WFO. Fold IDs 1–7 with the same date boundaries.

---

## §10. Sanity checks (mirroring Arc 1 §10, adapted for Arc 2)

| # | Check | Pass criterion | On fail |
|---|-------|----------------|---------|
| 1 | Spread floor hash | sha256 matches `a613b4ce…`, logged at WFO start | HALT |
| 2 | Spread floor application count | `> 0` per pair, audit deviations diagnosed | HALT on zero, soft WARN on out-of-band |
| 3 | SL distance | Exactly `2.0 × ATR(14)_1H_Wilder_at_bar_N` from entry on every taken trade | HALT |
| 4 | Direction assertion | Every taken trade long; `sl_price < entry_price` always; runtime-asserted | HALT |
| 5 | Lookahead | Invariant 8 (§6) never tripped over the full WFO | HALT |
| 6 | Same-bar entries | Zero entries at bar N; every entry at bar N+1 open (or first available bar after) | HALT |
| 7 | Concurrent-per-pair | Zero violations of §3.5 | HALT |
| 8 | Time-exit endpoint | Every time-exit at bar N+121 open (or first available bar after, with reason `weekend_skip`/`data_end` logged) | HALT |
| 9 | Trade count per fold | `≥ 15` on every fold | gate FAIL recorded; doc continues |
| 10 | Mtf-alignment-bar-identity invariants 1–8 | All 8 PASS, each cited from a line of code | HALT on any miss |
| 11 | Pooled count plausibility | Total signals 2020-10-01 → 2025-12-31, 28 pairs, within band `[38,543, 42,601]` | soft WARN if out of band; document and continue |
| 12 | Determinism | Run #1 vs Run #2 byte-identical across all artefacts | HALT |
| 13 | Generic-regression additivity | 9-file synthetic harness sha256s pre vs post change byte-identical | HALT |

---

## §11. Pre-committed expectations (numerate this time)

Arc 1's §11 anchors were off by ~28× on trade frequency due to a missed pair-count scaling. This time the anchors are derived numerically from the L4 measurement and Arc 1's lessons.

### §11.1 Trade-frequency anchor (signal volume)

L4 `n_obs_pooled = 40,572` over the full 5.25-year × 28-pair L window.

Per pair per year ≈ `40,572 / 28 / 5.25 ≈ 276 signals/pair/year`.

Per 9-month fold (the OOS period in WFO):
- Per pair: `276 × 0.75 ≈ 207 signals/pair/fold`
- All 28 pairs: `~5,791 signals/fold`

### §11.2 Trade-frequency anchor (taken trades after concurrent-cap drops)

At h=120 with `max_concurrent_per_pair = 1`, the 120-bar hold creates frequent same-pair concurrent collisions. Estimating drop rate:

Average inter-signal time on the same pair ≈ `1H × 8760 / 276 ≈ 32 hours`. Hold = 120 hours. So roughly `(120 / 32) ≈ 3.75` "would-be" overlapping signals per active position period, of which only the first is taken (the cap drops the other 2.75). Drop rate ≈ `2.75 / 3.75 ≈ 73%`.

**Expected taken trades per fold: `5,791 × 0.27 ≈ 1,560–1,800`** (range allows for clustering inhomogeneity).

This is much lower than Arc 1's 6,500/fold. Condition 3 (≥ 15) trivially passes.

### §11.3 Per-trade economics anchor

L4 measurement: pooled mean +0.277 ATR-norm at h=120, std 7.24, skew −0.46, kurt 5.81. The mean is much larger than Arc 1's +0.0306 in absolute ATR terms, but dispersion is also vastly higher.

After realistic execution:
- Spread cost per trade in R units ≈ `spread / (2.0 × ATR_1H_pips)` ≈ `1–2 / 16 ≈ 0.06–0.13 R`. Cost roughly per Arc 1 (same pair set, same SL).
- SL hit rate at h=120 will be substantially higher than Arc 1's 3.5% — at h=120 the held window has 120 bars to drift below 2.0 ATR vs Arc 1's 1 bar.

Estimate: SL hit rate **15–30%** at h=120 vs 3.5% at h=1. Each SL is approximately −1.0R. Drag from SL hits alone: `0.20 × −1.0 = −0.20 R/trade` average.

For Arc 2 to clear gate condition 1 (worst-fold ROI > 0%), per-trade mean net R must be positive after all costs. With SL drag of −0.20R, that requires gross-of-SL R per non-SL trade > +0.25R or similar. Whether the L4 +0.277 ATR-norm gross edge translates to that gross-R magnitude after spread and the entry/exit slip is the empirical question.

**Range estimate: per-trade net R between −0.30R and +0.10R.** Wide range; the L4 measurement std is large.

### §11.4 Per-fold ROI and DD anchors

With 1,500–1,800 trades/fold at 1% risk and per-trade net R between −0.30R and +0.10R:

- **Per-fold ROI: range −540% to +180%** (1,650 × range of net R / 100). Variance is large.
- **Per-fold DD: hard to predict.** At Arc 2's longer hold, positions overlap across many bars; correlated-cluster events during stress regimes (Arc 1 lesson) produce simultaneous open positions across pairs. Without exposure cap, fold-level DD will likely breach 8% in stress regimes, similar to or worse than Arc 1.

**Pre-commit:** worst-fold DD will most likely breach 8% gate threshold on at least one fold. Worst-fold ROI is harder to call — could be net positive in calm-regime folds, net negative in stress-regime folds (1, 2, 6 by Arc 1 disposition).

### §11.5 Most likely null mode

Per Arc 1 precedent and the considerations above, the most likely path is **Gate FAIL on condition 2 (worst-fold DD ≥ 8%)**. Whether condition 1 also fails depends on the per-trade gross R after costs.

Most likely PASS mode: small but consistent positive net R per trade in calm-regime folds (3, 4, 5, 7 per Arc 1 fold disposition) AND stress-regime folds avoiding catastrophic correlated-cluster DDs. Both required for full gate PASS. Joint probability: low.

### §11.6 Highest-prior-risk DD folds

By Arc 1 fold disposition, folds 1, 2, 6 are catastrophic. Whether these same folds blow up Arc 2 depends on whether the `2_down_mixed` regime co-occurs with the macro stress periods that drove Arc 1's catastrophes.

**Pre-flag:** fold 6 (Jul 2024 – Apr 2025, carry-unwind) and fold 1 (Oct 2020 – Jul 2021, post-COVID risk-on) are the two anchor stress folds. Fold 2 (Jul 2021 – Apr 2022) is the third candidate.

### §11.7 Sharp-divergence flag template

If actual taken-trade count diverges from anchor [1,500–1,800] by more than ±50% per fold, investigate concurrent-cap mechanics, signal frequency under the kijun-sign decision tree, or fold-window edge effects before drawing conclusions. Document the divergence in the result doc §8 per L6.0 §11 disclosure pattern.

---

## §12. What ships next

Per L6.0 v1.1:

- Arc 2 produces verbatim WFO output **AND** characterisation deliverable **AND** magnitude distribution natively (per §14 of the v1.1 lock).
- Output dir: `results/l6/arc2/`. Required files:
  - `trades_all.csv`, `wfo_fold_results.csv`, `wfo_summary.txt`, `signals_log.csv`
  - `mtf_alignment_bar_identity_check.txt` (8 invariants with code-line citations + pooled-count plausibility)
  - `pre_post_regression_manifest.txt`
  - `PHASE_L6_ARC2_RESULT.md` (mandatory regardless of gate disposition)
  - Characterisation set: `signals_features.csv`, `characterisation_report.md`, `regime_breakdown.csv`, `forward_horizon_curves.csv`, `pair_breakdown.csv`, `magnitude_distribution.md`, `feature_lag_audit.txt`, `run_manifest.txt`

- On gate PASS: arc closes as candidate-survives. Planner authors phase-2 (filters, exposure, etc.) in a separate doc; not bundled with this commit.
- On gate FAIL: arc closes as clean null. Findings record empirically observed effects in characterisation; candidate hypotheses (if any) graduate to `CANDIDATE_HYPOTHESES.md`; no filter is derived within Arc 2.
- Either way: **Arc 3 (registry rank 3, `volatility_regime__d1_atr_top_decile__any__h_120`) opens unconditionally on Arc 2 closure** per L6.0 §3.

---

## §13. Bundling scope (one change per phase)

This commit's bundling scope, exhaustively:

- §8.1 schema discriminator extension (new `signal.type` value)
- §8.2 Arc 2 contract block in validators
- §8.3 Arc 2 parallel signal+execution module (`core/signals/l4_mtf_alignment_2_down_mixed_kijun.py`)
- §8.4 verification of WFO-runner invariants (spread-floor hash check, trades-per-fold ≥ 15 gate); add if absent
- §6 Arc 2 mtf-alignment-bar-identity invariants in code with single-line citations
- Arc 2 YAML config at `configs/wfo_l6_arc2.yaml`
- Synthetic-harness regression re-run pre/post for additivity proof against Arc 1's locked manifest
- WFO run + outputs to `results/l6/arc2/`
- Characterisation pipeline reuse (per L6.0 v1.1 §14, the characterisation pipeline is already in place from Arc 1; Arc 2 invokes it on Arc 2's signal+trade set)
- `PHASE_L6_ARC2_RESULT.md`

DO NOT in this commit add: filters, exposure caps beyond per-pair=1, currency caps, daily DD guards, ATR trail, structural exits, regime restrictions, pair restrictions, alternative horizons (h=24 is registry rank 5, separate arc), alternative trend definitions (sma is registry rank 12, deferred), alternative state definitions (3_down is registry rank 7+, deferred), KH-24 modifications, L4 univariate-extreme path modifications, swap modelling, performance metrics beyond gate inputs.

If during execution a defect or improvement idea surfaces, log under "out-of-scope items observed" in the result doc. Do not act on it.

---

## §14. Lock acknowledgement

This document is locked when Keanu signs off and CI records the locking sha256. Post-lock modifications require a re-planning phase per L6.0 §17.

Signed: ________________
Lock sha256: ________________
Lock timestamp: ________________

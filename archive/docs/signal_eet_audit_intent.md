# Signal-Level EET Alignment Audit — Intent Doc

> **Branch:** `engine/signal-level-eet-alignment-audit`
> **Dispatch:** "Signal-Level EET Timezone Alignment Audit + Fix"
> **Status:** pre-implementation read-first. Ends turn here for chat review.

---

## 1. Greps performed

Ran the dispatch-mandated greps plus broader sweep across `core/`, `signals/`, `core/strategies/`. Targets:

- `.floor(` / `.ceil(` — UTC-anchored rounding
- `.normalize()` — UTC midnight truncation (works tz-aware, but downstream lookup is tz-sensitive)
- `pd.Timedelta` — rounding-by-arithmetic patterns
- `midnight` (case-insensitive)
- Plus `merge_asof`, `reindex`, `tz_*` for context

### Raw hits (in-scope: `core/signals/`, `core/strategies/`, `signals/`)

| File:line | Pattern | Used for |
|---|---|---|
| `core/strategies/kh24/signal.py:101` | `d1.index.normalize()` | KH-24 D1-lag-1 alignment (merge_asof key on D1 side) |
| `core/strategies/kh24/signal.py:108` | `df_h4.index.normalize() - pd.Timedelta(days=1)` | KH-24 D1-lag-1 alignment (merge_asof key on H4 side) |
| `core/strategies/kh24/exits/kijun_d1.py:58,63` | same pattern (KH-24 exit) | Same idiom |
| `core/strategies/kh24/filters/d1_regime.py:54,59` | same pattern (KH-24 D1 regime filter) | Same idiom |
| `core/strategies/kh24/filters/h1_cir.py:64` | `df_h4.index + pd.Timedelta(hours=3)` then `.reindex(target_h1)` | H1-CIR lookup (last H1 inside H4) |
| `signals/lchar_d1atr_top_decile.py:74` | `df_1h["date"].dt.normalize()` + `.map(idx_d1)` | Arc 3 D1-lag-1 alignment via exact-match map |
| `signals/lchar_dlr_long.py:131,132` | `d1_ts.normalize()` + `bar_dates_4h.normalize()` + `np.searchsorted` | Arc 10 D1-lag alignment via normalized searchsorted |

### Hits on origin/arc/l_arc_5 (NOT on main, but THE Arc that triggered this dispatch)

| File:line | Pattern | Used for |
|---|---|---|
| `origin/arc/l_arc_5:core/signals/mtf_alignment_2_down_mixed_kijun.py:101` | `h1_df.index.floor("4h")` | H1→H4 alignment via floored-key `.map()` |
| same:102 | `h1_df.index.normalize()` | H1→D1 alignment via floored-key `.map()` |

### Adjacent hits (outside dispatch scope but relevant — flagged for chat)

| File:line | Pattern | Used for | Scope decision |
|---|---|---|---|
| `core/features/multi_tf.py:39,44` | KH-24 normalize+merge_asof+Timedelta idiom | D1/W1 multi-TF features fed to Step 4 classifiers | **Same fix needed**; recommend including in Task 3 (touches downstream Step 4 features for all Pipeline E/D1 arcs) |
| `core/features/distance.py:26,54` | `df.index.normalize()` + groupby + Timedelta(1day) lookup | Prior-session high/low distance feature | **Same fix needed**; recommend including |
| `core/sim/risk/reset_floor.py:71` | `pd.Timestamp(t).normalize()` for daily reset bucketing | Daily reset-floor day key | **Out of dispatch scope**, but is **State B** under EET (UTC-day bucket ≠ EET-day bucket = 5ers daily-DD reset boundary). Flag for separate dispatch. |
| `core/step_6/statistical.py:187` | `df[ts_col].dt.tz_convert("UTC").dt.floor("D")` | Step 6 daily PnL bucketing for stats | Out of scope; explicitly tz_converts to UTC first → consistent with whatever timezone Step 6 trades use. Low risk; flag in audit doc, no fix. |

### No relevant hits in

- `core/arc/` — only fold-boundary arithmetic via `pd.Timedelta(days=1)` (timezone-independent)
- `core/runners/` — same
- `core/architectures/` — same
- `core/steps/` — same
- `core/sim/` (apart from reset_floor noted above)
- `core/data/` — aggregator has the tz_convert wiring but no signal-side bugs
- `signals/kb_exhaustion_bar*.py`, `signals/lchar_bar_range_top_decile.py`, `signals/lchar_pullback_resume_hhhl.py`, `signals/lchar_inside_bar_break_trend_long.py`, `signals/lchar_liquidity_sweep_reclaim.py`, `signals/lchar_swing_high_breakout_trend.py`, `core/strategies/shb/signal_module.py`, `core/signals/pullback_resume_hhhl.py` — single-timeframe signals, no HTF lookup, **State A by construction**

---

## 2. Engine timezone storage convention (load-bearing for classification)

Per PR #189 (`core/data/aggregator.py`):

- M1 input from `load_m1`: **tz-aware UTC** (`pd.to_datetime(..., utc=True)`).
- Aggregator output index for **both conventions**: tz-aware UTC (line 262 `out.index = out.index.tz_convert("UTC")`).
- Convention difference is in WHERE the bar labels fall on the UTC line:
  - `"utc"` (legacy): H4 at UTC `00, 04, 08, 12, 16, 20`; D1 at UTC `00:00`.
  - `"5ers_eet"`: H4 at UTC `22, 02, 06, 10, 14, 18` (winter) / `21, 01, 05, 09, 13, 17` (summer); D1 at UTC `22:00` (winter) / `21:00` (summer) of the **prior calendar day** relative to its EET date.

This is what breaks `.normalize()`-based date-key lookups under EET: the UTC bar label for an EET-day-N bar is in EET-day-N−1's UTC calendar slot.

---

## 3. Per-module State A/B/C classification

State definitions (from dispatch):
- **A** — safe; pure math or panel-native lookup; no UTC-anchor assumption.
- **B** — silently wrong under EET; lookup returns a shifted-but-non-empty value; no halt, wrong numbers.
- **C** — hard fail under EET; lookup returns nothing; pool comes out 0 or all-False.

Classifications are *under the 5ers_eet boundary convention*. Every module here is **State A under the legacy `"utc"` convention** that the current main-branch runtime defaults to.

### `core/signals/` (main)

| Module | Pattern | State (UTC conv.) | State (EET conv.) | Notes |
|---|---|---|---|---|
| `pullback_resume_hhhl.py` (Arc 8) | Single-TF H4, no HTF lookup | A | A | Already audited above |

### `core/signals/` (origin/arc/l_arc_5 — Arc 5 v3 signal, not on main)

| Module | Pattern | State (UTC) | State (EET) | Notes |
|---|---|---|---|---|
| `mtf_alignment_2_down_mixed_kijun.py` | `h1.index.floor("4h").map(idx_4h)` + `h1.index.normalize().map(idx_d1)` | A | **C** | THE Arc-5-v3.0.1 zero-pool bug. `.floor("4h")` returns UTC-anchored 4h floors; under EET those don't match the EET-anchored H4 index labels → `.map()` returns NaN for every bar → val mask all False → empty `signal_mask`. Reproduces the dispatch's "Arc 5 v3.0.1 0-trade pool under EET" symptom exactly. |

### `core/strategies/kh24/` (live KH-24 stack)

| Module | Pattern | State (UTC) | State (EET) | Notes |
|---|---|---|---|---|
| `signal.py` (C1-C9 evaluator + `_build_d1_lag1_arrays`) | `d1.index.normalize()` + `df_h4.index.normalize() - Timedelta(days=1)` + `merge_asof(backward)` | A | **B** | Under EET storage: D1 EET-day-N bar labelled UTC `22:00` of day N−1 → `.normalize()` strips to UTC day N−1. H4 inside EET day N at UTC `> midnight` normalizes to UTC day N; `−1 day` → UTC day N−1; merge_asof picks the D1 labelled UTC day N−1 — which IS the EET-day-N D1 (same-EET-day = lookahead). State B by silent value substitution. |
| `exits/kijun_d1.py` (`_build_d1_lag1_close_and_kijun`) | Same idiom | A | **B** | Same mechanism. KH-24 exit predicate would silently use same-EET-day D1 under EET. |
| `filters/d1_regime.py` (`evaluate_d1_regime`) | Same idiom | A | **B** | Same mechanism. |
| `filters/h1_cir.py` (`evaluate_h1_cir`) | `df_h4.index + pd.Timedelta(hours=3)` then `.reindex(target_h1)` | A | A | Pure additive arithmetic on a tz-aware index; works correctly under any storage convention as long as H4 + H1 panels share the same tz/convention (which they do by construction — same aggregator call). DST artifact noted in PR #189 docstring (3-hour offset is not literally the "last H1 inside H4" on DST-transition days), but that's a separate non-timezone-alignment concern. |

### `core/strategies/shb/`

| Module | State (UTC) | State (EET) |
|---|---|---|
| `signal_module.py` | A | A — delegates to `signals/lchar_swing_high_breakout_trend.py` which is single-TF H4 |

### `signals/` (lchar-era + KB)

| Module | Pattern | State (UTC) | State (EET) | Notes |
|---|---|---|---|---|
| `lchar_d1atr_top_decile.py` (Arc 3) | `df_1h["date"].dt.normalize()` + `.map(idx_d1 keyed by raw D1 ts)` | A | **C** | **Exact-match `.map()` (vs merge_asof) makes this hard-fail rather than silent-wrong**. H1 normalized → UTC midnight; idx_d1 keyed at UTC `22:00`-style EET-shifted D1 timestamps; lookup misses every row → all-NaN → mr_idx all `-1` → out all False → empty signal pool. |
| `lchar_dlr_long.py` (Arc 10) | `d1_norm = d1_ts.normalize()` + `bar_norm = ... .normalize()` + `np.searchsorted` | A | **B** | searchsorted picks a "close" neighbour rather than NaN. Under EET, bar_norm UTC midnight Jan N searchsorts among d1_norm where D1 EET-day-(N+1) maps to UTC midnight Jan N (because of the 22:00 shift) → picks same-EET-day D1 = lookahead. Silent wrong-value State B. |
| `lchar_pullback_resume_hhhl.py` (Arc 8) | Single-TF H4 | A | A | |
| `lchar_inside_bar_break_trend_long.py` (Arc 9) | Single-TF H4 | A | A | |
| `lchar_liquidity_sweep_reclaim.py` (Arc 7) | Single-TF H4 | A | A | |
| `lchar_swing_high_breakout_trend.py` (Arc 11/SHB) | Single-TF H4 | A | A | |
| `lchar_bar_range_top_decile.py` (Arc 4) | Single-TF H1 | A | A | |
| `kb_exhaustion_bar.py`, `kb_exhaustion_bar_adapter.py` (KH-24 Phase KC legacy) | Single-TF, no HTF lookups | A | A | KH-24 v3 lookup work lives in `core/strategies/kh24/signal.py` (audited above), not these files. |

### `core/features/` (multi-TF features — adjacent but in the same risk class)

| Module | Pattern | State (UTC) | State (EET) | Notes |
|---|---|---|---|---|
| `multi_tf.py` (`_build_d1_lag1_series`, `_w1_close_slope_sign`) | Same `.normalize() + merge_asof + Timedelta(1day)` idiom as KH-24 | A | **B** | Affects every Step 4 feature in `feature_class="multi_tf"`: `d1_close_slope_sign`, `d1_close_slope_magnitude`, `d1_atr_percentile_100`, `w1_close_slope_sign`. All used by `core/steps/path_classifier_per_fold.py` and Arc 10's `L1_minus_L0_atr`-class composite — would silently regress under EET. |
| `distance.py` (`_prior_session_high`, `_prior_session_low`) | `df.index.normalize()` + `groupby("_date")` + `_date - Timedelta(1day)` map | A | **B** | Same mechanism. Affects `prior_session_high_distance` and `prior_session_low_distance`. |

---

## 4. KH-24 audit — preliminary read (Task 6, pre-fix)

**Classification of KH-24's signal module under EET storage: State B**.

But — **KH-24 live deployment is currently safe**, for two reasons:

1. KH-24's current runtime is `aggregate(..., boundary_convention="utc")` for all live + WFO paths. PR #189 added the EET option but explicitly carved KH-24's path out of scope (`docs/dispatches/signal_parity_intent.md` and PR #189 commit message both confirm `core/backtester.py` + `core/signal_logic.py` untouched). KH-24 panels never flow through EET aggregation.
2. The MT5 EA runs natively on broker EET bars — that's a separate "EA vs backtest divergence" question already tracked under `docs/calibration/histdata_mt5_aggregation_parity_2026_05.md`. **Not** in the same fault-class as this audit; that one is about broker bar boundaries vs HistData M1 aggregation, not signal-module HTF alignment.

**The latent risk**: if anyone wires a KH-24 re-run or downstream arc through `boundary_convention="5ers_eet"` *without* this fix, the D1 regime / kijun-d1 exit / C8-C9 conditions will read same-EET-day D1 data → silently looks like a different signal, possibly with different (and falsely-leakage-positive) results. **Will surface this prominently in Task 1's audit doc with a top-of-document KH-24 block per the dispatch's emphasis rules.**

This is the worst-case finding from §6.2 of the dispatch but in the **manageable** form: KH-24 signal-module IS State B, but live deployment is not currently using the affected path, and the fix can land without pausing KH-24.

---

## 5. Arc 5 ad-hoc fix — search result

Dispatch table says Arc 5 was "already patched ad-hoc". Search:

- `arc/l_arc_5` (local): no diff vs main (clean tracking branch).
- `origin/arc/l_arc_5`: at Arc 5 closure commit `592a3f7` — contains the BUGGY `.floor("4h")` module, no fix applied.
- No other branch contains a State C fix for `mtf_alignment_2_down_mixed_kijun.py`.
- No stash entries match the pattern.
- No uncommitted ad-hoc fix in the working tree.

**Conclusion**: the "ad-hoc fix" referenced in the dispatch is **not committed anywhere CC can see**. Two interpretations:
- (a) The fix was made in a session that never persisted (CC's working-tree-only iteration during Arc 5 v3.0.1 attempt).
- (b) The dispatch is forward-looking and the "ad-hoc fix" terminology was placeholder language for what Task 5 will retroactively apply.

I'll proceed under interpretation (b): design the canonical utility from first principles, then port `mtf_alignment_2_down_mixed_kijun.py` to use it directly — no "ad-hoc fix" exists to be retroactively replaced. **Flagging for chat confirmation before Task 5.**

---

## 6. Canonical fix pattern — proposed design

The bug class is: **using a UTC- or local-midnight-anchored derived timestamp as a join key against HTF bar timestamps**. The fix class is: **use the HTF panel's native timestamps as the join key, via `merge_asof` or `searchsorted` directly, with no `.floor()` / `.normalize()` intermediate**.

Proposed API for `core/signals/htf_alignment.py`:

```python
def get_htf_value_at(
    current_timestamps: pd.DatetimeIndex,
    htf_panel: pd.DataFrame,
    column: str,
    *,
    require_fully_closed: bool = True,
) -> pd.Series:
    """Align HTF column to LTF anchor timestamps via native-timestamp search.

    For each ts in current_timestamps, finds the HTF bar at index k where
    htf_panel.index[k] is the largest index value satisfying:
        htf_panel.index[k] <= ts                   if not require_fully_closed
        htf_panel.index[k] + htf_bar_duration <= ts if require_fully_closed
    The latter ensures the HTF bar has fully CLOSED by time ts (not just started).

    Timezone-invariant: pure timestamp comparison; no UTC-anchor assumption.
    Works correctly under any panel storage convention (UTC, EET, tz-naive, etc.)
    as long as `current_timestamps` and `htf_panel.index` share the same
    timezone-awareness convention (which the engine guarantees post-PR-#189).

    Returns
    -------
    pd.Series indexed by `current_timestamps`. NaN where the lookup is out of
    range (LTF bars before any HTF bar has fully closed).
    """

def get_htf_row_at(
    current_timestamps: pd.DatetimeIndex,
    htf_panel: pd.DataFrame,
    *,
    require_fully_closed: bool = True,
) -> pd.DataFrame:
    """Same as get_htf_value_at but returns the entire matched HTF row per LTF ts.
    Useful where a signal needs multiple HTF columns at once (close + kijun + atr).
    """
```

**Implementation sketch** (full impl in Task 2; sketching for chat sign-off on the approach):

```python
def get_htf_value_at(current_timestamps, htf_panel, column, *, require_fully_closed=True):
    htf_idx_ns = htf_panel.index.asi8  # int64 ns; tz info irrelevant for ordering
    lt_idx_ns = current_timestamps.asi8

    # searchsorted(side='right') - 1: largest k where htf_idx[k] <= ts
    k = np.searchsorted(htf_idx_ns, lt_idx_ns, side='right') - 1
    if require_fully_closed:
        k = k - 1  # back off one HTF bar — current bar may still be active
    valid = (k >= 0) & (k < len(htf_panel))

    col = htf_panel[column].to_numpy()
    out = np.full(len(current_timestamps), np.nan, dtype=col.dtype if col.dtype.kind == 'f' else float)
    out[valid] = col[k[valid]]
    return pd.Series(out, index=current_timestamps, name=column)
```

The `require_fully_closed=True` default replicates the L_PROTOCOL §1 "one-day-lag rule" intent: at any LTF bar T, the HTF bar consulted must have FULLY CLOSED before T. For HTF=D1, this gives the prior-EET-calendar-day D1 (same as KH-24's design intent). For HTF=H4, this gives the most-recent-completed H4. For HTF=W1, the prior-week W1.

This pattern is what `merge_asof(direction='backward', allow_exact_matches=False)` already does for "most recent strictly prior bar". The `require_fully_closed` step is an additional `-1` on top — equivalent to running merge_asof against `htf_panel.index.shift(-1)`-style shifted starts. Net effect: matches `Arc 5 attic's "floor → idx → idx - 1"` design intent but with the actual HTF timestamps as the join key instead of UTC-anchored floors.

**Open question for chat**: should `get_htf_value_at` default `require_fully_closed=True` (matches KH-24 / L_PROTOCOL §1) or `False` (matches Arc 5's `floor → idx` design, which used `idx-1` separately as the "fully closed" step)? Recommendation: default `True` to match L_PROTOCOL §1 intent; document the contract clearly. Will encode in Task 2.

---

## 7. Files CC will modify in subsequent tasks

### Task 2 (canonical utility)
- **New**: `core/signals/htf_alignment.py`
- **New**: `tests/signals/test_htf_alignment.py`

### Task 3 (apply fixes)
**Core in-dispatch-scope signal-module fixes:**
- `core/signals/mtf_alignment_2_down_mixed_kijun.py` (state C) — *restore from `origin/arc/l_arc_5` AND* fix `.floor()` → canonical utility. **Open question for chat: is restoring this file in this PR in scope, or should Arc 5 v3.0.1 land in a separate dispatch?** This file is NOT on main currently. Recommendation: restore in this PR so Arc 5 v3.0.1 can re-run cleanly per Task 5. Will flag clearly in PR description.
- `signals/lchar_d1atr_top_decile.py` (state C) — replace `.normalize() + map` with canonical utility.
- `signals/lchar_dlr_long.py` (state B) — replace `.normalize() + searchsorted` with canonical utility.

**KH-24 stack (state B under EET) — surfaced per Task 6, fix per Task 3:**
- `core/strategies/kh24/signal.py:81-121` (`_build_d1_lag1_arrays`) — replace with canonical utility.
- `core/strategies/kh24/exits/kijun_d1.py:46-73` (`_build_d1_lag1_close_and_kijun`) — replace with canonical utility.
- `core/strategies/kh24/filters/d1_regime.py:33-75` (`evaluate_d1_regime`) — replace with canonical utility.

KH-24 byte-identical-output regression test will guard against any regression in the LEGACY UTC convention runtime path. **Open question for chat: is fixing the KH-24 stack in-scope for this PR, or should it land separately?** KH-24 is live; touching the live signal-evaluation code carries deployment risk even if the UTC-path output is byte-identical. Recommendation: fix in this PR with explicit byte-identical regression test against pre-fix output, AND a separate confirmation step before merge (chat-side spot check of one fold's trade list). Will not merge without explicit approval.

**Adjacent (out of strict dispatch scope but same fault class):**
- `core/features/multi_tf.py` — same idiom, affects all multi-TF features used in Step 4. Recommendation: include.
- `core/features/distance.py` — same idiom. Recommendation: include.
- `core/sim/risk/reset_floor.py` — daily-reset bucketing. **NOT a signal-module bug**; the daily reset is computed against trade-close timestamps and would be wrong-day-bucketed under EET. Recommendation: out of scope for this dispatch; spawn a separate task (will use `mcp__ccd_session__spawn_task` for chat visibility).

### Task 4 (regression tests)
- **New**: `tests/signals/test_htf_alignment_timezone_invariance.py`

### Task 7 (docs)
- `docs/PROTOCOL_RUNTIME.md` — add §"Timezone-invariant signal module patterns"
- `docs/BACKTESTER_ARCHITECTURE.md` — signal-module timezone responsibility note
- `docs/audits/engine_capability_audit_2026_05.md` — footer: signal-level EET MISSING → WIRED
- `docs/audits/signal_module_eet_audit_2026_05.md` — Task 1 output (NEW file)

### Task 8 (lint rule, optional)
- `.pre-commit-config.yaml` — grep-based hook: flag any `.floor(` introduction in `core/signals/` or `core/strategies/`.

---

## 8. Wave 1 retry triage (Task 5 plan — fill table at Task 1)

| Arc | Signal module | State found | Action |
|---|---|---|---|
| Arc 3 (`d1atr_top_decile`) | `signals/lchar_d1atr_top_decile.py` | C | Fix + re-run under EET (separate dispatch, this PR triages only) |
| Arc 4 (`bar_range_top_decile_neg`) | `signals/lchar_bar_range_top_decile.py` | A | Verdict valid as-is. No re-run needed. |
| Arc 5 (`mtf_alignment_2_down_mixed_kijun`) | `core/signals/mtf_alignment_2_down_mixed_kijun.py` (origin/arc/l_arc_5) | C | Fix in this PR; re-run as separate dispatch (THE triggering arc). |
| Arc 6 (failed_breakout_reversal) | (not on main; would need to be located) | TBD | Defer to triage. |
| Arc 7 (`liquidity_sweep_reclaim`) | `signals/lchar_liquidity_sweep_reclaim.py` | A | Verdict valid. No re-run. |
| Arc 8 (`pullback_resume_hhhl`) | `signals/lchar_pullback_resume_hhhl.py` + `core/signals/pullback_resume_hhhl.py` | A | Verdict valid. Original FAIL stays FAIL with reasoning intact. |
| Arc 9 (`inside_bar_break_trend_long`) | `signals/lchar_inside_bar_break_trend_long.py` | A | (If Arc 9 was run.) Verdict valid. |
| Arc 10 (`d1_swing_low_rejection_long` / DLR) | `signals/lchar_dlr_long.py` | **B** | **Verdict suspect**. Original verdict was PASS-VIABLE on path quality but Step 4 near-miss. The Step 1 trade pool may differ under EET because the D1 alignment was silently picking same-EET-day data. Re-run needed under EET to confirm. Currently halted on the partial-close exit-primitive blocker (CC_18) — flagging this as an ADDITIONAL prerequisite for Arc 10 resumption. |
| Arc 11 (`swing_high_breakout_trend_long`) | `signals/lchar_swing_high_breakout_trend.py` | A | Verdict valid. Original FAIL stays FAIL. |
| KH-24 (live) | `core/strategies/kh24/signal.py` + filters + exit | A under UTC conv. / B under EET | Live deployment NOT at risk (uses UTC conv.). Fix lands here to make future cross-convention re-runs safe. |

Final triage table will go in `docs/audits/signal_module_eet_audit_2026_05.md` Task 1 output.

---

## 9. Open questions for chat before proceeding

1. **Arc 5 `mtf_alignment_2_down_mixed_kijun.py` restoration**: this file is on `origin/arc/l_arc_5` only, not on main. Restore in this PR (so Task 3 can fix it + Task 5 can include it in triage)? Or merge separately first?
2. **KH-24 stack fix in-scope here**: Task 6 mandates auditing KH-24, but fixing KH-24's signal/exit/filter modules has deployment-adjacency. Confirm: fix in this PR with byte-identical UTC-path regression test, gated on chat approval before merge?
3. **`core/features/multi_tf.py` + `core/features/distance.py`**: adjacent to dispatch scope but exact same fault class (KH-24 normalize idiom). Confirm: include in Task 3?
4. **`core/sim/risk/reset_floor.py`**: same fault class but in `core/sim/risk/`, out of strict dispatch scope. Confirm: NOT in this PR, spawn separate task instead?
5. **`require_fully_closed` default**: `True` (matches KH-24 + L_PROTOCOL §1) or `False`? Recommendation: `True`.
6. **Arc 10 triage decision**: Arc 10's State B means its verdict may have been computed against the wrong D1 data. The original verdict (PASS-VIABLE after causal audit, then HALT pending exit primitive) survives causally on path quality, but the entry-time D1 features (`L1_minus_L0_atr` etc.) consumed in the WFO oracle would be different under EET. Confirm: surface as additional blocker on Arc 10 resumption alongside CC_18?

---

## 10. Risks summary (per dispatch §Risks 1-4)

1. **Audit found more State B/C than expected**: 4 modules across `signals/` + `core/signals/` + KH-24 stack + 2 in `core/features/`. Higher than baseline expectation. Time cost: each State B/C arc needs a separate re-run dispatch (Wave 1 retries).
2. **KH-24 IS State B under EET**: confirmed in §3. Mitigated because live deployment uses UTC convention. Surfaced prominently per §6.2 of dispatch. Will surface again at top of `docs/audits/signal_module_eet_audit_2026_05.md`.
3. **Regression test in Task 4 may surface unknown State B**: covered by the per-module UTC-vs-EET sha256 + drift-bound test design. Test outcome will be authoritative; will update audit doc if surprising hits surface.
4. **Lint rule false positives**: `.floor()` does have legitimate uses (e.g. `np.floor(size)` for position rounding — though this is in `analytics/metrics.py` and `core/step_6/statistical.py`, both outside the rule's grep scope of `core/signals/` + `core/strategies/`). Task 1 audit catalogs distinguish; rule scope can be narrowed.

---

**End turn. Awaiting chat review of intent doc + answers to §9 open questions before Task 1.**

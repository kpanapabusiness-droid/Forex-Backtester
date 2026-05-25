# Signal-Level EET Timezone Alignment Audit — 2026-05

> **Date:** 2026-05-25
> **PR:** `engine/signal-level-eet-alignment-audit`
> **Dispatch:** "Signal-Level EET Timezone Alignment Audit + Fix"
> **Trigger:** Arc 5 v3.0.1 attempted re-run under PR #189's `boundary_convention="5ers_eet"` produced 0-trade pool. Root-cause grep found the same UTC-anchored HTF-lookup idiom in several other signal + feature modules. Engine-level aggregation (PR #189) was already EET-correct; this audit + fix addresses the signal-module level.
> **Canonical fix utility:** [core/signals/htf_alignment.py](../../core/signals/htf_alignment.py)
> **Regression test:** [tests/signals/test_htf_alignment_timezone_invariance.py](../../tests/signals/test_htf_alignment_timezone_invariance.py)

---

## ⚠ KH-24 audit result (surfaced first, per Task 6 dispatch mandate)

**Classification under EET storage convention: State B (silent same-EET-day D1 lookahead). State A under UTC storage convention. NO live-deployment risk at audit time, but a latent landmine if anyone re-wires KH-24 to EET aggregation.**

### Mechanism

KH-24 signal evaluator + kijun_d1 exit + D1-regime filter all used the same pattern:

```python
d1["_date"] = d1.index.normalize()
shifted = pd.DataFrame({"_date": df_h4.index.normalize() - pd.Timedelta(days=1), ...})
merged = pd.merge_asof(shifted, d1, on="_date", direction="backward")
```

Under UTC storage convention, D1 bars are labelled at UTC `00:00` of their calendar date, and H4 bars are at `00, 04, 08, ..., 20`. The idiom correctly resolves to "the D1 bar of the prior UTC calendar day".

Under 5ers EET storage convention (PR #189), D1 bars are labelled at UTC `22:00` of the *prior* calendar day (= EET 00:00 of their EET-day). For an H4 bar inside EET day N (at UTC > midnight), the idiom's `.normalize() - Timedelta(days=1)` step produces a key that the same-EET-day D1 satisfies — **silently picking the SAME-EET-DAY D1 (a lookahead leak)** instead of the intended prior-EET-day D1.

### Current live-deployment status

- KH-24 runs through `aggregate(..., boundary_convention="utc")` for all WFO + live paths. PR #189 carved KH-24's path out of EET wiring (`core/backtester.py` + `core/signal_logic.py` untouched). The bug is dormant.
- The MT5 EA runs natively on broker EET bars; that's a separate "EA vs backtest divergence" question tracked under [docs/calibration/histdata_mt5_aggregation_parity_2026_05.md](../calibration/histdata_mt5_aggregation_parity_2026_05.md), unrelated to this audit class.

### Fix applied in this PR

All three KH-24 modules (`core/strategies/kh24/{signal,exits/kijun_d1,filters/d1_regime}.py`) switched to the canonical `core.signals.htf_alignment.get_htf_row_at(..., require_fully_closed=True)`. The utility produces:

- **Byte-identical output to the legacy idiom under UTC convention** — verified by [`tests/signals/test_htf_alignment::test_byte_identical_to_legacy_kh24_idiom_under_utc`](../../tests/signals/test_htf_alignment.py) plus two edge cases (`test_byte_identical_when_ltf_extends_past_last_htf_under_utc`, `test_byte_identical_when_ltf_inside_last_htf_under_utc`).
- **Correct prior-EET-day D1 alignment under 5ers EET convention** — verified by `test_default_returns_prior_d1_for_h4_under_eet` + `test_no_lookahead_at_first_h4_of_day_eet` + `test_canonical_disagrees_with_legacy_under_eet`.

### Q2 mitigation surface for chat-side merge check

Per chat's Q2 answer: byte-identical UTC-path regression test (✓ included) plus **explicit chat-side spot check of one fold's trade list pre/post fix before merge**. CC has not run a full WFO comparison. Recommended chat-side check before merging:

1. Check out main (pre-fix state).
2. Run KH-24 backtest on one fold (suggested: WFO fold 7 — currently has worst-fold ROI +1.92%) and capture trade list.
3. Check out this PR's branch.
4. Re-run same fold; diff trade lists.
5. **Expected:** identical (since UTC convention path is byte-identical at the unit-test level).
6. **If any deviation surfaces: DO NOT MERGE.** Chat resolution required.

### KH-24 existing unit + E2E test suite

All 22 KH-24 tests pass post-fix without modification:

```
tests/test_kh24_signal.py:   6 passed
tests/test_kh24_filters.py:  7 passed
tests/test_kh24_kijun_d1.py: 7 passed (lookahead-invariance, lag-1 alignment, predicate firing)
tests/test_kh24_e2e.py:      7 passed (including test_kh24_e2e_two_run_determinism)
tests/test_kh24_reset_floor.py: 8 passed
```

`test_kh24_e2e_two_run_determinism` is the strongest cross-component guard — KH-24's end-to-end output is byte-identical across two runs post-fix.

---

## Per-module classification + fix table

States: **A** = safe; **B** = silently wrong under EET (wrong-value lookup); **C** = hard fail under EET (empty pool).

### `core/signals/`

| Module | Pre-fix state (UTC / EET) | Post-fix state | Fix applied | Notes |
|---|---|---|---|---|
| [`mtf_alignment_2_down_mixed_kijun.py`](../../core/signals/mtf_alignment_2_down_mixed_kijun.py) (Arc 5) | A / **C** | A / A | **Restored from `origin/arc/l_arc_5` + canonical utility** (same PR) | THE Arc-5-v3.0.1 zero-pool bug. `.floor("4h").map()` returned NaN for every EET-anchored H1 bar. File was not on main; restore in this PR was chat-approved (Q1) so Arc 5 v3.0.1 can re-run cleanly post-merge. |
| [`pullback_resume_hhhl.py`](../../core/signals/pullback_resume_hhhl.py) (Arc 8 v3) | A / A | unchanged | none needed | Single-TF H4; no HTF lookup. |

### `core/strategies/kh24/`

| Module | Pre-fix (UTC / EET) | Post-fix | Fix applied |
|---|---|---|---|
| [`signal.py`](../../core/strategies/kh24/signal.py) (`_build_d1_lag1_arrays`) | A / **B** | A / A | `get_htf_row_at(..., require_fully_closed=True)` |
| [`exits/kijun_d1.py`](../../core/strategies/kh24/exits/kijun_d1.py) (`_build_d1_lag1_close_and_kijun`) | A / **B** | A / A | same |
| [`filters/d1_regime.py`](../../core/strategies/kh24/filters/d1_regime.py) (`evaluate_d1_regime`) | A / **B** | A / A | same |
| [`filters/h1_cir.py`](../../core/strategies/kh24/filters/h1_cir.py) | A / A | unchanged | none needed (uses `df_h4.index + pd.Timedelta(hours=3)` — pure additive on same-tz index, timezone-invariant) |

### `core/strategies/shb/`

| Module | State | Notes |
|---|---|---|
| [`signal_module.py`](../../core/strategies/shb/signal_module.py) | A / A | Delegates to `signals/lchar_swing_high_breakout_trend.py` (single-TF H4) |

### `signals/` (lchar-era + KB)

| Module | Arc | Pre-fix (UTC / EET) | Post-fix | Fix applied |
|---|---|---|---|---|
| [`lchar_d1atr_top_decile.py`](../../signals/lchar_d1atr_top_decile.py) | Arc 3 | A / **C** | A / A | `get_htf_value_at(...)` replaces `.dt.normalize().map(idx_d1)` |
| [`lchar_dlr_long.py`](../../signals/lchar_dlr_long.py) (`_date_to_d1_index`) | Arc 10 | A / **B** | A / A | `get_htf_index_at(..., require_fully_closed=False)` replaces `.normalize() + searchsorted` |
| [`lchar_bar_range_top_decile.py`](../../signals/lchar_bar_range_top_decile.py) | Arc 4 | A / A | unchanged | Single-TF H1 |
| [`lchar_pullback_resume_hhhl.py`](../../signals/lchar_pullback_resume_hhhl.py) | Arc 8 | A / A | unchanged | Single-TF H4 |
| [`lchar_inside_bar_break_trend_long.py`](../../signals/lchar_inside_bar_break_trend_long.py) | Arc 9 | A / A | unchanged | Single-TF H4 |
| [`lchar_liquidity_sweep_reclaim.py`](../../signals/lchar_liquidity_sweep_reclaim.py) | Arc 7 | A / A | unchanged | Single-TF H4 |
| [`lchar_swing_high_breakout_trend.py`](../../signals/lchar_swing_high_breakout_trend.py) | Arc 11 / SHB | A / A | unchanged | Single-TF H4 |
| [`kb_exhaustion_bar.py`](../../signals/kb_exhaustion_bar.py), [`kb_exhaustion_bar_adapter.py`](../../signals/kb_exhaustion_bar_adapter.py) | KH-24 Phase KC legacy | A / A | unchanged | Single-TF; v3 KH-24 alignment lives in `core/strategies/kh24/signal.py` |

### `core/features/` (multi-TF feature producers — adjacent to dispatch scope, included per chat Q3)

| Module | Pre-fix (UTC / EET) | Post-fix | Fix applied |
|---|---|---|---|
| [`multi_tf.py`](../../core/features/multi_tf.py) (`_build_d1_lag1_series`) | A / **B** | A / A | `get_htf_value_at(..., require_fully_closed=True)` |
| [`multi_tf.py`](../../core/features/multi_tf.py) (`_w1_close_slope_sign`) | A / A | unchanged | Already uses `merge_asof` against actual W1 timestamps; tz-invariant by construction |
| [`distance.py`](../../core/features/distance.py) (`_prior_session_high/_prior_session_low`) | A / **partial B** | unchanged | **DEFERRED — see follow-up [OPEN-FEATURES-DISTANCE-EET-SESSION-SEMANTICS](#open-features-distance-eet-session-semantics) below**. On closer inspection NOT the same fault class as `multi_tf.py`: the bug here is a session-boundary *semantic* question (UTC day vs EET day), not an HTF alignment alignment bug. The canonical utility doesn't fix the underlying groupby-by-UTC-date split. |

### Out of scope (flagged for separate dispatch)

| File:line | Issue | Follow-up |
|---|---|---|
| [`core/sim/risk/reset_floor.py:71`](../../core/sim/risk/reset_floor.py) | Daily reset-floor day key uses `pd.Timestamp(t).normalize()` — buckets by UTC midnight, not EET midnight (= 5ers daily-DD reset boundary). State B under EET. | **[OPEN-RESET-FLOOR-EET](#open-reset-floor-eet)** (below) |
| [`core/step_6/statistical.py:187`](../../core/step_6/statistical.py) | `df[ts_col].dt.tz_convert("UTC").dt.floor("D")` for Step 6 daily PnL bucketing. | Out of scope; explicitly tz-converts to UTC first → consistent semantic ("PnL bucketed by UTC day for stats"). Low risk; no fix needed. |

---

## Open follow-ups

### OPEN-RESET-FLOOR-EET

`core/sim/risk/reset_floor.py:71` uses `pd.Timestamp(t).normalize()` to bucket the daily reset floor. Under 5ers EET storage, this buckets by UTC midnight (not EET midnight = the 5ers broker daily-DD boundary). Effect: the daily floor may ratchet up at the wrong calendar day under EET, which could (a) compute daily DD against the wrong day's high-water mark or (b) shift drawdown reporting by 2-3 hours.

**Not in this PR per chat Q4** — flagged for separate dispatch. Same fault class as the signal-module bug but in `core/sim/risk/`, outside the dispatch's `core/signals/` + `core/strategies/` scope. The signal-EET fix in this PR does NOT touch this module.

### OPEN-FEATURES-DISTANCE-EET-SESSION-SEMANTICS

`core/features/distance.py` `_prior_session_high/_prior_session_low` group bars by `df.index.normalize()` (UTC calendar day). Under 5ers EET storage, the H4 bar labels are EET-shifted UTC timestamps (e.g. `UTC 22:00` for EET 00:00). The groupby then splits each EET-day into TWO UTC-date buckets: the EET-morning bars get the previous UTC date, the EET-afternoon bars get the current UTC date. The "session" boundary becomes UTC midnight (cutting EET days in half), not EET midnight.

This is **NOT an HTF alignment bug fixable by the canonical utility** — the lookup step is fine; the *grouping* is wrong for EET semantics. The fix would be to group by `df.index.tz_convert(BROKER_TZ).normalize()` or similar — a design decision about what "session" means under EET (UTC day for stats consistency vs EET day for broker-aligned analysis). Deferring to a separate dispatch.

Effect: under EET storage, the `prior_session_high_distance` and `prior_session_low_distance` features compute against UTC-day-bucketed maxes/mins instead of EET-day-bucketed. Same numerical noise class as the KH-24 D1-bucket bug; pre-existing behavior preserved by NOT touching the file in this PR.

### OPEN-ARC-10-RESUMPTION-PREREQS

Arc 10 v3.0.1 resumption blocked on TWO prerequisites:

1. **CC_18** (partial-close exit primitive `sl_partial_close_1r_runner_trail`) — see [`cdb8de2 [ARC 10 v3.0.1] HALT — canonical engine missing sl_partial_close_1r_runner_trail`](https://github.com/kpanapabusiness-droid/Forex-Backtester/commits/arc/l_arc_10_v3.0.1). Pre-existing.
2. **This PR** (signal-EET audit) — Arc 10's `lchar_dlr_long.py` was State B under EET; its v3.0 verdict (PASS-VIABLE → re-evaluated PASS-DEPLOYABLE per Amendment 3) was computed with the buggy D1 alignment. Post-merge re-run needed.

Phase 1 chat to dispatch Arc 10 re-run only after BOTH land.

---

## Wave 1 retry triage table (Task 5)

| Arc | Signal module | State found | Post-fix action |
|---|---|---|---|
| Arc 3 | `signals/lchar_d1atr_top_decile.py` | C | **Re-run needed under EET**; original v2.x verdict was CLEAN-NULL Step 3 (pre-EET work). |
| Arc 4 | `signals/lchar_bar_range_top_decile.py` | A | Verdict valid as-is. No re-run. |
| Arc 5 | `core/signals/mtf_alignment_2_down_mixed_kijun.py` | C | **Re-run needed under EET** — restored to main in this PR; Arc 5 v3.0.1 is THE triggering arc. Original v3.0 closure verdict FAIL Step 5; re-run validates whether new EET-correct pool changes that outcome. |
| Arc 6 | (failed_breakout_reversal; not on main) | TBD | Deferred — needs separate location step. |
| Arc 7 | `signals/lchar_liquidity_sweep_reclaim.py` | A | Verdict valid (FAIL). No re-run. |
| Arc 8 | `signals/lchar_pullback_resume_hhhl.py` + `core/signals/pullback_resume_hhhl.py` | A | Verdict valid (FAIL). No re-run. |
| Arc 9 | `signals/lchar_inside_bar_break_trend_long.py` | A | Single-TF, no EET risk. (Status: not yet run as full arc.) |
| **Arc 10** | `signals/lchar_dlr_long.py` | **B** | **Re-run needed under EET**. The v3.0 PASS-VIABLE / Amendment-3 re-evaluated PASS-DEPLOYABLE verdict was computed with silent same-EET-day D1 lookahead. Specifically the `L1_minus_L0_atr` family of features that carry 116% of HTF LOO drop in EXP-05 are D1-derived → directly affected. **Re-run is the load-bearing item for Arc 10 deployability decision**, gated also on CC_18 (per OPEN-ARC-10-RESUMPTION-PREREQS above). |
| Arc 11 | `signals/lchar_swing_high_breakout_trend.py` | A | Verdict valid (FAIL). No re-run. |
| KH-24 (live) | `core/strategies/kh24/` stack | A under UTC / B under EET | Live deployment NOT at risk (uses UTC convention). Q2 chat-side spot check pending before merge. No live-deployment action required. |

**Re-runs needed (3 arcs):** Arc 3, Arc 5, Arc 10. Each as separate post-merge dispatch.

---

## Tests landed

### Unit tests for the canonical utility ([`tests/signals/test_htf_alignment.py`](../../tests/signals/test_htf_alignment.py))

19 tests covering:
- Default-semantics (require_fully_closed=True): prior-day D1 under UTC + EET
- Boundary cases (first H4 of day, exact bar boundaries)
- Out-of-range handling (LTF before any HTF; LTF inside last HTF; LTF past last HTF)
- Tz-awareness contract (mismatch raises ValueError; matched ok)
- Series coercion (accepts Series of timestamps)
- Unknown column raises KeyError
- `get_htf_row_at` parity with per-column `get_htf_value_at`
- **Byte-identical to legacy KH-24 D1-lag-1 idiom under UTC** — load-bearing for Q2 mitigation
- **Canonical disagrees with legacy under EET** — proves the fix changes behavior

### Timezone-invariance regression suite ([`tests/signals/test_htf_alignment_timezone_invariance.py`](../../tests/signals/test_htf_alignment_timezone_invariance.py))

14 tests covering:
- KH-24 signal / d1_regime / kijun_d1 exit: no zero-pool, no all-NaN under EET
- Arc 5 mtf_alignment: no zero-pool, lookahead invariant holds under EET
- Arc 3 d1atr_top_decile: runs to completion under EET, produces 0/1 mask
- Arc 10 DLR `_date_to_d1_index`: aligns to containing-EET-day D1 (specifically asserts the prior bug's "next-EET-day D1" answer doesn't sneak back)
- **Static guard**: parametrized check that fixed modules don't reintroduce `.floor(` or `.normalize()` in non-docstring code

### KH-24 + features test suites (no test changes; all green post-fix)

```
tests/test_kh24_signal.py     6 passed
tests/test_kh24_filters.py    7 passed
tests/test_kh24_kijun_d1.py   7 passed
tests/test_kh24_e2e.py        7 passed  (including test_kh24_e2e_two_run_determinism)
tests/test_kh24_reset_floor.py 8 passed
tests/test_features_pipeline.py 20 passed (including test_pipeline_sha256_stable, test_pipeline_two_run_determinism)
tests/test_features_individual.py 23 passed
```

---

## Footer update: `engine_capability_audit_2026_05.md`

Signal-level EET alignment gap: MISSING → **WIRED**. See this audit doc + canonical utility at `core/signals/htf_alignment.py`.

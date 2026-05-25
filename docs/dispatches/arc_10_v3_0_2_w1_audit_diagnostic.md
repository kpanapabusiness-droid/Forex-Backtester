# Arc 10 v3.0.2 — W1 producer pre-Step-1 audit diagnostic

> **Status:** HALT triggered. Non-canonical HTF lookup pattern confirmed in `core/features/multi_tf.py::_w1_close_slope_sign`. Empirical lookahead mechanism reproduced from code + aggregator labelling rules.
> **Branch:** `arc/l_arc_10_v3.0.2` (intent doc landed at `7dd85d4`; no Step 1 compute started)
> **Triggering mandate:** dispatch §2 "W1 producer lookahead audit (pre-Step-1)"
> **Cross-arc signal:** Arc 8 v3.0.2 CC chat surfaced this; this diagnostic reproduces the finding independently for Arc 10 + flags the wider impact.

---

## §1 Per-producer status table

`core/features/multi_tf.py` — five HTF producers + one shared helper:

| Producer | File:line | Pattern | Status |
|---|---|---|---|
| `_build_d1_lag1_series` (shared helper) | [multi_tf.py:33-42](core/features/multi_tf.py:33) | `get_htf_value_at(pair_df.index, d1_df[[column]], column, require_fully_closed=True)` | **CANONICAL** ✓ |
| `_d1_close_slope_sign` | [multi_tf.py:45-57](core/features/multi_tf.py:45) | routes through `_build_d1_lag1_series` | **CANONICAL** ✓ |
| `_d1_close_slope_magnitude` | [multi_tf.py:76-83](core/features/multi_tf.py:76) | routes through `_build_d1_lag1_series` | **CANONICAL** ✓ |
| `_d1_atr_percentile_100` | [multi_tf.py:102-112](core/features/multi_tf.py:102) | routes through `_build_d1_lag1_series` | **CANONICAL** ✓ |
| `_w1_close_slope_sign` | [multi_tf.py:136-157](core/features/multi_tf.py:136) | raw `pd.merge_asof(df, w1_pos, on="_t", direction="backward", allow_exact_matches=False)` | **NON-CANONICAL — CRITICAL LOOKAHEAD** ⚠️ |

**All D1 producers (Arc 10's load-bearing class) are clean.** W1 producer is contaminated.

---

## §2 Mechanism — why `_w1_close_slope_sign` leaks

### Suspect pattern (verbatim, [multi_tf.py:147-157](core/features/multi_tf.py:147))

```python
slope = (w1_close - w1_close.shift(1)).rename("slope")
# Align by `merge_asof(direction='backward')` on week-start key, then enforce
# that the matched week's bar ended strictly before the signal bar.
df = pd.DataFrame({"_t": pair_df.index, "_idx": np.arange(len(pair_df))})
df = df.sort_values("_t")
w1_pos = slope.to_frame().reset_index().rename(columns={w1_df.index.name or "index": "_t"})
w1_pos.columns = ["_t", "slope"]
w1_pos = w1_pos.sort_values("_t")
merged = pd.merge_asof(df, w1_pos, on="_t", direction="backward", allow_exact_matches=False)
```

### Aggregator labelling ([core/data/aggregator.py:117](core/data/aggregator.py:117) + `_resample_kwargs`)

```python
_TF_TO_FREQ: dict[str, str] = {
    ...
    "W1": "W-MON",
}

def _resample_kwargs(freq: str, origin: str) -> dict:
    kw: dict = {"label": "left", "closed": "left"}
    ...
```

`label='left'` + `closed='left'` means the W1 bar for week N is timestamped at the **START of week N** (Monday 00:00 UTC under `boundary_convention="utc"`; UTC Sun 22:00 / 21:00 under `"5ers_eet"` per PROTOCOL_RUNTIME §15.3 — both anchor at the start of the new week).

### Why `allow_exact_matches=False` does NOT prevent the leak

The author's docstring claims:

> "Strictly prior W1 bar (no exact-match alignment — week N's bar isn't visible until week N+1 starts)."

This is incorrect. `allow_exact_matches=False` only prevents `merge_asof` from matching at the *exact* W1 timestamp. It does NOT prevent matching anywhere later in the same week.

**Concrete failure case:**
- H4 bar timestamp: Wed UTC 04:00 of week N
- Most recent W1 timestamp strictly before Wed 04:00: Mon 00:00 UTC of week N itself
- `merge_asof` picks the W1 bar for week N
- That W1 bar's `close` is computed from data spanning `[Mon 00:00 UTC week N, Mon 00:00 UTC week N+1)` — i.e., the END of week N (Sunday ~24:00 UTC)
- The slope `w1_close[N] - w1_close[N-1]` therefore contains week N's future close, **visible at every H4 bar from Mon 00:00:01 UTC through Sun 23:59:59 UTC of week N**

This is the "leaking future Sunday close" pattern Arc 8 v3.0.2 CC chat identified. Reproduced from first principles on Arc 10's code path.

Under `boundary_convention="5ers_eet"` (Arc 10 v3.0.2's locked convention) the W1 timestamp shifts to UTC Sun 22:00 / 21:00 (= EET Mon 00:00) — still the START of the week. Same mechanism.

### Canonical fix

Replace the body of `_w1_close_slope_sign` with the analog used in `_build_d1_lag1_series`:

```python
slope = (w1_close - w1_close.shift(1)).rename("slope")
aligned = get_htf_value_at(
    pair_df.index, slope.to_frame(), "slope",
    require_fully_closed=True,
)
return np.sign(aligned).astype("float64")
```

`require_fully_closed=True` backs off to the most-recently-FULLY-CLOSED W1 bar (week N-1) at any LTF timestamp inside week N. This is the L_PROTOCOL §1 non-negotiable's natural extension from "one-day lag" to "strict prior fully-closed HTF bar."

The canonical utility was specifically designed for this. PR #193 audit table at [docs/audits/signal_module_eet_audit_2026_05.md:113](docs/audits/signal_module_eet_audit_2026_05.md:113) classified `_w1_close_slope_sign` as State A under both UTC and EET ("Already uses merge_asof against actual W1 timestamps; tz-invariant by construction"). **That audit was wrong.** Timezone-invariance is not the issue; the bug is convention-independent — it's the label='left' aggregator + same-week merge_asof match. The PR #193 audit missed it because it focused on tz-shift failure modes (State B vs C), not on within-period lookahead.

---

## §3 Cross-arc impact

`core/features/multi_tf.py::_w1_close_slope_sign` is registered in the default v3.0 27-feature catalogue ([multi_tf.py:160-174](core/features/multi_tf.py:160)). Every arc using `core.features.pipeline.compute_feature_matrix` consumes it.

### Arcs that consumed `w1_close_slope_sign` while contaminated

| Arc | Verdict | W1 feature in winning config? | Impact |
|---|---|---|---|
| **Arc 10 v3.0** (closed, UTC) | PASS-DEPLOYABLE (Amendment 3 re-eval) | NO — A1 winner has empty `features_in_winning_config`, signal is D1-driven (`_date_to_d1_index` only) | Verdict survives. Step 1 pool clustering / Step 3 capturability / Step 4 AUC numbers used the contaminated `w1_close_slope_sign` for diagnostic/ranking — but the A1 winner is a rule-based no-classifier architecture. |
| **Arc 8 v3.0.2** (in flight) | discovered the bug | TBD | Audit in flight per Arc 8 CC chat. |
| **Arc 5 v3.0.2** (in flight, branch `arc/l_arc_5_v3.0.2`) | per recent commits Step 1 artefacts landed (pool 130,099 trades, +2.60% vs v3.0) | TBD | **Cross-arc HALT signal needed** — Arc 5 Step 1 ran on contaminated pipeline. |
| **Arc 11 v3.0.2** (in flight, branch `arc/l_arc_11_v3.0.2`) | TBD | TBD | **Cross-arc HALT signal needed** — same producer file. |
| **Arc 7 v3.0.2** (closed FAIL, branch `arc/l_arc_7_v3.0.2` head `dd7e247`) | FAIL (`step5_not_scalable`) | TBD | FAIL verdict cannot be upgraded by Step 6 OR by clean producers — but the FAIL itself may have been computed against a contaminated Step 1 pool. Re-audit needed only if conclusion direction matters cross-arc. |
| **Arc 8 v3.0.2** (HALT — w1_close_slope_sign confirmed lookahead) | HALT — see commit `1406468 [ARC 8 v3.0.2] Verification HALT — w1_close_slope_sign lookahead confirmed` | n/a | This commit on `arc/l_arc_8_v3.0.2` is the empirical confirmation that triggered the user's mandate. **Arc 8 CC chat already at HALT on this exact issue.** |
| KH-24 (live) | locked, deployed | n/a — KH-24 uses `core/strategies/kh24/` stack, NOT `core/features/multi_tf.py` | **No live-deployment risk.** KH-24's HTF alignment was canonicalised separately by PR #193. |

### Arc 10 v3.0.2 specific impact

- **Arc 10's D1-derived features are all CANONICAL.** All three D1 producers (`_d1_close_slope_sign`, `_d1_close_slope_magnitude`, `_d1_atr_percentile_100`) plus the upstream `L1_minus_L0_atr` family route through `_build_d1_lag1_series` which uses `get_htf_value_at(..., require_fully_closed=True)`.
- **Arc 10's signal `signals/lchar_dlr_long.py::_date_to_d1_index`** was independently canonicalised by PR #193 to `get_htf_index_at(..., require_fully_closed=False)`. Not affected.
- **`w1_close_slope_sign` is a feature in the 27-feature default catalogue**, computed at Step 1 and consumed by Step 2 clustering / Step 3 capturability / Step 4 classifier training. Under Arc 10's A1 winning architecture (no classifier features), the W1 leak does not appear at trade-decision time — but it polluted the Step 4 AUC (0.5199 c1 LGBM) the prior verdict cites.
- **Under 5ers_eet (Arc 10 v3.0.2's locked convention),** the W1 leak persists for the same mechanism (W1 label='left' anchors at week-start regardless of convention).

---

## §4 Decision per dispatch §2 decision rule

> "Any producer non-canonical: HALT immediately."

Triggered. **No Step 1 compute initiated on `arc/l_arc_10_v3.0.2`.**

### Recommendation for chat

1. **Land the canonical fix on main first.** Replace the body of `_w1_close_slope_sign` with the `get_htf_value_at(..., require_fully_closed=True)` analog (see §2 "Canonical fix" above). 5-15 line patch + regression test (re-run `tests/signals/test_htf_alignment_timezone_invariance.py` plus add a same-week-lookahead-invariant test specifically for W1).
2. **Audit `docs/audits/signal_module_eet_audit_2026_05.md`** — correct the State A classification of `_w1_close_slope_sign` to State B-equivalent (within-period lookahead, convention-independent). PR #193's audit family was about tz-shift; the W1 bug is in a different family and needs a footer / supplementary audit doc.
3. **Cross-arc HALT signal to Arc 5 v3.0.2 + Arc 11 v3.0.2.** Both branches have Step 1 artefacts already; the contaminated pool must be rebuilt post-fix. Arc 8 v3.0.2 is already at HALT on this issue. Arc 7 v3.0.2's FAIL verdict can survive (verdict direction unchanged by cleaner features), but the Step 4 numbers reported are suspect.
4. **No retroactive correction for Arc 10 v3.0.** The Amendment 3 re-evaluated PASS-DEPLOYABLE-PROVISIONAL verdict stands on the A1 winning architecture which has empty `features_in_winning_config`. The leak affected diagnostic outputs (Step 4 AUC), not the verdict-carrying numbers.
5. **Once fix lands, Arc 10 v3.0.2 resumes from intent doc §4 Step 1 with no other changes.**

### What this is NOT

- **Not a Step 6 patch.** This is feature-pipeline contamination at Step 1; orthogonal to the `engine/step_6_a1_vacuous_pass` patch that's gating Step 6 auto-dispatch. Both patches need to land before this arc can run end-to-end with full auto-dispatch — but they're independent.
- **Not a methodology change.** It's a producer bug-fix. One-change-per-phase discipline does not block it (per memory [feedback_framework_bug_vs_methodology.md](C:\Users\panap\.claude\projects\C--Users-panap-Documents-Forex-Backtester\memory\feedback_framework_bug_vs_methodology.md)).
- **Not blocking KH-24 live.** KH-24 doesn't use `core/features/multi_tf.py`.

---

## §5 What ran this turn

- Pre-flight §1 (minus Step 6 framework check): PASS
- Branch hygiene §2: PASS (fresh cut, prior deleted)
- Read-first §3: PASS (9 files read)
- Intent doc §4: written, committed `7dd85d4`, pushed
- **W1 producer audit (§2 of approval continuation):** FAIL → HALT
- Step 1 compute: NOT STARTED

---

## §6 What ends this turn

Branch `arc/l_arc_10_v3.0.2` pushed with:
- `docs/dispatches/arc_10_v3_0_2_intent.md` (intent doc, approved)
- `docs/dispatches/arc_10_v3_0_2_w1_audit_diagnostic.md` (this doc)

No code changes. No `results/l_arc_10_v3.0.2/` artefacts. No tracker delta. Awaiting chat resolution on the W1 producer fix sequencing (likely cross-arc HALT signal + landed fix PR before Arc 10 v3.0.2 resumes).

---

End of diagnostic.

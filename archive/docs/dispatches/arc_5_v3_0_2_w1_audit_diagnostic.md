# Arc 5 v3.0.2 — §I.6 W1 Producer Lookahead Audit — HALT DIAGNOSTIC

> **Audit target:** `core/features/multi_tf.py` — every HTF-derived feature producer
> **Mandate:** dispatch §I.6 of "Arc 5 v3.0.2 — Intent Doc APPROVED with Conditions"
> **Status:** **HALT** — one non-canonical producer found.
> **Branch:** `arc/l_arc_5_v3.0.2` (tip will be the diagnostic-doc commit; no Step 1 compute performed)
> **Triggered by:** Arc 8 CC chat's cross-arc surfacing of the `_w1_close_slope_sign` lookahead
> **Audit window:** ~10 min per dispatch promise; actual: ~5 min

---

## §1 Per-producer audit table

`core/features/multi_tf.py` registers four HTF-derived features. Producer-by-producer:

| Function | Lines | HTF source | Alignment method | Status |
|---|---:|---|---|---|
| `_d1_close_slope_sign` | 45-57 | D1 | `_build_d1_lag1_series` → `get_htf_value_at(..., require_fully_closed=True)` ([multi_tf.py:41](core/features/multi_tf.py:41)) | **CANONICAL** ✔ |
| `_d1_close_slope_magnitude` | 76-83 | D1 | same `_build_d1_lag1_series` path | **CANONICAL** ✔ |
| `_d1_atr_percentile_100` | 102-112 | D1 | same `_build_d1_lag1_series` path | **CANONICAL** ✔ |
| `_w1_close_slope_sign` | 136-157 | W1 | `pd.merge_asof(..., direction="backward", allow_exact_matches=False)` at [multi_tf.py:155](core/features/multi_tf.py:155) | **NON-CANONICAL (BUG)** ✗ |

Three of four producers route through the canonical `core.signals.htf_alignment.get_htf_value_at(..., require_fully_closed=True)` (PR #193). The fourth uses the pre-#193 idiom that PR #193's audit (`docs/audits/signal_module_eet_audit_2026_05.md`) was created to eliminate.

---

## §2 The suspect pattern — `_w1_close_slope_sign`

Quoted from [core/features/multi_tf.py:136-157](core/features/multi_tf.py:136):

```python
def _w1_close_slope_sign(pair_df: pd.DataFrame, panel=None) -> pd.Series:
    """Sign of the W1 close-on-close slope using the most recent fully-closed
    weekly bar (week N-1 visible during week N).

    Requires ``panel.aux["w1"]`` — a Panel keyed at W1.
    """
    if panel is None or not hasattr(panel, "aux") or "w1" not in panel.aux:
        return pd.Series(np.nan, index=pair_df.index, name="w1_close_slope_sign")
    pair = pair_df.attrs.get("pair") or panel.aux["w1"].pairs[0]
    w1_df = panel.aux["w1"].pair_dfs[pair]
    w1_close = (w1_df["close_bid"] + w1_df["close_ask"]) / 2.0
    slope = (w1_close - w1_close.shift(1)).rename("slope")
    # Align by `merge_asof(direction='backward')` on week-start key, then enforce
    # that the matched week's bar ended strictly before the signal bar.
    df = pd.DataFrame({"_t": pair_df.index, "_idx": np.arange(len(pair_df))})
    df = df.sort_values("_t")
    w1_pos = slope.to_frame().reset_index().rename(columns={w1_df.index.name or "index": "_t"})
    w1_pos.columns = ["_t", "slope"]
    w1_pos = w1_pos.sort_values("_t")
    merged = pd.merge_asof(df, w1_pos, on="_t", direction="backward", allow_exact_matches=False)
    merged = merged.sort_values("_idx").reset_index(drop=True)
    return np.sign(pd.Series(merged["slope"].values, index=pair_df.index)).astype("float64")
```

**Provenance:** introduced by [PR #163 (4e75598)](https://github.com/kpanapabusiness-droid/Forex-Backtester/pull/163) "infra(backtester-v3 PR-C): v3.0 WFO + KH-24 anchor WFO + broader Step-1 feature space" on 2026-05-22. PR #193 (signal-level EET alignment audit) fixed the D1 series in this same file but did NOT touch `_w1_close_slope_sign` — corroborated by `docs/audits/signal_module_eet_audit_2026_05.md`'s `multi_tf.py` row, which calls out only `_build_d1_lag1_series` as State B / fixed; the W1 producer is listed as "A / A unchanged" with the note "Already uses `merge_asof` against actual W1 timestamps; tz-invariant by construction." That note is **incorrect** for the reason in §3 below — the leak is in the resample-label semantics, not the timezone.

---

## §3 Mechanism — confirmed lookahead under BOTH UTC and 5ers_eet

The bug is in the interaction between two facts:

**Fact A — W1 bar labelling** ([core/data/aggregator.py:117,168,182](core/data/aggregator.py:117)):
- W1 is aggregated as `freq="W-MON"` with `label="left", closed="left"`.
- Each W1 bar is **left-edge labelled**: its index timestamp is the Monday start of the week; its `close` field holds the close of the **last underlying bar in the Mon→Sun window** = the Sunday end-of-week close.

**Fact B — `merge_asof` semantics in `_w1_close_slope_sign`** (line 155):
- `direction="backward", allow_exact_matches=False` matches each H1 signal bar to the most recent W1 bar with timestamp **strictly less than** the H1 bar's timestamp.
- For an H1 bar inside week N (any time strictly after Monday 00:00 of week N), the matched W1 bar is "Monday 00:00 of week N" — i.e. **the CURRENT week's W1 bar**, NOT week N-1's.

**Consequence:**
- The matched W1 bar's `slope = w1_close[week_N] − w1_close[week_{N-1}]`.
- `w1_close[week_N]` is **Sunday's close of the current week** — future data relative to the H1 signal bar (which is Mon-Sat inside week N).
- The producer returns `sign(slope)` computed against future Sunday's close. State B lookahead leak.

**The docstring's claim** ("week N-1 visible during week N") and the inline comment ("matched week's bar ended strictly before the signal bar") are **incorrect**. `allow_exact_matches=False` only excludes the literal Monday-00:00 instant; every other timestamp in the week matches the current-week W1 bar.

**Convention-independence:** the leak is in the `label='left' / closed='left'` semantics of the aggregator combined with the `merge_asof` matching. It manifests **under both UTC and 5ers_eet storage** — there is no timezone scenario under which the matched W1 bar represents week N-1 for a same-week signal bar. The PR #193 audit's "tz-invariant by construction" claim about this producer is wrong.

**Canonical replacement:** `core.signals.htf_alignment.get_htf_value_at(pair_df.index, slope.to_frame(), "slope", require_fully_closed=True)` — `require_fully_closed=True` enforces that `htf_index[matched]+1` is the next HTF bar that has already started by the signal bar's timestamp, which is the operational definition of "fully closed" (= the matched W1 bar's Mon→Sun window has fully elapsed). For a Mon-Sat signal bar inside week N, the canonical utility correctly returns week N-1's slope.

---

## §4 Downstream impact

`w1_close_slope_sign` is registered as a `FeatureSpec` at [core/features/multi_tf.py:160-174](core/features/multi_tf.py:160). It flows into:

- `core.features.pipeline.compute_feature_matrix` — one of the 27 features in the v3.0 default matrix.
- `core.steps.step_2_clustering` — path-feature clustering is on forward-path geometry NOT on the entry-time features, so Step 2 is unaffected.
- `core.steps.step_3_capturability` — per-cluster outcome stats; uses entry-time features only as cluster context, no direct dependence on `w1_close_slope_sign`. Unaffected.
- `core.steps.step_4_extraction` — **DIRECTLY AFFECTED.** Classifier training (RF + LGBM + LR) consumes the feature matrix; `w1_close_slope_sign` is one of the regressors. Any AUC ≥ 0.65 result that depends on `w1_close_slope_sign` is contaminated.
- `core.steps.step_5` — **DIRECTLY AFFECTED.** A2/A6 use Step 4's classifier; A1's filter rules can reference any feature; admit/reject decisions thus contaminated.

The lineage tag declares `CausalLineage.CLEAN` at [core/features/multi_tf.py:165](core/features/multi_tf.py:165) — this is the false declaration that the `_filter_lineage` enforcement (PR #185) takes at face value. The bug therefore evades the engine's existing lineage gate.

---

## §5 Cross-arc impact

Per dispatch §I.6 mandate to surface cross-arc impact explicitly. From `git worktree list` at audit time:

| Worktree branch | Worktree path | Status under this bug |
|---|---|---|
| `arc/l_arc_5_v3.0.2` (THIS arc) | `.claude/worktrees/compassionate-pare-bae095` | HALTED at intent-doc + audit; no contaminated artefacts produced |
| `arc/l_arc_7_v3.0.2` | `.claude/worktrees/clever-elbakyan-ad50ee` | **AT RISK** — uses same producer file. If Step 1+ already executed under this version of `multi_tf.py`, results contaminated. Needs cross-arc HALT check. |
| `arc/l_arc_10_v3.0.1` | `.claude/worktrees/gallant-pare-bed64b` | **AT RISK** — same producer file. Per chat §I.6, dispatch reads "Arc 10 v3.0.2 currently running"; the worktree list shows v3.0.1; either branch designation is current, same `multi_tf.py` → same risk. |
| `arc/l_arc_11_v3.0.2` | `.claude/worktrees/kind-leakey-f936c6` | **AT RISK** — same producer file. Tip `9e54a47` post-Amendment-5.1 merge. |
| `arc/l_arc_8_v3.0.1`, `arc/l_arc_8_v3.0.2`, `arc/l_arc_11_v3.0.1` | various | At risk by same reasoning if any are mid-Step-1+ execution. |
| Closed Wave 1 arcs (8 v3.0, 10 v3.0, 11 v3.0) | already on `main` | Closures are historical — contamination would affect their Step 4 AUC + Step 5 WFO outputs. Re-evaluation question for chat. |

**The Step 6 framework patch in flight (`engine/step_6_a1_vacuous_pass`) is orthogonal to this finding** — it addresses an audit-framework false-positive for A1 winners, not a feature-pipeline correctness issue. Both fixes are needed; neither subsumes the other.

---

## §6 Verdict + handoff

**HALT.** No Step 1 compute initiated on this branch.

The decision rule per dispatch §I.6:

> Any producer non-canonical: HALT. Write `docs/dispatches/arc_5_v3_0_2_w1_audit_diagnostic.md` with: per-producer status, function name + line numbers, quoted code snippet of the suspect pattern, push branch, end turn, await chat resolution (likely a separate engine fix dispatch).

All four artefacts produced; branch will be pushed; ending turn awaiting chat resolution.

**Likely chat actions** (not for me to take — surfaced for visibility):
1. Open a separate engine PR `engine/multi_tf_w1_canonical_alignment` replacing the `merge_asof` block with `get_htf_value_at(..., require_fully_closed=True)` (the 3 D1 producers already do this; the fix is mechanical).
2. Cross-arc HALT message to Arcs 7 v3.0.2 / 10 / 11 worktree chats — pause any Step 1+ work until the fix lands.
3. Update `docs/audits/signal_module_eet_audit_2026_05.md` `multi_tf.py` row's `_w1_close_slope_sign` entry from "A / A unchanged" to "B / B" with the convention-independent leak note from §3 above. PR #193 audit overstated this producer's coverage.
4. Decision on whether to re-evaluate closed Wave 1 arc closures (8 v3.0, 10 v3.0, 11 v3.0) for w1-feature contamination. Severity depends on whether `w1_close_slope_sign` made it into any winning config's feature importance — a quick scan of the closed closures' `best_architecture.features_in_winning_config` would answer that without re-running anything.

**After fix lands on main:** rebase `arc/l_arc_5_v3.0.2` onto main and re-run the §I.6 audit. If clean, proceed to Step 1 execution per the approved intent doc §E.

---

End of diagnostic.

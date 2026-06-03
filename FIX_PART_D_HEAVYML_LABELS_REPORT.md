# FIX PART D — heavy_ml label provenance + same-bar tie-break — **RESOLVED**

> Closes the two open items in `HONEST_ENGINE_SWEEP.md` Part D (the FLAG).
> heavy_ml's training labels are now SL-honest and provably in-tree, so the
> path is safe to UN-quarantine for discovery.
>
> Date: 2026-06-03 · Engine state: clean-base reset (fast replay retired 2026-06-02)

---

## What the sweep found (Part D = FLAG)

1. **FLAG-D1 — provenance gap.** `bars_to_1r_mfe` (the meta-label + survival
   target input) was *read* by `core/heavy_ml_probe` but **produced nowhere
   in-tree**, so any pool fed to heavy_ml had to be enriched out-of-tree —
   honesty unverifiable, possibly a retired-replay frame.
2. **FLAG-D2 — same-bar tie-break bug.** The label compared `exit_reason`
   against the literal `"sl"`, but the pool simulators emit `"hard_sl"` with
   no normalization. A same-bar (+1R-high **and** SL-low) trade therefore
   resolved to `y = 1` (**WIN**) — the Arc-10 defect (same-bar → win)
   reincarnated in label space.

---

## The fix

### D.1 — honest, in-tree `bars_to_1r_mfe` producer

New leaf module **`core/sim/honest_label.py`** — `reached_1r_before_sl(...)`
walks the actual forward bars with **take-the-loss ordering** (the hard stop
is checked BEFORE +1R on every bar), so "+1R reached before SL" is true ONLY
if +1R landed on a bar **strictly before** any stop breach; a same-bar
+1R/SL bar resolves SL-first and is **not** a reach:

```python
def reached_1r_before_sl(*, high_bid, low_bid, entry_idx, exit_off,
                         entry_price, sl_price, sl_distance,
                         exit_at_bar_open=False, r_threshold=ONE_R):
    if not math.isfinite(sl_distance) or sl_distance <= 0:
        return float("nan")
    last = exit_off - 1 if exit_at_bar_open else exit_off
    for off in range(0, last + 1):
        bidx = entry_idx + off
        lo = float(low_bid[bidx])
        if off > 0 and math.isfinite(lo) and lo <= sl_price:
            return float("nan")          # stop breach before any qualifying +1R
        hi = float(high_bid[bidx])
        if math.isfinite(hi) and (hi - entry_price) / sl_distance >= r_threshold:
            return float(off)            # +1R on a non-stop bar => strictly before SL
    return float("nan")
```

It is now **called by both pool simulators**, so the column is a native part
of the Step-1 pool (`pool.parquet`) that heavy_ml loads — no out-of-tree
enrichment, fully reproducible:

- `core/arc/arc_pool_builder.py:281` — the documented heavy_ml pool source
  (`run_probe.py --pool .../step_1/pool.parquet`).
- `core/discovery/pool_simulator.py:331` — the discovery pool source.

### D.2 — same-bar tie-break normalization (take-the-loss in label space)

`core/sim/honest_label.is_stop_loss_exit()` recognises every stop spelling a
producer emits — `{"sl", "hard_sl", "stop_loss", "stop"}`, case-insensitive.
Both label builders now resolve a same-bar +1R/SL tie to a **LOSS** for any
of these. The builders moved to a new sklearn-free module
**`core/heavy_ml_probe/labels.py`** (`build_meta_label_target`,
`build_survival_target`), re-exported from `meta_labeling.py` / `survival.py`
so every existing import is unchanged. The decoupling lets the regression
test pin the *real* label functions in CI's minimal (numpy/pandas) env.

With the honest producer in place the `bars_to_1r_mfe == bars_held` + stop
case can no longer arise for an intrabar stop (the producer never registers
+1R on the stop bar), so D.2 is belt-and-braces: the label agrees with the
engine even if the column is ever sourced elsewhere.

---

## Validation (per dispatch)

### Old vs new label — constructed same-bar +1R/SL trade

Trade: `bars_to_1r_mfe == bars_held == 5`, `exit_reason == "hard_sl"` (a bar
whose high hit +1R **and** whose low hit the stop).

| | comparison | label |
|---|---|---|
| **OLD** (`exit_reason == "sl"`) | `"hard_sl" == "sl"` → `False` | `y = 1` → **WIN (bug)** |
| **NEW** (`is_stop_loss_exit`) | `is_stop_loss_exit("hard_sl")` → `True` | `y = 0` → **LOSS (correct)** |

End-to-end through the real simulator (no hand-set columns), a same-bar tie
trade exits `hard_sl` with raw `mfe_r >= 1.0` (the high *did* touch +1R
intrabar) yet honest `bars_to_1r_mfe = NaN` → `build_meta_label_target → 0`,
`build_survival_target event → 0`. Pinned by
`tests/heavy_ml/test_label_take_the_loss.py::test_chain_same_bar_tie_simulator_to_label_is_loss`
and `::test_sim_same_bar_tie_nan_despite_raw_mfe_touching_1r`.

### Producer is in-tree + reproducible

`bars_to_1r_mfe` is now **written** by `reached_1r_before_sl` at
`core/arc/arc_pool_builder.py:281` and `core/discovery/pool_simulator.py:331`
(declared in the pool schema at `arc_pool_builder.py:425` and
`pool_simulator.py` `TradeRow`). Deterministic, standard-library walk —
provenance is the function + the bar data, nothing external.

### No replay/precomputed frame on the label path (grep)

```
grep -rniE "realized_r|_3p5|B_exit|simulate_path|np\.load|\.npy" \
     core/heavy_ml_probe/ core/sim/honest_label.py \
     core/discovery/pool_simulator.py core/arc/arc_pool_builder.py
→ (no matches)
```

The only `read_parquet` on the path is `pipeline.load_pool` reading the
**in-tree** Step-1 pool. The `realised_r` name in `meta_labeling.threshold_sweep`
is the kept/dropped mean-R **diagnostic** built from the honest `final_r`
column (the sweep itself classified this as honest), not a label input. The
`mfe_so_far_r` path column is a diagnostic and is **not** read by the label
builders (which consume only `bars_to_1r_mfe` / `bars_held` / `exit_reason` /
`final_r`).

### Tests — green

- **`tests/heavy_ml/test_label_take_the_loss.py`** — 18 cases, **CI-gated,
  not research-marked**, numpy/pandas only. Proven minimal-env safe: all its
  imports resolve with `sklearn` / `flaml` / `joblib` / `statsmodels` / `scipy`
  blocked (none load). Covers producer (before / after / same-bar / never),
  producer-through-real-simulator (engine bar indices), the real label
  functions, and the old-vs-new bug demonstration.
- `tests/heavy_ml_probe/test_meta_labeling.py` + `test_survival.py` — extended
  with `hard_sl` same-bar cases (real builders, via the re-export path).
- `tests/protocol_runtime/test_arc_pool_builder.py` + `tests/discovery/test_pool_simulator.py`
  — assert the emitted `bars_to_1r_mfe` is SL-honest.
- **Full non-research suite (CI parity):** `1649 passed, 294 skipped, 0 failed`.
- ruff clean on every changed/new file.

### Guards unchanged

heavy_ml's causal-lineage gate (`core/heavy_ml_probe/causal_lineage.py`) and
holdout-guard (`core/heavy_ml_probe/automl.py`) are **not modified** — they
defend against feature lineage / lookahead, not label dishonesty. This fix is
the missing label-honesty defense; it composes with them.

---

## Definition of done

- [x] `bars_to_1r_mfe` + the reach-1R-before-SL label have an honest, in-tree,
      reproducible producer sourced from the SL-honest forward walk
      (`core/sim/honest_label.reached_1r_before_sl`, called by both simulators).
- [x] Same-bar tie-break labels a +1R/SL `hard_sl` trade as a **LOSS**
      (`is_stop_loss_exit` normalization).
- [x] Regression test committed, CI-gated (minimal-env safe), green.
- [x] Old-vs-new label shown on the bug case (win → loss).
- [x] Grep confirms no replay-frame label source.

**heavy_ml is safe to UN-quarantine for discovery** on label-honesty grounds.
(Part C — broker-cost wiring on the gate path — remains a separate, open
blocker tracked in `HONEST_ENGINE_SWEEP.md`.)

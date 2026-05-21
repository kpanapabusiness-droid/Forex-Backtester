# Arc 8 Step 5 — Pre-check 2: D1 t=1 leak audit

> Dispatch §47-54: "D1 at t=1 hit recall 1.000 / precision 0.909 — suspiciously clean. Verify: close_r_at_t=1 uses bar t=1 CLOSE only, no high/low touches that could leak future SL info; mfe_so_far_r_at_t=1 and mae_so_far_r_at_t=1 use ONLY bars [0, 1]; No path-so-far feature reads beyond bar 1. Output verdict (clean / suspect)."

## Verdict

**CLEAN — no future-bar leak.** The D1 t=1 result is valid. The "suspiciously clean" recall 1.000 / precision 0.909 on the 80/20 holdout is fully explained by (a) c1's high positive class base rate (0.81) and (b) small holdout sample size (n=36). The classifier shows real but modest discriminative power (CV AUC 0.637, 50% specificity on holdout negatives), not a leak.

Step 5 WFO can proceed against the D1 classifier without leak-revision rework.

---

## Code path inspected

`scripts/l_arc_8/step4_extractability.py::compute_d1_features_at_t` (lines 257-373).

### Slicing logic — what bars are read at t=1

```python
slice_end = t + 1                                            # = 2 when t=1
seg = path_sorted.iloc[:slice_end]                           # bar_offset in {0, 1}
```

Verified empirically on a sample c1 trade (trade_id=25):

```
At t=1, slice_end=2; seg uses bars: [0, 1]
Max bar_offset in slice: 1
Min bar_offset in slice: 0
Any bar_offset > t in slice? False
```

Confirms no bar beyond t=1 enters the feature computation.

### Feature-by-feature trace at t=1

| Feature | Computation | Bars read | Leak risk |
|---|---|---|---|
| `close_r_at_t` | `close_new[idx_t]` where `idx_t = len(seg)-1 = 1` | bar 1 only | none |
| `mfe_so_far_r_at_t` | `mfe_new[1]` = scale × `mfe_so_far_r[1]` from Step 1 paths | bars 0..1 (running max from Step 1 emission, strictly causal) | none |
| `mae_so_far_r_at_t` | `mae_new[1]` = scale × `mae_so_far_r[1]` from Step 1 paths | bars 0..1 (running min, strictly causal) | none |
| `bars_in_profit_at_t` | `np.sum(close_new > 0)` over seg | bars 0..1 | none |
| `local_peaks_so_far_at_t` | `np.sum(mfe_new[1:] > mfe_new[:-1])` over seg | bars 0..1 (single pair) | none |
| `monotonicity_so_far_at_t` | `np.mean(in_profit[1:] >= in_profit[:-1])` over `close_new[close_new > 0]` | bars 0..1 | none |
| `velocity_first_t` | `mfe_at_t / max(t, 1)` | derived from mfe_at_t only | none |

**Step 1 emission audit (chain-of-trust):**

`mfe_so_far_r[i]` and `mae_so_far_r[i]` from `trades_paths.csv` are computed in `scripts/l_arc_8/step1_plumbing.py::_simulate_pair` via the loop:

```python
for k in range(entry_idx, last_bar_for_path + 1):
    ...
    if cand_mfe_price > mfe_so_far_price:
        mfe_so_far_price = cand_mfe_price
    if cand_mae_price > mae_so_far_price:
        mae_so_far_price = cand_mae_price
    ...
    row = _PathRow(..., mfe_so_far_r=mfe_so_far_r, mae_so_far_r=mae_so_far_r, ...)
```

`mfe_so_far_price` and `mae_so_far_price` are running max/min variables initialized to 0 at entry and updated only on each subsequent bar's `high` / `low`. The value at bar offset `i` is the max/min over bars `[0..i]` — strictly causal.

This is the canonical reference impl per protocol §15a Open-18 closure (mfe_so_far_r is running max of high_r intrabar, NOT max of close_r), consistent with all prior arcs.

### Eligibility check — observable in OOS at bar 1 close

```python
eval_ = _eval_trade_at_sl(path, unit_selected_sl, original_sl)
actual_exit_bar = eval_.truncated_at_bar
is_eligible = actual_exit_bar >= t
```

This uses future information (the eventual exit bar under SL=4.0×ATR) to determine eligibility. However: in OOS deployment at bar 1 close, we observably know whether the trade has been SL-hit between bars 0 and 1 — we've watched bars 0 and 1 unfold in real-time. So `actual_exit_bar >= 1` IS observable at bar 1 close; it's equivalent to "has SL not been hit during bars 0 or 1?". This is NOT a leak — it's the canonical D1 deferred-classification gate: only classify trades that are still open at the decision bar.

The dispatch's hypothetical concern was about future SL information leaking through `high_r` / `low_r` reads. Verified: features read `high_r` and `low_r` ONLY from bars 0..1 (via the `mfe_so_far_r` / `mae_so_far_r` reading from the Step 1 emission which itself is causal). No leak.

---

## Quantitative deflation of the "suspiciously clean" headline

Reproducing the 80/20 holdout split used by the threshold sweep:

```
c1 (n=177) at SL=4.0×ATR:
  positive rate (final_r >= 1.0 in 4×ATR frame): 0.808

Holdout 80/20: train_n=141, test_n=36
  test pos count: 30
  test neg count: 6
  test pos_rate: 0.833
```

c1 has an **extreme positive class imbalance (81%)** because its SL is very wide (4.0×ATR). Most trades that don't immediately fall apart will, over the 240-bar forward window, eventually reach +1.0 in the 4×ATR-denominated R-frame (only 0.5R in 2×ATR original units).

Decomposing the headline `recall=1.000, precision=0.909` at threshold 0.60:

```
TP = 30 (all 30 test positives admitted; recall = 30/30 = 1.000)
admit_total = 33 (TP / precision = 30 / 0.909 ≈ 33)
FP = 3 out of 6 test negatives admitted
specificity = 0.500 (classifier rejected 3 of 6 negatives)
admit_pct = 0.917 (vs base rate 0.833)
```

**Interpretation:**

- `recall = 1.000` looks dramatic but is partly mechanical at base_rate=0.833 + small test_n. The classifier passes the v2.2 §3 `recall ≥ 0.60` gate trivially because the gate is much weaker than the base rate.
- `precision = 0.909` is barely above the test base rate (0.833). The classifier's lift over the base rate (admit-all-strategy) is small but non-zero.
- **Specificity = 0.500** is the meaningful metric: classifier rejects 50% of negatives — real but modest discriminative power, consistent with CV AUC 0.637.

This is NOT a leak; it's a structural property of high-base-rate archetypes under v2.2 §3's recall floor. The recall ≥ 0.60 gate is binding for **low-base-rate** archetypes (c3 at pos_rate=0.13 — which DIED at threshold sweep) but trivially satisfied for **high-base-rate** archetypes like c1.

### Caveat for WFO interpretation

The threshold sweep's holdout (n=36) is small. WFO with 3-month OOS windows will give more robust per-window precision/recall, and the analyst should weight the WFO worst-window result more heavily than the Step 4 holdout numbers.

### Caveat for Pipeline D1 deployment economics

A classifier that admits 92% of trades at base rate 83% is, in deployment terms, **barely filtering** — it provides only 50% rejection of true negatives. The Pipeline D1 trade economics (per Open-22/23/24 cross-arc finding from Arc 4 RERUN and Arc 5) require:

```
(admit_rate × admit_mean_r) > (reject_rate × |reject_mean_r|) + (early_exit_rate × |early_exit_mean_r|)
```

With admit_rate ~0.92, reject_rate ~0.08, AND for c1 SL=4.0×ATR there is no early-exit pool (4.0×ATR is too wide to fire intrabar SL before t=1 in most trades), the deployment economics here are dominated by the admit pool. WFO will quantify whether the admit pool's mean_r is positive enough to absorb the small reject pool's negative drag. Step 5 WFO §10 ship gates apply.

---

## Files referenced

- `scripts/l_arc_8/step4_extractability.py` — `compute_d1_features_at_t` (lines 257-373)
- `scripts/l_arc_8/step1_plumbing.py` — `_simulate_pair` (path emission causality)
- `scripts/l_arc_8/step3_capturability.py` — `_eval_trade_at_sl` (eligibility check; observable in OOS)
- `results/l_arc_8/step1_verbatim/trades_paths.csv` — Step 1 path emission (causal mfe_so_far_r / mae_so_far_r)
- `results/l_arc_8/step2/clusters_K4.csv` — c1 trade IDs
- `results/l_arc_8/step4/archetype_v-shape_recovery_forward-geometry_weak_c1_D1_policy.yaml` — Step 4 closure: chosen_t=1, threshold=0.60, precision_holdout=0.909, recall_holdout=1.000

**Result: clean. Proceed to WFO without leak-revision rework.**

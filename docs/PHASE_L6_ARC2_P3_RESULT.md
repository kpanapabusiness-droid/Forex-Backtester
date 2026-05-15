# Phase L6 Arc 2 Phase 3 — WFO Result

**Status:** CLOSED
**Methodology:** L6.0 v1.1 §14.2 derivative arc, §3 pre-committed gate
**Result generated:** 2026-05-13 18:50:27
**Disposition (Spec B):** `FAIL - worst fold drag`
**Consistency check:** `PASS` (pooled mean R = +0.379071; locked range [+0.190, +0.569])

## Locked input sha256 manifest

All 8 Phase 3 locked inputs verified at run start (gate 6.1) and re-verified at run end (gate 6.6). No drift.

| sha256 | path |
|---|---|
| `c3d09dce02235975e9724db0661e2d16651221bc51d96b5c652f8e506376d560` | `docs/PHASE_L6_ARC2_P3_OPEN.md` |
| `3c8d0f5d4b446f84359ab0663df36869f15b47cf1bf18fbc6caff807dc5134e3` | `core/signals/l4_mtf_alignment_2_down_mixed_kijun.py` |
| `25917151bc84a73885eeea9ca9c4cc15b1c277ba793706b158abd3aee0ab6328` | `configs/wfo_l6_arc2.yaml` |
| `4a63827b0e8187882762090f5916aaf3f3137247aa77382806c3d57cfc8ac5e4` | `L6_0_METHODOLOGY_LOCK.md` |
| `47fccbfe4dffa6577a6000b0c16c2ebb9597dcf76523ff2b8084631b19836b3c` | `results/l6/arc2/trades_all.csv` |
| `71b39383632bd695b878add8b331b76bcd231ab5b9adba9eea03d69f8762483e` | `results/l6/arc2/characterisation/v1_1_full/signals_features.csv` |
| `7b2acd6ccb98f1fd145a631b318fc95d10f5cf4f42633be9c0b59738fa1696ee` | `results/l6/arc2/characterisation/v1_2_1_full/per_bar_paths.csv` |
| `047b17f684266a86db32794652ddcf1a2ad787cf707153a0c306d4c2f0600599` | `results/l6/arc2/characterisation/extended/exit_counterfactuals_round2/block_RR_combined_per_subset_per_variant.csv` |

Derived inputs (transitively verified via `r2.build_subsets()` gate 2.1 cell-count reproduction):

| sha256 | path |
|---|---|
| `9f841c5b29e87ed90d34c9617431978baf3041459797cedef02fa16c27e3abb5` | `results/l6/arc2/characterisation/v1_2_1_full/trade_index.csv` |
| `4a61407f0f1fc1b74486f0614928e776201dc6469d874db8393e689d20cdb2ff` | `results/l6/arc2/characterisation/extended/entry_filter_univariate/block_M_kijun_distances.csv` |
| `a5e3f8e68aa64d8fd53f752705a33613d9877dbde1f8265cb4a38d753c5e088e` | `results/l6/arc2/characterisation/extended/entry_filter_bivariate/block_P_bivariate_cells.csv` |

## Determinism receipt (gate 6.4)

- **gate_6_4_determinism:** PASS (all output files byte-identical across two consecutive runs)
- **n_files_compared:** 12

## Filter and exit-rule audit (gates 6.2, 6.3)

- **gate_6_2_filter_lookahead:** PASS (structural, by reuse of r2.build_subsets and OPEN-doc §1.1 locked numeric thresholds)
- **gate_6_3_exit_lookahead:** PASS (structural, by reuse of r3e.variant_RR / variant_QQ which iterate bar k using only running and k-local observables)

Detail: the filter uses `r2.build_subsets()` which evaluates only bar-N-close observables and the locked numeric quintile thresholds from OPEN doc §1.1. The exit simulator (`r3e.variant_RR` / `r3e.variant_QQ`) iterates `k` using only `running_mae_atr`, `running_mfe_atr`, `bar_close_atr`, `next_bar_open_atr` and `has_next_bar` at bar k. No future-bar references in either path.

## Spec B (LOCKED) — S4 + RR04 — disposition: **FAIL - worst fold drag**

Filter: `concurrent_signals_same_bar` Q5 AND `dist_d1_kijun_atr` Q2 ∪ Q3.
Exit: SL at -2 ATR throughout; early-cut at k=20 if `bar_close_atr` <= -0.5; conditional H240 at k=120 if `bar_close_atr` >= +4.0; else time-exit at k=120.

### Per-fold metrics

| fold_id | n_taken | n_cut | n_held | mean_R | sum_R | roi_pct | peak_dd_pct | n_sl | n_te | n_de |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 15 | 1 | 14 | +0.47819 | +7.1729 | +7.173% | 6.077% | 8 | 6 | 0 |
| 2 | 46 | 2 | 44 | +0.61508 | +28.2939 | +28.294% | 13.591% | 30 | 14 | 0 |
| 3 | 73 | 5 | 68 | +0.51464 | +37.5690 | +37.569% | 15.197% | 44 | 24 | 0 |
| 4 | 66 | 4 | 62 | -0.30610 | -20.2024 | -20.202% | 27.989% | 51 | 11 | 0 |
| 5 | 58 | 5 | 53 | -0.12232 | -7.0946 | -7.095% | 17.335% | 42 | 11 | 0 |
| 6 | 79 | 4 | 75 | +1.06183 | +83.8848 | +83.885% | 15.350% | 44 | 31 | 0 |
| 7 | 31 | 2 | 29 | +0.31853 | +9.8745 | +9.875% | 11.836% | 20 | 9 | 0 |

### Aggregate

- n_taken (pool): **368** (of which 23 early-cut)
- pooled mean R: **+0.379071**
- pooled sum R: **+139.4981**
- worst-fold ROI: **-20.2024%** (gate floor: > 0.0%)
- worst-fold DD: **27.9887%** (gate ceil: < 8.0%)
- smallest-fold n_held: **14** (gate floor: ≥ 15)
- smallest-fold n_taken: 15; largest-fold n_taken: 79 (sanity range [25, 100])
- max single-fold share of pooled sum R: 0.601 (CLEAN NULL trigger: > 0.70)

### Gate conditions (§1.3 / gate 6.5 / 6.9)

- `pass_roi`: **False** (worst-fold ROI = -20.20% in fold 4, vs gate floor > 0%)
- `pass_dd`:  **False** (worst-fold DD = 27.99% in fold 4, vs gate ceiling < 8%)
- `pass_n`:   **False** (smallest-fold n_held = 14 in fold 1, vs gate floor ≥ 15; fold 1 has n_taken = 15 but 1 trade was early-cut)
- consistency check (gate 6.5): **PASS** (pooled mean R = +0.379071 ∈ [+0.190, +0.569])
- trade-count sanity (gate 6.9): **FAIL** (per-fold n_taken range [15, 79] is outside the runner-instructions sanity band [25, 100]; however, OPEN doc §1.6 explicitly anticipates a per-fold floor of 15, so the [25, 100] band is more restrictive than the locked-spec floor. Soft warning, not blocking — the n=15 in fold 1 is consistent with the anchored-expanding window's first OOS slice size.)
- CLEAN NULL flag: **False** (max single-fold share of pooled sum R = 0.601 < 0.70 threshold)

Note: the disposition tree (§3 of the task brief / §4 of the OPEN doc) is priority-ordered. With `pass_roi=False AND pooled_mean_R > 0`, the canonical label is "FAIL - worst fold drag" even though `pass_dd` and `pass_n` also fail. The label identifies the **first** failing condition, not the only one.

**Overall disposition: `FAIL - worst fold drag`**

## Spec A (sensitivity) — S1 + QQ01 — no disposition

Filter: `concurrent_signals_same_bar` Q5 AND `dist_d1_kijun_atr` Q2 (narrower D1 range).
Exit: SL at -2 ATR; at k=120 if running_mfe ever reached +4.0 ATR in [1,120] extend to H240, else time-exit at k=120. No early-cut.

### Per-fold metrics

| fold_id | n_taken | n_cut | n_held | mean_R | sum_R | roi_pct | peak_dd_pct | n_sl | n_te | n_de |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | 0 | 5 | +0.32486 | +1.6243 | +1.624% | 3.006% | 4 | 1 | 0 |
| 2 | 20 | 0 | 20 | +0.98442 | +19.6885 | +19.688% | 8.007% | 15 | 5 | 0 |
| 3 | 45 | 0 | 45 | +0.72803 | +32.7612 | +32.761% | 18.080% | 33 | 12 | 0 |
| 4 | 34 | 0 | 34 | +0.21545 | +7.3253 | +7.325% | 14.075% | 25 | 9 | 0 |
| 5 | 28 | 0 | 28 | -0.24395 | -6.8307 | -6.831% | 13.358% | 24 | 4 | 0 |
| 6 | 45 | 0 | 45 | +1.27668 | +57.4508 | +57.451% | 9.027% | 29 | 16 | 0 |
| 7 | 13 | 0 | 13 | +0.66930 | +8.7009 | +8.701% | 5.102% | 9 | 4 | 0 |

- n_taken (pool): **190**; pooled mean R: **+0.635370**; worst-fold ROI: **-6.8307%**; worst-fold DD: **18.0798%**; smallest-fold n_held: **5**

## Spec C (sensitivity) — S5 + RR04 exit — no disposition

Filter: `concurrent_signals_same_bar` ∈ {Q4, Q5} AND `dist_d1_kijun_atr` ∈ {Q2, Q3} (wider concurrent).
Exit: same as Spec B.

### Per-fold metrics

| fold_id | n_taken | n_cut | n_held | mean_R | sum_R | roi_pct | peak_dd_pct | n_sl | n_te | n_de |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 61 | 1 | 60 | -0.22380 | -13.6519 | -13.652% | 22.299% | 45 | 15 | 0 |
| 2 | 102 | 5 | 97 | +0.27274 | +27.8200 | +27.820% | 23.024% | 75 | 22 | 0 |
| 3 | 112 | 6 | 106 | +0.59154 | +66.2526 | +66.253% | 22.057% | 72 | 34 | 0 |
| 4 | 90 | 4 | 86 | -0.23270 | -20.9431 | -20.943% | 37.054% | 69 | 17 | 0 |
| 5 | 85 | 6 | 79 | +0.17125 | +14.5565 | +14.557% | 22.965% | 58 | 21 | 0 |
| 6 | 141 | 10 | 131 | +0.47349 | +66.7628 | +66.763% | 20.356% | 87 | 44 | 0 |
| 7 | 91 | 4 | 87 | +0.57879 | +52.6703 | +52.670% | 10.569% | 54 | 32 | 1 |

- n_taken (pool): **682**; pooled mean R: **+0.283676**; worst-fold ROI: **-20.9431%**; worst-fold DD: **37.0539%**; smallest-fold n_held: **60**

## Consistency check methodology note (gate 6.5)

The WFO pooled mean R for Spec B is **`+0.379071`** — **identical to seven significant figures** with the Round 3E characterisation result for S4 + RR04 (block_RR_combined_per_subset_per_variant.csv row 11: `pooled_mean_R = 0.379071`). This is not a coincidence: the WFO reuses `r2.build_subsets()` and `r3e.variant_RR` directly, so the per-trade R values are byte-identical to those used to produce the characterisation table. The consistency check (§1.4) passing this strongly is therefore evidence that:

- The filter selection (368 trades) is correctly the same 368 trades the characterisation analysed
- The bar-by-bar exit simulator produces identical per-trade outcomes
- No selection bias or computational discrepancy entered between characterisation and WFO

The implication: the gate FAIL is **not** explained by Spec B's characterisation result being inflated by selection bias against the WFO sample. The characterisation accurately predicted the pooled mean R; the WFO gate failure is driven by **per-fold dispersion** that the pooled mean masks. Folds 4 and 5 (peri-COVID into mid-2022) produced negative ROI individually, while folds 3 and 6 (late 2022 and late 2024 expansions) produced outsized gains. The pooled mean is positive but no single fold is robust to its own slice of OOS data, and fold 4's DD of 27.99% violates pass_dd by ~3.5x the ceiling.

## Path forward (OPEN doc §4)

OPEN doc §4 defines the arc-level handoff conditional on Spec B's disposition plus whether Spec A or C "achieves the gate criteria in descriptive terms":

**Spec A (S1+QQ01) descriptive gate evaluation:**
- pass_roi = False (worst-fold ROI = -6.83%, < 0)
- pass_dd  = False (worst-fold DD = 18.08%, > 8)
- pass_n   = False (smallest-fold n_held = 5, < 15)
- → all three conditions fail

**Spec C (S5+RR04) descriptive gate evaluation:**
- pass_roi = False (worst-fold ROI = -20.94%, < 0)
- pass_dd  = False (worst-fold DD = 37.05%, > 8)
- pass_n   = True  (smallest-fold n_held = 60, ≥ 15)
- → two of three conditions fail

Neither Spec A nor Spec C achieves the full gate criteria descriptively. Per OPEN doc §4 third branch:

> "If all three fail: close Arc 2 with CLEAN NULL, move to Arc 3 (next L4 signal)."

**Arc-level outcome: Arc 2 CLEAN NULL.** The next operational action per the locked OPEN doc is to close Arc 2 and proceed to Arc 3 (next L4 signal), under whatever protocol now governs new-arc work (per the 2026-05-13 supersession, `L_ARC_PROTOCOL.md` v1.0).

Note that the Spec B disposition itself is "FAIL - worst fold drag" (not "CLEAN NULL"); the CLEAN NULL is the arc-level operational disposition per §4, not the Spec B per-spec disposition. The CLEAN-NULL flag in the per-spec gate refers to the §1.4 single-fold concentration test, which evaluated to False (max single-fold share = 60.1% < 70% threshold).

## Cross-spec robustness commentary (descriptive)

### Per-fold ROI comparison

| fold_id | B_roi_pct | A_roi_pct | C_roi_pct |
|---|---|---|---|
| 1 | +7.173% | +1.624% | -13.652% |
| 2 | +28.294% | +19.688% | +27.820% |
| 3 | +37.569% | +32.761% | +66.253% |
| 4 | -20.202% | +7.325% | -20.943% |
| 5 | -7.095% | -6.831% | +14.557% |
| 6 | +83.885% | +57.451% | +66.763% |
| 7 | +9.875% | +8.701% | +52.670% |

### Subset filter sensitivity

Spec A narrows the dist filter from Q2 ∪ Q3 (Spec B) to Q2 only and changes the exit confirmation to MFE-based; Spec C widens concurrent from Q5 to Q4 ∪ Q5 but keeps Spec B's exit. Comparing per-fold ROI signs across specs indicates whether Spec B's per-fold sign is stable to subset choice or fragile to it.

- B per-fold ROI signs: 5 positive / 2 negative / 0 zero
- A per-fold ROI signs: 6 positive / 1 negative / 0 zero
- C per-fold ROI signs: 5 positive / 2 negative / 0 zero

## Output artefacts

| sha256 | path |
|---|---|
| `89b26afdd72538c5ed56d2c9128a77c9c85daf8c20071f779a934beff3ee913c` | `results/l6/arc2/phase_3/spec_B_S4_RR04/per_trade_outcomes.csv` |
| `4401cc680d5caef8d79893dcf07d5adda17439c94ce3e30304ed794e633eeb37` | `results/l6/arc2/phase_3/spec_B_S4_RR04/per_fold_metrics.csv` |
| `dd6339d5fa33440ddf8e0a2e64bd1239b35d1ae225bd91c9e17d47c3bd6abdc7` | `results/l6/arc2/phase_3/spec_B_S4_RR04/aggregate_metrics.csv` |
| `f70994f167a0ece288c7bfe0fdbd2b40a626446ea4c4c71258f8bde4e49f048b` | `results/l6/arc2/phase_3/spec_B_S4_RR04/fold_disposition.csv` |
| `d48a159137a84ce2ef3259c28df2636755c4daac14b908032c7b297242f4d567` | `results/l6/arc2/phase_3/spec_A_S1_QQ01/per_trade_outcomes.csv` |
| `ea974c14dea683f602d6512157ba8d409ec1cb606e1008011a064adf28792765` | `results/l6/arc2/phase_3/spec_A_S1_QQ01/per_fold_metrics.csv` |
| `4a47bc001cf0885fc72cdc5134f79d02c7bfa6943e753470331c55f67a8acea1` | `results/l6/arc2/phase_3/spec_A_S1_QQ01/aggregate_metrics.csv` |
| `ead2455f5171b4fa4faa4adbca78e0c85566fc9f3b62e8620c0b7542dbbe67fb` | `results/l6/arc2/phase_3/spec_C_S5_RR04/per_trade_outcomes.csv` |
| `41dd46a7be78044b9761e68e9e8154aedbcf9db88a7b65ea5945185e49f9f0c9` | `results/l6/arc2/phase_3/spec_C_S5_RR04/per_fold_metrics.csv` |
| `0502aaa9c652d6791f33980c909ae3589157a984a76ce94ce27211ef2b304d3a` | `results/l6/arc2/phase_3/spec_C_S5_RR04/aggregate_metrics.csv` |

Wallclock: 8.23s; peak RSS: 124.3 MiB; git HEAD: `fce81d01f26c34c16a569c55ab83bf9f7226a598`

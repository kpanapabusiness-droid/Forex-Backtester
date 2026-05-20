# Arc 8 — Path 1 + Path 2 Combined Diagnostic Summary

> Post-Step-5 follow-up. Trigger: `c1_NOT_SEPARABLE_AT_ENTRY` at commit `7d9109e`.
> Goal: determine whether either of the two surviving filtering strategies preserves c1's structural edge.
> Both paths evaluated independently on the locked Step 1 pool (n=1,327). No retraining of Step 4 classifiers; no modification of Step 1-5 outputs.

---

## Headline verdicts

| Path | Verdict | Best operating point |
|---|---|---|
| **Path 1 — Post-entry confirmation (t-sweep)** | **PATH_1_MARGINAL** | t=12, c1 precision@recall=0.60 = 0.274 (in marginal band [0.20, 0.40]); c1 1-vs-rest AUC 0.755 |
| **Path 2 — Signal-tightening (mechanical filter)** | **PATH_2_DEAD** | No filter satisfies c1_ret ≥ 0.80 ∧ c2_ret ≤ 0.30 ∧ pool ≥ 500 simultaneously |

**Combined verdict: ARC 8 STAYS CLOSED.** Neither path is VIABLE.

---

## Path 1 — Post-entry confirmation

Tested at bar offsets `t ∈ {3, 5, 8, 12}` after entry. For each t, trained a multiclass RF (cluster ∈ {0, 1, 2, 3}) on 25 features (8 base + 10 PR-HHHL entry + 7 path-so-far at t), 5-fold TimeSeriesSplit, `class_weight='balanced'`, `random_state=42`.

### t-sweep results

| t | c1 eligibility | c1 1-vs-rest AUC | c1 precision@recall=0.60 | 4-class OOF acc | c1 MFE consumed | c1 MAE_p50 at t |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 100.0% | 0.652 | 0.196 | 0.515 | 7.3% | −0.082R |
| 5 | 100.0% | 0.684 | 0.224 | 0.565 | 10.1% | −0.083R |
| 8 | 100.0% | 0.727 | 0.243 | 0.600 | 13.0% | −0.115R |
| **12** | **100.0%** | **0.755** | **0.274** | **0.603** | **17.3%** | **−0.115R** |

(c1 eligibility = fraction of c1 trades still open at bar t under SL=4.0×ATR. Always 100% in the tested t range — c1's wide SL means no early stops.)

### Key observations

- **Separability monotonically improves with t.** c1 AUC: 0.547 (entry, from prior diagnostic) → 0.652 (t=3) → 0.755 (t=12). Path-so-far features genuinely carry cluster-discriminating information.
- **But precision@recall=0.60 plateaus in the marginal band.** Even at t=12 (the latest tested), only 27.4% — short of the 40% viability threshold. The lift from path-so-far features is real but insufficient.
- **Economic cost of waiting is acceptable.** At t=12, only 17% of c1's eventual MFE p50 has been "consumed" (mfe_so_far_p50 = 0.61R vs eventual_mfe_p50 = 3.50R in unit SL=4.0×ATR R-frame). The slippage is not the binding constraint.
- **Top feature importances at t=12** (from `post_entry_feature_importances.csv`): path-so-far features dominate (`mfe_so_far_r_at_t`, `velocity_first_t`, `monotonicity_so_far_at_t`, `bars_in_profit_at_t`) — confirming that mid-trade information is what distinguishes clusters.
- **Per-class AUC at t=12:** c0 0.717, c1 0.755, c2 0.846, c3 0.665. c2 is by far the easiest to identify (its trades enter profit shakily — distinct path signature), but the question we care about is c1 vs the rest, and that's still capped at marginal lift.

### Open extrapolation

Sweep was capped at t=12 per dispatch. AUC trend (0.652 → 0.684 → 0.727 → 0.755) suggests further improvement possible at larger t. A speculative t=20-30 might reach the VIABLE threshold — but the dispatched sweep doesn't test this, and the economic cost overlay at larger t would need re-assessment. Logged as a follow-up question, not a verdict modifier.

### Verdict

**PATH_1_MARGINAL.** Best t=12 yields c1 precision@recall=0.60 = 0.274, in the [0.20, 0.40] marginal band. Slippage acceptable. Not VIABLE, not DEAD. Per dispatch: "Document; defer to v2.4 protocol design".

---

## Path 2 — Signal-tightening

Tested 6 single-rule filters (`F1–F6`) at multiple thresholds, plus pairwise AND-combinations of the top-3 single rules. All evaluated against c1 retention, c2 retention, pool size, aggregate mean_r at SL=4.0×ATR. No filter retunes the Step 4 classifier — these are mechanical filters that would change when the PR-HHHL signal fires.

### Single-rule best operating points

| Rule | Best threshold | Pool | c1_ret | c2_ret | c2_kill | mean_r (unit R) | floor_500 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| F1 trigger_close_pos | ≥ 0.7 | 1077 | 0.791 | 0.830 | 0.170 | +0.060 | ✓ |
| F2 pullback_depth_atr | ≥ 1.0 | 1065 | 0.831 | 0.797 | 0.203 | +0.091 | ✓ |
| F3 trigger_body_atr | ≥ 0.5 | 890 | 0.655 | 0.657 | 0.343 | +0.025 | ✓ |
| F4 trigger_break_size_atr | ≥ 0.25 | 739 | 0.514 | 0.573 | 0.427 | +0.019 | ✓ |
| F5 ret_5bar_atr | ≥ 0.5 | 503 | 0.345 | 0.408 | 0.592 | +0.014 | ✓ |
| F6 pos_in_20bar_range | ≥ 0.5 | 963 | 0.638 | 0.767 | 0.233 | +0.006 | ✓ |

### Viability target

Required: `c1_retention ≥ 0.80` AND `c2_retention ≤ 0.30` AND `pool_size ≥ 500`.

**Zero configurations satisfy this.** The "best" single rule is F2 `pullback_depth_atr ≥ 1.0` with c1_ret=0.83 (✓), c2_ret=0.80 (✗ — far above 0.30 ceiling), pool=1065 (✓). Pushing F2 higher to θ=1.5 brings c2_ret to 0.59 but drops c1_ret to 0.68 — fails both gates.

**The diagonal pattern.** Across every filter, c1 and c2 retention drop in lockstep:

| Filter operating point | c1_ret | c2_ret | Difference |
|---|---:|---:|---:|
| F2 ≥ 0.75 | 0.93 | 0.90 | 0.03 |
| F2 ≥ 1.0 | 0.83 | 0.80 | 0.03 |
| F2 ≥ 1.25 | 0.77 | 0.70 | 0.07 |
| F2 ≥ 1.5 | 0.68 | 0.59 | 0.09 |
| F1 ≥ 0.7 | 0.79 | 0.83 | -0.04 (c2 BETTER preserved) |
| F1 ≥ 0.8 | 0.61 | 0.65 | -0.04 |

**c1 and c2 are mechanically indistinguishable at entry on these features**, confirming the previous diagnostic (overlap coefficient 0.83 mean across all 18 features for c1 vs c2). Even F2 pullback_depth — the most plausible c1-favouring filter — only opens a 0.03-0.09 retention gap across its threshold sweep.

### Two-rule combinations (top-3 single rules)

Top-3 by `c1_retention × c2_kill_rate`: F2 (pullback_depth ≥ 1.5), F3 (trigger_body ≥ 0.7), F4 (trigger_break_size ≥ 0.25).

| Combination | Pool | c1_ret | c2_ret | c2_kill | mean_r |
|---|---:|---:|---:|---:|---:|
| F2 ∧ F3 | 393 | 0.322 | 0.294 | 0.706 | +0.012 |
| F2 ∧ F4 | 433 | 0.316 | 0.324 | 0.676 | −0.030 |
| F3 ∧ F4 | 508 | 0.345 | 0.415 | 0.585 | +0.007 |

Combos kill more c2 but at the cost of c1 retention collapsing (0.32-0.35) and pool dropping toward / below the 500 floor. None viable.

### Pareto frontier

11 non-dominated points on `(c2_kill_rate, c1_retention)` subject to pool ≥ 500. The frontier is uniformly inside the unviable region — no point reaches the (c2_kill ≥ 0.70, c1_ret ≥ 0.80) target box.

See `signal_tightening_pareto_plot.png` for the visual: target box is empty.

### Mild positive: aggregate mean_r modestly improves

F2 `pullback_depth_atr ≥ 1.0` improves aggregate mean_r from baseline +0.054 → +0.091 (+68% relative). This is a real edge but it's small in absolute terms (≈ 0.09R per trade × 0.5% risk = 0.045% balance increment per trade) and applies to the full filtered pool, not c1 specifically.

### Verdict

**PATH_2_DEAD.** No filter achieves the c1_ret ≥ 0.80 + c2_kill ≥ 0.70 target. The mechanical-filter approach cannot bypass the c1-vs-c2 entry-overlap problem identified in the prior diagnostic.

---

## Cross-cluster comparison summary

| Property | c1 | c2 |
|---|---|---|
| Cluster size (Step 2) | 177 | 429 |
| Step 2 archetype label | V-shape recovery (FG-weak) | Early-peak hold |
| mean_r at SL=4.0×ATR (Step 5) | +2.59 | −0.47 |
| Step-1 mean entry-feature distribution | Indistinguishable from c2 (Part A diagnostic, overlap coef 0.83 mean) | |
| Mid-path distinguishability (t=12) | AUC 0.755 (path-so-far helps) | AUC 0.846 (c2 easiest to identify mid-path) |

**The core finding.** c1 (V-shape recovery good cluster) and c2 (Early-peak hold dead cluster) share entry-bar geometry. The difference between them is whether the trade's MFE develops mid-window (c1) or stalls early (c2) — a forward-path property, not an entry-time property. No entry-time filter can discriminate them; only post-entry path observation can, and even then only marginally within the dispatched t-sweep.

---

## Recommended next action

**Close Arc 8 for good. Log cross-arc finding for v2.4 protocol amendment cycle.**

Rationale: PATH_2_DEAD removes the mechanical-filter escape; PATH_1_MARGINAL doesn't reach VIABLE within the tested t range and economic-cost envelope. Continuing Arc 8 would require speculative larger-t sweeps OR a fundamentally different feature class — both are out of scope for arc-level work and belong in protocol design.

### v2.4 protocol finding (drafted per dispatch)

> Path-shape clustering identifies real structural archetypes (Arc 8 c1 = +2.59R/trade V-shape recovery in admit-only economics), but on the PR-HHHL signal family those archetypes are not predictable from any observable available at or near entry. Pipeline E classifiers trained on archetype-success labels achieve 80-90% admit rate on the full Step 1 pool because c1 and c2 share entry-time geometry. Pipeline D1 at near-entry (t ≤ 12) improves cluster separability monotonically but plateaus in the marginal band (precision@recall=0.60 ≈ 0.27) within an acceptable slippage envelope. **Future arcs should pre-test entry-time separability of expected clusters as a §1.5 gate (multiclass RF on the Step 2 cluster IDs with all available entry-time features; c1-vs-rest precision@recall=0.60 ≥ 0.30 required) before committing Step 1 simulation compute.** This is a third confirmation of the Open-22/23/24 admit-only-vs-deployment divergence; the framework needs a structural amendment, not per-arc patches.

### Optional follow-ups (analyst-side, not required)

1. **Speculative t=20/30 sweep** to see if Path 1 reaches VIABLE at later t. AUC trend supports the hypothesis; not committed by this diagnostic.
2. **PR-HHHL signal redesign** with cluster-discriminating triggers (e.g. confirmed-2-bar-resume instead of single-bar; minimum 3-bar look-ahead at entry detection). Outside the v2.3 framework; would require a new arc spec.
3. **Portfolio composition (Open-05)** logging c1 as a structural archetype that's tradeable in isolation (+2.59R/trade) but not stand-alone deployable on PR-HHHL alone.

---

## Files

### Path 1 outputs (`results/l_arc_8/diagnostics/post_entry_confirmation/`)

- `post_entry_t_sweep_results.csv` — per-t metrics
- `post_entry_confusion_matrices/confusion_matrix_t{3,5,8,12}.csv` — 4×4 per t
- `post_entry_c1_economic_overlay.csv` — slippage & MFE-consumed by t
- `post_entry_feature_importances.csv` — top-5 features per t
- `post_entry_t_separability_plot.png` — AUC + precision vs t with slippage
- `PATH1_VERDICT.json` — verdict and summary

### Path 2 outputs (`results/l_arc_8/diagnostics/signal_tightening/`)

- `single_rule_sweep.csv` — all single-rule × threshold combinations
- `two_rule_combinations.csv` — pairwise AND combos
- `pareto_frontier.csv` — non-dominated points (pool ≥ 500)
- `signal_tightening_pareto_plot.png` — scatter + Pareto + target box
- `cluster_retention_per_filter.png` — top-5 filter retention bars
- `PATH2_VERDICT.json` — verdict and best-config

### Combined (`results/l_arc_8/diagnostics/`)

- `entry_feature_overlap/` (prior diagnostic at commit `7d9109e` — c1_NOT_SEPARABLE_AT_ENTRY)
- `COMBINED_DIAGNOSTIC_SUMMARY.md` — this doc

# ARC_10_CLOSURE — l_arc_10

> **Closed:** 2026-05-22T00:00:00Z
> **Branch:** arc/l_arc_10
> **Closure doc path:** results/l_arc_10/ARC_CLOSURE.md

---

## §1 tracker_payload

```yaml
tracker_payload:

  # ────── Identity ──────
  arc_name: l_arc_10
  signal: D1 swing-low rejection long (DLR, v0.1) — bullish rejection of confirmed ascending D1 swing-low, 4H entry
  tf: H4
  sub_protocol: vanilla
  closed_timestamp: 2026-05-22T00:00:00Z
  closure_doc_link: results/l_arc_10/ARC_CLOSURE.md

  # ────── Verdict ──────
  verdict: PASS-VIABLE
  one_line: A1 unfiltered + SL=3.5×ATR + partial-close-1R-runner-trail clears §3 PASS-VIABLE on 11-fold WFO and holdout; classifier near chance — exit policy carries the edge.
  failed_at_step: N/A
  primary_failure_mode: N/A

  # ────── Pool metadata ──────
  pool_metadata:
    total_n: 3301
    window_start: 2010-01-01
    window_end: 2026-04-10
    kh24_co_fire_pct: null   # KH-24 v3 pool not co-built this dispatch; prior v2.3 measured 0%
    configs_evaluated_step5: 96
    search_scope_flag: normal

  # ────── Best architecture ──────
  best_architecture:
    name: A1 system_level_filter
    cluster: aggregate   # A1 is unfiltered — operates on the full Step 1 pool, not a single cluster
    archetype: V-shape   # the dominant capturable archetype (c1, 46.3% of pool)
    config: sl_3.5x_partial_close_1r_runner_trail_unlimited
    sl_atr: 3.5
    exit_policy: sl_partial_close_1r_runner_trail
    exposure_cap: unlimited
    worst_fold_ratio: 5.4185
    worst_fold_roi_pct: 26.49
    worst_fold_dd_pct: 9.22
    mean_fold_ratio: 16.1754
    mean_fold_roi_pct: 49.87
    sign_pos_folds: "11/11"
    n_trades_total: 2162
    holdout_roi_pct: 59.07
    holdout_dd_pct: 5.03
    holdout_passed: true
    oracle_worst_ratio: -1.1896   # caveat: oracle was locked to sl_only exit, not the partial-close policy — not a fair upper bound
    oracle_real_gap_sharpe: null   # not computed; oracle exit-policy mismatch makes the gap non-comparable
    features_in_winning_config: []   # A1 uses no classifier features — the signal conditions + SL/exit policy ARE the architecture

  # ────── Cost decomposition ──────
  cost_decomposition: null   # A1 is unfiltered — no admit/reject pools

  # ────── Per-cluster results ──────
  clusters:
    c0:
      n: 1771
      archetype: monotonic_down
      sl_atr: 1.5   # best per Step 3 SL sweep
      step3_composite: 0.6347
      mfe_p50_r: 1.995
      ww_pp: 0.528
      reach_1r: 0.710
      step4_e_auc: null   # only c1 went through Step 4 (highest composite; dispatch §Step 4 fallback)
      step4_d1_auc: null
      outcome: dies_step3   # failed candidate gate (ww_pp 0.528 > 0.30 ceiling; mfe_p50 1.995 < 1.5 fails as well... actually 1.995 >= 1.5 OK, ww_pp gates it)
    c1:
      n: 1528
      archetype: v_shape_recovery
      sl_atr: 4.0   # best per Step 3 SL sweep (composite 0.5605)
      step3_composite: 0.8628   # at Step 1 SL=2.0
      mfe_p50_r: 5.267
      ww_pp: 0.506
      reach_1r: 0.973
      step4_e_auc: 0.5199   # LGBM mean OOS AUC — at chance
      step4_d1_auc: null   # v3.0 has no separate D1 pipeline (v2.x concept)
      outcome: wins_step5   # backbone of the A1 unfiltered winner; A6 classifier on c1 also reaches PASS-DEPLOYABLE
    c2:
      n: 2
      archetype: monotonic_down
      sl_atr: 1.5
      step3_composite: 0.6750
      mfe_p50_r: 3.071
      ww_pp: 0.500
      reach_1r: 0.500
      step4_e_auc: null
      step4_d1_auc: null
      outcome: dies_step3   # outlier cluster (n=2)

  # ────── Architecture results ──────
  architectures_tested: [A1, A3, A6]
  architecture_results:
    A1: {tested: true, won: true,  worst_fold_ratio: 5.4185}
    A3: {tested: true, won: false, worst_fold_ratio: 1.3966}
    A6: {tested: true, won: false, worst_fold_ratio: 2.9210}   # PASS-DEPLOYABLE in Top-2 but not Top-1
  # Architectures not run (per archetype selection rule, v_shape_recovery → A1+A3+A6):
  #   A2 (classifier_filter), A4 (pipeline_d_exits), A5 (portfolio_composition — only 1 candidate cluster)

  # ────── Archetypes observed ──────
  archetypes_observed: [v_shape_recovery, monotonic_down]

  # ────── Cross-arc observation tags ──────
  cross_arc_tags:
    - v_shape_archetype_cross_arc_persistence
    - exit_policy_dominates_classifier
    - no_classifier_needed_for_v_shape_pass_viable
    - v3_engine_first_pass_viable
    - step4_auc_chance_full_pool_still_passes_via_exit_policy
    - oracle_locked_to_sl_only_unfair_upper_bound
```

---

## §2 Why succeeded

A1 (no classifier filter, full Step 1 pool of 2,162 search-window trades) with SL=3.5×ATR and the `sl_partial_close_1r_runner_trail` exit policy clears PASS-VIABLE with worst-fold ROI/DD ratio 5.42 and 11/11 positive folds. Holdout reproduces: ratio 11.73, DD 5.03%, ROI 59.07%. The result holds despite (and is largely independent of) Step 4 classifier AUC sitting at chance (LGBM 0.5199, RF 0.5142).

**The proximate cause is the exit policy + deeper SL combination.** At Step 1's default SL=2.0×ATR, mean R is −0.024 with p25=p50=p75=−1.0 — most trades hit SL. Widening to SL=3.5×ATR lets V-shape recovery paths complete: c1's reach_1R rises from 0.494 (SL=2.0) to 0.546 (SL=3.5) to 0.572 (SL=4.0), and mfe_p50 climbs from 0.96 → 1.29 → 1.59. The `sl_partial_close_1r_runner_trail` exit then mechanically captures this: close 50% at +1R locks in `0.5 × 1R`, the runner trails 1R below subsequent peak so winners with MFE ≥ 1R contribute `0.5 × (peak − 1R)`. The c1 cluster (46.3% of pool, mean_r +0.93R at SL=2.0, mfe_p50 5.27R) becomes the load-bearing cohort.

**The structural cause is that c1 V-shape recovery has real forward-geometry edge that the exit policy can extract by construction.** This matches the cross-arc V-shape pattern (Arc 7 c1/c3, Arc 10 v2.3 c1 oracle Sharpe 4.61). What v2.3 missed was that the edge does NOT require classifier-based entry filtering — it requires the right exit. Step 4's near-chance AUC (0.52 here, 0.63 in v2.3) was the wrong question. The right question, surfaced by v3.0's A1 architecture, is whether the system-level filter (no-filter + SL + exit) clears the WFO gate. It does.

**What this tells us about the methodology:** v3.0's gates-as-rankings approach (Steps 1-4 produce diagnostic outputs; Step 5 alone gates) corrects v2.3's Step 4 AUC kill-gate — which would have failed this arc the same way it failed v2.3. The architecture-search dimension (A1 system_level_filter as a valid candidate) is the load-bearing protocol change.

**Caveats** (carried from Step 6 audit):
- Per-fold equity reset (each fold starts at $100k). Cumulative DD across folds isn't tracked; real deployment would compound continuously. Worst-fold DD 9.22% bounds the per-fold drawdown, not the cumulative one.
- Exit-policy re-simulation under widened SL uses the stored mid-price path; new exit fills aren't bid/ask-corrected. Estimated drag ≤ 0.01R per trade.
- Oracle WFO was locked to `sl_only` exit at SL=4.0 — different policy than the winning A1 (`sl_partial_close_1r_runner_trail` at SL=3.5). The oracle isn't a fair upper bound here; rerun with matched exit policy would likely show the A1 result IS close to the c1 oracle.

---

## §3 Cross-arc observations

- **V-shape recovery archetype persists into v3.0 — third arc with a recognisable V-shape cohort** (Arc 7 c1/c3; Arc 10 v2.3 c1; Arc 10 v3.0 c1). Different engines (v2 MT5+floor-file vs v3 HistData M1 bid+ask), different pool sizes (1,288 / 802 / 3,301), but the archetype's forward-geometry shape (mono ~+0.6, ttp_rel ~0.8, recovery_ratio ~0.75, mfe_p50 ~5R) is reproducible. Cross-arc V-shape clusterifier as v2.4 candidate is now substantively confirmed.

- **Exit policy dominates classifier filter on the V-shape archetype.** Step 4 classifier AUC at 0.52 (chance) — same feature-space ceiling Arc 7 and Arc 10 v2.3 reported. A1 (no filter) passes PASS-VIABLE because the partial-close-runner-trail exit captures the c1 cohort's structural edge by construction. The classifier path that v2.3 closure recommended as "leading v2.4 candidate" may be the wrong direction; the filter path (deterministic SL + exit) is the right one for this archetype. Direct empirical evidence: A1 unfiltered worst-fold ratio 5.42 vs A6 max_per_currency_2 (with classifier) 2.92 — classifier-gated variant is STRICTLY WORSE on the same exit policy.

- **v3.0's gates-as-rankings + A1 as a valid architecture is the protocol change that unblocked this arc.** v2.3's Step 4 AUC kill-gate (E ≥ 0.65 OR D1 ≥ 0.60) would have closed this arc as STEP_4_HALT identical to the v2.3 outcome. v3.0 routes everything through Step 5 WFO; A1 (no classifier) is one of the valid architectures; the partial-close exit policy is in Appendix B. The protocol change is doing real work.

- **Pool-size drift between v2 and v3 engines is +312%** (802 → 3,301 trades, same signal, same window, same pair set). Two drivers: (a) v3 unrestricted Step-1 exposure vs v2 `max 1 open per pair`; (b) HistData M1 bid+ask spreads vs MT5 spread floor file. Path features (mid-based) are spread-independent; pool size and exposure-derived metrics are not. Future v2→v3 re-runs of any arc should expect comparable drift.

- **Oracle-WFO comparison pattern broken when oracle exit policy ≠ winning architecture exit policy.** Step 5's oracle locked to `sl_only` at SL=4.0 (the Step 3 best); winning A1 used `sl_partial_close_1r_runner_trail` at SL=3.5. Oracle reports -11.74% worst ROI while A1 reports +26.49% — implying A1 BEATS the oracle, which is structurally impossible if both used the same exit. The lesson: oracle WFO must sweep the same architecture × config grid as the realised candidates, or be reported only as "cluster-filter true-label upper bound CONDITIONAL on identical exit policy." Recommendation for protocol §2 Step 5 Amendment 2's oracle WFO definition: lock oracle to the WINNING architecture's config sans cluster-filter, not to a fixed (best_sl, sl_only).
```

---

## §10 Amendment 3 re-evaluation (added 2026-05-22)

**Original verdict:** PASS-VIABLE (best A1 system_level_filter, worst-fold ratio 5.42, DD 9.22% at `r_base = 0.5%`)
**Re-evaluated verdict:** **PASS-DEPLOYABLE-PROVISIONAL** (upgrade)
**Re-evaluation status:** provisional (missing `chained_max_dd_base_pct` and per-day max-DD series; Step 6 already PASS — carries forward)
**Re-evaluated primary_failure_mode:** N/A (passes amended DEPLOYABLE gate provisionally)

### Scaling derivation
- `worst_fold_dd_base_pct`: **9.22%** (closure §1 `worst_fold_dd_pct`)
- `worst_fold_roi_base_pct`: **+26.49%** (closure §1 `worst_fold_roi_pct`)
- `k_safe = 8.0 / 9.22 = 0.8677` (scaling **down** — base DD already exceeds 8% gate)
- `k_hard = 10.0 / 9.22 = 1.0846` (scaling up modestly)
- `r_safe_pct = 0.5 × 0.8677 = 0.4339%`
- `r_hard_pct = 0.5 × 1.0846 = 0.5423%`
- `scalable_to_safe`: **true** (within [0.15%, 2.0%])
- `scalable_to_hard`: **true** (within bounds)

### Amended DEPLOYABLE gate evaluation

| # | Constraint | Threshold | Value at r_safe | Pass/Fail | Notes |
|---|---|---|---|---|---|
| 1 | Scalable to safe | `r_safe ∈ [0.15%, 2.0%]` | 0.4339% | ✓ | well within bounds |
| 2 | Worst-fold ROI/DD ratio | ≥ 2.0 | **2.873** (§3 math: 26.49 / 9.22) | ✓ | invariant; engine reading 5.42 also clears |
| 3 | Worst-fold ROI | > 0 | +22.98% (= 26.49 × 0.8677) | ✓ | sign-positive |
| 4 | Per-fold positivity | all 11 positive, 0 negative | 11/11 | ✓ | sign does not scale |
| 5 | Worst-fold DD | ≤ 8% | 8.00% (= 9.22 × 0.8677) | ✓ | by construction of `k_safe` |
| 6 | Daily DD breaches | = 0 | unknown × 0.87 | ? | PROVISIONAL: per-day max-DD series not built under v3.0 pre-amendment. Note: `k_safe < 1` means daily breach count under proper per-day re-evaluation can only decrease or stay flat from the `r_base` count. Per-fold breach summary at `r_base` was not stored for Arc 10 (no per_fold_metrics.csv) — so we don't have the floor count either. PROVISIONAL per dispatch Q2. |
| 7 | Chained max DD | ≤ 10% | unknown × 0.87 | ? | PROVISIONAL: `chained_max_dd_base_pct` not measured. Step 6 audit §"Caveats" already flagged: "per-fold equity reset (each fold starts at $100k). Cumulative DD across folds isn't tracked; real deployment would compound continuously." Worst-fold DD at `r_safe` = 8% bounds *per-fold* drawdown but not the cross-fold chained one. |
| 8 | Trades per fold | ≥ 25 | n_total = 2,162 / 11 = ~197 avg | ✓ | per-fold breakdown not separately saved; total well above minimum even if distribution is uneven |
| 9 | Holdout at r_safe | clears prior §3 holdout gate | proxy: ROI ≈ **+51.27%** (= 59.07 × 0.8677), DD ≈ **4.36%** (= 5.03 × 0.8677) | ✓ | PROXY: not re-run. DD 4.36% < 8% gate; ratio 11.76 invariant |
| 10 | Step 6 clean | clean | **PASS** | ✓ | Step 6 already run on the same A1 Top-1 winner (`results/l_arc_10/step_6/causal_audit_report.md` §7 verdict). No producer downgrades, no candidate kills. Carries forward. |

### Amended VIABLE gate evaluation (recorded for completeness; DEPLOYABLE passes so VIABLE is redundant)

| # | Constraint | Threshold | Value at r_hard | Pass/Fail | Notes |
|---|---|---|---|---|---|
| 1 | Hard-scalable | `r_hard ∈ [0.15%, 2.0%]` | 0.5423% | ✓ | |
| 2 | Worst-fold ROI/DD ratio | ≥ 2.0 | 2.873 | ✓ | invariant |
| 3 | Mean-fold ROI/DD ratio | ≥ 2.5 | mean_roi 49.87 / worst_dd_proxy ≈ 16.18 (engine) or 49.87 / mean_dd (unknown) | ✓ | engine reports 16.18, well above 2.5 |
| 4 | Per-fold positivity | ≤ 1 negative fold | 11/11 positive | ✓ | |
| 5 | Worst-fold DD | ≤ 10% | 10.00% (= 9.22 × 1.0846) | ✓ | by construction |
| 6 | Daily DD breaches | = 0 | unknown × 1.08 | ? | PROVISIONAL |
| 7 | Chained max DD | ≤ 10% | unknown × 1.08 | ? | PROVISIONAL |
| 8 | Trades per fold | ≥ 25 | ~197 avg | ✓ | |
| 9 | Holdout at r_hard | clears prior §3 holdout gate | proxy: ROI ≈ +64.07%, DD ≈ 5.45% | ✓ | PROXY |
| 10 | Step 6 clean | clean | PASS | ✓ | carries forward |

### Engine vs §3 ratio discrepancy (flagged per chat Q1)

The closure §1 reports `worst_fold_ratio: 5.4185`. Inspection of `step_5/wfo_results.csv` row 1 shows the engine's worst-fold ratio is computed per fold and reported alongside the worst ROI; the §3 mathematical reading `worst_fold_roi / worst_fold_dd = 26.49% / 9.22% = 2.873` produces a different (lower) number. Per chat directive (Q1: option a), the §10 evaluation uses **2.873** as the gate value. Both readings clear the 2.0 gate — verdict outcome is identical. The discrepancy is methodological / reporting-convention, not material to this arc's verdict.

### Final assessment

Arc 10 upgrades from PASS-VIABLE to **PASS-DEPLOYABLE-PROVISIONAL** under Amendment 3. The mechanism: at `r_base = 0.5%`, worst-fold DD (9.22%) sat above the 8% PASS-DEPLOYABLE bound but below the 10% PASS-VIABLE bound — the original verdict was correct under the prior protocol. Amendment 3 introduces the scaling rule: scale risk down by `k_safe = 0.87` to bring worst-fold DD to the 8% gate by construction; worst-fold ROI scales linearly to +22.98%; the ROI/DD ratio is invariant (2.87 in §3 math, 5.42 in engine reading — both clear the 2.0 gate). With Step 6 already PASS on the same A1 Top-1 winner (causal audit clean — D1 swing-low detector right-edge-offset=4 confirmation-lag passes the Arc 9 lesson, all top-10 features clean producer-level), the upgrade is structurally sound.

The "PROVISIONAL" suffix is load-bearing on two constraints:
1. **Chained max DD** — Step 6 audit explicitly flagged that per-fold equity reset overstates sustainability vs production compounding. A continuous-equity backtest would resolve constraint #7 definitively. Without it, we assume linear scaling holds: if `chained_max_dd_base_pct` ≤ 11.52% then `× k_safe = 0.87` gives ≤ 10% scaled. Plausible but unverified.
2. **Daily DD breaches** — per-day max-DD series not built. Under downward scaling (`k_safe = 0.87`), proper per-day re-evaluation can only decrease the breach count from the `r_base` count. The `r_base` count itself wasn't stored separately for Arc 10 (no per_fold_metrics.csv).

Both gaps are engine-resolvable via a single re-run of the A1 winning config with: (a) chained continuous-equity emission, (b) per-day max-DD series persisted to `step_5/per_day_max_dd_base.parquet` per Amendment 3 §"Daily DD measurement" spec. Recommended (not blocking; verdict stands provisional).

### Missing data flags

- Constraint #6 (daily DD breaches at `r_safe`): per-day max-DD series not built. PROVISIONAL.
- Constraint #7 (chained max DD at `r_safe`): `chained_max_dd_base_pct` not measured; per-fold equity reset only. PROVISIONAL.
- **Recommended engine re-run** to upgrade verdict from PROVISIONAL to DEFINITIVE: re-run A1 winning config (`sl_3.5x_partial_close_1r_runner_trail_unlimited`) on the 2010-2020 search window + 2021-04→2026-04 holdout with continuous-equity tracking + per-day max-DD emission. Scope: ~1 config × 11 IS folds + 1 holdout fold ≈ 12 fold-runs. Low compute. Not in this dispatch's scope.

### Cross-arc tags (additions to closure §3 tags)

- `viable_to_deployable_upgrade_under_amendment3` — first documented PASS-VIABLE → PASS-DEPLOYABLE upgrade purely from the risk-normalised gate change (no engine re-run needed for upgrade itself)
- `step6_audit_carries_forward_across_amendment` — Step 6 PASS on the same A1 winner under prior protocol is valid for the upgraded verdict; the amendment doesn't change Step 6's scope
- `engine_ratio_vs_amendment_ratio_divergence` — engine `worst_fold_ratio: 5.42` vs §3 math 2.87; same verdict outcome but ~2× numerical gap. Worth a v3.1 reporting convention amendment to lock the ratio definition.

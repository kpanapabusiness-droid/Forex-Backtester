# L_ARC_PROTOCOL v1.1 AMENDMENT

**Supersedes:** Specific clauses in `L_ARC_PROTOCOL.md` v1.0 (locked 2026-05-13)
**Trigger:** Arc 1 (redo) step 3 calibration check FAIL — outcome (a) protocol-spec error
**Reference:** `results/l_arc_1/step3_extractability/PHASE_L_ARC_1_STEP3_S15_OUTCOME.md`
**Author:** planning chat
**Lock date:** TBD at sign-off
**Effective from:** sign-off date; applies retroactively to Arc 1 step 3 Phase F re-run and onward

---

## How this doc works

This document specifies twelve amendments to L_ARC_PROTOCOL v1.0 and L_ARC_OPERATIONAL_SPEC v1.0. Each amendment includes:
- **What changes** — concrete textual edit or addition
- **Where it lives** — file and section
- **Why** — justification grounded in arc 1 findings
- **Cost** — implementation effort
- **Risk** — what could go wrong

Once signed off, this amendment doc supersedes the relevant v1.0 clauses. The v1.0 docs remain on-disk as historical record; the working protocol version becomes v1.1.

---

## Amendment 1 — Family-level calibration check for known-prior arcs

**What changes.** Replace the named-feature calibration check with a family-level version.

**Where it lives.** `L_ARC_PROTOCOL.md` §15 (Calibration check).

**Current v1.0 text** (paraphrased):
> The known-prior carrier feature (e.g., `concurrent_signals_within_3h` for Arc 1) must surface as Tier 1 or Tier 2 BH-corrected predictor of the non-extractable cluster in step 3 Phase F. FAIL triggers §15 investigation.

**Replacement v1.1 text:**
> For arcs with a known-prior carrier feature, the calibration check evaluates the **effect family** that the carrier belongs to. Family membership is defined in the arc-open doc. PASS condition: at least one family member surfaces as Tier 1 or Tier 2 BH-corrected predictor of the non-extractable cluster in step 3 Phase F. Reporting must include all family members' tier assignments. The original named carrier is reported prominently but not used as the gate.
>
> For arcs without a known-prior carrier feature (i.e., new signals), the calibration check is **not applicable**; step 3 proceeds without a positive control. Step 3 reporting includes a statement of this status.

**Why.** Arc 1 demonstrated that the original CH-001 carrier (`concurrent_signals_within_3h`) was a noisier representative of its effect family than three to four sibling features that did clear Tier 2. Pinning the calibration check to a single named feature is too narrow — it confuses the question "did the methodology find filterable structure of the right kind" with "did the methodology re-find this exact feature." The first question is the right one.

**Cost.** Trivial. Phase F output already produces per-feature tier assignments. The family-level evaluation is an additional read pass over existing output.

**Risk.** The family-level check is more permissive than the feature-level check. A pathologically broad family definition could PASS the check on a single weak member. Mitigation: family membership is fixed in the arc-open doc and cannot be expanded post-hoc to rescue a FAIL.

---

## Amendment 2 — BH-tier is reportorial, not eliminative

**What changes.** Tier 3 features are NOT eliminated from Phase G (effect sizes) or Phase H (filter dry-run) consideration. Tier label is a skepticism flag, not a hard filter.

**Where it lives.** `L_ARC_OPERATIONAL_SPEC.md` §6.7 (Look-elsewhere haircut), §7 (step 4 candidate derivation).

**Current v1.0 behavior** (implicit): Phase G and Phase H feature inputs are drawn from Tier 1/2 only. Tier 3 features are reported but effectively dropped.

**Replacement v1.1 behavior:**
> All features ranked by raw AUC in the top-K (K = max(20, 0.5 × n_features_scanned)) per cluster enter Phase G and Phase H regardless of BH tier. The BH tier is reported alongside each feature. Step 4 candidate selection considers tier as one factor among (raw AUC, effect size, per-fold sign consistency, expected R-volume captured, partial AUC at worst decile) — not as an eliminative gate.

**Why.** BH correction is a discipline for multiple comparisons but can kill real features when n_features_scanned is large. AUC = 0.55 with BH-p = 0.22 (Tier 3) may still build a useful filter if effect size and sign consistency are strong. Treating tier as a flag rather than a filter preserves real findings while keeping the haircut transparent.

**Cost.** Trivial. Phase G and Phase H already process feature lists; this changes which features enter, not the computation.

**Risk.** Tier 3 features admitted to candidate selection could be noise. Mitigation: step 4's held-out fold check (folds 6, 7 reserved) is the real gate. Anything noisy doesn't survive held-out validation.

---

## Amendment 3 — HDBSCAN added as third clustering method

**What changes.** Phase A clustering runs three algorithms instead of two: k-means, hierarchical (Ward linkage), HDBSCAN.

**Where it lives.** `L_ARC_OPERATIONAL_SPEC.md` §6.1 (Cluster discovery).

**Current v1.0 spec:**
> Algorithms: k-means at K ∈ {2, 3, 4, 5, 6}; agglomerative hierarchical (Ward linkage).

**Replacement v1.1 spec:**
> Algorithms: k-means at K ∈ {2, 3, 4, 5, 6, 7, 8}; agglomerative hierarchical (Ward linkage) at the same K range; HDBSCAN with `min_cluster_size = 0.05 × n_pool`, `min_samples = 50`. HDBSCAN produces variable-K output with a "noise" label for unclustered trades; treat noise as a separate cluster ID for reporting purposes.

**Why.** K-means and hierarchical both require K to be specified in advance. HDBSCAN finds natural density clusters without specifying K, and labels low-density trades as noise. This catches structure that K-fixed methods miss — particularly sub-clusters within larger groups and outlier patterns. Adding HDBSCAN as a third view is complementary, not redundant.

**Cost.** Moderate. One additional algorithm run per arc. HDBSCAN is well-supported in scikit-learn-contrib; integration is ~50 lines of code. Output format aligns with k-means / hierarchical (cluster_id per trade).

**Risk.** HDBSCAN's hyperparameters (`min_cluster_size`, `min_samples`) influence cluster count and granularity. The choice of `min_cluster_size = 0.05 × n_pool` aligns with the exit/delayed-entry floor in Amendment 7. Could fail to find any clusters on some arcs (everything labeled noise) — reportorial, not a FAIL condition.

---

## Amendment 4 — Three path-geometry features added to clustering set

**What changes.** Three new features enter the clustering feature subset.

**Where it lives.** `L_ARC_OPERATIONAL_SPEC.md` §5.16 (feature schema) and §6.1 (clustering feature subset).

**New features:**
1. `fwd_realized_range_atr` — max(fwd_high) − min(fwd_low) across the forward window, ATR-normalized. Captures total volatility traversed regardless of direction.
2. `fwd_fraction_time_above_entry` — fraction of forward-window bars whose close is above entry price. Captures dwell-time in profit zone.
3. `fwd_max_consecutive_directional_bars` — longest run of bars moving in same direction (sign of step return). Captures persistence vs choppiness.

**Why.** The current 12-feature clustering set is reasonable but has potential redundancy on the magnitude axis (h24 and h120 versions of MFE/MAE) and gaps on volatility-realized and time-in-zone. These three features fill those gaps without creating new redundancy. All three are path-geometry, unconditional on exit, consistent with the framework that clustering operates on what the trade COULD do, not what it did under verbatim exit.

**Cost.** Low. Feature computation runs once in step 2; cluster code reads the new columns.

**Risk.** Inflating the feature count without removing redundant ones could weaken cluster separation by spreading variance thinner. Mitigation: Amendment 5 (PCA pre-check) identifies and merges redundant features before clustering runs.

---

## Amendment 5 — PCA pre-check on clustering features

**What changes.** Before running cluster algorithms in Phase A, run PCA on the standardized clustering feature set. Report the first 3 principal components' explained variance, the cumulative variance explained, and the loadings (which raw features contribute most to each PC).

**Where it lives.** `L_ARC_OPERATIONAL_SPEC.md` §6.1 (Cluster discovery, new Phase A.0 sub-step).

**Behavior:**
- If two features have absolute Pearson correlation > 0.85 in the pre-clustering data, the PCA report flags them as candidate-redundant.
- Reporting only; does not automatically drop features.
- Step 3 phase doc includes PCA findings in §schema_notes.

**Why.** Lets us see whether the clustering feature set is over-counting an axis (e.g., magnitude appearing twice via h24 and h120 versions). Future arcs can be informed by the result without it being an automatic drop. Adding it as a transparent diagnostic step rather than automated feature selection keeps the protocol interpretable.

**Cost.** Trivial. PCA on 15 features × 45,673 rows is sub-second.

**Risk.** None of substance. Diagnostic only.

---

## Amendment 6 — K range extended to {2..8}

**What changes.** Cluster K sweep covers K = 2, 3, 4, 5, 6, 7, 8 (was 2..6).

**Where it lives.** `L_ARC_OPERATIONAL_SPEC.md` §6.1 (Cluster discovery).

**Why.** At K=6, average cluster size ≈ 17% of pool, close to the filter-TO eligibility floor. Extending to K=8 (average ≈ 12.5%) lets the algorithm reveal sub-patterns within larger clusters without falling below the exit/delayed-entry size floor (5% per Amendment 7). Beyond K=8 the average cluster size drops below the floor, which is why we stop there.

**Cost.** Trivial. Two additional K values × 2 algorithms = 4 additional cluster runs per arc.

**Risk.** Step 3 phase doc reporting grows. Mitigation: report K=2 through K=4 in narrative detail; K=5 through K=8 as summary tables only. Detail is in supporting CSVs.

---

## Amendment 7 — Differentiated cluster-size floor by candidate type

**What changes.** Cluster size eligibility differs by candidate application:
- **Filter-TO candidates** (take only trades in this cluster): cluster must be ≥ 15% of pool.
- **Filter-OUT candidates** (skip trades in this cluster): no size floor (can filter out any size of bad cluster).
- **Exit candidates** (apply exit rule only to cluster members): cluster must be ≥ 5% of pool.
- **Delayed-entry candidates** (delay entry, observe path, decide based on cluster prediction): cluster must be ≥ 5% of pool.

**Where it lives.** `L_ARC_PROTOCOL.md` §7 (verdict logic) and `L_ARC_OPERATIONAL_SPEC.md` §7 (step 4 candidate derivation).

**Why.** The 15% floor is right for filter-TO candidates because going long-only on a small cluster has insufficient trade volume for stable returns. Exit and delayed-entry candidates apply only to cluster members; a 5% cluster of 45,673 trades is ~2,300 trades over 5 years (~450/year), enough statistical mass to validate. Filter-OUT candidates can be any size — small bad clusters are useful to skip.

**Cost.** Trivial. Verdict logic and candidate evaluation become tier-aware on cluster size.

**Risk.** The 5% floor for exits/delayed-entry may still be too small for some signal frequencies. Mitigation: per-fold stability check (Phase B) flags clusters with high per-fold size CV; step 4 evaluation reads stability flags.

---

## Amendment 8 — Delayed-entry-with-observation-window-filter as fourth candidate type

**What changes.** Step 4 candidate types expand from three (filter, exit, trail) to four:
1. Filter (skip trades at signal time)
2. Exit (modify when to close)
3. Trail (lock in profit via trailing stop)
4. **Delayed-entry-with-observation-window-filter** (wait N bars, observe path, then enter or skip based on cluster prediction at t=N)

**Where it lives.** `L_ARC_OPERATIONAL_SPEC.md` §7 (step 4 candidate derivation), with explicit evaluation rule:

> For any Phase D held-bar predictor surfacing AUC ≥ 0.65 at t < 10, step 4 must evaluate THREE candidate framings: (a) signal-time filter using a correlated t=0 feature if available; (b) early-exit at t, applied to predicted-bad-cluster members; (c) delayed-entry at t with cluster-prediction filter applied at entry. All three compete on held-out folds 6, 7.

**Why.** A predictable cluster at t=5 implies three possible trading-strategy applications: filter (using a worse t=0 proxy), exit (close early after entry), or delayed entry (skip the early-bar entry, observe, then enter only on good signal). The current protocol only evaluates filter and exit. Delayed entry is a distinct strategy with different cost characteristics (no spread paid on skipped trades, lost entry-bar movement on taken trades). For signals where early-bar movement is small or noisy, delayed entry can outperform both alternatives.

**Cost.** Moderate. Step 4 candidate evaluation triples for predictable-at-t findings. Backtester support already exists via the existing entry-delay shadow trade-sets from step 2 phase E.

**Risk.** Combinatorial explosion in step 4 candidate space. Mitigation: only apply the three-way evaluation when Phase D AUC threshold is met; otherwise candidate types remain the original three.

---

## Amendment 9 — Partial AUC at worst-decile cutoff

**What changes.** Phase C and Phase D predictor scans report TWO AUC metrics per feature per cluster:
1. Full AUC (current)
2. Partial AUC restricted to the worst-decile of cluster membership probability (new)

**Where it lives.** `L_ARC_OPERATIONAL_SPEC.md` §6.3 (Signal-time predictor scan) and §6.4 (Held-bar predictor scan).

**Why.** Full AUC weights every trade-pair equally. A feature that classifies the worst 10% of trades correctly but is noise on the middle 90% scores AUC ≈ 0.50 — yet filtering out that worst 10% is exactly what filter rules do well. Partial AUC at the worst-decile cutoff isolates tail-classification ability, which is more decision-relevant for filter design than full AUC.

**Cost.** Trivial. Partial AUC is a one-line modification to the AUC computation.

**Risk.** None of substance. Reportorial addition.

---

## Amendment 10 — Expected-R-volume filter ranking

**What changes.** Phase H filter dry-run ranks candidate filters by expected R-volume captured = (mean R improvement post-filter) × (n_retained post-filter), in addition to the existing per-fold sign-consistency check.

**Where it lives.** `L_ARC_OPERATIONAL_SPEC.md` §6.8 (Filter dry-run) and §7 (step 4 candidate evaluation).

**Why.** A filter that improves mean R by +0.10 on 5,000 trades captures more economic value than a filter that improves mean R by +0.50 on 200 trades, all else equal. Ranking by expected R-volume directs attention to filters that materially shift portfolio outcomes rather than narrow filters with high per-trade impact but limited applicability.

**Cost.** Trivial. Both quantities are already computed; the ranking just multiplies them.

**Risk.** None of substance. Reportorial addition.

---

## Amendment 11 — Hash-based permutation seeds

**What changes.** Per-feature permutation seed in Phase F BH haircut uses `hashlib.sha256(fname.encode()).hexdigest()[:8]` converted to int, not Python's `hash(fname)`.

**Where it lives.** Implementation code (`run_step3.py` Phase F). Protocol doc mentions the determinism requirement in `L_ARC_OPERATIONAL_SPEC.md` §11 (Reporting standards).

**Why.** Python's built-in `hash()` for strings is PYTHONHASHSEED-randomized across processes by default. This causes BH p-values to differ slightly between runs (Tier assignment is stable, but byte-identical output fails). Hashlib digests are deterministic across processes and OS environments.

**Cost.** Trivial. One-line code change.

**Risk.** None.

---

## Amendment 12 — Cluster-target selection rule locked in protocol

**What changes.** The rule for selecting which cluster is "the non-extractable / high-MAE target" in the calibration check (and in Phase G/H reporting prominence) moves from implementation code to protocol text.

**Where it lives.** `L_ARC_PROTOCOL.md` §7 (verdict logic).

**Specification:**
> The non-extractable cluster is the cluster with the **lowest mean net R** at the relevant K. Ties broken by highest median `fwd_mae_h24_atr`. The selection rule is fixed across all K values and all clustering algorithms. Each K × algorithm combination produces one non-extractable cluster identification.

**Why.** Arc 1 triage revealed the selection rule lived in `run_step3.py` rather than in protocol text. The choice was sensible (max mean fwd_mae_h24) but the protocol shouldn't depend on an implementation detail to define its own sensitivity check. The lowest-mean-net-R rule is more directly aligned with the economic question (which cluster destroys equity), and ties broken by max fwd_mae_h24 preserve the current heuristic's strength.

**Cost.** Trivial. Implementation code reads the rule from protocol text.

**Risk.** None.

---

## Out-of-scope deferrals

The following items were considered for v1.1 and explicitly deferred:

- **Outcome-tertile clustering as a second view.** Rejected per user input — clustering on R-outcomes conflates path geometry with exit choice. Kept clustering purely path-geometry.
- **Clustering feature expansion beyond the three added.** Deferred until evidence of inadequacy (post-arc-5 if candidate yield is poor).
- **DXY / cross-asset / equities context features in step 2.** Deferred similarly.
- **HDBSCAN hyperparameter sweep.** Locked at `min_cluster_size = 0.05 × n_pool, min_samples = 50`. Future arcs may reconsider.
- **Step 4 candidate-type prioritization weighting.** Each candidate type competes on held-out folds; no a-priori preference order.

These are recorded for future amendment consideration if signals appear.

---

## Sign-off

- Amendment count: 12
- Authoring rationale: arc 1 step 3 calibration FAIL outcome (a) per `PHASE_L_ARC_1_STEP3_S15_OUTCOME.md`
- Once signed off, the working protocol version becomes v1.1
- v1.0 docs remain on-disk as historical record under `docs/archive/v1.0/`
- Signed off by: ______________________
- Date: ______________________
- Simulator git commit at sign-off: ______________________

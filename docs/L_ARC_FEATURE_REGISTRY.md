# L_ARC_FEATURE_REGISTRY

**Purpose:** Cross-arc record of which features, filters, and cluster patterns surfaced as useful, useless, or borderline across L arcs. Maintained at each arc closure. Reference for arc-open expectations and arc-N+1 feature-set audits.

**Maintenance rule:** Append entries at each arc closure. Never delete entries. If an entry needs correction, append a correction row with reference to the original.

**Status of this doc:** active. First entries populated at arc 1 closure (2026-05-14).

---

## §1. Predictor features — cross-arc performance

Tracks which signal-time and held-bar predictor features cleared which tiers in step 3 Phase F per arc.

| Arc | Signal | Algorithm | K | Target cluster | Feature | Raw AUC | Partial AUC (worst-decile) | BH tier | Notes |
|---|---|---|---|---|---|---:|---:|---|---|
| 1 (CLOSE) | univariate_extreme/abs_return_top_decile/neg/h=1 | k-means | 2 | cid=1 (high-MAE) | currency_basket_3h_USD | 0.509 | n/a | 1 | Cross-pair family §5.15 |
| 1 (CLOSE) | univariate_extreme/abs_return_top_decile/neg/h=1 | k-means | 2 | cid=1 (high-MAE) | currency_basket_3h_JPY | 0.512 | n/a | 1 | Cross-pair family §5.15 |
| 1 (CLOSE) | univariate_extreme/abs_return_top_decile/neg/h=1 | k-means | 2 | cid=1 (high-MAE) | currency_basket_3h_GBP | 0.484 | n/a | 1 | Cross-pair family §5.15 |
| 1 (CLOSE) | univariate_extreme/abs_return_top_decile/neg/h=1 | k-means | 2 | cid=1 (high-MAE) | concurrent_signals_same_bar | 0.491 | n/a | 1 | Cross-pair family §5.15 |
| 1 (CLOSE) | univariate_extreme/abs_return_top_decile/neg/h=1 | k-means | 2 | cid=1 (high-MAE) | trade_overlap_at_execution_time | 0.505 | n/a | 2 | Cross-pair family §5.15 |
| 1 (CLOSE) | univariate_extreme/abs_return_top_decile/neg/h=1 | hierarchical | 2 | cid=1 (high-MAE) | currency_basket_3h_USD | 0.508 | n/a | 1 | Cross-pair family §5.15 |
| 1 (CLOSE) | univariate_extreme/abs_return_top_decile/neg/h=1 | hierarchical | 2 | cid=1 (high-MAE) | currency_basket_3h_JPY | 0.524 | n/a | 1 | Cross-pair family §5.15 |
| 1 (CLOSE) | univariate_extreme/abs_return_top_decile/neg/h=1 | hierarchical | 2 | cid=1 (high-MAE) | currency_basket_3h_GBP | 0.476 | n/a | 1 | Cross-pair family §5.15 |
| 1 (CLOSE) | univariate_extreme/abs_return_top_decile/neg/h=1 | hierarchical | 2 | cid=1 (high-MAE) | concurrent_signals_same_bar | 0.484 | n/a | 1 | Cross-pair family §5.15 |
| 1 (CLOSE) | univariate_extreme/abs_return_top_decile/neg/h=1 | hierarchical | 2 | cid=1 (high-MAE) | concurrent_signals_within_3h | 0.493 | n/a | 2 | Historical CH-001 carrier — Tier 2 at hierarchical, Tier 3 at k-means |

**Arc 1 cross-arc summary:**
- Cross-pair / portfolio family (§5.15) is the dominant predictor class for the high-MAE cluster
- 5/8 family members cleared Tier 1/2 in both algorithms
- Currency-basket directional features (USD, JPY, GBP) consistently Tier 1
- Historical CH-001 carrier (`concurrent_signals_within_3h`) is a noisier representative of the family

**Query patterns to surface after arcs 1-5:**
- Features at Tier 1/2 in ≥ 3 arcs → reliable predictors, prioritize in feature-set audit
- Features at Tier 3 in all arcs → candidates for removal from feature set
- Features at Tier 1 in some arcs, Tier 3 in others → conditional predictors, document conditioning factor

---

## §2. Filter candidates — cross-arc performance

Tracks filter specs derived in step 4 and their step 6 WFO outcomes. Arc 1 Phase H descriptive candidates included for cross-arc visibility even though arc 1 closed before step 4.

| Arc | Filter spec | Cluster-size disposition | Expected R-volume captured | Per-fold sign consistency | Step 6 WFO outcome | Step 6 worst-fold ROI | Step 6 worst-fold DD | Notes |
|---|---|---|---:|---|---|---:|---:|---|
| 1 (CLOSE) | currency_basket_3h_JPY above p25 | filter-OUT below p25 | +198.9 R-units | 5/7 | n/a (CLOSE at step 3) | n/a | n/a | Best Phase H by E[R-volume] |
| 1 (CLOSE) | session=asia | filter-TO asia session | +157.1 R-units | 5/7 | n/a | n/a | n/a | Phase H candidate |
| 1 (CLOSE) | day_of_week below p25 | filter-OUT high-day | +154.0 R-units | 4/7 | n/a | n/a | n/a | Below sign-consistency floor |
| 1 (CLOSE) | currency_basket_3h_JPY above p75 | filter-TO high-JPY | +142.2 R-units | 5/7 | n/a | n/a | n/a | Phase H candidate |
| 1 (CLOSE) | concurrent_signals_same_bar below p25 | filter-OUT high-density | +127.4 R-units | 3/7 | n/a | n/a | n/a | Below sign-consistency floor |

**Arc 1 filter learnings:**
- Best filter improves mean R by only +0.0087 R, retaining ~50% of pool
- Per-fold sign consistency at best filter is 5/7 (2 negative OOS folds)
- No filter spec clears PASS-VIABLE thresholds in arc 1 step 3 dry-run

**Cross-arc carry:**
- `currency_basket_3h_{USD/JPY/GBP}` directional family marked for explicit testing in arcs 2, 3, 5 step 4 candidate derivation, regardless of arc-specific predictor scan results

---

## §3. Cluster archetypes — cross-arc recurrence

Tracks cluster patterns that recur across arcs at the same K.

| Arc | Algorithm | K | Cluster ID | Archetype label | Size (% of pool) | Mean net R | Median fwd_mfe_h24 | Median fwd_mae_h24 | Race condition p50 | Notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| 1 (CLOSE) | k-means | 2 | 0 | high-MFE/MFE-first | 50.5% | positive | high | low | negative | Mirror archetype |
| 1 (CLOSE) | k-means | 2 | 1 | high-MAE/MAE-first | 49.5% | −0.140 | 1.08 | 3.38 | +18 | Target archetype |
| 1 (CLOSE) | hierarchical | 2 | 0 | high-MFE/MFE-first | 44.2% | positive | high | low | negative | Mirror archetype |
| 1 (CLOSE) | hierarchical | 2 | 1 | high-MAE/MAE-first | 55.8% | −0.099 | 1.35 | 2.91 | +7 | Target archetype; CV=0.79 stability flag |

**Arc 1 archetype summary:**
- K=2 produces a clean directional split at both algorithms (~50/50)
- Path geometry asymmetry: MAE 3× MFE in target, MFE 3× MAE in mirror
- Silhouette ≈ 0.166, ARI ≈ 0.508 — modest but real structure
- HDBSCAN found 0 non-noise clusters → no high-density sub-structure beyond what K=2 captures

**Cross-arc question to monitor:** does the K=2 directional archetype recur in arcs 2-5 at similar proportions? If yes → generic FX 1H/4H feature. If no → arc-1 signal-specific.

---

## §4. Methodology learnings

Free-form section. One paragraph per arc capturing methodology lessons that didn't make it into the protocol spec but are worth remembering.

### Arc 1 (redo) — CLOSE 2026-05-14

The univariate_extreme / abs_return_top_decile / neg / h=1 signal carries small detectable edge (L4 Sharpe 0.037) that does not translate to a deployable filter under realistic execution. Pool mean net R = −0.006 (essentially zero with spread drag). Forward path is Brownian (sqrt-t scaling at h=120→240→480). Time-exit curve degrades monotonically from h=1, foreclosing exit-only candidate paths. K=2 cluster discovery produces real directional archetypes (50/50, MAE 3× MFE vs MFE 3× MAE), but signal-time predictors of cluster membership cap at AUC 0.52 — multivariate ML (RF/GB) and partial AUC at worst-decile both confirm. No held-bar predictor crosses AUC 0.65 at t < 10. The cross-pair effect class is real and detected (5/8 family Tier 1/2), but does not translate to a per-fold-stable filter at this horizon. CLOSE verdict, no step 4-6 work.

**Project-level implication:** L4 small-Sharpe (≤ 0.05) high-DSR signals at h=1 should be expected to CLOSE under L_ARC_PROTOCOL because the per-trade edge floor at h=1 in FX is too thin relative to spread + variance. The protocol correctly produces a CLOSE without requiring exhaustive step 4-6 effort.

**Pipeline validation:** v1.1 amendment executed cleanly. All 12 amendments specified worked as designed in arc 1 (family calibration PASS, HDBSCAN integration clean, new clustering features stable, partial AUC computed, hash-seed determinism PASS at 24/24). Pipeline is production-ready for arcs 2-5.

**Interaction-effect test:** Phase C RF (depth 5) and GB (depth 3) tests 2-way and 3-way feature interactions implicitly. Both peaked at AUC 0.521-0.531, barely above logistic regression's 0.506-0.511. This is a robust negative result on the "we missed an interaction effect" hypothesis — the protocol's current model ensemble adequately tests interaction effects at standard depth, and they did not surface in arc 1.

**Calibration check working as designed:** The v1.0 feature-level check FAILed (correctly halted), surfacing a spec design issue (CH-001 carrier was noisier than its family). v1.1 amendment generalized to family-level. The discipline of halt-on-FAIL prevented premature closure on weak evidence.

---

## §5. Permanently superseded features

Features in earlier protocol versions that are now NOT in the active feature schema. Reference: which arc(s) and what reasoning led to removal.

| Feature | Removed in version | Reason | Arcs where it failed | Notes |
|---|---|---|---|---|

_Currently empty — no features have been removed from v1.0/v1.1 feature schema._

---

## §6. Maintenance log

| Date | Arc closed | Updated by | Entries appended | Notes |
|---|---|---|---|---|
| 2026-05-14 (TBD at sign-off) | Arc 1 (redo) | planning chat | 10 §1 rows (predictors), 5 §2 rows (filter candidates), 4 §3 rows (cluster archetypes), 1 §4 paragraph (methodology learnings) | First populated entries. Verdict CLOSE. Pipeline validated. |

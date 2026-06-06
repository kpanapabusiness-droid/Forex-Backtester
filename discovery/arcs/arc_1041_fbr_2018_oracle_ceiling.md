# arc 1041 — oracle-best-cluster CEILING on fbr's 2018 fold: is the fbr→PASS route salvageable in principle?

**Chat:** 1000s · **Date:** 2026-06-06 · **Verdict: DIAGNOSTIC → KILL** (fbr→PASS route DEAD in principle; fbr UNCHANGED, PORTFOLIO) · OOS never touched.

## Frame
Continued 1000s after arc 1040. Loop state: range 1000-1999, next id 1041. arc 1040 established that fbr —
the corpus's strongest edge, one fold from all-folds-positive PASS — has an 2018 fold that D1-level
confluence (the 3rd collinear level-significance filter, after shadow-depth [1025] + same-TF touch-count
[2030]) only MITIGATES (−4.39%→−2.57%), never flips. That left the load-bearing question open: is fbr-2018
unsalvageable in PRINCIPLE (no selection rescues it), or did the three causal filters just miss a real
sub-structure? Settling it definitively closes (or re-opens) the corpus's top near-miss.

## Method (§5d oracle ceiling — trustworthy-asymmetric)
The oracle-best-cluster ceiling is the right instrument: nothing can exceed perfect-hindsight cluster
selection, so a weak ceiling is a trustworthy KILL. Built fbr's IS pool (`build_arc_pool`, 237 trades),
clustered path shapes (canonical `run_step_2` → k=4; `run_step_3` flags clusters 0 & 1 as candidates), then
scored EACH post-hoc cluster's **2018** ROI on the honest engine (`OracleFoldRunner` = perfect-hindsight
cluster membership → `A1Architecture` → `MultiPairBacktester`, FundedNext, fbr's committed
`sl_plus_trailing_atr` exit) over the 2018 build_v3_folds fold. Driver: `_arc1040_work/oracle_fbr_2018.py`.

## Result — fbr-2018 is UNSALVAGEABLE by path-shape selection

| fbr-2018 selection | 2018 ROI | n_2018 | pool members |
|---|---|---|---|
| baseline (no selection) | −4.39% | 18 | 237 |
| cluster 1 (candidate, bulk) | −4.11% | 8 | 96 |
| cluster 2 | −2.64% | 5 | 52 |
| cluster 3 | −2.59% | 5 | 50 |
| cluster 0 (candidate) | +0.00% | **0 (empty)** | 39 |

**Every path-shape cluster that actually trades in 2018 is negative (−2.59% to −4.11%).** The single
non-negative cell is cluster 0 — which has ZERO 2018 fires, i.e. its only way to be "non-losing" is to not
trade the year (a vacuous +0.00% on n=0). So the perfect-hindsight ceiling on fbr-2018 is: lose ≥2.59%, or
sit out. **No selection — not even with perfect hindsight on the realised forward path — makes fbr-2018
positive.**

## Why / what it means
fbr-2018's loss is distributed across EVERY traded path-shape, not concentrated in one separable bad
cluster — consistent with arc 2014 (strong-USD turns failed breakdowns into real breakdowns *systematically*)
and arc 1025 (fbr-2018 is regime-intrinsic, density-invariant) and arc 1040 (TF-invariant). The only
"salvage" is to NOT trade 2018, which requires ex-ante knowledge that 2018 will be a strong-USD breakdown
regime — the unsolved regime-prediction problem; no causal proxy provides it (D1-confluence/touch/depth all
still FIRE in 2018 and still lose, arc 1040/2030/1025). **Therefore the fbr→PASS route is dead in
principle: fbr cannot reach all-folds-positive PASS; it is capped at 9/10 by an unsalvageable 2018 and stays
the strongest PORTFOLIO component.**

**Weak residual thread (noted, not pursued):** cluster 0 (a step_3 candidate, 39 pool members, capturable
path-shape) never fires in 2018 — its setup type is simply absent in the strong-USD regime. IF a CAUSAL
feature proxied cluster-0 membership, selecting on it would naturally sit out 2018. But cluster membership
is post-hoc (forward-path shape), and the obvious causal proxies (D1-confluence/shadow-depth/touch-count)
do NOT isolate it (they fire in 2018 and lose). Finding a causal cluster-0 proxy would be a NEW hunt against
a known-hard target; the three nearest attempts failed.

## Verdict
DIAGNOSTIC → KILL (no new component; fbr UNCHANGED, PORTFOLIO). Definitively closes the fbr→PASS route:
fbr-2018 is unsalvageable by selection. Operative lever unchanged = operator path-A (gate governance 1032 +
deployment risk-appetite 1033/2033). OOS never touched.

## Lesson (new)
The oracle-best-cluster ceiling settles "is a binding fold salvageable" definitively where a single causal
filter cannot: clustering fbr's pool and scoring each cluster's 2018 ROI shows EVERY traded path-shape is
2018-negative (best non-empty −2.59%), the only non-losing option being an empty cluster (don't trade). A
binding fold whose loss is distributed across all post-hoc path-shapes — not concentrated in one separable
bad cluster — is regime-intrinsic and unsalvageable by ANY entry selection (causal or oracle); only a
regime-timing overlay (the unsolved prediction problem) could avoid it. Generalizes arcs 1040/2030/1025 from
"no causal filter rescues fbr-2018" to "no selector does."

## Tooling
No new BUILT tool (reused canonical `build_arc_pool` / `run_step_2` / `run_step_3` / `OracleFoldRunner`).
Driver scratch (uncommitted): `_arc1040_work/oracle_fbr_2018.py`. **FLAGS:** none. No canonical-core change.
OOS NOT touched.

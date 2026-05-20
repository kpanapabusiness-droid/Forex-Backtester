# Arc 10 — Closure (updated)

**Disposition:** `STEP_4_HALT` per §16a Path A — unchanged
**Pipeline branch:** `claude/charming-mcnulty-8160e0` (worktree)
**Last updated:** 2026-05-18

This document supersedes the original closure written at HALT (commit `994f642`). The post-closure experimentation pass and WFO pair — run over §16a at chat-side direction as research probes — produced material findings and two corrections to claims in the original. The disposition is unchanged. Nothing here authorises deployment.

---

## Commit trail

| Commit | Phase | Result |
|--------|-------|--------|
| `362a085` | Step 1 — plumbing | PASS |
| `f39dcd9` | Steps 2–3 — clustering + capturability | PASS |
| `994f642` | Step 4 — extractability | HALT |
| `73ba1f0` | Experimentation pass (EXP-01–06) | Complete |
| `07e41c1` | WFO pair (base + oracle c1) + synthesis | Complete |

`results/ARC_QUEUE.md` not touched at any commit — parallel CC session running Arcs 8 / 9 / 11 retains queue ownership.

---

## Pipeline results

| Step | Gate | Result |
|------|------|--------|
| 1 | Pool ≥ 500, deterministic, lookahead-free, D1-lag verified | **PASS** — 802 trades, byte-identical, 5/5 lookahead, 3/3 D1-lag NaN-perturbation, KH-24 co-fire 0% |
| 2 | Path-shape clustering | **PASS** — K=3, silhouette 0.4525, 0/4 degenerate |
| 3 | Capturability | **PASS** — c1 V-shape, SL=3.0×ATR, composite 0.4934, fwd_mfe_p50 3.08 R, wrong_way_pp 0 |
| 4 | E ≥ 0.65 OR D1 ≥ 0.60 | **FAIL near-miss** — E AUC 0.6296 (margin −0.0204), D1 AUC 0.5897 (margin −0.0103) |

Disposition reasoning (§16a Path A, compound vs strict): see original closure at `994f642`; unchanged.

---

## Post-closure research findings

### Experimentation pass — commit `73ba1f0`

| Exp | Headline finding |
|---|---|
| EXP-01 | E and D1 AUC 95% CIs straddle thresholds; joint P(either clears) = 40.5%. Arc 10 alone is in the noise zone. |
| EXP-02 | `L1_minus_L0_atr` (D1 HL slope) carries 116% of HTF LOO drop; age features 18%. Load-bearing single feature. |
| EXP-03 | Triple-pass threshold E ≥ 0.536 (Arc 7 binding); pair-pass at 0.600. 0 documented FP across arc history at any relaxed threshold. |
| EXP-04 | Fold-2 date correction: **2023-07 → 2024-06**, not Q2 2022. No entry-time regime descriptor places fold-2 ≥ 1.5σ from cohort mean. Drop unexplained. |
| EXP-05 | Cross-arc V-shape pool (Arc 7 c3 + Arc 10 c1) AUC 0.6348 (gap −0.015). +0.029 over Arc 10 alone. **Arc 6 reclassified Stepwise, not V-shape** — BLOCKED for pool. |
| EXP-06 | Open-04 probes (D1 Kijun distance, session dummies) both negative on Arc 10 alone. Informational only; needs Arc 8 / 9 / 11 reproduction. |

### WFO pair — commit `07e41c1`

| Metric | Base | Oracle c1 | Gap |
|---|---|---|---|
| Sharpe (annualised) | −1.29 | 4.61 [CI 3.13–6.09] | +5.90 |
| Calmar | −14.2 | 71.2 | +85.4 (+602%) |
| Expectancy (R/trade) | 0.40 | 1.55 | +1.15 |
| Win rate | 33% | 65% | +32 pp |
| Profit factor | 0.19 | 8.10 | +7.91 |
| Max drawdown | 1.12% | 0.50% | −0.62 (lower is better) |
| Admits / fold | 1.4 | 3.9 | +2.5 |

Synthesis recommendation: **build** (clusterifier) — both material thresholds (Sharpe +0.30, Calmar +50%) cleared by wide margin.

Critical caveats:

- Oracle is upper bound. Real classifier is strictly worse. Per EXP-01, P(realisable AUC ≥ 0.65) = 12.5%; realised lift is a fraction of the +5.90 Sharpe gap.
- Base optimiser admitted ~1.4 trades/fold; 4/8 folds admitted zero. Inner-CV could not find a parameter set producing positive expected edge on the unfiltered signal. This is the P&L-language version of the AUC verdict at Step 4 — same conclusion, two pipelines, same pool.
- Oracle fold sizes 14–15 trades — wide per-fold bootstrap CIs. Small-N uncertainty does not invalidate the gap but limits its precision.

---

## Material corrections to prior documents

1. **Arc 6 archetype.** Previously classified as the third V-shape near-miss (Arc 6 / Arc 7 / Arc 10). Per EXP-05, Arc 6 is **Stepwise, not V-shape**. The "three V-shape cohorts" framing in the original closure and in `ARC_10_LIVE.md` is incorrect. Two genuine V-shape near-misses: Arc 7 and Arc 10.
2. **Fold-2 date.** Previously stated as Q2 2022. Actual fold-2 window: **2023-07 → 2024-06** (EXP-04).
3. **Open-06 case.** Weakened by (1). Threshold relaxation that admits the Arc 6 + 10 pair mixes archetypes. A clean V-shape pair (Arc 7 + Arc 10) requires deeper relaxation (binding at ~0.536 via Arc 7). Reassess in v2.4 cycle.

---

## Why we can't filter to c1 — diagnostic framing

c1 is defined by realised path shape. Entry-time discriminability depends on the mutual information between entry-time features and the future-path label. The current feature envelope caps that mutual information at ≈0.6296 AUC on Arc 10. Step 4 confirms this is below the gate; it does **not** say what features would lift it above. Two distinct paths forward, not exclusive:

- **Classifier path.** Extend the entry-time feature envelope via qualitative characterisation of c1 entries → new feature families: multi-TF trend alignment, pre-entry pattern context (failed-break-reversal, compression-expansion sequences, distance-to-recent-swing), volatility-regime descriptors, within-c1 sub-clustering. Re-test through Step 4 on the extended envelope.
- **Filter path.** Skip the classifier abstraction. Hand-engineer deterministic entry-time conditions (e.g. `D1 slope > X AND compression ratio < Y AND realised-vol-percentile > Z`) that select c1-like setups by construction. Validate on post-filter trade-set properties (Sharpe, expectancy) rather than AUC. More robust at N = 802; interpretable; sidesteps the AUC gate entirely.

Step 4 implements the classifier path within a fixed envelope. The filter path is what you'd reach for after a Step 4 HALT if you wanted a different validation regime entirely. EXP-02's `L1_minus_L0_atr` finding is a fragment of both: a single structural condition that turned out to carry the lift, suggesting reverse-engineered conditions can pull weight.

The highest-leverage diagnostic before any new feature work: distributional comparison of every existing feature between c1 and non-c1 trades, ranked by separation. If the biggest separations are features the classifier already uses heavily → feature-limited (need new families). If there are high-separation features the classifier under-weights → model-limited (need different model class or interaction features). Different fixes; one experiment distinguishes them.

---

## Recommended next dispatches

In rank order for the v2.4 cycle. Each is a separate dispatch.

1. **Reverse-FE diagnostic.** c1-vs-rest distributional comparison across the existing feature set, ranked by KL / Mann-Whitney separation. Plus SHAP and permutation importance on the existing 0.6296 classifier. Outcome routes everything downstream. Cheap; should run first.
2. **Cross-arc clusterifier build.** Arc 7 c3 + Arc 10 c1, slope feature (`L1_minus_L0_atr`) mandatory in catalog. Extended feature envelope informed by (1). Validation via Step 4 on the cross-arc pool. Strongest single piece of evidence for cross-arc V-shape edge sits behind this.
3. **Filter-path probe (parallel to 2).** Hand-engineered c1 filter on Arc 10, validated on post-filter trade properties — Sharpe, expectancy, max DD. Different validation regime than (2); comparison is informative either way. Cheap.
4. **v2.4 calibration packet.** Bundle EXP-01–06 + WFO pair + corrections (Arc 6 reclass, fold-2 date) for the cross-arc cycle. Open-06 threshold case weakened; Open-04 deferred pending multi-arc reproduction.

---

## Hard non-goals

- **Do not deploy.** Arc 10 remains `STEP_4_HALT`.
- **Do not promote the oracle WFO result.** Cluster ID is not available at trade-entry time in production.
- **Do not touch `results/ARC_QUEUE.md`.** Parallel CC session owns queue transitions.
- **Do not advance Open-04 commission** on Arc 10 evidence alone. EXP-06 is informational; commission needs Arc 8 / 9 / 11 reproduction.

---

## Header to carry into all downstream docs

> ⚠️ Arc 10 disposition is `STEP_4_HALT`. Post-closure research (experimentation + WFO) was conducted over §16a as a research probe at chat-side direction. Not a deployment evaluation. Not a re-disposition.

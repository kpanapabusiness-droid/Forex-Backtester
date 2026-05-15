# L_ARC_PROTOCOL — Master Methodology for L Arc Signal Testing

**Status:** LOCKED v1.0
**Locked date:** 2026-05-13
**Scope:** Every arc that tests a signal from `docs/LCHAR_TOPN_REGISTRY.md` (currently Arcs 1–5) and every arc that originates from a `results/CANDIDATE_HYPOTHESES.md` entry.
**Supersedes:** `docs/L6_0_METHODOLOGY_LOCK.md` §9 (no filter rescue, verbatim-as-gate), §14 disposition rules. The L6.0 feature schema (§14.3) and pair-set / WFO structure (§5, §4) carry forward unchanged. Spread-floor lock unchanged.
**Companion doc:** `L_ARC_OPERATIONAL_SPEC.md` — deliverables, folder layout, angle catalogues, scoring tables, effect size definitions.
**Modification rule:** No modifications without an explicit re-planning phase. Each new arc is treated as a confirmation that the protocol holds; if it breaks, modification requires a new locked version (v1.1, v2.0, etc.) with version history.

---

## Document History

| Version | Date | Scope |
|---------|------|-------|
| v1.0 | 2026-05-13 | Initial lock. 6-step extractability protocol replaces verbatim-as-gate framing. Includes dual-tier WFO disposition (PASS-DEPLOYABLE / PASS-VIABLE), effect-size + AUC verdict logic, component-ranked candidate scoring, post-arc routing, Arc 1 redo as protocol calibration check with fail procedure, fold-duration-aware annualisation and trade-count scaling, creativity-vs-rigidity discipline. |

---

## 1. Purpose

This document is the single methodology of record for every signal-testing arc in the L research line. It is read first in every new chat that opens or continues an arc. It answers, at every step: what question is this step asking, what evidence does it produce, what is the verdict logic, what closes the step.

The objective is a profitable, deployable trading system — maximising ROI under prop-firm constraints (DD < 8%). The protocol is the barrel of the gun that keeps every arc pointed at that objective. Verbatim WFO at the start of an arc is not a viability test; raw L registry signals tested at h=1 with a 2×ATR SL and no exposure cap will essentially always blow the WFO gate (Arcs 1 and 2 verbatim FAILed catastrophically — DD 39%–91% across folds). Closing arcs on verbatim FAIL leaves real extractable structure on the table. The L arc characterisation already established that the registry signals have measurable forward drift; the question this protocol answers is whether that drift can be extracted under realistic execution, packaged into a system that meets the deployability bar.

## 2. Scope

In scope:
- Every L registry signal tested as its own arc (1, 2, 3, 4, 5 — all run as full separate arcs through steps 1–6, no conditional-duplicate scheduling).
- Every arc spawned from `results/CANDIDATE_HYPOTHESES.md`.
- All cross-arc registries and feature aggregations produced by these arcs.

Out of scope:
- Modifications to the L registry, spread floor, or any locked artefact.
- Retroactive re-evaluation of completed arcs when later arcs surface new features (see §16).
- Decisions about what to do with PASS-DEPLOYABLE survivors (deployment, sizing, portfolio context) — those are post-arc decisions made by the planner at the time, not pre-committed in this protocol (see §17).

---

## 3. The Six Steps — Overview

| Step | Question asked | Produces | Closure condition |
|------|----------------|----------|-------------------|
| 1 | Does the signal fire cleanly under realistic execution? | Trade-set + sanity-check report | Plumbing passes; trade-set delivered. No gate disposition. |
| 2 | What does the trade-path distribution look like, in full? | Full descriptive analysis | All angle-catalogue tests run; full distributions reported. |
| 3 | Is the signal extractable — can we partition trades into archetypes by features observable at or after entry? | Cluster geometry + predictor scan + **verdict** | Verdict: PROCEED / AMBIGUOUS-PROBE / CLOSE. |
| 4 | Among candidate filters/exits, which look best on which metrics? | One doc covering all candidates | All viable candidates evaluated; ranked on multi-component table. |
| 5 | If a filter is applied, what does the resulting trade-set look like under steps 2 and 3 templates? | Re-characterised population | Step 2 + step 3 templates re-run; verdict on filtered population. |
| 6 | Does the final (filter + exit) system pass the integrity test out-of-sample at deployable or viable economics? | WFO dual-tier disposition | Final arc verdict: PASS-DEPLOYABLE / PASS-VIABLE / clean-null. |

Each step has the gate-disposition logic explained in §5–§10 below.

---

## 4. Discipline Rules (Apply To Every Step)

1. **Full distributions, never medians-only.** Every reported metric reports: mean, std, skew, kurt, min, p1/5/10/20/30/40/50/60/70/80/90/95/99, max. 2D distributions reported as full heatmaps, not just correlations. Reading a single number tells you almost nothing about a trade population.

2. **No lookahead anywhere.** Filters use only data computable at bar N close (signal time, t=0). Exits at bar t use only data computable at bar t (t ≥ 0 after entry). Every step's deliverables include a lookahead-invariant test: a synthetic data perturbation that confirms future bars do not affect the current decision. Hard fail on violation.

3. **Honest scan documentation.** Step 3 features are scanned in bulk. Every step 3 report states how many features were examined and applies an explicit multiple-comparisons haircut (Benjamini-Hochberg) producing reporting tiers — not gates (see operational spec §6.7).

4. **Effect size before significance.** AUC and p-values measure discrimination and significance — they do not measure whether the discriminated thing matters economically. Cluster differentiation requires forward-geometry effect size thresholds (defined in operational spec §8) AND predictor AUC thresholds. Both gates conjunctive. Mean R is supplementary, not primary, because pre-exit mean R is dominated by the wrong exit policy.

5. **Determinism.** Every step's outputs are byte-identical on re-run (same git commit, same data). All scripts log input/output sha256s. CI-enforced.

6. **One change per phase within a step.** Step 4 candidates are isolated from each other (one filter or one exit at a time, not stacked). Stacking happens at step 6.

7. **Risk per trade follows L6 convention (0.5%).** Position sizing fixed at 0.5% × reset floor balance until and unless a sizing arc opens.

8. **Creativity belongs in interpretation, not in threshold-moving.** Chats running arcs are expected to think creatively about: what patterns mean and what they imply about market structure; which features to explore beyond the auto-generated candidates; how to construct unconventional filter or exit candidates that combine multiple features or borrow from cross-arc registry insights; which probes to design when verdicts are ambiguous; how to explain surprising findings; how to reconcile apparent contradictions between sub-analyses. The creativity is in how we use what we observe; the rigidity is in what we are not allowed to move. Chats are NOT free to move locked thresholds (effect size, AUC, DD, trade count, BH tier boundaries) within an arc, change folder structure or naming, skip catalogue angles, or rationalise post-hoc to flip a verdict. The thresholds are the barrel — they keep us pointed at building a profitable system.

---

## 5. Step 1 — Verbatim Run (Plumbing Test)

**Question asked:** Does the L registry signal fire correctly under realistic 1H execution? Are the trade counts, spread-floor application rates, and lookahead invariants all within audit ranges?

**This is not a gate.** The verbatim trade-set is the input to step 2 regardless of how the WFO would judge it. The pre-WFO numbers (mean R, fold ROI, etc.) are recorded for context but no disposition is taken on them.

**Configuration:**
- Signal: per `docs/LCHAR_TOPN_REGISTRY.md` entry (the registry definition is the source of truth — verify against `scripts/lchar/run_layer4.py` for the canonical computation).
- Entry: bar N+1 open.
- SL: 2.0 × ATR(14)_1H from entry price.
- Time exit: bar N+1+h open (h from registry entry).
- Spread: `configs/spread_floors_5ers.yaml` (locked, sha256 in arc-open doc).
- Exposure cap: max 1 open position per pair. No currency cap, no concurrent-trade cap.
- Risk: 0.5% × reset floor balance.

**Cost accounting:**
- Spread per the floor file is the only per-trade cost modelled in steps 1–6 trade accounting.
- Commissions, overnight swap, and slippage are NOT modelled at trade level. Slippage is assumed neutral over many trades (the few pips that go against you cancel against the few that go for you).
- An aggregate operational-cost haircut is applied to ROI figures at PASS-DEPLOYABLE evaluation only (operational spec §7.4). PASS-VIABLE evaluation uses gross-of-haircut numbers.

**Plumbing requirements:**
- Signal definition re-validation: the L registry entry's mathematical definition is computed against the canonical source (`scripts/lchar/run_layer4.py`) on a 100-bar deterministic sample; expected outputs match canonical bit-identical. If not, signal definition has drifted and must be reconciled before continuing.
- Trade count within ±5% of L4 `n_obs_pooled` expectation (soft WARN if outside; investigate).
- Spread-floor hash matches the locked sha256.
- SL distance = 2.0 × ATR on all trades.
- D1 lag-1 verified.
- Same-bar entries = 0.
- Lookahead-invariant test passes (per operational spec §10).

**Closure:** plumbing-pass sanity report committed; trade-set delivered to step 2. If plumbing fails (lookahead detected, signal definition mismatch, spread-floor hash mismatch, signal count outside expected band by >5%), halt and investigate before proceeding.

---

## 6. Step 2 — Descriptive Trade-Path Analysis

**Question asked:** What does the trade-path distribution look like, in full, across every angle that informs step 3 and step 4?

**Operates on:** the step 1 trade-set, augmented with per-bar held-trade observations from N+1 to exit, and forward-horizon observations from N+1 to N+240 unconditional on SL.

**Forward-horizon coverage and stability check:** standard cap is 240 bars (10 days at 1H). Step 2 includes a stability check comparing the fwd_mfe/mae distributions at h=120 vs h=240 — if medians or p95s differ by more than 10% at the cap, the distribution is still evolving and step 2 extends to h=480 for that arc. Result documented; future arcs on similar-horizon signals inherit the extended cap if needed.

**Standard angle catalogue:** the full list lives in `L_ARC_OPERATIONAL_SPEC.md` §5. Every arc runs every angle. The catalogue is comprehensive — angles can be added if a future arc surfaces a useful one, never removed.

**Output discipline:** every metric reported with the full distribution shape (per discipline rule #1). Marginals, 2D joint distributions, conditional breakdowns (by pair, fold, session, regime, etc.), shadow trade-sets (entry-delay sensitivity, SL-distance sensitivity, time-exit horizon sensitivity, cost stress), random-entry baseline comparison.

**Closure:** all angles run; signals_features.csv and trade_paths.csv delivered to step 3; PHASE_L_ARC_N_STEP2.md committed. No verdict — descriptive only.

---

## 7. Step 3 — Extractability Assessment + Verdict

**Question asked:** Are there stable archetypes in the trade-path distribution, and is membership in any archetype predictable from features observable at signal time (t=0) or during the trade (t > 0, no lookahead), AND does that archetype membership matter economically (forward-geometry effect size)?

**Operates on:** step 2's signals_features.csv and trade_paths.csv. Step 3 does not re-process raw trade data; it analyses the step 2 output.

**The sub-analyses:**

**3a — Cluster discovery.** Cluster on trade-path features (mfe, mae, bars-held, time-to-peak, time-to-trough, peak/final ratio, oscillation count, monotonicity ratio, MFE/MAE sequence). K-means at K=2,3,4,5,6 plus hierarchical clustering. Per-cluster: count, mean R full distribution, fwd_mfe and fwd_mae full distributions, exit-reason mix.

**3b — Cluster stability.** Does each cluster appear in every fold with similar size and similar economics? Cluster present only in F2 with 80% of its trades there is fold-specific, not structural — discount accordingly.

**3c — Signal-time predictor scan (t=0).** For each cluster, fit logistic regression and shallow tree (depth ≤3) on the full signal-time feature set to predict cluster membership; supplement with random forest and gradient boosting for interaction detection. Per-cluster AUC, top features by importance, per-fold AUC stability. Multicollinearity check on top-20 predictors. The "is this filterable" question.

**3d — Held-bar predictor scan (t > 0).** Same as 3c, but progressively adding features observable at t=1, 3, 5, 10, 20. Per-cluster AUC at each t cutoff. Report the earliest t at which AUC crosses 0.65, 0.70, 0.75 per cluster. The "is this archetype-conditional-exit-able" question.

**3e–3h — Stratification axes** (pair, volatility regime, session/DOW/hour, hour-in-4H-bar, hour-in-D1-bar, pre-momentum). Per-stratum cluster mix and conditional forward-geometry distribution.

**Verdict logic (dual-gate: AUC AND effect size AND cluster size AND stability):**

A cluster is **extractable** when ALL FOUR conditions hold:
- **AUC condition:** signal-time predictor AUC > 0.65 on the cluster (3c) OR held-bar predictor AUC > 0.70 at t ≤ 20 on the cluster (3d).
- **Effect size condition:** the cluster differs from the pool on forward geometry by at least the thresholds in operational spec §8 (fwd_mfe_h24 median difference, fwd_mfe-to-mae ratio difference, or bars-to-+1ATR vs bars-to--1ATR race).
- **Cluster size condition:** the cluster comprises ≥15% of the pool. Smaller clusters can only be filtered OUT (filter the small bad cluster, keep the rest), not filtered TO (filter to the small good cluster).
- **Stability condition:** cluster passes 3b stability across folds.

| Verdict | Trigger condition | Disposition |
|---------|-------------------|-------------|
| **PROCEED — extractable** | At least one cluster passes all four conditions above. | Step 4 opens with the predictive features as candidate filter/exit axes. |
| **AMBIGUOUS — needs probe** | Cluster(s) pass AUC and stability but fall short on effect size (within 50% of threshold) OR on size (10–15% of pool). | One probe budget: planner applies the most-promising filter or delayed-exit-by-feature descriptively to the trade-set, with recorded reasoning for the probe choice, re-runs step 3 on the modified population, sees if verdict sharpens to PROCEED or CLOSE. |
| **CLOSE — non-extractable** | No cluster passes all four conditions across the scan, AND probe (if used) did not sharpen. | Arc closes clean. Findings logged in `results/CANDIDATE_HYPOTHESES.md` and `NEGATIVE_FINDINGS.md` as appropriate; arc folder marked CLOSED-NULL. |

**Decision tree for step 4 routing (when verdict is PROCEED):**

- **Non-extractable cluster present + signal-time separable** → filter that cluster out, exit-engineer the survivors (filter path).
- **All clusters partially extractable, differ in exit-optimal policy** → cluster-conditional exit (exit path).
- **Both available, ambiguous which dominates** → both paths tested in step 4 as parallel candidates; step 6 WFO decides.

**Prefer exits over filters when both work equally.** Exits don't reduce N. A filter that drops 60% of trades to lift mean R is structurally weaker than an exit that keeps all trades and lifts captured MFE, even if both produce similar improvement, because the filter shrinks the statistical base and the live trade frequency. Exception: when a filter cleanly removes a structurally non-extractable cluster, filter is preferred — those trades carry risk with no upside under any exit.

---

## 8. Step 4 — Filter / Exit Derivation

**Question asked:** Among the candidate filters and exits surfaced by step 3, which look best on which metrics?

**Output discipline:** one document covering all candidates (`PHASE_L_ARC_N_STEP4.md`). Per-candidate subfolder for config, spec, and per-candidate evaluation outputs.

**Candidate auto-generation rule.** Step 4 candidates are auto-generated from step 3 outputs:
- Every predictor in the §6.8 filter dry-run with AUC > 0.65 AND effect size threshold cleared on at least one cluster becomes a candidate filter.
- Every held-bar predictor with AUC > 0.70 at t ≤ 20 on a cluster with extractable-cluster path characteristics becomes a candidate cluster-conditional exit.
- The unfiltered + best-time-exit-from-step-2-sensitivity-sweep is always a candidate (the exit-only baseline).
- Planner may add additional candidates beyond auto-generation (combined-feature filters, unconventional exit constructions, etc.), recording reasoning — this is where chat creativity belongs per discipline rule #8.

**Filter candidate specification.** Each filter candidate spec names the target step-3 cluster it intends to keep (or filter out, for the opposite path), described by the cluster's path characteristics: median mfe_held, median mae_held, median bars_held, dominant MFE/MAE sequence class. This is a documentation requirement — it makes the filter's intent legible and lets step 5 re-characterisation report whether the filter actually achieved its intent. Not a gate — a filter that doesn't do what was intended but still produces a step 6 PASS-DEPLOYABLE survivor is fine; the WFO is the truth. The named target just makes any disconnect visible.

**Filter candidate evaluation — component-ranked, not single-composite.**

Each filter candidate is reported across the components in operational spec §7.1. Candidate is **viable** only if ALL components clear their floors. The planner reads the component table for all candidates side-by-side and picks which advance to step 5 / step 6. No single composite score — false precision avoided. Multiple candidates may advance if their component profiles are meaningfully different.

**Exit candidate evaluation** — exits operate on a fixed population, so mean R is back in play but still supplementary. Metrics per operational spec §7.2.

**Held-out fold check before step 6:** derive exit parameters on folds 1–5, evaluate on folds 6 and 7. Last-two convention maximizes temporal generalisation test. Held-out folds must show mean R > 0 and positive capture ratio for the exit to advance.

**Lookahead invariant for exits:** an exit decision at bar t may use only data with timestamps ≤ t (or, for higher-TF features, the most-recently-completed convention). Same synthetic-perturbation test as filters. Hard fail on violation.

**Zero-candidate disposition.** If no filter candidate clears the component-table viability floors AND no exit candidate passes the held-out fold check, arc closes as clean null. No retry with relaxed floors within the same arc. Findings logged.

**Closure:** all candidates evaluated; component table committed; planner records which candidate(s) advance to step 5 / step 6 with reasoning in the phase doc.

---

## 9. Step 5 — Re-characterisation

**Question asked (when a filter was applied):** What does the filtered trade-set look like under the step 2 and step 3 templates? Does the filter change the exit-optimal policy? Did the filter achieve what its spec intended?

**This step exists because filters shift the population.** A cluster that was 20% of trades pre-filter may be 60% post-filter, with materially different exit-optimal behaviour. The whole step 2 + step 3 pipeline reruns on the filtered population, and the step 3 verdict on the filtered population is what informs step 6 configuration.

**Operates on:** the step 4 filtered trade-set (one re-characterisation per filter candidate that progresses).

**Filter intent check (sanity, not gate):** the re-characterisation report includes a comparison of the post-filter cluster mix vs the step 4 candidate spec's named target. Specifically: does a cluster matching the named target's path characteristics now exist in the filtered population, and is it over-represented relative to the original pool? Discrepancy between intent and outcome is documented but does not block advancement.

**Output:** subfolder per candidate. Full step 2 + step 3 templates re-run. Standard verdicts.

**Closure:**
- Filtered-population step 3 verdict = PROCEED → step 6 opens with the filter + the exit-optimal-on-filtered-population.
- Filtered-population step 3 verdict = AMBIGUOUS → one probe budget applies; if probe sharpens to PROCEED, continue; otherwise drop the candidate.
- Filtered-population step 3 verdict = CLOSE → candidate dropped, not advanced to step 6.

**Note on candidates that take the exit-only path (no filter):** step 5 is skipped — there's no population shift to re-characterise. The step 4 exit candidate goes directly to step 6.

---

## 10. Step 6 — Joint WFO (Integrity Test) — Dual-Tier Disposition

**Question asked:** Does the final (filter + exit) system pass the prop-firm gate out-of-sample across all 7 anchored expanding folds at deployable economics, viable economics, or neither?

**This is where the gate lives.** All prior steps are diagnostic; step 6 is judgement.

### 10.1 Calculation conventions (apply to all tiers)

These conventions are required because the L6.0 anchored expanding WFO produces folds with very different OOS durations (fold 1 covers ~3–6 months; fold 7 covers 5+ years). Tier thresholds quoted in §10.2 assume the conventions below; without them, "worst-fold annualised ROI > 5%" is ambiguous and tier disposition will hit ad-hoc decision points.

- **Per-fold annualised ROI:** `fold_raw_ROI × (365 / fold_OOS_days)`. Calculation uses calendar days, not trading days. `fold_OOS_days` is read from the WFO fold spec.
- **Short-fold exclusion from worst-fold annualisation:** folds with OOS duration < 90 calendar days are excluded from the worst-fold annualised ROI calculation. Their raw economics are reported but their annualised numbers are too noisy to anchor a tier decision (a 3-month fold with 12% raw ROI annualises to ~48%; the same system at 12% over 12 months annualises to 12% — same evidence, wildly different "annualised" figure).
- **Short folds still contribute to:** mean fold annualised ROI calculation (weighted by fold OOS days), DD constraint, trades-per-fold count, fold-to-fold sign consistency.
- **Trades-per-fold floor scaling:** the trades-per-fold floor scales with fold OOS duration:
  - Folds with OOS duration ≥ 180 days: floor = 15 trades.
  - Folds with OOS duration < 180 days: floor = `max(5, round(15 × fold_OOS_days / 180))`. Minimum 5 regardless of duration.
  - Short-fold trade counts below the scaled floor inform robustness but don't drive auto-disqualification.
- **Net of spread, gross of haircut:** all ROI figures at PASS-VIABLE evaluation are net of spread (per the floor file applied in step 1), gross of operational-cost haircut. PASS-DEPLOYABLE evaluation applies the additional haircut per operational spec §7.4.

### 10.2 Tier thresholds

**Tier 1 — PASS-DEPLOYABLE** (system is deployment-ready, subject to operational-cost haircut and planner decision):
1. Worst-fold annualised ROI > 5% (net of spread, net of haircut, calculated per §10.1).
2. Worst-fold max DD < 8%.
3. Mean fold annualised ROI > 8% (net of spread, net of haircut, calculated per §10.1).
4. Trades per fold ≥ scaled floor per §10.1.

All four conjunctive.

**Tier 2 — PASS-VIABLE** (system has measurable edge but does not meet deployment thresholds):
1. Worst-fold ROI > 0% (any positive, net of spread, gross of haircut).
2. Worst-fold max DD < 8%.
3. Trades per fold ≥ scaled floor per §10.1.

All three conjunctive. PASS-VIABLE is the floor for "this is real edge worth documenting" — below this is clean-null.

**Tier 3 — CLEAN-NULL:** does not meet PASS-VIABLE thresholds across all 7 folds, or any worst-fold DD ≥ 8%.

The DD constraint (< 8%) applies to both PASS tiers regardless of haircut — it's a prop-firm hard limit and is not subject to haircut adjustments.

### 10.3 Configuration and disposition

**No cap on candidates entering step 6.** Run as many as the step 4 / step 5 work surfaces, judgement call at the time. Compute is not the binding constraint; clarity is. Common-case expectation: 2–4 candidate configs per arc reach step 6.

**Configuration matrix** — typically:
- One unfiltered + exit candidate (the exit-only path)
- One or more (filter + exit) candidates (the filter path)

Both paths run as full WFO.

**Composite verdict between passing candidates:**
1. Tier disposition first. PASS-DEPLOYABLE candidates outrank PASS-VIABLE.
2. If multiple within same tier: worst-fold ROI, worst-fold DD, fold-to-fold ROI consistency (std of fold ROI), trade count headroom.
3. Tiebreaker: prefer the system that holds more trades at equal performance.

**Closure:** arc closes with the highest-tier disposition of any candidate. Arc-level `PHASE_L_ARC_N_CLOSURE.md` summarises across all steps, names the best candidate per tier, and registers findings to `results/CANDIDATE_HYPOTHESES.md`, `CROSS_ARC_FEATURE_REGISTRY.md`, and `NEGATIVE_FINDINGS.md` as appropriate. Post-arc routing per §17.

---

## 11. Conditional-Duplicate Handling (Arc 4 / Arc 5)

Per user direction at v1.0 lock: Arc 4 (L registry rank 4, Class A duplicate of Arc 1) and Arc 5 (L registry rank 5, Class B duplicate of Arc 2) run as **full separate arcs through steps 1–6**, not as ablations within parent arcs. This supersedes L6.0 §3 conditional-duplicate scheduling.

Rationale: they are different signals (different bases / horizons) and benefit from the full protocol run. Cross-arc class-validation evidence (does the Class-A class survive consistently across rank 1 and rank 4?) is logged in `CROSS_ARC_FEATURE_REGISTRY.md` but does not affect individual arc disposition.

---

## 12. Cross-Arc Artefacts

Maintained at the `results/` root, append-only across all arcs:

| File | Content |
|------|---------|
| `CANDIDATE_HYPOTHESES.md` | Existing registry, repurposed. Effects observed in any arc that might inform future arcs. Append-only. |
| `CROSS_ARC_FEATURE_REGISTRY.md` | Signal-time and held-bar features that show predictive power on more than one arc (per operational spec §14.1 threshold). Effect size per arc, predicted-cluster description. Helps identify structural market features vs spurious cross-arc selection. |
| `NEGATIVE_FINDINGS.md` | Features that consistently DON'T predict across 2+ arcs. Saves future arcs time by surfacing known dead ends. |

Entries appended at arc closure by the closure-doc author.

---

## 13. Chat Boundaries

Each chat reads its predecessor's closure doc and writes its own. Chats are disposable; docs are the persistent record.

| Chat | Scope | Closes with |
|------|-------|-------------|
| A | Step 1 + Step 2 + Step 3 (tightly coupled — step 3 verdict needs full step 2 context) | Step 3 verdict committed; handover to step 4. |
| B | Step 4 (one chat per mechanism type — filters in one chat, exits in another, or both in one if scope is contained) | Step 4 component table; planner decides which candidates advance. |
| C | Step 5 re-characterisation (one chat per filter candidate that advances) | Re-characterisation complete; filtered-population step 3 verdict committed. |
| D | Step 6 WFO (one chat covering all configs) | Arc closure doc; cross-arc registry entries appended. |

If a chat hits context-length pressure mid-step (especially in step 2 or step 3 with heavy outputs), break at a natural sub-boundary with a handover doc and start a continuation chat. Heavy compute (cluster fits, predictor scans, WFO runs) goes through Claude Code; planning and verdict happen in the chat.

---

## 14. Modification Policy

The protocol is locked at v1.0. Modifications require:
1. A re-planning chat that articulates what changed and why.
2. A new locked version (v1.1 for additive amendment, v2.0 for protocol redesign).
3. A version history entry in §Document History.

In-flight arcs are not retroactively modified by protocol changes unless explicitly stated. Each arc runs under the protocol version locked at its arc-open doc.

The angle catalogue in `L_ARC_OPERATIONAL_SPEC.md` may be extended (new angles added) without bumping the protocol version, because the discipline is "every arc runs every catalogued angle" — adding angles only makes the analysis more comprehensive, not different. Removals require a version bump.

---

## 15. Migration: Arc 1 and Arc 2 Redo + Protocol Calibration Check

Arc 1 and Arc 2 were run under L6.0 verbatim-as-gate framing and closed FAIL on the verbatim WFO. Under this protocol they are reopened: step 1 trade-sets are retained (no signal re-computation needed beyond re-validation per §5), step 2 and step 3 templates run on those trade-sets, step 3 verdict drives onward routing.

Arc 1's known CH-001 finding (concurrent_signals_within_3h) will be revisited as one of multiple candidate features in step 3's predictor scan, not pre-committed. The new step 3 may surface a sharper threshold or an interaction effect that the original analysis missed; do not anchor on the threshold 13.

**Protocol calibration check (Arc 1 redo).** The original Arc 1 P2 work passed WFO under L6.0 framing — `concurrent_signals_within_3h` is known real edge. Arc 1 redo therefore doubles as a sensitivity check on the new protocol: before accepting Arc 1's verdict under v1.0 framing, verify that `concurrent_signals_within_3h` surfaces as ≥ Tier 2 predictor of the non-extractable (high-MAE) cluster in step 3's 3c signal-time predictor scan. If it does not surface — i.e. the new protocol's predictor scan or thresholds fail to detect a known-real effect — the protocol is miscalibrated. Halt and investigate before proceeding to Arc 2.

**Calibration-check failure procedure.** The investigation produces one of three outcomes:

- **(a) Implementation bug.** Step 3 implementation has a bug (feature not computed correctly, model fit broken, AUC calculation wrong). Bug identified, fixed, regression test added, Arc 1 redo re-runs from step 3 (steps 1–2 outputs retained if unaffected). Documented as the bug + fix in the Arc 1 phase doc and in `CHANGELOG.md`.
- **(b) Threshold miscalibration.** Step 3 implementation is correct but thresholds (AUC, effect size, BH tier boundaries) are mis-set relative to the strength of real-edge effects. v1.1 patch with revised thresholds, justified by the calibration-check evidence per §9 of operational spec. Arc 1 redo re-runs under v1.1.
- **(c) Data-window-induced detectability change.** Implementation and thresholds are reasonable but the feature is no longer detectable due to data window changes (5ers feed vs FTMO feed used in original CH-001 derivation, period coverage differences). Documented as a known caveat affecting interpretation of subsequent arcs. Protocol proceeds without calibration certainty — Arc 2 onwards run with explicit acknowledgement that the sensitivity check did not establish a confident calibration. Tier 3 candidates and AMBIGUOUS-PROBE verdicts are treated with extra scrutiny by the planner.

Outcome (c) is the worst case but not catastrophic — it means the protocol is operating without a positive control. Subsequent arcs effectively become the calibration evidence. Cross-arc trends in §9 patterns become the primary recalibration trigger.

Order: Arc 1 redo first (debug the protocol and standard scripts on a known signal; serves as the calibration check), Arc 2 redo second (validate the protocol on a different signal class), then proceed to Arc 3 cleanly.

---

## 16. No Retroactive Re-evaluation

Arc results are not retroactively re-evaluated when later arcs surface new features or effects. If Arc 3 discovers a feature that, if applied to Arc 1's data, would change Arc 1's verdict, Arc 1 does not reopen. Cross-arc findings inform future arcs only.

Rationale: every arc runs under the protocol version locked at its arc-open doc; re-evaluating completed arcs every time a new feature surfaces would create an unbounded re-work loop and reintroduce post-hoc rationalisation pressure. Cross-arc learning happens forward, via the registries (§12) and the next arc's design.

Exception: if a cross-arc pattern is strong enough to justify a fresh arc *based on* a prior arc's data (e.g., "Arc 1's signal with Arc 3's discovered filter feature as a pre-committed filter"), that's a new arc opening — it runs the full step 1–6 pipeline on its own merits and is registered separately. It does not modify the prior arc's record.

---

## 17. Post-Arc Routing

| Arc disposition | What happens at closure |
|-----------------|-------------------------|
| **PASS-DEPLOYABLE survivor** | Closure doc names the survivor candidate, records WFO summary, registers cross-arc features. Survivor's config and trade-set archived under `results/l_arc_N/step6_wfo/<config_slug>/`. Deployment is a separate decision made by the planner at the time — not pre-committed in this protocol. |
| **PASS-VIABLE-only survivor** | Closure doc records the survivor with explicit PASS-VIABLE-not-DEPLOYABLE flag. Findings registered. May feed into future hypothesis arcs (composite signals, parameter variations). |
| **Clean-null arc** | Findings registered to `results/CANDIDATE_HYPOTHESES.md` and `NEGATIVE_FINDINGS.md` as appropriate. Move to next arc per registry order. |

**L registry exhaustion fallback:** if all 5 L registry arcs close clean-null or PASS-VIABLE-only, the planner reviews `results/CANDIDATE_HYPOTHESES.md` for composite hypothesis arcs and `CROSS_ARC_FEATURE_REGISTRY.md` for cross-arc feature combinations that warrant a fresh arc. Re-planning chat opens at that point.

---

## 18. Sign-Off

Once committed, the protocol governs every L arc until v1.1 or later supersedes it.

**Signed off by:** _______________
**Date locked:** _______________

---

*End of L_ARC_PROTOCOL v1.0. Operational reference at `L_ARC_OPERATIONAL_SPEC.md`.*

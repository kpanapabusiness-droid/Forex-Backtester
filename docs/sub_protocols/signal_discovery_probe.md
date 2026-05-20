# Sub-Protocol — signal_discovery_probe

> **Status:** locked v1.0
> **Predecessor:** L_PROTOCOL v3.0 overseer (this sub-protocol overrides Step 1 signal generation; Steps 2-5 run normally on discovered signals)
> **Purpose:** when no pre-defined signal hypothesis exists, search the rule space to discover a signal that survives WFO and holdout.

---

## When invoked

An arc's `ARC_OPEN.md` declares `sub_protocol: signal_discovery_probe`. Typical use:

- Vanilla and heavy_ml_probe arcs across multiple signal classes have failed to surface a deployable
- We want to let an automated search find rule combinations humans wouldn't try
- We accept higher overfit risk in exchange for broader hypothesis space

Not invoked by default. The most overfit-prone work in trading research. Bonferroni correction is the binding defence.

---

## What overrides the overseer

### Step 1 — replaced

Vanilla Step 1 (apply a known signal definition) is replaced with rule search:

1. **Search space:** rule combinations expressible as logical predicates over the standard feature space (price geometry, session, multi-TF state, vol regime, cross-pair coherence — same features available in vanilla Step 1).

2. **Rule grammar (locked):**
   - **Atoms:** `feature OP threshold` where OP ∈ {>, <, ≥, ≤, ==, !=} and threshold is from a discrete quantile grid (10th, 25th, 50th, 75th, 90th percentile of feature distribution on training data)
   - **Combinators:** AND, OR, NOT
   - **Cap:** maximum 5 atoms per rule
   - **Direction:** rule declares long or short on trigger; mixed-direction rules require explicit declaration

3. **Search method:** random search (default) or genetic algorithm (declarable per arc).
   - Random search: 10,000 rule combinations evaluated
   - GA: 50 generations × 100 population = 5,000 evaluations
   - Budget is HARD per arc

4. **Causal audit constraint:** every rule's atoms must use features carrying a clean causal lineage tag from feature engineering. Rules touching uncertain-causality features are rejected from the search before evaluation.

5. **Per-rule evaluation during search:**
   - Generate trade pool ex-ante on the rule
   - Pool size < 200 → reject (too thin)
   - Pool size ≥ 200 → compute mean R, sample-size-adjusted Sharpe (Lo's correction), and pass to selection
   - Selection scoring during search: Sharpe-adjusted-by-Bonferroni-correction

6. **Selection-bias accounting:**
   - All rules evaluated count toward the Bonferroni denominator
   - Final reported "discovered" signals must clear `p < 0.05 / N_evaluated`
   - With N=10,000, threshold becomes p < 5e-6 — extremely strict
   - This is the point: the search-space size sets the bar

7. **Output of Step 1:** top-K rules by Bonferroni-corrected significance. K=3 by default. Each becomes a "discovered signal" treated as the arc's signal for Steps 2-5.

### Steps 2-5 — run normally

Each top-K discovered rule runs through standard L_PROTOCOL Steps 2-5 as if it were a hand-designed signal. Same clustering, same capturability, same extraction, same WFO architecture search, same gates.

Critical: **the holdout (2021-2025) was NOT used during rule search.** Rule search ran on 2010-2020 IS only. The standard Step 5 holdout evaluation applies normally to each discovered rule.

---

## Validation discipline

Three independent defences against the AI-bot failure mode:

1. **Bonferroni correction** during search (p < 0.05 / N_evaluated)
2. **Causal audit** of every feature used (Step 6 of overseer applies)
3. **One-shot holdout** at Step 5 (2021-2025 untouched during search)

Any rule that clears all three has earned at least cautious interest. Even then, treat as "interesting hypothesis worth re-running" not "deployable system." Discovered signals SHOULD be re-run as standard arcs with sub_protocol: vanilla before any deployment decision.

---

## Output (in addition to standard arc outputs)

`step_1/discovery/`:
- `search_log.csv` — every rule evaluated with score (10k rows for random search)
- `top_k_rules.md` — final K=3 with Bonferroni-corrected p-values and rule specifications
- `bonferroni_summary.md` — N evaluated, corrected threshold, rules clearing threshold
- `causal_audit_rejections.md` — rules rejected pre-evaluation for causality reasons
- `compute_budget_used.md` — actual evaluations vs cap

For each top-K rule, Steps 2-5 produce standard arc artefacts in `results/<arc_name>/discovered_<K>/step_<N>/`.

---

## Discipline rules specific to this sub-protocol

- Bonferroni denominator is the SEARCH SPACE SIZE, not the number of "interesting-looking" rules. All evaluated rules count.
- Causal lineage is a pre-evaluation gate. A rule with an unclear feature is rejected before it's evaluated, not flagged afterwards.
- The holdout window (2021-2025) is NEVER touched during rule search. If it is, the arc is invalid.
- Top-K rules are TREATED AS HYPOTHESES, not as systems. Discovered rules feed a follow-up vanilla arc for clean re-validation before deployment.
- "Sharpe-adjusted-by-Bonferroni" during search is a ranking aid only; the actual decision is the L_PROTOCOL §3 gate applied to each discovered signal's Step 5 results.

---

## Expected compute cost

- Random search: ~6-12 hours per arc (depends on pool generation speed per rule)
- GA: ~10-20 hours
- Plus Steps 2-5 on top-K=3 rules: ~6-12 hours
- Total: full day to two days per signal discovery arc

Run sparingly. The discovery framework is most valuable when standard signal classes have been exhausted.

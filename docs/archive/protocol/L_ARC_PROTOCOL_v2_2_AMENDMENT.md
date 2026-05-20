# L_ARC_PROTOCOL v2.2 — Amendment for Unattended CC Execution

> Status: DRAFT for review. Lands on top of v2.1.2 (2026-05-17).
> Purpose: mechanise the remaining chat-judgement carve-outs in steps 1-5, close the Step 4 max-F1 fallback gap surfaced by Arc 7, and make explicit the live-execution equivalence asserted across the pipeline. CC can now run arcs 1→5 unattended in parallel without analyst sign-off mid-arc.
> Scope: governance + protocol mechanisation only. No engine change. No methodology change. No threshold change.
> Anchor preservation: KH-24 K=4 archetype 3 unaffected — none of the items being mechanised were invoked on the anchor under v2.0 or v2.1.x.

---

## §0. What v2.2 changes from v2.1.2

| Dimension | v2.1.2 | v2.2 |
|---|---|---|
| §9 single-fold sign-flip | "chat-level judgement" override | No override. §9 gate 1 conjunctive without exception. Diagnostic logged. |
| §12 Tier 2 lift candidates | "stack freely" | ≤ 5 candidates per archetype, intersection-only |
| Arc selection | implicit (analyst picks) | FIFO over `results/ARC_QUEUE.md`; supports registry entries AND standalone signal-spec docs |
| Closure disposition (steps 1-5) | KILL / HALT analyst-decided | Mechanical rule: §16a |
| Mid-arc analyst sign-off | Implicit at multiple points | Explicit: none required between arc-open and end of step 5 |
| §8 Step 4 threshold sweep failure | Implementation-defined (Arc 7: max-F1 fallback) | No fallback. Archetype dies at Step 4. |
| Live-execution equivalence | §1.2 covers entry timing | §1.2 carried forward; new §1a explicit assertion for steps 1 and 6 |

These changes close every named chat-judgement carve-out in steps 1-5 plus the Arc 7 Step 4 gap. Step 6 ship decision remains analyst-reviewed because it commits a system to deploy or archive — that review is deliberate, not a carve-out.

Rationale: under serial execution analyst-judgement carve-outs are cheap. Under parallel CC execution they become serialisation bottlenecks. Every carve-out mechanised here was historically resolved the way the new rule resolves it. The two non-carve-out additions (max-F1, live-execution) close real gaps surfaced in Arc 7 + the PR2 + spread-fix landings.

---

## §1. §9 single-fold sign-flip mechanisation

### Amended text

§9 gate 1 (sign consistency) is conjunctive without override. If any fold has `final_r_mean ≤ 0`, archetype fails §9 and dies at step 5.

The v2.1.2 carve-out ("Single-fold-outlier sign flips → chat-level judgement") is removed. Replaced by mandatory diagnostic logging:

When an archetype dies on §9 gate 1, the closure doc records:
- Per-fold `final_r_mean`, `n_archetype_in_fold`, `t_stat`
- Flag any fold with `n_archetype_in_fold < 10` as "thin-fold flip" — informational only, does not change disposition
- Per-fold contribution to overall final_r_mean

### Why

The carve-out was a "but the failing fold was tiny" reasoning path. Never invoked in a closed arc — every archetype that died on sign consistency also failed another gate. Removing it loses nothing.

If a future arc surfaces a thin-fold flip killing a structurally clean archetype, the mandatory diagnostic captures it for the next cross-arc calibration cycle.

### Anchor preservation

KH-24 K=4 archetype 3 — all 7 folds positive under v2.0 evaluation. No interaction.

---

## §2. §12 Tier 2 lift candidate cap

### Amended text

§12 Tier 2 — lifting above the gate:

Once Tier 1 succeeds with a single classifier, additional classifiers may be stacked as lift candidates, subject to:

1. **Cap:** ≤ 5 lift candidates per archetype per arc
2. **Intersection-only:** lift admits trade iff baseline AND lift classifier both admit
3. **Validation:** at step 6, lift must improve WFO worst-fold ROI by ≥ 1% vs baseline without DD degradation
4. **Fallback:** baseline (Tier 1 single classifier) always retained — lift either replaces or doesn't, never modifies

Candidate selection within the cap: top-5 feature-subset classifiers ranked by 5-fold CV AUC on hold-out data, excluding the Tier 1 baseline's feature subset.

### Why

"Stack freely" under v2.1.2 was unbounded once Tier 1 cleared. Under parallel CC execution this fans out arbitrarily. No archetype has tested more than 3 lift candidates historically — cap at 5 is bounded permission without practical constraint.

### Anchor preservation

KH-24 anchor measured under §14 — no Tier 2 lift evaluated. No interaction.

---

## §3. §8 Step 4 threshold sweep failure rule

### Amended text

§8 Pipeline assignment & artefact production, augmented:

For both Pipeline E and Pipeline D1 threshold sweep:

> Sweep threshold {0.40, 0.50, 0.60, 0.70}; select max precision with recall ≥ 0.60.

**If no threshold in the sweep satisfies recall ≥ 0.60, the archetype fails Step 4.** No max-F1 fallback. No "best effort" threshold selection. Archetype dies, closure doc per §16a.

This rule applies to:
- Pipeline E threshold sweep at gate-clearance
- Pipeline D1 threshold sweep at gate-clearance
- Tier 2 lift candidate threshold sweep (each lift candidate must independently pass)

### Why

Arc 7 closure: Pipeline D1 classifier nominally cleared AUC ≥ 0.60, but the threshold sweep produced no threshold satisfying recall ≥ 0.60. Implementation fell back to max-F1, selecting a threshold that admitted 3-4 total trades. A classifier admitting <10 trades is not a system; it's noise dressed up as a signal.

The recall ≥ 0.60 constraint is structural — it captures "the classifier admits a meaningful fraction of the cohort's trades." Failing that constraint means the cohort + classifier + threshold combination doesn't constitute a deployable filter. Allowing max-F1 fallback was implementation drift, not protocol intent.

The fix is one sentence: "no fallback." Step 6 trade-count floors (≥15 trades/fold for pass-deployable, ≥5 for pass-viable) are downstream — they would have killed Arc 7 at WFO anyway. v2.2 just kills it earlier and explicitly.

### Anchor preservation

KH-24 K=4 archetype 3, Pipeline D1 at t=3, RF AUC 0.638. Threshold sweep produces a threshold satisfying recall ≥ 0.60 (the cohort is large enough that admitting 60% of positive cases is mechanical). No interaction.

---

## §4. Arc selection — FIFO via state file

### Amended text

New §15b added to §15 operational requirements:

CC consults `results/ARC_QUEUE.md` at arc start. The file lists signal sources with status: `Unrun`, `Active`, `Closed-{KILL|HALT|SHIPPED}`.

Signal sources may be:
- **Registry entries** from `LCHAR_TOPN_REGISTRY.md` (Entry N) — discovered signals from the L characterisation atlas
- **Standalone signal-spec docs** under `signal_spec_<name>_v<version>.md` — analyst-designed signals (Arc 6 failed-breakout, Arc 7 liquidity sweep, etc.)

CC selects the topmost `Unrun` entry, transitions it to `Active` with a timestamp + branch name, opens the arc, and proceeds. On arc close, CC transitions the entry to `Closed-{disposition}` with closure doc path.

Multiple `Active` entries may coexist (parallel arcs from independent CC sessions). Each occupies its own branch and its own `results/arc_<N>/` folder.

Analyst override: analyst may reorder the Unrun section manually at any time (direct edit). CC reads the current state on each arc-open dispatch.

### Concurrency semantics

Git-level coordination. Each CC session pulls before queue edit; on push conflict, pulls again and retries with whatever Unrun entry is now topmost. Sufficient for dispatch cadences of minutes apart.

### File format

```markdown
# Arc Queue

## Active (in-flight)
- [ ] Arc <N>: <signal_name> — branch phase/arc-<N>, opened YYYY-MM-DD, live doc results/arc_<N>/ARC_<N>_LIVE.md
  - Spec: <registry entry N OR signal_spec_<name>.md>

## Unrun
- [ ] Arc <M>: <signal_name>
  - Spec: <registry entry M OR signal_spec_<name>.md>

## Closed
- [x] Arc <K>: <signal_name> — KILL/HALT/SHIPPED YYYY-MM-DD, results/arc_<K>/ARC_<K>_RESULT.md
```

### Why

Implicit arc selection ("analyst picks") doesn't scale to parallel execution. Explicit queue state lets CC pick deterministically and lets analyst rebalance priorities without per-arc coordination. Supporting both registry entries and standalone specs matches actual project practice (Arcs 6, 7, 8, 9 are standalone specs).

---

## §5. §16a — KILL vs HALT mechanical disposition rule

### Amended text

New §16a added to §16 governance:

When an arc fails at any of steps 1-5, CC writes the closure doc and assigns disposition mechanically per the rule below. No analyst review required for disposition assignment.

**HALT** iff ALL three conditions hold:

1. **Single criterion fail.** At the killing step, exactly one gate criterion fails. All other criteria at that step pass cleanly.
2. **Cohort viability.** `size_fraction_of_pool ≥ 0.10` (evaluated against the failing cohort: cluster or aggregate at step 3; archetype at steps 4-5 inheriting size from step 3).
3. **Near-miss OR strong-magnitude:**
   - **Path A (numeric near-miss):** failing criterion is numeric (monotonicity, frac_reach_1R, frac_wrong_way_pre_peak, fwd_mfe_p50, size_fraction, AUC, recall, size variance ratio, DD ratio) AND fails by margin < 0.03 absolute.
   - **Path B (categorical fail with strong magnitude):** failing criterion is categorical (shape_tag classification, sign consistency boolean) AND the cohort's `fwd_mfe_h240_p50 ≥ 3.0R` at the cluster's evaluated SL.

**KILL** otherwise.

### Step 4 threshold sweep failure under §16a

When Step 4 fails because no threshold satisfies recall ≥ 0.60 (per §3 of this amendment):
- The failing criterion is `recall ≥ 0.60`. Treat as a numeric criterion.
- Margin = (0.60 − best_recall_achieved). If margin < 0.03 → Path A near-miss eligible for HALT.
- Otherwise KILL.

### Closure doc requirements

**HALT closures** include a `## Cross-arc calibration candidate` section enumerating:
- Failing criterion and margin (or categorical failure mode)
- Magnitude evidence: size_fraction, fwd_mfe_h240 percentiles (p25/p50/p75/p95), final_r distribution, t-stat
- Suggested calibration item type: floor recalibration / new admit condition / measurement definition / etc.
- Reference to any matching open Open-N items in §16

These accumulate as input for the next protocol amendment cycle. Analyst reviews in batch (per §12 cross-arc governance), not per-arc.

**KILL closures** do not include this section. They record the failure mode cleanly and archive.

### Why

HALT vs KILL has been analyst-decided across every closed arc to date. The pattern is consistent. The 0.03 numeric threshold matches the historical near-miss bandwidth (KH-24 v2.0 c4 monotonicity miss by 0.020, Arc 2 redo c2 wrong_way miss by 0.0051). The 3R magnitude threshold matches the Arc 3 Stepwise lower bound (mfe_p50 3.34R). The single-criterion clause keeps HALT high-signal — multi-criterion failures are rationalisation candidates and excluded.

### Anchor preservation

KH-24 anchor doesn't trigger this rule — it passes all gates under v2.0 evaluation. No interaction.

---

## §6. §13 — No mid-arc analyst sign-off

### Amended text

New clause appended to §13:

Between arc-open and end of step 5, CC requires no analyst sign-off, no dispatch, no review. The single CC dispatch (per `prompts/cc_arc_orchestrator_template.md`) executes steps 1-5 end-to-end and halts only at:

- End of step 5 (PASS) — returns halt summary, awaits step 6 WFO dispatch from chat
- Mid-step KILL — closure doc per §16a, queue update, dispatch ends
- Mid-step HALT — closure doc per §16a with calibration candidate section, queue update, dispatch ends
- Engine-touching boundary breach — halt with halt summary, no commits to engine paths, dispatch ends
- Protocol ambiguity — halt with halt summary citing clause, dispatch ends

Chat does NOT review mid-arc state. Chat reviews:
- Step 5 halt summaries (for step 6 dispatch decisions)
- HALT closure docs in batch (for cross-arc calibration cycles)
- KILL closure docs by skim (informational)

### Why

Explicit declaration prevents drift. Without this clause, an over-cautious CC implementation might prompt chat at step 2 K-selection or step 3 archetype routing. Those decisions are mechanical, CC owns them.

---

## §7. §1a — Live-execution equivalence

### Amended text

New §1a inserted after §1 methodology principles, before §2:

**Steps 1 and 6 must execute trades under live-equivalent semantics.** This is an assertion, not a behaviour change — the Python backtester already implements live-equivalent execution per `SPREAD_SEMANTICS_LOCK.md` and the v1.3 forward-window extension. The assertion is that arcs may not deviate.

Required for Step 1 (plumbing) and Step 6 (WFO):
1. **Entry timing.** Signal at bar t close → entry at bar t+1 open. No same-bar entries. No mid-bar entries. Per §1.2.
2. **Spread costs.** Real per-bar MT5 bid/ask spread from the execution bar (t+1 for next-open entry, t for intrabar SL/TS). Fallback to `configs/spread_floors_5ers.yaml` only when raw spread = 0. No hardcoded defaults. Per `SPREAD_SEMANTICS_LOCK.md`.
3. **Stop / trail / TP execution.** Intrabar SL/TS triggers on mid; fills on bid/ask using execution-bar spread. No synthetic mid-fills. Per `SPREAD_SEMANTICS_LOCK.md` §"Intrabar stop exits".
4. **D1 features.** One-day lag (`merge_asof` backward, pre-shifted date). Same-day D1 close is not available intraday. Per §1.4.
5. **Volume veto.** Veto-on-volume produces no entry, no trade row, no spread. Per `SPREAD_SEMANTICS_LOCK.md` §"Spread sourcing".

Step 6 additionally must:
6. **Apply §11 archetype-specific exit policy.** Per-archetype MFE-lock / TP1 + trail / custom trail distances / etc. — executable in the engine post-PR-#131 + per-archetype-§11-exit-policies PR. Backtester is source of truth; EA mirrors. Per §1.12.

Step 1 does NOT apply §11 exit policies — at Step 1 the archetype assignment is unknown (clustering happens at Step 2, archetype routing at Step 3). Step 1 uses a uniform simulation-SL (default 2.0×ATR, or arc-specified default) to construct the pool. SL sweep at Step 3 re-evaluates pool trades under candidate SLs using the v1.3 forward-window extension (per §15a).

### Why explicit

Two recent landings make this assertion actionable:
- **PR #131 + the per-archetype §11 exit policies PR** (PR2) — Pipeline D1 with §11 archetype-specific exit policies now executes in the engine. Step 6 WFO can faithfully evaluate the live execution path.
- **Spread fix** — real per-bar bid/ask spread costs are now enforced in the backtester; the `spread_floors_5ers.yaml` is reduced to a fallback floor when raw spread = 0.

Pre-PR2 + pre-spread-fix, Step 6 had latent gaps where the evaluation diverged from the live execution path. Those gaps are closed. v2.2 makes the contract explicit so CC cannot silently re-introduce them.

### Anchor preservation

KH-24 K=4 archetype 3 evaluation under v2.0 already complied with all six requirements. No interaction.

---

## §8. What v2.2 does NOT change

Explicit non-changes, to keep amendment scope clean:

- No §2 floor changes
- No §6 K-selection or clustering changes
- No §7 SL sweep or capturability composite changes
- No §8 extractability AUC threshold or feature-budget changes (only the threshold-sweep-failure rule is new)
- No §10 multi-cluster ship decision changes — step 6 dispatch from chat remains the single analyst-review checkpoint
- No §11 archetype map changes
- No §14 anchor changes
- No engine change of any kind
- No data change
- No CI gating change

v2.2 is a seven-amendment governance + calibration pass. Methodology unchanged.

---

## §9. Migration

| Action | Owner | Required |
|---|---|---|
| Land v2.2 as PR to `L_ARC_PROTOCOL.md` | Engineering (Cursor) | Yes — protocol doc is PR-required scope |
| Create `results/ARC_QUEUE.md` initial state | Chat | Yes — direct-to-main |
| Create `prompts/cc_arc_orchestrator_template.md` | Chat | Yes — direct-to-main |
| Update STATUS.md / SESSION_ZERO.md (Arc 4-7 closure batch + v2.2 active) | Chat | Yes — direct-to-main |
| Re-evaluate past arcs under v2.2 | N/A | No — v2.2 applies to Arc 8 onward |

---

## Document control

| Field | Value |
|---|---|
| Version | v2.2 (DRAFT) |
| Supersedes | v2.1.2 (2026-05-17) |
| Active for arcs | Arc 8+ |
| Methodology change | None |
| Engine change | None |
| Anchor preservation | Verified — none of the seven items were invoked on KH-24 anchor under v2.0/v2.1.x |
| Drafted | 2026-05-18 |
| Companion files | `prompts/cc_arc_orchestrator_template.md`, `results/ARC_QUEUE.md` |

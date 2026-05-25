# Amendment 5 — AUC-Gated A2/A6 Architecture Selection

> **Status:** Ratified 2026-05-23. Landed into `L_PROTOCOL.md` §2 Step 5 architecture-selection rule.
> **Supersedes:** Amendment 1 "Architecture-selection rule per Step 3 archetype" (partial — A2/A6 selection logic only; A3/A4 archetype gating preserved)
> **Trigger:** Two-instance empirical evidence (Arc 7 v3.0 c0 Bimodal AUC 0.6758; Arc 7 v3.0.1 c1 Unclassified AUC 0.6642) of architecture-map gap producing false-negative arc verdicts
> **Engine impact:** None. All six architectures wired post-PR-186. Selection is dispatch-time logic.

---

## Section 1 — Diagnosis

L_PROTOCOL Amendment 1 introduced archetype-to-architecture mapping, gating each architecture's evaluation on Step 3 archetype tag. The rule conflated two structurally different architecture categories:

**Category A — Shape-required architectures (mechanism depends on path shape):**
- **A3 (deferred entry)** — Waits 3-5 bars, classifies dip-then-recover pattern. Mechanism requires a dip phase. Running on Monotonic-up = no dip exists, late entry costs.
- **A4 (path-classifier exits)** — Exits when path classifier loses confidence. Designed for Stepwise/Bimodal give-back patterns. Running on V-shape = exits during dip phase right before recovery.

For Category A, archetype gating is methodologically correct. Running on mismatched shapes produces misleading FAIL artefacts that pollute cross-arc evidence.

**Category B — Classifier-driven architectures (mechanism depends on classifier quality):**
- **A2 (admit/reject classifier filter)** — Consumes Step 4 classifier output, admits trades above threshold, rejects below.
- **A6 (confidence-driven sizing)** — Consumes Step 4 classifier output, scales position size by predicted probability.

For Category B, the architecture's mechanism is path-shape-agnostic by construction. The only relevant input quality measure is the classifier's predictive power — measured by AUC. Gating these architectures on archetype tags is a measurement-mechanism mismatch: using shape tags to make decisions that should depend on classifier quality.

The original protocol-design conflated these categories under uniform archetype gating. Empirical surfacing has now confirmed the gap produces false-negative arc verdicts (Arc 7 v3.0 + v3.0.1).

## Section 2 — Amended architecture selection rule

The Amendment 1 architecture-selection rule is replaced with the following four-gate procedure. A cluster's architecture set equals the UNION of architectures qualifying under all gates.

### Gate 1 — Shape-required architectures (archetype-driven)

| Archetype | Architectures added by Gate 1 |
|---|---|
| V-shape recovery | A3 |
| Stepwise climber | A4 |
| Bimodal | A4 |
| Monotonic up | (none) |
| Monotonic down | (none) |
| Choppy | (none) |
| Unclassified | (none) |

Gate 1 logic unchanged from Amendment 1 with respect to A3/A4 selection.

### Gate 2 — Classifier-driven architectures (AUC-driven)

If Step 4 mean OOS AUC ≥ 0.65 for the cluster: add A2 and A6.

The 0.65 threshold matches Amendment 1's classifier deployability bar (same number, same purpose: classifier predictive power sufficient to act on).

Gate 2 fires REGARDLESS of archetype. A2 and A6 are now archetype-agnostic.

### Gate 3 — Universal architecture

Always add A1 (system-level filter).

A1 has no shape or classifier dependence; baseline architecture for every cluster.

### Gate 4 — Portfolio architecture

If two or more candidate clusters survive Step 3 within the arc: add A5 (portfolio composition).

A5 selection logic unchanged from Amendment 1.

### Cluster skip condition

**Choppy clusters: skip all architectures.** Step 5 evaluation not performed. Choppy archetype indicates no structural edge worth extracting under any architecture.

## Section 3 — Examples

Worked examples for cluster classification under Amendment 5:

| Cluster | Archetype | Step 4 AUC | Architectures (Gates 1+2+3) |
|---|---|---|---|
| V-shape recovery | V-shape | 0.72 | A1, A3, A2, A6 |
| V-shape recovery, weak classifier | V-shape | 0.55 | A1, A3 |
| Stepwise, strong classifier | Stepwise | 0.68 | A1, A4, A2, A6 |
| Stepwise, weak classifier | Stepwise | 0.52 | A1, A4 |
| Bimodal, strong classifier | Bimodal | 0.70 | A1, A4, A2, A6 |
| Bimodal, weak classifier | Bimodal | 0.55 | A1, A4 |
| Monotonic up, strong classifier | Monotonic up | 0.75 | A1, A2, A6 |
| Monotonic up, weak classifier | Monotonic up | 0.50 | A1 |
| Unclassified, strong classifier | Unclassified | 0.66 | A1, A2, A6 |
| Unclassified, weak classifier | Unclassified | 0.50 | A1 |
| Choppy | Choppy | anything | (skip all) |

## Section 4 — What does NOT change

- **Verdict gates** — Amendment 3 risk-normalised gates apply unchanged to all architectures evaluated under Amendment 5
- **Failure-mode taxonomy** — no new failure modes
- **A3, A4 archetype gating** — preserved; running on mismatched shapes still prohibited
- **Step 4 classifier methodology** — unchanged; AUC computed per existing protocol
- **Step 5 search space per architecture** — unchanged per-architecture config grids
- **Closure template** — no schema change; `architectures_tested` field already captures the actual set
- **Engine code** — zero changes required; all six architectures wired post-PR-186

## Section 5 — Dispatch template updates

Arc dispatches generated post-Amendment 5 must apply the four-gate selection rule when listing architectures to test. Specific updates:

1. **Wave 2 dispatches** (in flight at time of ratification) — must reflect Amendment 5 from first dispatch
2. **Phase 2 dispatches** — same
3. **Discovery probe follow-up arcs** — same
4. **Sub-protocol arcs (heavy_ml_probe, etc.)** — same

Dispatches issued under prior protocol versions are not retrofitted; see Section 7 for retroactive policy.

## Section 6 — Compute cost

Amendment 5 typically adds A2 (1-6 configs depending on threshold grid) and A6 (12-18 configs depending on sizing curve grid) per qualifying cluster.

Net effect on Step 5 search space per arc:
- Arcs with no Unclassified / Monotonic / Bimodal clusters at AUC ≥ 0.65: zero change
- Arcs with 1-2 such clusters: 20-50% increase in Step 5 configs
- Arcs with 3+ such clusters: up to 70% increase

Wave 2 arcs (6 arcs in parallel) compute increase: bounded by individual-arc factor, no cross-arc compounding.

Acceptable cost. The compute spent is exactly the compute that should have been spent under Amendment 1's design intent.

## Section 7 — Retroactive policy

Closed arcs are audited for the criterion: "FAIL verdict closed with at least one Step 3 surviving cluster whose Step 4 AUC ≥ 0.65 was not evaluated under A2 AND A6 under prior protocol."

| Arc | Status | Action |
|---|---|---|
| Arc 7 v3.0.2 | In flight under dispatch-scoped Amendment 5 override | Closes Amendment 5's empirical test case |
| Arc 8 v3.0 (closed FAIL) | Audit at Amendment 5 ratification | Retry only if criterion met |
| Arc 8 v3.0.1 (in progress) | Audit at closure | Apply Amendment 5 from current dispatch |
| Arc 10 v3.0 (closed PROVISIONAL) | V-shape mapping already includes A6 | Likely unaffected; verify at audit |
| Arc 10 v3.0.1 (queued post-CC_18 + CC_19) | Apply Amendment 5 from first run | Includes Amendment 5 by sequence |
| Arc 11 v3.0 (closed FAIL) | Audit at Amendment 5 ratification | Retry only if criterion met |
| Arc 11 v3.0.1 (in progress) | Audit at closure | Apply Amendment 5 from current dispatch |

**Retroactive retries are not blanket.** Audit each closure; retry only where the criterion materially applies. If Arc 7 v3.0.2 produces a verdict change (FAIL → PASS-*), expand retroactive scope confidence. If Arc 7 v3.0.2 confirms FAIL despite Amendment 5, the protocol change is still methodologically correct but doesn't unlock prior arcs.

Arc 7 v3.0 (the original Bimodal c0 instance) is queued for retroactive audit post-Amendment 5 ratification. Arc 7 v3.0.1 (Unclassified c1) is superseded by in-flight Arc 7 v3.0.2.

**Operational ordering (per landing decision 2026-05-23):** Arc 8 v3.0 and Arc 11 v3.0 retroactive audits are queued in `TODO.md` but deferred until Arc 7 v3.0.2 closes. Arc 7 v3.0.2's verdict informs whether retroactive retries are worth running at all.

## Section 8 — Ratification

Amendment 5 ratified by mastermind decision dated **2026-05-23**.
Author: mastermind chat with Phase 1 chat empirical evidence.
Effective: from ratification timestamp for all new dispatches.
Engine PR required: none.
Documentation updates landed alongside:
- L_PROTOCOL.md (this section)
- Closure template v1.3.1 (`architectures_skipped_by_amendment_5` field added)
- Parser v1.3 (optional field acceptance + Phase 2 enforcement for post-ratification PASS)
- TODO.md (retroactive audit queued item)
- Wave 2 dispatch templates (Phase 1 chat-side; separate from this PR)
- Discovery probe follow-up dispatch templates (Phase 1 chat-side; separate from this PR)

## Section 9 — Implementation discipline

The four-gate selection rule is enforced AT DISPATCH TIME, not at engine time. Arc dispatches must:

1. Explicitly list the architectures-tested set per cluster
2. Cite the gate that admitted each architecture (e.g. `A1: Gate 3; A3: Gate 1 V-shape; A2, A6: Gate 2 AUC=0.72`)
3. Include a Step 4 AUC summary in dispatch preamble for gate-2 transparency

Engine receives a fully-resolved architecture set; engine does NOT apply the selection rule itself. This separation preserves engine determinism and allows the rule to evolve without engine PRs.

## Section 10 — Audit trail

Each arc closure §1 tracker_payload records:
- `architectures_tested` (set of architecture IDs) — existing
- `architectures_skipped_by_amendment_5` (set; empty list if none) — new informational field for cross-arc analytics

No schema change to closure template structure beyond the new optional field; `architectures_tested` already exists.

`architectures_skipped_by_amendment_5` rollout follows the Amendment 3 field pattern:
- **Phase 1:** OPTIONAL on all closures (parser accepts presence or absence).
- **Phase 2 (post-ratification):** REQUIRED for any PASS verdict whose `closed_timestamp > <PR-merge timestamp>Z`. Closures landed before the PR-merge timestamp are grandfathered.

The exact cutoff timestamp is the merge timestamp of the PR landing this amendment, backfilled into `scripts/tracker_parser/schema.py::AMENDMENT_5_CUTOFF_ISO` post-merge. The placeholder value pinned at landing is `2026-05-23T00:00:00Z` (start of ratification day UTC) — functionally equivalent for all practical purposes since the PR cannot merge before that instant and no PASS closure will have a `closed_timestamp` between `2026-05-23T00:00:00Z` and the actual merge timestamp.

End of Amendment 5.

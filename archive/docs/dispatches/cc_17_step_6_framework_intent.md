# CC_17 — Step 6 Framework Intent

> **Branch:** `engine/step-6-causal-audit-framework`
> **Authored:** 2026-05-24 by CC_17 from `CC_17_STEP_6_FRAMEWORK.md` Read-first phase.
> **Status:** intent only. No engine code written. Chat review required before implementation begins.

---

## Read-first artefacts consumed

| Source | Take-away that shapes this intent |
|---|---|
| [L_PROTOCOL.md](L_PROTOCOL.md) §2 Step 6 | Sparse: trigger = ≥ 1 PASS-tier candidate; mechanics = producer-trace + byte-compare; downgrade or kill candidate. No category framework. |
| [L_PROTOCOL.md](L_PROTOCOL.md) §3 Evaluation order | Step 6 is constraint #10 — runs ONLY after #1-9 clear. `step6_causal_audit_fail` already in the failure-mode taxonomy at priority 11. |
| [docs/audits/engine_capability_audit_2026_05.md](docs/audits/engine_capability_audit_2026_05.md) §"Step 6 — Causal audit" | All 5 audited items MISSING: producer-trace, byte-compare, D1-lag check, auto-trigger, manifest format. Tier 4 of the Recommendations. |
| [core/wfo/amended_gates.py:248](core/wfo/amended_gates.py:248) | `classify_amended_fold_stats(..., causal_audit_clean: bool = True)` is the plumbed hook — Step 6 already has a callsite shape; today the orchestrator never passes the parameter. |
| [core/arc/arc_orchestrator.py:818](core/arc/arc_orchestrator.py:818) | Today's Step 6 = `"(lazy — deferred to chat at PASS verdict)"`. `ArcConfig.invoke_step_6: bool = False` at [:176](core/arc/arc_orchestrator.py:176) is never flipped. |
| [core/arc/arc_orchestrator.py:779-836](core/arc/arc_orchestrator.py:779) | Amendment 3 evaluation runs per top-K candidate, picks best by verdict rank. Step 6 dispatch lands AFTER `_run_amendment_3_evaluation`. |
| [docs/templates/ARC_CLOSURE_TEMPLATE.md](docs/templates/ARC_CLOSURE_TEMPLATE.md) v1.2.1 | §1 tracker_payload has no `step_6` block today. Version bump path v1.2 → v1.2.1 (additive `chained_dd_method` field). Dispatch wants v1.3. |
| [scripts/tracker_parser/schema.py:41](scripts/tracker_parser/schema.py:41) | Schema detection precedence template_version → v1.2-exclusive fields → v1.1-exclusive fields → v1.0. Detection function is the v1.3 extension point. |
| [scripts/l_arc_10_v3/step_6_byte_compare.py](scripts/l_arc_10_v3/step_6_byte_compare.py) (250 LOC) + [results/l_arc_10/step_6/](results/l_arc_10/step_6/) | Arc 10's hand-written precedent. Manifest schema is informal; report is prose. Useful template, NOT directly reusable as a feature-producer-agnostic harness. |
| [ARC_TRACKER.md](ARC_TRACKER.md) | No "Step 6 audit registry" section today. The new section (Task 6) is a fresh append at line ~175 (after "Cross-arc tag registry"). |

---

## Component map + LOC estimates

> All LOC numbers are budget targets. Final counts will vary ±20%.

### New engine modules (`core/step_6/`)

| File | LOC | Role |
|---|---:|---|
| `__init__.py` | 30 | Public exports (orchestrator, CategoryAuditResult, CheckResult). |
| `manifest.py` | 180 | `CategoryAuditResult`, `CheckResult`, `Step6Manifest` dataclasses; severity enum; manifest.json serde + sha256. |
| `orchestrator.py` | 250 | `run_step_6(arc_result, audit_config) -> Step6Result` — dispatches the 6 categories, applies severity rules, decides verdict_impact. |
| `lookahead.py` | 280 | §6.1 — per-feature producer-trace + D1 lag-rule + cluster-feature audit + threshold-selection lineage. |
| `selection_bias.py` | 220 | §6.2 — Bonferroni denominator from `pool_metadata.configs_evaluated_step5`; holdout-touch detector; cluster-selection justification. |
| `execution_realism.py` | 300 | §6.3 — spread regime delta vs broker; fill realism (M1 next-bar-open feasibility); lot-rounding; margin/leverage at `r_safe`; mid-price refactor + UTC bar boundary checks. |
| `statistical.py` | 240 | §6.4 — Lo-corrected Sharpe sample-size check; pair-set survivorship; vol-regime coverage; cross-pair correlation. |
| `determinism.py` | 180 | §6.5 — verify on-disk `sha256_manifest.json` matches recomputed sha256 of Step 4 + Step 5 artefacts; verify seed pinning in producer code (NOT a sim re-run — see §"Open questions" Q5). |
| `deployment_readiness.py` | 200 | §6.6 — §4 deployment_spec field-by-field validator; feature live-computability; latency-tolerance metadata; restart-safety metadata; `config_artefact_path` self-containment. |
| `byte_compare.py` | 300 | Generic feature-producer-agnostic harness (factored from Arc 10's `step_6_byte_compare.py`). Iterates over registered `FeatureSpec` set, samples N trades, recomputes from raw OHLC, byte-compares to pool. |
| `artefacts.py` | 240 | Per-category markdown writers + `summary.md` + `sha256_manifest.json`. |
| **Engine subtotal** | **2,420** | |

### CLI (`scripts/`)

| File | LOC | Role |
|---|---:|---|
| `scripts/run_step_6.py` | 180 | Manual invocation per Task 7 — flags: `--category`, `--no-block`, `--dry-run`; outputs to `results/<arc>/step_6_manual_<timestamp>/`. |

### Tests (`tests/step_6/`)

| File | LOC | Role |
|---|---:|---|
| `__init__.py` + `conftest.py` + `_fixtures.py` | 200 | Synthetic arc-result fixtures; minimal pool + step-1/4/5 artefacts for fast unit tests. |
| `test_lookahead.py` | 200 | Per-feature clean / leaky / D1-same-day / threshold-from-full-data cases. |
| `test_selection_bias.py` | 150 | Bonferroni denominator math; holdout-touch detection. |
| `test_execution_realism.py` | 200 | Spread delta thresholds; lot-rounding at broker minimums; mid-price/UTC checks. |
| `test_statistical.py` | 150 | Lo-corrected Sharpe arithmetic on toy returns; pair-set survivorship. |
| `test_determinism.py` | 120 | sha256_manifest match + mismatch cases. |
| `test_deployment_readiness.py` | 200 | §4 deployment_spec field validation + missing-field FAIL paths. |
| `test_byte_compare.py` | 150 | Harness on a 2-feature synthetic producer; happy path + drift detection. |
| `test_orchestrator.py` | 250 | End-to-end synthetic arcs covering Task 10 cases 1-6 (PASS clean, PASS leaky → downgrade, FAIL with manual invocation, --no-block, two-run determinism, schema-validation HALT). |
| `test_manual_cli.py` | 200 | CLI invocation matrix; exit codes; output directory layout. |
| **Test subtotal** | **1,820** | |

### Modified files

| File | LOC delta | Change |
|---|---:|---|
| `core/arc/arc_orchestrator.py` | +120 | Step 6 dispatch after `_run_amendment_3_evaluation`; verdict re-classification when Step 6 FAILs; persist `Step6Manifest` to `results/<arc>/step_6/`; populate the existing `invoke_step_6` flag (currently dead code at [:176](core/arc/arc_orchestrator.py:176)). |
| `core/wfo/amended_gates.py` | +20 | No structural change. Document the `causal_audit_clean` callsite invariant. Adjust `_fail` reason text for `STEP6_CAUSAL_AUDIT_FAIL` to surface which category failed. |
| `core/arc/_closure_template.py` | +60 | Emit §1 tracker_payload `step_6` block keyed off `Step6Manifest`. |
| `L_PROTOCOL.md` | +150 | §2 Step 6 expanded from sparse spec (lines 333-354 today) to six-category framework per Amendment 4. §3 PASS-* conditional on Step 6. |
| `docs/templates/ARC_CLOSURE_TEMPLATE.md` | +80 | v1.2.1 → v1.3 (see §"Open questions" Q8). New §1 `step_6` block schema. v1.3 row in the Schema versioning table. |
| `archive/L_PROTOCOL_v3_0_AMENDMENT_4.md` | +250 (new file) | Full Amendment 4 text. Authored by mastermind chat per Task 8, applied here. |
| `ARC_TRACKER.md` | +30 | New "Step 6 audit registry" section after "Cross-arc tag registry" (line ~175). Append-only per arc, auto-PASS dispatch only. |
| `scripts/tracker_parser/schema.py` | +80 | `BestArchitectureV13`, `TrackerPayloadV13`, `step_6` block model. `detect_schema_version` v1.3 path. Phase 2 tightening per Task 9. |
| `scripts/tracker_parser/mapping.py` | +50 | New Section 4-M: Step 6 audit registry mutation. |
| `scripts/tracker_parser/extract.py` | +20 | Step 6 block presence check helper (for v1.3 PASS validation). |
| `scripts/tracker_parser/README.md` | +60 | v1.3 schema notes; Phase 2 enforcement rules. |
| `docs/PROTOCOL_RUNTIME.md` | +80 | New §"Step 6 framework" section. |
| `docs/audits/engine_capability_audit_2026_05.md` | +20 | Footer note: Step 6 MISSING items → WIRED post-PR. |
| **Modified subtotal** | **+1,020** | |

### Grand total

| Bucket | LOC |
|---:|---:|
| New engine | 2,420 |
| New CLI | 180 |
| New tests | 1,820 |
| Modified | +1,020 |
| **Total** | **~5,440** |

Sits at the upper end of "multi-day (3-5 days CC)". 35% is tests, 19% is docs+schema — matches the dispatch's stated bias toward audit-grade rigour over feature volume.

---

## Open questions for chat review

These need resolution BEFORE the implementation phase begins. Each has a recommended default; chat may override.

### Q1 — Gate-pass / Step-6 ordering (CRITICAL)

[`classify_amended_fold_stats`](core/wfo/amended_gates.py:248) already takes `causal_audit_clean: bool = True`. Two readings of Task 1 ("After Step 5 verdict assignment, check if verdict starts with PASS-… automatically dispatch"):

- **(A) Pre-gate path:** Step 6 runs FIRST for every top-K candidate, then `classify_amended_fold_stats` is called with the real `causal_audit_clean` value. Cost: Step 6 runs even on candidates that fail #1-9 (waste).
- **(B) Post-gate path:** Amended gate runs first with `causal_audit_clean=True` default; if any candidate clears all of #1-9, Step 6 runs; if Step 6 FAILs, re-call the gate with `causal_audit_clean=False` to downgrade. Matches §3 "Evaluation order" item 2 exactly.

**Recommend (B)** — matches §3 ordering verbatim and avoids wasted Step 6 invocations on candidates that fail #1-9.

### Q2 — Step 6 per-candidate scope (CRITICAL)

Amendment 3 emits per top-K results. Top-1 is the verdict-carrier; Top-2/Top-3 are reported for selection-bias accounting. Dispatch says "automatically dispatch `run_step_6(arc_result)`" — singular arc_result, ambiguous on top-K.

- **Per-Top-1:** Only the verdict-carrying candidate gets audited. Top-2/Top-3 marked `step_6.ran = false`. Cheapest, matches "winning candidate's filter/classifier" language in §2 Step 6.
- **Per-top-K:** Every PASS-tier candidate audited. Expensive (byte-compare is heavy); produces redundant reports when Top-1/Top-2 share feature sets.

**Recommend Per-Top-1**, with a note in the manifest if Top-2 / Top-3 use a feature set disjoint from Top-1 (rare). Chat can override per-arc via CLI flag if a Top-2 candidate becomes interesting.

### Q3 — Scope creep beyond protocol §2 Step 6 (CRITICAL — Amendment 4 ratification)

L_PROTOCOL §2 Step 6 today specifies only producer-feature trace + byte-compare. Dispatch §6.2-§6.6 add five categories, several of which overlap existing engine paths:

| Dispatch category | Overlap |
|---|---|
| §6.2 Selection bias | Step 5 already emits `configs_evaluated_step5` + `search_scope_flag` (thin/normal/broad). Bonferroni accounting today in closure prose, not engine. |
| §6.3 Execution realism — spread delta | Step 1 integrity check `spread-floor activation rate per pair` is on the engine-capability-audit "MISSING" list — different check. Spread delta vs broker is genuinely new. |
| §6.3 Execution realism — lot rounding, margin | Belongs in closure template v1.2 §4 deployment_spec §4.8 by current template wording. |
| §6.5 Determinism | `tests/test_determinism.py` (CI) + per-step `sha256_manifest.json` already enforce this at engine-side. Step 6 re-runs would be expensive duplication. |
| §6.6 Deployment readiness | Closure template v1.2 §4 deployment_spec already has a "Deployment readiness checklist" with 6 items, parser-enforced via Section 4-L. |

Risk: Step 6 becomes a kitchen sink and no real-world PASS verdict ever clears it cleanly.

**Recommend** chat confirm Amendment 4 ratifies all six categories as Step 6 scope and clarifies the overlap rules (e.g., §6.6 deployment readiness READS the existing §4 deployment_spec rather than duplicating the checks). If full §6.2-§6.6 scope is too aggressive, propose narrower v1 = §6.1 lookahead + §6.6 deployment readiness only; defer §6.2-§6.5 to a follow-up amendment when Wave 2 produces the first auto-dispatched PASS.

### Q4 — Severity rules + warnings-only category (MINOR)

Dispatch §"Severity rules": `critical` fails block; `warning` fails flag-only; `info` records. Dispatch §"CategoryAuditResult.passed = AND of all check.passed" contradicts this — a check with severity=warning that fails would set `check.passed = false` and tank the category.

**Recommend** redefine: `CategoryAuditResult.passed = AND over only the critical-severity checks` (warnings tracked separately in `n_warnings`). Document explicitly in `manifest.py` docstrings.

### Q5 — Determinism category cost (MINOR)

Dispatch §6.5 says "Two-run sha256 reproduction on Step 4 + Step 5 artefacts". Full-arc Step 4 + Step 5 re-run is expensive (10+ min on a real arc). `tests/test_determinism.py` already covers this at mini-pipeline level in CI.

**Recommend** Step 6 determinism check = verify the existing `step_4/manifest.json` + `step_5/sha256_manifest.json` sha256 entries match a recomputed sha256 of the artefacts on disk (cheap; catches post-emission tampering). DO NOT re-run sims. The two-run sha256 guarantee is provided by CI, not by Step 6.

### Q6 — Arc 10 retroactive (MINOR)

Dispatch §"Backwards compatibility" says Arc 10 must run Step 6 manually if its PROVISIONAL verdict becomes definitive PASS. Arc 10 already has hand-written Step 6 outputs in [results/l_arc_10/step_6/](results/l_arc_10/step_6/) including a PASS verdict.

**Recommend** clarify in Amendment 4 §"Backwards compatibility": Arc 10's existing hand-written Step 6 stays canonical. If re-run under the framework via manual CLI, results write to `results/l_arc_10/step_6_manual_<timestamp>/` and do NOT clobber the existing audit. Arc 10's `template_version` stays at v1.2; no v1.3 retrofit required.

### Q7 — Parser v1.3 grandfathering date (COSMETIC)

Task 9 references "PR-186 merge date" as the timestamp for the v1.2 PASS Amendment-3-field-required cutoff. Need the literal ISO timestamp from `git log --format="%aI" -1 7c238e8^` (the merge commit predecessor; PR-186 is commit `1b15774`). Will resolve at implementation time, but flag now so chat is aware the cutoff date is a load-bearing parser constant.

### Q8 — Template version v1.3 vs v1.2.2 (COSMETIC)

Closure template is at v1.2.1 today (PR-186). Dispatch says v1.3. Amendment 4 is genuinely a bigger schema change than v1.2.1 (additive `chained_dd_method` field only). v1.3 is the right call — locks the schema for "Wave 2 from-scratch PASS arcs run Step 6 from first close."

**Recommend** confirm v1.3 (not v1.2.2). v1.3 row added to the Schema versioning table; v1.2.1 closures grandfathered.

---

## Risks beyond design questions

- **Risk #1 — Wave 2 dispatch coupling.** Engine audit Tier 4 lists Step 6 framework as separate from Tier 1 (Amendment 3 gate rewrite). Both PR-185 and PR-186 already landed; Wave 2 is not strictly blocked on Step 6. But once Step 6 lands, every PASS verdict gates on it — if Step 6 framework has bugs at landing, every Wave 2 PASS arc tanks. Recommend a buffer week between PR landing and Wave 2 dispatch to shake out false-positive critical fails.
- **Risk #2 — Hand-written Arc 10 Step 6 as test fixture.** Tempting to use Arc 10's existing outputs as a golden-test fixture. But Arc 10's manifest was hand-written and its format predates the spec the framework will lock. Will produce a synthetic golden fixture instead and treat Arc 10 as a separate manual-CLI-validation case.
- **Risk #3 — `core/step_6/byte_compare.py` generic harness vs producer signature variability.** Arc 10's byte-compare uses producer-specific imports (`signals.lchar_dlr_long`). A truly generic harness needs to introspect `FeatureSpec` producers and reconstruct their inputs from raw OHLC. Manageable for the v3.0 registered feature classes; future bespoke signal-side features will need to register a `step_6_recompute(panel, signal_row) -> dict` hook. Will add this to the protocol-level `SignalModule` interface as part of the engine PR (~30 LOC extra in `core/arc/signal_protocol.py`).

---

## What this intent does NOT cover

- A4 portfolio composition spec (separate amendment per dispatch §"After this lands")
- Heavy ML probe sub-protocol engine path (Phase 2 prerequisite per engine audit Tier 2)
- Full-window-sim chained DD method (`chained_dd_method: full_window_sim`, deferred per Amendment 3 chat directive Q6)
- Cross-platform CI determinism check (engine audit Tier 6 housekeeping)

---

End of intent. Awaiting chat review of the 8 open questions before implementation begins.

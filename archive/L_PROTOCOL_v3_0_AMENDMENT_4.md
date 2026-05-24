# L_PROTOCOL Amendment 4 — Step 6 Causal Audit Framework (v1, locked)

> **Date:** 2026-05-24
> **Status:** locked and landed into `L_PROTOCOL.md` §2 Step 6 + §3 on 2026-05-24.
> **Triggered by:** L_PROTOCOL v3.0 §2 Step 6 was sparse (producer trace + byte-compare). Engine capability audit 2026-05 surfaced five MISSING items: producer trace as runnable check, byte-compare from raw OHLC, D1 lag verification, auto-dispatch on PASS, manifest format. Arc 10's PROVISIONAL PASS-DEPLOYABLE made the gap concrete — the backtester said PASS, the EA tank ran post-closure.
> **Scope:** Step 6 framework as runnable engine code. Auto-dispatches on any candidate that clears §3 constraints #1-9 (post-gate per evaluation order). Manual CLI invokable on any closure regardless of verdict (read-only diagnostic). Six audit categories, severity rules, manifest schema, closure template v1.3 bump, parser v1.3 schema + Phase 2 tightening.
> **Supersedes:** L_PROTOCOL v3.0 §2 Step 6 (sparse 22-line spec). §3 verdict definitions clarified to make PASS-* conditional on Step 6 clean. Failure-mode taxonomy `step6_causal_audit_fail` retained from Amendment 3 §"Failure-mode priority" at priority 11.

---

## Core principle

Step 6 is the **causal audit** — it verifies that the system Step 5 produced is causally clean and ready for live deployment. It does NOT re-run the search or duplicate Step 5 work. It VERIFIES that recorded paths produced clean output, that no lookahead crept in, that the deployment spec is complete, and that the artefact set is reproducible.

Per chat resolution Q3: **audit = verification, not re-computation.** Several categories overlap existing engine paths (selection-bias accounting at Step 5, determinism in CI, deployment-readiness in §4 deployment_spec). Step 6's job is to verify those paths produced the right record, not redo the work.

---

## Trigger (post-gate per chat Q1)

Step 6 dispatches AFTER `classify_amended_fold_stats` (Amendment 3) clears §3 constraints #1-9 with `causal_audit_clean=True` (default) for at least one top-K candidate. Constraints #1-9 are evaluated FIRST per Amendment 3 §"Evaluation order"; Step 6 is constraint #10.

Auto-dispatch runs on the **Top-1 verdict-carrying candidate only** (chat Q2). When Top-2 / Top-3 use a feature set materially different from Top-1, Step 6 surfaces this as a `top_k_feature_set_divergence` warning on the lookahead category — chat may then opt to audit them manually.

Step 6 critical-failure downgrades the Top-1's verdict by re-classifying its amended gate with `causal_audit_clean=False`, which routes to `primary_failure_mode = step6_causal_audit_fail`. Other top-K candidates are NOT re-classified.

`skip_step_6: bool = False` on `ArcConfig` is the escape hatch for diagnostic / dev runs that don't need the audit.

---

## Six audit categories

Each category is a self-contained module under `core/step_6/` that exports `audit(inputs, audit_config) -> CategoryAuditResult`. Categories CANNOT depend on each other (each runs independently).

### §6.1 Lookahead

Producer-level feature trace + D1 lag rule enforcement + cluster-feature audit + byte-compare from raw OHLC + threshold-selection lineage.

Six checks (4 critical, 1 warning, 1 info):
- `per_feature_lineage_clean` (critical)
- `no_path_features_in_entry` (critical)
- `d1_lag_rule_enforced` (critical, static source check)
- `byte_compare_no_drift` (critical, N samples × M features)
- `threshold_selection_lineage` (warning)
- `feature_lineage_table_exists` (info)

### §6.2 Selection bias

Verifies Step 5's recorded selection-bias accounting: configs evaluated, Bonferroni-equivalent noise floor, holdout reuse detector, cluster-selection record.

Five checks (3 critical, 1 warning, 1 info).

### §6.3 Execution realism

Real-spread source (HistData M1 bid+ask), per-pair spread regime delta, fill realism, lot rounding at `r_safe`, mid-price refactor active, UTC bar boundary.

Six checks (4 critical, 1 warning, 1 info).

### §6.4 Statistical integrity

Per-fold + total trade count sufficiency, 28-pair survivorship, vol-regime coverage (≥ 3 calendar years), cross-pair daily-bucket correlation, Lo-corrected Sharpe with lag-1 autocorrelation correction.

Five checks (2 critical, 2 warning, 1 info).

### §6.5 Determinism

`step_4/classifiers/manifest.json` sha256 match against on-disk file hashes, required arc artefacts present, seed pinning across producer modules, LF line terminators in text artefacts. Per chat Q5: sha256-manifest verify ONLY — does NOT re-run sims.

Four checks (2 critical, 1 warning, 1 info).

### §6.6 Deployment readiness

`## §4 deployment_spec` heading present, `config_artefact_path` resolvable, every `§4.X` subsection (1-11) present, features live-computable per FeatureSpec registry, deployment-readiness checklist marked.

Five checks (3 critical, 1 warning, 1 info).

---

## Severity rules (chat Q4)

- `critical` failure → category FAIL → Step 6 FAIL → verdict downgrade (auto-dispatch only).
- `warning` failure → category PASS but flagged; recorded in `n_warnings` total.
- `info` failure → recorded; informational only.

`CategoryAuditResult.passed = AND over critical-severity checks only`. Warnings and info do not affect the category's pass/fail decision.

`--no-block` CLI flag demotes critical failures to warnings in the rendered report only; manual invocations never modify the verdict regardless.

---

## Manifest + artefact format

`results/<arc>/step_6/` per auto-dispatch:
- `manifest.json` — per dispatch Task 4 schema (arc_name, ran_at, trigger, overall_passed, verdict_impact, categories summary, report_paths, critical_failures, n_warnings)
- `summary.md` — one-page roll-up
- `<category>_report.md` × 6 — per-category check table + evidence
- `sha256_manifest.json` — per-file sha256 for two-run determinism comparison

`results/<arc>/step_6_manual_<timestamp>/` per manual CLI invocation — identical layout; never clobbers the auto-dispatched run (chat Q6).

---

## Closure template v1.3 — `§1 tracker_payload.step_6` block

```yaml
step_6:
  ran: <bool>
  trigger: auto_pass | manual | not_applicable
  overall_passed: <bool or null>
  manifest_path: results/<arc>/step_6/manifest.json
  categories:
    lookahead: <bool or null>
    selection_bias: <bool or null>
    execution_realism: <bool or null>
    statistical: <bool or null>
    determinism: <bool or null>
    deployment_readiness: <bool or null>
  critical_failures: [<list of "category.check_name" entries>]
  warnings_count: <int>
  verdict_impact: none | downgraded_to_fail
```

REQUIRED for any v1.3 PASS verdict. For FAIL / HALT closures the block records `ran: false` with all other fields null.

---

## Parser v1.3 — Phase 2 tightening (chat Q7)

Cutoff: `2026-05-23T06:20:59Z` (PR-186 merge).

For any PASS verdict with `closed_timestamp > 2026-05-23T06:20:59Z`:
- REQUIRE Amendment 3 fields in `best_architecture` (`chained_max_dd_base_pct`, `k_safe`, `k_hard`, `r_safe_pct`, `r_hard_pct`, `scalable_to_safe`, `scalable_to_hard`).

For `template_version: v1.3` PASS verdicts:
- ADDITIONALLY require the `step_6` block with `overall_passed: true`.

Pre-cutoff closures (v1.0/v1.1/v1.2/v1.2.1) grandfathered. Failed validation → parser HALT, blocking PR merge.

---

## §3 verdict definitions (clarification)

§3 PASS-DEPLOYABLE and PASS-VIABLE both require "Step 6 causal audit clean" (constraint #10 in both tiers — pre-existing in v3.0). Amendment 4 makes the framework concrete:

- Pre-Amendment-4: "Step 6 causal audit clean" was a manual gate, dispatched at chat's discretion.
- Post-Amendment-4: "Step 6 causal audit clean" = `Step6Result.overall_passed == True`, evaluated on the Top-1 candidate after §3 constraints #1-9 clear. Engine-enforced.

The `step6_causal_audit_fail` failure mode at priority 11 in §3 "Failure-mode priority" remains unchanged.

---

## Backwards compatibility (chat Q6)

- v1.0 / v1.1 / v1.2 / v1.2.1 closures (Arcs 8, 10, 11, all retroactives) grandfathered. Parser handles them under the existing detection precedence. No retroactive Step 6 required.
- Arc 10's PROVISIONAL PASS-DEPLOYABLE: existing hand-written Step 6 at `results/l_arc_10/step_6/` stays canonical. Re-running the framework via manual CLI lands in `results/l_arc_10/step_6_manual_<ts>/` and does NOT clobber. Arc 10's `template_version` stays v1.2; no v1.3 retrofit required.
- Future arcs (Wave 2 onward) close at v1.3; Step 6 auto-dispatches on PASS-tier candidates.

---

## Discipline rules

- Step 6 FAIL on an auto-dispatched candidate → MUST downgrade to FAIL. No override mechanism.
- Manual invocations are read-only — never modify the verdict.
- Severity rules locked: critical = blocking, warning = flagging, info = recording.
- Step 6 modules are engine-side; audit categories CANNOT depend on each other.
- All Step 6 outputs deterministic — same arc result → same Step 6 output bytes (modulo `ran_at` timestamp).

---

## Definition of done (PR scope)

- [x] Six audit category modules implemented with check functions
- [x] Auto-dispatch wired into orchestrator (post-gate per Q1, Top-1 per Q2)
- [x] Manual CLI (`scripts/run_step_6.py`) working
- [x] Closure template v1.3 with `step_6` block
- [x] Tracker schema extended (new "Step 6 audit registry" section)
- [x] Parser v1.3 detection + Step 6 block model + enum validation
- [x] Parser Phase 2 tightening (PR-186-merge-date cutoff + v1.3 PASS Step 6 requirement)
- [x] Amendment 4 landed in `archive/L_PROTOCOL_v3_0_AMENDMENT_4.md`
- [x] L_PROTOCOL.md §2 Step 6 expanded with Amendment 4 reference
- [x] L_PROTOCOL.md §3 verdict-Step 6 conditional clarified
- [x] Full test suite passing (existing + new tests/step_6/)
- [x] Documentation updated (PROTOCOL_RUNTIME.md, tracker_parser/README.md, engine_capability_audit footer)

---

## End

Amendment locked. Future Step 6 framework changes require explicit redesign event documented in chat.

# closure_template_v12_log.md

> **Dispatch:** `CC Dispatch — Closure Template v1.2 (deployment spec) + Retrofit Arcs 8, 10, 11`
> **Branch (worktree):** `claude/romantic-nash-219b96` — to be renamed `infra/closure-template-v1.2` at PR time per chat direction.
> **Intent doc:** [closure_template_v12_intent.md](closure_template_v12_intent.md)
> **Outcome:** all 7 tasks executed; parser dry-run on all three retrofitted closures exit 0; full tracker_parser test suite passes (60/60).

---

## Task-by-task summary

### Task 1 — Template v1.1 → v1.2

`docs/templates/ARC_CLOSURE_TEMPLATE.md` updated:
- Version header bumped v1.1 → v1.2 with the dispatch's verbatim summary line.
- Added `config_artefact_path` + `deployment_spec_section_present` fields to the `best_architecture` block in §1 YAML. Bumped `template_version` placeholder to `v1.2`.
- Inserted **§4 deployment_spec** template (4.1-4.11) verbatim from the dispatch between §3 and the Section 4 closure-→-tracker mapping checklist.
- Added **Section 4-L** mapping entry: parser-enforced PASS-verdict validation (config path + file + heading + flag).
- Section 5 parser notes updated to describe v1.2 detection precedence (1.2 → 1.1 → 1.0 → error).
- Schema-versioning table extended with the v1.2 row (date 2026-05-23, deployment-spec rationale).

### Task 2 — L_PROTOCOL.md §6 patch

`L_PROTOCOL.md` §6 "ARC_CLOSURE.md format" subsection replaced verbatim with the dispatch's 4-section spec (§1 / §2 / §3 / §4 conditional). v1.0 reference bumped to v1.2.

### Task 3 — Parser v1.2 support + tests

`scripts/tracker_parser/schema.py`:
- Module docstring updated to describe v1.2.
- `SchemaVersion` extended to `Literal["1.0", "1.1", "1.2"]`.
- Added `V12_EXCLUSIVE_FIELDS = {"config_artefact_path", "deployment_spec_section_present"}`.
- `detect_schema_version()` now returns `"1.2"` for `template_version v1.2/1.2` OR presence of v1.2-exclusive fields. Detection order 1.2 → 1.1 → 1.0.
- `VALID_VERDICTS` extended with `PASS-VIABLE-PROVISIONAL`, `PASS-DEPLOYABLE-PENDING-STEP6`, `PASS-VIABLE-PENDING-STEP6` (anticipating §4 trigger cases).
- New `BestArchitectureV12` (extends V11 with the two deployment-spec fields) + `TrackerPayloadV12` + `normalize_to_v12()`.
- Added `_coerce_legacy_field_names()` helper called from `parse_payload` when version detected is v1.2 — handles the retrofit case where a v1.2 closure retains v1.0-style `worst_fold_roi_pct` / `worst_fold_dd_pct` field names. Idempotent.
- `parse_payload()` extended with v1.2 dispatch branch.

`scripts/tracker_parser/extract.py`:
- Added `DEPLOYMENT_SPEC_HEADING_RE` and `has_deployment_spec_heading(closure_path)` helper for Section 4-L item 3.

`scripts/update_tracker_from_closure.py`:
- Added `_validate_v12_pass_verdict(payload, closure_path)` — runs the four Section 4-L checks in priority order, returns 0 on pass / 1 on any failure.
- Wired between schema validation and tracker mutations: if `template_version == "1.2"` AND `verdict` starts with `PASS-`, validation runs and HALTs before any tracker IO if it fails.

**New tests:**
- `tests/tracker_parser/test_schema_v12.py` — 9 tests covering v1.2 detection (by template_version, by exclusive field), preservation of deployment-spec fields through normalisation, backwards-compat for v1.0/v1.1, unknown template_version rejection, new verdict enum acceptance.
- `tests/tracker_parser/test_pass_verdict_validation.py` — 6 tests covering each Section 4-L failure mode (null config path → HALT, missing config file → HALT, missing §4 heading → HALT, flag false → HALT), full PASS case, FAIL-verdict skip-validation case.
- `tests/tracker_parser/test_v12_golden_arc10.py` — 4 tests: Arc 10 retrofit parses as v1.2, §4 heading present, §1 metric fields unchanged from v1.1 (idempotency), full CLI validation passes against live repo.

Test results: **60/60 pass**. No regressions in existing 41 tests.

### Task 4 — Retrofit Arcs 8, 10, 11

**Three reconstructed config YAMLs** (per chat Q2):

- [configs/l_arc_8/winning_config.yaml](configs/l_arc_8/winning_config.yaml) — A6 meta_labeling, SL=1.5×ATR, sl_plus_tp_3r, unlimited exposure, RF classifier with thresholds (0.3, 0.5). Reconstructed from closure §1 + wfo_results.csv row 2 (config_id=30) + scripts/l_arc_8/{shared.py, run_step5_wfo.py}.
- [configs/l_arc_10/winning_config.yaml](configs/l_arc_10/winning_config.yaml) — A1 system_level_filter, SL=3.5×ATR, sl_partial_close_1r_runner_trail, unlimited exposure. Reconstructed from closure §1 + wfo_results.csv row 2 + scripts/l_arc_10_v3/step_5.py + configs/l_arc_10_v3/arc_open.yaml.
- [configs/l_arc_11/winning_config.yaml](configs/l_arc_11/winning_config.yaml) — A2 classifier_filter, SL=2.0×ATR, sl_only, per-pair-1 + per-currency-2, RF classifier with threshold 0.11937665572820917 (cluster 0). Reconstructed from closure §1 + wfo_results.csv row 3 + run_summary.json + scripts/l_arc_11/run.py.

All three carry the required header block per chat Q2: "RECONSTRUCTED RETROSPECTIVELY at 2026-05-23 from <sources>. Original engine ran from driver script at <path> with these parameters." Each parameter annotated with its source.

**Three closure retrofits** (per chat Q1 — section numbering insert-with-gap; per chat Q3 — abbreviated §4 for FAIL arcs):

- [results/l_arc_8/ARC_CLOSURE.md](results/l_arc_8/ARC_CLOSURE.md) — added `template_version: v1.2` + `config_artefact_path: configs/l_arc_8/winning_config.yaml` + `deployment_spec_section_present: true` to §1. Inserted §4 deployment_spec (FAIL arc — abbreviated, §4.4 and §4.11 marked "FAIL arc — not applicable for deployment"; §4.10 includes the chat-required config-YAML-reconstruction caveat). §2 / §3 / §10 untouched.
- [results/l_arc_10/ARC_CLOSURE.md](results/l_arc_10/ARC_CLOSURE.md) — same §1 additions. Inserted §4 deployment_spec as primary spec (PASS-VIABLE / PASS-DEPLOYABLE-PROVISIONAL). All sub-sections written fully; §4.10 caveats include chat-required config-YAML-reconstruction language + Step 6 PROVISIONAL flag. §2 / §3 / §10 untouched.
- [results/l_arc_11/ARC_CLOSURE.md](results/l_arc_11/ARC_CLOSURE.md) — same §1 additions. Inserted §4 deployment_spec (FAIL arc — abbreviated, same convention as Arc 8). §2 / §3 / §10 untouched.

Numbering: §1, §2, §3, §4, §10 (insert-with-gap per chat Q1). §10 "Amendment 3 re-evaluation" preserved as universal cross-arc anchor.

### Task 5 — WORKFLOW.md

`WORKFLOW.md` §2 arc-close artefact set extended with 2 bullets for PASS verdicts: §4 deployment_spec requirement + populated `config_artefact_path` with file presence. §7 cadence-table template row updated to reference v1.2.

### Task 6 — Parser dry-run re-validation

```
py scripts/update_tracker_from_closure.py results/l_arc_8/ARC_CLOSURE.md  --dry-run  → exit 0
py scripts/update_tracker_from_closure.py results/l_arc_10/ARC_CLOSURE.md --dry-run  → exit 0
py scripts/update_tracker_from_closure.py results/l_arc_11/ARC_CLOSURE.md --dry-run  → exit 0
```

**Arc 10 specifically:** v1.2 PASS-verdict validation runs (verdict `PASS-VIABLE` triggers it) and passes all four Section 4-L checks. No error emitted; tracker diff content is the (expected) re-application of Arc 10's contributions per the rolling-state seeding state documented in `scripts/tracker_parser/README.md` (Arcs 8 and 10 not seeded into rolling state during the prior backfill — separate work item out of scope here).

**Arcs 8 and 11 (FAIL):** v1.2 PASS-verdict validation is skipped per spec (template Section 4-L last paragraph). Both parse cleanly and emit a dry-run diff matching pre-retrofit behaviour.

### Task 7 — Documentation

`scripts/tracker_parser/README.md`:
- Schema-versioning section extended with v1.2 detection rule + precedence note.
- New "v1.2 PASS-verdict validation" subsection mirroring template Section 4-L.
- Error-modes (HALT) list extended with the v1.2-specific failures.
- File-layout block updated to include the three new test files.
- Test count updated from 41 → 60.

---

## Anything I want to flag for chat

1. **Tracker diff in the v1.2 dry-run is expected, not a regression.** The dispatch said "Tracker state should be unchanged from current — these retrofits do NOT add tracker rows." Arc 11's pre-retrofit sha256 is in `parsed.log`, so pre-retrofit Arc 11 was a no-op; the retrofit changes the sha256 so on a real (non-dry-run) invocation, the parser would re-apply Arc 11's contributions to the tracker. Arcs 8 and 10 were never seeded into rolling state (per README "Bootstrap state (2026-05-22)" — `Arcs 8 and 10 are NOT seeded`), so they would also re-apply. **This is a pre-existing condition; the dispatch's expectation does not hold because rolling state ≠ tracker rows.** If chat wants the tracker bytes truly unchanged post-retrofit, the cleanup path is to seed Arcs 8 and 10 into rolling state alongside this PR — but that's the `results/re_evaluation_2026_05/SUMMARY.md §"Tracker work surfaced"` work item the README already flags as out of scope. Recommend leaving as-is and addressing in the separate cleanup PR; the v1.2 retrofit is structurally correct regardless.

2. **Verdict enum extended.** Added `PASS-VIABLE-PROVISIONAL`, `PASS-DEPLOYABLE-PENDING-STEP6`, `PASS-VIABLE-PENDING-STEP6` to `VALID_VERDICTS` to match the template v1.2 §4 trigger spec ("REQUIRED if verdict ∈ {PASS-DEPLOYABLE, PASS-VIABLE, PASS-DEPLOYABLE-PROVISIONAL, PASS-VIABLE-PROVISIONAL, PASS-*-PENDING-STEP6}"). The schema rejects unknown verdicts; without this extension, future closures using the new variants would HALT in schema validation. Chat may want to track this as a v3.0 schema-versioning note — though materially it's a forward extension, not a breaking change.

3. **Section 4-L specifically prohibits adding `config_artefact_path` to the tracker schema** ("this column is NOT in the tracker schema; do NOT add it to the tracker. It's read by the parser for validation only"). Implemented exactly: the v1.2 field is read by the new CLI helper but never written to any tracker cell. The `mapping` module is unchanged.

4. **Branch rename pending.** Current worktree branch is `claude/romantic-nash-219b96`. Per chat direction, will rename to `infra/closure-template-v1.2` at PR open time.

5. **Rebase onto main mid-dispatch — Arc 10 §10 finalisation reconciled.** Mid-dispatch, `origin/main` advanced by one commit ([#178](https://github.com/kpanapabusiness-droid/Forex-Backtester/pull/178)) which finalised Arc 10's §10 Amendment 3 re-evaluation from `PASS-DEPLOYABLE-PROVISIONAL` to `PASS-DEPLOYABLE (confirmed)` after Step 6 audit clean. I stashed the v1.2 retrofit, fast-forwarded the worktree to main (cleanly — no merge), restored the stash (no conflicts — main's edits in §10 / my edits in §1 + §4 touch different file regions), and updated three Arc 10 §4 references to reflect the post-finalisation state: §4 header (PASS-DEPLOYABLE instead of PASS-DEPLOYABLE-PROVISIONAL), §4.10 caveats (chained DD + per-day DD missing-data flags now framed as "forwarded out of Step 6 scope per main #178" with revert clause), §4.11 Step-6 checklist row (checked, points to `step_6/audit_report.md` from #178). §10 itself was not modified by me — that's main's content, preserved per dispatch rule. The v1.2 golden test for Arc 10 + the full 60-test suite + all three parser dry-runs (exit 0) re-validated post-reconciliation.

---

## Definition-of-done checklist

- [x] `docs/templates/ARC_CLOSURE_TEMPLATE.md` at v1.2 with §4 deployment_spec template + `config_artefact_path` field.
- [x] `L_PROTOCOL.md` §6 patched.
- [x] Parser updated with v1.2 schema detection + PASS-verdict validations + test coverage (60/60).
- [x] Arcs 8, 10, 11 retrofitted with §4 deployment_spec + updated §1 tracker_payload.
- [x] Parser dry-run exits 0 on all three retrofitted closures.
- [x] `WORKFLOW.md` updated (§2 arc-close artefact set + §7 cadence table).
- [x] `scripts/tracker_parser/README.md` updated.

End of log. Ready for PR.

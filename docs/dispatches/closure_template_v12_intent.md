# closure_template_v12_intent.md

> **Dispatch:** `CC Dispatch — Closure Template v1.2 (deployment spec) + Retrofit Arcs 8, 10, 11`
> **Branch:** `claude/romantic-nash-219b96` (running in a worktree cut from main; dispatch names `infra/closure-template-v1.2` — will rename / open PR from this branch unless chat directs otherwise).
> **Scope:** template + parser + 3 closure retrofits + WORKFLOW.md + parser README. No engine changes. No verdict logic changes.
> **Stage:** Read-first complete. Awaiting chat answers to three open questions before executing Tasks 1-7.

---

## Reads completed

1. **`docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.1 (284 lines)** — current schema. §1 YAML block lives inside an outer `markdown` fence with an inner `yaml` fence; the parser opens the inner one. `best_architecture:` block already carries 19+ Amendment-3 risk-normalised fields. Section 4 A-K mapping checklist is the parser contract; Section 5 references `scripts/update_tracker_from_closure.py`. Schema-versioning table at the foot already anticipates `template_version` as a per-closure declaration.
2. **`L_PROTOCOL.md` §3, §6.** §3 (gates) is unchanged by this dispatch — Amendment 3 risk-normalised gate language stays untouched. §6 has an "ARC_CLOSURE.md format" subsection (lines 561-571) that needs to be replaced with the v1.2 spec (4 sections + §4 conditional REQUIRED rule). The §6 subsection currently reads "All arc closure docs MUST follow `docs/templates/ARC_CLOSURE_TEMPLATE.md` v1.0" — bumps to v1.2 and gains the §4 conditional rule.
3. **The three closure docs** (`results/l_arc_{8,10,11}/ARC_CLOSURE.md`). Confirmed each currently has §1 / §2 / §3, then a `## §10 Amendment 3 re-evaluation` section. No §4 sections present. `template_version` field is absent in all three closures' §1 YAML (parser infers v1.1 from Amendment-3 fields under v1.1-exclusive-field detection, then normalises). All three are written under the v1.1 schema effectively.
4. **`scripts/update_tracker_from_closure.py` + `scripts/tracker_parser/{schema,extract,mapping,registry,rolling_state,tracker_io}.py`.** Detection precedence: `template_version` field → v1.1-exclusive-field presence → default v1.0 (schema.py:70-92). Extraction is regex on `## §1 tracker_payload` heading + the next fenced `yaml` block (extract.py:21-77). Schema validation is Pydantic; current `parse_payload` returns a v1.1-normalised dict and rejects unknown verdict / failure_mode / architecture enums.
5. **Canonical config YAML for each arc's winning architecture.** This is where the **load-bearing finding** sits — see "Interpretive call 2" below.
6. **`WORKFLOW.md`** — §2 has an "Arc-close artefact set" subsection (lines 38-47). §7 cadence table includes a row for `docs/templates/ARC_CLOSURE_TEMPLATE.md` (line 120). Both touched.
7. **`scripts/tracker_parser/README.md`** — Schema versioning section at lines 46-56 explicitly enumerates the v1.0/v1.1 rules; v1.2 row needs adding. README references `tests/tracker_parser/` test set count (currently 41 tests per line 137).
8. **`tests/tracker_parser/`** — 10 existing test files. `conftest.py` exposes `closure_arc_8`, `closure_arc_10`, `closure_arc_11` fixtures pointing at the live closure docs. The three new test files (Task 3d) drop alongside existing ones.

---

## File paths CC will create / modify (in execution order)

| Order | Path | Action | Source |
|---|---|---|---|
| T1 | `docs/templates/ARC_CLOSURE_TEMPLATE.md` | Edit — bump v1.1→v1.2, add 3 `best_architecture` fields, insert §4 deployment_spec template block, update Section 4 mapping checklist, update Section 5 parser notes, add schema-versioning row | Dispatch Task 1a-1e |
| T2 | `L_PROTOCOL.md` | Edit §6 — replace "ARC_CLOSURE.md format" subsection verbatim with the v1.2 spec | Dispatch Task 2 |
| T3a | `scripts/tracker_parser/schema.py` | Add v1.2 detection, add `BestArchitectureV12` model (extends v1.1 with 3 new fields), add `TrackerPayloadV12`, extend `parse_payload` dispatcher, extend `normalize_to_v11` → `normalize_to_v12` semantics | Dispatch Task 3a-3c |
| T3a | `scripts/tracker_parser/extract.py` | Add `verify_deployment_spec_section(closure_path)` helper that scans for `## §4 deployment_spec` heading | Dispatch Task 3b |
| T3a | `scripts/update_tracker_from_closure.py` | Wire PASS-verdict validation step between schema validation and tracker mutation: if v1.2 + PASS-verdict → run config_artefact_path + §4-heading + flag-true checks | Dispatch Task 3b |
| T3b | `tests/tracker_parser/test_schema_v12.py` | NEW | Dispatch Task 3d |
| T3b | `tests/tracker_parser/test_pass_verdict_validation.py` | NEW | Dispatch Task 3d |
| T3b | `tests/tracker_parser/test_v12_golden_arc10.py` | NEW (Arc 10 retrofit idempotency vs current tracker state) | Dispatch Task 3d |
| T4a | `configs/l_arc_8/winning_config.yaml` | CREATE (per Q2 chat direction; reconstructed from `step_5/wfo_results.csv` + closure §1 + driver script) | Dispatch Task 4 |
| T4a | `configs/l_arc_10/winning_config.yaml` | CREATE (per Q2 chat direction) | Dispatch Task 4 |
| T4a | `configs/l_arc_11/winning_config.yaml` | CREATE (per Q2 chat direction) | Dispatch Task 4 |
| T4b | `results/l_arc_8/ARC_CLOSURE.md` | Edit — insert §4 deployment_spec (marked FAIL-consistency), update §1 tracker_payload (add `template_version: 1.2`, `config_artefact_path`, `deployment_spec_section_present: true`) | Dispatch Task 4 |
| T4b | `results/l_arc_10/ARC_CLOSURE.md` | Edit — insert §4 deployment_spec (primary deployment spec; PROVISIONAL caveats), update §1 tracker_payload | Dispatch Task 4 |
| T4b | `results/l_arc_11/ARC_CLOSURE.md` | Edit — insert §4 deployment_spec (marked FAIL-consistency), update §1 tracker_payload | Dispatch Task 4 |
| T5 | `WORKFLOW.md` | Edit §2 arc-close artefact set — add 2 bullets (PASS verdicts require §4 deployment_spec + populated `config_artefact_path`) | Dispatch Task 5 |
| T6 | (parser dry-run, no file write) | Run on all 3 retrofitted closures | Dispatch Task 6 |
| T7a | `scripts/tracker_parser/README.md` | Edit Schema-versioning section — add v1.2 row; update HALT error modes; update test count | Dispatch Task 7 |
| T7b | `WORKFLOW.md` | Edit §7 cadence table — update template-row note to reference v1.2 (`docs/templates/ARC_CLOSURE_TEMPLATE.md` line is already there) | Dispatch Task 7 |

Plus this intent doc + a `closure_template_v12_log.md` at end (per WORKFLOW §2 pattern).

**Net:** 13 files modified, 3 files created in `configs/`, 3 test files created, 2 dispatch artefacts. Sequence is mostly serialisable — only T6 (dry-run validation) depends on T3+T4 finishing first.

---

## Interpretive calls flagged for chat

Three items where the dispatch needs an explicit chat decision before execution. **Items 1 + 3 are restatements of the dispatch's "Open questions for chat" with my recommendation. Item 2 is a load-bearing finding that affects whether the dispatch can execute as written.**

### 1. Section numbering of retrofitted closures (dispatch Q1)

The three closure docs currently use `## §1` / `## §2` / `## §3` / `## §10 Amendment 3 re-evaluation`. Inserting `## §4 deployment_spec` between §3 and §10 leaves a 5..9 gap.

- **Option A — insert with gap.** Numbering becomes 1, 2, 3, 4, 10. Preserves §10 as the universal "Amendment 3 re-evaluation" cross-arc anchor — Arc 8 / 10 / 11 / 5 / 7 (re-eval pending) all keep §10 = Amendment 3 re-eval. Dispatch's recommendation.
- **Option B — renumber.** Numbering becomes 1, 2, 3, 4, 5 (renaming §10 → §5 "Amendment 3 re-evaluation"). Clean numerical order but breaks any prose / cross-doc references to `§10` (CLAUDE.md references arc closure §10 indirectly via "Amendment 3 re-evaluation"; the existing arc-closure-content references §10 internally).

**Recommendation: Option A (insert with gap), per dispatch's own suggestion.** Confirm before Task 4 retrofit.

### 2. Canonical config YAML availability is uneven across the three arcs — load-bearing finding (dispatch Q2)

I expected each arc's winning architecture to be backed by a canonical config YAML file referenced by the engine. That is partly true for Arc 8 and false for Arcs 10 and 11.

**Arc 8.** `configs/l_arc_8/step5.yaml` exists but is **v2.3-era** (`phase: L_ARC_8_STEP_5_WFO`, references `results/l_arc_8/step1_verbatim/...`, talks about Pipeline E / Pipeline D1 — pre-v3.0 framing). The Arc 8 v3 re-run that produced the closure's `A6_SL1.5_TP3R_unlimited_thr_0.3_0.5` winning config was driven by `scripts/l_arc_8/run_step5_wfo.py` with inline parameters; no v3-era YAML lives in repo for it.

**Arc 10.** No `configs/l_arc_10/winning_config.yaml` exists. The v3 winning config (`sl_3.5x_partial_close_1r_runner_trail_unlimited`) lives entirely as constants + a config table inside `scripts/l_arc_10_v3/step_5.py` (lines 73-91 + the per-arch config builders). The arc_open YAML at `configs/l_arc_10_v3/arc_open.yaml` covers Step 1 only; Step 5 parameters are inline in the driver.

**Arc 11.** No `configs/l_arc_11/winning_config.yaml` exists. The v3 winning config (`a2_shb_cluster0`, A2 classifier_filter with primary_threshold ≈ 0.119) lives inline in `scripts/l_arc_11/run.py` (lines 346-388 — A1Config / A2Config / A6Config builders).

**What the dispatch says (Q2):** "if no canonical YAML exists at retrofit time, create one at `configs/<arc>/winning_config.yaml` matching the parameters the engine actually ran, then reference it. Document in the retrofit's §4.10 caveats that the YAML was reconstructed from artefacts."

**Recommendation: take the dispatch's option — reconstruct 3 YAMLs from artefacts.** Reconstruction sources for each arc would be:

- Arc 8: closure §1 `best_architecture` block + `results/l_arc_8/step_5/wfo_results.csv` (winning config row) + the existing v2.3 `configs/l_arc_8/step5.yaml` for the Step-1 signal definition (re-stated under v3.0 framing) + `scripts/l_arc_8/run_step5_wfo.py` for inline parameter values.
- Arc 10: closure §1 + `results/l_arc_10/step_5/wfo_results.csv` + `scripts/l_arc_10_v3/step_5.py` constants + `configs/l_arc_10_v3/arc_open.yaml` for the signal-side fields.
- Arc 11: closure §1 + `results/l_arc_11/run_summary.json` (winning-config primary_threshold ≈ 0.11937665572820917) + `scripts/l_arc_11/run.py` constants + `results/l_arc_11/ARC_OPEN.md` for signal-side fields.

Each reconstructed YAML will have a header comment: `# Reconstructed retrospectively from artefacts (closure §1, step_5/wfo_results.csv, driver script). Not the live file the engine consumed at run time.` and the §4.10 caveat will mirror this.

**Confirm reconstruction is acceptable before Task 4.** Alternative is to leave `config_artefact_path: null` for all three and explicitly mark the v1.2 retrofit as "schema-compliant but config-pointer-empty" — but that defeats the purpose of the field for the one arc (Arc 10) that's PASS-track and would actually benefit from EA portability foundation.

### 3. FAIL arc §4 strictness (dispatch Q3)

Arcs 8 and 11 are FAIL verdicts. Dispatch says §4 is "technically optional but write it anyway for consistency." Three readings:

- **Full spec** — write §4.1 through §4.11 in full, same depth as Arc 10. Cost: ~250-400 lines per FAIL closure. Benefit: full structural symmetry.
- **Abbreviated** — sub-sections present; concrete content where applicable; `N/A` or 1-line "FAIL arc — not deployment-relevant" where not. Dispatch's recommendation.
- **Heading-only stub** — single line under each sub-heading saying "FAIL arc — not deployment-relevant." Cheapest; least useful.

**Recommendation: abbreviated.** §4.1 / 4.2 / 4.3 / 4.5 / 4.6 / 4.7 / 4.8 / 4.9 written normally (these are observable from artefacts regardless of verdict). §4.4 (filter chain) written for A2/A6 winners but tagged "FAIL — not deployed." §4.10 caveats written normally. §4.11 deployment-readiness checklist marked "N/A — FAIL verdict, not for deployment."

---

## Parser implementation sketch (Task 3 preview)

Schema detection (T3a): extend `detect_schema_version()` to return `'1.2'` when `template_version` is `'v1.2'` / `'1.2'` OR when `config_artefact_path` or `deployment_spec_section_present` fields are present in `best_architecture`. Detection order becomes 1.2 → 1.1 → 1.0 → error. The existing v1.0/v1.1 codepaths are untouched.

PASS-verdict validation (T3a, wired into `update_tracker_from_closure.py` between step 2 (`schema.parse_payload`) and step 3 (`tracker_io.read_tracker`)):

```python
# After payload validated; before tracker mutations
verdict_starts_with_pass = payload["verdict"].startswith("PASS-")
if payload.get("template_version") == "1.2" and verdict_starts_with_pass:
    ba = payload.get("best_architecture") or {}
    config_path = ba.get("config_artefact_path")
    if not config_path:
        logging.error(...)  # HALT
        return 1
    if not (_REPO_ROOT / config_path).exists():
        logging.error(...)  # HALT
        return 1
    if not _closure_has_deployment_spec_heading(closure_path):
        logging.error(...)  # HALT
        return 1
    if ba.get("deployment_spec_section_present") is not True:
        logging.error(...)  # HALT
        return 1
```

Backwards compat (T3c): v1.0 and v1.1 closures skip the v1.2-only validation block. Tracker-mutation logic is unchanged across versions — the three new fields are tracked in the v1.2 schema model but **not** written to tracker columns (per dispatch: "this column is NOT in the tracker schema; do NOT add it to the tracker. It's read by the parser for validation only").

Idempotency contract: re-running the parser on a retrofitted closure produces the same tracker bytes as the pre-retrofit run because the three new fields don't affect any tracker A-J mapping. This is the test `test_v12_golden_arc10.py` confirms.

---

## What I will NOT do without further direction

- Bump the `template_version` from `v1.1` to `v1.2` in any closure beyond Arcs 8/10/11 (no Wave 1 arc 5 / 7 / 9 closures touched).
- Add `config_artefact_path` to the tracker schema (dispatch explicitly forbids this).
- Modify any verdict / failure_mode / amendment-3 metric in §1 of the three closures (dispatch: "Do NOT modify §2, §3, or §10/Amendment 3 content").
- Run the engine, re-fit any classifier, or recompute any per-fold metric.
- Open the closure PR on a branch named `infra/closure-template-v1.2` — current branch is `claude/romantic-nash-219b96`; will rename / cherry-pick if chat directs.

---

## Pre-execution checklist (after chat green-light)

- [ ] Q1 answered (section numbering: Option A insert-with-gap)
- [ ] Q2 answered (config YAML reconstruction: take dispatch's option; create 3 YAMLs)
- [ ] Q3 answered (FAIL arc §4 strictness: abbreviated)
- [ ] T1 template v1.1 → v1.2
- [ ] T2 L_PROTOCOL §6 patch
- [ ] T3 parser v1.2 + tests
- [ ] T4 retrofit 3 closures (+ 3 reconstructed config YAMLs)
- [ ] T5 WORKFLOW.md §2
- [ ] T6 parser dry-run on all 3 — expects exit 0 × 3
- [ ] T7 README + WORKFLOW §7
- [ ] Log doc at `docs/dispatches/closure_template_v12_log.md`
- [ ] PR `[INFRA] Closure template v1.2 + Arc 8/10/11 deployment spec retrofit`

End of intent. Ending turn for chat review.

# tracker_parser

Ingests an `ARC_CLOSURE.md` file, validates the `§1 tracker_payload` YAML block, and applies the `docs/templates/ARC_CLOSURE_TEMPLATE.md` Section 4 A-K mapping to `ARC_TRACKER.md`. Side-effects three files: the tracker, a rolling-state sidecar, and an idempotency registry.

## Quick start

```bash
python scripts/update_tracker_from_closure.py results/<arc>/ARC_CLOSURE.md
```

Run from the repo root (the script auto-locates `ARC_TRACKER.md`, `scripts/tracker_parser/rolling_state.json`, and `scripts/tracker_parser/parsed.log`).

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--dry-run` | off | Parse + validate; print intended diff; do NOT write. |
| `--tracker-path PATH` | `ARC_TRACKER.md` | Override tracker location. |
| `--rolling-state PATH` | `scripts/tracker_parser/rolling_state.json` | Override rolling-state sidecar. |
| `--registry PATH` | `scripts/tracker_parser/parsed.log` | Override idempotency registry. |
| `--verbose` | off | Print parsed payload + intended mutations before writing. |

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success or no-op (closure already in registry). |
| 1 | Schema / payload error (HALT — fix the closure doc). |
| 2 | Tracker IO error. |

## Workflow — when to invoke

Parser invocation is part of the **standard arc-close workflow**, run on the arc branch **before opening the closure PR**:

1. CC writes `results/<arc>/ARC_CLOSURE.md` per `docs/templates/ARC_CLOSURE_TEMPLATE.md`.
2. CC runs:
   ```bash
   python scripts/update_tracker_from_closure.py results/<arc>/ARC_CLOSURE.md
   ```
3. CC commits both the closure doc **and** the tracker delta (`ARC_TRACKER.md`, `scripts/tracker_parser/rolling_state.json`, `scripts/tracker_parser/parsed.log`) on the arc branch in **one atomic commit**.
4. PR opens with closure + tracker update in the same diff.
5. Chat reviews + merges.

Pre-PR invocation (not post-merge) ensures reviewers see the tracker change alongside the closure that drives it. This is the canonical location for the workflow — `L_PROTOCOL.md` §6, `WORKFLOW.md` §2, and `docs/templates/ARC_CLOSURE_TEMPLATE.md` §5 all reference back to this section.

## Schema versioning

The parser detects the closure doc's template version automatically:

| Detection rule | Outcome |
|---|---|
| `template_version` field present in `§1 tracker_payload` | Use it (accepts `v1.0`, `v1.1`, `v1.2`, `v1.3`, `1.0`, `1.1`, `1.2`, `1.3`). |
| Top-level `step_6` block present | Infer v1.3. |
| Any v1.2-exclusive field present in `best_architecture` (`config_artefact_path`, `deployment_spec_section_present`) | Infer v1.2. |
| Any v1.1-exclusive field present (`worst_fold_dd_base_pct`, `k_safe`, etc.) | Infer v1.1. |
| Otherwise | v1.0. |

Detection precedence: 1.3 → 1.2 → 1.1 → 1.0 → error.

v1.3.1 (L_PROTOCOL Amendment 5) shares the `template_version: v1.3` declaration. The discriminator is the presence of the optional top-level `architectures_skipped_by_amendment_5` field — mirrors the v1.2 / v1.2.1 `chained_dd_method` rollout pattern. There is no separate `v1.3.1` literal; the parser treats v1.3 and v1.3.1 as one detection bucket and applies the Amendment-5 Phase-2 cutoff check below.

v1.3.1+ (Amendment 5.1, 2026-05-25) extends the `architectures_skipped_by_amendment_5` vocabulary to accept the reason string `a5_gate_4_admission_blocked_by_no_pass_tier_constituent` alongside `{A1..A6}` architecture IDs. No template version bump — same `template_version: v1.3` declaration, same v1.3 detection bucket. Parser adds a Phase 1 WARNING-level check for Gate 4 qualifier compliance on post-cutoff PASS closures (see Phase 2 tightening section below).

v1.0 closures use legacy field names (`worst_fold_roi_pct`, `worst_fold_dd_pct`). The parser internally renames them to v1.1 names (`*_base_pct`) and fills v1.1-exclusive Amendment 3 fields with `null`. v1.1 closures are passed through.

v1.2 closures add `config_artefact_path` + `deployment_spec_section_present` to the `best_architecture` block plus a new `§4 deployment_spec` section in the closure doc. Retrofitted v1.2 closures may retain v1.0-style field names (`worst_fold_roi_pct`); the parser renames them pre-validation via `_coerce_legacy_field_names` — purely additive on §1, no manual rewrite required.

v1.3 closures (L_PROTOCOL Amendment 4) add a top-level `step_6` block to `tracker_payload`. The block records Step 6 framework outcome (ran/trigger/overall_passed/categories/critical_failures/warnings_count/verdict_impact). REQUIRED for any v1.3 PASS verdict; OPTIONAL for FAIL/HALT closures (records `ran: false`).

The mapping logic operates on a unified v1.1+-shaped dict — historical closures are never rewritten. v1.3 adds Section 4-M (Step 6 audit registry) which is silently no-op'd against trackers that lack the section.

## v1.2 PASS-verdict validation

When `template_version` ∈ `{1.2, 1.3}` AND `verdict` starts with `PASS-`, the parser enforces (after schema validation, BEFORE any tracker mutation):

1. `best_architecture.config_artefact_path` MUST be non-null.
2. The file at that path (resolved relative to repo root) MUST exist.
3. The closure doc MUST contain a `## §4 deployment_spec` heading.
4. `best_architecture.deployment_spec_section_present` MUST be `true`.

Any failure → HALT exit code 1, no tracker write, no rolling-state update. Fix the closure (re-create the missing config YAML, add the §4 section, set the flag) and re-run.

For non-PASS verdicts (FAIL / HALT / DISCOVERY_COMPLETE), the validation block is skipped — §4 is optional and `config_artefact_path` may be `null`.

For v1.0 and v1.1 closures, the v1.2 validation block is not evaluated.

## Phase 2 tightening (v1.3 / Amendment 4)

Cutoff: `2026-05-23T06:20:59Z` (PR-186 merge per chat resolution Q7).

For any PASS verdict whose `closed_timestamp > 2026-05-23T06:20:59Z`, the parser REQUIRES Amendment 3 fields in `best_architecture` (`chained_max_dd_base_pct`, `k_safe`, `k_hard`, `r_safe_pct`, `r_hard_pct`, `scalable_to_safe`, `scalable_to_hard`). Missing any field → HALT.

For `template_version: v1.3` PASS verdicts, the parser ADDITIONALLY requires:

1. `step_6` block present.
2. `step_6.ran == true`.
3. `step_6.overall_passed == true`.

PASS verdicts cannot ship without a clean Step 6 dispatch. Manual invocations cannot satisfy this requirement (they always carry `trigger: manual` and `verdict_impact: none`); only auto-dispatched runs produce `overall_passed: true` with engine confidence.

Pre-cutoff closures (v1.0 / v1.1 / v1.2 / v1.2.1 closed before the cutoff) are grandfathered — none of the Phase 2 checks fire.

### v1.3.1 — Amendment 5 architecture-skip field

For any PASS verdict whose `closed_timestamp > AMENDMENT_5_CUTOFF_ISO` (defined in `scripts/tracker_parser/schema.py`; pinned at the placeholder `2026-05-23T00:00:00Z` pending post-merge backfill), the parser additionally requires the optional top-level `architectures_skipped_by_amendment_5` field. The field is a subset of `{A1..A6}` and MAY be empty (`[]`) when the Amendment-5 four-gate set equals or supersets the Amendment-1 set. Pre-cutoff closures grandfathered. Mirrors the v1.2.1 `chained_dd_method` Phase 2 pattern.

**Amendment 5.1 (2026-05-25 — Gate 4 PASS-tier-constituent qualifier):** Post-`AMENDMENT_5_1_CUTOFF_ISO` PASS closures with ≥2 candidate clusters surviving Step 3 must declare A5 admission state explicitly. If A5 was admitted and ran, no entry needed in `architectures_skipped_by_amendment_5`. If A5 was not admitted because no constituent cluster cleared Step 5 PASS-tier, the field MUST include `a5_gate_4_admission_blocked_by_no_pass_tier_constituent`. Phase 1 (current): WARNING-level check; closures parse with warnings logged. Phase 2 (future): upgrade to ERROR-level after backfill of `AMENDMENT_5_1_CUTOFF_ISO` to PR merge timestamp.

## Idempotency

Each parse records the closure doc's sha256 in `scripts/tracker_parser/parsed.log`. Re-running the parser on a closure whose sha256 is already in the registry produces a no-op (exit 0, registry unchanged). If a closure doc is edited after parsing, its sha256 changes and the parser will re-apply — but only forward (no rollback of the prior state). Closure docs should be treated as immutable after merge.

The registry is git-tracked so reparses are consistent across machines.

## Rolling state

`rolling_state.json` is the parser's bookkeeping sidecar — git-tracked, append-only writes. It holds per-row counters and sums not visible in the tracker:

- `features.<name>` — `n_with` / `sum_with` / `n_without` / `sum_without` (for rolling avgs in the Per-feature contribution table).
- `architectures.<short_name>` — `n_tested` / `n_won` / `sum_won_ratio`.
- `archetypes.<canonical_label>` — `n_arcs` / `n_clusters` / `sum_mfe` / `sum_reach`.
- `tags.<name>` — `count` / `arcs[]`.

The parser treats this sidecar as the source of truth for rolling avgs (not the tracker cells). Keeping the sidecar in sync with the tracker is essential — if they diverge, the next parse will recompute averages from the sidecar and overwrite the tracker cells.

## Error modes (HALT)

The parser exits with code 1 on any of:

- `§1 tracker_payload` heading missing.
- Fenced ```yaml block missing or malformed.
- YAML parse error.
- Required schema field missing or wrong type (Pydantic ValidationError).
- `primary_failure_mode` not in the locked enum.
- `verdict` not in the locked enum.
- `architectures_tested` entry not in `{A1, A2, A3, A4, A5, A6}`.
- Archetype label not in the normalisation map (closures use varying casing; new archetypes require an explicit map update before the parser will accept them).
- A `per_failure_mode_count` or `per_archetype_recurrence` row referenced in the closure does not exist in the tracker — parser will not create these rows; add them manually first.
- **v1.2 PASS-verdict only:** `best_architecture.config_artefact_path` is null OR the file at that path does not exist OR the closure doc lacks a `## §4 deployment_spec` heading OR `best_architecture.deployment_spec_section_present` is not `true` (template Section 4-L).

## Determinism

Same closure doc + same starting tracker + same starting rolling state → byte-identical output tracker. The `Last auto-update` line uses the closure's `closed_timestamp` (formatted UTC `YYYY-MM-DD HH:MM:SS`), making the timestamp deterministic by construction.

## Bootstrap state (2026-05-22)

The live `rolling_state.json` and `parsed.log` were seeded from Arc 11's closure as part of this PR. Arc 11's row contributions are in rolling state; Arc 11's sha256 is in `parsed.log` (so re-running the parser on Arc 11 is a no-op).

**Arcs 8 and 10 are NOT seeded** — they have a Closed arcs summary row in the live tracker from a prior manual backfill, but their other-section contributions are missing. Running the parser on either of those closure docs against the live tracker today would append a duplicate Closed arcs summary row. See `results/re_evaluation_2026_05/SUMMARY.md` §"Tracker work surfaced" for the cleanup path.

## File layout

```
scripts/
├── update_tracker_from_closure.py   # CLI entrypoint
└── tracker_parser/
    ├── __init__.py
    ├── README.md                    # this file
    ├── schema.py                    # Pydantic models + version detection
    ├── extract.py                   # YAML extraction from §1
    ├── tracker_io.py                # ARC_TRACKER.md read/write
    ├── mapping.py                   # Section 4 A-K mapping
    ├── rolling_state.py             # rolling-state sidecar IO
    ├── registry.py                  # parsed.log IO
    ├── rolling_state.json           # bookkeeping sidecar (git-tracked)
    └── parsed.log                   # idempotency registry (git-tracked)

tests/tracker_parser/
├── conftest.py
├── fixtures/
│   ├── tracker_blank_state.md
│   └── tracker_after_arc_11.md
├── test_schema_detection.py
├── test_schema_v12.py
├── test_pass_verdict_validation.py
├── test_yaml_extraction.py
├── test_mapping_A_through_K.py
├── test_idempotency.py
├── test_determinism.py
├── test_golden_arc8.py
├── test_golden_arc10.py
├── test_golden_arc11.py
└── test_v12_golden_arc10.py
```

## Testing

```bash
python -m pytest tests/tracker_parser/ -v
```

60 tests cover: schema detection + normalisation (v1.0 / v1.1 / v1.2), YAML extraction HALT cases, every Section 4 A-K mapping, v1.2 PASS-verdict validation (Section 4-L), idempotency, determinism, and four golden closures (Arc 11 full A-K; Arcs 8 & 10 Closed arcs summary row only per Q-3; Arc 10 v1.2-retrofit metric-invariance + Section 4-L pass).

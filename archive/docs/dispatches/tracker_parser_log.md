# Tracker Parser — Log Doc

> Dispatch: build `scripts/update_tracker_from_closure.py` + tests
> Intent doc: [tracker_parser_intent.md](tracker_parser_intent.md)
> Branch: `infra/tracker-parser` (this PR)
> Status: complete; 41/41 tests pass.

---

## What landed

### Code (10 files)
- `scripts/update_tracker_from_closure.py` — CLI entrypoint.
- `scripts/tracker_parser/__init__.py`
- `scripts/tracker_parser/schema.py` — Pydantic v2 models for v1.0 / v1.1, schema detection, v1.0 → v1.1 normalisation, enum validation.
- `scripts/tracker_parser/extract.py` — §1 fenced-block YAML extraction; HALTs on any of 6 malformed-input cases.
- `scripts/tracker_parser/tracker_io.py` — line-level surgical edits to `ARC_TRACKER.md`; byte-stable on no-op round-trip; preserves inline HTML-comment suffixes on rows.
- `scripts/tracker_parser/mapping.py` — Section 4 A-K orchestrator; archetype normalisation map; rolling-avg arithmetic per feature / architecture / archetype / tag.
- `scripts/tracker_parser/rolling_state.py` — sidecar JSON read/write (sorted-keys deterministic).
- `scripts/tracker_parser/registry.py` — sha256 idempotency registry.
- `scripts/tracker_parser/rolling_state.json` — bootstrap state (Arc 11 contributions).
- `scripts/tracker_parser/parsed.log` — bootstrap state (Arc 11 sha256).

### Tests (8 files, 41 tests)
- `tests/tracker_parser/conftest.py` — fixtures.
- `tests/tracker_parser/fixtures/tracker_blank_state.md` — synthetic pre-Arc-11 tracker.
- `tests/tracker_parser/fixtures/tracker_after_arc_11.md` — Arc 11 full golden.
- `test_schema_detection.py` — 11 tests (4 version-detection paths, v1.0/v1.1 normalisation, 3 enum HALT cases).
- `test_yaml_extraction.py` — 7 tests (happy path + 6 HALT cases).
- `test_mapping_A_through_K.py` — 17 tests (every A-K step on synthetic payloads).
- `test_idempotency.py` — 1 test (re-parse no-op).
- `test_determinism.py` — 2 tests (two runs byte-identical, Arc 8 + Arc 11).
- `test_golden_arc11.py` — 1 test (full A-K verification against `tracker_after_arc_11.md`).
- `test_golden_arc8.py`, `test_golden_arc10.py` — Closed arcs summary row only (per Q-3 scope).

### Documentation
- `scripts/tracker_parser/README.md` — canonical "when to invoke" + workflow + schema versioning + error modes.
- `L_PROTOCOL.md` §6 — line replaced: parser implemented, points at README.
- `ARC_TRACKER.md` footer — "Update mechanism" reflects parser as default; manual updates are now the exception.
- `docs/templates/ARC_CLOSURE_TEMPLATE.md` §5 — Section 5 "Parser implementation notes (deferred)" replaced with current parser location + workflow reference.
- `WORKFLOW.md` §2 — new "Arc-close artefact set" sub-section describing parser invocation as part of standard pre-PR workflow.
- `results/re_evaluation_2026_05/SUMMARY.md` — housekeeping note added about Arc 8/10 partial backfill + parser bootstrap state.

---

## Q-1 through Q-6 resolutions and how they landed

| Q | Resolution | Where it landed |
|---|---|---|
| Q-1 | Sidecar JSON, git-tracked | `scripts/tracker_parser/rolling_state.json` (bootstrapped with Arc 11), `rolling_state.py` for read/write |
| Q-2 | Closure's `closed_timestamp` for determinism | `mapping.apply_last_auto_update` + `_format_parser_timestamp` |
| Q-3 | Arc 11 full A-K; Arcs 8/10 Closed arcs row only | Three golden tests sized accordingly; housekeeping note in SUMMARY.md |
| Q-4 | Parser writes empty string for `Re-evaluated verdict` column | `mapping.apply_closed_arcs_summary` cell 8 = `""` |
| Q-5 | Use §1 `worst_fold_ratio` (engine reading) | Tracker row uses `best_architecture.worst_fold_ratio` verbatim, formatted via `str()` to match YAML repr |
| Q-6 | Warning + exit 0 on absent Active arcs row | `mapping.apply_active_arcs` logs WARNING and continues |

---

## Verification (dispatch Task 11)

Per Q-3 the verification is:

- **Arc 11 full A-K golden** — `test_arc_11_full_golden` runs the full pipeline against `tracker_blank_state.md` and asserts byte-identical output against `tracker_after_arc_11.md`. Pass.
- **Arcs 8 + 10 Closed arcs summary row golden** — `test_arc_8_closed_arcs_summary_row` and `test_arc_10_closed_arcs_summary_row` apply the parser and assert the appended row's 10 cells match the spec. Pass.

The Arc 11 generated fixture matches the live `ARC_TRACKER.md` content section-by-section, with two intentional deltas:
1. `Re-evaluated verdict` column on the Arc 11 Closed arcs summary row is empty (parser convention per Q-4) vs `FAIL` (manual backfill).
2. `Last auto-update` line is `parser: 2026-05-22 12:33:35` (derived from Arc 11's `closed_timestamp`) vs the manual `manual: 2026-05-22` + parenthetical.

Both deltas are correct parser behaviour.

Dry-run on the live tracker (Arcs 8/10/11) confirms expected behaviour:
- Arc 11 → registry hit, exit 0, no-op (sha256 already in `parsed.log`).
- Arc 8 → would append a duplicate Closed arcs summary row (Arc 8 not in registry; row already in tracker from manual backfill) AND new rows in all other sections (gap from partial backfill — documented in SUMMARY.md).
- Arc 10 → same pattern as Arc 8.

The Arc 8/10 dry-run behaviour matches Q-3's anticipated discovery. The bootstrap state in `rolling_state.json` reflects Arc 11 only; cleanup of Arc 8/10 contributions is documented as a separate work item.

---

## Surprises / deviations from intent

- **Arc 8/10 duplicate-append on live tracker.** Anticipated in intent §Q-3 and confirmed at validation time. Per Q-3 scope decision, parser was not modified to handle this case automatically — it's a housekeeping cleanup item.
- **Pydantic v2 syntax confirmed at code-time** (intent §6 flagged this as a "verify before coding" item). Repo uses v2.12; models use `ConfigDict(extra="allow")` per v2 convention.
- **Markdown round-trip stability** worked first try via line-level surgical edits. Inline HTML-comment suffixes (e.g., `step5_dd_above_gate` row's `<!-- deprecated by Amendment 3 -->` comment) are preserved through `update_row`.

No deviations from dispatch scope. No deviations from chat answers Q-1 through Q-6.

---

## Test summary

```
tests/tracker_parser/test_determinism.py ............................ 2 passed
tests/tracker_parser/test_golden_arc10.py ........................... 1 passed
tests/tracker_parser/test_golden_arc11.py ........................... 1 passed
tests/tracker_parser/test_golden_arc8.py ............................ 1 passed
tests/tracker_parser/test_idempotency.py ............................ 1 passed
tests/tracker_parser/test_mapping_A_through_K.py .................. 17 passed
tests/tracker_parser/test_schema_detection.py ..................... 11 passed
tests/tracker_parser/test_yaml_extraction.py ........................ 7 passed
======================== 41 passed in ~1.0s ==========================
```

---

## Definition-of-done checklist

- [x] Parser implementation (8 Python files; spec listed 5 — added `rolling_state.py`, `extract.py`, `registry.py` for clean separation).
- [x] Test suite covers schema detection, YAML extraction, all Section 4 A-K mappings, idempotency, determinism, three golden closures.
- [x] All tests pass.
- [x] Parser reproduces current tracker state for Arc 11 (full A-K) and Arcs 8/10 (Closed arcs summary row only) per Q-3 scope.
- [x] README, L_PROTOCOL §6, ARC_TRACKER convention line, template §5, WORKFLOW §2, SUMMARY.md all updated.
- [x] Bootstrap state seeded (rolling_state.json + parsed.log with Arc 11 contributions).

# closure_template_intent.md

> **Dispatch:** `CC_08_CLOSURE_TEMPLATE_LAND.md`
> **Branch:** `infra/closure-template-v1` (cut from main, this commit forward)
> **Scope:** docs-only. No engine, no arc, no backtester.
> **Stage:** Read-first complete. Awaiting chat review before executing Tasks 1-7.

---

## Reads completed

1. `L_PROTOCOL.md` §6 — current "ARC_CLOSURE.md required sections" subsection (lines 444-452) and "ARC_TRACKER.md auto-update" subsection (lines 454-464). Both to be replaced verbatim per Task 3.
2. `WORKFLOW.md` §7 — cadence table at lines 103-117. `docs/sub_protocols/*` row is at line 109. Template row inserts immediately after.
3. `ARC_TRACKER.md` — confirmed **empty initial state**. No active arcs, no closed-arc rows, no per-feature rows, all counter tables zeroed. `Last auto-update: never (empty initial state)`. **Safe to overwrite per Task 2** — HALT clause does not trigger.
4. `TODO.md` — Round 2 row `Draft 'ARC_TRACKER.md' skeleton (empty schema)` is at line 54, status 🔴 (matches dispatch precondition). Round 4 ends at line 80; Phase 0 header begins line 83 — Round 5 inserts between these. `Last updated:` line at line 5 currently reads `2026-05-22 (PR-E.1.7 closure of CC_06 dispatch chain)`.

---

## File paths CC will touch (in execution order)

| Order | Path | Action | Source |
|---|---|---|---|
| T1 | `docs/templates/` | Create directory | — |
| T1 | `docs/templates/ARC_CLOSURE_TEMPLATE.md` | Create (verbatim) | Attached `ARC_CLOSURE_TEMPLATE.md` |
| T2 | `ARC_TRACKER.md` | Overwrite (verbatim) | Attached `ARC_TRACKER (1).md` |
| T3 | `L_PROTOCOL.md` | Edit §6 — replace 2 subsections | Dispatch text (Task 3) |
| T4 | `WORKFLOW.md` | Edit §7 — insert 1 cadence row | Dispatch text (Task 4) |
| T5 | `TODO.md` | Edit — flip Round 2 row to 🟢, insert Round 5 block, bump `Last updated` | Dispatch text (Task 5) |
| T7 | PR `[INFRA] Closure template v1.0 + tracker schema extension` | gh pr create | — |

Plus this intent doc + a `closure_template_log.md` at the end (per WORKFLOW §2 pattern).

Net: 5 files modified, 1 directory created, 2 dispatch artefacts (intent + log). All edits verifiable via `git diff`.

---

## Verbatim-source diffs (pre-execution sanity)

- Template file: 239 lines as provided. Contains nested fences (outer ```markdown wraps inner ```yaml at L27-114, then closes at L138). Lands as-is per dispatch ("CC does not edit their content").
- ARC_TRACKER replacement: 141 lines as provided. New sections vs current:
  - Cross-arc cluster registry (NEW)
  - Cost-decomposition registry (NEW)
  - Cross-arc tag registry (NEW)
  - Per-failure-mode table — enum keys now snake_case (`pool_too_small`, etc.) matching template §1 `tracker_payload.primary_failure_mode` enum; row count expands from 11 → 12 (adds `admit_only_vs_deployment`, splits `step5_*` into three rows, etc.)
  - Per-archetype table unchanged in shape
  - Per-architecture table unchanged in shape (still 6 rows A1-A6)
  - Update mechanism rewritten to reference template Section 4 mapping + parser path

---

## Interpretive calls flagged for chat

Three items where the dispatch is ambiguous or where I noticed something worth surfacing before execution:

### 1. `TODO.md` "Last updated" — today is already 2026-05-22

The dispatch line is `Bump the file's 'Last updated' line at the top to today's date`. The file already shows `2026-05-22`. Same calendar day. Default plan: rewrite the parenthetical to reference this dispatch — `2026-05-22 (CC_08 closure template + tracker schema extension)` — replacing the prior `(PR-E.1.7 closure of CC_06 dispatch chain)` annotation. Flag if chat prefers verbatim-no-change or a different annotation.

### 2. TODO.md has a duplicate cadence table at §"Reminder — what gets updated when" (lines 200-213)

This duplicate mirrors WORKFLOW §7. Task 4 updates WORKFLOW §7 only; it does not mention this mirror. Default plan: **leave the TODO mirror untouched** (dispatch scope is explicit, and the discipline rules say "All edits verifiable via git diff" — i.e. only the specified edits). Consistency cleanup can be a separate trivial follow-up. Flag if chat wants the mirror synced in this dispatch.

### 3. Failure-mode enum row split — historical-counter implication is zero (no prior arcs under v3.0)

The new tracker's failure-mode table replaces 11 human-readable rows with 12 snake_case enum rows. Since the tracker is empty initial state, no counts are lost. Flagging only because the change is semantically a schema break — but the dispatch explicitly authorises verbatim replacement, and the tracker is empty, so this is in-scope and zero-risk.

---

## What CC will NOT do (boundaries)

- Will not modify content of the two attached files (template + tracker). Verbatim only.
- Will not edit any §6 content beyond the two specified subsections.
- Will not touch the TODO duplicate cadence table without explicit go-ahead.
- Will not touch any engine code, config YAML, or backtester artefact.
- Will not delete or restructure pre-existing tracker rows if they appear (they don't — verified empty).
- Will not open the PR until Tasks 1-6 are complete and `git diff` is clean.

---

## Halt-conditions during execution

CC will HALT and surface to chat if:
- Either verbatim-source file fails byte-identical write (e.g. line-ending corruption on Windows).
- §6 subsection match fails (heading drift since this read).
- WORKFLOW §7 `docs/sub_protocols/*` row not findable.
- TODO Round 2 row text drifts from the read snapshot.
- Any other unexpected file state.

---

## Ready for chat review

Standing by. On chat approval, execute Tasks 1-7 in sequence, write `closure_template_log.md`, open PR, end turn.

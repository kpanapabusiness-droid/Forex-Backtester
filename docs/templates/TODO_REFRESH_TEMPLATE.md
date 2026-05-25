# CC Dispatch — TODO.md Refresh (TEMPLATE — fill in placeholders)

> **How to use this template:** replace every `<PLACEHOLDER>` with current-state content. Sections marked `[FILL]` need current context. Sections marked `[STABLE]` rarely change. Save customised version as `CC_<N>_TODO_REFRESH_<date>.md` before firing.

> **Branch:** `infra/todo-refresh-<YYYY-MM-DD>`
> **Deliverable:** `TODO.md` fully rewritten to reflect current project state.
> **Scope:** docs-only. No code changes. No protocol changes.
> **Prerequisites:** <FILL: list PRs that must be merged before this refresh runs, e.g. "PR #XXX merged"; if no prerequisites, write "none">.

---

## Why [STABLE]

Current `TODO.md` drifts as PRs land, arcs close, and decisions are made. Periodic full rewrites are faster than continuous patches when drift is material. This dispatch refreshes the operational TODO to reflect actual project state.

---

## Read-first [STABLE]

Before any edits, produce `todo_refresh_intent.md`:

1. Read current `TODO.md` to assess staleness.
2. Read `ARC_TRACKER.md` to confirm current arc state.
3. Read `L_PROTOCOL.md` to confirm protocol version + amendments landed.
4. Read `docs/audits/engine_capability_audit_2026_05.md` (or latest audit) for capability state.
5. Read recent PR descriptions for what landed. Use `gh pr list --state merged --limit 20` to list recent merges.
6. Cross-reference against existing `TODO.md` claims; identify drift.
7. End turn for chat review.

---

## Current state to capture [FILL]

Fill in for THIS refresh:

### Recently merged PRs since last refresh
- PR #<NUM>: <one-line description, e.g. "Step 6 framework + Amendment 4 + parser v1.3">
- PR #<NUM>: <description>
- (add as many rows as needed)

### Currently open / in-flight PRs
- PR #<NUM>: <description, status: open / draft / awaiting review>
- (add rows)

### Active parallel chat workstreams
- <e.g. "heavy_ml_probe build chat — Phase 2 sub-protocol engine implementation, multi-week build">
- <e.g. "Wave 1 retry chat — 5 arcs running under amended engine">
- <e.g. "signal discovery probe — local 10k run by user">

### Protocol / template state
- L_PROTOCOL version: <e.g. v3.0>
- Active amendments: <e.g. Amendments 1, 2, 3, 4 inline>
- Closure template version: <e.g. v1.3>
- Parser supports: <e.g. v1.0 / v1.1 / v1.2 / v1.2.1 / v1.3>

### Architecture build status
- Wired: <e.g. A1, A2, A3, A4, A6>
- Not built: <e.g. A5 — deferred until VIABLE candidates exist>

### Engine capability gaps (per latest audit)
- WIRED: <count>
- PARTIAL: <count>
- MISSING: <count + key items>

---

## Target structure [STABLE]

Refreshed `TODO.md` follows the same section structure but with current content:

### 1. Header
- `Last updated:` = today's date
- Brief description unchanged

### 2. Status legend [STABLE — never changes]
🟢 DONE | 🟡 IN PROGRESS | 🔴 NOT STARTED | ⚪ BLOCKED / WAITING | ⚫ HALTED / CANCELLED

### 3. Current state — quick view
Rewrite to reflect:
- Phase 0: <current status>
- Phase 1: <current status, sub-detail on waves>
- Phase 2: <current status>
- Engine state: <one-line summary>
- Architectures wired: <list>
- Active parallel chats: <count>

### 4. Reset checklist (Rounds 1-N)
- Add new Round (next number) for any consolidated body of work since last refresh
- Re-status existing rounds as needed
- DO NOT delete completed rounds — they're historical record

### 5. Phase 0, Phase 1, Phase 2 sections
- Phase 0: <status + path verdict if applicable>
- Phase 1: per-wave breakdown, per-arc state, retry tracking
- Phase 2: sub-protocol status, signal-class queue

### 6. Engine consolidation [FILL if applicable]
If engine work has happened since last refresh, document:
| Item | Status | PR | Notes |
|---|---|---|---|
| <item> | 🟢 / 🟡 / 🔴 | PR #<num> | <one line> |

### 7. Standing items / open questions
Keep relevant existing items; add new ones surfaced since last refresh; remove resolved ones.

### 8. Doc set state
- Locked / ready: <list with versions>
- To create: <list>

### 9. Ideas parked
Keep parked items unless explicitly resolved.

### 10. Reminder — what gets updated when [STABLE — table unchanged unless new doc types added]

---

## Discipline rules [STABLE]

- Do NOT touch `ARC_TRACKER.md` (parser-managed).
- Do NOT touch `ARC_HISTORY.md` (frozen).
- Do NOT touch per-arc closure docs in `results/`.
- Do NOT modify L_PROTOCOL.md or sub-protocol docs.
- Refresh covers ONLY `TODO.md`.
- Preserve file's existing structure; replace content within.
- Use GitHub PR numbers (`#XXX`), not internal dispatch labels (`CC_XX`).
- If dispatch labels and PR numbers diverge from earlier docs, add a one-time §"PR numbering convention" annotation.

---

## Common corrections to apply [STABLE]

Most TODO refreshes will need to apply some subset of these corrections:

- **Falsely marked NOT STARTED that's actually DONE** — happens when state advanced but TODO wasn't updated
- **Missing entirely** — new work since last refresh
- **Items now obsolete** — decisions reversed or features deprecated
- **Stale PR numbers** — earlier dispatch labels superseded by GitHub numbers

---

## Definition of done [STABLE]

1. `TODO.md` fully rewritten with current state.
2. Header `Last updated` line reflects today.
3. All status emojis (🟢 / 🟡 / 🔴 / ⚪ / ⚫) accurate.
4. New sections included where work since last refresh warrants them.
5. Stale references removed.
6. PR title: `[INFRA] TODO.md refresh — post-PR-#<latest> state alignment` (use GitHub-actual number).

End turn at PR.

---

## Notes for chat [STABLE]

After this lands, the operational TODO matches actual project state. Any future arc closes / PR merges should add a one-line update to TODO with a short reference. Full rewrites should be rare — every 2-3 weeks at major state transitions or after substantial PR clusters.

The mastermind chat's working doc (if one exists) is separate and NOT part of any TODO refresh — chat-side memory only, not repo state.

---

## Open questions for chat [FILL if any]

List any items the read-first phase should surface for chat decision before the rewrite:

- <e.g. "PR numbering mismatch between dispatch labels and GitHub numbers — confirm GitHub numbers preferred">
- <e.g. "Phase 0a/0b framing — keep distinct or merge as completed phase">
- <e.g. "Arc N status uncertain — surface actual repo state, chat reconciles">

(Leave empty if no expected open questions; CC will surface anything it finds during read-first regardless.)

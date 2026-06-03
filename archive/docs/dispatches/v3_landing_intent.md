# v3.0 Landing & Reorg — Intent

Dispatch source: `CC_05_V3_LANDING_AND_REORG.md` (read from `~/Downloads/`).
Branch: `claude/busy-jones-cdb5e6` (worktree), targeting `main` via PR per Task 6.

## Confirmations

1. **Listed file changes understood and accepted.** Six new docs land at specified paths (`L_PROTOCOL.md`, `TODO.md`, `ARC_TRACKER.md`, `project_brief.md` overwrite at root; `docs/sub_protocols/heavy_ml_probe.md` and `docs/sub_protocols/signal_discovery_probe.md`). All other root `.md` files except the 8-file keep-list move to `docs/`. All `results/` subdirs (and `results/ARC_QUEUE.md`) move under `results/archived/`. Docs audit per Task 4.

2. **Pre-v3.0 results directory layout preserved under `results/archived/`.** `git mv` only; no content changes, no flattening. Each subdir retains its internal structure.

3. **Repo root post-dispatch matches keep-list (8 files):** `README.md`, `CLAUDE.md`, `project_brief.md`, `L_PROTOCOL.md`, `TODO.md`, `ARC_TRACKER.md`, `ARC_HISTORY.md`, `REPO_INVENTORY.md`. Verified by `find . -maxdepth 1 -name '*.md'` at end of Task 5.

4. **`docs/sub_protocols/` is a new subdirectory.** Will be created before placing the two sub-protocol files.

## Prerequisite check (per dispatch)

Consolidation PR from `claude/wonderful-ellis-8d3091` already merged to main (commit `c4b2b90`, PR #157, "inventory: repo reorganisation + ARC_HISTORY consolidation"). Proceeding.

## Inventory observations (pre-dispatch)

- Root `.md` files currently: 17. Keep-list keeps 8; 12 will move to `docs/` (no name collisions detected by pre-check against existing `docs/` entries). Movable: `AGENTS.md`, `CHANGELOG.md`, `L_ARC_PROTOCOL.md`, `L_ARC_PROTOCOL_v2_2_AMENDMENT.md`, `L_ARC_PROTOCOL_v2_3_AMENDMENT.md`, `L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md`, `PROTOCOL_IMPROVEMENT_BACKLOG.md`, `SESSION_ZERO.md`, `SHELVED_ARCS.md`, `STATUS.md`, `WORKFLOW.md`, `inventory_intent.md`.
- `results/` currently contains 16 subdirs plus `ARC_QUEUE.md` at top level. All move to `results/archived/`. The `ARC_QUEUE.md` file is pre-v3.0 (superseded by `ARC_TRACKER.md`) — moved with the rest. Noted as a minor interpretive call beyond the dispatch's literal "subdirs only" wording.
- `docs/` contains 14 `.md` files at its top level (excluding subdirs). Task 4 audits each against the new root keep-list.

## Deviations from dispatch (minor)

- **Intent and log artefacts live in `docs/dispatches/`, not at repo root.** The whole dispatch is about cleaning root to the 8-file keep-list, and `docs/dispatches/` already exists for dispatch artefacts. Putting them at root would mean Task 2 immediately moves them again, or leaves the log at root after Task 2 commits — neither is clean. Will note in PR body.
- **`results/ARC_QUEUE.md` archived as part of Task 3.** The dispatch's procedural step only enumerates subdirs, but its intent ("the raw `results/` content from pre-v3.0 should move to `results/archived/`") covers the file too. Will note in PR body.

If chat disagrees with either, easy to revert before merge.

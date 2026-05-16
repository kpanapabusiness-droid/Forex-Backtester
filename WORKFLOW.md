# WORKFLOW — Phase Management and Documentation

> **v2.0 protocol note (2026-05-16):** `L_ARC_PROTOCOL.md` v2.0 §13 introduces the
> one-live-arc-doc model (`ARC_<N>_LIVE.md`, finalised at arc end as
> `ARC_<N>_RESULT.md`) and direct-to-main workflow for analysis. This supersedes
> the per-step phase-doc / phase-close-prompt workflow described below for Arcs
> 3+. The folder-convention sections below remain valid. The Phase Close Checklist
> and templates are kept for v1.x historical reference and should not be applied
> to v2.0 arcs — candidates for rewrite when WORKFLOW reaches v3.

## The Problem This Solves
Previously, closing a phase required manual copy-paste across multiple documents,
separate prompts, and manual uploads to the Claude project folder. This document
standardises the process so phase close is one prompt and one upload action.

## Folder Convention (Locked, v2 — 2026-05-13)

**All result documents go in the arc's results folder.** Phase docs, result CSVs,
artefacts, configs — all co-located under the arc folder.

The previous `docs/` convention is retired. Co-location of phase docs with their
result artefacts is the new permanent standard, derived from the L arc protocol
(`L_ARC_PROTOCOL.md` v2.0; v1.x folder convention inherited from
`archive/L_ARC_OPERATIONAL_SPEC_v1_0.md` §2) and applied to all future work.

### For L arc research (current and future arcs)
Per v1.x ops spec §2 folder structure (still in force):
- Phase docs: `results/l_arc_N/PHASE_L_ARC_N_STEPK[_<candidate>].md`
- Arc open/closure: `results/l_arc_N/PHASE_L_ARC_N_OPEN.md`, `PHASE_L_ARC_N_CLOSURE.md`
- Per-step subfolders: `step1_verbatim/`, `step2_descriptive/`, etc.

### For non-L-arc work (future workstreams)
Follow the same co-located pattern:
- Phase docs: `results/<arc_name>/PHASE_<NAME>.md`
- Artefacts live in subfolders within the same `results/<arc_name>/` folder.
- `docs/` is reserved for system specs that aren't arc-specific (e.g., `KH24_SYSTEM_LOCK.md`).

### Migration of existing root-level phase docs
Existing root-level phase docs (`PHASE_L6_ARC*.md`) are SUPERSEDED. Do not move
them; their replacements live under `results/l_arc_N/` per the new protocol.

The Claude project folder contains only files that change frequently:
- `SESSION_ZERO.md`
- `STATUS.md`
- `CHANGELOG.md`
- `L_ARC_PROTOCOL.md` (v2.0, self-contained — read at start of every arc chat)
- The current phase's open and result docs

Static reference docs (GOLDEN_STANDARD_LOGIC, BACKTESTER_TEMPLATE, KH24_SYSTEM_LOCK
etc) do not need re-uploading unless they change.

## SESSION_ZERO Convention (Locked)
SESSION_ZERO.md is a living document. Do not replace it wholesale.
At phase close, update only the "Current State" section at the top.
The Phase History section below accumulates — never delete prior entries.

## Phase Close Checklist
Run this at the end of every phase/step, in order:

1. Write phase document to `results/<arc_name>/<location_per_protocol>.md`
2. Update SESSION_ZERO.md "Current State" section only
3. Append entry to CHANGELOG.md (most recent first)
4. Update STATUS.md to reflect new current step
5. Write handover note for next chat (v1.x convention; v2.0 supersedes with one-live-arc-doc per `L_ARC_PROTOCOL.md` §13)
6. Report which files changed — human uploads only those to Claude project folder

## Phase Close Prompt Template
Use this template when closing a phase. Fill in the bracketed sections.

---
PHASE CLOSE — [ARC NAME / STEP NUMBER]

Read `L_ARC_PROTOCOL.md` before starting (v2.0 is self-contained; v1.x ops spec
in `archive/` if working on a historical Arc 1/2 doc). Follow the phase close
checklist exactly. **For v2.0 arcs use the one-live-arc-doc model instead — see
`L_ARC_PROTOCOL.md` §13.**

Phase document to write (`results/<arc>/<path>.md`):
[Full phase document content]

SESSION_ZERO current state update:
[New current state block — replaces only the Current State section]

CHANGELOG entry:
[Single entry, 4-8 bullet points, most recent first]

STATUS.md update:
[Reflect new current step and any disposition]

Handover note for next chat (appended to phase doc):
[What's complete, path to phase doc, what next chat reads first, deferred decisions]

Report all files changed when complete. Do not touch any other files.
---

## What Goes In A Phase Document
- Why this phase exists (what question it answers — per `L_ARC_PROTOCOL.md` step definitions for L arc work)
- What was decided and why (decisions table)
- Key findings (numbered, significance flagged)
- Full-distribution outputs per discipline rule (no medians-only)
- Data location (file paths, how to load)
- Verdict / disposition (per protocol step gate logic)
- What comes next
- Permanently eliminated by this phase (if anything)
- Handover note for next chat at the bottom

## What Goes In An Arc-Open Document
**v1.x convention** (per `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md` §12). For
v2.0 arcs there is no separate arc-open doc — see `L_ARC_PROTOCOL.md` §13 for
the one-live-arc-doc structure (`ARC_<N>_LIVE.md`).

## Modification Policy
This workflow document is locked v2. Modifications require an explicit
re-planning chat and a version bump. The folder convention is the permanent
standard; do not revert to `docs/` for phase outputs.

---

*Last updated: 2026-05-13 — folder convention migrated to `results/<arc>/` co-location, permanent.*

# WORKFLOW.md — Operational Conventions

> **Purpose:** how to operate the project around `L_PROTOCOL.md`. Captures conventions that emerged in practice but aren't methodology.
> **Status:** living document. Updated as conventions evolve.
> **Scope:** all chat sessions and CC dispatches. Applies regardless of arc or sub-protocol.
>
> `L_PROTOCOL.md` is methodology. This is operations.

---

## §1 Tool boundaries

| Tool | Role |
|---|---|
| Chat (Claude in this conversation) | Strategy, analysis, decisions, verdicts, drafting CC dispatches, reviewing CC outputs |
| Claude Code (CC) | Multi-file features, code, tests, runs, computation. Executes dispatches from chat. |
| Cursor (Sonnet) | Single-file patches, YAML, doc updates. Small changes only. |

GPT-4 and Aider are permanently excluded.

Per L_PROTOCOL §1: chat decides what data means. CC produces data. The rule: "what does the data show?" → CC prompt. "What does the data mean?" → chat.

---

## §2 Dispatch artefact pattern

Every non-trivial CC dispatch follows this pattern:

1. **Chat drafts dispatch** as a markdown file. Lives at `docs/dispatches/<dispatch_name>.md` or `/mnt/user-data/outputs/<dispatch_name>.md`.
2. **CC produces intent doc FIRST** at `docs/dispatches/<work>_intent.md`. Lists file paths CC will touch, plan summary, any interpretive calls for chat. End turn for chat review.
3. **Chat reviews intent**, confirms or redirects.
4. **CC executes**, producing artefacts + a **log doc** at `docs/dispatches/<work>_log.md` recording verification results, deviations from dispatch, and any flags for chat.
5. **CC opens PR**, ending turn.
6. **Chat reviews PR**, merges (or requests changes), pulls main locally.

The intent → log → PR pattern is mandatory for any CC dispatch touching more than 2-3 files. For trivial single-file edits the pattern can be skipped.

### Arc-close artefact set

When the work is an arc closure, the artefact set on the arc branch includes the tracker update produced by the parser:

1. CC writes `results/<arc>/ARC_CLOSURE.md` per `docs/templates/ARC_CLOSURE_TEMPLATE.md`.
2. CC runs `python scripts/update_tracker_from_closure.py results/<arc>/ARC_CLOSURE.md`.
3. CC commits the closure doc + the tracker delta (`ARC_TRACKER.md`, `scripts/tracker_parser/rolling_state.json`, `scripts/tracker_parser/parsed.log`) in one atomic commit on the arc branch.
4. CC opens the closure PR. Chat reviews + merges.

The parser invocation is pre-PR, not post-merge — so reviewers see the tracker change in the same diff as the closure. See `scripts/tracker_parser/README.md` for full usage.

**For PASS verdicts (DEPLOYABLE / VIABLE / *-PROVISIONAL / *-PENDING-STEP6)** the artefact set additionally includes:
- `§4 deployment_spec` in the closure doc per template v1.2 §4 — self-contained porting specification.
- `best_architecture.config_artefact_path` populated in §1 tracker_payload and the referenced YAML file present at that path (relative to repo root). The parser HALTs at exit code 1 if either is missing or the file does not exist (template Section 4-L). Reconstruct the YAML from artefacts if it does not exist, document reconstruction in §4.10.

---

## §3 Branch + worktree conventions

- Canonical: `main`
- Arc branches: `arc/<arc_name>` (per L_PROTOCOL §7)
- Infra branches: `infra/<work>` (e.g., `infra/backtester-v3-pr-a`)
- Diagnostic / probe branches: `diagnostic/<topic>` or `probe/<topic>`
- One worktree per active CC session. Worktrees enabled in CC session config.
- After PR merge: GitHub "Delete branch" button + `git branch -d <name>` locally.
- L_PROTOCOL §7 auto-cleanup hook handles arc branches once implemented.

Multiple arcs run on parallel worktrees with no conflicts (independent `results/<arc>/` folders). Append conflicts at simultaneous-close on `ARC_TRACKER.md` resolve chronologically — no content dropped.

---

## §4 Diff-before-code discipline

For any work that ports / reimplements existing logic (e.g., porting a strategy from MQL5 to Python), CC produces a **side-by-side diff doc** before touching code:

- Section per component being ported
- Aspect table: spec / source-of-truth vs current implementation vs mismatch flag
- Committed BEFORE code changes

This pattern emerged in PR-E.1.5 / PR-E.1.6 after the original PR-E.1 was found to have ported KH-24 incorrectly. Once a port has bugs, side-by-side comparison catches them before they compound.

Applies to: strategy ports, methodology amendments, feature engineering changes that reimplement prior logic.

---

## §5 Staged-PR pattern

Large work (>500 LOC, multiple components, anchor or gate-driven outcomes) splits into staged PRs:

- Each PR is one logical layer, reviewable in isolation
- Sequential dependencies (PR-B depends on PR-A); chat merges in order
- Final PR is typically the verification / gate (anchor reproduction, baseline run, etc.)

Backtester reconfig used 5+ staged PRs (PR-A through PR-E with subdivisions). PR-E.2 became the critical gate.

For arc work: a typical Phase 1 arc is single-PR scope (one signal, one closure). Stage only if scope is genuinely large (e.g., sub-protocol that adds engine infrastructure).

---

## §6 Halt discipline

CC HALTS rather than tries to fix unexpected outcomes when:

- Anchor reproduction fails outside explainability
- A bisect or diagnostic surfaces ambiguous causes
- A dispatch's verification step doesn't pass
- Any other condition where "just trying to fix" risks compounding the problem

On HALT:
- No PR opened
- Branch pushed for chat review
- Diagnostic doc produced at `docs/dispatches/<work>_diagnostic.md`
- End turn

This prevents the "kept trying things until something worked" failure mode that produces unreliable code and hidden assumptions.

Chat decides next step from the diagnostic.

---

## §7 Documentation update cadence

| Doc | Update cadence |
|---|---|
| `L_PROTOCOL.md` | Only at major redesign events |
| `docs/sub_protocols/*` | When sub-protocol is amended |
| `docs/templates/ARC_CLOSURE_TEMPLATE.md` | Only at major redesign events (template version bump). Current: v1.2 (2026-05-23, deployment_spec addition). |
| `WORKFLOW.md` (this file) | When operational conventions evolve |
| `ARC_TRACKER.md` | Auto on arc open / close per L_PROTOCOL §6 |
| `ARC_HISTORY.md` | Never (frozen at v3.0 start) |
| `TODO.md` | Manually at meaningful state transitions (wave closes, phases close, milestones) — NOT per arc |
| `README.md`, `CLAUDE.md`, `project_brief.md` | Rarely |
| `BACKTESTER_ARCHITECTURE.md`, `DATA_FOUNDATION.md` | When backtester or data source changes |
| Per-arc `ARC_OPEN.md` / `ARC_CLOSURE.md` | Write-once |
| Dispatch artefacts in `docs/dispatches/` | Append-only (intent → log → diagnostic if needed) |

---

## §8 Communication conventions

Chat preferences in this project (descriptive, not normative):

- Direct, analyst-mode, minimal back-and-forth
- Concise without hedging
- Push back when something looks off
- Domain shorthand fine (fold numbers F1-F7, R-multiples, MAE/MFE, WFO gate, PASS-DEPLOYABLE / PASS-VIABLE)
- Result tables before prose; details after if asked
- One-line verdicts where possible

Applies to chat → CC dispatches (specify exactly what's expected) and chat → other chats (handovers should mirror these expectations).

---

## §9 When in doubt

- Methodology question → L_PROTOCOL.md
- Operational question → this doc
- Sub-protocol question → `docs/sub_protocols/<name>.md`
- Project history / past arc results → ARC_HISTORY.md
- Live state → ARC_TRACKER.md (closed-arc summary) or TODO.md (operational phase)
- Eliminated approaches → CLAUDE.md

If a question doesn't fit any of these, it belongs in this doc — add a section.

---

End. Living document. Update as conventions evolve.

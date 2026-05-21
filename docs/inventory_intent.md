# Repo Inventory + Reorganisation — Intent Doc

> Plan doc. Tasks 1-4 in the dispatch execute against this plan. If a classification proves wrong during execution, halt and update this doc before continuing.
> Generated: 2026-05-19. Branch: `claude/wonderful-ellis-8d3091` (worktree on `main`).

---

## Scope

`git ls-files '*.md'` returns **178** tracked `.md` files. Counts by tree:

- 22 at repo root
- 4 under `archive/`
- 7 under `attic/2025-11-08/`
- ~40 under `docs/` (mixed: subdirs `arc_results/`, `calibration_decisions/`, plus flat phase/era/signal-spec docs)
- ~95 under `results/` (per-arc artefact subfolders, mostly step/exp summaries)
- The rest under `.cursor/`, `.github/`, `prompts/`, `indicators/mq5/`, `mt5/`, `scripts/`, etc.

`all_md_files.txt` has the full sorted list.

---

## Classification scheme (per dispatch)

| Class | Action |
|---|---|
| `ACTIVE` | Keep at current path |
| `ARC_RESULT` | Source for ARC_HISTORY.md; move to `docs/archive/arc_results/` |
| `PROTOCOL_HISTORICAL` | Move to `docs/archive/protocol/` |
| `CALIBRATION_DECISION` | Move to `docs/archive/calibration/` (or stay if subdir already current) |
| `SPEC_HISTORICAL` | Move to `docs/archive/signal_specs/` |
| `NNFX_ERA` | Move to `docs/archive/nnfx_era/` |
| `DISPATCH_HISTORICAL` | Move to `docs/archive/dispatches/` |
| `OPEN_PROPOSAL` | Keep at current path; flag in inventory |
| `UNCERTAIN` | Flag for chat; do NOT move |

---

## Target repo-root contents (post-move)

ONLY these stay at root:
- `README.md`, `CLAUDE.md`, `STATUS.md`, `CHANGELOG.md`
- `WORKFLOW.md`, `AGENTS.md`
- `L_ARC_PROTOCOL.md` (active v2.1.2 base)
- `L_ARC_PROTOCOL_v2_2_AMENDMENT.md` (active, landed)
- `L_ARC_PROTOCOL_v2_3_AMENDMENT.md` (active, landed)
- `L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md` (OPEN_PROPOSAL)
- `PROTOCOL_IMPROVEMENT_BACKLOG.md` (active backlog)
- `SESSION_ZERO.md` (active primer)
- `SHELVED_ARCS.md` (active register)
- `ARC_HISTORY.md` (NEW — this dispatch)
- `REPO_INVENTORY.md` (NEW — this dispatch)
- `inventory_intent.md` (NEW — this plan; removed at end of dispatch or left for record per chat decision)

Existing subdirs that stay as-is:
- `.cursor/`, `.github/` — IDE/repo config
- `archive/` — v1.x protocol archive (already explicit archive)
- `attic/2025-11-08/` — quarantined NNFX-era files (already explicit archive)
- `docs/` (remaining current contents after archived items move)
- `docs/calibration_decisions/` — current calibration records
- `prompts/` — active arc-orchestrator template
- `indicators/`, `mt5/`, `core/`, `scripts/`, etc. — code subdirs
- `results/` — per-arc artefacts (co-location convention preserved)
- `tests/`

---

## Classification — root-level files

| File | Class | Reason |
|---|---|---|
| README.md | ACTIVE | Repo entry doc |
| CLAUDE.md | ACTIVE | First-read context |
| STATUS.md | ACTIVE | Current-state snapshot (chat may reset content) |
| CHANGELOG.md | ACTIVE | Recent closures log |
| SESSION_ZERO.md | ACTIVE | 5-minute primer |
| WORKFLOW.md | ACTIVE | Phase workflow |
| AGENTS.md | ACTIVE | Coding-agent contract |
| L_ARC_PROTOCOL.md | ACTIVE | Active v2.1.2 protocol |
| L_ARC_PROTOCOL_v2_2_AMENDMENT.md | ACTIVE | Active landed amendment |
| L_ARC_PROTOCOL_v2_3_AMENDMENT.md | ACTIVE | Active landed amendment |
| L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md | OPEN_PROPOSAL | DRAFT, not yet landed |
| PROTOCOL_IMPROVEMENT_BACKLOG.md | ACTIVE | Cross-arc backlog |
| SHELVED_ARCS.md | ACTIVE | Shelved register (v2.3+) |
| ARC_9_CANDIDATE_A_SPEC.md | SPEC_HISTORICAL | Arc 9 KILL_REAFFIRMED — classifier invalidated; deployment candidate no longer valid. Duplicate exists at `results/l_arc_9/`. |
| NEW_CHAT_HANDOVER.md | DISPATCH_HISTORICAL | Self-marked SUPERSEDED 2026-05-16 |
| C1_SWEEP_GUIDE.md | NNFX_ERA | NNFX 57-C1-indicator sweep guide |
| EXIT_INDICATOR_SETUP.md | NNFX_ERA | NNFX exit-indicator infrastructure |
| VOLUME_INDICATOR_SETUP.md | NNFX_ERA | NNFX volume-indicator setup |
| RESULTS_SCHEMA_AUDIT.md | NNFX_ERA | Jan-2025 baseline-vs-volume schema audit |
| project_brief.md | UNCERTAIN | Pre-v2.0 brief (2026-05-09) — superseded by CLAUDE.md/STATUS.md but referenced from arc closure docs (JL invalidation). Chat decides. |
| cleanup_intent.md | DISPATCH_HISTORICAL | One-time cleanup-dispatch record (2026-05-19) |
| cleanup_log.md | DISPATCH_HISTORICAL | One-time cleanup-dispatch record (2026-05-19) |

---

## Classification — archive/ and attic/

Already-archived. No moves.

- `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md` → stays (PROTOCOL_HISTORICAL, already in archive/)
- `archive/L_ARC_PROTOCOL_v1_0.md` → stays
- `archive/L_ARC_PROTOCOL_v1_1_AMENDMENT.md` → stays
- `archive/L_ARC_PROTOCOL_v1_2_AMENDMENT.md` → stays
- `attic/2025-11-08/*` (7 files) → stays (NNFX_ERA, already in attic/)

Note: `archive/` and `attic/` are pre-existing explicit-archive trees. I will not relocate their contents to `docs/archive/` — that would shuffle archives uselessly. Both are recorded in REPO_INVENTORY as already-archived.

---

## Classification — docs/ (flat files)

| File | Class | Reason |
|---|---|---|
| docs/KH24_SYSTEM_LOCK.md | ACTIVE | Live system spec, referenced for active deployment |
| docs/GOLDEN_STANDARD_LOGIC.md | ACTIVE | Execution-truth invariants (project-permanent) |
| docs/BACKTESTER_ARCHITECTURE.md | ACTIVE | Engine architecture reference |
| docs/BACKTESTER_AUDIT.md | ACTIVE | Engine audit reference |
| docs/BACKTESTER_USER_GUIDE.md | ACTIVE | Engine user guide |
| docs/BACKTESTER_EXTENSION_CLOSURE.md | DISPATCH_HISTORICAL | Closure record for PR #131/135/138 — work complete |
| docs/CLAUDE_PROJECT_INSTRUCTIONS.md | ACTIVE | Canonical copy of Claude project settings |
| docs/CANDIDATES.md | NNFX_ERA | Arc 1 L6+ candidate hypotheses (NNFX/L6.0 era) |
| docs/L0_METHODOLOGY_LOCK.md | ACTIVE | L characterization arc methodology — still referenced by LCHAR docs |
| docs/L6_0_METHODOLOGY_LOCK.md | PROTOCOL_HISTORICAL | Self-marked SUPERSEDED 2026-05-13 |
| docs/LCHAR_ATLAS.md | ACTIVE | L atlas reference (used by L_ARC arc-open dispatches) |
| docs/LCHAR_TOPN_REGISTRY.md | ACTIVE | Live registry of signals under test |
| docs/L_ARC_DEFERRED_CANDIDATES.md | ACTIVE | Active deferred-candidate reference |
| docs/L_ARC_FEATURE_REGISTRY.md | ACTIVE | Cross-arc feature registry, append-only |
| docs/L_ARC_PLAN.md | NNFX_ERA | Self-marked COMPLETED 2026-05-13 |
| docs/KH_Research_Roadmap.md | NNFX_ERA | KH arc closed; superseded by KH24_SYSTEM_LOCK + L_ARC_PROTOCOL |
| docs/PHASE6_PLAN.md | NNFX_ERA | Phase 6 NNFX-era |
| docs/PHASE_B1_C1_ARCHETYPES.md | NNFX_ERA | Phase B.1 NNFX-era |
| docs/PHASE_B_INDICATOR_QUALITY.md | NNFX_ERA | Phase B NNFX-era |
| docs/PHASE_C1_PARAMETER_SENSITIVITY.md | NNFX_ERA | Phase C.1 NNFX-era |
| docs/PHASE_C_C1_IDENTITY_WFO.md | NNFX_ERA | Phase C NNFX-era |
| docs/PHASE_D2_2_FEATURE_DIAGNOSTICS.md | NNFX_ERA | Phase D-2.2 NNFX-era |
| docs/PHASE_D2_LIFT_HARNESS.md | NNFX_ERA | Phase D-2 NNFX-era |
| docs/PHASE_D6F_CLEAN_LABELS.md | NNFX_ERA | Phase D-6F NNFX-era |
| docs/PHASE_L6_ARC1_OPEN.md | NNFX_ERA | Self-marked SUPERSEDED L6.0 era |
| docs/PHASE_L6_ARC1_P2_OPEN.md | NNFX_ERA | Self-marked SUPERSEDED L6.0 era |
| docs/PHASE_L6_ARC2_OPEN.md | NNFX_ERA | Self-marked SUPERSEDED L6.0 era |
| docs/PHASE_L6_ARC2_P3_OPEN.md | NNFX_ERA | L6.0 Arc 2 P3 open |
| docs/PHASE_L6_ARC2_P3_RESULT.md | ARC_RESULT | L6.0 Arc 2 P3 result (v1.x Arc 2 closure material) |
| docs/ARCHETYPE_REGISTRY.md | NNFX_ERA | Ignition-pool archetype registry (pre-KH/L) |
| docs/SPREAD_FLOOR_AUDIT_FINDING.md | CALIBRATION_DECISION | Resolved cross-arc finding (Arc 4 surface) |
| docs/SPREAD_SEMANTICS_LOCK.md | ACTIVE | Protocol-level lock referenced from active protocol |
| docs/cleanup_plan_2025-11-08.md | DISPATCH_HISTORICAL | 2025-11-08 cleanup plan record |
| docs/phase8_execution_truth.md | NNFX_ERA | Phase 8 execution-truth scratch (NNFX-era) |

### docs/arc_results/ — all ARC_RESULT

These are the canonical closure docs for Arcs 3, 4, 4-rerun, 5, 6, 7, 10. All move to `docs/archive/arc_results/`:

- docs/arc_results/ARC_3_RESULT.md
- docs/arc_results/ARC_4_RESULT.md
- docs/arc_results/ARC_4_RERUN_RESULT.md
- docs/arc_results/ARC_5_RESULT.md
- docs/arc_results/ARC_6_RESULT.md
- docs/arc_results/ARC_7_RESULT.md
- docs/arc_results/ARC_10_RESULT.md

### docs/calibration_decisions/ — stays

- docs/calibration_decisions/SPREAD_FLOOR_CALIBRATION_DECISION_2026-05-17.md → CALIBRATION_DECISION; stays at `docs/calibration_decisions/` (current subdir, dispatch-permitted).

### docs/signal_spec_*.md

| File | Arc | Status | Class |
|---|---|---|---|
| signal_spec_failed_breakout_long_v0.2.md | 6 | CLOSED (DIES Step 4) | SPEC_HISTORICAL |
| signal_spec_pullback_resume_hhhl_long_v0.1.md | 8 | CLOSED HALT_DEPLOYMENT | SPEC_HISTORICAL |
| signal_spec_inside_bar_break_trend_long_v0.1.md | 9 | CLOSED KILL_REAFFIRMED | SPEC_HISTORICAL |
| signal_spec_three_bar_reversal_trend_long_v0.1.md | 12 | UNRUN | UNCERTAIN |
| signal_spec_asia_range_breakout_htf_trend_long_v0.1.md | 13 | UNRUN | UNCERTAIN |
| signal_spec_mean_reversion_stretch_long_v0.1.md | 14 | UNRUN | UNCERTAIN |
| signal_spec_failed_breakdown_reversal_uptrend_long_v0.1.md | 15 | UNRUN | UNCERTAIN |
| signal_spec_persistent_momentum_continuation_long_v0.1.md | 16 | UNRUN | UNCERTAIN |

Unrun arcs (12-16) sit in the queue (`results/ARC_QUEUE.md`). The queue may reset post-data-foundation-rebuild. Their specs are flagged UNCERTAIN — chat decides whether they remain as queue inputs or get archived alongside the queue itself.

---

## Classification — results/ (artefact-folder closures)

The `results/` tree is artefact territory. Closure docs co-located with artefacts (per WORKFLOW v2 convention) stay in place — moving them would break co-location with their step1/step2/... subfolders.

These are recorded in REPO_INVENTORY as ARC_RESULT-class for ARC_HISTORY.md source-material purposes, but **left in place**:

- `results/arc_kh24_v2/ARC_KH24_V2_RESULT.md` (KH-24 v2.0 self-test)
- `results/l_arc_2_redo/ARC_2_REDO_RESULT.md` (Arc 2 v2.0 redo)
- `results/l_arc_8/ARC_8_CLOSURE.md` (Arc 8)
- `results/l_arc_8/ARC_8_LIVE.md` (Arc 8 live tracker)
- `results/l_arc_9/ARC_9_CLOSURE.md` (Arc 9)
- `results/l_arc_9/ARC_9_LIVE.md` (Arc 9 live tracker)
- `results/l_arc_11/ARC_11_CLOSURE.md` (Arc 11)
- `results/l_arc_11/ARC_11_LIVE.md` (Arc 11 live tracker)

Step/experiment/diagnostic summaries under `results/l_arc_*/` (~70 .md files) also stay in place. They are artefact subfolder contents, not root-level clutter.

`results/ARC_QUEUE.md` → ACTIVE (queue state file).

---

## Classification — other subdirs

| Path | Class | Notes |
|---|---|---|
| `.cursor/rules.md` | ACTIVE | IDE config |
| `.github/pull_request_template.md` | ACTIVE | Repo config |
| `prompts/cc_arc_orchestrator_template.md` | ACTIVE | v1.1 orchestrator template |
| `indicators/mq5/README.md` | ACTIVE | Component README |
| `mt5/README.md` | ACTIVE | Component README |
| `scripts/v1_3_calibration/loader_decisions.md` | UNCERTAIN | Active script's companion doc; superseded by replays? Chat decides. |

---

## Target structure post-move

```
<root>/
  README.md
  CLAUDE.md
  STATUS.md
  CHANGELOG.md
  SESSION_ZERO.md
  WORKFLOW.md
  AGENTS.md
  L_ARC_PROTOCOL.md
  L_ARC_PROTOCOL_v2_2_AMENDMENT.md
  L_ARC_PROTOCOL_v2_3_AMENDMENT.md
  L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md   (OPEN_PROPOSAL)
  PROTOCOL_IMPROVEMENT_BACKLOG.md
  SHELVED_ARCS.md
  ARC_HISTORY.md                              (NEW)
  REPO_INVENTORY.md                           (NEW)
  inventory_intent.md                         (NEW; this plan)
  project_brief.md                            (UNCERTAIN — flagged)
  archive/                                    (pre-existing v1.x archive)
  attic/2025-11-08/                           (pre-existing NNFX archive)
  docs/
    KH24_SYSTEM_LOCK.md
    GOLDEN_STANDARD_LOGIC.md
    BACKTESTER_*.md (3 active)
    CLAUDE_PROJECT_INSTRUCTIONS.md
    L0_METHODOLOGY_LOCK.md
    LCHAR_ATLAS.md
    LCHAR_TOPN_REGISTRY.md
    L_ARC_DEFERRED_CANDIDATES.md
    L_ARC_FEATURE_REGISTRY.md
    SPREAD_SEMANTICS_LOCK.md
    signal_spec_*.md (Arcs 12-16, UNCERTAIN)
    calibration_decisions/
      SPREAD_FLOOR_CALIBRATION_DECISION_2026-05-17.md
    archive/                                  (NEW)
      arc_results/
        ARC_3_RESULT.md
        ARC_4_RESULT.md
        ARC_4_RERUN_RESULT.md
        ARC_5_RESULT.md
        ARC_6_RESULT.md
        ARC_7_RESULT.md
        ARC_10_RESULT.md
        PHASE_L6_ARC2_P3_RESULT.md
      protocol/
        L6_0_METHODOLOGY_LOCK.md
      calibration/
        SPREAD_FLOOR_AUDIT_FINDING.md
      signal_specs/
        signal_spec_failed_breakout_long_v0.2.md
        signal_spec_pullback_resume_hhhl_long_v0.1.md
        signal_spec_inside_bar_break_trend_long_v0.1.md
        ARC_9_CANDIDATE_A_SPEC.md
      nnfx_era/
        CANDIDATES.md
        L_ARC_PLAN.md
        KH_Research_Roadmap.md
        PHASE6_PLAN.md
        PHASE_B1_C1_ARCHETYPES.md
        PHASE_B_INDICATOR_QUALITY.md
        PHASE_C1_PARAMETER_SENSITIVITY.md
        PHASE_C_C1_IDENTITY_WFO.md
        PHASE_D2_2_FEATURE_DIAGNOSTICS.md
        PHASE_D2_LIFT_HARNESS.md
        PHASE_D6F_CLEAN_LABELS.md
        PHASE_L6_ARC1_OPEN.md
        PHASE_L6_ARC1_P2_OPEN.md
        PHASE_L6_ARC2_OPEN.md
        PHASE_L6_ARC2_P3_OPEN.md
        ARCHETYPE_REGISTRY.md
        phase8_execution_truth.md
        C1_SWEEP_GUIDE.md
        EXIT_INDICATOR_SETUP.md
        VOLUME_INDICATOR_SETUP.md
        RESULTS_SCHEMA_AUDIT.md
      dispatches/
        NEW_CHAT_HANDOVER.md
        cleanup_intent.md
        cleanup_log.md
        cleanup_plan_2025-11-08.md
        BACKTESTER_EXTENSION_CLOSURE.md
  results/                                    (per-arc artefacts — untouched)
  prompts/                                    (untouched)
  (code subdirs untouched)
```

---

## UNCERTAIN files — flagged for chat

1. **project_brief.md** — Pre-v2.0 brief (2026-05-09). Referenced from `docs/arc_results/ARC_3_RESULT.md` ("JL invalidation"). Either ACTIVE-with-stale-content or NNFX_ERA. Content not modified per dispatch rules.
2. **docs/signal_spec_three_bar_reversal_trend_long_v0.1.md** (Arc 12 unrun) — queue input
3. **docs/signal_spec_asia_range_breakout_htf_trend_long_v0.1.md** (Arc 13 unrun)
4. **docs/signal_spec_mean_reversion_stretch_long_v0.1.md** (Arc 14 unrun)
5. **docs/signal_spec_failed_breakdown_reversal_uptrend_long_v0.1.md** (Arc 15 unrun)
6. **docs/signal_spec_persistent_momentum_continuation_long_v0.1.md** (Arc 16 unrun)
7. **scripts/v1_3_calibration/loader_decisions.md** — companion to a calibration script. Chat to confirm whether the calibration is current or superseded.

Reasoning for each UNCERTAIN entry is repeated in REPO_INVENTORY.md.

---

## Notes on `results/l_arc_*/ARC_*_CLOSURE.md`

Per WORKFLOW v2, arc closure docs may live at `docs/arc_results/ARC_<N>_RESULT.md` (per L_ARC_PROTOCOL §13 starting Arc 7) OR co-located with their artefact folder under `results/l_arc_<N>/`. The repo has both patterns:

- Arcs 3, 4, 4-rerun, 5, 6, 7, 10 — closure at `docs/arc_results/ARC_<N>_RESULT.md` (canonical pattern from CLAUDE.md)
- Arcs 8, 9, 11 — closure at `results/l_arc_<N>/ARC_<N>_CLOSURE.md` (parallel-CC-session pattern; ran in their own worktrees)
- KH-24 self-test, Arc 2 redo — closure at `results/<arc>/ARC_<N>_RESULT.md` (co-located legacy)

For this dispatch:
- `docs/arc_results/*` move to `docs/archive/arc_results/` (root-level clutter cleanup)
- `results/l_arc_*/ARC_*_CLOSURE.md` and `results/<arc>/ARC_<N>_RESULT.md` STAY in place (artefact co-location preserved)

All are source material for ARC_HISTORY.md regardless of location.

---

## Discipline

- No file deletions.
- No content modifications outside fixing internal-link breaks between ACTIVE docs.
- All moves via `git mv` to preserve history.
- One commit per category (per dispatch instruction).

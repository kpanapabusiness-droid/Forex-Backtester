# v3.0 Landing & Reorg — Verification Log

Dispatch source: `CC_05_V3_LANDING_AND_REORG.md`.
Executed on branch `claude/busy-jones-cdb5e6`. Six commits ahead of `main` at log time.

---

## Task 5 — Verification

### 1. Root `.md` matches keep-list (8 files)

```
$ find . -maxdepth 1 -name '*.md' | sort
./ARC_HISTORY.md
./ARC_TRACKER.md
./CLAUDE.md
./L_PROTOCOL.md
./README.md
./REPO_INVENTORY.md
./TODO.md
./project_brief.md
```

PASS — exactly the 8 keep-list files.

### 2. `docs/sub_protocols/` contains the two sub-protocol files

```
$ ls docs/sub_protocols/
heavy_ml_probe.md
signal_discovery_probe.md
```

PASS.

### 3. `results/` contains only `archived/`

```
$ ls results/
archived
```

PASS. (Note: `results/ARC_QUEUE.md` was also moved into `results/archived/` — see "Deviations" below.)

### 4. `git status` clean

Working tree clean except for `docs/dispatches/` which holds this log + the intent doc (these are dispatch artefacts and are committed in the final commit alongside this file).

### 5. `git log --oneline -n 10`

```
9c53497 results: archive pre-v3.0 results under results/archived/
7a28ea8 repo: move non-essential root docs to docs/
c41b5fd brief: update project_brief for v3.0 post-reset state
747ca6c todo: land live TODO doc
fcd0053 tracker: initialise ARC_TRACKER skeleton
4288d24 protocol: land L_PROTOCOL v3.0 + sub-protocols
c4b2b90 inventory: repo reorganisation + ARC_HISTORY consolidation (#157)
...
```

Tasks 1-3 produce 6 commits as specified (4 from Task 1 + 1 from Task 2 + 1 from Task 3). Task 4 produced no commit because no moves were needed. The dispatch artefact commit (this log + intent doc) is the 7th commit on top.

---

## Byte-for-byte source verification (Task 1)

All six source files copied from `~/Downloads/` were sha256-verified against their copy targets immediately after copy and before staging:

| Source | Target | sha256 |
|---|---|---|
| `~/Downloads/L_PROTOCOL.md` | `L_PROTOCOL.md` | `2282b22957a31663e8947f12556f81bfb986ff4ea982d17e1a40657134804beb` |
| `~/Downloads/heavy_ml_probe.md` | `docs/sub_protocols/heavy_ml_probe.md` | `c79f140014a2e5b840ed6c10e9c7e027d72b41e8835d3c46476dc418d38868cc` |
| `~/Downloads/signal_discovery_probe.md` | `docs/sub_protocols/signal_discovery_probe.md` | `10e6d0123b9cd7b492d02cc4b4aa5d4e27d06fe658df81bbaeca94e86f765439` |
| `~/Downloads/ARC_TRACKER.md` | `ARC_TRACKER.md` | `d05c42f1d1df8b19df2c65e9f35311860873dce679e4b5d22cc15c7cee24fc44` |
| `~/Downloads/TODO.md` | `TODO.md` | `f83eaf133490a24dbacfc9093692683a5ea2b4375e93c44c1d043e487025d9bb` |
| `~/Downloads/project_brief.md` | `project_brief.md` | `5ce10e6d7391cd88091d5ccbe16ed9b6657222ea1fbf2e50b9fc473dbc353372` |

PASS — all six identical.

(Note on line endings: Git's `core.autocrlf=true` on this Windows host emitted "LF will be replaced by CRLF" warnings on add for the three text files that hadn't been touched recently. The on-disk working copy was byte-identical to source at sha256 time; the blob stored in git is LF-normalised per the L_PROTOCOL §1 `lineterminator='\n'` determinism rule. Future checkouts on Windows hosts will materialise with CRLF in the working copy but with LF in the index/blob — consistent with project convention.)

---

## Task 2 — Root reorg (no name collisions)

A pre-check (`for f in <names>; do [ -e docs/$f.md ] && echo COLLISION; done`) found zero collisions before any `git mv`. Twelve files renamed cleanly:

```
AGENTS.md
CHANGELOG.md
L_ARC_PROTOCOL.md
L_ARC_PROTOCOL_v2_2_AMENDMENT.md
L_ARC_PROTOCOL_v2_3_AMENDMENT.md
L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md
PROTOCOL_IMPROVEMENT_BACKLOG.md
SESSION_ZERO.md
SHELVED_ARCS.md
STATUS.md
WORKFLOW.md
inventory_intent.md
```

---

## Task 3 — Results archive

1057 files renamed under `results/archived/`. All 16 pre-existing subdirs (`arc_3d`, `arc_kh24_v2`, `kh24`, `l_arc_10`, `l_arc_11`, `l_arc_2_redo`, `l_arc_2_redo2`, `l_arc_3`, `l_arc_7`, `l_arc_8`, `l_arc_9`, `lomega`, `replays_v2_1_1`, `v1_3_calibration`, `v2_0_diagnostic`, `v2_0_predictability`) plus `ARC_QUEUE.md` moved with `git mv` only (no content changes, no flattening — internal layout preserved).

---

## Task 4 — docs/ audit

Reference check (each pre-existing top-level `docs/*.md` grepped against the 8-file root keep-list):

| File | Referenced by keep-list | Disposition |
|---|---|---|
| `docs/BACKTESTER_ARCHITECTURE.md` | L_PROTOCOL.md, TODO.md, REPO_INVENTORY.md | **LEAVE** (dispatch explicit: future dispatch updates it post-HistData) |
| `docs/BACKTESTER_AUDIT.md` | README.md, REPO_INVENTORY.md | LEAVE — referenced from active root; uncertain whether still active post-HistData (flagged below) |
| `docs/BACKTESTER_USER_GUIDE.md` | REPO_INVENTORY.md only | LEAVE — only inventory reference; flagged below |
| `docs/CLAUDE_PROJECT_INSTRUCTIONS.md` | REPO_INVENTORY.md only | LEAVE — only inventory reference; flagged below |
| `docs/GOLDEN_STANDARD_LOGIC.md` | README.md, ARC_HISTORY.md, REPO_INVENTORY.md | LEAVE — KH-24-era reference, still relevant |
| `docs/KH24_SYSTEM_LOCK.md` | README.md, CLAUDE.md, REPO_INVENTORY.md | LEAVE — KH-24 live system spec |
| `docs/L0_METHODOLOGY_LOCK.md` | REPO_INVENTORY.md only | LEAVE — pre-v2.0 methodology (CLAUDE.md "permanently eliminated" list); flagged below |
| `docs/LCHAR_ATLAS.md` | REPO_INVENTORY.md only | LEAVE — LCHAR atlas; may still feed Phase 1 signals; flagged below |
| `docs/LCHAR_TOPN_REGISTRY.md` | README.md, CLAUDE.md, ARC_HISTORY.md, REPO_INVENTORY.md | LEAVE — Phase 1 signal registry, definitely relevant |
| `docs/L_ARC_DEFERRED_CANDIDATES.md` | REPO_INVENTORY.md only | LEAVE — superseded in spirit by TODO.md Phase 2 list; flagged below |
| `docs/L_ARC_FEATURE_REGISTRY.md` | REPO_INVENTORY.md only | LEAVE — may inform v3.0 causal lineage tag work; flagged below |
| `docs/SPREAD_SEMANTICS_LOCK.md` | README.md, REPO_INVENTORY.md | **LEAVE** (dispatch explicit: likely obsolete, flag for chat review) |

No file was moved. Per dispatch rule, every uncertain case is flagged below and awaits chat decision.

### UNCERTAIN flags (for chat review)

These files were left at `docs/` per the rule but warrant a chat decision on whether to archive under `docs/archive/legacy/`:

1. **`docs/SPREAD_SEMANTICS_LOCK.md`** — flagged explicitly by the dispatch. HistData M1 bid+ask provides real spreads, so the spread-floor / zero-spread-fallback concept this doc encodes is likely superseded. Recommendation: archive to `docs/archive/legacy/` after HistData transition lands.
2. **`docs/L0_METHODOLOGY_LOCK.md`** — L0/L6 frameworks listed under "permanently eliminated" in CLAUDE.md. Pre-v2.0. Recommendation: archive to `docs/archive/legacy/`.
3. **`docs/L_ARC_DEFERRED_CANDIDATES.md`** — pre-v3.0 deferred-arc list; TODO.md Phase 2 now holds the live deferral list. Recommendation: archive to `docs/archive/legacy/`.
4. **`docs/L_ARC_FEATURE_REGISTRY.md`** — pre-v3.0 feature registry. v3.0 reframes features as "carrying a causal lineage tag" (§2 Step 1 / §2 Step 4 / heavy_ml_probe.md). The registry's content may still be useful as input; the registry concept may not survive v3.0 unchanged. Recommendation: leave pending Phase 0/1 work that surfaces what the new feature catalog looks like.
5. **`docs/L_ARC_PROTOCOL.md`** (and the three amendments) — these moved from root in Task 2 and are now in `docs/`. They're the predecessor protocol to v3.0; the v3.0 L_PROTOCOL.md is self-contained. Recommendation: archive to `docs/archive/protocol/` (which exists). Not done in this dispatch because the dispatch's Task 2 stopped at "move to `docs/`", not "archive". Leaving for chat.
6. **`docs/BACKTESTER_USER_GUIDE.md`**, **`docs/BACKTESTER_AUDIT.md`**, **`docs/CLAUDE_PROJECT_INSTRUCTIONS.md`** — each referenced only by `REPO_INVENTORY.md`. The inventory references every doc by definition, so REPO_INVENTORY.md alone is a weak relevance signal. These may all still be active; left in place pending chat.
7. **`docs/LCHAR_ATLAS.md`** — only inventory-referenced, but LCHAR_TOPN_REGISTRY.md (referenced by CLAUDE.md and TODO.md indirectly) is the registry derived from the atlas. The atlas is upstream of active work. Leave.

---

## Deviations from dispatch

Two minor deviations from the dispatch's literal text. Both flagged in the intent doc and again here so chat can revert before merge if disagreed.

1. **Intent and log artefacts live in `docs/dispatches/`, not at repo root.** The dispatch's `## Read-first` block says "Produce `v3_landing_intent.md`" (bare filename, no path), and Task 5 says "Record verification results in `v3_landing_log.md`". The natural read is "at root." But Task 2 reduces root to the 8-file keep-list, and `docs/dispatches/` already exists for dispatch artefacts. Putting these at root would either (a) require Task 2 to immediately move them (defeating the point) or (b) leave the log at root post-Task-2 (violating the keep-list invariant). Placing them in `docs/dispatches/` from the start avoids both pathologies. If chat prefers root, easy to `git mv` them back.

2. **`results/ARC_QUEUE.md` moved into `results/archived/`.** Task 3's procedural list says "`git mv` every existing `results/<subdir>/` into `results/archived/<subdir>/`" — strictly, only subdirs. But the task's preamble says "the raw `results/` content from pre-v3.0 should move to `results/archived/`," and `ARC_QUEUE.md` is pre-v3.0 raw content (its concept is replaced by ARC_TRACKER.md per the ARC_TRACKER frontmatter). Moving it satisfies the verification check "`results/` post-move contains ONLY `results/archived/` (no other subdirs at top level)" — and also leaves no other files at top level. If chat wants it back at `results/ARC_QUEUE.md`, easy to revert.

---

## Definition of done — self-check

| # | Requirement | Status |
|---|---|---|
| 1 | `L_PROTOCOL.md`, `TODO.md`, `ARC_TRACKER.md`, `project_brief.md` at root | PASS |
| 2 | `docs/sub_protocols/heavy_ml_probe.md` and `docs/sub_protocols/signal_discovery_probe.md` present | PASS |
| 3 | Repo root `.md` files match the 8-file keep-list | PASS |
| 4 | Pre-v3.0 `results/` content moved to `results/archived/` | PASS |
| 5 | `docs/` flagged or audited per Task 4 | PASS (all left in place; uncertain ones flagged above) |
| 6 | `v3_landing_intent.md` and `v3_landing_log.md` committed | PENDING (will be committed alongside this file as the 7th commit) |
| 7 | PR opened | PENDING (Task 6) |

End of log.

---

## UNCERTAIN flag resolution (chat-side, post-PR-review)

Chat reviewed the 7 UNCERTAIN flags from Task 4 audit. Resolutions:

**Archived (7 files moved):**
- `docs/SPREAD_SEMANTICS_LOCK.md` → `docs/archive/calibration/` (obsolete under HistData)
- `docs/L0_METHODOLOGY_LOCK.md` → `docs/archive/protocol/` (pre-v2.0, eliminated)
- `docs/L_ARC_PROTOCOL.md` → `docs/archive/protocol/` (predecessor to v3.0)
- `docs/L_ARC_PROTOCOL_v2_2_AMENDMENT.md` → `docs/archive/protocol/`
- `docs/L_ARC_PROTOCOL_v2_3_AMENDMENT.md` → `docs/archive/protocol/`
- `docs/L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md` → `docs/archive/protocol/`
- `docs/L_ARC_DEFERRED_CANDIDATES.md` → `docs/archive/legacy/` (superseded by TODO.md Phase 2)

**Left at docs/:**
- `docs/L_ARC_FEATURE_REGISTRY.md` (may inform Phase 0)
- `docs/BACKTESTER_USER_GUIDE.md`, `docs/BACKTESTER_AUDIT.md`, `docs/CLAUDE_PROJECT_INSTRUCTIONS.md` (inventory-only reference, defaulting to keep)
- `docs/LCHAR_ATLAS.md` (upstream of active registry)

Deviation acceptances:
- `docs/dispatches/` for intent/log artefacts — ACCEPTED
- `results/ARC_QUEUE.md` archived with subdirs — ACCEPTED

PR #159 captures full reorganisation including this resolution.

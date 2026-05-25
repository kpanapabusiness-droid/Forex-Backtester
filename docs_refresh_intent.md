# docs_refresh_intent — CC_22 comprehensive documentation refresh

> **Dispatch:** `CC_22_DOCS_COMPREHENSIVE_REFRESH.md`
> **Branch:** `infra/docs-comprehensive-refresh-2026-05-25`
> **Worktree:** `xenodochial-montalcini-e95b3f` (cut from main @ `198d78f`)
> **Scope confirmed by chat:** full Tier 1-4 audit; commit `TODO_REFRESH_TEMPLATE.md` into `docs/templates/`
> **Status:** READ-FIRST complete; awaiting chat sign-off before any edits per dispatch §"Read-first" item 5

---

## §0 Recent-PR landscape (since last TODO refresh @ PR #190)

| PR | Date | Topic | Doc footprint expected |
|---|---|---|---|
| #193 | 2026-05-25 | Signal-level EET timezone alignment + canonical utility | PROTOCOL_RUNTIME §15.4, BACKTESTER_ARCHITECTURE signal-tz note, `signal_module_eet_audit_2026_05.md` (new), engine_capability_audit footer |
| #194 | 2026-05-25 | Amendment 5 — AUC-gated A2/A6 selection + parser v1.3.1 field | L_PROTOCOL header + §2 Step 5, `archive/L_PROTOCOL_v3_0_AMENDMENT_5.md`, ARC_CLOSURE_TEMPLATE v1.3.1, parser schema |
| #195 | 2026-05-25 | Canonical exit policy registry + `sl_partial_close_1r_runner_trail` primitive | PROTOCOL_RUNTIME §8c, BACKTESTER_ARCHITECTURE Exit-policies subsection, engine_capability_audit footer |
| #197 | 2026-05-25 | EET session semantics: distance.py / reset_floor.py / `compute_per_day_max_dd` + Amendment 6 | PROTOCOL_RUNTIME §15.5, BACKTESTER_ARCHITECTURE session-bucketing note, L_PROTOCOL §3 Boundary, **Amendment 6 archive + prefatory block (NOT YET DONE)**, engine_capability_audit footer |

PR #197's body literally says: *"Chat ratified **Amendment 6** — engine here; protocol text in parallel docs PR."* This dispatch IS that parallel docs PR — see §6 open question A6.

---

## §1 Doc inventory (44 active .md candidates; archive/attic excluded)

### Tier 1 — Active orientation docs (MUST be current)

| Doc | Current classification | Planned action |
|---|---|---|
| [L_PROTOCOL.md](L_PROTOCOL.md) | **PARTIAL** — has Amendments 1-5 prefatory blocks ✓; §"Boundary" inline-mentions Amendment 6 but no prefatory block; title still bare "v3.0" | Add Amendment 6 prefatory block (matches A1-A5 pattern); verify §3 / §2 Step 5 / §2 Step 6 unchanged elsewhere |
| [CLAUDE.md](CLAUDE.md) | **STALE — near-total rewrite** — header says "CC_07 / v3.0 runtime infrastructure landed"; body still references `L_ARC_PROTOCOL v2.0 / v2.1.2 / v2.2 / v2.3`, Pipeline E/D1, Arc 10 §16a Path A, etc. Pre-dates v3.0 protocol + all 8 engine PRs + Wave 1 closures | Rewrite to reflect: L_PROTOCOL v3.0 + Amendments 1-6 are active protocol; KH-24 still live; engine post-PR-#197; Phase 1 Wave 1 status (Arcs 8/10/11 closed, 5/7 closures in flight, retries pending); Phase 2 sub-protocols building. Preserve "Permanently Eliminated" list (still authoritative) and Risk Parameters. Drop v2.x vocabulary that no longer applies. |
| [project_brief.md](project_brief.md) | **STALE** — last updated 2026-05-20; says "HistData rebuild in progress" (done), "backtester awaits reconfiguration" (done), Phases all "ready to dispatch" (Phase 1 partially closed) | Update §1 "Current state" + "Where we are operationally" to reflect: HistData layer done; v3.0 protocol locked w/ Amendments 1-6; Phase 0 closed; Phase 1 Wave 1 in flight; engine post-PR-#197. Methodology section §2 mostly already correct (cite Amendments 1-6 explicitly). |
| [README.md](README.md) | **STALE** — "Current State" references `L_ARC_PROTOCOL.md v2.3`, Phase 0 GO under Path B, doc paths that don't exist (`SESSION_ZERO.md`, `STATUS.md`, `GOLDEN_STANDARD_LOGIC.md`, `docs/BACKTESTER_AUDIT.md` etc.). "Repository Layout" mentions `spread_floors_5ers.yaml` as "permanently deleted in PR-B"; "How to Run a Backtest" still pre-Phase-1. Documentation Hierarchy lists v1.x-era docs as Tier 1. | Rewrite "Current State" for post-PR-#197 reality. Rewrite "Start Here" pointing to `L_PROTOCOL.md` (not L_ARC_PROTOCOL). Rewrite Documentation Hierarchy table to remove dead doc references; align with current canonical Tier 1-4 set. Update "How to Run a Backtest" to reference the orchestrator path. Keep KH-24/Tool Stack/Risk Parameters sections (still accurate). |
| [TODO.md](TODO.md) | **PARTIAL** — TODO was refreshed in PR #190 (last full pass), then partially patched in PR #194 (Round 5 + Amendment 5 references + PR convention table). But: Round 6 stops at #189; "Engine state" line says "post-PR-#189"; PRs #193/#195/#197 not folded in; no Round 7 | FULL REFRESH per `docs/templates/TODO_REFRESH_TEMPLATE.md` (committed in this same PR). Fill `[FILL]` with content per CC_22 Task 1. Add Round 7 — Final Engine Consolidation. Update Engine state, Architectures wired, Doc set state, Standing items. Preserve historical rounds. |
| [WORKFLOW.md](WORKFLOW.md) | **PARTIAL** — §7 cadence table says template "Current: v1.2"; should be v1.3.1. Doesn't reference Amendment 4 Step 6 auto-dispatch, Amendment 5 dispatch-time four-gate rule, Amendment 6, or any PR #185-#197 work. §2 dispatch artefact pattern still accurate; §3 branch conventions still accurate. | Update §7 cadence table closure-template version → v1.3.1. Add §2.3 reference to Amendment 4 Step 6 auto-dispatch (post-§3-pass dispatch is now an engine action, not chat). Add §2 paragraph noting Amendment 5 four-gate enforcement is dispatch-time. Other sections unchanged. |
| [TODO_REFRESH_TEMPLATE.md → docs/templates/](docs/templates/TODO_REFRESH_TEMPLATE.md) | **NEW** — chat-confirmed to commit | Copy from `C:\Users\panap\Downloads\TODO_REFRESH_TEMPLATE.md` to `docs/templates/TODO_REFRESH_TEMPLATE.md`. Add row to TODO.md "Reminder — what gets updated when" cadence table. No content modification. |

### Tier 2 — Engine reference docs

| Doc | Current classification | Planned action |
|---|---|---|
| [docs/PROTOCOL_RUNTIME.md](docs/PROTOCOL_RUNTIME.md) | **PARTIAL** — has §8b (Amendment 3), §8c (exit policy registry, PR #195), §10b (Step 6 / Amendment 4), §13-14 (warmup), §15.1-15.4 (signal parity + signal-EET), §15.5 (EET session semantics, PR #197) ✓. But: §15 header still labels PR #187 (canonical is PR #189); §8b line "UTC broker-day boundary (locked)" pre-Amendment-6; no dedicated Amendment 5 architecture-selection subsection. | (a) Replace all "PR #187" references in §15 header + §15.1-15.4 with "PR #189". (b) Update §8b "Boundary" bullet to point to §15.5 / Amendment 6. (c) Add brief subsection (or §8d) describing Amendment 5 four-gate dispatch-time selection — currently only L_PROTOCOL has it. (d) Verify TOC/index lines consistent. |
| [docs/BACKTESTER_ARCHITECTURE.md](docs/BACKTESTER_ARCHITECTURE.md) | **PARTIAL** — has PR #189 / #195 / #197 sections ✓, A1-A6 architecture catalogue (incl. A5 not built), exit policy subsection, signal-module timezone responsibility note, session-bucketing responsibility note (PR #197). But: "Data layer (PR-A; PR #187 EET extension)" header is stale; "satisfies PR #187 Sub-change B" reference stale | Replace "PR #187" → "PR #189" in Data layer + Spread+Sim section headers. Verify A1-A6 catalogue current with A5 status (NOT BUILT). |
| [docs/audits/engine_capability_audit_2026_05.md](docs/audits/engine_capability_audit_2026_05.md) | **SEVERELY STALE — full refresh** — file is from PR #184 (2026-05-23 baseline). "Amendment 3 — Risk-normalised gates" section says "All items in this section are MISSING in the engine" — but PR #186 implemented all of them. "Step 6" section says "mostly MISSING" — but PR #188 built it. Executive summary counts (WIRED 30 / PARTIAL 24 / MISSING 13) totally outdated. | FULL refresh per dispatch Task 2. Re-evaluate every capability against current main. Update WIRED / PARTIAL / MISSING. Add footer section: "Post-Phase-1-engine-build summary — items resolved across PRs #185-#197". Estimated count shift: most Amendment-3 + Step-6 MISSING items → WIRED; signal-EET items → WIRED; orchestrator-step-5-run-context gap → status TBD (verify). |
| [docs/audits/signal_module_eet_audit_2026_05.md](docs/audits/signal_module_eet_audit_2026_05.md) | **CURRENT** — landed in PR #193; references current state accurately | No changes. (Optional: tiny housekeeping — update "Last updated" if convention demands; this doc doesn't have one.) |
| [docs/calibration/histdata_mt5_aggregation_parity_2026_05.md](docs/calibration/histdata_mt5_aggregation_parity_2026_05.md) | **PARTIAL** — content current; PR # references stale ("Source PR: PR #187") | Replace "PR #187" → "PR #189" throughout. Otherwise unchanged. |
| [docs/calibration/arc_10_signal_parity_rerun_2026_05.md](docs/calibration/arc_10_signal_parity_rerun_2026_05.md) | **PARTIAL** — content current for PR #189 baseline; PR # references stale; missing post-PR-#193 note on Arc 10 DLR State-B finding (Arc 10 v3.0 actually has another rerun trigger from #193) | Replace "PR #187" → "PR #189". Add §X note on PR #193 signal-EET finding (Arc 10's `lchar_dlr_long.py` was State B; rerun now also covers EET-correct D1 alignment). Add note that PR #197 daily-DD boundary change (Amendment 6) doesn't affect Arc 10's existing PASS-VIABLE numbers under UTC convention. |

### Tier 3 — Schema / parser docs

| Doc | Current classification | Planned action |
|---|---|---|
| [docs/templates/ARC_CLOSURE_TEMPLATE.md](docs/templates/ARC_CLOSURE_TEMPLATE.md) | **PARTIAL** — at v1.3.1 with `architectures_skipped_by_amendment_5` ✓, `step_6` block ✓, all Amendment 3 fields ✓. But §4.11 deployment-readiness checklist references "PR #187 hard requirement" (stale) | Update §4.11 reference "PR #187" → "PR #189". Otherwise current. |
| [scripts/tracker_parser/README.md](scripts/tracker_parser/README.md) | **PARTIAL** — Schema versioning table covers v1.0-v1.3 but does NOT mention v1.3.1 (Amendment 5 `architectures_skipped_by_amendment_5` field). Phase 2 tightening section accurate. | Add v1.3.1 row to Schema versioning detection table + brief mention in Phase 2 tightening section (`AMENDMENT_5_CUTOFF_ISO` cutoff requires field for post-cutoff PASS verdicts). |

### Tier 4 — Sub-protocol docs

| Doc | Current classification | Planned action |
|---|---|---|
| [docs/sub_protocols/heavy_ml_probe.md](docs/sub_protocols/heavy_ml_probe.md) | **CURRENT** — spec-only; engine build via parallel chat (PR #187) | No changes. (Could add status footer noting "build in flight via PR #187 (heavy_ml_probe PR-A)" but spec itself is unchanged and the dispatch's preference is structural changes only.) |
| [docs/sub_protocols/signal_discovery_probe.md](docs/sub_protocols/signal_discovery_probe.md) | **CURRENT** — spec-only; engine path landed via PR #175; 10k run pending | No changes. |

### Excluded (per dispatch §"Tier 5 / Tier 6" + discipline rules)

- All `archive/**` (frozen; Tier 5)
- `ARC_HISTORY.md` (frozen pre-v3.0 record; Tier 5)
- `ARC_TRACKER.md` (parser-managed; Tier 6)
- `results/<arc>/**` (per-arc, write-once; Tier 6)
- `attic/**` (quarantined MT5-era; not in CC_22 scope)
- `data/histdata/**.md` (operational artefacts from data download; out of scope)
- `docs/archive/**` (historical; Tier 5)
- `docs/CHANGELOG.md`, `docs/STATUS.md`, `docs/SESSION_ZERO.md` if they exist — per TODO.md §"Standing items": *"STATUS.md and CHANGELOG.md untouched by reset; user decides if reset/delete or retain as historical record"*. Surface to chat — see §6 open question A2.

### Other Tier-1-adjacent docs found (not in dispatch but exist)

| Doc | Action |
|---|---|
| `REPO_INVENTORY.md` | Out of scope per "Rarely" cadence + dispatch's listed Tier 1; classify CURRENT, skip. |
| `docs/AGENTS.md` | Not in dispatch; classify as out-of-scope (skip unless chat says otherwise). |
| `docs/BACKTESTER_AUDIT.md`, `docs/BACKTESTER_USER_GUIDE.md`, `docs/DATA_FOUNDATION.md`, `docs/GOLDEN_STANDARD_LOGIC.md`, `docs/KH24_SYSTEM_LOCK.md`, `docs/LCHAR_ATLAS.md`, `docs/LCHAR_TOPN_REGISTRY.md`, `docs/L_ARC_FEATURE_REGISTRY.md`, `docs/PROTOCOL_IMPROVEMENT_BACKLOG.md`, `docs/SHELVED_ARCS.md`, `docs/STATUS.md`, `docs/SESSION_ZERO.md`, `docs/CHANGELOG.md`, `docs/CLAUDE_PROJECT_INSTRUCTIONS.md` | Not listed in dispatch Tier 1-4. Skip unless chat explicitly opts in. KH24_SYSTEM_LOCK + DATA_FOUNDATION may be worth a quick scan for staleness — see §6 open question A3. |

---

## §2 Cross-reference verification plan (CC_22 Task 4)

After per-doc updates, verify these consistency triples:

| Reference | Where it appears | Action |
|---|---|---|
| Amendments 1-6 list | L_PROTOCOL.md header, PROTOCOL_RUNTIME.md §8b / §10b / §15.5, BACKTESTER_ARCHITECTURE.md, project_brief.md, CLAUDE.md | Verify all six referenced consistently after Amendment 6 backfill |
| Closure template version | L_PROTOCOL §6, WORKFLOW.md §7, ARC_CLOSURE_TEMPLATE.md, parser README | All should say v1.3.1 post-refresh |
| Parser version + Amendment-5 + Amendment-6 cutoff ISO | parser schema (`AMENDMENT_5_CUTOFF_ISO`), parser README, TODO.md standing items | Cutoff still placeholder pending PR-merge backfill (one-line follow-up PR per existing TODO standing item); flag if any doc claims final timestamp |
| PR # references for signal parity | PROTOCOL_RUNTIME §15, BACKTESTER_ARCHITECTURE Data/Spread sections, both calibration docs, ARC_CLOSURE_TEMPLATE §4.11 | All should say PR #189 (not PR #187) post-refresh |
| Daily-DD boundary convention | L_PROTOCOL §3 "Boundary", PROTOCOL_RUNTIME §8b + §15.5, BACKTESTER_ARCHITECTURE session-bucketing note, Amendment 6 archive | All point to Amendment 6 / 5ers EET / `compute_per_day_max_dd(boundary_convention="5ers_eet")` |
| Architecture-selection rule | L_PROTOCOL §2 Step 5, ARC_CLOSURE_TEMPLATE §1 `architectures_skipped_by_amendment_5`, parser README v1.3.1 row | All reference Amendment 5 four-gate dispatch-time enforcement |

Inconsistencies surface to chat per dispatch §"Risks" item 2.

---

## §3 Definition-of-done checklist for the eventual PR

Mirrors dispatch §"Definition of done":

1. `docs_refresh_intent.md` (this file) approved by chat
2. Tier 1: CLAUDE.md, README.md, project_brief.md, TODO.md, WORKFLOW.md, L_PROTOCOL.md updated
3. Tier 2: PROTOCOL_RUNTIME.md, BACKTESTER_ARCHITECTURE.md, engine_capability_audit_2026_05.md, both calibration docs updated; signal_module_eet_audit unchanged (already current)
4. Tier 3: ARC_CLOSURE_TEMPLATE.md + tracker_parser README updated
5. Tier 4: no changes
6. New artefacts: `docs/templates/TODO_REFRESH_TEMPLATE.md` committed; `archive/L_PROTOCOL_v3_0_AMENDMENT_6.md` created (pending §6 A6 confirmation)
7. Cross-reference triples verified
8. `Last updated:` dates set to 2026-05-25 on modified docs that carry one
9. Markdown linters / link checkers pass (if any exist — likely none, per audit doc)
10. PR title: `[DOCS] Comprehensive refresh — post-Phase-1-engine-build state alignment`

---

## §4 Estimated touch surface (files-modified count, post-this-intent)

- Created: 2 (`docs/templates/TODO_REFRESH_TEMPLATE.md`, `archive/L_PROTOCOL_v3_0_AMENDMENT_6.md` pending A6)
- Modified: ~12 (Tier 1: 6; Tier 2: 5; Tier 3: 2 — minus signal_module_eet_audit and sub-protocols)
- Total: ~14 files in PR

---

## §5 Discipline self-check (per dispatch §"Discipline rules")

- [x] No `archive/**` modification EXCEPT creating `archive/L_PROTOCOL_v3_0_AMENDMENT_6.md` per §6 A6 (NEW file is documentation completion of a chat-ratified amendment per PR #197 body, not new methodology — but explicit chat sign-off requested)
- [x] No `ARC_HISTORY.md`, `ARC_TRACKER.md`, `results/<arc>/**` changes planned
- [x] No code (`core/`, `scripts/`, `tests/`) changes planned
- [x] No new protocol amendments planned (Amendment 6 is already ratified per PR #197)
- [x] No methodology changes — documentation accuracy only
- [x] All identified Tier 1-4 docs covered or explicitly excluded

---

## §6 Open questions for chat (surface before edit phase per dispatch instruction)

### A1. PR numbering convention for in-flight refresh

TODO.md §6 "PR numbering convention" table currently records `CC_15 "PR #187 signal parity" | PR #189` etc. The TIER 2 + TIER 3 docs we'll update (PROTOCOL_RUNTIME §15, both calibration docs, ARC_CLOSURE_TEMPLATE §4.11) currently use "PR #187" for the signal-parity engine — confirmed stale per TODO mapping. **Plan: globally replace "PR #187" → "PR #189" in the affected docs and remove the bridging note where possible.** Risk: any closure doc cross-link that uses the dispatch's notional label would need its own bridge — but closure docs are out of scope (Tier 6).

Confirm: ✅ globally rewrite PR numbers in docs being touched, OR ❌ preserve dispatch-historical references with bridging notes.

### A2. STATUS.md / CHANGELOG.md / SESSION_ZERO.md

TODO §"Standing items" leaves these as "user decides if reset/delete or retain as historical record". They're referenced by README.md ("Start Here" sequence) which we're rewriting. Two options:

- **Option α:** Drop references from README; leave the files untouched as historical record.
- **Option β:** Drop references AND mark the files as obsolete (e.g., move under `docs/archive/`).

Recommend Option α — least invasive, matches dispatch's "docs-only" scope.

Confirm: α / β / other.

### A3. Tier-1-adjacent docs not in dispatch — quick-scan scope?

Several docs are referenced by `WORKFLOW.md` §9 "When in doubt" or `CLAUDE.md` reads-first that aren't in the dispatch's Tier 1-4: `docs/KH24_SYSTEM_LOCK.md`, `docs/DATA_FOUNDATION.md`, `docs/SHELVED_ARCS.md`, `docs/PROTOCOL_IMPROVEMENT_BACKLOG.md`. If they're stale and we leave them, the refreshed CLAUDE.md will still link to potentially-stale targets.

- **Option α:** Strict dispatch scope — touch only docs in dispatch Tier 1-4. Document any drift surfaced for a follow-up PR.
- **Option β:** Expand scope to include quick-scan + targeted update of any of the above showing material drift.

Recommend Option α (matches dispatch's scope-creep risk explicitly).

Confirm: α / β / list specific files to include.

### A4. CLAUDE.md "Permanently Eliminated" + "Cross-arc lessons" sections

Current CLAUDE.md has long "Permanently Eliminated" + "Cross-arc lessons" + "Vocabulary (post-Arc-10)" + "Activity catalog" sections written under v2.x framing. Under v3.0 protocol many of these terms (Pipeline E, Pipeline D1, Reverse FE, §16a Path A) are no longer protocol-canonical — they live in archive/. Plan options:

- **Option α:** Preserve "Permanently Eliminated" verbatim (still authoritative — KH-24 era + v2.x conclusions still bind). Drop or trim "Cross-arc lessons" / "Vocabulary" / "Activity catalog" sections that reference v2.x-only constructs. Add a one-line pointer to ARC_HISTORY.md for full cross-arc context.
- **Option β:** Preserve all sections verbatim, add a header note "Sections N-M are v2.x-era cross-arc context, retained for historical reading."

Recommend Option α (cleaner; reduces drift surface; matches dispatch preference for accuracy over preservation).

Confirm: α / β / other.

### A5. README.md "Documentation Hierarchy" table

Current README has a 13-row Documentation Hierarchy table; ~half reference docs that may not exist or are stale (LCHAR_TOPN_REGISTRY, GOLDEN_STANDARD_LOGIC, BACKTESTER_AUDIT, BACKTESTER_SCHEMA.json, BACKTESTER_TEMPLATE.yaml, SPREAD_SEMANTICS_LOCK, etc.). Plan: replace with a much shorter hierarchy aligned to the canonical Tier 1-4 docs from this refresh (≤8 rows).

Confirm: ✅ shorten to canonical set, OR keep full table + only remove confirmed-dead rows.

### A6. Amendment 6 archive file — does CC_22 own it?

**Strongest open question.** PR #197 body explicitly says: *"Chat ratified Amendment 6 — engine here; protocol text in parallel docs PR."* L_PROTOCOL.md §3 "Boundary" line 452 confirms: *"version-amended per Amendment 6...full text in the parallel L_PROTOCOL amendment docs PR."*

But the dispatch's discipline rules say *"DO NOT introduce new protocol amendments (Amendments 1-6 already locked)."*

Reading: Amendment 6 IS ratified by chat (PR #197 says so); only its archive file + L_PROTOCOL prefatory block are unwritten. CC_22 is the natural home for that documentation completion. The discipline rule prevents NEW methodology amendments, not documentation backfill for an already-ratified one.

**Plan (pending confirmation):** create `archive/L_PROTOCOL_v3_0_AMENDMENT_6.md` mirroring the Amendment 3/4/5 archive structure; add Amendment 6 prefatory block to L_PROTOCOL.md header (matching A1-A5 pattern); document scope (`compute_per_day_max_dd` boundary convention change + propagation pattern). Source the content from PR #197 body + L_PROTOCOL §3 "Boundary" current text + PROTOCOL_RUNTIME §15.5 "Amendment 6" paragraph.

Confirm: ✅ CC_22 owns Amendment 6 archive + L_PROTOCOL prefatory block, OR ❌ split into separate PR (then CC_22 leaves L_PROTOCOL's "parallel L_PROTOCOL amendment docs PR" reference as a future-PR pointer).

---

## §7 Notes / observations from read-first that didn't fit elsewhere

- `engine_capability_audit_2026_05.md` is the largest workitem by far (643 lines, requires recomputing WIRED/PARTIAL/MISSING per capability). Most prior MISSING items are now WIRED per PRs #185/#186/#188/#189/#193/#195/#197. Plan: targeted rewrite of each capability paragraph rather than full re-audit from scratch; preserve format + section structure; add post-PR-#197 footer per dispatch Task 2. Estimated 2-3 hours of careful work; largest single doc in PR.
- TODO.md current state is hybrid (PR #190 baseline + PR #194 partial patch). Full refresh per template is cleaner than continued patching.
- Most Tier 2-3 changes are mechanical (PR # rewrites + small additions). Bulk of writing time is CLAUDE.md + README.md + project_brief.md + engine_capability_audit refresh.
- No CI is configured for markdown link validation; cross-reference verification (§2) is manual but per dispatch Task 5.

---

End of intent. Awaiting chat sign-off per dispatch §"Read-first" item 5 before any further edits.

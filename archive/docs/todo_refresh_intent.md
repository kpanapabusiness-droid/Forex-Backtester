# TODO.md Refresh — Read-First Intent

> **Dispatch:** `CC_16_TODO_REFRESH.md`
> **Branch (planned):** `infra/todo-refresh-2026-05-23`
> **Status:** Read-first only. No TODO.md edits. Awaiting chat review.
> **Author:** CC (jovial-mcnulty-1b0855)
> **Date:** 2026-05-24
> **Prerequisite status:** PR #187 (signal parity, dispatch's notional number) = GitHub PR #189, **OPEN, not yet merged**.

---

## TL;DR — three things to resolve before TODO.md rewrite

1. **PR numbering mismatch.** The CC_16 dispatch was authored assuming "PR #187 = signal parity engine". The actual GitHub numbering disagreed because two parallel chats opened PRs ahead of mine:
   - GitHub **PR #187** = `[INFRA] heavy_ml_probe PR-A` (open, parallel heavy_ml_probe build chat)
   - GitHub **PR #188** = `[ENGINE] Step 6 causal audit framework + Amendment 4 + parser v1.3` (open, parallel mastermind chat)
   - GitHub **PR #189** = `[ENGINE] Signal parity` (my CC_15 PR, open)

   Recommendation: refreshed TODO.md should refer to the GitHub PR number with a one-time annotation noting the dispatch's notional number when necessary. Less confusion than referring to "PR #187 signal parity" forever.

2. **Some dispatch status claims conflict with current repo state.** Eight conflicts surfaced (§3 below). The biggest: dispatch marks Step 6 framework 🔴 NOT STARTED, but GitHub PR #188 (open) implements it; should be 🟡 IN PROGRESS.

3. **CC_15 (my PR #189) is not yet merged.** The dispatch's prerequisite says it should fire AFTER signal parity lands. Strictly, I should wait. Recommendation: chat decide whether to (a) push TODO refresh now in parallel (and update the "PR #189 merged" claim post-hoc), or (b) defer TODO rewrite until #189 merges. Either is defensible; (a) saves a round trip.

---

## §1 PR numbering reconciliation

### Dispatch's notional PR sequence

| Dispatch label | Dispatch intent | Actual GitHub mapping |
|---|---|---|
| PR #182 / #183 stack (parser) | Parser ship + backfill | Actually PR #179 (parser build, merged 2026-05-22) + #181 (template v1.2 retrofit) + #182 (CC_11 follow-up backfill). PR #183 was a separate ENGINE PR (reverted, superseded by #185). |
| PR #184 | Engine capability audit | ✓ matches (merged 2026-05-24) |
| PR #185 | Step 4 classifier persistence + holdout exclusion | ✓ matches (merged 2026-05-23) |
| PR #186 | Amendment 3 + A3/A4 + Step 4/5 fixes | ✓ matches (merged 2026-05-23) |
| **PR #187** | **Signal parity engine (CC_15)** | **MISMATCH — actually PR #189 (open). GitHub PR #187 = heavy_ml_probe PR-A.** |
| (not in dispatch) | Step 6 framework + Amendment 4 + parser v1.3 | GitHub PR #188 (open) — Step 6 framework is referenced as 🔴 in §8 of the dispatch but the implementation is in flight |

### Recommendation

In the refreshed TODO.md, refer to PRs by **GitHub number** throughout. Add a single annotation in the Round 6 section noting that "PR #187 in the CC_15/CC_16 dispatch numbering = GitHub PR #189." Avoid carrying the notional number forward.

---

## §2 Current repo state (read-first findings)

### §2.1 Closures landed (`docs/archive/arc_results/`)

```
ARC_3_RESULT.md, ARC_4_RESULT.md, ARC_4_RERUN_RESULT.md,
ARC_5_RESULT.md, ARC_6_RESULT.md, ARC_7_RESULT.md,
ARC_10_RESULT.md, PHASE_L6_ARC2_P3_RESULT.md
```

Arc 8 closure lives at `results/l_arc_8/ARC_CLOSURE.md` per ARC_TRACKER.
Arc 11 closure lives at `results/l_arc_11/ARC_CLOSURE.md`.

### §2.2 ARC_TRACKER (auto-updated by parser)

| Arc | Verdict | Re-eval verdict | Closure |
|---|---|---|---|
| l_arc_10 | PASS-VIABLE | PASS-DEPLOYABLE | results/l_arc_10/ARC_CLOSURE.md |
| l_arc_8 | FAIL (step 5) | FAIL | results/l_arc_8/ARC_CLOSURE.md |
| l_arc_11 | FAIL (step 5) | FAIL | results/l_arc_11/ARC_CLOSURE.md |

Active arcs:
- `arc_discovery_01` (signal_discovery_probe, locked H1, Step 1 infrastructure landed; full 10k run pending)

Arc 7 closure under v3.0 (PR #180) is OPEN — re-closure of an arc that already had a v2.1.2 closure (per CLAUDE.md). Status under v3.0 framework: "in flight."

### §2.3 Sub-protocol docs

- `docs/sub_protocols/heavy_ml_probe.md` ✓ exists
- `docs/sub_protocols/signal_discovery_probe.md` ✓ exists

### §2.4 Tracker parser

- `scripts/update_tracker_from_closure.py` ✓ exists
- `scripts/tracker_parser/` ✓ full package (schema.py, mapping.py, extract.py, rolling_state.py, README.md, parsed.log)
- Parser version: supports v1.0 / v1.1 / v1.2 currently. PR #188 bumps to v1.3.

### §2.5 Engine capability audit

- `docs/audits/engine_capability_audit_2026_05.md` ✓ exists, last edited via my CC_15 PR #189 (footer note added).

### §2.6 PRs in flight (OPEN, parallel chats)

- **PR #180** `[ARC 7 v3.0]` liquidity_sweep_reclaim_long — FAIL closure
- **PR #187** `[INFRA] heavy_ml_probe PR-A` — scaffolding + lineage gate + IO. PR-B/C/D/E/F to follow per build plan.
- **PR #188** `[ENGINE] Step 6 causal audit framework + Amendment 4 + parser v1.3` — implements Amendment 4, parser v1.3, manual CLI at `scripts/run_step_6.py`, six audit categories, Phase 2 cutoff `2026-05-23T06:20:59Z`. Locks closure template at v1.3 (dispatch's "v1.2.1" should be v1.3).
- **PR #189** `[ENGINE] Signal parity` — my CC_15 PR.

---

## §3 Conflicts between dispatch's prescribed statuses and actual repo state

| # | Dispatch claim | Actual repo state | Recommended status |
|---|---|---|---|
| 1 | Round 2: "L_PROTOCOL.md landed: 🟢"; old TODO has 🔴 | L_PROTOCOL.md present at repo root with Amendments 1+2+3; landed across PR #167 / #176. | 🟢 (dispatch correct) |
| 2 | Round 2: "sub-protocol docs landed: 🟢"; old TODO has 🔴 | `docs/sub_protocols/heavy_ml_probe.md` + `signal_discovery_probe.md` exist | 🟢 (dispatch correct) |
| 3 | Round 5: "parser BUILT: 🟢"; old TODO has ⚪ deferred | `scripts/update_tracker_from_closure.py` + `scripts/tracker_parser/` shipped via PR #179 | 🟢 (dispatch correct) |
| 4 | Dispatch §8: "Step 6 framework PR: 🔴" | PR #188 implements it. OPEN, with full Amendment 4 + parser v1.3 + 44 step_6 tests passing. | 🟡 IN PROGRESS (dispatch under-states) |
| 5 | Dispatch §10 doc-state: "Closure template v1.2.1: 🟢" | Latest merged template = v1.2 (PR #181). PR #188 (open) bumps to v1.3, not v1.2.1. | 🟡 v1.3 in flight via PR #188 |
| 6 | Dispatch §10 doc-state: "parser v1.0/1.1/1.2/1.2.1 schemas: 🟢" | Parser supports v1.0/v1.1/v1.2; v1.3 in flight via PR #188. No v1.2.1 schema version exists or is planned. | 🟡 v1.0/v1.1/v1.2 🟢 ; v1.3 in flight |
| 7 | Dispatch §6: "Arc 5/7 in flight on old engine" + "Arc 5/7 v3.0.1 retry under signal-parity engine: 🔴" | Arc 7 v3.0 closure PR #180 is OPEN (FAIL verdict). Arc 5 v3.0 — no visible PR or active branch found. Could be in a parallel chat unknown to this session. | 🟡 PR #180 in flight (Arc 7); Arc 5 status needs chat confirmation |
| 8 | Dispatch §5: "Phase 0a skipped by chat decision; Phase 0b skipped" | Old TODO: Phase 0a 🟢 DONE (completed PR-E.1.6, chat Path B verdict); Phase 0b 🔴 ready to dispatch | 🟢 Phase 0a CLOSED (Path B verdict); Phase 0b 🔴 still ready but framed as "skipped by chat decision" needs explicit chat confirm before changing |

### Conflict resolution recommendations

- Conflicts 1, 2, 3, 4, 5, 6: refresh per dispatch's framing but use correct GitHub PR numbers + correct template/parser versions (v1.3, not v1.2.1).
- Conflict 7 (Arc 5 status): chat needs to confirm Arc 5 v3.0's actual state. If in a parallel chat, what's the branch name?
- Conflict 8 (Phase 0a/0b): chat needs to confirm the "skipped" framing vs the existing "DONE under Path B" framing. The two framings differ in intent — "skipped" implies a deliberate non-execution; "DONE Path B" implies execution-with-attribution. Per current TODO + the BACKTESTER_ARCHITECTURE doc, the reality is Path B (executed with attribution). Recommend keeping the Path B framing.

---

## §4 Proposed refreshed TODO.md outline

Following the dispatch's §"Target structure" §§1-12. Diff from current TODO.md:

### §4.1 Header (§1 dispatch)
- Update `Last updated` to current date.
- Keep existing description sentence.

### §4.2 Status legend (§2 dispatch)
- Unchanged.

### §4.3 Current state — quick view (§3 dispatch)
Major rewrite. Proposed content:

- **Phase 0 — Framework validation:** 🟢 CLOSED (Path B verdict per PR-E.1.7; Phase 0b ready but deferred — see §"Standing items" for chat policy on whether to keep deferred or mark skipped).
- **Phase 1 — Arc 1-11 re-runs:** 🟡 IN PROGRESS. Wave 1 closures: Arcs 8 (FAIL), 10 (PASS-DEPLOYABLE under Amendment 3 re-eval), 11 (FAIL). Arcs 5 + 7 closures: in flight (Arc 7 PR #180 open; Arc 5 status pending chat confirmation). Wave 1 retries (5, 7, 10) under signal-parity engine: 🔴 NOT STARTED.
- **Phase 2 — Sub-protocols + new signals:** 🔴 NOT STARTED (blocked on Phase 1 closure + parallel sub-protocol engine builds).
- **Engine state:** post-PR-186 + signal-parity (PR #189 open). Steps 1-5 wired; A1/A2/A3/A4/A6 architectures wired; A5 not built; Step 6 framework in flight via PR #188.
- **Active parallel chat work:**
  - heavy_ml_probe build (PR #187 PR-A open; PR-B/C/D/E/F to follow)
  - Step 6 framework + Amendment 4 (PR #188 open)
  - signal_discovery_probe 10k local run by user (per dispatch §7)

### §4.4 Reset checklist Rounds 1-5 (§4 dispatch)
Update per dispatch + conflicts:

- Round 1: 🟢 (unchanged)
- Round 2: all 🟢 (4 task statuses flip 🔴 → 🟢 per dispatch §4)
- Round 3: 🟢 (unchanged)
- Round 4: 🟢 (project folder cleanup done per dispatch; sync notes added)
- Round 5: 🟢 closure template (v1.2 merged, v1.3 in PR #188); 🟢 parser BUILT via PR #179 (NOT #182/#183 as dispatch incorrectly claims). Add note that v1.3 is in flight via PR #188.

### §4.5 NEW Round 6 — Engine consolidation (§4 dispatch)

| PR # GitHub | Description | Status |
|---|---|---|
| #184 | Engine capability audit 2026-05 | 🟢 merged 2026-05-24 |
| #185 | Step 4 fitted-classifier persistence + holdout-window training fix | 🟢 merged 2026-05-23 |
| #186 | Amendment 3 implementation + A3/A4 wiring + Step 4/5 fixes | 🟢 merged 2026-05-23 |
| #189 (dispatch label "#187") | Signal parity engine (mid features + trail mid + 5ers EET) | 🟡 OPEN at refresh time; will be 🟢 once merged |

### §4.6 Phase 0 — Framework validation (§5 dispatch)
Mark 🟢 CLOSED. Keep Path B framing (existing TODO accurate). Drop the "ready to dispatch" line for Phase 0b — replace with "deferred per chat decision (KH-24 base through v3.0 protocol not expected to add information beyond Path B verdict)".

### §4.7 Phase 1 — Arc 1-11 (§6 dispatch)
Update with current Wave 1 closure state. Add Wave 1 retry queue under signal-parity engine. Wave 2 dispatch gating: signed up but blocked on Wave 1 retries.

### §4.8 Phase 2 — Sub-protocols + new signals (§7 dispatch)
Per dispatch. Note heavy_ml_probe build is in PR #187 (GitHub) PR-A; further PRs B/C/D/E/F follow. signal_discovery_probe 10k run is user-side workstation operation.

### §4.9 NEW §8 — Engine consolidation queue (§8 dispatch)
Per dispatch's table. Notes adjustments:
- "Step 6 framework PR" → 🟡 IN PROGRESS via PR #188 (NOT 🔴)
- "Audit doc footer (items MISSING → WIRED)" → 🟡 PR #189 begins this with the signal-parity footer; further audits as engine items WIRE up
- "Parser v1.2.1 PASS-validation Phase 2 tightening" → reframe as "parser v1.3 PASS validation Phase 2 tightening (cutoff 2026-05-23T06:20:59Z)" per PR #188 — version is v1.3, not v1.2.1

### §4.10 §9 — Standing items / open questions
Per dispatch §9 ADD section:
- **Signal parity gap (NEW)** — IN PROGRESS via PR #189; will be RESOLVED on merge.
- Keep KH-24 live VPS health, cross-asset data (DXY/US10Y/SPX on hold), discovery follow-up dependencies.

### §4.11 §10 — Doc set state
Per dispatch §10 with version corrections (v1.3 not v1.2.1). Add my two new calibration docs (`histdata_mt5_aggregation_parity_2026_05.md` + `arc_10_signal_parity_rerun_2026_05.md`).

### §4.12 §11 — Ideas parked
Per dispatch — mostly unchanged. Confirm 4 items.

### §4.13 §12 — Reminder — what gets updated when
Per dispatch — add `docs/calibration/*` row; confirm parser auto-updates ARC_TRACKER.

---

## §5 Files CC will touch in the rewrite

Only `TODO.md`. Per dispatch discipline rules: do NOT touch ARC_TRACKER.md, ARC_HISTORY.md, per-arc closure docs, L_PROTOCOL.md, or sub-protocol docs.

Plus, `todo_refresh_intent.md` (this file) lands as a sibling artefact under repo root, like `signal_parity_intent.md` did for CC_15.

LOC estimate: TODO.md grows from 232 lines → ~280-320 lines after refresh (new Round 6, refactored Phase 0 framing, expanded Phase 1 status, new Engine consolidation §8 section, updated calibration doc rows).

---

## §6 Discipline rule confirmations

- ✓ Will NOT touch `ARC_TRACKER.md` (parser-managed)
- ✓ Will NOT touch `ARC_HISTORY.md` (frozen)
- ✓ Will NOT touch per-arc closure docs in `results/`
- ✓ Will NOT modify L_PROTOCOL.md or sub-protocol docs
- ✓ Refresh covers ONLY `TODO.md` (+ intent doc)
- ✓ Preserve file's existing structure and section ordering

---

## §7 Open questions for chat before rewrite

1. **Push refresh now or after PR #189 merges?** Recommendation: push now in a separate branch (`infra/todo-refresh-2026-05-23`) and use "PR #189 → merging" / "expected merge today" framing. Update post-merge with a one-line edit if needed.

2. **PR numbering convention in refreshed TODO.md.** Recommendation: use GitHub numbers everywhere; one-time annotation in Round 6 that dispatch's "PR #187 signal parity" = GitHub PR #189.

3. **Phase 0 framing.** Dispatch says "skipped by chat decision"; current TODO says "DONE under Path B." Recommendation: keep Path B framing (it's truthful; the work happened); add a note that Phase 0b is deferred per chat decision (also true).

4. **Arc 5 v3.0 status.** No PR or branch found in this session. Is it in a parallel chat? What's its current step?

5. **Closure template version target.** PR #188 implements v1.3 (not v1.2.1 as dispatch §10 says). Recommendation: track v1.3 as the target. The "v1.2.1" was likely a placeholder in the dispatch text.

6. **MASTERMIND_LIVE_STATE.md.** Dispatch notes this is chat-side memory, not part of refresh. Confirm: should refreshed TODO.md make any reference to it at all, or just skip?

---

## §8 Read-first end-of-turn

Per dispatch: **end turn for chat review**. No `TODO.md` modifications. No git changes.

When chat resumes:
1. Confirm answers to §7 open questions
2. Confirm PR numbering convention
3. CC executes the rewrite per §4 outline
4. PR opens with title `[INFRA] TODO.md refresh — post-PR-187 state alignment` (or updated number if chat prefers GitHub-actual numbering)

End of intent.

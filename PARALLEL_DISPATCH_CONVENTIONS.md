# Parallel Dispatch Conventions

This document governs the four substantive CC dispatches running in parallel after the v2.5 amendment landed on `main`. Each dispatch reads this file as part of its read-first sequence.

## Branches and file ownership

| Dispatch | Branch | Owns (read+write) | Reads only |
|---|---|---|---|
| System-level re-eval | `system-level/closed-arc-reeval` | `results/system_level_track/**` | All canonical refs below |
| Currency strength descriptive | `analysis/currency-strength-descriptive` | `results/currency_strength/**` | All canonical refs below |
| t=x contamination test | `probe/tx-contamination-test` | `results/tx_contamination/**` | All canonical refs below |
| Free-reign discovery | `discovery/free-reign-v1` | `results/free_reign/**`, `scripts/free_reign/**` | All canonical refs below |

**Canonical references (read-only across all parallel dispatches):**

- `L_ARC_PROTOCOL.md`
- `L_ARC_PROTOCOL_v2_3_AMENDMENT.md`
- `L_ARC_PROTOCOL_v2_5_AMENDMENT.md`
- `L_ARC_PROTOCOL_v2_x_AMENDMENT_PROPOSAL.md` (if present)
- `docs/KH24_SYSTEM_LOCK.md`
- `docs/SPREAD_SEMANTICS_LOCK.md`
- `docs/calibration_decisions/SPREAD_FLOOR_CALIBRATION_DECISION_2026-05-17.md`
- `configs/spread_floors_5ers.yaml`
- `CLAUDE.md`
- `project_brief.md`
- `PARALLEL_DISPATCH_CONVENTIONS.md` (this file)

**No parallel dispatch modifies a canonical reference.** If a parallel dispatch believes a canonical reference needs to change, it logs the concern to `PROTOCOL_IMPROVEMENT_BACKLOG.md` (NEW item only, never modify existing) and continues without the change.

## Shared meta-files (write-coordinated)

These files may be touched by parallel dispatches, but writes are append-only and section-scoped:

- `STATUS.md` — each parallel dispatch appends a status entry under its own dated section when it begins, when it produces a meaningful milestone, and when it closes. Format below.
- `CHANGELOG.md` — each parallel dispatch appends a top-of-file entry on closure (PR open). Format below.
- `PROTOCOL_IMPROVEMENT_BACKLOG.md` — each parallel dispatch may append NEW items under "Open" with unique IDs. Existing items (open or closed) are NOT modified by parallel dispatches under any circumstances.
- `results/ARC_QUEUE.md` — owned by main chat / coordination dispatch only. Parallel dispatches do NOT modify this file.

If two parallel dispatches need to modify the same meta-file, the later merge resolves the conflict by appending both entries in chronological order. No content from either dispatch is dropped.

## STATUS.md entry format (append-only by parallel dispatches)

```
### YYYY-MM-DD — <dispatch short name> — <milestone>

<one-paragraph description of what was done or completed, with commit hash and result file pointer>
```

## CHANGELOG.md entry format (append-only by parallel dispatches at PR open)

```
## <DISPATCH NAME> | YYYY-MM-DD | <verdict>

<two-paragraph summary: what was done, headline result, any cross-arc items raised>

- Branch: <branch name>
- Final commit: <commit hash>
- PR: <PR URL>
- Result document: <path>
```

## Anchor preservation rule (binding on all parallel dispatches)

KH-24 deployed configuration is locked. No parallel dispatch may:

- Modify `docs/KH24_SYSTEM_LOCK.md` (wherever it lives)
- Modify the deployed `KH24_EA.mq5` EA code
- Modify the live KH-24 spread floor, exposure cap, or 1H filter configuration
- Recommend deployment changes to KH-24 directly (recommendations go to chat via the dispatch's closure document)

Parallel dispatches MAY:

- Evaluate KH-24 base or KH-24 full as benchmark / anchor checks (read-only on KH-24 config)
- Recommend modifications to KH-24 in their closure documents for chat-side review
- Compare candidates against KH-24's worst-fold ROI / DD numbers

## Spread floor and data source rules (binding)

All parallel dispatches use:

- `configs/spread_floors_5ers.yaml` at the version on `main` at branch-cut time (do not update mid-dispatch)
- HistData M1 OHLC as primary data source for any 2010-2019 IS work (depends on separate HistData pipeline being on `main`; if not present, dispatches falling back to the interim partition per v2.5 §D are permitted)
- 5ers MT5 data as secondary / ratification source for OOS

If a parallel dispatch finds a spread floor or data source issue, it logs to `PROTOCOL_IMPROVEMENT_BACKLOG.md` (new item) and continues using the existing config. It does NOT change the config mid-dispatch.

## Determinism rule (binding)

All result files produced by parallel dispatches must:

- Be reproducible from seed
- Carry a sha256 in the result manifest
- Use `lineterminator='\n'` on any `to_csv()` call for cross-platform reproducibility

## Cross-dispatch communication

Parallel dispatches do not communicate with each other during execution. Cross-dispatch findings (e.g. free-reign discovery finds something that informs the system-level re-eval) are surfaced via:

1. The dispatch's closure document and CHANGELOG entry
2. New items in `PROTOCOL_IMPROVEMENT_BACKLOG.md`

Chat reviews findings after all four dispatches close and decides next steps (which may include a follow-up coordination dispatch).

## PR merge order (recommended)

When the four parallel dispatches close and open PRs, the recommended merge order is:

1. `CC_TX_CONTAMINATION_TEST.md` — smallest, fastest, informs Pipeline D design downstream
2. `CC_CURRENCY_STRENGTH_DESCRIPTIVE.md` — descriptive only, no protocol mutation
3. `CC_SYSTEM_LEVEL_TRACK_REEVAL.md` — substantive, references v2.5
4. `CC_FREE_REIGN_STRATEGY_DISCOVERY.md` — largest, most likely to surface cross-cutting findings

This order is a recommendation, not a constraint. PRs may be merged as they pass review.

## Dispatch halt conditions

A parallel dispatch HALTS and ends turn (no PR merge attempt) if any of:

- A canonical reference is found to be inconsistent with the dispatch's needs
- Lookahead is detected in any feature or pipeline component
- A determinism failure is encountered that cannot be resolved within the dispatch scope
- The HistData 2010-2019 pipeline dependency is required but not satisfied AND the interim partition fallback is not permitted for the work in question
- KH-24 anchor reproduction check fails (worst-fold ROI / DD drift beyond tolerance)

On halt, the dispatch:

1. Writes a HALT entry in its result document explaining what triggered the halt
2. Adds a new entry to `PROTOCOL_IMPROVEMENT_BACKLOG.md`
3. Opens a PR (do-not-merge) with title `[HALT] <dispatch name> — <reason>`
4. Ends turn

## End of file

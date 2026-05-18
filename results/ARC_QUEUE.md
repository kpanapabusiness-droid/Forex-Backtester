# Arc Queue

> State file for L arc parallel execution under L_ARC_PROTOCOL v2.2.
> Read by CC at arc-open. Updated by CC at queue transitions. Analyst owns Unrun ordering and signal source selection.
> Coordination: git-level. Queue file is the only shared state between concurrent CC sessions. Each session pulls before edit; on push conflict, pulls and retries.

---

## Active (in-flight)

_None._

---

## Unrun

In FIFO order. Topmost is next. Analyst adds entries here as signals become ready.

_Empty._

### Entry format

Each Unrun entry references either a registry entry or a standalone signal spec:

```markdown
- [ ] **Arc <N>** — <signal name>
  - Spec: `LCHAR_TOPN_REGISTRY.md` Entry <K>     # for registry-derived signals
  - Spec: `signal_spec_<name>_v<version>.md`     # for analyst-designed signals
```

---

## Closed

Most recent first. Populated from actual closure docs in `results/arc_<N>/` and `docs/arc_results/`.

_To be populated by the housekeeping pass — see `CC_HOUSEKEEPING_DISPATCH.md`._

---

## Conventions

### Selection

**FIFO:** CC picks the topmost Unrun entry at arc-open dispatch. Analyst reorders Unrun by direct edit if priorities shift.

### Signal source

Each Unrun entry must reference either a registry entry or a standalone signal spec doc. CC reads the referenced doc at arc-open. If neither is provided, CC halts with halt summary "queue entry missing signal source."

### Status transitions

- `Unrun → Active` at arc-open. CC adds timestamp + branch name + live doc path. Commits `arc-<N> open`.
- `Active → Closed-{KILL|HALT|SHIPPED}` at arc close. CC adds disposition + closure doc path. Commits `arc-<N> closed <disposition>`.

### Parallel arcs

Multiple Active entries permitted. Each on its own `phase/arc-<N>` branch, own `results/arc_<N>/` folder. One Active entry per CC chat session.

To run N arcs in parallel: open N CC chats. Each session reads this queue file, picks the topmost Unrun, and proceeds.

### Concurrency

Git-level coordination. Each session pulls before queue edit; on push conflict, pulls and retries with whatever Unrun entry is now topmost. Sufficient for dispatch cadences of minutes apart.

### Re-runs

A closed arc re-evaluated under a new protocol version gets a new Unrun entry (`Arc K redo`), not by mutating the existing Closed entry. Closed entries are historical record.

### SHIPPED disposition

Only assigned after step 6 WFO pass-deployable + analyst ship decision (per §10). Steps 1-5 dispatches never produce SHIPPED — only KILL, HALT, or step-5-complete-pending-WFO (Active remains).

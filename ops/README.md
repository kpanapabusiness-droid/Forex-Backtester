# ops/ — Operator Infrastructure (OUTSIDE the discovery gate)

> **This directory is operator orchestration, NOT gated code.** Nothing here is part of
> the L_PROTOCOL discovery gate, the canonical measurement core, or the `discovery/`
> protocol. `run_fleet.py` only creates worktrees, syncs them to `origin/main`, spawns
> the `claude` CLI, and writes logs. It never commits, never touches the core, never
> edits a `discovery/` doc. Treat it like a launchd/systemd unit that happens to live in
> the repo.

## What `run_fleet.py` is

A hands-off launcher for the continuous discovery fleet. It runs the fleet for a bounded
session by respawning **fresh** `claude -p` sessions, one per arc-id range, until a
wall-clock budget elapses or the `discovery/STOP` sentinel appears on `origin/main`.

Each spawned session is the real per-chat discovery run (the dispatch you pass via
`--dispatch`). The launcher is deliberately dumb: its entire blast radius is

* create / reuse sibling git worktrees (one per range, **outside** the repo tree),
* hard-sync each worktree to `origin/main` before every spawn,
* spawn one fresh `claude` session per range and stream its output to a log,
* respawn a fresh session when one exits (never `--resume`).

Why fresh every time: a chat cannot clear its own context window (the transcript is
append-only), so it has a FINITE arc budget — it runs to a graceful handoff (finish the
current arc → commit + push → stop). Continuity is **handoff + bootstrap, NOT self-debloat**
(protocol §10): step-(k) re-orientation drops working detail from active attention but does
NOT free the context window. A fresh session then bootstraps purely from the shared log
(pull main, read protocol + log + LESSONS + registry, resume at the highest arc-id in its
range + 1). The launcher embodies exactly this — it **never** `--resume`s; it always spawns a
NEW session per range, and the LOG (not the prior transcript) carries the learning forward.

## The model value

`--model` defaults to **`claude-opus-4-8`** (the full name). This was verified:

* `claude --help` lists the full name `claude-opus-4-8` as the canonical `--model` example,
* the CLI parser accepts it, and
* a live session launched with it runs (see test results below) — the backend accepts it.

If a future CLI/model change rejects the full string, fall back to the alias: `--model opus`.

## Run command

The operator runs this from a shell where **conda `base` is activated** (so the spawned
sessions inherit the engine environment — pandas/sklearn/lightgbm/etc.). On this machine:

```powershell
# from a conda-base-activated PowerShell, repo root:
conda activate base
python ops\run_fleet.py --dispatch <path-to-RESUME_DISPATCH.md> `
  --ranges 1000,2000,3000 --hours 8
```

`--dispatch` points at the resume-dispatch file (e.g. `RESUME_DISPATCH.md`, or the
in-repo `discovery/CONTINUOUS_RUN_DISPATCH.md`). Its **full contents** become each
session's prompt, with `\n\nRange: <range>. Resume.` appended so the session knows its
assigned arc-id block.

> **Environment matters.** The child `claude` processes inherit the launcher's environment.
> Launch from an activated conda `base` shell so the sessions' `python`/engine calls
> resolve to the env that has the discovery dependencies. Launching from a bare shell will
> spawn sessions whose engine calls fail.

## CLI

```
python ops/run_fleet.py --dispatch <path> [options]

  --dispatch <path>            (required) resume-dispatch file; its contents become the prompt
  --repo <path>                repo root (default: C:\Users\panap\Documents\Forex-Backtester)
  --ranges 1000,2000,3000      comma-separated arc-id ranges, one CC chat each
  --hours 8                    wall-clock budget; after it elapses, stop LAUNCHING new
                               sessions (in-flight sessions are NEVER killed mid-arc)
  --model claude-opus-4-8      --model passed to claude (fall back to 'opus' if rejected)
  --max-turns 250              --max-turns per session (turn-runaway guard)
  --max-sessions-per-range 100 hard ceiling on respawns per range (session-runaway guard)
  --min-healthy-secs 120       a session exiting faster than this counts as a failed start
  --worktrees-root <path>      where sibling worktrees live
                               (default: C:\Users\panap\Documents\fx-worktrees)
  --claude <path>              path to claude.exe
                               (default: C:\Users\panap\.local\bin\claude.exe, then PATH)
  --permission-mode acceptEdits  --permission-mode for claude
  --allowed-tools Bash,Read,Edit,Write,Agent  --allowedTools for claude (Agent = council sub-agents; see note)
  --add-dir <dir>              extra --add-dir for claude (repeatable)
  --output-format <fmt>        optional --output-format for claude (default: text)
  --prompt-stdin               deliver the prompt via stdin instead of an argv positional
  --setup-only                 create/reuse + hard-sync the worktrees, then exit (no spawns)
  --dry-run                    print exactly what WOULD happen, then exit (spawns nothing)
```

### The exact session that gets spawned

```
claude -p "<dispatch contents>\n\nRange: <range>. Resume." \
  --model claude-opus-4-8 \
  --permission-mode acceptEdits \
  --allowedTools "Bash,Read,Edit,Write,Agent" \
  --max-turns 250
```

> **`Agent` is included on purpose** (the dispatch's literal spec was `Bash,Read,Edit,Write`). The
> mandatory `/llm-council-discovery` survivor stress-test spawns sub-agents (5 lenses + 5 reviewers +
> 1 chairman); the sub-agent tool is named **`Agent`** (legacy alias `Task`). Without it the council is
> hard-denied at the spawn step in headless mode. See the note below for the verification.

cwd = `<worktrees-root>\<range>`. stdin is `/dev/null` (so claude does not stall waiting on
stdin when the prompt is an argv positional). Output (stdout+stderr) streams to
`ops/logs/<range>/<timestamp>.log`.

## Worktrees

* One per range, on branch `disco/<range>`, at a sibling path **outside** the repo:
  `<worktrees-root>\<range>` (e.g. `C:\Users\panap\Documents\fx-worktrees\1000`).
* Created on demand: `git -C <repo> worktree add <wt> -b disco/<range> origin/main`
  (or reused if the branch/worktree already exists).
* Before **every** spawn the worktree is hard-synced:
  `git fetch origin` → `git reset --hard origin/main` → `git clean -fd`.
  This is deliberate: a crashed session's in-flight, unpushed arc is discarded — clean slate
  each session. All *real* work is pushed per-arc, so nothing of value is lost.

## Safety rails (this runs unattended)

| Rail | Behaviour |
|------|-----------|
| `--hours` budget | After it elapses, **stop launching** new sessions. In-flight sessions finish; they are never killed mid-arc. |
| `--max-sessions-per-range` | Hard ceiling on respawns per range (default 100). |
| `--max-turns` | Per-session turn cap (default 250) so one session can't run away. |
| Crash-loop backoff | A session exiting in `< --min-healthy-secs` (default 120s) is treated as a failed start. Backoff grows 10s → 30s; after **3 consecutive** fast exits the range **HALTs** with a diagnosis (auth / PATH-to-claude / bad `--model`). A healthy session resets the counter. |
| STOP sentinel | Checked (after a `git fetch`) **before every spawn**, both as a local file `<repo>\discovery\STOP` and on `origin/main:discovery/STOP`. Present → finish the range cleanly, no new spawn. |
| Ctrl+C | First SIGINT: graceful — stop launching, let in-flight finish. Second SIGINT: force-kill in-flight children. Exits non-zero with a per-range summary. |

> **`--max-turns` vs `--min-healthy-secs`:** if you deliberately set a tiny `--max-turns`
> for testing, a *successful* session can exit in well under `--min-healthy-secs` and be
> miscounted as a crash-loop fast-exit. Lower `--min-healthy-secs` to match (the live tests
> below use `--min-healthy-secs 20`). In production with `--max-turns 250` a real session
> runs minutes and a failed start dies in seconds, so the default 120s cleanly separates them.

## Logs

* Per session: `ops/logs/<range>/<timestamp>.log` (header with cmd + cwd, the child's full
  stdout/stderr, footer with exit code + duration).
* Fleet-level heartbeats: `ops/logs/_fleet/fleet_<timestamp>.log` (also echoed to the console).
* `ops/logs/` is **git-ignored** (`ops/.gitignore`) — runtime artifacts, never committed.
  Logs are written under the **main repo's** `ops/logs/`, not the worktrees, so the
  per-spawn `git clean -fd` in the worktrees never touches them.

## What it will NOT do

* It never `git commit`s or `git push`es anything — the spawned CC sessions commit + push
  their own `discovery/` docs.
* It never edits the canonical core or any `discovery/` protocol/doc.
* It never `--resume`s a session.

## Notes / gotchas

* **Council skill / tools (verified).** The mandatory `/llm-council-discovery` survivor stress-test
  spawns sub-agents (its steps literally "spawn all 5 lenses as sub-agents", then "5 fresh reviewers",
  then a chairman). The sub-agent tool is **`Agent`** (legacy alias `Task`), so the default allow-list
  here is `Bash,Read,Edit,Write,Agent` — NOT the dispatch's literal `Bash,Read,Edit,Write`, which would
  break the council. Verified live, headless, `claude-opus-4-8`:
  - `--allowedTools "Agent"` → a headless session spawns a sub-agent successfully.
  - The skill **does** invoke in `-p` mode (returns its step-0 input-validation refusal) — the `Skill`
    tool is NOT gated by `--allowedTools`; you do not need to allow-list it.
  - With only `Bash,Read,Edit,Write` the skill reaches step 0/1 but is hard-denied at the sub-agent
    spawn (headless `-p` cannot prompt for approval); `acceptEdits` does NOT auto-approve non-edit tools.
  - Sub-agents inherit the parent allow-list and cannot spawn further sub-agents — no escalation beyond
    what the parent can already do.
* **Concurrent pushes to main** occasionally race; the CC sessions resolve that in-session
  (fetch/rebase/retry). The launcher does not serialize pushes — worktree `fetch`/`reset`
  are local. (Git plumbing inside the launcher is serialized with a lock to avoid ref races.)
* **Throughput.** WFO is CPU-bound and single-threaded; 2–3 concurrent ranges is the sweet
  spot on a 6-core box (per the overseer handover). More ranges degrades per-arc throughput.

## Test results

See `TEST_RESULTS.md` in this directory for the full test-plan run (dry-run, worktree
creation, single-range respawn, STOP detection, crash-loop, 3-range parallel).

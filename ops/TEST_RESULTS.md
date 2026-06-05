# run_fleet.py — Test-Plan Results

Operator infra; tested live on Windows 11 / conda `base` / `claude` v2.1.163 on 2026-06-05.
All six dispatch-specified tests were run. Live-spawn tests were bounded (low `--max-turns`,
`--max-sessions-per-range`, low `--min-healthy-secs`) to validate harness behaviour at low
cost — the harness's job is spawn/log/respawn/sync, not running full discovery arcs.

## Environment verified (don't-assume items)

| Item | Finding |
|------|---------|
| `claude` CLI | `C:\Users\panap\.local\bin\claude.exe`, v2.1.163, resolvable + runs from a Python subprocess. |
| `--model` value | **`claude-opus-4-8`** accepted by the parser AND backend (a live session ran 110s of real work under it, stopped only by the turn cap). Fallback alias `opus` documented. |
| `--max-turns` | Real flag (NOT shown in `--help`, but the parser consumes it — verified by a forced parse error). Hitting it returns **exit code 1** — same as a failed start — which is why crash detection keys on **duration**, not exit code. |
| `--permission-mode acceptEdits`, `--allowedTools` | Confirmed in `--help`; accepted. |
| Python / engine env | Harness launched from conda `base` (`C:\Users\panap\miniconda3\python.exe`, pandas/numpy/sklearn/lightgbm present); children inherit it. `py` (WindowsApps 3.14) is NOT the engine env — launch from activated `base`. |
| stdin | Children get `stdin=/dev/null` in argv-prompt mode, so claude does not stall on the "no stdin in 3s" wait. |

## Test 1 — dry-run (`--dry-run --ranges 1000,2000,3000`)  ✅ PASS

Printed, per range, the correct: sibling worktree path (`C:\Users\panap\Documents\fx-worktrees\<range>`,
outside the repo), branch (`disco/<range>`), the conditional `git worktree add ... -b disco/<range>
origin/main`, the pre-spawn sync (`fetch` + `reset --hard origin/main` + `clean -fd`), the STOP check
(local + remote), the exact spawn cmd (`-p <prompt> --model claude-opus-4-8 --permission-mode
acceptEdits --allowedTools Bash,Read,Edit,Write --max-turns 250`), cwd = worktree, and the per-range
`...Range: <n>. Resume.` prompt tail. Nothing spawned or modified.

## Test 2 — worktree creation + hard-sync (`--setup-only --ranges 1000`)  ✅ PASS

Created `C:\Users\panap\Documents\fx-worktrees\1000` on `disco/1000`, tracking `origin/main`
(`## disco/1000...origin/main`), hard-synced to `origin/main` (HEAD `5bae5c5`). Re-running detected the
existing worktree and **reused** it (idempotent), no recreate.

## Test 3 — single-range live, respawn (`--ranges 1000 --max-turns 10 --max-sessions-per-range 2 --min-healthy-secs 20 --hours 0.5`)  ✅ PASS

```
[1000] SPAWN pid=4828  ... 
[1000] EXIT  pid=4828 code=1 dur=110s          # session 1 ran real discovery work, hit max-turns
[1000] SPAWN pid=36816 ...                      # <-- RESPAWN: fresh session, never --resume
[1000] EXIT  pid=36816 code=1 dur=148s
[1000] max-sessions-per-range (2) reached; stopping.
range 1000: max-sessions (sessions=2);  exit=0
```

The harness spawned a real session, it ran (read protocol/log/registry, began observation), exited at
the turn cap, and the harness **respawned a fresh session**, then stopped cleanly at the session cap.
Both 110s/148s exits are > `--min-healthy-secs 20`, so correctly classified healthy (not crash-loop).

## Test 4 — STOP sentinel  ✅ PASS (both paths)

* **4a, local file:** created `<repo>\discovery\STOP`, ran the harness → it reused the worktree,
  detected `local file discovery/STOP`, and **halted before any spawn** (`sessions=0`, exit 0). STOP removed.
* **4b, pushed to origin/main:** to avoid halting any *live* discovery chats (they poll the real
  `origin/main` STOP), this was exercised in an **isolated git fixture** — a local bare origin whose
  `origin/main` had `discovery/STOP` while the working tree did not (the faithful "operator pushed STOP,
  my repo hasn't pulled" case). The harness `fetch`ed, detected `origin/main:discovery/STOP`, and halted
  before spawn (`sessions=0`, exit 0). This is the exact remote-detection code path the production
  harness uses against the real `origin/main`.

## Test 5 — crash-loop (`--ranges 1000 --model bogus-model-xyz-does-not-exist`)  ✅ PASS

```
[1000] SPAWN ... EXIT code=1 dur=3s ; fast exit #1 ; crash-loop backoff 10s
[1000] SPAWN ... EXIT code=1 dur=3s ; fast exit #2 ; crash-loop backoff 30s
[1000] SPAWN ... EXIT code=1 dur=3s ; fast exit #3
[1000] HALT: 3 consecutive fast exits -> ... Likely cause: auth / PATH-to-claude / bad --model
       ('bogus-model-xyz-does-not-exist'). Last exit code=1, inspect log: ...\20260605_100452.log
range 1000: crash-halt (sessions=3);  exit=1
```

Three fast exits, backoff grew 10s → 30s, **HALT after 3** with a clear diagnosis pointing at the log,
non-zero harness exit. The per-range log captured the child's `There's an issue with the selected
model (...)` error.

## Test 6 — 3-range parallel (`--ranges 1000,2000,3000 --max-turns 10 --max-sessions-per-range 1 --min-healthy-secs 20 --hours 0.2`)  ✅ PASS

```
[1000] worktree exists -> reuse: ...\fx-worktrees\1000
[2000] worktree created: ...\fx-worktrees\2000 on disco/2000 (tracking origin/main)
[3000] worktree created: ...\fx-worktrees\3000 on disco/3000 (tracking origin/main)
[1000] SPAWN pid=24512 log=20260605_101616.log
[2000] SPAWN pid=32872 log=20260605_101619.log     # three concurrent sessions,
[3000] SPAWN pid=36780 log=20260605_101622.log     # three distinct worktrees, three distinct logs
[3000] EXIT code=1 dur=147s ; max-sessions-per-range (1) reached; stopping.
[2000] EXIT code=1 dur=214s ; max-sessions-per-range (1) reached; stopping.
[1000] EXIT code=1 dur=244s ; max-sessions-per-range (1) reached; stopping.
==== FLEET SUMMARY ====
   range 1000: max-sessions (sessions=1)
   range 2000: max-sessions (sessions=1)
   range 3000: max-sessions (sessions=1)
FLEET END  exit=0
```

All three sessions overlapped (10:16:16–10:20:20), confirming genuine parallel execution under the
3-range CPU-bound load profile, then each stopped cleanly at its session cap.

Three sessions spawned concurrently into three distinct worktrees on three distinct branches, each
streaming to its own `ops/logs/<range>/<ts>.log`. The serialized git plumbing (worktree-add / fetch /
reset under a lock) produced **no git collision**. `origin/main` and all `disco/*` tips stayed at
`5bae5c5` throughout — the turn-capped sessions never reached a commit, so **zero main pollution**.

## Cleanup / residue

The test runs left `fx-worktrees\{1000,2000,3000}` on `disco/{1000,2000,3000}` — these are exactly the
worktrees the production fleet creates and reuses (each is hard-synced + `clean -fd` before every spawn),
so they are safe to leave for the real run, or removable with `git worktree remove`.

## Follow-up — council sub-agent tool (post-6-test verification)

The 6 tests above ran with the dispatch's literal `--allowedTools "Bash,Read,Edit,Write"`. A subsequent
check found that set **insufficient** for the mandatory `/llm-council-discovery` survivor stress-test,
which spawns sub-agents (5 lenses → 5 reviewers → 1 chairman). Verified live, headless, `claude-opus-4-8`:

| Probe | `--allowedTools` | Result |
|-------|------------------|--------|
| A — skill invokes headless? | `Read,Glob,Agent,Skill` | ✅ skill ran, returned its step-0 input-validation refusal (no sub-agents — step 0 refuses thin input first) |
| B — does `Agent` spawn? | `Agent` | ✅ `SPAWNED=yes, RETURNED=PING` — `Agent` is the correct token, sub-agent spawning works headless |
| C — minimal fix | `Bash,Read,Edit,Write,Agent` (no `Skill`) | ✅ skill still invoked — the `Skill` tool is NOT gated by `--allowedTools`; only `Agent` was missing |

Conclusion: the **only** addition needed is **`Agent`** (sub-agent spawning; legacy alias `Task`).
`DEFAULT_ALLOWED_TOOLS` was updated to `Bash,Read,Edit,Write,Agent`. Without `Agent`, the council reaches
step 0/1 then is hard-denied at the sub-agent spawn (headless `-p` cannot prompt; `acceptEdits` does not
auto-approve non-edit tools) — silently breaking the mandatory survivor council. This corrected the one
real defect in the dispatch's spawn spec.

# Wave 1 Branch Audit — 2026-05

> Read-only inventory. No state changes made. Audit branch: `audit/wave_1_branch_inventory` (off `main` @ `219fbf6`).
> Scope: Arcs 5, 8, 10, 11. Arc 7 explicitly excluded per dispatch (`arc/l_arc_7` in-flight elsewhere — not touched).

## Executive summary

| Arc | Closure on main | Branch on remote | Open PR | Verdict |
|---|---|---|---|---|
| 5  | **N** (no folder at all) | Y (`arc/l_arc_5`) | **#172** | **OPEN PR — DECISION NEEDED** (deletion loses entire arc record) |
| 8  | Y | N | none | BRANCH ALREADY GONE |
| 10 | Y | Y (`arc/l_arc_10_v3.0.1` — note suffix) | none | **CONTENT-ON-BRANCH-ONLY** (2 dispatch docs; no `results/` delta) |
| 11 | Y | N | none | BRANCH ALREADY GONE |

**Headline issues for chat:**
1. **Arc 5 has no presence on main at all** — no `results/l_arc_5/` folder, no `ARC_TRACKER.md` row. The entire arc record (closure + steps 1–5) lives only on `arc/l_arc_5` and inside open PR [#172](https://github.com/kpanapabusiness-droid/Forex-Backtester/pull/172). Deleting the branch (or closing the PR without merging) loses every artefact of Arc 5.
2. **Arc 10 branch is `arc/l_arc_10_v3.0.1`, not `arc/l_arc_10`.** Plain `arc/l_arc_10` is not on remote — already gone. The v3.0.1 branch is the retry attempt that HALTed; results identical to main but two dispatch docs (intent + HALT diagnostic) exist only on the branch.

---

## Arc 5

### What's on main

```
$ ls -la results/l_arc_5/
ls: cannot access 'results/l_arc_5/': No such file or directory
```

- `results/l_arc_5/ARC_CLOSURE.md` on main: **N**
- `results/l_arc_5/ARC_OPEN.md` on main: **N**
- Step folders on main: **none** (no folder exists)
- Total file count on main: **0**
- ARC_TRACKER.md mentions of `l_arc_5` on main: **none** (no row in any of the tracker sections)

### What's on `arc/l_arc_5`

- `results/l_arc_5/ARC_CLOSURE.md` on branch: **Y** (27 KB)
- Step folders on branch: **step_1, step_2, step_3, step_4, step_5** (no step_6 — Arc 5 closed at Step 5 FAIL per PR body)
- Total file count under `results/l_arc_5/` on branch: **17**
- Commits ahead of main: `git rev-list --count main..origin/arc/l_arc_5` → **1**
- Commits behind main: `git rev-list --count origin/arc/l_arc_5..main` → **21**
- Last commit: `592a3f7 2026-05-23 07:42:15 +1000 [ARC 5 v3.0] mtf_alignment.2_down_mixed.kijun.h_120 — FAIL`

Files on branch (17):
```
results/l_arc_5/ARC_CLOSURE.md
results/l_arc_5/step_1/integrity_report.md
results/l_arc_5/step_1/manifest.json
results/l_arc_5/step_1/pool.parquet
results/l_arc_5/step_2/cluster_assignments.parquet
results/l_arc_5/step_2/cluster_summary.csv
results/l_arc_5/step_2/cluster_summary.md
results/l_arc_5/step_3/capturability.csv
results/l_arc_5/step_3/capturability_summary.md
results/l_arc_5/step_4/extraction_metrics.csv
results/l_arc_5/step_4/extraction_summary.md
results/l_arc_5/step_4/feature_importance.csv
results/l_arc_5/step_5/architectures_ranked.md
results/l_arc_5/step_5/best_candidate.md
results/l_arc_5/step_5/holdout.csv
results/l_arc_5/step_5/manifest.json
results/l_arc_5/step_5/wfo_results.csv
```

### Delta — branch vs main, scoped to `results/l_arc_5/`

All 17 files on branch are absent from main. Deletion of `arc/l_arc_5` without merging PR #172 (or another preservation path) **loses the complete closure and every step artefact** for Arc 5.

### Open PRs

- **[#172 — [ARC 5 v3.0] mtf_alignment.2_down_mixed.kijun.h_120 — FAIL](https://github.com/kpanapabusiness-droid/Forex-Backtester/pull/172)**
  - State: OPEN | Base: `main` | Head: `arc/l_arc_5` | Created: 2026-05-22
  - +2445 / −10 across 23 changed files
  - Body verdict: FAIL Step 5 (worst-fold ratio −0.90, 8/10 negative folds, DD 51.7%)
  - First L_PROTOCOL v3.0 vanilla arc to run end-to-end; replaces prior HALT closure
  - Mergeable status: UNKNOWN (branch is 21 commits behind main — likely needs rebase)

### Verdict

**OPEN PR — DECISION NEEDED.** This is not a stale leftover — it is the live, unmerged closure record for Arc 5. Three plausible paths for chat:
- (a) Rebase + merge PR #172 → arc record lands on main, branch can then be deleted.
- (b) Close PR without merge + delete branch → loses entire arc record (no tracker row either; effectively erases Arc 5 from project history).
- (c) Preserve branch as archival reference without merging → unusual; the project pattern is closure-on-main per L_ARC_PROTOCOL §13.

---

## Arc 8

### What's on main

```
$ ls -la results/l_arc_8/
-rw-r--r-- 27901 May 25 12:30 ARC_CLOSURE.md
drwxr-xr-x       May 25 12:30 step_1
drwxr-xr-x       May 25 12:30 step_2
drwxr-xr-x       May 25 12:30 step_3
drwxr-xr-x       May 25 12:30 step_4
drwxr-xr-x       May 25 12:30 step_5
```

- `results/l_arc_8/ARC_CLOSURE.md` on main: **Y** (27 KB)
- `results/l_arc_8/ARC_OPEN.md` on main: **N**
- Step folders on main: **step_1–step_5** (no step_6 — FAIL Step 5 per tracker)
- Total file count on main: **29**
- ARC_TRACKER.md row on main (line 26): `| l_arc_8 | pullback_resume_hhhl_long_v0.1 ... | 4H | vanilla | A6 meta_labeling | 1.749 | FAIL | FAIL | 5 | results/l_arc_8/ARC_CLOSURE.md |` — path matches reality on main.

### What's on `arc/l_arc_8`

`origin/arc/l_arc_8` is **not present on remote** (verified via `git branch -r | grep arc_8` returns nothing).

### Delta

N/A — no branch to compare.

### Open PRs

```
$ gh pr list --state open --head "arc/l_arc_8"
[]
```

No open PRs.

### Verdict

**BRANCH ALREADY GONE.** Closure and all step artefacts present on main; tracker row points to correct file; nothing to clean up. Likely already deleted in a prior cleanup pass.

---

## Arc 10

### What's on main

```
$ ls -la results/l_arc_10/
-rw-r--r-- 22914 May 25 12:30 ARC_CLOSURE.md
-rw-r--r--  1124 May 25 12:30 ARC_OPEN.md
-rw-r--r--  4793 May 25 12:30 run_summary.json
drwxr-xr-x       May 25 12:30 step_1
drwxr-xr-x       May 25 12:30 step_2
drwxr-xr-x       May 25 12:30 step_3
drwxr-xr-x       May 25 12:30 step_4
drwxr-xr-x       May 25 12:30 step_5
drwxr-xr-x       May 25 12:30 step_6
```

- `results/l_arc_10/ARC_CLOSURE.md` on main: **Y** (23 KB)
- `results/l_arc_10/ARC_OPEN.md` on main: **Y**
- Step folders on main: **step_1–step_6** (only audited arc with step_6)
- Total file count on main: **26**
- ARC_TRACKER.md row on main (line 25): `| l_arc_10 | D1 swing-low rejection long (DLR, v0.1) ... | H4 | vanilla | A1 system_level_filter | 5.4185 | PASS-VIABLE | PASS-DEPLOYABLE | N/A | results/l_arc_10/ARC_CLOSURE.md |` — path matches reality on main.

### What's on `arc/l_arc_10` / `arc/l_arc_10_v3.0.1`

**Plain `arc/l_arc_10` is NOT on remote.** The remote-present branch is `arc/l_arc_10_v3.0.1` — the retry attempt against canonical post-PR-189 engine, not the original closure branch. Treating it as the in-scope arc 10 branch:

- `results/l_arc_10/ARC_CLOSURE.md` on branch: **Y**
- `results/l_arc_10/ARC_OPEN.md` on branch: **Y**
- Step folders on branch: **step_1–step_6**
- Total file count under `results/l_arc_10/` on branch: **26**
- Commits ahead of main: `git rev-list --count main..origin/arc/l_arc_10_v3.0.1` → **2**
- Commits behind main: `git rev-list --count origin/arc/l_arc_10_v3.0.1..main` → **3**
- Last commit: `cdb8de2 2026-05-25 01:45:21 +1000 [ARC 10 v3.0.1] HALT — canonical engine missing sl_partial_close_1r_runner_trail`

The 2 branch-only commits:
- `01489b9` — Intent doc, adds `docs/dispatches/arc_10_v3_0_1_intent.md` only
- `cdb8de2` — HALT diagnostic, adds `docs/dispatches/arc_10_v3_0_1_diagnostic.md` only

### Delta — branch vs main

```
$ git diff --name-only main origin/arc/l_arc_10_v3.0.1 -- results/l_arc_10/
(no output — identical)
```

`results/l_arc_10/` is **byte-identical** between main and branch. The branch-only content is two dispatch docs in `docs/dispatches/`:
- `docs/dispatches/arc_10_v3_0_1_intent.md`
- `docs/dispatches/arc_10_v3_0_1_diagnostic.md`

What would be lost on branch deletion: those two dispatch markdown files (the v3.0.1 retry intent and the HALT explanation). Both are narrative records of a HALT decision — not engine code, not arc results, not tracker mutations. Per the HALT commit body: *"No results/l_arc_10_v3.0.1/ artefacts. No tracker mutation. No PR."*

### Open PRs

```
$ gh pr list --state open --head "arc/l_arc_10"        → []
$ gh pr list --state open --head "arc/l_arc_10_v3.0.1" → []
```

No open PRs from either name.

### Verdict

**CONTENT-ON-BRANCH-ONLY** (mild). Original Arc 10 closure is safely on main and tracker-referenced. The `arc/l_arc_10_v3.0.1` retry branch contains zero unique arc artefacts — just two dispatch docs (intent + HALT diagnostic). Chat decides whether those dispatch docs should be preserved (cherry-pick to main) or are disposable retry narrative. Note for chat: plain `arc/l_arc_10` is already gone — only the v3.0.1 suffix branch remains.

---

## Arc 11

### What's on main

```
$ ls -la results/l_arc_11/
-rw-r--r-- 30170 May 25 12:30 ARC_CLOSURE.md
-rw-r--r--  1949 May 25 12:30 ARC_OPEN.md
drwxr-xr-x       May 25 12:30 step_1
drwxr-xr-x       May 25 12:30 step_2
drwxr-xr-x       May 25 12:30 step_3
drwxr-xr-x       May 25 12:30 step_4
drwxr-xr-x       May 25 12:30 step_5
```

- `results/l_arc_11/ARC_CLOSURE.md` on main: **Y** (30 KB)
- `results/l_arc_11/ARC_OPEN.md` on main: **Y**
- Step folders on main: **step_1–step_5** (no step_6 — FAIL Step 5)
- Total file count on main: **17**
- ARC_TRACKER.md row on main (line 27): `| l_arc_11 | swing-high breakout in trend (SHB) long ... | H4 | vanilla | A2 classifier_filter | -0.7687 | FAIL | FAIL | 5 | results/l_arc_11/ARC_CLOSURE.md |` — path matches reality on main.

### What's on `arc/l_arc_11`

`origin/arc/l_arc_11` is **not present on remote**.

### Delta

N/A — no branch to compare.

### Open PRs

```
$ gh pr list --state open --head "arc/l_arc_11"
[]
```

No open PRs.

### Verdict

**BRANCH ALREADY GONE.** Closure + tracker row + all step artefacts on main. Likely already cleaned up.

---

## Tracker cross-check

Grepped `ARC_TRACKER.md` on main for `l_arc_<N>` entries:

| Arc | Tracker row present? | Tracker closure path | Path exists on main? | Inconsistency? |
|---|---|---|---|---|
| 5  | **N** (no row in any tracker section) | n/a | n/a | **YES — Arc 5 entirely absent from tracker, but full closure exists on `arc/l_arc_5` + PR #172** |
| 8  | Y (line 26) | `results/l_arc_8/ARC_CLOSURE.md` | Y | none |
| 10 | Y (line 25) | `results/l_arc_10/ARC_CLOSURE.md` | Y | none |
| 11 | Y (line 27) | `results/l_arc_11/ARC_CLOSURE.md` | Y | none |

Additional tracker rows: Arc 8 and Arc 11 also appear in cluster registry, failure-mode counts, and cross-arc observation tables (lines 97, 110, 123–129, 147, 157–171). Arc 10 also appears in the cluster registry (lines 130–132) and cross-arc observations (lines 166–171). Arc 5 has **zero** mentions across the entire tracker.

---

## Recommended actions

Stating these as one-liners; chat decides:

- `arc/l_arc_5`: **DO NOT DELETE without merging PR #172 first** — it carries the only copy of Arc 5's closure + 17 step artefacts AND there is no tracker row pointing anywhere else. Chat should decide between (a) rebase + merge #172, (b) close #172 without merge + accept loss of arc record, or (c) preserve branch as archival-only.
- `arc/l_arc_8`: **No action needed** — already absent from remote, closure on main.
- `arc/l_arc_10_v3.0.1` (note: plain `arc/l_arc_10` already gone): **safe to delete** if chat is comfortable losing the two dispatch docs `docs/dispatches/arc_10_v3_0_1_{intent,diagnostic}.md`; **cherry-pick those two files to main first** if they should be preserved. Original Arc 10 closure on main is unaffected either way.
- `arc/l_arc_11`: **No action needed** — already absent from remote, closure on main.

---

## Notes

- **Arc 5 tracker absence is the most unexpected finding.** Every other audited arc has a tracker row that points to its on-main closure file. Arc 5's closure and tracker entries (`closed_arcs`, failure-mode, cluster registry, etc.) would land in a single merge of PR #172 — both the artefacts and the tracker mutation are inside that PR. Closing the PR without merging effectively erases Arc 5 from project state.
- **`arc/l_arc_10_v3.0.1` is a retry of a closed arc, not the original closure branch.** The dispatch named `arc/l_arc_10`; that branch is already gone. The v3.0.1 branch was cut from `4fccff0` (post-Amendment 4) to retry the diagnostic on canonical engine, then HALTed because the canonical sim layer lacks `sl_partial_close_1r_runner_trail`. The HALT commit message (`cdb8de2`) outlines three forward paths (engine PR / re-dispatch with narrower policies / hand-roll workaround) — that decision is open and orthogonal to branch deletion.
- **No legacy arc branch is up-to-date with main.** `arc/l_arc_5` is 21 commits behind; `arc/l_arc_10_v3.0.1` is 3 commits behind. Any merge attempt would need rebase / conflict resolution.
- **Arc 7 was not touched** per dispatch §6 (`arc/l_arc_7` is in-flight in a parallel session).
- **No state changes made.** Audit branch `audit/wave_1_branch_inventory` was created off `main` and used only for read-only inspection. No commits, no pushes, no deletions, no PR mutations.

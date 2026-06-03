# heavy_ml_probe Merge Orchestration — HALT Diagnostic

> **Status:** HALT per merge-dispatch §8.
> **Trigger:** CI failure on PR-A / PR-B / PR-C (pre-existing baseline issue, not heavy_ml_probe-introduced).
> **No merges performed.** Stack remains intact and untouched.
> **Branch:** `infra/heavy_ml_probe_pr_e` (worktree `elegant-ride-ad9a57`).

---

## §1 Topology — clean (passes pre-merge checks)

The §1 topology check passed. Linear stack matches dispatch §2 exactly:

```
origin/main (88d44de — 8 commits ahead of stack baseline)
  └─ infra/heavy_ml_probe_build (7c238e8 — stack baseline)
       └─ pr_a (1a4ca83, +1)
            └─ pr_b (a6e43d4, +2)
                 └─ pr_c (7918bb4, +3)
                      └─ pr_d (6f03ffc, +4)
                           └─ pr_e (c895cfc, +5)
```

Every parent-is-ancestor-of-child relationship holds. Origin SHAs == local SHAs. Merge-base of every PR vs build == `7c238e8`. **Topology PASS.**

---

## §2 The HALT trigger — CI red on PR-A / PR-B / PR-C

```
PR #187 (PR-A): tests fail (4m12s, 4m27s)
PR #191 (PR-B): tests fail (4m31s, 4m25s)
PR #192 (PR-C): tests fail (4m50s, 4m16s)
PR #196 (PR-D): tests pass (3m50s, 4m38s)  ← unexpected
PR #198 (PR-E): tests pass (5m27s, 5m23s)  ← unexpected
```

### §2.1 Failing tests are NOT heavy_ml_probe

The 10 failing tests are in pre-existing analytics code:

```
tests/test_phaseD6A_precursor.py — 5 failures
tests/test_phaseD6B_ignition_recall.py — 2 failures
tests/test_phaseD6C_value_of_capture.py — 3 failures
```

Failure mode: `ValueError: cannot insert direction, already exists` — pandas's `DataFrame.insert()` rejecting a duplicate column. The bug lives in `analytics/phaseD6A_ignition_precursor_analysis.py` or its callees. **heavy_ml_probe never touches `analytics/`, `tests/test_phaseD6*`, or any related code.**

### §2.2 Root cause: stale stack baseline

- Stack baseline: `7c238e8` (main's tip at PR-A open time, 2026-05-22).
- `origin/main` is now `88d44de` — 8 commits ahead. Among them: `d97430e` ("EET session semantics: distance.py + reset_floor.py + compute_per_day_max_dd") + `88d44de` ("rename core/utils → core/time_utils"). These engine changes touch the files that the phaseD6 tests indirectly depend on.
- The failing test files have **identical content SHAs** on both `7c238e8` and `origin/main` — the test files were never the problem; the engine code they invoke was.
- Main's most recent CI: `tests: success`. Main is green; the stack inherits a pre-fix engine state.

### §2.3 Why PR-D / PR-E mysteriously pass

PR-D's `requirements-dev.txt` removes `lifelines` + `scikit-survival` and adds `statsmodels`. This is a material change to pip's resolution graph — likely picks up different versions of pandas / numpy / a transitive that no longer triggers the buggy code path in the stale engine.

PR-A/B/C still have the original (lifelines + sksurv) deps which produce the buggy resolution. PR-E inherits PR-D's deps and also passes.

This is empirical, not load-bearing — the root issue is the stale baseline, and the dep-resolver shift is a red herring that happens to mask it on the later PRs.

---

## §3 Why a merge can't proceed cleanly

Per merge-dispatch §8 HALT triggers:

- ✗ "CI fails on `infra/heavy_ml_probe_build` after any individual PR merge" — would trigger immediately on PR-A merge (PR-A's CI is currently red; merging propagates red to the build branch).
- ✓ Topology clean
- ✓ No test pass-count divergence on `tests/heavy_ml_probe` locally
- ✓ No conflicts in `core/heavy_ml_probe/**`
- ✓ requirements-dev.txt resolves cleanly (locally; CI resolves differently and triggers the latent bug)

The merge-dispatch's pytest pass-count expectations (31/31, 50/50, ..., 149/149) refer to `tests/heavy_ml_probe/` specifically — those DO pass locally and presumably in CI. But CI runs the **whole** test suite, including the broken `tests/test_phaseD6*` modules.

So technically:
- `pytest tests/heavy_ml_probe -q` passes on every PR's branch (verified locally + CI passes on PR-D / PR-E)
- `pytest` (whole suite) fails on PR-A / PR-B / PR-C

Whether to consider that a merge-blocker is a chat-level decision.

---

## §4 Options for chat

### Option A — Rebase the stack onto current `origin/main` (cleanest)

```
git fetch origin
for pr in a b c d e; do
  git checkout infra/heavy_ml_probe_pr_$pr
  git rebase origin/main
  # resolve any conflicts (most likely in pipeline.py from core/utils → core/time_utils rename)
  git push --force-with-lease
done
```

- **Pros:** picks up all 8 main commits; clean baseline; CI should be green across the stack
- **Cons:** introduces 8 commits' worth of potential conflicts. Likely flashpoints:
  - `core/utils/` → `core/time_utils/` rename — if any heavy_ml_probe code imports `core.utils.*`, those imports break. (Spot check: heavy_ml_probe imports `core.features.*` and `core.steps.*`, not `core.utils.*` directly. Should be clean.)
  - `core/sim/risk/reset_floor.py` / `core/features/distance.py` changes — heavy_ml_probe doesn't touch these directly either.
  - `requirements-dev.txt` — additive merge; resolution should be clean.
- **Cost:** 5 sequential rebases + force-pushes + CI re-runs. Maybe 30-60 min.

### Option B — Cherry-pick `88d44de` into the build branch first (surgical)

```
git checkout infra/heavy_ml_probe_build
git cherry-pick 88d44de  # the rename fix
git push
# then proceed with stack merge as originally planned
```

- **Pros:** minimal change; brings the relevant fix without 7 other commits' worth of churn
- **Cons:** assumes `88d44de` alone fixes the phaseD6 tests. Likely true but unverified; might need other commits (d97430e the EET session semantics is the engine fix; 88d44de is the follow-up rename). Most conservative: cherry-pick `d97430e` AND `88d44de`.
- **Cost:** 2 cherry-picks; risk of partial-fix-not-actually-green.

### Option C — Accept CI red on PR-A/B/C; merge anyway (pragmatic)

Per the merge-dispatch's spirit ("don't pollute main until PR-F"), the build branch CI being red is acceptable IF chat plans to rebase or cherry-pick the fix before PR-F merges to main anyway.

- **Pros:** zero rework on the stack; PR-D / PR-E already pass; PR-F can land the rebase on a single later PR
- **Cons:** explicit acceptance that CC isn't blocked by red CI on auxiliary tests; sets a precedent. Auditor-unfriendly.
- **Cost:** zero now; deferred to PR-F.

### Option D — Update CI workflow to skip phaseD6 tests (hack)

- **Pros:** unblocks merges immediately
- **Cons:** sets the wrong precedent; ignores the underlying bug; auditor-unfriendly
- **Cost:** small workflow change; CI fix needs to be reverted later when phaseD6 fix lands

### Option E — Wait for chat

Default of HALT discipline. Surface, gather chat input, proceed with the chosen option.

---

## §5 CC recommendation

**Option B with safety:** cherry-pick BOTH `d97430e` (the EET session-semantics engine fix) and `88d44de` (the follow-up rename) into `infra/heavy_ml_probe_build`, push, verify build branch CI passes. THEN proceed with the stack-merge order PR-A → ... → PR-E. PR-F can later cherry-pick the remaining 6 commits or rebase the build branch onto main if chat wants the full bring-forward.

Rationale: minimum touches; surgical; preserves the stack as authored. PR-A's CI will then re-run against an advanced build branch (its base) and should turn green if Option B's cherry-picks fixed the issue. If PR-A's CI is still red after Option B, fall back to Option A.

If chat prefers Option A (full rebase), that's also defensible — cleaner long-term but more rework.

If chat prefers Option C (defer to PR-F), I'll merge the stack now with CI red on PR-A/B/C and surface the cleanup task at PR-F time.

---

## §6 What I'm waiting on

Chat decision on Option A / B / C / D / E. Once decided, I can resume merge orchestration in a single follow-up run.

No PR-F work in this run.

---

## §7 State snapshot at HALT time

```
Branch                                     Origin SHA       Local SHA       Match
origin/main                                88d44de          88d44de         ✓
infra/heavy_ml_probe_build                 7c238e8          7c238e8         ✓
infra/heavy_ml_probe_pr_a                  1a4ca83          1a4ca83         ✓
infra/heavy_ml_probe_pr_b                  a6e43d4          a6e43d4         ✓
infra/heavy_ml_probe_pr_c                  7918bb4          7918bb4         ✓
infra/heavy_ml_probe_pr_d                  6f03ffc          6f03ffc         ✓
infra/heavy_ml_probe_pr_e                  c895cfc          c895cfc         ✓
```

```
PR        Number    Mergeable    CI Status
PR-A      #187      MERGEABLE    UNSTABLE (tests fail — phaseD6)
PR-B      #191      MERGEABLE    UNSTABLE (tests fail — phaseD6)
PR-C      #192      MERGEABLE    UNSTABLE (tests fail — phaseD6)
PR-D      #196      MERGEABLE    CLEAN (tests pass)
PR-E      #198      MERGEABLE    CLEAN (tests pass)
```

All PRs are mergeable (no conflicts). HALT is purely on CI verification, not on stack integrity.

---

End of diagnostic.

# heavy_ml_probe Rebase Run — HALT Diagnostic

> **Status:** HALT after rebase. Rebase itself completed cleanly (no conflicts). CI on PR-A/B/C still red.
> **Trigger:** §9 — the rebase did not fix CI red on PR-A/B/C. The original Option A premise (stale baseline) was wrong; the actual cause is a stack-internal dep-resolver bug.
> **No merges performed.** All five PR branches force-pushed to rebased state; build branch fast-forwarded to `198d78f`.
> **Branch:** `infra/heavy_ml_probe_pr_e` (worktree `elegant-ride-ad9a57`).

---

## §1 What completed cleanly

Rebase phase succeeded end-to-end:

| Step | Old SHA | New SHA | Notes |
|---|---|---|---|
| FF `infra/heavy_ml_probe_build` → `origin/main` | `7c238e8` | `198d78f` | clean FF, 9 commits pulled |
| Rebase PR-A onto build | `1a4ca83` | `ee63cd8` | no conflicts, force-pushed |
| Rebase PR-B onto PR-A | `a6e43d4` | `4cbc30a` | no conflicts, force-pushed |
| Rebase PR-C onto PR-B | `7918bb4` | `80545a5` | no conflicts, force-pushed |
| Rebase PR-D onto PR-C | `6f03ffc` | `ba6b774` | no conflicts, force-pushed |
| Rebase PR-E onto PR-D | `c895cfc` | `f6cfd65` | no conflicts, force-pushed |

Local test pass-counts after each rebase (per dispatch §4 verification):

- PR-A: **31/31** ✓
- PR-B: **50/50** ✓
- PR-C: **80/80** ✓
- PR-D: **111/111** ✓
- PR-E: **149/149** + sibling `tests/discovery` **42/42** ✓
- ruff: clean

Zero conflicts in any file. Stack topology preserved — PR-A is ancestor of PR-B is ancestor of PR-C etc., still a perfect linear stack but now on top of `198d78f` instead of `7c238e8`.

---

## §2 What's still broken

CI on PR-A / PR-B / PR-C is **still red post-rebase** on the same `tests/test_phaseD6A_*.py` failures:

```
ValueError: cannot insert direction, already exists
```

in `pandas/core/frame.py:5180`, triggered by `analytics/phaseD6A_ignition_precursor_analysis.py`.

PR-D / PR-E CI is **green** (same as pre-rebase).

This is **NOT** a stale-baseline issue. The Option A rebase premise was wrong. Root cause found below.

---

## §3 Root cause — pip-resolver downgrade chain

### §3.1 Smoking gun

Main's CI install log (commit `198d78f`):

```
Successfully installed ... pandas-3.0.3 ...
1572 passed, 307 skipped, ... in 219.04s
```

PR-A's CI install log (commit `ee63cd8`, post-rebase):

```
Downloading pandas-3.0.3-cp312-cp312-manylinux_2_24_x86_64.whl
... (later in same install) ...
Downloading pandas-2.3.3-cp312-cp312-manylinux_2_24_x86_64.whl
10 failed, 1593 passed, ... in 197.57s
```

Two pandas downloads on PR-A. First is the initial install of `numpy pandas pyarrow pytest ruff` (pulls 3.0.3). Then `pip install -r requirements-dev.txt` runs and the heavy-ML deps **downgrade pandas to 2.3.3**. The downgraded pandas is the one running when tests execute.

### §3.2 What forces the downgrade

PR-A's `requirements-dev.txt` includes:

```
flaml
lifelines           ← deferred per PR-D log §1, but still listed
scikit-survival     ← deferred per PR-D log §1, but still listed
xgboost
catboost
lightgbm
```

`lifelines` or `scikit-survival` (or both — both depend on `ecos`) carry a pandas pin like `pandas<3.0` that conflicts with main's pandas-3.0.3. Pip resolves by downgrading. PR-A through PR-C all carry these two deps.

PR-D's `requirements-dev.txt` REMOVES `lifelines` + `scikit-survival` (replaced by `statsmodels` per chat-side RSF-deferral disposition). PR-D's CI install pulls only `pandas-3.0.3` → pandas stays modern → no phaseD6 failures.

### §3.3 Why pandas 2.3.3 triggers the bug

`analytics/phaseD6A_ignition_precursor_analysis.py` calls `pd.DataFrame.insert(column='direction', ...)`. pandas 2.3.3 raises `ValueError: cannot insert direction, already exists` when the column is already present; pandas 3.0.3 has different semantics (likely silently overwrites or guards). The pre-existing analytics code passes pandas-3.0.3 contract and fails the pandas-2.3.3 contract.

This is a real `analytics/phaseD6A_*` bug — but it's a bug that only manifests under pandas-2.3.3, which is not what main uses. heavy_ml_probe's deferred deps (`lifelines` + `scikit-survival`) expose it by forcing the downgrade.

---

## §4 Why Option A rebase didn't help

The Option A premise was: "stack baseline `7c238e8` is pre-bugfix on `analytics/phaseD6A_*`; rebase onto `198d78f` picks up the fix." That premise was **WRONG** in two ways:

1. The failing test files have **identical content SHAs** on `7c238e8` and `198d78f` (verified in `heavy_ml_probe_merge_diagnostic.md` §2.2). The fix never existed; the tests have always failed on this code under pandas-2.3.3.
2. The bug isn't in `analytics/phaseD6A_*` getting fixed on main; it's in main's CI never installing pandas-2.3.3, so the bug never gets exposed on main's runs.

The rebase moved the stack to a current base — but the dep-resolver behavior is unchanged because PR-A/B/C's deps still list `lifelines` + `scikit-survival`.

---

## §5 Fix paths for chat

### Option F1 (recommended) — Amend PR-A to use PR-D's clean dep set

Pull PR-D's `requirements-dev.txt` change forward to PR-A. Remove `lifelines` + `scikit-survival`; add `statsmodels`. Propagate via rebase through PR-B, PR-C. PR-D's commit would still appear in PR-D's diff but be a no-op for `requirements-dev.txt` (or PR-D's commit can be amended to skip the dep change since PR-A already did it).

**Pros:**
- Principled: PR-A through PR-E all install the same dep set → pandas stays at 3.0.3 throughout → CI green
- Mirrors what would have happened if PR-D's lib choices had been known at PR-A authoring time
- Single-line change per PR

**Cons:**
- Requires amending PR-A's commit (force-push to rewrite the merge-tree); needs to propagate via re-rebase through B/C/D/E
- Sequential operation, ~15-30 min
- Touches code, not just branch ops (dispatch's "no code changes" spirit may want chat sign-off)

### Option F2 — Pin pandas in requirements-dev.txt

Add `pandas>=3.0` to `requirements-dev.txt` so pip refuses to downgrade. May fail at install time if heavy deps genuinely cannot resolve with pandas 3.0 (would surface as install error rather than test failure — at least loud).

**Pros:**
- Minimal change; one line per PR
- Forces resolver to find a heavy-dep version that supports pandas 3.0 or fail loudly

**Cons:**
- May not be possible (lifelines / sksurv may simply not have a pandas-3.0-compatible release yet)
- Pinning pandas across heavy_ml_probe deps puts policy on a future operator's plate

### Option F3 — Fix `analytics/phaseD6A_*` to handle the duplicate column

Out-of-scope for heavy_ml_probe (the code lives in `analytics/`, owned by whoever maintains the phaseD6 work). Would be the most principled global fix.

**Pros:**
- Real fix; benefits everyone
- Removes the time-bomb regardless of pandas version

**Cons:**
- Out of scope for this work
- Requires identifying who owns phaseD6 + their bandwidth

### Option F4 — Accept CI red on PR-A/B/C; merge anyway

Merge PR-A → PR-B → PR-C even though their CI is red (PR-D's merge fixes the dep set, build branch CI then goes green). Final state of build branch is green; transient red state on PR-A/B/C accepted.

**Pros:**
- Zero rework; ships fastest
- Build branch ends up correct

**Cons:**
- Sets precedent of merging red CI
- The interim PRs as merged on origin are "stained" with red CI

### Option F5 — Wait for chat

Default per HALT discipline.

---

## §6 CC recommendation

**Option F1.** PR-D's dep change is what should have been in PR-A from the start. The reason it wasn't is that chat's dispatch for PR-A authored before chat made the "drop RSF + lifelines, use statsmodels" decision at PR-D time. Pulling that decision forward to PR-A:

- Eliminates the pandas downgrade everywhere in the stack
- Makes every PR's CI green
- Matches the actual library choices the build ended up with
- Surface-area: edit one file (`requirements-dev.txt`) per PR via interactive rebase OR amend PR-A only and re-rebase B/C/D/E

I can perform Option F1 in a follow-up run if chat confirms. Approximately 30 min of git operations + waiting for 5 CI runs.

If chat prefers F4 (accept red CI), I can resume the original merge orchestration immediately — 0 rework, but the merged PR-A/B/C remain in red-CI state on origin.

---

## §7 Current branch state at HALT time

```
Branch                                  Origin           Local            Match
origin/main                             198d78f          198d78f          ✓
infra/heavy_ml_probe_build              198d78f          198d78f          ✓ (FF'd from 7c238e8)
infra/heavy_ml_probe_pr_a               ee63cd8          ee63cd8          ✓ (rebased from 1a4ca83)
infra/heavy_ml_probe_pr_b               4cbc30a          4cbc30a          ✓ (rebased from a6e43d4)
infra/heavy_ml_probe_pr_c               80545a5          80545a5          ✓ (rebased from 7918bb4)
infra/heavy_ml_probe_pr_d               ba6b774          ba6b774          ✓ (rebased from 6f03ffc)
infra/heavy_ml_probe_pr_e               f6cfd65          f6cfd65          ✓ (rebased from c895cfc)
```

| PR | # | State | Mergeable | CI |
|---|---|---|---|---|
| PR-A | 187 | OPEN | ✓ | FAIL (phaseD6 — pandas 2.3.3 downgrade) |
| PR-B | 191 | OPEN | ✓ | FAIL (same) |
| PR-C | 192 | OPEN | ✓ | FAIL (same) |
| PR-D | 196 | OPEN | ✓ | PENDING (will pass — proven previously) |
| PR-E | 198 | OPEN | ✓ | PENDING (will pass) |

---

## §8 What's deferred to chat decision

- Option F1 vs F2 vs F3 vs F4 vs F5
- If F1: I can do the surgery; chat just signals go-ahead
- If F4: I can resume the original merge orchestration immediately

No PR-F work in either case.

---

End of diagnostic.

# heavy_ml_probe Rebase — State Report (§1)

> **Status:** §1 state report — produced before any git mutations.
> **Goal:** rebase the entire heavy_ml_probe stack onto current `origin/main`
> (`198d78f`), then proceed with the original merge plan.
> **HALT preconditions:** all checked, none triggered. Proceeding.

---

## §1.1 Branch SHAs (origin = local, all match)

| Branch | Origin SHA | Local SHA | Match |
|---|---|---|---|
| `main` | `198d78f259abc61a9594d4cf742b2e3858cbddfb` | `198d78f` | ✓ |
| `infra/heavy_ml_probe_build` | `7c238e822f7db1ddaa033bbcdc576e01fe0c97be` | `7c238e8` | ✓ |
| `infra/heavy_ml_probe_pr_a` | `1a4ca83a15c70f96e6d806303a9c6188a9406342` | `1a4ca83` | ✓ |
| `infra/heavy_ml_probe_pr_b` | `a6e43d4ac9751d124d6a10bfd7c0afe0883a3669` | `a6e43d4` | ✓ |
| `infra/heavy_ml_probe_pr_c` | `7918bb40e6d9038b3296a3b56788de88fef346d5` | `7918bb4` | ✓ |
| `infra/heavy_ml_probe_pr_d` | `6f03ffc1e559869f85c5b631d4b407e6e10b263d` | `6f03ffc` | ✓ |
| `infra/heavy_ml_probe_pr_e` | `c895cfc86773627889269c78fefd1d955f09c34f` | `c895cfc` | ✓ |

No drift. `git fetch --all --prune` completed cleanly (deleted two stale remote refs: `claude/heuristic-zhukovsky-247bd2`, `engine/sl-partial-close-runner-trail-primitive` — neither stack-related).

## §1.2 Stack topology (still clean linear from prior diagnostic)

```
origin/main (198d78f)
  └─ infra/heavy_ml_probe_build (7c238e8 — still at stack baseline; never advanced)
       └─ pr_a (1a4ca83, +1)
            └─ pr_b (a6e43d4, +2)
                 └─ pr_c (7918bb4, +3)
                      └─ pr_d (6f03ffc, +4)
                           └─ pr_e (c895cfc, +5)
```

Every parent-is-ancestor-of-child holds (verified previously).

## §1.3 Commits on `origin/main` since stack baseline (9 total)

```
198d78f [ENGINE] Canonical exit policy registry + sl_partial_close_1r_runner_trail primitive + per-arc migration (#195)
88d44de [FIX] CC_20 follow-up: rename core/utils/ -> core/time_utils/ (deshadow legacy core/utils.py)
ff8e0b9 [CHORE] CC_20 ruff fix — remove extra blank line after import block
d97430e [ENGINE] EET session semantics: distance.py + reset_floor.py + compute_per_day_max_dd (#197)
219fbf6 [PROTOCOL] Amendment 5 — AUC-gated A2/A6 architecture selection + parser v1.3 field (#194)
e824f5f [ENGINE] Signal-level EET timezone alignment audit + fix + canonical utility (#193)
4fccff0 [INFRA] TODO.md refresh — post-PR-#189 state alignment (#190)
1eedccd [ENGINE] Signal parity: mid-feature leaks + trail mid + 5ers EET bar boundaries (#189)
e47ab04 [ENGINE] Step 6 causal audit framework + Amendment 4 + parser v1.3 (#188)
```

**1 new commit since the prior diagnostic** (`198d78f` — exit policy registry). Otherwise the bring-forward set is identical to what was surfaced in `heavy_ml_probe_merge_diagnostic.md` §2.2.

## §1.4 `core/utils` → `core/time_utils` import scan (HALT-risk check)

`git grep "from core\.utils\|import core\.utils"` against `core/heavy_ml_probe/**`, `scripts/heavy_ml_probe/**`, `tests/heavy_ml_probe/**`:

```
No matches found.
```

Zero `core.utils.*` imports anywhere in the heavy_ml_probe surface. The rename (`88d44de`) is structurally invisible to the stack — rebase will not require any code fixups for this rename.

## §1.5 Per-PR status

| PR | # | State | Mergeable | mergeStateStatus | CI |
|---|---|---|---|---|---|
| PR-A | 187 | OPEN | ✓ | UNSTABLE | **FAIL** (phaseD6 pre-existing — see diagnostic) |
| PR-B | 191 | OPEN | ✓ | UNSTABLE | **FAIL** (same) |
| PR-C | 192 | OPEN | ✓ | UNSTABLE | **FAIL** (same) |
| PR-D | 196 | OPEN | ✓ | CLEAN | PASS |
| PR-E | 198 | OPEN | ✓ | CLEAN | PASS |

All 5 PRs are open and mergeable (no conflicts). CI failure on PR-A/B/C is the pre-existing baseline issue that the rebase will fix.

## §1.6 `origin/main` CI status (HALT precondition §9.1)

```
tests: success (HEAD = 198d78f)
```

Most recent runs on `main`:

```
26381541253  success  [ENGINE] Canonical exit policy registry (#195)               main  4m29s
26380930998  success  [FIX] CC_20 follow-up: rename core/utils/ -> core/time_utils/  main  3m46s
```

Two intermediate commits (`ff8e0b9` ruff fix + `d97430e` EET session semantics) showed CI failures in their direct push runs (43s + 40s — short, likely setup/install failures rather than test failures), but the immediately-following commits succeeded. **Current `main` HEAD CI is green.** HALT precondition cleared.

## §1.7 HALT-trigger check (per §9)

| Trigger | Status |
|---|---|
| `main` CI red at start of run | ✗ (green) |
| Any PR branch auto-closed / deleted | ✗ (all 5 open) |
| Local SHAs diverge from origin | ✗ (all match) |
| Conflicts in any non-`heavy_ml_probe` file | not yet evaluable; pre-rebase |
| Rebased PR CI fails for heavy_ml_probe-specific reasons | not yet evaluable; pre-rebase |
| Test pass counts diverge | not yet evaluable; pre-rebase |
| `requirements-dev.txt` doesn't resolve cleanly | not yet evaluable; pre-rebase |

No triggers fire at §1. Proceeding to §2 rebase.

---

## §1.8 Outcome

PASS. Proceeding to rebase sequence per §2 in the order:

1. Fast-forward `infra/heavy_ml_probe_build` to `origin/main` (`198d78f`)
2. Rebase `pr_a` onto updated build
3. Rebase `pr_b` onto rebased `pr_a`
4. Rebase `pr_c` onto rebased `pr_b`
5. Rebase `pr_d` onto rebased `pr_c`
6. Rebase `pr_e` onto rebased `pr_d`

Each rebase: `git rebase --onto <new-base> <old-base> <branch>` then `git push --force-with-lease`. Per-PR local test verification (31 → 50 → 80 → 111 → 149) before next rebase. CI re-runs in the background.

---

End of §1 state report.

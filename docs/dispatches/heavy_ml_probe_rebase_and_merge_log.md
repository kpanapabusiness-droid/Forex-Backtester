# heavy_ml_probe Rebase + Merge Log

> **Phase:** F1 fix complete; stack ready for the original merge plan.
> **Status:** ALL FIVE PRs CI GREEN, MERGEABLE, CLEAN. Holding per dispatch §8 — chat must signal to start merges.
> **Dispatches in this work-stream (in order):**
>  1. `heavy_ml_probe_merge_orchestration.md` — original stack-merge plan (HALTed after §1 topology pass; pre-existing CI red on PR-A/B/C)
>  2. `heavy_ml_probe_merge_diagnostic.md` — root-cause writeup of first HALT
>  3. `heavy_ml_probe_rebase_state.md` — §1 pre-rebase state report for Option A
>  4. `heavy_ml_probe_rebase_diagnostic.md` — second HALT; rebase succeeded but PR-A/B/C CI still red (real cause: dep-resolver downgrade, not stale baseline)
>  5. F1 dispatch (this run) — amend PR-A to carry the dep cleanup
> **Branch:** `infra/heavy_ml_probe_pr_e` (worktree `elegant-ride-ad9a57`).

---

## §1 Rebase phase (Option A) — completed

| Step | Old SHA | New SHA | Conflicts | Notes |
|---|---|---|---|---|
| FF `infra/heavy_ml_probe_build` → `origin/main` | `7c238e8` | `198d78f` | none | 9 commits pulled forward (Step 6 framework, Amendment 4/5, EET semantics, exit policy registry, core/utils → core/time_utils rename) |
| Rebase PR-A onto build | `1a4ca83` | `ee63cd8` | none | 31/31 ✓ |
| Rebase PR-B onto PR-A | `a6e43d4` | `4cbc30a` | none | 50/50 ✓ |
| Rebase PR-C onto PR-B | `7918bb4` | `80545a5` | none | 80/80 ✓ |
| Rebase PR-D onto PR-C | `6f03ffc` | `ba6b774` | none | 111/111 ✓ |
| Rebase PR-E onto PR-D | `c895cfc` | `f6cfd65` | none | 149/149 + 42/42 ✓ |

Force-pushes all with `--force-with-lease`. No conflicts in any file. The Option A premise turned out to be wrong (CI red was NOT a stale-baseline issue) → HALT diagnostic written → F1 fix path approved by chat.

## §2 F1 fix phase — completed

PR-D's `requirements-dev.txt` change (remove `lifelines` + `scikit-survival`, add `statsmodels`) pulled forward to PR-A. Reason: those two deps pin `pandas<3.0`, forcing pip to downgrade pandas to 2.3.3 in PR-A/B/C's CI install, which exposes a pre-existing `analytics/phaseD6A_*` bug that doesn't manifest on pandas 3.0.3.

### §2.1 Per-step result

| Step | Old SHA | New SHA | requirements-dev.txt conflict | Local pytest |
|---|---|---|---|---|
| Amend PR-A (swap lifelines/sksurv → statsmodels) | `ee63cd8` | `cb735ed` | n/a (direct edit) | 31/31 ✓, lint clean |
| Re-rebase PR-B onto amended PR-A | `4cbc30a` | `c0ba093` | **none** — PR-B didn't touch the file | 50/50 ✓ |
| Re-rebase PR-C onto re-rebased PR-B | `80545a5` | `b4e7085` | **none** — joblib pin on line 10, swap on lines 17-22; non-overlapping | 80/80 ✓ |
| Re-rebase PR-D onto re-rebased PR-C | `ba6b774` | `b8b5ec7` | **none** — git auto-detected the swap as redundant; PR-D's diff for this file collapsed to zero lines | 111/111 ✓ |
| Re-rebase PR-E onto re-rebased PR-D | `f6cfd65` | `1ed32fb` | **none** — PR-E didn't touch the file | 149/149 + 42/42 ✓, lint clean |

All five re-rebases force-pushed with `--force-with-lease`.

### §2.2 Conflict expectations vs reality

Dispatch §4 anticipated multiple `requirements-dev.txt` conflicts during re-rebase. **Actual result: zero conflicts.** Reasons:

- PR-B / PR-E never touched `requirements-dev.txt` — automatic clean.
- PR-C's joblib pin (lines 10) and PR-A's amended swap (lines 17-22) occupy non-overlapping line ranges → git's auto-merger handled cleanly.
- PR-D's swap commit was textually identical to what PR-A now carries → git's auto-merger recognised it as already-applied → PR-D's `requirements-dev.txt` diff collapsed to zero lines. PR-D's commit still exists with its other code/test/doc changes, just no longer touches that one file.

The cleanest possible F1 outcome.

### §2.3 PR-A's final `requirements-dev.txt`

```
numpy
pandas
pyarrow
pytest
ruff
pyyaml
pydantic
scipy
scikit-learn
joblib

# heavy_ml_probe sub-protocol (docs/sub_protocols/heavy_ml_probe.md v1.0)
# ...
flaml
# Cox PH via statsmodels.duration.hazard_regression.PHReg (PR-D). The
# original dispatch spec listed `lifelines` for Cox PH and
# `scikit-survival` for RSF, but both are blocked on Python 3.14 because
# their transitive dep `ecos` has no cp314 wheel. statsmodels has a
# clean cp314 wheel and PHReg covers the Cox PH path; RSF is deferred
# per PR-B flag-1 disposition.
statsmodels
xgboost
catboost
lightgbm
```

`joblib` still un-pinned (PR-C's commit upgrades it to `joblib>=1.4,<1.6`).

---

## §3 CI status per PR (post-F1, all green)

```
PR-A #187: tests pass (4m28s)
PR-B #191: tests pass (4m22s)
PR-C #192: tests pass (5m16s)
PR-D #196: tests pass (5m43s)
PR-E #198: tests pass (5m30s)
```

All five mergeStateStatus: **CLEAN**, mergeable: **MERGEABLE**. Pandas-version verification (spot-checked PR-A's install log):

```
Successfully installed ... pandas-3.0.3 ...
```

No downgrade. The F1 hypothesis (deferred `lifelines` + `scikit-survival` were forcing the downgrade) is confirmed empirically.

---

## §4 Final SHAs (origin = local, all match)

| Branch | SHA |
|---|---|
| `infra/heavy_ml_probe_build` | `198d78f` (= `origin/main`) |
| `infra/heavy_ml_probe_pr_a` | `cb735ed` (1 commit ahead of build) |
| `infra/heavy_ml_probe_pr_b` | `c0ba093` (2 commits) |
| `infra/heavy_ml_probe_pr_c` | `b4e7085` (3 commits) |
| `infra/heavy_ml_probe_pr_d` | `b8b5ec7` (4 commits) |
| `infra/heavy_ml_probe_pr_e` | `1ed32fb` (5 commits) |

Stack topology preserved — clean linear stack on top of `198d78f`.

---

## §5 Per-PR `requirements-dev.txt` diff size vs build

| PR | Diff lines vs build (`198d78f`) | Content |
|---|---|---|
| PR-A | 25 | F1-amended dep swap (flaml + statsmodels + xgboost/catboost/lightgbm replacing the original joblib-only baseline) |
| PR-B | 25 | inherits PR-A's diff; no PR-B-specific changes |
| PR-C | 33 | adds joblib pin on top of PR-A's diff |
| PR-D | 33 | inherits PR-C's diff; no PR-D-specific changes (PR-D's original dep swap collapsed to zero per §2.2) |
| PR-E | 33 | inherits PR-D's diff; no PR-E-specific changes |

Pattern confirms: every PR's `requirements-dev.txt` strictly extends its predecessor's. PR-D no longer competes with PR-A on the same lines — clean stack-internal ownership.

---

## §6 Holding for chat — do NOT proceed to merges

Per dispatch §8 deliverable 4: "After all five PRs are green: report ready-state to chat. Do NOT auto-proceed to merges."

Ready-state is the table above. All gates per dispatch §3 / §4 cleared:

- ✓ Per-PR CI passes
- ✓ Per-PR local pytest pass-count matches expected (31/50/80/111/149)
- ✓ Sibling regression `tests/discovery` 42/42
- ✓ ruff clean
- ✓ Pandas version verified at 3.0.3 in CI install
- ✓ All 5 PRs mergeStateStatus: CLEAN, mergeable: MERGEABLE
- ✓ No HALT triggers fired during F1

Waiting on chat to dispatch the merge orchestration prompt (original `heavy_ml_probe_merge_orchestration.md`) for the PR-A → PR-B → PR-C → PR-D → PR-E sequential merges into `infra/heavy_ml_probe_build`.

---

## §7 What stays open after merges (forward reference, not this run)

After all five PRs merge:

- `infra/heavy_ml_probe_build` advances by 5 commits past `origin/main` (`198d78f`)
- PR-A through PR-E: closed (merged)
- PR-F: not yet opened (final docs-polish PR per original build plan)
- `main` untouched until eventual `build → main` PR after PR-F

---

## §8 Phase 1 final-merge run — completed

Per the "Final Merge" dispatch, executed sequentially. Each merge used GitHub's "Create a merge commit" method (not squash) to preserve per-PR commit detail as audit trail. Build branch CI verified green after each merge (~5-6 min runs).

| PR | # | Merge commit | New build HEAD | Build CI |
|---|---|---|---|---|
| PR-A | 187 | `2d52f1e` | `2d52f1e` | green |
| PR-B | 191 | `b76dcd5` | `b76dcd5` | green |
| PR-C | 192 | `864b001` | `864b001` | green |
| PR-D | 196 | `67c8b68` | `67c8b68` | green |
| PR-E | 198 | `8be174c` | `8be174c` | green |

Phase 1 outcome: `infra/heavy_ml_probe_build` advanced from `198d78f` to `8be174c` — 10 commits ahead of `origin/main` (5 PR commits + 5 merge commits). All 5 PRs in MERGED state.

Local verification on freshly-pulled `infra/heavy_ml_probe_build`: `pytest tests/heavy_ml_probe` → 149/149 ✓; sibling `tests/discovery` → 42/42 ✓ (verified earlier in the F1 phase, unchanged by merge commits). No conflicts encountered during any merge.

---

## §9 Phase 2 — PR-F (docs + polish) opened

Branch cut: `infra/heavy_ml_probe_pr_f` off `infra/heavy_ml_probe_build` (`8be174c`).

Changes (2 files, +109 / -8 lines):

- `docs/sub_protocols/heavy_ml_probe.md`: expanded "Output artefacts" with per-file descriptions; added Step 5 adapter manifest schema reference block; added "A4 hazard math (the CoxPHAdapter contract)" section; added "Cox PH minimum-N discipline" section; added "RSF deferral status" section.
- `core/heavy_ml_probe/survival.py`: 4-line docstring expansion on `_predicted_risk` cross-referencing PR-D flag-1 disposition (the rationale was already there; PR-F makes the audit-trail explicit).

Verification: `pytest tests/heavy_ml_probe` → 149/149 ✓ (no semantics changed); `ruff check` clean.

Log doc: `docs/dispatches/heavy_ml_probe_pr_f_log.md`.

PR-F is now open + awaiting chat review per the final-merge dispatch §2.5 (hold-for-chat gate before Phase 3).

---

## §10 PR-F amendment (Phase 3 dispatch §1) — landed

Chat review of PR-F caught one piece of stale text: spec doc's "Expected compute cost" section still cited the pre-build "2-6 hours per cluster" ceiling. PR-B's empirical wall-clock + PR-D's production-scale probe established the as-built figure at ~10-30 min per cluster.

Fix landed as a follow-up commit on PR-F's branch (NOT amending the original PR-F commit per dispatch §1.1):

- `abf4163` — `docs(heavy_ml_probe): sync compute-cost estimate to empirical measurement`
- Updated: `docs/sub_protocols/heavy_ml_probe.md` "Expected compute cost" section rewrite; `docs/dispatches/heavy_ml_probe_pr_f_log.md` §4.2 + §5.1 marked-as-fixed; new §6 "Future tech-debt" section recording the two non-urgent items chat flagged (adapter manifest schema duplication; possible operator-guide split).
- No grep hits for stale "2-6 hours" / "6 hour" in `core/heavy_ml_probe/**` docstrings.
- Verification: 149/149 + lint clean.
- PR-F CI re-ran on push and passed.

---

## §11 Phase 3 — Build → Main landed

| Step | Action | Result |
|---|---|---|
| §3.1 | Merge PR-F #203 → build branch | Merge commit `41928e9`; build CI green |
| §3.2 | Verify main hasn't advanced into heavy_ml_probe surface | Main advanced 6 commits since stack baseline check, but **zero file overlap** with build branch; no rebase needed. Cross-branch file-overlap check: empty. |
| §3.3 | Open `infra/heavy_ml_probe_build → main` PR | PR #206 opened |
| §3.4 | Wait for CI | Both runs `pass`; mergeStateStatus CLEAN, mergeable MERGEABLE |
| §3.5 | Merge to main with merge-commit | Merge commit **`0db7851`** |
| §3.6 | Post-merge verification | Main CI: `success` ✓; local pytest tests/heavy_ml_probe: 149/149 ✓; sibling tests/discovery: 42/42 ✓ |
| §3.7 | Branch cleanup | 6 PR branches deleted on origin; `infra/heavy_ml_probe_build` retained for 24-48h rollback safety net |

### §11.1 Final SHA on main

```
0db7851 Merge pull request #206 from kpanapabusiness-droid/infra/heavy_ml_probe_build
```

### §11.2 PR-A → PR-F commit history visible on main

The merge-commit method preserved the full audit trail:

```
0db7851 Merge pull request #206 from .../infra/heavy_ml_probe_build       ← final build → main
41928e9 Merge pull request #203 from .../infra/heavy_ml_probe_pr_f        ← PR-F
abf4163 docs(heavy_ml_probe): sync compute-cost estimate to empirical measurement  ← PR-F amendment
d5ec992 [INFRA] heavy_ml_probe PR-F: docs polish + final-merge orchestration logs
8be174c Merge pull request #198 from .../infra/heavy_ml_probe_pr_e        ← PR-E
67c8b68 Merge pull request #196 from .../infra/heavy_ml_probe_pr_d        ← PR-D
864b001 Merge pull request #192 from .../infra/heavy_ml_probe_pr_c        ← PR-C
b76dcd5 Merge pull request #191 from .../infra/heavy_ml_probe_pr_b        ← PR-B
2d52f1e Merge pull request #187 from .../infra/heavy_ml_probe_pr_a        ← PR-A
1ed32fb [INFRA] heavy_ml_probe PR-E: integration + Step 5 adapters (gate PR)
b8b5ec7 [INFRA] heavy_ml_probe PR-D: Cox PH survival via statsmodels.PHReg
b4e7085 [INFRA] heavy_ml_probe PR-C: meta-labeling target + threshold sweep
c0ba093 [INFRA] heavy_ml_probe PR-B: FLAML AutoML + 11-fold TimeSeriesSplit
cb735ed [INFRA] heavy_ml_probe PR-A: scaffolding + lineage gate + IO
```

### §11.3 Branch state on origin after cleanup

| Branch | Status |
|---|---|
| `main` | at `0db7851`, CI green, heavy_ml_probe landed |
| `infra/heavy_ml_probe_build` | retained for 24-48h rollback safety net (at `41928e9`) |
| `infra/heavy_ml_probe_pr_a` through `pr_f` | DELETED on origin |

### §11.4 ARC_TRACKER question — surfaced for chat (not actioned)

Per Phase 3 dispatch §4.5: should `ARC_TRACKER.md` get a row for heavy_ml_probe as available infrastructure? It's not an arc, but it's discoverable infrastructure that future arcs can invoke. Chat decides — CC does not action this.

---

End of log.

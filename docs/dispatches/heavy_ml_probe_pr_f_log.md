# heavy_ml_probe PR-F — Log Doc (Docs + Polish, Final PR)

> **Branch:** `infra/heavy_ml_probe_pr_f` → `infra/heavy_ml_probe_build` (cut from build branch HEAD `8be174c` after PR-E merged).
> **Dispatch:** chat's "Final Merge" dispatch §2 (PR-F scope).
> **Prior:** PR-A through PR-E merged sequentially into `infra/heavy_ml_probe_build` in Phase 1; this is the final docs-cleanup PR before the build → main PR.
> **Authoring CC session:** worktree `elegant-ride-ad9a57`.

---

## §1 Scope landed (verbatim against dispatch §2.2)

**Documentation in `docs/sub_protocols/heavy_ml_probe.md`:**

| Item | Status | Location |
|---|---|---|
| Operator notes for invocation | ✓ extended | "Output artefacts" section (per-artefact descriptions for all step_4/heavy_ml/ + step_5/heavy_ml_augmented/ files) |
| Reference adapter manifest format | ✓ added | "Step 5 adapter manifest schema" subsection with full JSON shape + `STEP5_MANIFEST_SCHEMA_VERSION = "1.0"` pin |
| Minimum-N discipline for Cox PH | ✓ added | "Cox PH minimum-N discipline" section — warning-not-skip policy at n_train < 200, no-events fold handling, statsmodels convergence-failure trapping, closure-doc audit guidance |
| RSF deferral | ✓ promoted to dedicated section | "RSF deferral status" — three-level documentation (spec / `integrated_brier_score` error message / `requirements-dev.txt` comment block) |
| A4 hazard math (`P(reach +1R in next K bars)` formula) | ✓ added | "A4 hazard math (the `CoxPHAdapter` contract)" section — full formula + step-function semantics + numerical guards + NaN propagation |
| Exit code mapping (0/1/2/3) | already present | "Invocation" section + "When invoked" exit codes block |
| Fold selection strategies | already present | "Invocation" section last paragraph |

**Inline doc cleanup:**

| Item | Status | Location |
|---|---|---|
| `_predicted_risk` rationale comment | ✓ extended | `core/heavy_ml_probe/survival.py:289-302` — PR-D flag-1 cross-reference added to the existing docstring (rationale was already there; reference makes the chat-side audit-trail explicit) |

**Spec sync items already up-to-date from PR-B/C/D/E** (no additional PR-F work):

- Library swaps (statsmodels for Cox PH, FLAML for AutoML, RSF deferred) — documented under "Step 4 — replaced" item 3 since PR-D
- Per-stage independence — implicit from the skip-reason taxonomy in `pipeline.py` (referenced by the new operator notes)
- Narrow Step 5 scope (PR-E §6 decision) — documented in "Scope clarification (PR-E)" block since PR-E

---

## §2 What PR-F intentionally does NOT touch (per dispatch §2.2 "What NOT to touch")

- `core/heavy_ml_probe/**` — single change in `survival.py` is a docstring expansion only (no code semantics change)
- `tests/heavy_ml_probe/**` — zero changes
- `core/architectures/**` — dispatch §3 boundary preserved
- `L_PROTOCOL.md` — not a heavy_ml_probe-owned doc

---

## §3 Verification

### §3.1 Test suite

```
py -3 -m pytest tests/heavy_ml_probe -q -W ignore::UserWarning
........................................................................ [ 96%]
.....                                                                    [100%]
149 passed in 177.53s
```

149/149 pass — no code changes that affect runtime. PR-F is docs + one docstring-only edit.

### §3.2 Lint

```
py -3 -m ruff check core/heavy_ml_probe scripts/heavy_ml_probe tests/heavy_ml_probe
All checks passed!
```

### §3.3 Diff stats

```
core/heavy_ml_probe/survival.py      |   4 ++
docs/sub_protocols/heavy_ml_probe.md | 113 ++++++++++++++++++++++++++++++++---
2 files changed, 109 insertions(+), 8 deletions(-)
```

Modest surface area as expected for a docs-cleanup PR.

---

## §4 Deviations from dispatch

### §4.1 `_predicted_risk` already had the rationale comment

The dispatch §2.2 asked for "Inline comment in `survival.py` near `_predicted_risk` explaining why `exp(X @ params)` is computed directly vs `results.predict()`". That rationale was already in the docstring as of PR-D's landing — survival.py:290-297 explains the predict-bunch-shape stability argument. PR-F added a single trailing paragraph cross-referencing the PR-D flag-1 disposition so the audit trail is explicit (chat → CC → spec).

If chat would rather see this as a sidebar comment (vs an extended docstring), the 4-line addition is trivial to refactor — flag for re-review.

### §4.2 Compute-cost section preserved verbatim

The "Expected compute cost" section at the bottom of the spec doc still cites the original spec's "2-6 hours per cluster" estimate, but PR-B's empirical wall-clock probe showed ~6 minutes per cluster at production scale. The dispatch §2.2 didn't list compute-cost calibration as a PR-F item, so it stays unchanged — should chat want to update with PR-B's empirical numbers, a one-line tweak in a follow-up.

---

## §5 Flags for chat

### §5.1 Non-blocking

- **Compute cost spec text is stale.** Spec says "2-6 hours per cluster"; PR-B log §4.5 measured ~6 minutes per cluster at production scale (n=5000, n_folds=11). Not within PR-F dispatch scope; surfacing for future doc-touch.
- **`docs/sub_protocols/heavy_ml_probe.md` length** went from 140 lines to ~330 lines after PR-F additions. The doc remains organised (clear sections) but it's now denser. Chat may want to split the operator-notes block into a separate `docs/sub_protocols/heavy_ml_probe_operator_guide.md` in a future polish pass; out of scope here.
- **Adapter manifest schema is duplicated** between the spec doc (PR-F) and `core/heavy_ml_probe/adapters.py::STEP5_MANIFEST_SCHEMA_VERSION`'s docstring. Spec doc is the human-facing source of truth; the constant is the machine-facing version pin. They should stay in lockstep on `schema_version` bumps.

### §5.2 No HALT

No HALT conditions encountered. WORKFLOW §6 unused. The build is complete and ready for `build → main` once PR-F merges.

---

## §6 Holding for chat per dispatch §2.5

Per the final-merge dispatch §2.5: "End turn after PR-F opens. Do NOT proceed to Phase 3 until chat approves PR-F."

After PR-F opens:

1. End turn
2. Wait for chat approval
3. Then Phase 3: merge PR-F → build → CI verify → open `infra/heavy_ml_probe_build → main` PR → chat approves final merge to main → post-merge cleanup (delete 6 PR branches, surface ARC_TRACKER question per §6 of the final-merge dispatch)

---

End of PR-F log.

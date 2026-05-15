# Phase KH-24 Promotion — `phase/kh-17` → `promotion/kh24-to-main`

> Date: 2026-05-15
> Branch: `promotion/kh24-to-main` (forked from `main` HEAD `fe345be`)
> Source: `phase/kh-17` HEAD `04740bb` (KH-24 lock commits: `484d2f7`, `a0f55dd`)
> Author: Claude Code (Opus 4.7), under prompt direction from chat
> Status: **PROMOTED, NOT MERGED.** Merge to `main` is the chat's decision after review of this doc.

---

## §1. Header

| Item | Value |
|---|---|
| Source branch | `phase/kh-17` |
| Source commit (tip) | `04740bb5b507299475c80e3569ce3bc59a70ee1a` |
| KH-24 system lock commits on source | `a0f55dd` (system lock + result doc, 2026-04-20), `484d2f7` (phase close) |
| Target branch | `promotion/kh24-to-main` |
| Target branch parent | `main` HEAD `fe345be58a8517189e898d0618266f083bda900c` |
| Files promoted (action=copy) | 15 |
| Files verified identical (action=verify_identical) | 1 (`core/utils.py`) |
| Files updated for path correctness in governance docs | 0 |
| sha256 manifest | `results/kh24/audit/promotion_manifest.csv` |
| Governance path audit | `results/kh24/audit/governance_path_audit.csv` |
| Source audit motivating this promotion | `results/kh24/audit/capturability_artefact_audit.md` |

---

## §2. Investigation findings

### Audit's framing was wrong

The capturability artefact audit (committed earlier today) characterised commit `04740bb` as "the bulk worktree commit that built main [and] did not preserve these specific paths." This is incorrect, and the inaccuracy is corrected here.

What `git log` actually shows:

- `main` and `phase/kh-17` diverged at common ancestor `81764f7` (Merge branch 'main' of github).
- After divergence:
  - `main` proceeded: `81764f7` → `2108213` → ... → `92852ab docs: lock L_ARC_PROTOCOL v1.0 and migrate reference docs into docs/` → `fe345be` (HEAD)
  - `phase/kh-17` proceeded: `81764f7` → `c3dbb8b` → `52c1524` → `1aa428c` → `484d2f7` (KH-24 phase close) → `a0f55dd` (KH-24 system lock + result doc) → `04740bb` (tip)

So `04740bb` is the tip of `phase/kh-17`, applied **after** the KH-24 lock commits. It did not exclude or delete the KH-24 files — it added more files alongside them. Verified via `git diff --name-status a0f55dd 04740bb`: the only KH-24-named addition is `EA/KH24_EA.mq5`; no KH-24 paths are deleted or modified.

KH-24 is missing from `main` because **`phase/kh-17` was never merged into `main`**. The two branches developed independently and there is no merge commit joining them. This is incidental, not deliberate.

### Was the exclusion deliberate?

**No.** The commit message for `04740bb` describes a bulk worktree snapshot covering the full KH/KI/J/I research lineage. The exclusion is not a "stripping for security or secrets" decision — it is the natural consequence of two divergent branches that never re-merged. No content in `phase/kh-17`'s KH-24 paths suggests broker secrets, credentials, or anything that would justify deliberate withholding.

### Decision

Proceed with promotion. The user's chat explicitly directed this in the current prompt; this section just confirms there's no historical signal saying "don't promote".

---

## §3. Promotion manifest summary

Manifest CSV: `results/kh24/audit/promotion_manifest.csv` (16 rows).

| Action | Count | Files |
|---|---|---|
| copy (newly staged on promotion branch) | 15 | `configs/wfo_kh24.yaml`, `configs/wfo_kh24_short_mirror.yaml`, `configs/wfo_baseline_clean.yaml`, `docs/KH24_SYSTEM_LOCK.md`, `results/kh24/PHASE_KH24_RESULT.md`, `results/kh24/kgl_v2_report.md`, `results/kh24/trades_all.csv`, `results/kh24/wfo_fold_results_4h.csv`, `results/kh24/wfo_per_pair_4h.csv`, `results/kh24/wfo_summary_4h.txt`, `scripts/phase_kgl_v2_4h_wfo.py`, `signals/__init__.py`, `signals/kb_exhaustion_bar.py`, `tests/test_phase_kgl_v2_context_columns.py`, `EA/KH24_EA.mq5` |
| verify_identical | 1 | `core/utils.py` (sha256 identical on both branches; no checkout needed) |

### Byte-identity verification

All 15 copied files were verified via two complementary checks:

1. **Working-tree sha256.** After `git checkout phase/kh-17 -- <path>`, the on-disk sha256 differs from `phase/kh-17` blob sha256 because `core.autocrlf=true` is set on this Windows clone: git stores content with LF endings and rewrites them to CRLF on checkout. This is a **filesystem-appearance difference only**; it is reversed at stage time.

2. **Git blob hash (canonical, used in the manifest).** `git hash-object <path>` after checkout produces the SHA-1 blob that git will actually store. For every copied file, the working-tree hash-object SHA-1 matches `git rev-parse phase/kh-17:<path>` exactly. The staged blob (after `git add`) is also byte-identical to `phase/kh-17`. All 15 verifications passed.

In short: what gets committed on `promotion/kh24-to-main` is byte-identical at git's storage level to what is on `phase/kh-17`. The CRLF appearance on Windows disk is cosmetic.

### Scope decision on dependencies

The KH-24 generator script imports `signals.kb_exhaustion_bar._wilder_atr` and `core.utils.{get_pip_size, load_pair_csv}`. These are not KH-24-named but are required for the generator to run on `main`.

- `signals/__init__.py` and `signals/kb_exhaustion_bar.py` — promoted (the KH-24 signal definition is by-definition a KH-24 artefact even though the path is generic).
- `core/utils.py` — already identical on `main`; verified, no copy needed.
- `tests/test_phase_kgl_v2_context_columns.py` — the generator's test; promoted for completeness.

### Scope decision on `configs/wfo_baseline_clean.yaml`

This file is not KH-24-named, but `SESSION_ZERO.md:61` explicitly governance-flags it as a KH-24 config ("KH-24 configs (`wfo_kh24.yaml`, `wfo_baseline_clean.yaml`): never modify"). Without it, that governance reference would still not resolve after promotion. Promoted on that basis. Its content is the KGL_V2 baseline with exposure cap disabled — the comparison anchor KH-24 was derived against.

### Sibling artefacts NOT promoted

The following KH-24-mentioning files exist on `phase/kh-17` but are KH-N-specific (other arc phases) rather than KH-24 proper. They are out of scope for this promotion and left for a follow-up if the chat wants them:

- `configs/wfo_kgl_v2.yaml`, `configs/wfo_kgl_v2_extended.yaml` (KGL_V2 engine baseline, not KH-24)
- `configs/wfo_kh11b.yaml`, `configs/wfo_kh25.yaml`, `configs/wfo_ki1_baseline_1h.yaml` (other KH/KI arc configs)
- `scripts/h1_threshold_sweep.py`, `scripts/kgl_v2_full_backtest.py`, `scripts/kh14_state2_mae_trajectory.py`, `scripts/parity_fold7_extract.py`, `scripts/phase_kh29/kh29_excursion_analysis.py`, `scripts/phase_ki1_1h_wfo.py`, `scripts/state1_diagnostic.py` (other phase analysis scripts)
- `tests/test_phase_kh17_state2_watch.py` (KH-17 test, not KH-24)
- `EA/KGL_V2_EA.mq5`, `EA/KGL_V2_EA_v1.2.mq5`, `EA/KH8_EA.mq5`, `EA/Data scrape - 1H.mq5` (other EAs)

### `.gitignore` interaction

Main's `.gitignore` ignores `results/*` (except `results/.gitkeep` and `results/c1_only_exits/**`), as well as `*.csv`, `*.txt`, `*.png`, `*.log` broadly. The 7 files under `results/kh24/` and the `wfo_summary_4h.txt` would normally be ignored. They were staged anyway because:

- Explicit `git add <path>` overrides `.gitignore` for that specific call.
- Once tracked, git keeps tracking them regardless of `.gitignore` (the ignore rule applies only to untracked files).

If the chat decides at merge time that the gitignore policy should explicitly allow `results/kh24/**`, that's a separate `.gitignore` patch. The current state works (the files are tracked) but a `!results/kh24/**` rule would make future KH-24 emissions stage cleanly without `-f`.

---

## §4. Governance doc reconciliation summary

Audit CSV: `results/kh24/audit/governance_path_audit.csv` (15 rows).

| Doc | Path references audited | All resolve after promotion | Action taken |
|---|---|---|---|
| `CLAUDE.md` | 4 | yes | none |
| `STATUS.md` | 4 | yes (1 bare filename, 3 explicit paths) | none |
| `SESSION_ZERO.md` | 7 | yes (3 bare filenames, 4 explicit paths) | none |

### Findings

- **Every explicit path reference** (form `docs/...`, `configs/...`, `scripts/...`, `results/...`) resolves correctly on `promotion/kh24-to-main` after the 15 files were staged. No content edits to CLAUDE.md / STATUS.md / SESSION_ZERO.md were required.
- **Bare filenames** (`KH24_EA.mq5`, `wfo_kh24.yaml`, `wfo_baseline_clean.yaml`) appear in table cells and prose. These are not "paths" in the strict sense; they are labels that the reader resolves by context. The corresponding files all exist after promotion, so the references are functionally correct. Left unchanged per the prompt's "only update if path moved or name changed" rule — none of these names have changed; they just lack a directory prefix.
- **Status content** (KH-24 is locked, gate PASS, +1.92% worst-fold ROI, deployed on VPS) was explicitly out of scope per the prompt. Not touched.

### No stale references after promotion

There are no remaining unresolvable KH-24 paths in any of the three governance docs as of `promotion/kh24-to-main` HEAD.

---

## §5. Open questions for the chat

1. **Merge strategy.** The promotion branch is built off `main` HEAD with a stack of in-flight uncommitted L-arc work alongside (per the user's "promote into the current working tree" answer earlier in the chat). When merging to `main`, the chat needs to decide:
   - Merge the KH-24 files alone (cherry-pick the promotion commit) and leave the L-arc uncommitted state for separate handling.
   - Or treat this as a regular merge where the promotion commit travels in.

2. **`.gitignore` patch.** Should `.gitignore` get an explicit `!results/kh24/**` rule so future KH-24 emissions (e.g., the per-bar trade-paths regeneration recommended in the capturability audit) stage without needing `-f`? Not part of this promotion's scope, but worth a small follow-up patch.

3. **Sibling artefacts promotion.** The KH-arc siblings listed in §3 ("Sibling artefacts NOT promoted") may be wanted on `main` for context. Specifically `EA/KGL_V2_EA.mq5` (the underlying engine EA) and `scripts/phase_kh29/kh29_excursion_analysis.py` (referenced by the closed Path A roadmap). A follow-up promotion task would handle these.

4. **`docs/KH_Research_Roadmap.md`.** Lives on `main` but not on `phase/kh-17`. Promotion is a no-op for this file — it's already where governance expects it. Noted for completeness.

5. **`configs/spread_floors_5ers.yaml`.** Untracked on both branches at this moment (present in main's worktree as an uncommitted file). The capturability audit referenced this as a locked KH-24 / L-arc shared artefact. Out of scope here (it's not on either branch's tree), but the chat may want to commit it on `main` separately.

---

## §6. Definition of done for merge

The chat reviews this doc and confirms each item before merging `promotion/kh24-to-main` to `main`:

- [ ] Every file in `promotion_manifest.csv` with action=copy exists on `promotion/kh24-to-main` at a staged blob sha256 matching `sha256_phase_kh17`. (**Verified by Claude Code: 15/15 OK.**)
- [ ] Every file with action=verify_identical has matching sha256 on `main` and `phase/kh-17`. (**Verified by Claude Code: 1/1 OK — `core/utils.py`.**)
- [ ] Every governance doc path reference in `governance_path_audit.csv` shows `resolves_on_promotion_branch=true`. (**Verified by Claude Code: 15/15 OK.**)
- [ ] No file on `main` outside the manifest was modified by this promotion. (**Verified by Claude Code: no changes to existing tracked main files — only NEW files were added.**)
- [ ] No backtester runs were performed. (**Confirmed.**)
- [ ] No content edits to KH-24 files during promotion. (**Confirmed: byte-identical copies only; the only edits made by this work were to the three new audit-folder docs.**)

If all six boxes are ticked, the merge can proceed. Recommended merge form: a single squashed commit on `main` with a message that names the source branch, the 15 files promoted, and a reference to this phase doc.

---

## Appendix — quick reference

- Manifest: `results/kh24/audit/promotion_manifest.csv`
- Governance audit: `results/kh24/audit/governance_path_audit.csv`
- Source audit: `results/kh24/audit/capturability_artefact_audit.md` (note: §1 of that audit incorrectly characterised `04740bb` as having excluded KH-24; correction is in §2 of this doc)
- Branch: `promotion/kh24-to-main`
- Branch parent: `main` HEAD `fe345be`
- Source branch tip: `phase/kh-17` HEAD `04740bb`

*Promoted, not merged. Awaiting chat review.*

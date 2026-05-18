# Arc 8 — Closure Doc

**Signal:** PR-HHHL long (`signal_pullback_resume_hhhl_long_v0.1`)
**Protocol:** v2.3 stack (base v2.1.2 + v2.2 + v2.3 amendments)
**Branch:** `claude/magical-zhukovsky-bd69d9` → dispatcher to merge into `phase/l_arc_8`
**Closure date:** 2026-05-18
**Disposition:** **HALT_DEPLOYMENT** (no portfolio candidate from §10 ship gates; c1 logged as cross-arc structural finding for Open-05)
**Lifecycle:** Arc 8 → CLOSED. Queue state: removed from Active.

---

## TL;DR

Arc 8 PASSED Steps 1–4 (one archetype survived: c1 V-shape recovery FG-weak, admit-only economics genuinely strong). Step 5 WFO failed §10 ship gates: classifier admits 70–89% of full pool; non-c1 trades dilute c1's +2.59R/trade edge to negative aggregate ROI (-13% to -15% worst window). Two post-Step-5 diagnostics confirmed the failure is structural — c1 and c2 share entry-bar geometry, so no entry-time filter (mechanical or classifier-based) can preferentially admit c1. Mid-path classifiers (D1 at t≤12) improve separability monotonically but plateau in the marginal band. Arc 8 closes; cross-arc evidence base now sufficient for v2.4 protocol amendment.

---

## Step-by-step result table

| Step | Gate | Result | Headline |
|---|---|---|---|
| 1 | Plumbing | PASS | 1,327 trades, 28 pairs; all 5 gates clear; KH-24 co-fire 0.0% |
| 2 | Path-shape clustering | PASS | K=4, silhouette 0.4762, 0/4 degenerate; 2 V-shape clusters |
| 3 | Capturability | PASS | 3 units survive (c1 SL=4.0×ATR, c3 SL=2.0×ATR, agg SL=3.0×ATR); c2 dies §2 floors |
| 4 | Extractability | PASS | 1 archetype: c1 E+D1 (E AUC 0.697, D1 AUC 0.637 at t=1); c3 + agg die per v2.2 §3 |
| 5 | WFO ship gates | **FAIL** | Admit-only PASS (E Sharpe 1.44 / D1 Sharpe 1.14) but full-pool FAIL (worst DD 15.6% / 19.0%, ROI -13% / -15%) |

---

## Step 5 ship-gate detail

| Pipeline | View | Worst ROI | Worst DD | Sharpe | n | 4-gate pass |
|---|---|---:|---:|---:|---:|:---:|
| E | admit-only | +18.66% | 1.00% | 1.44 | 103 | ✓ |
| E | full-pool | **−13.21%** | **15.58%** | 0.012 | 751 | ✗ |
| D1 | admit-only | +25.73% | 0.54% | 1.14 | 133 | ✓ |
| D1 | full-pool | **−14.69%** | **19.02%** | 0.001 | 943 | ✗ |

**Failure mechanism:** Pipeline E (5 features) and Pipeline D1 (15 features at t=1) both admit 70–89% of full Step 1 OOS pool. Per-cluster mean_r at SL=4.0×ATR:

| Cluster | n (Step 1) | mean_r | Pool share |
|---|---:|---:|---:|
| c0 | 316 | −0.46R | 23.8% |
| **c1** (trained-on) | 177 | **+2.59R** | 13.3% |
| c2 | 429 | −0.47R | 32.3% |
| c3 | 405 | −0.10R | 30.5% |

c1 is genuinely +2.59R/trade. But only 13.3% of pool. Non-c1 admissions (~80% × ~1150 trades, mean ≈ −0.3R) drown the signal.

---

## Post-Step-5 diagnostics

### Diagnostic 1 — Entry-feature overlap (commit `7d9109e`)
**Result:** `c1_NOT_SEPARABLE_AT_ENTRY`. Multiclass RF on full pool: c1 one-vs-rest AUC 0.547, precision@recall=0.60 = 0.149 (vs base rate 0.133 — zero lift). All 18 entry features have mean overlap coefficient > 0.78; c1 vs c2 pairwise overlap 0.83 (highest, structural — both share entry geometry). Multiclass reframe path: DEAD.

### Diagnostic 2 — Path 1 + Path 2 combined (commit `4756c66`)

**Path 1 (post-entry confirmation, t ∈ {3, 5, 8, 12}):** `PATH_1_MARGINAL`. c1 AUC climbs 0.547 (entry) → 0.652 (t=3) → 0.755 (t=12); c1 precision@recall=0.60 climbs 0.196 → 0.274. Plateau below 0.40 VIABLE threshold. Slippage not the constraint (17% c1 MFE consumed by t=12).

**Path 2 (mechanical signal-tightening):** `PATH_2_DEAD`. Zero filter configurations satisfy c1_ret ≥ 0.80 ∧ c2_ret ≤ 0.30 ∧ pool ≥ 500. c1 and c2 retention drop in lockstep across all 6 single-rule sweeps (~0.03–0.09R gap). Two-rule combinations kill more c2 but collapse c1 retention to ~0.32. Best filter (F2 `pullback_depth_atr ≥ 1.0`) improves aggregate mean_r +68% but doesn't separate c1 from c2.

---

## Root cause

c1 (V-shape recovery, +2.59R) and c2 (Early-peak hold, −0.47R) are **mechanically indistinguishable at entry**. The difference between them is whether MFE develops mid-trade (c1) or stalls early (c2) — a forward-path property, not an observable at entry. Mid-path observation helps (Path 1 AUC trend) but the lift plateaus before deployable precision within the economic envelope.

This is the **third consecutive arc** with the same failure mode:

| Arc | Admit-only | Full-pool | Failure mode |
|---|:---:|:---:|---|
| 4 RERUN | PASS | FAIL | Reject pool 32% × −0.232R + early-exit 11% × −0.685R |
| 5 | PASS | FAIL | Rejected pool 78% × −0.46R |
| **8** | PASS | FAIL | Admit 70–89% × non-c1 mean ≈ −0.3R |

---

## Cross-arc structural findings (logged for v2.4 protocol design)

### Finding 1 — v2.4 §1.5 entry-separability gate (recommended)

Path-shape clustering identifies real structural archetypes, but on PR-HHHL (and likely other trend-continuation signals) those archetypes are not predictable from entry-time observables. Pipeline E classifiers trained on archetype-success labels admit 70–90% of full pool because winning and losing clusters share entry-time geometry.

**Proposed §1.5 pre-Step-1 gate:**
- Run a quick multiclass RF on a Step-1-spec smoke pool (n ≥ 200) with target = path-shape cluster ID
- Required: target cluster (winning archetype candidate) one-vs-rest precision@recall=0.60 ≥ 0.30 on entry features alone
- If below 0.30 → arc auto-halts before committing full Step 1 simulation compute

This is the structural amendment the framework needs. Saves arc-level compute on signals that can't deploy.

### Finding 2 — c1 V-shape recovery FG-weak as portfolio archetype (Open-05)

c1 admit-only economics are tradeable in isolation: Pipeline E Sharpe 1.44, worst DD 1.00%, worst ROI +18.66%; Pipeline D1 Sharpe 1.14, worst DD 0.54%, worst ROI +25.73%. The archetype itself is real edge; the problem is identifying its trades in advance from PR-HHHL signal alone. Log for portfolio-composition work — c1 may be routable from other signals or combinable with confirming archetypes.

Key features (Pipeline E top-5 importance): `ret_5bar_atr`, `pos_in_20bar_range`, `pullback_depth_atr`, `range_to_atr_14`, `hl_range_atr`. Could serve as cross-arc filter candidates for other long trend-continuation signals.

### Finding 3 — pullback_depth_atr observation (signal-design)

The single filter F2 `pullback_depth_atr ≥ 1.0` improves aggregate Step 1 mean_r from +0.054 → +0.091 (+68% relative). Doesn't pass §10 viability gates as a system, but useful as a signal-redesign input if PR-HHHL is revisited or as a cross-arc feature for trend-continuation work.

---

## Variance from dispatch (recorded)

- **Branch name:** worktree `claude/magical-zhukovsky-bd69d9` instead of `phase/l_arc_8` (stale local branch conflict — pre-Arc-3 commits from `phase/v2_2_housekeeping`)
- **Signal spec:** written from analyst-supplied content (was unmerged on `tmp/post-v2_3`); byte-equivalent to `tmp/post-v2_3` commit `9e9bf0a`
- **Data wiring:** `data/4hr` directory junction to parent repo (worktree had `data/` gitignored)
- **Engine PR:** `feat/open-24-pre-t-sl-per-archetype` merged locally to main mid-arc; NOT pushed to origin (analyst-side decision pending)
- **Step 1 wfo_l_arc_8.yaml sha:** `9785a5b...` (dispatch text said `accba985...` — was the smoke-test manifest sha; corrected in arc-open doc)
- **Pool size:** 1,327 trades (below spec prior of 2,500–4,000 but clears §5 floor with 2.65× margin); plausibly attributable to strict HH-AND-HL + 0.5×ATR pullback floor in signal spec

---

## Files

### Step outputs (locked, do not modify)
- `results/l_arc_8/step1_verbatim/` — Step 1 plumbing
- `results/l_arc_8/step2/` — clustering
- `results/l_arc_8/step3/` — capturability
- `results/l_arc_8/step4/` — extractability (E + D1 classifiers + policies)
- `results/l_arc_8/step5_wfo/` — WFO results + STEP5_SUMMARY.md

### Diagnostics (post-Step-5, locked)
- `results/l_arc_8/diagnostics/entry_feature_overlap/` (commit `7d9109e`)
- `results/l_arc_8/diagnostics/post_entry_confirmation/` (commit `4756c66`)
- `results/l_arc_8/diagnostics/signal_tightening/` (commit `4756c66`)
- `results/l_arc_8/diagnostics/COMBINED_DIAGNOSTIC_SUMMARY.md`

### Live arc doc
- `results/l_arc_8/ARC_8_LIVE.md` — to be updated to closed status by separate CC dispatch (see prompt)

### Closure doc (this file)
- `results/l_arc_8/ARC_8_CLOSURE.md`

---

## Recommended next dispatch (analyst-side)

1. **Push engine PR** (`feat/open-24-pre-t-sl-per-archetype`) to origin/main — currently merged locally only.
2. **Update tracking files:**
   - `STATUS.md` — Arc 8 → CLOSED (HALT_DEPLOYMENT)
   - `CHANGELOG.md` — append closure entry
   - `SESSION_ZERO.md` — log Arc 8 closure + 3rd Open-22/23/24 confirmation
   - `PROTOCOL_IMPROVEMENT_BACKLOG.md` — add v2.4 §1.5 entry-separability gate as proposal with Arcs 4/5/8 evidence base
3. **Draft v2.4 protocol amendment** (chat-side design work, not arc-level) — §1.5 gate spec, threshold justification, regression test plan.
4. **Open-05 portfolio composition log** — add c1 V-shape recovery FG-weak archetype with admit-only economics and key features.
5. **Resolve stale `phase/l_arc_8` branch** — `git branch -m phase/l_arc_8 phase/l_arc_8_pre_arc8_archive` or delete; fast-forward to `claude/magical-zhukovsky-bd69d9`.
6. **Move to next arc (9 / 10 / 11)** — apply §1.5 gate informally before Step 1 spend.

---

## Commit history (Arc 8 worktree)

- `a80972b` arc-8 open
- `3c5f943` arc-8 step 1 PASS
- `9583947` arc-8 step 2 PASS
- `c34cc6b` arc-8 step 3 PASS
- `a5eb6e6` arc-8 step 4 PASS
- (Step 5 commit sha — see live arc doc)
- `7d9109e` arc-8 diagnostic: entry-feature overlap (c1_NOT_SEPARABLE_AT_ENTRY)
- `4756c66` arc-8 diagnostic: post-entry confirmation + signal-tightening (path1+path2)
- (closure commit — pending this file landing)

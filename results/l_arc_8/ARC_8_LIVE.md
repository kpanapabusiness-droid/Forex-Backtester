# Arc 8 — Pullback-and-resume in HH/HL uptrend (PR-HHHL, long)

## Status

- **Current step:** Arc open (pre-Step-1)
- **Verdict:** none yet
- **Last updated:** 2026-05-18
- **Branch:** worktree `claude/magical-zhukovsky-bd69d9` (dispatcher-target merge to `phase/l_arc_8`)
- **Live doc:** `results/l_arc_8/ARC_8_LIVE.md`
- **Dispatch:** `cc_dispatch_arc_8.md` (under L_ARC_PROTOCOL v2.3 stack)

## Arc-open

### Signal under test

| Field | Value |
|---|---|
| Trial id | `signal_pullback_resume_hhhl_long_v0.1` |
| Source | `docs/signal_spec_pullback_resume_hhhl_long_v0.1.md` (sha256 `d1c04841a1079973e411748c49347a07d59741d7c42483cf8539d97b4454414e`) |
| Family | Trend continuation (structural) |
| Direction | Long only |
| Signal TF | 4H |
| Anchor TF | n/a (single-TF) |

### Hypothesis

This signal carries structural edge surface-able by path-shape clustering and v2.3 capturability + extractability gates. PR-HHHL is the pool-size anchor of the Arc 8-11 parallel batch — if Pipeline E fails here, it informs whether the entry-time-features hypothesis fails on trend-continuation signals broadly.

### Locked parameters (Step 1 sim)

| Field | Value |
|---|---|
| Initial SL | `entry − 2.0 × ATR(14)_4H[t]` (entry-price anchor) |
| ATR period | 14, Wilder, 4H |
| Forward window | 240 bars (4H) |
| Pair set | 28 FX (KH-24 set) |
| Data window | 2020-10-01 → 2026-01-31 |
| Exposure cap | Max 1 open position per pair |
| Risk per trade | 0.5% × reset floor balance |
| Population builder | `build_ex_ante_bounded_population` (single pass, no folds at Step 1) |
| Spread | Per-bar MT5 native points; floor file fallback when raw = 0 |
| Spread floor file | `configs/spread_floors_5ers.yaml` (sha256 `f5f5c584b7181278c0d4ecbcd3383023fb68af81cd7a320da2dc92524392411b`) |
| Spread semantics | `docs/SPREAD_SEMANTICS_LOCK.md` (sha256 `ef0fb938ce37a029b58a6c76b0c13380dc4f73c08a35576c2b200d3dcf951f5c`) |

### Step 3 SL sweep candidates

Default `{0.5, 1.0, 1.5, 2.0, 3.0, 4.0} × ATR_4H` (no spec override).

### Protocol stack

| Doc | sha256 |
|---|---|
| `L_ARC_PROTOCOL.md` (v2.1.2 base) | `fac9a7a8f7c7f81e7a3da664d8866c23f5902a57ebf2ba99127500c65cca438e` |
| `L_ARC_PROTOCOL_v2_2_AMENDMENT.md` | `bf8cc2f8d036111abd354e9c56ff8c34d140bb6d8d56c0a8c41aa01b8f4e08ab` |
| `L_ARC_PROTOCOL_v2_3_AMENDMENT.md` | `db95bbd98a70297acb9934b83852d7efcc4523abf87624075ff27ef558a5cdb6` |

Effective protocol version: **v2.3** (Step 5 cross-fold stability removed; Step 6 → Step 5 = WFO; orchestrator halt at end of Step 4; max-F1 closure under v2.2 §3; Tier 2 lift cap ≤ 5 under v2.2 §2).

### Pre-committed step gates

Per v2.3 (no overrides, no mid-arc sign-off, halt end of Step 4):

| Step | Gate |
|---|---|
| 1 — Plumbing | Pool ≥ 500; byte-identical determinism; lookahead-invariant; §15a schema; spread semantics tests green |
| 2 — Path-shape clustering | K-sweep {3,4,5,6,7}; pick highest silhouette satisfying §6 gate; smaller K within 0.01 tolerance preferred |
| 3 — Capturability | §2 floors at chosen SL per candidate sweep; capturability composite per §7 with tiebreakers; bimodal_separated + ≠ scattered floors |
| 4 — Extractability | Angle E A→B→C → 0.65 AUC lock; Angle D1 smallest-t rule (≥ 0.60 AND exclusion ≤ 30%); threshold sweep with recall ≥ 0.60 (no max-F1 fallback per v2.2 §3); Tier 2 lift ≤ 5 candidates per archetype (v2.2 §2) |

Halt: end of Step 4 (Step 5 WFO is a separate chat-dispatched item).

### Co-fire matrix expectations (informational, per signal spec §66-68)

- **KH-24** (`kb_exhaustion_bar`): bearish exhaustion vs PR-HHHL bullish resume — independence expected; flag if > 10%
- **Arc 9 / 10 / 11**: Step 1 not yet landed; signals/specs unavailable on `main` (Arc 9-11 specs live on `tmp/post-v2_3` only). Co-fire vs Arc 9/10/11 deferred to whichever arc's Step 1 lands second.

### Boundaries per dispatch §24-30

This session owns: arc-open doc, all Step 1-4 scripts, live arc doc, closure doc on early arc death, queue state transitions, branch creation, all commits on the worktree branch.

This session does NOT own: Step 5 WFO dispatch, engine PRs (`scripts/phase_kgl_v2_4h_wfo.py`, `signals/` once written goes into a PR-required scope strictly — but per dispatch boundaries the per-arc signal module is arc-owned), `L_ARC_PROTOCOL.md` edits, ship/archive decisions on Step 5 output.

### Dispatcher-flagged variance

| Item | Variance | Reason |
|---|---|---|
| Branch name | Worktree branch `claude/magical-zhukovsky-bd69d9` instead of `phase/l_arc_8` | This session was opened as a git worktree. Existing local `phase/l_arc_8` branch contains stale pre-v2.2 work (unique commits descend from `phase/v2_2_housekeeping`); not touched. Dispatcher decides at session end whether to fast-forward `phase/l_arc_8` to this worktree branch, or rename the stale branch and create fresh. |
| Spec source | Spec file written from analyst-provided content (matches `tmp/post-v2_3` commit `9e9bf0a` byte-equivalently, sha256 `d1c04841...`) | Spec was not on `main`; lived only on unmerged `tmp/post-v2_3`. Analyst supplied content via dispatcher channel for this session. |
| Data wiring | `data/4hr` is a directory junction inside the worktree → parent repo `..\..\..\..\data\4hr` | `data/` is `.gitignore`d; the worktree had only `data/test/`. Junction is reversible and tracks no new files. |

## Step results

| Step | Gate | Result | Notes |
|---|---|---|---|
| 1 | Plumbing | _pending_ | |
| 2 | Clustering | _pending_ | |
| 3 | Capturability | _pending_ | |
| 4 | Extractability | _pending_ | |

## Detailed analysis

_(none yet)_

## Cross-arc candidates

_(none yet)_

## Interesting observations

_(none yet)_

# Backtester Extension — Closure

> Scope: PR #131 + PR #135 + PR #138. Backtester infrastructure shipped 2026-05-17. This chat closes here.

---

## What Shipped

### PR #131 — Pipeline D1 plumbing
**Merged 2026-05-17.** Hook architecture for mid-trade classifier intervention per L_ARC_PROTOCOL v2.0 §3.

- `core/features_path_so_far.py` — path-shape feature builder, single source of truth for offline training + runtime engine
- `core/d1_pipeline.py` — `D1Hook`, `Close` / `ApplyPolicy` / `Hold` decision dataclasses, YAML config loader
- Engine patch: hook fires at line ~1352 (pre-Priority-1 SL, post-bar_path append); `D1_HOOK = None` byte-identical to pre-PR
- 41 tests
- §3 protocol wording fix: pre-t SL = 2.0 × ATR, "close at N+1 open" replaces "break-even close" framing

### PR #135 — D1 PR 2: Stepwise climber policy + per-fold classifiers
**Merged 2026-05-17.** ApplyPolicy made concrete for §11 row 2.

- `core/exit_policies.py` — `StepwiseClimberPolicy` with MFE-lock + trail-from-high primitives
- Per-fold classifier dispatch: `D1Hook` loads one joblib per fold per archetype; engine threads `fold_id` from WFO loop
- Audit fields: `mfe_lock_fired_bar`, `trail_active_from_bar`, `classifier_fold_id` persisted to trades_all.csv
- Engine patch: per-bar policy `update_per_bar` between bar_path append and SL check
- 72 tests (20 exit-policy + 52 D1-pipeline)
- Trail-suppression resolved: lock-fire bar arms AND uses trail in same-bar SL check, matching Step 5 simulator semantics verified at `scripts/arc_4/step5_stability.py:269-318`

### PR #138 — Engine generalisation
**Merged 2026-05-17.** Engine becomes config-driven for signal, timeframe, time-exit, spread-floor.

- `core/signal_adapter.py` — `SignalAdapter` Protocol + `import_class` + `validate_aux_declaration`
- `signals/kb_exhaustion_bar_adapter.py` — wraps existing KH-24 signal logic; deferred import to avoid module-load cycle
- Engine: `signal_adapter` config drives adapter loading; `signal_tf` + `data.{signal_tf_dir, aux.{h1,d1}}` route data sources; aux loading gated by adapter declaration
- `time_exit_bars` config (optional): max-life cap, fires last in exit cascade, last-bar fallback to current close
- `_apply_spread_floor` wired at all four entry-side spread reads via existing `core/spread_floor.py`
- 22 new tests
- KH-24 byte-identical sha256: `08118567a6ef…58e80ab0` pre- and post-PR

---

## Key Decisions

**Trail suppression dropped.** Initial PR #135 implementation suppressed trail on lock-fire bar (gave SL = entry). Verified against Step 5 simulator: trail is armed AND used same-bar. Dropped suppression; SL at lock-fire = max(BE_floor, trail_stop) = entry + 0.25R for the canonical Arc 4 cluster 1 case.

**Cap-bind 240 became TimeExitPolicy, not hardcoded.** Original PR 2 prompt asked to hardcode 240 as `PATH_FORWARD_BARS`. Correct framing per engine generalisation: `time_exit_bars` is a config-driven optional exit policy that fires last. KH-24 doesn't set it → no behaviour change.

**Engine generalisation kept tight.** Initial draft proposed 3-PR refactor (~15-25 hours) with exit-policy plugin architecture, full schema migration, multi-stage timeframe abstraction. Revised to single ~600-900 LOC PR after recognising the existing engine was already mostly correct; only signal adapter + TF routing + time-exit + spread-floor were genuinely missing.

**Pre-t SL stays uniform at 2.0 × ATR.** Per §3 / PR #131. Archetype-specific SL replaces at bar t via ApplyPolicy. Cluster 1's pre-t SL (2×ATR) being tighter than archetype SL (3×ATR) is structural — flagged for cross-arc backlog if Step 6 surfaces empirical truncation.

---

## What's Unblocked

- Any future arc can run an alternate signal via `signal_adapter` config
- Any future arc can use 1H or other timeframes via `signal_tf` config
- Any future arc surfacing a Stepwise climber archetype can use the D1 hook + StepwiseClimberPolicy directly
- Arc 3 Stepwise climber replay under v2.1.1 (Open-18 #1) has engine support to test extractability properly
- Arc 4 (when it opens under v2.1.1) has all engine prerequisites: signal pluggability, 1H TF, D1 hook, Stepwise climber policy, per-fold classifiers, time-exit, spread floor

---

## What's Deferred

- **§11 row 1, 3, 4, 5, 6, 7 exit policies.** Only row 2 (Stepwise climber) shipped. Other rows (Monotone ascent, Early-peak hold, Peak-and-collapse, V-shape recovery, Random walk, Bimodal) added when an arc surfaces a Step-4 archetype consumer.
- **Step 5C joblib dump.** Operational fix in `scripts/l_arc_4/step5c_refit.py` — train-in-memory classifiers need `joblib.dump` per fold. ~30 min when Arc 4 reaches Step 6.
- **Convention (b) hourly MTM DD + daily DD reporting.** Utility exists in `scripts/l_arc_4/step5b_refit_risk_sweep.py`; needs wrapper to consume engine output. ~1-2 hours when Arc 4 reaches Step 6.

---

## Pre-Commits for Future Step 6 Runs

If a Step 6 real-engine run on a Stepwise climber archetype underperforms its Step 5 leakage-haircut-adjusted simulator estimate by > 20% on worst-fold ROI, two pre-committed competing hypotheses discriminate the cause:

- **H1:** §11 row 2 calibration on pullback tolerance (if archetype's pullback exceeds row 2's ≤ 0.5R rule)
- **H2:** Pre-t SL truncation (2×ATR pre-t < archetype SL causing trades the simulator gave for free)

Discrimination via trade-level audit (`mfe_lock_fired_bar`, `trail_active_from_bar`, pre-t exit timing). Resolution = cross-arc backlog item, not within-arc rescue.

---

## Recovery Note

PR #138 implementation went through a parallel-CC-session worktree collision mid-build. Two adapter files (`core/signal_adapter.py`, `signals/kb_exhaustion_bar_adapter.py`) were untracked + lost during a checkout collision. Recovered by:
1. `git worktree add ../Forex-Backtester-engine-gen feat/engine-generalisation`
2. `git stash pop` for engine + config changes
3. Rewrote the two adapter files from in-context design
4. Re-verified byte-identical KH-24 sha256

Going forward: PR work in this repo uses dedicated `git worktree` per PR to prevent recurrence.

---

## Chat Closure

This chat is closed. Future backtester engine work opens a new chat with this doc as context.

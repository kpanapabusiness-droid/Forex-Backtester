# Execution Truth Invariants

> Project-permanent methodology invariants for the Forex Ignition Rebuild backtester.
> All code, tests, configs, and research must conform to this document.
> Last updated: 2026-05-09 — slimmed to project-permanent invariants only after KH arc closure.
> Supersedes: GOLDEN_STANDARD_LOGIC.md v1.9.8 (NNFX-era architecture spec, archived).

---

## Why This Document Exists

The previous version of this file (v1.9.8) described an NNFX-style entry/exit architecture (C1/C2/baseline/volume catalysts, TP1+runner half-split, WIN/LOSS/SCRATCH classification tied to TP1). That architecture was abandoned with KGL_V2 in early 2026 and superseded again by KH-24 in April 2026. KH-24 has no TP1, no half-split, and uses a single-direction trail-only management.

The architecture-specific content in v1.9.8 is therefore stale. What carries forward — and is genuinely project-permanent — are the execution-truth invariants that any signal architecture in this project must respect. Those invariants are below. They apply to KH-24 today, the L characterization arc going forward, and any signal architecture this project ever ships.

---

## 1. No Lookahead

The defining invariant. No decision at bar N may use any data that wasn't available at bar N's close.

Specific applications:

- Signal evaluation uses bars `[..., N]` only. Entry fills at bar N+1 open. Same-bar entry is forbidden under all circumstances.
- D1 alignment to lower timeframes uses lag-1 convention: a 4H bar at timestamp T uses the prior calendar day's D1 close, never same-day.
- Indicator computation uses backward-looking windows only (`pd.merge_asof` direction='backward', never 'forward' or 'nearest').
- Population construction uses ex-ante labeling only. `build_ex_ante_bounded_population` is mandatory for any signal research.
- Clean path-dependent labels (e.g., MFE-before-MAE labels) are evaluation tools only. Never used in population selection.

Detection: any code that reads bar N+k for k > 0 to make a decision at bar N is a lookahead violation regardless of intent.

The D1 lookahead bug uncovered in early 2026 cost the project significant rework and motivated this invariant being the first one in the document. The lesson is non-negotiable.

## 2. Audit Field Immutability

Fields recorded at trade entry time are immutable for the life of the trade.

- `entry_price`, `entry_time`, `sl_at_entry_price`, `initial_sl_distance` set once at entry. Never mutated.
- Any dynamic stop level (trail level, breakeven shift, kijun-based exit level) lives in a separate field (e.g., `current_sl`, `trail_level`) so the original entry-time values remain auditable.
- Exit-time fields (`sl_at_exit_price`, `exit_time`, `exit_reason`) recorded at exit, never modified after.

This invariant exists so trade reconstruction from CSV is unambiguous and trade-level diagnostics (KH-29 style excursion analysis) are always possible.

## 3. Spreads Affect PnL Only

Per-bar MT5 spread data is applied at entry. Enabling, disabling, or changing spread data:

- Changes per-trade PnL.
- NEVER changes trade counts.
- NEVER changes trade entry/exit timing.

Hardcoded spread defaults are forbidden. Always use the per-pair, per-bar `spread` column from MT5 data, converted to pips as `spread_points / 10`.

The 5ers parity arc (early 2026) confirmed that broker-specific spread data is materially different from defaults and that signal counts must remain identical across spread on/off comparisons.

## 4. Date Ranges Strictly Enforced by Config

WFO fold boundaries, IS/OOS windows, and any other temporal scope are enforced via config (`date_range.start`, `date_range.end`). Code never overrides these values.

Fold separation is strict: an OOS bar in fold N may not appear in any IS window of any later fold. Verified by parameter hash and seed before each WFO run.

## 5. Config-Driven, No Hardcoded Parameters

All system parameters live in YAML configs. Any change to a parameter is a config edit.

This applies to: entry conditions, ATR multiples, risk percent, exposure caps, filter thresholds, spread settings, fold structure, output paths.

Hardcoded values in Python code are forbidden except for default fallbacks marked explicitly as such (and ideally still overridable via config).

## 6. Deterministic Outputs

Same input data + same config + same code = same output bytes. Always.

Random seeds are explicit in config when randomness is used. Bootstrap, Monte Carlo, and any other stochastic computation is controlled by `rng_seed` parameters.

CI verifies this on every PR via smoke tests against locked baseline configs (`wfo_baseline_clean.yaml`, `wfo_kh24.yaml`). These baseline configs may NEVER be modified.

## 7. CI Required Gates

Every PR runs:

- Lint (`ruff check .`)
- Unit tests (`pytest -q`)
- Smoke test (one fast deterministic backtest run)

CI green is mandatory before merge. Failing CI is not negotiable. Even doc-only PRs run CI to catch accidental file changes.

## 8. Phase Discipline

The discipline that produced KH-24 and the three clean diagnostic verdicts (KH-27, KH-28, KH-29) is project-permanent. It is the methodology, not a phase-specific convention.

- One change per phase. Pre-commit the gate before running. Accept the result, including nulls.
- Every phase produces a result document regardless of pass or fail.
- Locked gates are not loosened post-hoc. KH-29's closest-miss (0.016R off the threshold) was still a miss. Loosening thresholds after seeing results is the rationalization that destroys the methodology.
- Pre-flight analysis is preferred when it can falsify a hypothesis cheaply. KH-27 falsified its hypothesis in one day of analysis instead of weeks of WFO.
- Post-hoc rationalization of failed phases is forbidden. "The gate was too strict" or "the signal would have worked except..." are the patterns that destroy research integrity.

## 9. Population Integrity

Population selection for any signal research must be ex-ante.

- Use `build_ex_ante_bounded_population` for any test where the population determines what gets evaluated.
- No forward-conditioned logic anywhere in dataset construction.
- No outcome-aware gating in classification.
- No clean labels in selection.

Phase JL invalidated an entire promising research arc on this exact violation. The bounded event system showed 79.8% win rate and +0.53R expectancy under forward-biased population selection; rebuilt ex-ante it produced -0.05R and -50% ROI. The lesson is permanent.

## 10. Permanently Excluded Tools

The following are permanently excluded from any implementation work on this project:

- **GPT-4** — produces hallucinated MQ4→Python conversions. "Indicator failures" attributed to it during early eras were implementation failures, not signal failures.
- **Aider** — replaced by Claude Code.

Implementation tools currently in use:

- Claude (this chat) for planning, research interpretation, and decisions
- Claude Code for multi-file work and correctness-critical changes
- Cursor for single-file patches and YAML edits

No other AI tools touch the code without an explicit methodology change.

---

## Acceptance Tests (Must Pass)

These tests verify the invariants above. Any backtester change must keep them passing. Specific systems (e.g., KH-24) layer additional system-level acceptance tests on top.

1. **No lookahead in signal evaluation.** Set up a synthetic signal that fires at bar N. Verify entry executes at bar N+1 open, never same-bar.

2. **D1 lag verification.** Construct 4H bars at calendar-day boundaries. Verify the D1 value used at the first 4H bar of day T is from D1 bar T-1 close, never T.

3. **Audit field immutability.** Run a trade through trail/exit cycles. Verify `entry_price`, `entry_time`, `sl_at_entry_price`, `initial_sl_distance` are unchanged at every bar of the trade life. Verify dynamic levels (trail, kijun-based exit) are written to separate fields.

4. **Spread invariance to trade count.** Run the same config with `spreads.enabled=true` and `spreads.enabled=false`. Verify trade count and entry/exit timing are byte-identical; only PnL differs.

5. **Determinism.** Run the same config twice in two clean environments. Verify byte-identical output (trades CSV, equity curve, summary).

6. **Fold separation.** Run a 7-fold WFO. Verify no OOS bar in fold N appears in any IS window of fold N+k for k > 0.

7. **Config-driven parameters.** Grep the codebase for hardcoded values that match config-managed parameters (risk %, ATR multiples, spread defaults). Should produce zero results except in test fixtures and documented default fallbacks.

8. **Ex-ante population.** For any signal research script, verify population construction uses `build_ex_ante_bounded_population` and no clean-label or outcome-aware fields appear in the selection logic.

---

## Relationship to Other Documents

- **KH24_SYSTEM_LOCK.md** — describes the specific entry/exit architecture, parameters, and acceptance tests for the KH-24 live system. KH-24 inherits all invariants in this document and adds system-specific rules.
- **L_ARC_PLAN.md** — describes the L characterization arc methodology. L inherits all invariants in this document; the descriptive-not-predictive rule and pre-registered bridge are L-arc-specific additions.
- **L_ARC_PROTOCOL.md §1 + §12** — methodology principles and operational rules that apply across all arcs.
- **WORKFLOW.md** — phase management conventions.
- **BACKTESTER_AUDIT.md** — implementation-level audit of the backtester engine.

---

*The invariants in this document are project-permanent. They survived the KGL_V2 → KH transition and the KH → L transition because they are not tied to any specific signal architecture. They apply to whatever the project ships next.*

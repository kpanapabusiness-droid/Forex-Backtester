> # ⚠️ SUPERSEDED 2026-05-13
>
> This document is SUPERSEDED by `L_ARC_PROTOCOL.md` v1.0 (now archived at `archive/L_ARC_PROTOCOL_v1_0.md`) and `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md`. As of 2026-05-16 the active protocol is `L_ARC_PROTOCOL.md` v2.0.
>
> **What is superseded:** §9 (no filter rescue, verbatim-as-gate) and §14 disposition rules — replaced by the six-step extractability protocol with dual-tier WFO disposition.
>
> **What carries forward unchanged:**
> - §14.3 feature schema (referenced by `archive/L_ARC_OPERATIONAL_SPEC_v1_0.md` §5.16; v2.0 has its own feature set per `L_ARC_PROTOCOL.md` §8)
> - §5 pair set (28 FX pairs)
> - §4 WFO structure (7 anchored expanding folds, OOS Oct 2020 – Jan 2026)
>
> **Why this file is kept:** historical record of the methodology that governed Arcs 1 and 2 in their original runs. Useful for understanding the redesign rationale documented in `L_ARC_PROTOCOL.md` §1.
>
> **Do not use this doc as authority for new arc work.** Read `L_ARC_PROTOCOL.md` first.
>
> ---

# L6.0 — Methodology Lock

**Arc:** L6+ signal-testing arc (post-L characterization)
**Status:** LOCKED (pending sign-off)
**Current version:** v1.1 (additive amendment, see Document History)
**Locked date:** 2026-05-10 (v1.0); 2026-05-10 (v1.1 amendment)
**Predecessor docs:** `docs/L0_METHODOLOGY_LOCK.md`, `LCHAR_TOPN_REGISTRY.md`, `docs/L_ARC_DEFERRED_CANDIDATES.md`, `results/lchar/L_ARC_CLOSURE.md`, `KH24_SYSTEM_LOCK.md`, `SPREAD_SEMANTICS_LOCK.md`, `DATA_AUDIT_REPORT.md`
**Modification rule:** No modifications to this document after sign-off without an explicit re-planning phase. Re-planning requires a new phase document, justification for the change, and acknowledgement that the change resets the relevant downstream work. Mirrors L0 §1 / §12.

---

## Document History

| Version | Date | Scope | Notes |
|---------|------|-------|-------|
| v1.0 | 2026-05-10 | Original lock, §1–§18 | Initial methodology lock for the L6+ signal-testing arc. |
| v1.1 | 2026-05-10 | Additive amendment — Characterisation Outputs | Adds non-gating per-arc characterisation deliverable + `CANDIDATE_HYPOTHESES.md` registry. Does not modify v1.0 §1–§13. Appended at end of document; see "Amendment v1.1 — Characterisation Outputs". |

---

## 1. Purpose

L6.0 is the methodology lock for the L6+ signal-testing arc. It commits, in advance of any signal test, to which candidates are tested, in what order, under what gate, with what direction handling, with what spread / risk / pair-set treatment, and under what stopping conditions. Once L6.0 is signed off, the arc's degrees of freedom are constrained to *what the data shows under this methodology* — no post-hoc gate changes, no candidate re-ordering, no scope expansion.

L6.0 contains no code and no data analysis. It is documentation only. The lock is what makes everything that follows safe.

L6.0 does **not** specify the tradable conversion of any individual L registry condition (entry rule, exit rule, SL spec). That is the job of each arc-opening phase document.

---

## 2. Scope

### In scope

- Verbatim WFO testing of the 5 entries in `LCHAR_TOPN_REGISTRY.md` (signal_TF = 1H, all directions long per §6).
- Subsequent filter / exit / sizing phases on any registry candidate that passes its verbatim WFO gate, under one-change-per-phase discipline.

### Out of scope

- KH-24 modifications of any kind. KH-24 is the live production system on the VPS and is unaffected by L6+ outcomes.
- Portfolio construction questions (whether any L6+ system complements KH-24).
- The 7 deferred candidates in `docs/L_ARC_DEFERRED_CANDIDATES.md`. They are not in scope unless an explicit re-planning trigger fires per §12.
- Short-side testing on any registry condition. All 5 registry entries imply long trades (positive pooled conditional mean); short mirrors are mathematically degenerate and not run. See §6.
- Any signal_TF other than 1H. The full registry is at 1H. Cross-TF testing of the same conditions is a re-planning question.
- Window extension. The 2020-10-01 → 2025-12-31 OOS window from the L arc is inherited verbatim. Window extension reopens L1–L5.

---

## 3. Hypothesis-Class Sequencing

The 5 registry entries collapse structurally to 3 hypothesis classes:

| Class | Registry ranks | Common feature |
|-------|----------------|----------------|
| A — extreme-bar mean-reversion at 1H, h=1 | 1, 4 | `abs_return_top_decile_neg` (rank 1) and `bar_range_top_decile_neg` (rank 4) measure the same phenomenon (large-magnitude down bar) via different bases |
| B — MTF 2-down-mixed kijun alignment at 1H | 2, 5 | Same condition, two horizons (h=120 and h=24) |
| C — high D1-vol regime at 1H, h=120 | 3 | Standalone |

### Arc ordering (locked)

| Arc | Registry rank | Class | Trigger |
|-----|---------------|-------|---------|
| Arc 1 | 1 | A | Unconditional — opens immediately on L6.0 sign-off |
| Arc 2 | 2 | B | Unconditional — opens after Arc 1 closes |
| Arc 3 | 3 | C | Unconditional — opens after Arc 2 closes |
| Arc 4 | 4 | A duplicate | Conditional on Arc 1 outcome (see below) |
| Arc 5 | 5 | B duplicate | Conditional on Arc 2 outcome (see below) |

### Conditional-duplicate scheduling

The scheduling of Arcs 4 and 5 is decided in writing **after** the parent arc closes:

- If parent arc passed verbatim WFO: the duplicate is run as a parameter-stability ablation within the parent arc's subsequent phase work. Result documented but not required to pass independently to validate the parent.
- If parent arc failed verbatim WFO: the duplicate is downweighted but not eliminated. It may be run as a separate arc only after all other unconditional registry candidates have been tested, with the explicit framing that the parent failure reduces the prior on the duplicate passing.

The decision is recorded in the parent arc's closure record, not pre-committed here.

### Why this ordering

Arcs 1–3 cover all three distinct hypothesis classes in priority of DSR rank. This maximises information per compute unit: if all three classes verbatim-fail, the registry's edge thesis is broadly rejected and Arcs 4–5 are unlikely to rescue it. If any class passes, the duplicate becomes a meaningful sensitivity check on the passing structure.

---

## 4. Per-Arc Gate

The gate is applied to the verbatim WFO of every arc and to every subsequent filter phase within an arc. It is locked at L6.0 and may not be modified without re-planning.

### Gate definition

A WFO PASSES the gate if and only if all three conditions hold across all 7 anchored expanding folds:

1. **Worst-fold ROI > 0%** (every fold returns positive)
2. **Worst-fold max DD < 8%** (no fold exceeds the prop-firm-aligned drawdown ceiling)
3. **Trades per fold ≥ 15** (per-fold sample floor)

If any of the three fails on any fold, the gate FAILS and the phase is a clean null.

### WFO structure (inherited from KH-24 §3)

| Property | Value |
|----------|-------|
| Method | Anchored expanding window |
| Folds | 7 |
| IS period | 4 years per fold start |
| OOS period | 9 months per fold |
| OOS coverage | 2020-10-01 → 2026-01-01 |

### Inheritance from KH-24

Conditions 1 and 2 are KH-24 verbatim. Condition 3 is the only addition: an explicit per-fold trade-count floor at 15. KH-24's worst-fold trade count was 23 (well above 15); the floor codifies what KH-24 implicitly satisfied and prevents thin-fold luck on rare-condition signals at 1H.

### What the gate does NOT include

- No Sharpe gate, no win-rate gate, no profit-factor gate, no expectancy gate.
- No portfolio-equity-curve gate.
- No comparison to KH-24 performance.

The two-threshold simplicity of KH-24's gate is preserved. The trade-count floor is structural, not performance-tuning.

---

## 5. Pair-Set Scope

**All 28 pairs, every arc, every phase.**

Pair set inherited from KH-24 §2 verbatim:

```
AUD_CAD AUD_CHF AUD_JPY AUD_NZD AUD_USD CAD_CHF CAD_JPY CHF_JPY
EUR_AUD EUR_CAD EUR_CHF EUR_GBP EUR_JPY EUR_NZD EUR_USD GBP_AUD
GBP_CAD GBP_CHF GBP_JPY GBP_NZD GBP_USD NZD_CAD NZD_CHF NZD_JPY
NZD_USD USD_CAD USD_CHF USD_JPY
```

### No pre-WFO pair selection

The L4 atlas's per-pair Sharpe diagnostic distributions are computed over the full 2020–2025 window. Selecting pairs from them — in any form, including "top-N before WFO" — leaks future-window information into IS partitions. This is the same JL trap that invalidated the bounded-event arc; it is not relaxed here.

### Pair-restriction as a subsequent phase

If a verbatim universe-wide WFO passes the gate but is borderline (e.g., worst-fold ROI < 1%), a *separate subsequent phase* may test "top-N pairs by per-fold IS-only per-pair Sharpe." Restriction to a sub-universe is a different hypothesis from universe-wide and falls under one-change-per-phase. L6.0 does not pre-commit pair-set restriction parameters; arc planners decide if and how to apply them per phase.

---

## 6. Direction Handling

### Direction derivation rule

Trade direction for any L registry candidate is derived mechanically from the sign of the pooled conditional mean return at the registry's stated horizon:

- Sign(pooled_mean_r) > 0 → long
- Sign(pooled_mean_r) < 0 → short

For the 5 current registry candidates:

| Rank | Condition | Pooled mean_r (ATR-norm) | Direction |
|-----:|-----------|-------------------------:|-----------|
| 1 | abs_return_top_decile_neg, h=1 | +0.0306 | LONG |
| 2 | 2_down_mixed kijun, h=120 | +0.2773 | LONG |
| 3 | d1_atr_top_decile any, h=120 | +0.1290 | LONG |
| 4 | bar_range_top_decile_neg, h=1 | +0.0268 | LONG |
| 5 | 2_down_mixed kijun, h=24 | +0.1058 | LONG |

**All 5 registry candidates are long-direction trades.** The `_neg` and `_down_mixed` strings in the trial IDs describe the **condition** (price went down), not the trade. The trade is long mean-reversion (price expected to revert up after a down event).

### Short-mirror requirement

Not required for any of the 5 registry candidates. Short P&L on a positive-mean condition is, by construction, the negative of long P&L (modulo asymmetric execution costs); a short mirror on these conditions is degenerate and produces a guaranteed gate fail. No compute is spent on it.

If a future arc tests a candidate with a negative pooled mean (none currently in scope), a short-mirror sanity test before WFO commits compute is required — to be specified in that arc's opening doc.

### Direction-explicit execution-engine assertion (CI-enforced)

The backtester must enforce, on every taken trade, that the SL is on the opposite side of the entry from the trade direction:

- Long trade: `assert sl_price < entry_price`
- Short trade: `assert sl_price > entry_price`

Violation halts the WFO with a hard error. This guards against the SIGNAL_DIRECTION class of silent bug observed in KH-arc short-mirror work. CI-enforced; never bypassed regardless of phase or arc.

---

## 7. Spread Cost Modelling

### Per-bar MT5 spread, with per-pair non-zero floor

The KH-24 / SPREAD_SEMANTICS_LOCK convention of per-bar MT5 spread on entry only is inherited. The only divergence is a deterministic floor for zero-spread bars.

### Floor rule

```
effective_spread[bar, pair] = max(observed_spread[bar, pair], min_nonzero_spread[pair])
```

where `min_nonzero_spread[pair]` is the minimum non-zero spread observed for that pair across the full dataset (all timeframes pooled, full 2010–2025 window).

### Why a floor is needed at 1H but was not at 4H

Per `DATA_AUDIT_REPORT.md` §5, 1H zero-spread bar rates are 5.14% (GBP_NZD) to 30.97% (EUR_USD). The most-traded majors are worst — consistent with broker-side quantization to zero, not missing data. KH-24 ran at 4H, where zero-spread rates are negligible; the issue does not arise there. At 1H, treating recorded spread = 0 as authentic execution cost would inflate net edge in WFO across the very pairs that matter most.

### Floor semantics

- Deterministic, data-driven, zero free parameters.
- Persisted in `configs/spread_floors_5ers.yaml`, sha256-hashed, CI-checked at every WFO run, never modified after L6.0 lock.
- Conservative direction: execution can never be priced tighter than the tightest real spread observed for that pair.

### High-spread outliers — no cap

Per audit, max-spread outliers (e.g., AUD_JPY 460 pips, GBP_NZD 369 pips) are rare (≤ 22 bars per pair, < 0.05%) and represent real broker behavior at news / Sunday-open events. No cap is applied. Live execution would face the same spreads.

### What is NOT permitted

- Hardcoded default spread (e.g., "0.5 pip for majors"). This is the failure mode SPREAD_SEMANTICS_LOCK exists to prevent.
- Per-arc spread overrides.
- Modifying `spread_floors_5ers.yaml` after L6.0 lock without explicit re-planning.

---

## 8. Risk per Trade and Drawdown Limits

### Risk per trade

**1% of reset floor balance per trade.** KH-24 inheritance verbatim.

### Drawdown limits

| Limit | Value | Enforcement |
|-------|-------|-------------|
| Max DD per fold | < 8% | Gate condition (§4.1) — fail closes the phase |
| Daily DD | < 4% | Operational only — logged per fold, not gated. Daily DD is enforced live by the prop firm separately from WFO economics. |

### Position sizing

- Position size = (1% × reset_floor_balance) / (sl_distance_in_currency_units)
- Reset floor balance recomputed at start of each calendar month per 5ers convention
- No compounding within the WFO economics beyond what the reset-floor rule produces

---

## 9. One Change Per Phase

### Verbatim Phase 1 of every arc

The first WFO test of any registry candidate uses the **minimum-implementation tradable conversion** of the L4 condition. Specifically:

| Property | Default for verbatim phase 1 |
|----------|------------------------------|
| Entry trigger | L4 condition fires at signal_TF=1H bar N close |
| Entry execution | Bar N+1 open (never same-bar) |
| Stop loss | 2.0 × ATR(14) from entry price (KH-24 convention) |
| Time exit | At horizon h bars (h=1, 24, or 120 per registry entry) |
| Filters | None |
| Trail | None |
| Structural exits | None |
| Pair set | All 28 |
| Direction | Per §6 sign rule |
| Risk | Per §8 |
| Spread | Per §7 |

The arc-opening doc may specify minor implementation details (e.g., handling of conflicting same-bar signals across pairs, rounding conventions) but may not add filters, trails, or structural exits to phase 1.

### No filter rescue

If the verbatim WFO fails the gate, the candidate is closed. **No post-hoc filter additions to rescue a failed verbatim.** Filter additions, exit improvements, exposure caps, trails — these are subsequent phases of an arc that **already passed verbatim**, never a way to revive a null.

This is the strict version of one-change-per-phase. It is more restrictive than KH-arc's actual path (KH-1 baseline alone did not pass; KH-24 passed only after layered additions). The reason for the strict version: the L registry candidates have already been pre-screened by L4's DSR with selection-bias correction. Verbatim-pass is the realistic-execution counterpart of that pre-screen. If a candidate cannot survive plain execution costs, layered tuning to rescue it is the JL failure mode under a different name.

### Subsequent phases

Once a verbatim phase 1 passes, the arc may add one structural change at a time, each with its own gate test, each with its own pre-committed expectation. Examples (not exhaustive):

- Filter addition (e.g., MTF confirmation overlay)
- Exit improvement (e.g., ATR trail, structural exit)
- Exposure cap (e.g., currency cap analogous to KH-24 Component 2)
- Pair-set restriction (per §5)

Each subsequent phase produces its own result document. Phase ordering and selection is at the arc planner's discretion within one-change-per-phase.

---

## 10. Time Budget

| Bound | Value |
|-------|-------|
| Active per-arc WFO + result doc | ≤ 5 working days |
| Active total across all arcs | ≤ 2 weeks (with parallelism between arcs where Claude Code permits) |
| Calendar hard cap | 4 weeks |

If the calendar cap is hit before all unconditional arcs (1–3) close, the L6+ arc closes with whatever is documented; subsequent direction is a separate planning phase. Cap is not extensible without re-opening L6.0.

---

## 11. Stopping Criteria

The L6+ arc closes on the first of the following:

1. **All 5 registry candidates have a verbatim WFO result documented** (pass or fail). Default closure path.
2. **First verbatim pass** — the passing arc continues into filter phases for that candidate; remaining unrun verbatim tests deferred to a successor planning phase. Closure document records what is run vs deferred.
3. **Calendar hard cap (4 weeks)** — close with whatever is documented.

Each closure produces a closure document at `results/l6/L6_ARC_CLOSURE.md` analogous to `results/lchar/L_ARC_CLOSURE.md`.

---

## 12. Deferred-Pool Re-Planning Trigger

The 7 deferred candidates in `docs/L_ARC_DEFERRED_CANDIDATES.md` are NOT in scope for L6+ by default.

A deferred candidate may enter L6+ scope only via an **explicit re-planning phase document** that:

1. States which deferred candidate(s) are being added and why
2. Justifies the addition under one of the two permitted triggers:
   - **Trigger A:** All 5 registry candidates have failed verbatim WFO. Re-planning may add deferred candidates to extend the search.
   - **Trigger B:** A registry candidate has passed and the planner wants to expand the related-hypothesis universe (e.g., test class duplicates from the deferred pool).
3. Opens a new arc per the same L6.0 rules (verbatim phase 1, gate per §4, all the same constraints).

No silent expansion. Any deferred-pool entry without a written re-planning phase is a methodology breach and resets the work.

---

## 13. Project-Permanent Rules Carried (from L0 §9)

The following apply unchanged to all L6+ arc work and to anything that follows:

- No lookahead anywhere in any signal, indicator, alignment, stop, or exit logic
- `build_ex_ante_bounded_population` mandatory on every signal test
- WFO worst-fold is the only judge of signal success
- One change per phase, pre-committed gate, accept the result (including nulls)
- Per-bar MT5 spread (with §7 floor); never hardcoded defaults
- Risk = 1% of reset floor balance
- Clean labels are evaluation-only, never used for population selection
- GPT-4 and Aider permanently excluded from implementation
- Every phase produces a result document regardless of pass or fail
- Config-driven via YAML, no hardcoding, deterministic outputs, CI-enforced

---

## 14. What Does NOT Carry from L Arc

These were L-arc-specific and do not apply to L6+:

- DSR ranking metric (replaced by WFO worst-fold gate per §4)
- 300-trial family budget (replaced by per-phase one-change discipline per §9)
- Descriptive-only language (L6+ tests; signals either pass or fail)
- N=5 registry cap (operational, can be revisited via §12 re-planning)
- 192-trial L4 budget bookkeeping (no longer relevant)

---

## 15. Risk Register

| Risk | Countermeasure |
|------|----------------|
| Forward bias re-entering via post-hoc gate or pair-set tuning | Gate locked at §4. Pair-set locked at §5. Modifications require explicit re-planning. |
| Filter-tuning rescue of a failed verbatim | §9 explicitly prohibits filter rescue. Failed verbatim closes the candidate. |
| Silent SIGNAL_DIRECTION bug (KH short-mirror class) | §6 CI-enforced SL/entry direction assertion on every taken trade. |
| 1H zero-spread inflating net edge | §7 per-pair non-zero floor, deterministic, CI-checked. |
| Thin-fold luck producing false gate pass | §4 condition 3: trades-per-fold ≥ 15 floor. |
| Time budget overrun without closure | §10 4-week calendar hard cap. Close with whatever is documented. |
| Deferred-pool silent expansion | §12 explicit re-planning required, no silent additions. |
| KH-24 confusion | §2 explicit out-of-scope. KH-24 is live, unaffected by L6+. |

---

## 16. Tool Usage

- **This chat (Opus 4.7, extended thinking on):** planning, methodology decisions, gate evaluation, arc-result interpretation
- **Cursor:** documentation patches, YAML config edits, single-file changes
- **Claude Code (Opus 4.7, xhigh effort, 1M context where available):** backtester runs, WFO execution, signal computation, multi-file backtester changes
- **GPT-4 and Aider:** permanently excluded

---

## 17. Out of Scope (consolidated)

- KH-24 modifications of any kind
- Live system, broker, VPS, or EA changes
- Portfolio construction questions (KH-24 + L6+ combinations)
- Cross-arc performance comparisons
- Short-side testing on any registry condition (degenerate per §6)
- Signal_TF other than 1H without window re-extension
- Window extension (2020-10-01 → 2025-12-31 OOS inherited verbatim)
- Modifications to `configs/spread_floors_5ers.yaml` post-lock
- Modifications to the registry pair set post-lock

---

## 18. Sign-Off

Once this document is committed, L6+ methodology is locked. Arc 1 opens immediately on sign-off with its own arc-opening phase document at `results/l6/arc1/PHASE_L6_ARC1_OPEN.md`.

**Signed off by:** _______________
**Date:** _______________

---

*End of L6.0 v1.0 Methodology Lock. This document is the operational reference for the L6+ signal-testing arc until the arc closes per §11.*

---

# Amendment v1.1 — Characterisation Outputs (Non-gating, Standard Per Arc)

**Status:** Additive amendment to L6.0, locked at v1.1.
**Locked date:** 2026-05-10
**Scope:** Does not modify v1.0 §1–§13. Adds standard per-arc characterisation deliverable, hypothesis registry, and v1.1 magnitude-distribution outputs.
**Numbering note:** Section numbers in this amendment (§14.1–§14.7) are **local to the amendment** and refer only to its own internal structure. They are independent of the v1.0 headers §14 ("What Does NOT Carry from L Arc") through §18 ("Sign-Off"). Cross-references to §4, §9, etc. inside the amendment refer to v1.0 sections.

---

## §14. Characterisation Outputs (Non-gating, Standard Per Arc)

**Status:** additive amendment to L6.0, locked at v1.1. Does not modify §1-§13.

### §14.1 Mandate

Every arc produces, alongside its verbatim WFO, a characterisation deliverable on the
same trade-set. The characterisation is descriptive only: it produces information
about what the signal does, not gates, not filters, not system modifications.

The verbatim WFO disposition (PASS / FAIL per §4) is determined solely by the WFO
result and is unaffected by characterisation output. A FAIL gate disposition closes
the arc per §9; characterisation findings do not rescue it.

### §14.2 Disposition rule (preserves §9 no-filter-rescue)

Patterns observed in characterisation are candidate hypotheses for **fresh arcs**, not
modifications to the current arc. The graduation path:

1. Characterisation reveals a candidate effect (e.g., signal performance varies with
   feature X).
2. The candidate is logged in a permanent `CANDIDATE_HYPOTHESES.md` registry with
   the arc that produced it, the effect description, and effect size.
3. If the candidate is selected for testing, a fresh arc opens with the filter
   pre-committed (threshold, definition, gate) BEFORE any fold result is seen.
4. The fresh arc runs its own verbatim WFO with the new filter as part of the locked
   spec, producing its own characterisation output.

The "fresh arc with pre-committed filter" rule is non-negotiable. Deriving a filter
from one arc's characterisation and applying it within the same arc is a §9 violation
and reproduces the JL forward-bias trap.

### §14.3 Standard feature set (locked)

Every arc's characterisation produces per-signal features in the following groups.
Definitions are inherited from `configs/<arc>_characterisation.yaml` which extends
the locked feature schema.

- **Identity / context**: pair, signal_bar_ts, fold_id, arc1_fold_disposition (or
  current arc fold disposition for arc-internal use).
- **Pre-signal 1H context**: cum_logret_1h_{3,6,12,24}, run_length_into_signal,
  atr_1h_at_n, atr_1h_regime, atr_1h_slope_5, range_position_20, acf1_returns_20,
  range_expansion_5, realized_vol_24h, realized_vol_120h.
- **Signal-bar properties**: bar_size_atr, bar_body_atr, close_position_in_bar,
  signal_zscore_100, spread_at_signal_pips, spread_floored_at_signal.
- **Volume (all TFs)**: volume_{1h,4h,d1,w1}_at_lag1 (or _at_n for 1H), median_50,
  ratio, zscore_100. Volume null handling: zero → NaN, propagate, NaN-rate audit per
  pair × session.
- **Baseline distances (all TFs, ATR-normalised, signed)**: dist_to_kijun_{1h,4h,d1}_atr,
  dist_to_ema20_{1h,4h,d1}_atr, dist_to_ema50_{4h,d1}_atr, dist_to_ema8_{w1}_atr,
  dist_to_ema20_{w1}_atr.
- **MTF lag-1 features**: cum_logret per TF, ATR per TF, ATR regime per TF, EMA slopes
  ATR-normalised. Lag-1 timestamps logged: ts_4h_used, ts_d1_used, ts_w1_used.
  Runtime assertions: 4H ≤ T_N, D1 strict prior day, W1 strict prior week. Hard-fail
  on violation.
- **Time / session / liquidity**: hour_utc, dow, session, bars_to_weekend_close,
  bars_since_weekend_open, pair_floor_rate_100h.
- **Cross-pair / portfolio context**: concurrent_signals_same_bar,
  concurrent_signals_within_3h, currency basket moves (usd_basket_3h, eur_basket_3h,
  jpy_basket_3h, gbp_basket_3h). NB: cross-pair density is computed post-hoc on the
  full signal set; this is descriptive use, not a runtime filter input.
- **Classification labels**: d1_trend_label, h4_trend_label, h1_trend_label,
  mtf_alignment, pre_momentum_label, structural_pattern, cluster_label.
- **Trade-level outcome**: entry/exit prices and timestamps, exit_reason, gross_r,
  net_r, spread_cost_r, mfe_held_atr, mae_held_atr.
- **Forward horizon outcome (descriptive only)**: fwd_logret_h{1,6,24,72,120,240},
  fwd_mfe_h{H}_atr, fwd_mae_h{H}_atr, bars_to_{plus,minus}_{1,2}atr_capped_240h.
- **Magnitude distribution outputs (locked from v1.1)**: net R full distribution
  (mean, std, skew, kurt, min, p1/p5/p25/p50/p75/p95/p99/max); gross R same; SL hit
  rate; conditional means by exit_reason; winner / loser conditional distributions;
  wasted-MFE statistic (held-bar MFE not captured by the time-exit).

Arc-specific signal features may be added to this set per arc (e.g., kijun-distance
features for kijun-conditional arcs). They are documented in the arc-opening doc and
the characterisation config; they do not replace the standard set.

### §14.4 Standard outputs (locked)

Every arc writes to `results/<arc>/characterisation/`:

1. `signals_features.csv` — one row per taken signal, full feature set above.
2. `characterisation_report.md` — narrative with: header (sha256s, determinism),
   population summary, NaN audit, classification breakdown, conditional outcome
   tables (per-label, per-decile), forward-horizon curves, per-fold-disposition
   breakdown (lift ratios), edge-vs-cost decomposition, **magnitude distribution
   tables (locked from v1.1)**, findings summary (empirical observations only — no
   recommendations).
3. `regime_breakdown.csv` — pattern × MTF cells.
4. `forward_horizon_curves.csv` — long-format pattern × horizon.
5. `pair_breakdown.csv` — per-pair aggregations.
6. `magnitude_distribution.md` — distribution tables, SL hit rate, wasted-MFE
   (locked from v1.1).
7. `feature_lag_audit.txt` — lag-1 receipts on a deterministic 100-row sample.
8. `run_manifest.txt` — input / output sha256s, determinism confirmation, byte-
   identicality across two consecutive runs.

### §14.5 Disposition discipline (enforced)

The characterisation report's findings section is **empirical observations only**.
Statements of the form "filter X would help" or "this pattern should be excluded"
are forbidden. Permitted statements are factual: "label Y has mean net_r of Z on
n=W trades vs pooled mean of P." Anything action-shaped goes to
`CANDIDATE_HYPOTHESES.md` for future-arc selection.

The same forbidden-language rule applies to any analyst (human or AI) reading the
characterisation. The temptation to act on a strong observed effect within the
current arc is the JL trap. The methodology safeguard is the fresh-arc-with-pre-
committed-filter rule, not analyst self-restraint.

### §14.6 Backfill rule

Arc 1's characterisation has been produced under spec v1.0. The v1.1 magnitude
distribution outputs are backfilled to Arc 1 as a one-time supplementary run on
the existing `signals_features.csv` (no re-run of the pipeline; no schema change;
read-existing-CSV-only). Future arcs produce v1.1 outputs natively.

### §14.7 Hypothesis registry

A new file `docs/CANDIDATE_HYPOTHESES.md` is created at v1.1 lock. It records each
candidate effect with: source arc, observation summary, effect size, n, what a
fresh-arc gate would look like (definition + threshold), recommended priority.
Selection of which candidate becomes the next arc is a planning decision separate
from characterisation; the registry is evidence, not a roadmap.

---

*End of Amendment v1.1.*

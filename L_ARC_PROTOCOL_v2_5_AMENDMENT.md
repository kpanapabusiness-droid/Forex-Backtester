# L_ARC_PROTOCOL v2.5 — AMENDMENT

> **Version:** 2.5
> **Predecessor:** v2.3 (Step 5 = WFO; Step 5 stability removed)
> **Note on numbering:** This amendment is numbered v2.5, NOT v2.4, to avoid collision with the existing v2.4 proposal (entry-separability gate, Open-25). Either may land first. If both land, sequence is determined by chat at merge time; both are additive and do not conflict at the mechanical level.
> **Effective:** on merge to `main`
> **Status:** ADDITIVE — does not deprecate or modify v2.3 mechanics for arcs using the trade-classifier track. Coexists with any v2.4 amendment landing for entry-separability.

---

## What this amendment does

Adds a parallel **system-level evaluation track** to the L_ARC protocol. The existing **trade-classifier track** (Steps 1–6 with cluster → capturability → extractability → WFO) is preserved unchanged for arcs that use it. Each new arc declares which track it operates under at arc open. Once declared, the track is locked for that arc.

The system-level track exists because cross-arc evidence over 11 arcs has shown the trade-classifier track's Step 4 AUC gate is filtering against a question (predict individual trade success from entry features) that may not correspond to deployability. KH-24 — the only currently-deployed system — would have died at Step 4 under the trade-classifier track applied to its base signal. The system-level track asks the deployment question directly: does this signal + filter + exposure configuration produce a worst-fold WFO pass on held-out data.

---

## What does NOT change

- v2.3 protocol remains in full force for any arc declared under the trade-classifier track.
- v2.3 closures stand. Open-22, Open-23, Open-24 closed by v2.3 remain closed by v2.3; this amendment does NOT re-close or re-attribute them.
- Any v2.4 amendment (entry-separability gate, Open-25) coexists with v2.5. If v2.4 lands first, v2.5 inherits any v2.4 mechanics where applicable.
- All non-negotiables: no lookahead, no repainting, real execution only, ex-ante population construction, deterministic outputs, config-driven via YAML, Python backtester as source of truth.
- Worst-fold WFO as the deployment gate. ROI ≥ 5% worst-fold, DD ≤ 8%, daily DD ≤ 4%, sign-consistency.
- KH-24 deployment status. KH-24 is locked. Changes to KH-24 require an explicit modification phase opened against it, NOT a system-level re-evaluation.
- The L0 registry as a historical artefact. New arcs are not bound to the registry.
- Permanently eliminated items in `CLAUDE.md`. Nothing on that list returns under either track.
- Spread floor convention (post-2026-05-17 calibration from HistData audit).
- D1 one-bar lag rule (`iClose(D1, 1)` semantics).
- Anchor preservation principle: cross-arc work must reproduce KH-24's worst-fold ROI (1.92%) and DD (6.37%) within tolerance as a methodology check.

---

## What this amendment closes from the open backlog

**Nothing.** This amendment is purely additive infrastructure. It introduces a parallel track but does not resolve any open backlog item by itself. Open items remain open; their resolution is the work product of arcs run under the new track, not the track's existence.

(This is an intentional departure from earlier draft text that claimed closures of Open-22, Open-04, etc. Those claims were incorrect — Open-22/23/24 were already closed by v2.3, and the others are not mechanically resolved by adding a track.)

---

## §A. Track declaration

Every arc-open document must contain a locked field at the top:

```
track: trade-classifier | system-level
declared_by: <author>
declared_at: <ISO timestamp>
amendment_version: v2.5
```

Once an arc commits a step-1 result on its branch, the track is locked. A track switch requires arc closure and a new arc-open under the alternative track.

Existing closed arcs (Arcs 1–11) are eligible for re-opening under the system-level track via the migration rules in §F.

---

## §B. System-level track — step structure

The system-level track replaces the trade-classifier track's step structure (1→6) with the following four-step structure:

### Step 1 — Plumbing and pool construction

Identical to the trade-classifier track Step 1. Build the trade pool using the arc's declared signal definition. All Step 1 gates from v2.3 apply unchanged:

- Pool size ≥ 500 trades
- ≥ 28 pairs
- Smallest per-pair n ≥ 20
- Determinism (byte-identical across re-runs)
- Lookahead spot-check (5 randomly sampled trades, manually verified)
- D1-lag verification (3 NaN-perturbation tests)
- KH-24 co-fire rate < 5% (independence from anchor)
- Cap-binding rate ≤ 20% under §5 auto-extend
- Spread floor applied per post-2026-05-17 calibration

**Pass:** all gates clean.
**Fail:** Step 1 fails → arc closes with `STEP_1_HALT`. Same as trade-classifier track.

### Step 2 — Pool characterisation

Replaces the trade-classifier track's cluster → capturability → extractability sequence with a single descriptive step.

The arc produces a pool characterisation document containing:

- Full distribution of trade outcomes (R, bars_held, MFE, MAE, time-to-peak)
- Per-pair trade-count and outcome distribution (small-multiples table)
- Per-session, per-hour, per-day-of-week trade-count and outcome distribution
- Per-vol-regime, per-D1-state outcome distribution
- Forward-path geometry distribution (mono, peaks, ttp_rel — same metrics as trade-classifier track Step 2, but used descriptively only; no clustering required)
- Conditional outcome distributions across the filter menu in §C (one row per filter, no combinations yet)

**Pass:** the pool has at least one ≥ 5pp delta in mean R between the filtered and unfiltered cohort across at least one filter in the menu. (This is a very weak gate — it asks only "does any filter move the needle at all on this pool". If no filter moves it, the pool itself doesn't carry differential information and the search in Step 3 will not succeed.)
**Fail:** no filter moves mean R by ≥ 5pp → `STEP_2_HALT_NULL_POOL`. Close arc.

Clustering is **optional** under the system-level track. An arc may run clustering as auxiliary analysis if the candidate hypothesis benefits from it, but no gate depends on cluster quality.

### Step 3 — Filter and exposure search (IS only)

The arc declares its filter menu at arc-open as a subset of (or extension to) the canonical menu in §C. The complexity cap is ≤ 3 filters per config. Exposure rules are searchable independently.

Search procedure:

1. Enumerate the full discrete search space (filters × thresholds × exposure rules × SL multipliers).
2. For each config, run 7-fold IS WFO on the IS window (2010-01-01 → 2019-12-31, HistData).
3. Record per-config: worst-fold ROI, mean-fold ROI, worst-fold DD, daily DD breach count, trades per fold, sign-consistency.
4. Rank configs by worst-fold ROI subject to: worst-fold DD ≤ 8%, daily DD ≤ 4%, trades per fold ≥ 5, sign-consistency.
5. Keep top-K configs (K = 5 by default, declarable at arc-open up to K = 10).

**Pass:** at least one config clears the IS gates with worst-fold ROI ≥ 5%.
**Fail:** no config clears IS → `STEP_3_HALT_NO_IS_PASS`. Close arc.

The IS window is searched blind. No human inspection of IS data prior to Step 3 results, no inspection of intermediate IS distributions during the search, no peek at OOS at any point during Step 3.

Selection-bias log is mandatory. The total number of configs evaluated is reported in the arc closure document. This number bounds the multiple-testing exposure for the Step 4 verdict.

### Step 4 — OOS validation (one-shot per config)

For each of the K top-K winners from Step 3:

1. Lock the config (filters, thresholds, exposure rule, SL multiplier). No further tuning.
2. Run 7-fold OOS WFO on the OOS window (2020-01-01 → present, HistData primary).
3. Record same metrics as Step 3.
4. Optionally: re-run on 5ers MT5 data over the OOS window as a data-source ratification (informational, not a gate).

**This is one-shot per config.** Each config gets exactly one OOS evaluation. If a config fails OOS, that config is dead. The arc may evaluate other configs from the top-K list, but no config may be re-tuned and re-validated within the same arc.

**Deployable verdict** per config requires ALL of:

- IS worst-fold ROI ≥ 5% (already verified in Step 3)
- IS worst-fold DD ≤ 8%
- IS daily DD ≤ 4% (zero breaches across IS window)
- IS sign-consistency (no negative folds; or ≤ 1 mildly negative fold if mean-fold ≥ 8% — same convention as v2.3)
- IS trades per fold ≥ 5
- OOS worst-fold ROI ≥ 5%
- OOS worst-fold DD ≤ 8%
- OOS daily DD ≤ 4% (zero breaches across OOS window)
- OOS sign-consistency (same convention)
- OOS trades per fold ≥ 5
- 5ers MT5 ratification: worst-fold ROI within ±3pp of HistData OOS, DD within ±2pp (if ratification is run)

Anything short of all of the above is **NOT_DEPLOYABLE**. Per-config; an arc may produce 0, 1, or more deployable configs from its top-K.

**Arc closure:**

- If ≥ 1 deployable config: arc closes with `DEPLOYABLE` disposition. The deployable config(s) are flagged for chat-side review. Deployment requires explicit chat-side decision; the protocol does not auto-deploy.
- If 0 deployable configs but ≥ 1 OOS near-miss (within 1pp of worst-fold ROI gate, ≤ 1pp over DD gate): arc closes with `STEP_4_HALT` and the near-miss is flagged for protocol calibration review.
- If 0 deployable configs and no near-miss: arc closes with `CLEAN_NULL`.

---

## §C. Canonical filter menu

The system-level track operates against a canonical filter menu. Arcs may use any subset, with the ≤ 3 complexity cap. Extensions require an explicit amendment to this menu, declared at arc-open.

| Filter ID | Description | Grid |
|---|---|---|
| F-session | Session window | {London, NY, Tokyo, London∪NY, NY∪Tokyo, any} |
| F-hod | Hour-of-day | {0-7, 8-15, 16-23, custom 4h windows} |
| F-dow | Day-of-week | {Mon-Fri, Mon-Thu, Tue-Thu, Tue-Fri} |
| F-d1-slope | D1 close − D1 close[N] sign | sign ∈ {+, -, any}, N ∈ {1, 3, 5} |
| F-d1-atr-pct | D1 ATR percentile vs 100-bar trailing | {>50, >70, <30, <50, any} |
| F-vol-regime | 4H ATR / 100-bar trailing 4H ATR ratio | {>1.0, >1.2, <0.8, <1.0, any} |
| F-1h-close-in-range | 1H close position within 1H bar range | T ∈ {0.20, 0.28, 0.35, 0.50}, dir ∈ {top, bottom, any} |
| F-cross-pair-density | Count of same-direction signals in trailing 24 bars across 28 pairs | {<2, <5, <8, ≥3, ≥6, any} |
| F-dollar-bloc | Mean rank of USD pairs over trailing 24 bars | {top tercile, bottom tercile, middle, any} |
| F-currency-strength | Rank of base or quote currency at entry | {rank ≤ 2, rank ≥ 7, diff ≥ 4, any} |

Exposure rule (searched separately, applied as outermost gate):

| Parameter | Grid |
|---|---|
| max_concurrent_per_currency | {1, 2, 3, unlimited} |
| max_concurrent_total | {3, 5, 10, unlimited} |

SL multiplier (searched as part of config):

| Parameter | Grid |
|---|---|
| SL × ATR(14) | {1.5, 2.0, 2.5, 3.0, 3.5, 4.0} |

TP and trailing policies are NOT searched under the canonical menu — they require their own amendment. Default exit policy: stop-loss only, no fixed TP, trade exits when SL is hit or when the signal's natural exit condition triggers (defined per signal at arc-open).

---

## §D. IS / OOS partition convention

```
IS  window:  2010-01-01 → 2019-12-31  (HistData M1 OHLC, harmonised)
OOS window:  2020-01-01 → present     (HistData primary)
Ratification (optional, informational): 5ers MT5 over the OOS window
```

The IS window is searched. The OOS window is one-shot per config. No exceptions.

If HistData 2010-2019 is unavailable for any reason (build failure, data gap), the fallback is:

```
IS  (interim):  2020-01-01 → 2024-12-31, 7-fold WFO
OOS (interim): 2025-01-01 → present
```

Arcs run on the interim partition must be flagged as `INTERIM_PARTITION` in their closure. Such results are not deployment-deciding; they are exploratory. A `DEPLOYABLE` verdict requires the full partition.

---

## §E. Selection-bias and multiple-testing accounting

System-level Step 3 searches discrete config spaces that may contain hundreds or thousands of candidates. The Step 4 one-shot OOS rule is the primary defence against in-sample overfitting, but multiple-testing exposure remains via the K = 5 (or up to 10) top-K promoted to OOS.

The arc closure document must include a selection-bias section reporting:

- Total configs evaluated in Step 3
- Number of configs passing IS gates
- Number of configs promoted to OOS (= top-K)
- Number of configs passing OOS gates
- Bonferroni-corrected significance threshold for the OOS pass rate, given K trials

A pass rate at K trials that is statistically indistinguishable from random is grounds for `STEP_4_HALT_SELECTION_BIAS` rather than `DEPLOYABLE`, even if individual configs pass the per-config criteria. This rule prevents a single noise-driven OOS pass from being treated as deployment evidence.

Concretely: if K = 5 and only 1 of 5 passes OOS, the binomial P(≥1 random pass | per-config null rate p₀) must be small (Bonferroni-style threshold: P ≤ 0.05 after correction for the implicit null rate). If P > 0.05 after correction, the pass is logged as `OOS_PASS_NOT_BIAS_DEFENSIBLE` and the arc closes with `STEP_4_HALT`.

---

## §F. Migration of closed arcs

Arcs 1–11 closed under v2.0–v2.3 (trade-classifier track) are eligible for re-evaluation under the system-level track. The procedure:

1. Open a re-evaluation dispatch (not a new arc) targeting one or more closed arcs.
2. Reuse the original arc's Step 1 trade pool. **Do not rebuild.** Extend backward to 2010 using the same signal definition for the IS window.
3. Run system-level Steps 2 → 3 → 4 as defined above.
4. Closure of the re-evaluation produces:
   - For each closed arc: a verdict (`DEPLOYABLE` / `STEP_4_HALT` / `CLEAN_NULL`) under the system-level track.
   - The original closure status (trade-classifier track) is preserved unchanged. The system-level verdict is added as a parallel record.

A `DEPLOYABLE` verdict from a re-evaluation does NOT automatically reopen the arc as live. It produces a deployable config that is then flagged for chat-side deployment decision.

The system-level re-evaluation of closed arcs is the canonical first use of v2.5. See `CC_SYSTEM_LEVEL_TRACK_REEVAL.md` for the dispatch.

---

## §G. Anchor preservation

Under either track, any cross-arc evaluation framework must reproduce KH-24's deployed worst-fold ROI (1.92%) and worst-fold DD (6.37%) within tolerance (±0.5pp on ROI, ±1pp on DD) when KH-24 is run through the framework. If reproduction fails, the framework has a methodology bug and any results from that framework are suspect.

KH-24 itself is locked. Re-evaluation of KH-24 base under the system-level track is permitted for benchmarking purposes only — it does NOT authorise modifications to the deployed KH-24 configuration. Changes to the deployed KH-24 require a dedicated modification phase opened against KH-24, separate from any L-arc work.

---

## §H. Discipline rules (track-agnostic)

- Track declaration is binding. An arc may not switch tracks mid-flight.
- Locked thresholds within an arc do not move. Calibration adjustments are cross-arc.
- The selection-bias log is mandatory and material.
- The one-shot OOS rule binds at the config level, not the arc level. An arc may evaluate multiple configs on OOS as long as each is locked before its single OOS evaluation.
- Cross-arc findings raised by a system-level arc are logged to `PROTOCOL_IMPROVEMENT_BACKLOG.md`. Resolution is cross-arc, never mid-arc.
- The system-level track does NOT lower the deployment bar. All gates from v2.3 (worst-fold ROI, DD, daily DD, sign-consistency) apply equally. What changes is the path to those gates, not their stringency.

---

## §I. What this amendment does NOT permit

- Lowering the worst-fold ROI gate (≥ 5%) or DD gate (≤ 8%).
- Lowering the daily DD breach threshold (no breaches above 4% permitted).
- Skipping the OOS validation step.
- Iterating on OOS after seeing OOS results.
- Lookahead in any feature, anywhere.
- Outcome-aware filters in population construction.
- Modifying KH-24 deployment without an explicit modification phase.
- Treating clustering as required for the system-level track.
- Treating an interim-partition pass as deployment-deciding.
- Re-using an OOS block for a re-tuned config.
- Closing existing open backlog items by reference to v2.5 mechanics. v2.5 is additive; closures of existing items require their own work.

---

## §J. Cross-reference

This amendment is consumed by the following parallel dispatches:

- `CC_SYSTEM_LEVEL_TRACK_REEVAL.md` — re-evaluation of closed arcs 6, 7, 8, 10, 11 under the system-level track
- `CC_FREE_REIGN_STRATEGY_DISCOVERY.md` — free-reign discovery operates under system-level track conventions for any candidate it promotes to OOS validation
- `CC_CURRENCY_STRENGTH_DESCRIPTIVE.md` — descriptive phase; gates an eventual system-level arc on currency strength
- `CC_TX_CONTAMINATION_TEST.md` — diagnostic probe informing Pipeline D (trade-classifier track feature design); independent of v2.5 mechanics

The infrastructure dispatch `CC_PARALLEL_COORDINATION.md` lands this amendment on `main` before the substantive dispatches launch.

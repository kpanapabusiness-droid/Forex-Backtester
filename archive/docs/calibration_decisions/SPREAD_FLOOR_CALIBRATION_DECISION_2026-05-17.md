# SPREAD_FLOOR_CALIBRATION_DECISION — 2026-05-17

> Cross-arc calibration decision document under `L_ARC_PROTOCOL.md` §12 ("Cross-arc governance").
> Required before any change to `configs/spread_floors_5ers.yaml`.
> Status: **APPROVED 2026-05-17 by Keanu (in-chat)**.
> Predecessor: `docs/SPREAD_FLOOR_AUDIT_FINDING.md` (2026-05-17).

---

## §1. Decision

Replace the uniform 0.1 pip spread floor in `configs/spread_floors_5ers.yaml` with **per-pair p50 values from the HistData 2024-2025 audit**, taken from the first-5-minute spread distribution at 1H execution bars, active-session pooled (London / NY / overlap windows). The file becomes calibration-curated; the generator script [scripts/lchar/compute_spread_floors.py](../../scripts/lchar/compute_spread_floors.py) is retired.

This decision OVERRIDES the prior recommendation in `docs/SPREAD_FLOOR_AUDIT_FINDING.md` "Required resolution §1" (which specified p10). The override is intentional; reasoning in §3–§4 below. The historical p10 recommendation is preserved in the audit doc as a forward-pointer; that wording is not edited in place.

---

## §2. Context

`configs/spread_floors_5ers.yaml` is the per-pair spread-floor table loaded by every L arc backtest config. Its semantic, defined in [core/spread_floor.py](../../core/spread_floor.py): when the per-bar raw spread observation is zero, the floor value is substituted. When raw is non-zero, raw passes through unchanged.

The existing file, generated 2026-XX-XX by [scripts/lchar/compute_spread_floors.py](../../scripts/lchar/compute_spread_floors.py), encoded `min_nonzero_spread_native: 1` for all 28 pairs — that is, 1 raw MT5 point per pair, which the engine converts to 0.1 pip uniformly via `pips = native / 10`. The Arc 4 closure surfaced (via independent HistData validation, recorded in `SPREAD_FLOOR_AUDIT_FINDING.md`) that this uniform 0.1 pip floor under-models real spreads by 3-48× per pair, with the worst gap on GBP/NZD (real p50 ≈ 4.8 pips, gap = 4.7).

The audit doc's "Required resolution §1" specified p10 of the first-5-minute distribution as the replacement value, on the reasoning that "the floor is a minimum, not a typical" and p10 represents the cheapest 10% of execution bars. STATUS.md backlog mirrored that recommendation.

This calibration revisits that p10 choice and lands on p50 instead. Rationale in §3–§4.

---

## §3. Why p10 is wrong-direction conservatism

The floor's mechanical role is **to synthesise a value when raw is zero**, not to bound the cheapest possible spread. The two are different:

- "Bound the cheapest possible" → floor as a *minimum allowed cost* → p10 fits.
- "Synthesise a missing value" → floor as a *best estimator under uncertainty* → an unbiased prior fits; p50 (median) is the unbiased prior, p10 (low-tail) is system-favouring.

When raw = 0 (broker quantization on a slow bar), we have no observation. The right substitute is an estimator that minimises expected error in either direction. The mean or median of the empirical distribution fits that role; the p10 systematically picks the lowest-cost tail.

If the floor were applied as a *clamp on the raw observation* (`effective = max(raw, floor)`), p10 would be sensible — it would catch implausibly-tight raw values without affecting typical ones. But the engine's semantic is *substitute-when-raw-equals-zero*, not clamp. Under substitution semantics, p10 systematically under-prices the synthesised cost.

The audit doc's framing — "the floor is a minimum, not a typical" — is correct as a definition of what a floor *is*. The framing is wrong for what this particular floor *does* in this engine. The semantic gap was not surfaced when the audit doc was authored; it surfaces now.

---

## §4. LP-vs-retail bias compounds the problem

The HistData feed publishes liquidity-provider (LP) quotes. 5ers retail fills are tighter than LP only on tightest-tier majors; on crosses and minors, 5ers retail is typically 10-30% wider than the LP feed (broker markup). This is a known structural delta, currently unmeasured for this specific broker (Arc 4 open follow-up: one-week MT5 snapshot from Contabo VPS).

A p10 choice stacks two layers of optimistic bias:
1. **Within-distribution bias.** p10 picks the bottom 10% of the available samples — a tail, not a centre.
2. **LP-vs-retail bias.** HistData LP is tighter than 5ers retail by ~10-30%. The retail-equivalent of an LP-p10 is roughly LP-p5 or LP-p3 — even deeper in the tail.

The product: HistData-p10 maps to roughly 5ers-retail-p5. The floor would substitute a value below the empirical 5th percentile of what the broker actually quotes. Every floor application would under-price spread.

p50 removes the within-distribution bias layer. The LP-vs-retail layer remains and is acknowledged: the floor is still slightly optimistic relative to true 5ers retail, but only by the 10-30% feed-vs-broker delta — one layer of bias rather than two. The remaining layer will be measured directly (Arc 4 open follow-up: MT5 snapshot) and applied as a multiplier in a subsequent recalibration.

p50 is the floor below which we will not go without explicit empirical justification.

---

## §5. Alternatives considered

Per-pair percentile distribution from the HistData 2024-2025 active-session pool. Representative pairs (active sessions: London, NY, overlap):

| Pair | p10 (pips) | p25 | p50 | p75 | p90 |
|---|---|---|---|---|---|
| EUR_USD | 0.2 | 0.2 | 0.3 | 0.5 | 0.7 |
| GBP_USD | 0.6 | 0.7 | 0.8 | 0.9 | 1.2 |
| USD_JPY | 0.3 | 0.5 | 0.6 | 0.7 | 1.0 |
| EUR_JPY | 0.7 | 0.8 | 1.0 | 1.3 | 1.7 |
| GBP_CAD | 2.0 | 2.2 | 2.8 | 3.3 | 4.2 |
| GBP_NZD | 3.6 | 4.2 | 4.5 | 6.0 | 7.3 |

Sources: `results/spread_validation/per_pair_distributions.csv` (p10, p25, p75 — active-session pooled) and `results/spread_validation/execution_bar_spreads.csv` (p50, p95 — first-5-minute at 1H execution bar).

Aggregate lift ratios across the 28 pairs:
- p50 / p10 ≈ 1.25× (median per-pair lift)
- p75 / p50 ≈ 1.11×
- p90 / p50 ≈ 1.23×

p50 sits 25% above p10 in expectation and 11–23% below p75–p90.

---

## §6. Choice rationale

**p50 over p10:** removes the within-distribution bias layer that compounds with the LP-vs-retail bias (§4). Under substitution semantics (§3), the median is the right estimator.

**p50 over p75 / p90:** under uncertainty, an unbiased prior is preferable to a one-sided conservative prior. Conservative cost stress-testing is a separate methodology (run the same WFO with floor = p75 or p90 and compare deltas) — not a substitute for the calibrated central value. Bias direction matters more than bias magnitude here; p50 has the right direction.

**p50 over LP-multiplier-corrected p10:** the LP-vs-retail multiplier is not yet measured. Once it is (Arc 4 open follow-up: one-week MT5 snapshot from Contabo VPS), a subsequent recalibration can apply it as a deliberate uniform multiplier on the p50 base. Until the multiplier is measured, p50 is the most honest central value we can defend.

**p50 over alternatives that change methodology entirely** (e.g. min of broker quotes, smoothed over rolling N-bar window): out of scope for this recalibration. Methodology change requires a new audit; this is a value-only refresh within the existing methodology shape (per-pair scalar floor).

---

## §7. Scope and impact

**Affected (re-evaluation under new floor recommended at next visit):**
- Arc 4 closure verdict (CLEAN-NULL on transaction-cost truth) — directionally stands but the magnitude of the F6 sign flip softens under p50 vs uniform-0.1; cluster 1 D1 AUC 0.667 is unaffected; a re-run under p50 produces a finer-grained verdict.
- Arc 5 (queued) — runs first under p50.
- Arcs 6+ open under p50 by default.
- Open-18 cross-replays (previously blocked) — unblock under p50.

**Unaffected:**
- KH-24 live deployment. KH-24 does not load this file (uses raw 5ers MT5 per-bar spread directly via `configs/wfo_kh24.yaml`). EA v2.01 on Contabo VPS unchanged.
- KH-24 WFO claim downgrade (pass-deployable → pass-viable). The downgrade stands on the independent audit reconciliation, not on this floor change.
- All §10 PASS-DEPLOYABLE / PASS-VIABLE / CLEAN-NULL threshold definitions in L_ARC_PROTOCOL.md.

---

## §8. Open follow-ups

Carried to STATUS.md active backlog:

- **LP-to-retail multiplier calibration (MEDIUM, Arc 4).** One-week MT5 bid/ask snapshot from live VPS to derive HistData-LP to 5ers-retail multiplier. Refines floor values upward by estimated 10-30%. Tracked separately from session-aware floor item.
- **Phase Zero spread audit (HIGH, Arc 4).** Add to L arc workflow before Step 1 plumbing; refresh tick audit every 6 months.
- **Session-aware floors per-pair × per-session (MEDIUM, Arc 4).** Off-hours / weekend-edge spreads can be 2–3× active-session; per-session calibration would tighten the model further. Defer until per-pair p50 is in place and next arc's behaviour observed.
- **Governance doc consolidation (LOW, Arc 4 closure).** `docs/SPREAD_SEMANTICS_LOCK.md` and `docs/L6_0_METHODOLOGY_LOCK.md` overlap on spread floor governance. Resolve before next calibration cycle.

---

## §9. Approval

| Item | Value |
|---|---|
| Decision date | 2026-05-17 |
| Source arcs | Arc 4 closure + Arc 5 spread validation reconciliation |
| Source data | `results/spread_validation/per_pair_distributions.csv`, `results/spread_validation/execution_bar_spreads.csv` |
| Window | 2024-01-01 → 2025-12-31 |
| Pooling | Active sessions (London, NY, overlap); mean of per-session percentiles for §5 impact table; canonical floor values from `execution_bar_spreads.csv` `p50_spread_pips_first5min` |
| Percentile | p50 |
| Replaces | Uniform 0.1 pip across all 28 pairs |
| Overrides | `SPREAD_FLOOR_AUDIT_FINDING.md` "Required resolution §1" (p10 recommendation) |
| Author | Chat-side analyst, 2026-05-17 |
| Approval | Approved 2026-05-17 by Keanu (in-chat) |
| KH-24 anchor impact | None — KH-24 does not load this file |
| Governance | `L_ARC_PROTOCOL.md` §12 "Cross-arc governance" + new "Spread floor calibration" subsection |

---

## §10. The 28 per-pair p50 floor values (final, in pips)

Listed alphabetically. Native-points encoding lives in `configs/spread_floors_5ers.yaml`; the pip values here are the authoritative calibration record.

| Pair | p50 (pips) | Pair | p50 (pips) | Pair | p50 (pips) | Pair | p50 (pips) |
|---|---|---|---|---|---|---|---|
| AUD_CAD | 1.9   | CHF_JPY | 2.2   | EUR_USD | 0.3   | GBP_USD | 0.8 |
| AUD_CHF | 1.4   | EUR_AUD | 2.4   | GBP_AUD | 4.0   | NZD_CAD | 2.3 |
| AUD_JPY | 0.8   | EUR_CAD | 1.8   | GBP_CAD | 2.8   | NZD_CHF | 1.7 |
| AUD_NZD | 2.05  | EUR_CHF | 1.0   | GBP_CHF | 1.85  | NZD_JPY | 1.0 |
| AUD_USD | 1.0   | EUR_GBP | 0.8   | GBP_JPY | 2.0   | NZD_USD | 1.1 |
| CAD_CHF | 1.5   | EUR_JPY | 1.0   | GBP_NZD | 4.8   | USD_CAD | 1.3 |
| CAD_JPY | 1.2   | EUR_NZD | 3.8   |         |       | USD_CHF | 0.9 |
|         |       |         |       |         |       | USD_JPY | 0.6 |

Encoding contract (per `docs/SPREAD_SEMANTICS_LOCK.md` "Floor file encoding"): `pips = native_points / 10` uniform across all 28 pairs including JPY. Non-integer pip values (1.85, 2.05) round up to the next integer when encoded as native points (GBP_CHF → 19, AUD_NZD → 21). Round-up is the conservative choice consistent with this section's calibration logic.

---

## §11. Document control

| Field | Value |
|---|---|
| Date | 2026-05-17 |
| Author | Claude Code (under chat-level analyst direction) |
| Predecessor finding | `docs/SPREAD_FLOOR_AUDIT_FINDING.md` (2026-05-17, RESOLVED section dated same) |
| Governance | `L_ARC_PROTOCOL.md` §12 "Cross-arc governance" + new "Spread floor calibration" subsection |
| Storage convention | `docs/calibration_decisions/<TOPIC>_<YYYY-MM-DD>.md` — establishes dated-decision precedent |
| Status | APPROVED 2026-05-17 by Keanu (in-chat) |

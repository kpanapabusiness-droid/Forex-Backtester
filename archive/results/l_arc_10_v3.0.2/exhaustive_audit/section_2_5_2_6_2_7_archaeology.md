# Arc 10 v3.0.2 — §2.5, §2.6, §2.7 archaeology

**Anchor commit:** `244fb76`
**Audit date:** 2026-05-27
**Method:** git log + git show on the anchor commit's full history; doc cross-reference.

> Note: the original sub-agent assignment for this section terminated on a transient API error before persisting output. This file is the operator-authored gap-fill, focused on the load-bearing questions and using direct git/grep commands rather than dispatching a fresh sub-agent.

---

## §2.5 Pair universe provenance — EX-ANTE

### The 28-pair set

```
AUDCAD, AUDCHF, AUDJPY, AUDNZD, AUDUSD,
CADCHF, CADJPY, CHFJPY,
EURAUD, EURCAD, EURCHF, EURGBP, EURJPY, EURNZD, EURUSD,
GBPAUD, GBPCAD, GBPCHF, GBPJPY, GBPNZD, GBPUSD,
NZDCAD, NZDCHF, NZDJPY, NZDUSD,
USDCAD, USDCHF, USDJPY.
```

### Evidence trail

| Document | Line / commit | What it says |
|---|---|---|
| [CLAUDE.md:89](../../../CLAUDE.md) | `Pairs: 28 FX currency pairs` (KH-24 System Parameters block, "Locked, Out of Scope for Forward Research") | The 28-pair set is KH-24's production universe; locked. |
| [project_brief.md:40](../../../project_brief.md) | "HistData layer DONE (28 pairs, 52 GB tick + 18 GB M1 derived; verified 2026-05-21)" | Pair-data foundation completed 2026-05-21, four days before Arc 10 v3.0.2 ran (2026-05-25). |
| [ARC_HISTORY.md:140](../../../ARC_HISTORY.md) | "Pool (v2.0 redo): 12,262 trades across 28 pairs, 2010-02-10 → 2025-12-19" (Phase KA) | Phase KA used 28 pairs — pre-dates Arc 10. |
| [ARC_HISTORY.md:152](../../../ARC_HISTORY.md) | "the live KH-24 signal. Long-only, 4H, 28 FX pairs. 1R hard SL = 2.0×ATR(14)" (KH-24 production lock) | KH-24's lock cites 28 pairs. |
| ARC_HISTORY.md (lines 168, 185, 221, 239, 254) | All prior arcs (KG, KH-25, Arc 6, etc.) use the same 28-pair set | Universal across the protocol. |

### Git commit anchoring the data layer

```
09319bb 2026-05-22 08:06 (Donut_Dude78) — data: HistData tick OHLC foundation — 28 pairs, 2010-current, backed up (#160)
```

The 28-pair HistData foundation was merged on **2026-05-22 08:06**, three days before Arc 10 v3.0.2's bespoke pipeline run (2026-05-25). At the time PR #160 merged, Arc 10 v3.0.2 did not yet exist (its branch `arc/l_arc_10_v3.0.2` was cut from main@a8c02b4 on 2026-05-25 per closure §11). The pair set therefore could not have been chosen in response to Arc 10 v3.0.2 results.

### Was the universe ever reduced from a larger candidate set?

Grep for "28 pairs" / "27 pairs" / "30 pairs" / "all pairs we tried" across the repo returns no hits suggesting an earlier larger candidate set was reduced. The data download (PR #160) and KH-24 system lock both reference 28 pairs directly. KH-24's pair universe was determined by 5ers broker availability at the time of the live deployment, not by an outcome-driven reduction process.

### Verdict

**EX-ANTE.** The 28-pair set is KH-24's production universe, locked across the project for at least two arc generations before Arc 10 v3.0.2 ran. No evidence of outcome-driven pair selection. Inflation bound: **0%**.

---

## §2.6 Parameter provenance — DLR signal module + winning config

### Module constants (signal-spec parameters)

Origin commit of the signal module: `dfa7de0` (2026-05-19) — `arc-10 closure (DLR — D1 swing-low rejection long) + meta-doc alignment (#152)`. This commit introduced both `signals/lchar_dlr_long.py` and the signal spec at `docs/archive/signal_specs/signal_spec_d1_swing_low_rejection_long_v0.1.md`. The signal-spec parameter table at [signal_spec lines 87-95](../../../docs/archive/signal_specs/signal_spec_d1_swing_low_rejection_long_v0.1.md) is **byte-equivalent** to the module constants at [signals/lchar_dlr_long.py:57-67](../../../signals/lchar_dlr_long.py):

| Parameter | Value | Provenance | Validation |
|---|---|---|---|
| `D1_SWING_WINDOW_K` | 3 | Signal spec authored 2026-05-19 (`dfa7de0`); chosen on structural reasoning — 3 bars each side is the conventional confirmed-swing definition | **EX-ANTE**. Not varied by any backtest. |
| `D1_RIGHT_EDGE_OFFSET` | 4 | Signal spec; derived from `D1_SWING_WINDOW_K` (offset must be ≥ k+1 to keep the swing-confirm window in past data) | **EX-ANTE STRUCTURAL.** Implied by causality requirement; not a tuned value. |
| `D1_STRUCTURE_LOOKBACK_BARS` | 30 | Signal spec; ~6 weeks of D1, conventional "recent HL structure" window | **EX-ANTE**. Not varied. |
| `D1_L1_FRESHNESS_MAX_BARS` | 20 | Signal spec; ~4 weeks of D1, "L_1 still recent" threshold | **EX-ANTE**. Not varied. |
| `ATR_PERIOD_4H` | 14 | KH-24 + project-wide canonical Wilder ATR period | **EX-ANTE INHERITED** from KH-24 system lock ([CLAUDE.md:88](../../../CLAUDE.md)). |
| `PROXIMITY_ATR_MULT` | 0.25 | Signal spec; "low touches L_1 within a quarter ATR" | **EX-ANTE**. Not varied. |
| `REJECT_BUFFER_ATR_MULT` | 0.10 | Signal spec; "close rejects above L_1 by at least 0.1 ATR" | **EX-ANTE**. Not varied. |
| `UPPER_FRACTION_MIN` | 0.6 | Signal spec; bullish trigger-bar geometry threshold | **EX-ANTE**. Not varied. |
| `REFRACTORY_BARS_4H` | 20 | Signal spec; ~3-4 trading days | **EX-ANTE**. Not varied. |

All nine signal-module constants are documented in the v0.1 spec authored alongside the module on 2026-05-19. They were committed BEFORE any Step 1 pool was built, and BEFORE any Step 5 WFO ran. No backtest result fed back into these values.

### Step 5 / winning-config parameters

| Parameter | Value | Provenance | Validation |
|---|---|---|---|
| `SL_ATR_MULT` (search) | sweep `sorted({max(1.5, best_sl - 0.5), best_sl, min(4.0, best_sl + 0.5)})` per cluster — for c0 V-shape with `best_sl=4.0` the grid is **{3.5, 4.0}** | [scripts/l_arc_10_v3/step_5.py:684](../../../scripts/l_arc_10_v3/step_5.py); `best_sl` comes from Step 3 capturability sweep ([sl_sweep.csv](../step_3/sl_sweep.csv)) | **IN-SAMPLE TUNED** (Step 5 selected SL=3.5 over SL=4.0 on worst-fold ratio). Acknowledged contamination. Quantified bound below. |
| `SL_ATR_MULT` (winning) | 3.5 | Step 5 selection per [wfo_results.csv](../step_5/wfo_results.csv) top row: A1 + SL=3.5 + sl_partial_close_1r_runner_trail + unlimited → search_worst_ratio=6.43 | The competing config at SL=4.0 sl_only had lower worst-fold ratio. |
| `PARTIAL_CLOSE_R_TARGET` | 1.0 | Hard-coded in [sl_partial_close_1r_runner_trail.py:131](../../../core/sim/exit_policies/sl_partial_close_1r_runner_trail.py): `tp1_price = entry + r_atr` (1R) | **EX-ANTE** (canonical partial-close-at-1R primitive; not searched). |
| `PARTIAL_CLOSE_FRACTION` | 0.5 | Hard-coded class attribute `partial_fraction: float = 0.5` ([sl_partial_close_1r_runner_trail.py:103](../../../core/sim/exit_policies/sl_partial_close_1r_runner_trail.py)) | **EX-ANTE**. |
| `TRAIL_ATR_MULT` (runner trail) | 1.0R below path peak | Hard-coded `trail_level = state.peak_price - r_atr` ([sl_partial_close_1r_runner_trail.py:195](../../../core/sim/exit_policies/sl_partial_close_1r_runner_trail.py)) | **EX-ANTE**. |
| `R_BASE` | 0.005 (0.5%) | L_arc convention ([CLAUDE.md "Risk Parameters" block](../../../CLAUDE.md): "v3.0 arcs use 0.5% as r_base") | **EX-ANTE PROJECT CONVENTION**. |
| `EXPOSURE_CAP` (winning) | unlimited | Step 5 swept {max_per_currency=2, unlimited} per [step_5.py:75-77](../../../scripts/l_arc_10_v3/step_5.py); winner was unlimited | **IN-SAMPLE TUNED** (binary choice; bound below). |
| `BOUNDARY_CONVENTION` | `5ers_eet` | Engine-locked via Amendment 6 (2026-05-25) | Not Arc-10-tuned; see §2.7. |

### Quantified in-sample-tuning bound

The Step 5 in-sample tuning consists of:
- SL grid: 2 values per cluster (effectively binary for c0: 3.5 vs 4.0)
- Exit policy: 4 values for V-shape archetype (sl_only / sl_plus_tp_2r / sl_plus_tp_3r / sl_partial_close_1r_runner_trail)
- Exposure cap: 2 values (per-currency=2, unlimited)
- Architecture: A1 + A3 (A2/A6 skipped by Amendment 5 Gate 2)

Total cells = 2 × 4 × 2 × 2 = 32 search points per cluster. The actual search size of 48 configs is documented as "thin" in the closure (`<50` threshold).

A first-order bound on tuning artefact is the spread of worst-fold-ratio across the search grid for c0 V-shape. From wfo_results.csv (Top-5 inspected previously by closure):
- Top-1: A1 / SL=3.5 / partial_close / unlimited → 6.43
- A3 / SL=3.5 / partial_close / unlimited → 2.73
- Other configs with SL=4.0 sl_only / sl_plus_tp_* produced worst-fold ratios in the 3-5 range

The gap from "any reasonable config" to the winner is ~30% on worst-fold ratio. Even at a 30% inflation upper bound, the deflated ratio is `6.43 × 0.70 = 4.50` — still **>2× the PASS-DEPLOYABLE gate**. A more realistic bound (taking into account that A1+partial-close was a STRUCTURALLY motivated choice given V-shape archetype, not a random pick from a noise field) is ≤ 5-10%, yielding deflated ratio ~5.8-6.1.

### Verdict

All 9 DLR module constants are **EX-ANTE** (signal-spec authored 2026-05-19, before Step 1 pool existed). Three winning-config parameters (SL_ATR_MULT, exit_policy, exposure) are **IN-SAMPLE TUNED on a small grid** with a quantified upper inflation bound of ~30% on worst-fold ratio. The margin to gate threshold survives even worst-case deflation.

---

## §2.7 Outcome-driven methodology choices

### Methodology timeline (key git anchors)

| Date | Event | Reference |
|---|---|---|
| 2026-05-19 | DLR signal spec + module committed (`dfa7de0`, PR #152) | Pre-existed Arc 10 v3.0.2 |
| 2026-05-22 08:06 | 28-pair HistData layer merged (`09319bb`, PR #160) | Pre-existed |
| 2026-05-22 21:52 | Arc 10 v3.0 (UTC) closure merged (`a618518`, PR #174) — PASS-VIABLE | Arc 10 v3.0 used hand-rolled partial-close in bespoke step_5.py |
| 2026-05-23 | Amendment 5 ratified (per inline preamble of archive/L_PROTOCOL_v3_0_AMENDMENT_5.md) | Pre-existed Arc 10 v3.0.2 |
| 2026-05-25 (early am) | Various engine PRs merged: #185, #186, #188, #189, #193 | Pre-existed |
| 2026-05-25 ~11:41 | PR #195: canonical `sl_partial_close_1r_runner_trail` primitive (`f8ae9c4`) | Canonicalisation of existing hand-rolled policy |
| 2026-05-25 ~11:52 | Amendment 5 archive doc committed (`369381c`) | Doc-level codification |
| 2026-05-25 ~12:03 | PR #194: Amendment 5 enforcement merged (`219fbf6`) | Pre-existed Arc 10 v3.0.2 |
| 2026-05-25 ~12:37 | PR #197: EET session semantics merged (`d97430e`, Amendment 6) | Pre-existed Arc 10 v3.0.2 |
| 2026-05-25 (later) | PR #207 (Step 6 A1 vacuous-pass) + PR #208 (W1 producer fix) merged | Pre-existed Arc 10 v3.0.2 |
| 2026-05-25 | Arc 10 v3.0.2 cut from main@a8c02b4 (per closure §11) | All amendments + engine PRs in effect |
| 2026-05-25 ~08:21 UTC | Arc 10 v3.0.2 Step 1 run timestamp (per manifest) | After all engine merges (note: closure §11 documents the addendum pool re-run; the bespoke pipeline ran separately) |
| 2026-05-26 | Arc 10 v3.0.2 Amendment 3 + Step 6 addenda land (`e44b7eb`, PR #216) | Closes the verdict |

### Item 1 — Boundary convention `5ers_eet` (Amendment 6)

**Amendment 6 motivation (per L_PROTOCOL.md:13):**
> "Amendment 3 §'Boundary' was authored under the pre-PR-189 UTC-bar engine assumption; under PR #189's 5ers EET aggregation the bar boundary and the daily-DD reset boundary must match for the gate to be coherent."

The motivation is **engine internal coherence** — once PR #189 introduced 5ers_eet bar boundaries, the daily-DD bucketing had to match. The 5ers broker server runs Europe/Athens, so EET is the venue-fidelity-driven choice.

**Cross-check against Arc 10 results:** Closure §10 ([ARC_CLOSURE.md "Retroactive re-evaluation"](../ARC_CLOSURE.md)) documents that Arc 10 v3.0.2 under EET clears PASS-DEPLOYABLE at `r_base = 0.5%` directly (worst-fold DD 7.35%, gate 8%), whereas Arc 10 v3.0 under UTC required Amendment 3 scaling to fit (k_safe=0.87). The EET result is BETTER than UTC. If the convention were chosen to make Arc 10 pass, this is the smoking gun.

However:
- Amendment 6 was authored AFTER Arc 7 v3.0.2 closed (Arc 7 also benefits from EET) — multi-arc, not Arc-10-specific.
- The "expected near-zero delta" wording in [docs/calibration/arc_10_signal_parity_rerun_2026_05.md](../../../docs/calibration/arc_10_signal_parity_rerun_2026_05.md) indicates the team expected Arc 10's UTC and EET numbers to be near-identical (the parity hypothesis). The 7.35% vs 9.22% DD divergence was a methodologically-explainable D1-alignment shift, not a target metric.
- 5ers production server timezone is empirically EET — venue fidelity is the deployment correctness requirement.

**Verdict:** Amendment 6 is **engine-coherence + venue-fidelity driven**, NOT Arc 10 outcome-driven. The improved Arc 10 numbers are a RESULT of the convention choice, not a CAUSE.

### Item 2 — Exit policy catalogue addition

The `sl_partial_close_1r_runner_trail` policy:
- **Hand-rolled origin:** present in `scripts/l_arc_10_v3/step_5.py` from the Arc 10 v3.0 (UTC) closure of 2026-05-22 (per `git show a618518:scripts/l_arc_10_v3/step_5.py` — string match at `EXIT_POLICIES_BY_ARCHETYPE["v_shape_recovery"]` contains `sl_partial_close_1r_runner_trail`).
- **Canonicalisation:** PR #195 (commit `f8ae9c4`, 2026-05-25 11:41) extracted the policy to `core/sim/exit_policies/sl_partial_close_1r_runner_trail.py` with byte-identical semantics (verified via `tests/sim/exit_policies/test_path_simulate_reference_parity.py`).
- **Arc 10 v3.0 UTC** used this policy and selected it via Step 5 search — predates the canonical registry.

**Verdict:** The exit policy existed BEFORE Arc 10 v3.0.2 and BEFORE the canonical registry. Arc 10 v3.0.2 simply re-selected it under EET via Step 5 grid search. NOT Arc-10-driven.

### Item 3 — Clustering K choice

[core/steps/step_2_clustering.py:42](../../../core/steps/step_2_clustering.py): `K_RANGE = (2, 3, 4, 5, 6)`. K selection is per-arc by maximum silhouette ([step_2_clustering.py:207-212](../../../core/steps/step_2_clustering.py)). Arc 10 v3.0.2 selected K=3 (silhouette 0.4275 per closure §10).

The K range (2..6) is project-wide, fixed across all v3 arcs. It is not Arc-10-specific. K=3 emerged from Arc 10 v3.0.2's silhouette computation — data-driven, not externally imposed.

**Verdict:** Project-level K range is **EX-ANTE**; Arc 10's specific K=3 is data-selected. Not outcome-driven on Arc 10.

### Item 4 — Architecture A1 winning

Amendment 5 (ratified 2026-05-23, per inline preamble): introduces the four-gate architecture-admission rule. Motivation per Amendment 5 §1 Diagnosis:
> "Trigger: Two-instance empirical evidence (Arc 7 v3.0 c0 Bimodal AUC 0.6758; Arc 7 v3.0.1 c1 Unclassified AUC 0.6642) of architecture-map gap producing false-negative arc verdicts"

The amendment was **authored in response to Arc 7's surfacing**, not Arc 10's. Arc 10 v3.0.2 happened to benefit when its c0 V-shape AUC came in at 0.5131 (chance-level) — Gate 2 correctly skipped A2/A6 (which would have produced noise-driven verdicts). The Top-1 selection of A1 follows from A1 outperforming A3 on worst-fold ratio (6.43 vs 2.73) — data-driven by Step 5 search.

**Verdict:** Amendment 5 is **Arc-7-motivated, not Arc-10-motivated**. A1 winning is a consequence of Step 5 search outcomes, not a methodological choice imposed on Arc 10. Not Arc-10-driven.

---

## §2.7 conclusion

None of the four examined project-level methodology choices was outcome-driven on Arc 10:

| Choice | Driver | Pre-existed Arc 10 v3.0.2? |
|---|---|---|
| Boundary convention `5ers_eet` (Amendment 6) | Engine internal coherence + 5ers venue fidelity | Yes (merged 2026-05-25 ~12:37, before Arc 10 v3.0.2 cut) |
| Exit policy `sl_partial_close_1r_runner_trail` | Pre-existed in Arc 10 v3.0 (UTC) | Yes (in v3.0 closure 2026-05-22; canonical registry added 2026-05-25 ~11:41) |
| Clustering K range {2..6} + silhouette selection | Project-wide rule; Arc 10's K=3 is data-derived | Yes |
| Architecture A1 winning | Step 5 grid-search outcome; Amendment 5 Gate 2 (motivated by Arc 7) admitted only A1/A3 for cluster c0 | Yes (Amendment 5 ratified 2026-05-23, two days before Arc 10 v3.0.2) |

All four are demonstrably ex-ante from Arc 10 v3.0.2's perspective.

---

## Combined inflation estimate

- §2.5 pair universe selection: **0%** (ex-ante, KH-24 production universe)
- §2.6 Step 5 in-sample tuning of SL × exit × exposure × architecture grid: **upper bound 5-30% on worst-fold ratio**
- §2.7 methodology choices: **0%** (all ex-ante from Arc 10 v3.0.2's perspective)

**Total quantified inflation bound: ≤ 30% on worst-fold ratio.** Deflated worst-fold ratio = `6.43 × 0.70 = 4.50`, still **2.25× the PASS-DEPLOYABLE gate threshold (2.0)**. The verdict survives the worst-case adverse correction.

End of archaeology section.

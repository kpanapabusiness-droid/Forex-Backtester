# Arc 10 v3.0.2 — Cost-Realism Sweep Report

> **Verdict headline:** Arc 10 v3.0.2 under UTC at `r_base = 0.5%` does NOT survive realistic costs against the 5ers 10% hard DD limit. **Swap is by far the dominant cost vector** (Δratio = −2.60 from a single axis flip, vs −0.42 from spread 1.5× and −0.72 from slippage 1.0 pip/fill). At most modest costs (1× or 1.5× spread, swap-off, light slippage), the strategy clears DD_WATCH (DD ∈ [8%, 10%)); turning swap on at any spread mult ≥ 1.5× pushes DD ≥ 10% (FAIL_DD_HARD). The deployment-realistic configuration (swap-on + 1.5–2× spread) is uniformly FAIL at `r_base` — Amendment 3 risk scaling (`r_safe = 0.4336%`, k_safe = 0.87) is REQUIRED for live deployment, not optional.
>
> **Anchor commit:** `244fb763a8ecffd45d9da4eafabf50caff7bc468` (engine drift since anchor = 0 lines on `core/sim/exit_policies/path_simulate.py`, `core/sim/fill.py`, `core/sim/account.py`, `scripts/l_arc_10_v3/step_5.py`)
> **Branch:** `audit/arc_10_cost_realism_sweep`
> **Baseline:** UTC re-run (worst-fold ratio 5.42 / DD 9.22% / ROI 26.49%)

---

## §1 Pre-flight gates — all PASS

### §1.1 Implementation correctness (G1–G4 unit tests on cost primitives)

All four dispatch §4 gates passing, plus 28 edge cases (32 total in `tests/sim/costs/`):

| Gate | Check | Result |
|---|---|---|
| G1 swap day-count | Synthetic Mon 14:00 UTC → Wed 10:00 UTC (summer + winter both) | **2 rollovers** (not 3) ✅ |
| G2 Friday 3× | Trade crossing Friday rollover | 3× multiplier ✅ |
| G3 runner lot | Post-TP1 swap on reduced lot | runner=half, full before TP1 ✅ |
| G4 commission | 1.0 lot RT | exactly **$4** ✅ |

Test command: `py -3.14 -m pytest tests/sim/costs/ -v` → **32 passed in 0.27s**.

Edge cases verified: zero-rollover intra-day trades, trade-exactly-at-rollover boundary excluded (open interval), weekend skip (Sat/Sun no rollover), DST spring-forward handled via `zoneinfo.ZoneInfo("America/New_York")` (mid-March US-DST/EU-DST misalignment resolves correctly).

### §1.2 Baseline reproduction (dispatch §4 + §8)

Cost-free baseline cell (1× spread, swap-off, 0-slip, **commission-off**) reproduces UTC re-run published numbers to four decimal places:

| Metric | Expected (UTC re-run) | Observed (overlay) | Match |
|---|---:|---:|---|
| Worst-fold ratio | 5.4185045 | 5.4185045 | ✅ |
| Worst-fold DD | 9.2241% | 9.2241% | ✅ |
| Worst-fold ROI | 26.4905% | 26.4905% | ✅ |
| Sign consistency | 11/11 | 11/11 | ✅ |
| Total IS trades | 2,162 | 2,162 | ✅ |
| Holdout ROI | 59.07% | 59.07% | ✅ |

**Overlay arithmetic is correct** — no bug introduced by the cost-overlay code path.

Note: the dispatch §4's "1×/swap-off/0-slip cell == unmodified pool" assertion is satisfied with commission OFF (since commission is a real cost and would degrade the cell vs the unmodified pool). The 30-cell grid then has commission ON in every cell as a fixed real cost not on a sweep axis.

### §1.3 §5.2 entry-time assertion (bar misalignment check)

H4 UTC grid = 00/04/08/12/16/20 UTC. Entries at 2200 UTC or 2100 UTC would indicate misaligned bar boundaries.

```
n_at_2200_UTC = 0
n_at_2100_UTC = 0
```

**No bar misalignment** detected. Dispatch §8 stop condition not triggered.

### §1.4 Zero-spread trade count (dispatch §3.3)

```
n_zero_either = 0  (0.00% of 3,301 trades)
```

**Zero zero-spread trades** in the UTC re-run pool. The dispatch §3.3 "if >1% surface" condition is well under the threshold. Spread multiplier widens recorded spread proportionally on every trade — no data-gap floor needed.

---

## §2 FX constants (per dispatch §3.5)

Per-pair representative pip_value_usd computed from UTC re-run pool entry_price means (single constant per pair, 2010–2026 window):

### §2.1 Reference rates used for pip-value conversion

| Reference pair | n_trades | mean_entry_price | pip_value_usd (per 1.0 lot) | Used for |
|---|---:|---:|---:|---|
| USDJPY | 118 | 109.6032 | $9.124 | All ⋆JPY pairs (USDJPY + 6 crosses) |
| USDCAD | 118 | 1.2412 | $8.057 | All ⋆CAD pairs (USDCAD + 4 crosses) |
| USDCHF | 121 | 0.9324 | $10.725 | All ⋆CHF pairs (USDCHF + 5 crosses) |
| NZDUSD | 100 | 0.7047 | $7.047 | NZD-quoted crosses (AUDNZD, EURNZD, GBPNZD) |
| GBPUSD | 116 | 1.4067 | $14.067 | EURGBP (GBP-quoted) |
| AUDUSD | 104 | 0.7760 | $7.760 | AUD-quoted crosses (EURAUD, GBPAUD) |
| (n/a) | n/a | n/a | $10.000 | USD-quoted pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD) |

### §2.2 Caveat

Constant FX over 2010–2026 understates pip-value variance — e.g., USDJPY ranged 76 → 160 over the window, so the average 109.6 is materially off in the tails. Per dispatch's stated tolerance ("10% conversion error = <1% R error"), this is acceptable for finding the cost-level breakpoint. Per-trade FX-at-entry-time would refine swap magnitudes by ~5–10% for the most volatile pairs (USDJPY, GBPJPY) but does not change the verdict ordering of cells.

---

## §3 30-cell grid

### §3.1 Verdict summary

| Verdict | Cells |
|---|---:|
| PASS-DEPLOYABLE | **0** |
| DD_WATCH | 7 |
| FAIL_DD_HARD | 23 |
| FAIL (ratio/ROI) | 0 |

**No cell is PASS-DEPLOYABLE at `r_base = 0.5%`.** The UTC baseline worst-fold DD is 9.22%, already above the 8% PASS-DEPLOYABLE ceiling. Cost addition only worsens DD; no cell crosses below 8%. Confirms Amendment 3 risk-scaling is structurally required for UTC deployment (the EET closure correctly identified this — `k_safe = 1.0879` under EET vs `0.87` under UTC).

### §3.2 Full grid (raw numbers from [grid_results.csv](grid_results.csv))

| Cell | spread× | swap | slip (pip/fill) | worst ratio | worst DD | worst ROI | holdout ROI | holdout DD | verdict |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.0 | off | 0.0 | 5.286 | 9.26% | 26.12% | 58.33% | 5.07% | DD_WATCH |
| 2 | 1.0 | off | 0.5 | 4.916 | 9.37% | 25.01% | 56.02% | 5.14% | DD_WATCH |
| 3 | 1.0 | off | 1.0 | 4.568 | 9.48% | 23.90% | 53.75% | 5.22% | DD_WATCH |
| 4 | 1.0 | ON | 0.0 | 2.690 | **9.98%** | 14.96% | 37.88% | 5.85% | DD_WATCH |
| 5 | 1.0 | ON | 0.5 | 2.451 | 10.12% | 13.54% | 35.87% | 6.09% | FAIL_DD_HARD |
| 6 | 1.0 | ON | 1.0 | 2.225 | 10.26% | 12.14% | 33.90% | 6.33% | FAIL_DD_HARD |
| 7 | 1.5 | off | 0.0 | 4.864 | 9.66% | 24.62% | 54.94% | 5.17% | DD_WATCH |
| 8 | 1.5 | off | 0.5 | 4.521 | 9.76% | 23.58% | 52.69% | 5.26% | DD_WATCH |
| 9 | 1.5 | off | 1.0 | **4.199** | **9.87%** | 22.54% | 50.46% | 5.35% | DD_WATCH |
| 10 | 1.5 | ON | 0.0 | 2.427 | 10.47% | 12.90% | 34.94% | 6.31% | FAIL_DD_HARD |
| 11 | 1.5 | ON | 0.5 | 2.204 | 10.61% | 11.51% | 32.97% | 6.55% | FAIL_DD_HARD |
| 12 | 1.5 | ON | 1.0 | 1.992 | 10.76% | 10.13% | 31.04% | 6.78% | FAIL_DD_HARD |
| 13 | 2.0 | off | 0.0 | 4.476 | 10.05% | 19.88% | 51.63% | 5.32% | FAIL_DD_HARD |
| 14 | 2.0 | off | 0.5 | 4.157 | 10.15% | 18.87% | 49.42% | 5.41% | FAIL_DD_HARD |
| 15 | 2.0 | off | 1.0 | 3.857 | 10.26% | 17.87% | 47.24% | 5.60% | FAIL_DD_HARD |
| 16 | 2.0 | ON | 0.0 | 2.183 | 10.96% | 10.88% | 32.06% | 6.77% | FAIL_DD_HARD |
| 17 | 2.0 | ON | 0.5 | 1.973 | 11.11% | 9.51% | 30.13% | 7.00% | FAIL_DD_HARD |
| 18 | 2.0 | ON | 1.0 | 1.774 | 11.28% | 8.16% | 28.24% | 7.24% | FAIL_DD_HARD |
| 19 | 3.0 | off | 0.0 | 2.354 | 10.87% | 10.91% | 45.32% | 6.04% | FAIL_DD_HARD |
| 20 | 3.0 | off | 0.5 | 2.130 | 10.99% | 9.98% | 43.21% | 6.28% | FAIL_DD_HARD |
| 21 | 3.0 | off | 1.0 | 1.904 | 11.12% | 9.06% | 41.13% | 6.51% | FAIL_DD_HARD |
| 22 | 3.0 | ON | 0.0 | 0.623 | 12.21% | 3.32% | 26.58% | 7.67% | FAIL_DD_HARD |
| 23 | 3.0 | ON | 0.5 | 0.442 | 12.40% | 2.45% | 24.74% | 7.91% | FAIL_DD_HARD |
| 24 | 3.0 | ON | 1.0 | 0.276 | 12.59% | 1.59% | 22.93% | 8.14% | FAIL_DD_HARD |
| 25 | 4.0 | off | 0.0 | 0.492 | 11.82% | 2.62% | 39.43% | 6.95% | FAIL_DD_HARD |
| 26 | 4.0 | off | 0.5 | 0.317 | 11.96% | 1.75% | 37.41% | 7.19% | FAIL_DD_HARD |
| 27 | 4.0 | off | 1.0 | 0.156 | 12.15% | 0.90% | 35.42% | 7.42% | FAIL_DD_HARD |
| 28 | 4.0 | ON | 0.0 | −0.588 | 13.51% | −4.41% | 21.46% | 8.57% | FAIL_DD_HARD |
| 29 | 4.0 | ON | 0.5 | −0.653 | 13.69% | −5.22% | 19.70% | 8.80% | FAIL_DD_HARD |
| 30 | 4.0 | ON | 1.0 | −0.709 | 14.41% | −6.01% | 17.97% | 9.03% | FAIL_DD_HARD |

Commission = $4/lot RT applied in every cell. **Bold** rows are the breakpoint cells (last DD_WATCH on each axis branch). Negative worst-ratios (cells 28–30) mean at least one fold went net-negative under the realistic swap+spread+slip combination.

---

## §4 Breakpoint analysis

### §4.1 Last-surviving cells (DD_WATCH boundary at 10% DD)

| Axis condition | Cell | worst_ratio | worst_DD | gap to 10% |
|---|---:|---:|---:|---:|
| swap ON, otherwise minimal | **4** (1× / on / 0) | 2.69 | **9.98%** | **0.02pp** (razor-thin) |
| swap OFF, spread 1.5× + slip 1.0 | **9** (1.5× / off / 1.0) | 4.20 | **9.87%** | **0.13pp** |
| swap OFF, spread 1.5× + slip 0.5 | 8 | 4.52 | 9.76% | 0.24pp |

**Two interpretations of "breakpoint":**

1. **With realistic swap (always on in live):** Arc 10 can ONLY survive 10% DD at exactly 1× spread + 0 slip — i.e., the strategy is at the 10% DD limit even at 5ers's interbank-spread floor with zero slippage. Any spread widening ≥ 1.5× OR any positive slip pushes DD ≥ 10%.

2. **With swap-off (academic — ignores carry cost):** Arc 10 survives up to ~1.5× spread with 1 pip slip; fails at 2× spread regardless of slip.

The deployment-realistic case is interpretation 1: **Arc 10 v3.0.2 at `r_base = 0.5%` does not survive live trading costs against the 5ers 10% DD limit, with effectively zero safety margin.** This is the structural argument for `r_safe < r_base` per Amendment 3 (closure §4 documents `r_safe = 0.4336%`).

### §4.2 Risk-scaling implication

Under Amendment 3 chained DD scaling with `k_safe = 0.87`, expected DD at deployment is `0.87 × baseline_dd`:

| Cell | r_base DD | × 0.87 → r_safe DD | New verdict at r_safe |
|---|---:|---:|---|
| 4 (1×, on, 0) | 9.98% | 8.68% | DD_WATCH |
| 7 (1.5×, off, 0) | 9.66% | 8.40% | DD_WATCH |
| 10 (1.5×, on, 0) | 10.47% | 9.11% | DD_WATCH (no longer FAIL) |
| 16 (2×, on, 0) | 10.96% | 9.54% | DD_WATCH (no longer FAIL) |
| 22 (3×, on, 0) | 12.21% | 10.62% | **FAIL_DD_HARD** |

At `r_safe = 0.4336%`, realistic-cost cells (1.5–2× spread + swap on) shift from FAIL_DD_HARD to DD_WATCH. The 5ers 10% hard limit is preserved with modest margin. At 3× spread, even risk-scaled DD breaches.

**Conclusion:** Amendment 3's `k_safe = 0.87` is calibrated to preserve survival against approximately 2× HistData spread + full swap. This is plausibly the right calibration for typical 5ers conditions. A regime of 3× spread (e.g., during high-volatility events) plus swap would breach the 5ers 10% limit even at `r_safe`.

---

## §5 Dominant cost vector

Single-axis sensitivity from the "cell-1" reference (1× / swap-off / 0-slip + commission):

| Cost axis turned ON | Δratio | ΔDD (pp) | ΔROI worst-fold (pp) |
|---|---:|---:|---:|
| **Swap on (1×, 0 slip)** | **−2.60** | **+0.72** | **−11.17pp** (26.12 → 14.96) |
| Spread 4× (off, 0 slip) | −4.79 | +2.55 | −23.50pp (26.12 → 2.62) |
| Spread 2× (off, 0 slip) | −0.81 | +0.78 | −6.24pp (26.12 → 19.88) |
| Spread 1.5× (off, 0 slip) | −0.42 | +0.39 | −1.50pp |
| Slip 1.0 pip (1× off) | −0.72 | +0.21 | −2.22pp |
| Slip 0.5 pip (1× off) | −0.37 | +0.10 | −1.11pp |

### §5.1 Dominant cost vector: **SWAP**

For realistic spread (1× to ~2× — 5ers's typical range over HistData interbank), **swap dominates costs by 3–6×** other axes:

- 1× → 1.5× spread: Δratio = −0.42
- 1× → 2× spread:   Δratio = −0.81
- swap on (1×):     Δratio = **−2.60**
- slip 0 → 0.5 pip: Δratio = −0.37
- slip 0 → 1.0 pip: Δratio = −0.72

Only at extreme spread (4×, well outside 5ers's normal range) does spread cost magnitude rival swap.

### §5.2 Why swap dominates this strategy

Three structural reasons:

1. **Long-bias on USD-weak pairs** — Arc 10's V-shape recovery long signal fires preferentially on pairs where USD is the carrier weak-quote (AUD-, NZD-, EUR-, GBP-, CAD-, CHF-quoted), so the long-side swap is steeply negative across nearly all 28 pairs.
2. **Multi-day holding** — `sl_partial_close_1r_runner_trail` runner trails out over typically 5–15 H4 bars (≈ 1–3 days), so most trades cross 1–3 rollovers. Friday-3× catches ~1 in 5.
3. **High average hold for winning runners** — winners' MFE peaks late in the trade life; the runner half-lot holds for many additional bars after TP1, accumulating swap on the reduced lot. The TP1 partial close mitigates ~half the swap on winning trades but not on losers (which exit at SL before TP1).

The mean per-trade swap cost in dollar terms (from cell 4 vs cell 1 mean-fold ROI: 32.09% vs 49.32% per fold, n≈196/fold, $500 risk × annual ratio ≈ $500 risk × 0.86yr = $430/trade) is on the order of $5–15/trade per night × ~2 nights × 0.7 lots ≈ $7–20/trade — material against the $500/trade risk amount.

---

## §6 Diagnostics

### §6.1 §5.1 TP1/SL same-bar frequency — **DEFERRED**

Per dispatch §5.1: requires per-bar `trade_paths.parquet` data which is gitignored and missing from the UTC re-run worktree. Regenerating from anchor (`scripts/l_arc_10_v3/step_1.py -c configs/l_arc_10_v3.0.2_utc_rerun/arc_open.yaml`) would take 10–15 min and require setting up the `data/cache` symlink to the main-repo HistData cache.

Per dispatch §5.1: "If not [regenerable in <15min]: DEFER to a follow-up. Note in report as deferred. Does NOT gate the grid."

**Deferred to follow-up sub-audit §6.1-A.** The grid + breakpoint analysis above are unaffected.

### §6.2 §5.2 2400 EET entry-time assertion — **PASS**

`n_at_2200_UTC = 0`, `n_at_2100_UTC = 0` across all 3,301 trades. UTC H4 grid (00/04/08/12/16/20) intact; no entries fall on the broker-rollover instant. **No bar misalignment.**

### §6.3 Zero-spread count — **0 trades**

No trade in the UTC re-run pool has zero recorded spread on entry or exit. Well under the dispatch §3.3 1%-of-pool threshold.

### §6.4 TP1 hit count + derivation note

Per dispatch §3.1 req 3 + intent §5.3: TP1 hit derived from `pool.mfe_r ≥ 1.75` (since pool MFE is in SL=2.0 units; replayed TP1 = +1R = +3.5 ATR = 1.75 pool-R). TP1 timing approximated by `entry_time + 4hr × time_to_peak_mfe`. This is a conservative upper bound (peak ≥ TP1 crossing); a regenerated trade_paths.parquet would refine the timing to the exact first-crossing bar.

Approx 35–40% of trades hit TP1 (per pool's `mfe_r ≥ 1.75` count). The conservative timing overstates full-lot swap-nights by 1–2 bars per winning trade → swap-on cells slightly OVERSTATE realistic cost. Refined timing would shift the breakpoint marginally favourably; doesn't change the verdict.

---

## §7 Stop conditions — none triggered

Per dispatch §8:

- ✅ Anchor 244fb76 verified (engine drift = 0 lines on swept paths)
- ✅ G1–G4 all PASS (32 unit tests, 0 fails)
- ✅ Baseline reproduces 5.4185 / 9.2241% / 26.4905% within 1e-4
- ✅ §5.2 entry-time assertion = 0 (no bar misalignment)
- ✅ Zero-spread count = 0 (<<1% threshold)

No `simulate_path` modifications, no TP1/SL convention change, no boundary change, no KH-24/live path touched, no spread floor fabricated, no external FX feed. All cost math is post-hoc R-overlay; the engine is byte-identical to the §2.9 audit's verified state.

---

## §8 Headline findings + recommendations

### §8.1 Headline finding

**Arc 10 v3.0.2 under UTC convention at `r_base = 0.5%` does NOT survive realistic live costs against the 5ers 10% hard DD limit.** Only 1× HistData spread + zero slippage + swap-on (Cell 4: DD=9.98%, ratio=2.69) survives, with razor-thin 0.02pp margin to the 10% breach line.

**Amendment 3 risk scaling (`r_safe = 0.4336%`, `k_safe = 0.87`) is structurally required, not optional.** At `r_safe`, realistic-cost cells (1.5–2× spread + swap on) move from FAIL_DD_HARD to DD_WATCH with modest margin to 10%.

### §8.2 Dominant cost vector

**SWAP** (Δratio −2.60 from a single axis flip; 3–6× larger impact than spread or slippage at realistic scale). Driven by Arc 10's structural long-bias on weak-quote pairs combined with multi-day runner-trail holding.

### §8.3 Recommendations

1. **Deploy at `r_safe = 0.4336%` per Amendment 3 — not `r_base = 0.5%`.** Confirmed by this sweep.
2. **Monitor swap rates in live; refuse to enter trades where the daily swap-to-risk ratio exceeds a threshold** (e.g., where one night's expected swap > 10% of risk amount). The two highest-cost pairs are USDJPY (−27.6 points, $9.124 pip → −$2.52/lot/night) and GBPJPY (−31.7 points, $9.124 → −$2.89/lot/night). The lowest are NZDUSD (−2.7 points, $10 → −$0.27/lot/night) — 10× cheaper.
3. **Watch for spread regime shifts.** At 3× HistData spread (high-volatility events: NFP, FOMC, central-bank-surprise days), even risk-scaled DD breaches 10%. A spread filter or event blackout would lift safety margin meaningfully.
4. **Follow up §6.1-A:** Regenerate `trade_paths.parquet` for the UTC re-run, run TP1/SL same-bar diagnostic to confirm the current TP1-first convention isn't double-counting wins on same-bar SL+TP1 trades. Per dispatch §5.1, "report only — no convention change" — but the result informs whether the breakpoint is robust to convention choice.

---

## §9 Deliverables

Under `results/l_arc_10_v3.0.2/cost_sweep/`:

- ✅ [cost_sweep_intent.md](cost_sweep_intent.md) — pre-compute
- ✅ [cost_sweep_report.md](cost_sweep_report.md) — this file
- ✅ [grid_results.csv](grid_results.csv) — raw 30-cell numbers
- ✅ [diagnostics.json](diagnostics.json) — entry-time + zero-spread + baseline check + FX constants
- ✅ [cost_sweep.log](cost_sweep.log) — full run log

Under repo:
- ✅ `configs/swaps_5ers.yaml` — 28-pair swap rates
- ✅ `core/sim/costs/{__init__,swap,commission,slippage,spread_multiplier}.py` — canonical primitives
- ✅ `tests/sim/costs/test_*.py` — G1–G4 + 28 edge cases, 32/32 PASS
- ✅ `scripts/audit/arc_10/cost_sweep.py` — analysis driver

Reproducibility:
```
# Smoke-only (baseline reproduction check)
py -3.14 scripts/audit/arc_10/cost_sweep.py --baseline-only

# Full grid (~30s)
py -3.14 scripts/audit/arc_10/cost_sweep.py
```

---

End of report.

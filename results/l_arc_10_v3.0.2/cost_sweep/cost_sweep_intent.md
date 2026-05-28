# Cost-Realism Sweep — Intent (pre-compute)

> **Dispatch:** CC Dispatch — Arc 10 v3.0.2 Cost-Realism Sweep (REVISED, Option B), 2026-05-27
> **Anchor commit:** `244fb763a8ecffd45d9da4eafabf50caff7bc468` (tag `arc-10-v3.0.2-DEPLOYABLE`)
> **Branch:** `audit/arc_10_cost_realism_sweep` (cut from anchor; merged with `origin/main` to acquire UTC re-run pool, which post-dates 244fb76)
> **Engine drift since 244fb76:** zero diff on `core/sim/exit_policies/path_simulate.py`, `core/sim/fill.py`, `core/sim/account.py`, `scripts/l_arc_10_v3/step_5.py`. `simulate_path` byte-identity from §2.9 audit preserved.

This doc is committed BEFORE any cost-sweep compute, per dispatch §1.

---

## §1 Baseline numbers being compared against

UTC re-run (per [results/l_arc_10_v3_0_2_utc_rerun/COMPARISON_REPORT.md](../../../results/l_arc_10_v3_0_2_utc_rerun/COMPARISON_REPORT.md) + [step_5/best_candidate.md](../../../results/l_arc_10_v3_0_2_utc_rerun/step_5/best_candidate.md)):

| Metric | UTC baseline |
|---|---:|
| Worst-fold ratio | **5.4185** (F6, 2015) |
| Worst-fold DD | **9.2241%** (F4, 2013) |
| Worst-fold ROI | **26.4905%** (F6, 2015) |
| Sign consistency | 11/11 |
| Total IS trades | 2,162 |
| Holdout ROI | 59.07% |
| Holdout DD | 5.30% |
| Holdout ratio | 11.13 |

Winning config: A1, SL=3.5×ATR, exit_policy=`sl_partial_close_1r_runner_trail`, exposure=`unlimited`, cluster c1.

**Baseline reproduction assertion** (per dispatch §4): the 1× spread mult / swap-off / 0-slip cell must reproduce 5.4185 / 9.22% / 26.49% exactly. The cell is the unmodified pool by construction; the assertion catches overlay arithmetic bugs.

---

## §2 The 30-cell grid

| Axis | Levels | Count |
|---|---|---:|
| Spread multiplier (recorded spread × mult, no floor per dispatch §3.3) | 1×, 1.5×, 2×, 3×, 4× | 5 |
| Swap | off, current-5ers-constant | 2 |
| Slippage (adverse pip per fill × 3 fills per trade if TP1 hit else 2) | 0, 0.5, 1.0 pip/fill | 3 |
| **Commission** (constant in all cells, not swept) | $4/lot RT, always ON | — |

5 × 2 × 3 = **30 cells**. Per cell record: `worst_fold_roi`, `worst_fold_dd`, `worst_fold_ratio`, `holdout_roi`, `holdout_dd`, `mean_fold_roi`, `verdict` (PASS-DEPLOYABLE / DD-WATCH / FAIL).

---

## §3 Correctness gates (must pass before grid runs)

| Gate | Check | Expected |
|---|---|---|
| G1 swap day-count | Synthetic Mon 14:00 UTC → Wed 10:00 UTC | 2 rollovers crossed |
| G2 Friday 3× | Synthetic trade crossing Friday rollover | 3× multiplier applied to that night |
| G3 runner lot | Post-TP1 swap on reduced (half) lot | reduced, not original |
| G4 commission | 1.0 lot RT | exactly $4 |

Plus baseline cell == 5.4185 / 9.2241% / 26.4905%.

Any G1–G4 failure or baseline mismatch → STOP and surface (dispatch §8). Grid invalid until fixed.

---

## §4 FX-conversion constants per pair

Per dispatch §3.5: "Use a single representative FX constant per pair, sourced by averaging the relevant price column from the pool over the backtest period." Sourced from `pool.entry_price.mean()` per pair across the UTC re-run pool (3,301 trades, 2010-01-01 → 2026-04-30).

### §4.1 Reference rates used for pip-value conversion to USD

| Reference pair | n_trades in pool | mean_entry_price | used for |
|---|---:|---:|---|
| USDJPY | 118 | 109.6032 | All ⋆JPY pairs (USDJPY + 6 crosses); pip_value_usd = 1000 / 109.6032 ≈ $9.124/lot |
| USDCAD | 118 | 1.2412 | All ⋆CAD pairs (USDCAD + 4 crosses); pip_value_usd = 10 / 1.2412 ≈ $8.057/lot |
| USDCHF | 121 | 0.9324 | All ⋆CHF pairs (USDCHF + 5 crosses); pip_value_usd = 10 / 0.9324 ≈ $10.725/lot |
| NZDUSD | 100 | 0.7047 | NZD-quoted crosses (AUDNZD, EURNZD, GBPNZD); pip_value_usd = 10 × 0.7047 = $7.047/lot |
| GBPUSD | 116 | 1.4067 | EURGBP (GBP-quoted); pip_value_usd = 10 × 1.4067 = $14.067/lot |
| AUDUSD | 104 | 0.7760 | AUD-quoted crosses (EURAUD, GBPAUD); pip_value_usd = 10 × 0.7760 = $7.760/lot |
| (n/a) | n/a | n/a | USD-quoted pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD): pip_value_usd = $10/lot (no conversion) |

### §4.2 Derived pip_value_usd per pair

Computed at sweep time by the driver from §4.1 reference rates. Documented in `cost_sweep_report.md` for traceability.

### §4.3 Caveat

Constant FX over 2010-2026 understates pip-value variance — e.g., USDJPY ranged 76 → 160 over the window, so the average 109.6 is materially off in the tails. Per dispatch's stated tolerance ("10% conversion error = <1% R error"), this is acceptable. Cross-checks for swap-axis cells will confirm cost magnitude is in the expected range (swap-on swing of ~5pp ROI per fold on the order of $5/lot × ~5 nights × ~1k trades ≈ $25k cumulative).

---

## §5 Cost-model implementation details

### §5.1 Swap (`core/sim/costs/swap.py`)

Per trade:
1. Identify all rollover instants in `(entry_time, exit_time)` open interval (strictly between, per dispatch §3.1 requirement 1).
2. For each rollover instant `t_rollover`:
   - Determine if `t_rollover` falls on Friday → apply 3× multiplier (dispatch §3.1 requirement 2).
   - Determine if `t_rollover` is before or after TP1 hit. If TP1 hit (mfe_r ≥ 1.0): runner lot = original / 2 from that point. Per dispatch §3.1 requirement 3.
   - Compute: `nightly_swap_usd = (swap_long_points / 10) × pip_value_usd(pair) × lots_at_t × (3 if Friday else 1)`
3. Sum over all crossings → total swap_usd.
4. Convert to R: `swap_r = total_swap_usd / (risk_pct × INITIAL_BAL)` (simplified — does not compound).

### §5.2 Rollover instant in UTC

Standard FX rollover is 17:00 NY time (close of NY trading day) = 24:00 broker EET/EEST. Per dispatch §3.1 requirement 4 (with **label correction** — see below):

| Period | NY local time | UTC | EU local time |
|---|---|---|---|
| US-DST on (≈ Mar 2nd Sun → Nov 1st Sun): "summer" | 17:00 EDT | **21:00 UTC** | 24:00 EEST |
| US-DST off (≈ Nov 1st Sun → Mar 2nd Sun): "winter" | 17:00 EST | **22:00 UTC** | 24:00 EET |

**Dispatch §3.1 requirement 4 had the labels swapped** (stated "2200 UTC (summer/EDT) / 2100 UTC (winter/EST)"). The above table reflects the standard 17:00 NY convention. Implementation will use `zoneinfo.ZoneInfo("America/New_York")` to resolve the rollover UTC time per actual date, eliminating the manual DST table; pytz/zoneinfo handle the US-DST vs EU-DST misalignment windows (mid-March + late-October) correctly.

### §5.3 TP1 timing derivation (dispatch §3.1 requirement 3)

Pool has `mfe_r` (peak MFE in R over trade life) and `time_to_peak_mfe` (bar offset at which MFE peaked) but no explicit TP1 timestamp. Without per-bar path data:

- If `mfe_r < 1.0`: TP1 never hit → full lot for entire trade.
- If `mfe_r ≥ 1.0`: TP1 hit at some bar `b_tp1` ≤ `time_to_peak_mfe`. Conservative upper bound (penalizes swap cost more, i.e. counts more full-lot nights): assume `b_tp1 = time_to_peak_mfe`. **Documented as derivation.**

Alternative if `trade_paths.parquet` is regenerable (§5.1 of dispatch): use exact bar where `mfe_so_far_r ≥ 1.0` first. Listed under §5 follow-up.

### §5.4 Commission (`core/sim/costs/commission.py`)

`commission_usd = 4.0 × lots_original` (round-turn, once per trade, on original lot per dispatch §2.2 + §3.2). Convert to R as in §5.1 step 4.

### §5.5 Slippage (`core/sim/costs/slippage.py`)

`n_fills = 3 if mfe_r ≥ 1.0 else 2` (entry + final exit always; TP1 fill only if TP1 hit). `slip_pips_total = slip_per_fill × n_fills` (always adverse). Convert to R: `slip_r = slip_pips_total × pip_size(pair) / sl_distance_price`.

### §5.6 Spread multiplier (`core/sim/costs/spread_multiplier.py`)

Recorded per-trade spread (in price units): `spread_total = spread_close_at_entry + spread_close_at_exit`. Multiplier widens proportionally; the original spread is already in `final_r`. Extra cost from widening: `extra_spread_price = spread_total × (mult − 1)`. R: `extra_spread_r = extra_spread_price / sl_distance_price`.

Zero-spread trades (`spread_close_at_entry == 0` or `_exit == 0`): per dispatch §3.3, do NOT floor. Cost remains 0 for that trade in the spread axis. Count + report; surface if >1% of pool.

### §5.7 Lot sizing per trade

Step 5 compounds at fixed 0.5% R, never materializes lot sizes. For cost conversion to R, lot size at entry is:

```
risk_amount_usd  = risk_pct × INITIAL_BAL  =  0.005 × 100,000  =  $500
sl_distance_usd  = sl_distance_price × pip_value_per_pip_per_unit × lot_size_units
                   ↘ to solve for lots: lots_per_trade = risk_amount_usd / (sl_distance_pips × pip_value_per_lot_usd)
sl_distance_pips = sl_distance_price / pip_size(pair)
```

Per dispatch §5.7 simplification: uses INITIAL_BAL as a constant rather than compounding equity_at_entry. Justified because (a) cost effects are small relative to compounding effects in deciding gate breakpoint; (b) the goal is finding breakpoint *cost level*, not exact dollar deductions.

### §5.8 R-adjustment composition

```
adjusted_final_r = final_r − (swap_r + commission_r + slippage_r + extra_spread_r)
```

Per-fold metrics re-aggregate via the same `_fold_metrics` math step_5.py uses (ROI = compounded R-multiple × risk_pct; DD = max drawdown of equity curve).

---

## §6 Pass/fail per cell (dispatch §7)

- **PASS-DEPLOYABLE**: worst-fold ratio ≥ 2.0 AND worst-fold ROI ≥ 5% AND no daily-DD breach
- **DD-WATCH**: worst-fold DD ∈ [8%, 10%]
- **FAIL**: ratio < 2.0 OR ROI < 5% OR DD ≥ 10% OR daily-DD breach

**Daily-DD note:** Step 5 doesn't emit per-day equity; daily-DD requires the Amendment 3 / per-day-bucketing addendum primitive. Within scope: approximate via worst-fold DD; flag if any cell triggers DD-WATCH or DD ≥ 10%. A full daily-DD recheck would require running `core.runners._fold_stats_helpers.compute_per_day_max_dd` per cell over the adjusted equity. Listed as follow-up if any cell hits DD-WATCH.

---

## §7 Deliverables (post-compute)

Under `results/l_arc_10_v3.0.2/cost_sweep/`:

- ✅ `cost_sweep_intent.md` (this file, pre-compute)
- ⏳ `cost_sweep_report.md` — 30-cell grid, G1–G4 results, baseline assertion result, breakpoint cell + dominant-vector analysis, zero-spread count, FX constants table, §5 diagnostics
- ⏳ `grid_results.csv` (lineterminator='\n') — raw per-cell numbers
- ⏳ `cost_sweep.log` — run log

Under repo:
- ✅ `configs/swaps_5ers.yaml` — committed pre-compute
- ⏳ `core/sim/costs/{__init__,swap,commission,slippage,spread_multiplier}.py` — canonical primitives
- ⏳ `tests/sim/costs/test_{swap,commission,slippage,spread_multiplier}.py` — G1–G4 unit tests + edge cases
- ⏳ `scripts/audit/arc_10/cost_sweep.py` — analysis driver

---

## §8 Stop conditions (mirrored from dispatch §8)

- Anchor-gated paths drift since 244fb76 (engine code modification)
- Any G1–G4 unit test fails
- Baseline cell ≠ 5.4185 / 9.2241% / 26.4905% (overlay arithmetic bug)
- §5.2 (2400 EET entry) returns non-zero (bar misalignment)
- Zero-spread trade count > 1% of pool

Do NOT: change TP1/SL convention, change boundary to EET, touch `simulate_path` / KH-24 / live paths, fabricate spread floors, pull external FX rates.

---

## §9 Environment record

```
HEAD                : (post-merge of origin/main into audit branch; engine paths sha-identical to 244fb76)
anchor              : 244fb763a8ecffd45d9da4eafabf50caff7bc468
git branch          : audit/arc_10_cost_realism_sweep
python              : 3.14.0
numpy               : 2.4.4
pandas              : 3.0.2
sklearn             : 1.8.0   (not used in cost sweep; recorded for consistency with §2.9 env)
pyarrow             : 23.0.1
platform            : Windows-11-10.0.26200-SP0
cpu_count           : 12
boundary_convention : UTC (per dispatch §0 — NOT 5ers_eet)
pool source         : results/l_arc_10_v3_0_2_utc_rerun/step_1/pool.parquet  (3,301 trades, post-PR-#218)
config source       : configs/l_arc_10_v3.0.2_utc_rerun/arc_open.yaml
```

---

End of intent doc.

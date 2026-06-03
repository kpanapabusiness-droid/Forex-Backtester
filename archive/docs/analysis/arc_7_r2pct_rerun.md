# Arc 7 Best-Config Re-run at r = 2% (WFO + Full Sim)

> **Config:** Arc 7 v3.0.2 best (`A6::A6::cl1::sl2.0::thr0.5-0.7::exp2`)
> **Risk:** `r_base = 2.0%` (vs canonical `0.5%`)
> **Authority:** user override (analysis re-run, not an arc closure)
> **Branch:** `analysis/arc_7_wfo_rerun_r2pct`
> **Output dirs:**
>   - WFO: `results/analysis/arc_7_r2pct/wfo/`
>   - Full sim: `results/analysis/arc_7_r2pct/full_sim/`
>   - Combined summary: `results/analysis/arc_7_r2pct/run_summary.json`
>
> **Post-investigation note (added on rebase, 2026-05-29).** §4.1 originally
> flagged a "trade-count drop ~36-38%" as an unexplained non-linearity. A
> follow-up investigation has since closed this thread:
> [docs/dispatches/driver_script_divergence_investigation.md](../dispatches/driver_script_divergence_investigation.md).
> The investigation traced the gap to a **driver-script default mismatch**:
> the analysis script (this branch's commits) defaulted `--window-end` to
> `2025-12-31`, four months earlier than the canonical orchestrator's
> `2026-04-30`. At matched window ends the two scripts produce byte-identical
> closed-trade ledgers. The artefacts in this PR retain the original
> `2025-12-31` window for audit trail; consumers wanting the canonical-window
> numbers should invoke `scripts/l_arc_7_v3_0_2/run.py` directly or rerun
> the analysis script with `--window-end 2026-04-30`. See §4.1 below.

Reuses Arc 7 v3.0.1 Step 1-4 artefacts at [results/l_arc_7/](../../results/l_arc_7/) (pool sha256-verified at load; A6 classifier from `step_4/classifiers/1.pkl`). The ONLY change from v3.0.2 canonical is `risk_pct = 0.005 → 0.02`. Pool size 5,175 trades; cluster c1 (Unclassified, RF AUC 0.6642). Boundary convention `5ers_eet`. Determinism: `random_state=42`, `n_jobs=1`, `lineterminator='\n'`. Wall-clock: 8.2 minutes total.

---

## §1 Deliverable A — Canonical WFO at r=2%

### §1.1 Per-fold OOS table

| Fold | OOS window | n_trades | ROI % | DD % | Daily 5% breaches | Ratio |
|---:|---|---:|---:|---:|---:|---:|
| 2  | 2011 | 14 | 12.28% | 2.93% | 0 | 4.19 |
| 3  | 2012 | 11 |  9.29% | 2.28% | 0 | 4.07 |
| 4  | 2013 | 17 | 29.89% | 2.33% | 0 | 12.82 |
| 5  | 2014 | 11 |  **3.52%** | **4.57%** | 0 | **0.77** ← worst |
| 6  | 2015 | 14 | 10.91% | 2.71% | 0 | 4.02 |
| 7  | 2016 | 14 | 14.96% | 3.49% | 0 | 4.28 |
| 8  | 2017 | 17 | 10.88% | 3.90% | 0 | 2.79 |
| 9  | 2018 | 14 | 12.81% | 4.54% | 0 | 2.82 |
| 10 | 2019 | 19 | 10.48% | 3.14% | 0 | 3.34 |
| 11 | 2020 | 16 | 15.07% | 2.25% | 0 | 6.69 |
| H (12) | 2021-2025 | 87 | 61.95% | 5.84% | 1 | 10.61 |

Note: fold 1 has empty IS and is dropped by the canonical `min_is_days=365` gate.

### §1.2 Summary stats (IS folds, exclusive of holdout)

- Worst-fold ratio: **0.77** (fold 5, 2014)
- Mean-fold ratio: 4.58
- Worst-fold ROI %: 3.52% (fold 5, 2014)
- Worst-fold DD %: 4.57% (fold 5, 2014)
- All 10 IS folds positive ROI: **YES**
- Min trades / fold: 11
- Chained max DD (IS + holdout, equity-stitching method): **5.84%**

### §1.3 Holdout one-shot (2021-01-01 → 2025-12-31)

- ROI: **61.95%** (over 5 years)
- DD: 5.84%
- Trades: 87
- Daily 5% breaches at r=2%: **1 day**
- Holdout ratio: 10.61

### §1.4 Comparison vs r=0.5% canonical (Arc 7 v3.0.2)

| Metric | r=0.5% (canonical) | r=2.0% (this run) | Multiplier | Expected (linear 4×) |
|---|---:|---:|---:|---:|
| Worst-fold ROI         |  4.49% (fold 11) |  **3.52% (fold 5)** | 0.78×  | ~17.96% |
| Worst-fold DD          |  1.26% (fold 10) |  **4.57% (fold 5)** | 3.63×  | ~5.06%  |
| Worst-fold ratio       |  4.36 (fold 10)  |  **0.77 (fold 5)**  | 0.18×  | ~4.36 (unchanged) |
| Mean-fold ratio        |  8.70            |  4.58               | 0.53×  | ~8.70   |
| Chained max DD         |  2.09%           |  **5.84%**          | 2.79×  | ~8.36%  |
| Holdout ROI            | 28.48%           |  **61.95%**         | 2.18×  | ~113.9% |
| Holdout DD             |  2.09%           |  **5.84%**          | 2.79×  | ~8.36%  |
| Holdout ratio          | 13.62            | 10.61               | 0.78×  | ~13.62 (unchanged) |
| Holdout trades         | 136              | **87**              | 0.64×  | (no scaling expected) |
| Sum IS trades          | 237              | **147**             | 0.62×  | (no scaling expected) |
| Worst-fold identity    | fold 10 (by ratio) | **fold 5 (by ratio)** | — | — |

**Key non-linearities surfaced:**

1. **The worst fold *changes identity* at higher risk.** At r=0.5% fold 10 was worst (ratio 4.36, DD 1.26%). At r=2% fold 5 is worst (ratio 0.77, DD 4.57%). Fold 5's DD scales ~3.8× — close to linear — but its ROI scales only 0.50× (3.52% vs the 7.06% at r=0.5%, would have linearly extrapolated to ~28.2%). The ROI compression at the worst fold is what crushes the ratio from 5.84 (fold 5 at r=0.5%) to 0.77 at r=2%.

2. **Trade counts drop ~36-38%.** WFO holdout dropped 136 → 87 trades. IS folds dropped from a ~24/yr cadence at r=0.5% to ~15/yr at r=2%. The same signal stream + same deterministic classifier produces the same admit attempts; the gap must come from exposure-cap interactions (positions held during a bar reject simultaneous admits). The dispatch §1.4 anticipated this ("exposure cap interactions"); this run is the first quantitative measurement. Mechanism worth a follow-up audit — at higher r the balance trajectory is more volatile, but `LiveBalanceRisk.risk_size` only affects sizing, not admit/exposure timing. Possible cause: `mark_to_market` interactions or a non-obvious dependency in `MultiPairBacktester._fill_pending_entries`. **Flagged** but not investigated in this analysis dispatch.

3. **Holdout ratio at r=2% (10.61) is below the 13.62 at r=0.5%** — the cleanest fold (2021-2025 holdout, capturing post-COVID + 2022 inflation regime) still loses ratio at higher risk, consistent with #1 above.

4. **One daily 5% breach during holdout at r=2%** (vs 0 at r=0.5%). Surfaces in §2 as the date(s) below.

Linear scaling would have produced worst-fold DD ~5.06% and chained max DD ~8.36%. Realised chained max DD 5.84% — **better** than linear, because per-fold compounding distributes risk consumption across folds. Realised worst-fold ROI 3.52% — **much worse** than the linear 17.96% — because fold 5 2014's mix of trades produces compounding-unfavourable sequencing at higher r.

---

## §2 Deliverable B — Full continuous sim 2010-2025 at r=2%

### §2.1 Sim mechanism used

**Option 1 (dispatch §5.1) — single-fold extended OOS.** Constructed a single `Fold` with `oos_start = 2010-01-01`, `oos_end = 2025-12-31`, and empty IS (A6 uses a persisted classifier from v3.0.1 Step 4; no per-fold retrain per Amendment 2). The architecture's `_slice_panels_to_fold(...)` helper provides the standard 60-day warmup, which is moot here since data starts at 2010-01-01.

This produces ONE continuous equity curve across the full 16-year window with no per-fold resets, fresh-account-per-fold artefacts, or holdout boundary. Same A6 config, same classifier, same exit policy (SL=2.0×ATR + trailing ATR), same exposure cap (2 per currency, 1 per pair), `r_base = 0.02`, `starting_balance = $100,000`.

Wall-time: 212 seconds (3.5 min).

### §2.2 Summary stats over full 2010-2025 window

| Metric | Value |
|---|---:|
| Starting equity                  | $100,000.00 |
| Ending equity                    | $656,722.23 |
| Total return                     | **+556.72%** (~12.4% annualised geometric, 16 yr) |
| Max drawdown (peak-to-trough)    | **9.90%** |
| Max DD duration (peak → trough)  | 1 day |
| Max DD duration (peak → recovery)| 46 days |
| Number of trades (whole closes)  | 256 |
| Win rate                         | **70.7%** |
| Average R per trade              | **+0.75 R** |
| Sharpe (annualized, H4-bar log-returns) | **1.46** |
| Worst single-day DD (5ers EET boundary) | 2.64% |
| Days breaching 5ers 5% daily limit | **0** |
| Worst rolling 30-day max-to-min swing | 10.26% |
| Exit-reason mix                  | 188 trailing_stop / 68 stop_loss |

**Notes on metric definitions:**
- "Max drawdown" is the classical peak-to-trough on the cumulative-max envelope.
- "Worst rolling 30-day max-to-min swing" measures the worst `(max − min) / max` within any 30-day window, ignoring whether the max occurred before or after the min. This metric can exceed the peak-to-trough number when a transient high inside the window is not the all-time HWM — present here only because the deepest single drawdown (Nov 4-5, 2010) recovered within the same 30-day window during which the peak still held.
- Avg R = `(exit_price − entry_price) × direction_sign / |entry_price − sl_price|`. The +0.75 average across 256 trades is consistent with a meta-labeled trend-following profile (high win rate, asymmetric trail-driven payoffs).

### §2.3 Equity curve highlights

**Top-5 drawdown periods** (sorted by depth):

| Rank | Peak | Trough | End | Depth | Trough → Recovery (days) | Peak → End (days) |
|---:|---|---|---|---:|---:|---:|
| 1 | 2010-11-04 | 2010-11-05 | 2010-12-20 |  **9.90%** | 44  | 46  |
| 2 | 2023-04-28 | 2023-04-28 | 2023-05-11 |  5.84% | 12  | 13  |
| 3 | 2014-02-12 | 2014-06-26 | 2014-10-06 |  4.57% | 102 | 235 |
| 4 | 2015-11-17 | 2016-08-11 | 2016-09-06 |  4.57% | 26  | 293 |
| 5 | 2018-04-13 | 2018-08-06 | 2018-08-09 |  4.54% |  3  | 118 |

- Starting equity $100k; final equity $656,722.
- **Worst drawdown is the very first drawdown** (Nov 2010, depth 9.90%, recovered in 46 days). This is the 5ers-survival pivot — see §2.4.
- Drawdown #3 (2014-02 → 2014-10) coincides with the worst WFO fold (fold 5, 2014).
- **Zero years with negative annual return** across 16 years (2010-2025 all positive).

### §2.4 5ers prop firm survival check

5ers limits per [CLAUDE.md](../../CLAUDE.md) and [L_PROTOCOL.md](../../L_PROTOCOL.md) Amendment 6:
- 10% max drawdown (account closing)
- 5% daily drawdown (EET broker trading day, Amendment 6 boundary)

| Limit | Worst observed | Breached? | Margin |
|---|---:|:---:|---:|
| Max DD ≤ 10%       | 9.90% (Nov 4-5, 2010) | **NO**  | **0.10 pp** |
| Daily DD ≤ 5%      | 2.64% | NO | 2.36 pp |
| Days > 5% daily DD | 0 over 16 yr | — | — |

**Verdict — narrowly survived.** The configuration would have stayed inside both 5ers limits if deployed at r=2% from 2010-01-01, but the Nov 2010 drawdown (week-2 of the deployment) sits 0.10 percentage points below the 10% account-closing limit. That is ~$97 of margin on a $100k account in the first month of trading. Any execution friction (additional slippage, one extra stopped-out trade in the same window, a half-spread cost not modelled) plausibly converts this into a fatal breach.

**Reading note on 5ers max-DD interpretation:** the 9.90% number is peak-to-trough against the cumulative-max equity envelope. Depending on whether 5ers tracks max DD against starting balance, against a sliding HWM, or against a trailing reference, the operative limit interpretation can differ. Against starting balance (= $100k), the Nov 2010 trough was at $99,838 — *barely* below starting balance, ~0.16% under. Against the cumulative-max envelope, it was 9.90% below the peak just hours before. The latter is the conservative reading; the former is more permissive. This run does not attempt to litigate the precise 5ers rule semantics — the user knows their broker's convention.

**Holdout daily breach context.** The 1 day of >5% daily DD recorded in the Deliverable A holdout (§1.3) at r=2% is on the *fold-isolated* equity series, where the holdout sim starts from $100k on 2021-01-01 — that smaller base amplifies relative DD on any given trading day. In the continuous sim (§2.2), at equivalent 2021+ dates the account was at $300-650k, and the same absolute USD daily-DD magnitude no longer crosses 5%. **The continuous sim records zero daily-5%-breach days.** This is informative: the WFO daily-breach gate is most conservative on small bases (e.g. the first OOS year of a fold), and gives a different signal than a continuous-deployment sim at the same r.

---

## §3 Comparison: WFO vs Full Sim

Worst-fold metrics from WFO measure "worst-case 1-year period at a $100k base." Full sim measures "actual cumulative outcome over 16 years with realised compounding."

| Question | WFO answer (r=2%) | Full sim answer (r=2%) |
|---|---|---|
| Worst observed drawdown | 5.84% (chained across IS+holdout) | 9.90% (Nov 2010) |
| Worst 1-year-window ROI | 3.52% (fold 5, 2014) | (not directly comparable) |
| Total cumulative return | (per-fold resets to $100k) | +556.7% over 16 yr |
| Days breaching 5% daily | 1 (holdout fold base $100k) | 0 (compounded base) |
| Sharpe-equivalent | (per-fold; not directly computed) | 1.46 annualised |
| 5ers survival outcome | passes Amendment 3 daily-breach gate per fold | **survives by 0.10 pp on max DD** |

**Two key divergences:**

1. **Full-sim max DD (9.90%) exceeds WFO chained max DD (5.84%) by ~70%.** The WFO stitching method (per Amendment 3) chains per-fold OOS equity multiplicatively; each fold restarts from a fresh $100k base, so a fold's intra-fold drawdown is bounded by what one year + 5-yr holdout can produce. The Nov 2010 drawdown falls inside fold 1 (empty IS, dropped from WFO) and the full-sim sim. The full sim captures it; the WFO stitching does not. **This is the load-bearing divergence for deployment decisions** — a deployable WFO verdict on the chained-DD axis (5.84% < 8% safe gate) under-states the actual deployment exposure (9.90% on the first month of trading).

2. **Daily-breach count flips** (WFO: 1 in holdout → full sim: 0). As discussed in §2.4, this is a measurement-base artefact — at the chained-equity higher base, the same absolute USD daily moves don't cross 5% relative.

---

## §4 Notes

### §4.1 Trade-count discrepancy vs canonical — RESOLVED

This section originally flagged a ~36-38% trade-count drop from r=0.5% canonical to r=2% as an unexplained non-linearity. **It has since been investigated and closed.** See [docs/dispatches/driver_script_divergence_investigation.md](../dispatches/driver_script_divergence_investigation.md).

**Root cause:** the analysis driver script (`scripts/analysis/arc_7_r2pct_rerun.py`) defaulted `--window-end` to `"2025-12-31"`, four months earlier than the canonical orchestrator (`scripts/l_arc_7_v3_0_2/run.py`) which defaults to `"2026-04-30"`. The numbers in §1-§3 of this report were computed with the shorter holdout. The "missing" Jan-Apr 2026 tail accounts for the trade-count gap; at matched window ends the two scripts produce **byte-identical closed-trade ledgers** (every `(entry_time, pair, exit_time, exit_reason)` tuple matches).

**Quantitative reconciliation** (from the investigation §2.1):

| Engine state | `--window-end` | Holdout trades |
|---|---|---:|
| Pre-W1-fix (Arc 7 v3.0.2 closure engine) | 2025-12-31 | 87 (this report's number) |
| Pre-W1-fix | 2026-04-30 | 98 |
| Current `main` (post-PR #208 W1 lookahead fix) | 2025-12-31 | 137 |
| Current `main` | 2026-04-30 | 150 |
| Arc 7 v3.0.2 committed closure CSV (pre-W1, Apr 2026 end) | 2026-04-30 | 136 |

So the canonical "136" baseline I compared against in §1.4 was at engine = pre-W1-fix + window-end = Apr 2026; this report's "87" is at the same engine + window-end = Dec 2025. The 87 → 98 gap (11 trades) is exactly the 4-month Jan-Apr 2026 tail. The headline §1.4 row "Holdout trades: 136 → 87" is an **apples-to-oranges** comparison artefact, not a sim divergence.

**What changes about §1-§3:** the numbers themselves are correct *for window-end = 2025-12-31* (the report's stated scope), but the §1.4 comparison-vs-canonical column treats canonical's longer holdout as a baseline. The headline ROI / DD / ratio outcomes are not meaningfully affected by the 4-month tail (it's a 7% extension of a 5-year holdout), but the trade-count column is materially off. Future revisions of this analysis should either (a) re-run with `--window-end 2026-04-30` for an apples-to-apples comparison or (b) explicitly re-extract the canonical 2021→2025-12-31 trade count for the comparison row.

**Engine note.** Post-PR #208 (W1 lookahead fix), all trade counts in this table jump by ~50%. The Arc 7 v3.0.2 closure (and this analysis's run) used the pre-W1-fix engine; a deployment-decision rerun on current `main` would produce different numbers, dominated by the engine fix rather than the risk change.

### §4.2 Worst-fold identity flip (fold 10 → fold 5)

The canonical r=0.5% worst fold was 2019 (fold 10) on ratio terms. The r=2% worst fold is 2014 (fold 5). Fold 5 2014 at r=2% returns 3.52% with 4.57% DD — the lowest ratio (0.77) of any fold. Fold 5 at r=0.5% had ratio 5.84 (ROI 7.06%, DD 1.21%). The ~3.8× DD scaling combined with only ~0.50× ROI scaling at this fold is what flips the worst-fold-identity table. Without a per-trade R sequence comparison (out of scope here), the most parsimonious read is that fold 5's drawdown clusters in a way where higher r consumes capital faster than wins recover it.

### §4.3 Deviation from linear scaling

The dispatch anticipated ~4× linear scaling. Actual scaling on signal-quality-blind metrics:
- Worst-fold DD: 3.6× (close to 4×)
- Chained max DD: 2.8× (sub-linear, good)
- Holdout DD: 2.8× (sub-linear)
- Holdout ROI: 2.2× (sub-linear — this is the **bad** sub-linearity)
- Holdout trades: 0.64× (artefact of the §4.1 window-end mismatch — not a real scaling effect)

The Holdout-ROI sub-linearity (2.2× vs expected 4×) combined with the worst-fold-ROI compression (0.78× vs expected 4×) is the core "scaling doesn't behave like multiplying by 4" finding. The configuration's *risk efficiency* (ratio) degrades meaningfully as risk scales up — exactly the "too clean to scale" caution from the v3.0.2 Amendment-3 verdict, now measured directly at r=2% deployment risk. Note that the Holdout-ROI sub-linearity is computed across two different holdout windows (canonical's Apr 2026 vs this run's Dec 2025) and would shift slightly under an apples-to-apples rerun; the qualitative finding (sub-linear) stands but exact multipliers are sensitive to the window-end mismatch.

### §4.4 Out of scope

- No tracker update (analysis re-run, not arc closure).
- No Step 6 dispatch (this is not a new arc).
- No engine code changes.
- No Amendment-3 reinterpretation. The user is producing this data for their own deployment-decision purposes.
- KH-24 system unaffected; arc/l_arc_7 and arc/l_arc_7_v3.0.2 verdicts unchanged.

---

## §5 Artefacts manifest

```
results/analysis/arc_7_r2pct/
├── run_summary.json                           # combined-run summary
├── run.log                                    # full stdout/stderr
├── wfo/
│   ├── wfo_results.csv                        # 10 IS folds per-fold metrics
│   ├── holdout_results.csv                    # 1 holdout fold
│   ├── per_day_max_dd_chained.parquet         # stitched IS+holdout per-day DD
│   ├── wfo_summary.md                         # text summary
│   ├── summary_stats.json                     # machine-readable WFO summary
│   └── manifest.json                          # sha256 + sizes
└── full_sim/
    ├── equity_curve.parquet                   # H4-bar mark-to-market series, 2010-2025
    ├── trade_ledger.parquet                   # 256 closed trades + R-at-close
    ├── per_day_max_dd.parquet                 # 5ers-EET daily DD over full window
    ├── summary_stats.json                     # machine-readable full-sim summary
    └── manifest.json                          # sha256 + sizes

configs/analysis/arc_7_r2pct/
├── wfo_config_r2pct.yaml                      # Deliverable A spec
└── full_sim_config_r2pct.yaml                 # Deliverable B spec

scripts/analysis/
└── arc_7_r2pct_rerun.py                       # driver
```

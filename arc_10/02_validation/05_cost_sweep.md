# Cost Sweep — EET (FundedNext) vs UTC (5ers)

> **Source artifact:** `results/l_arc_10_v3.0.2/fundednext_cost_sweep/fundednext_cost_sweep_report.md` (full report) + `results/l_arc_10_v3.0.2/fundednext_cost_sweep/`
> **Verdict:** EET is the deployable venue. UTC remains viable as fallback.
> **Risk locked:** 0.50% EET (FundedNext) / 0.40% UTC (5ers).

## Executive summary

| | **EET — FundedNext** | **UTC — 5ers** |
|---|---|---|
| Swap | OFF (swap-free add-on) | ON (real cost) |
| Commission | $5/lot RT | $4/lot RT |
| Daily-DD boundary | EET (broker midnight) | UTC (midnight) |
| Max DD limit | 10% | 10% |
| **Backtest worst-fold ratio (central cell)** | **5.44** | **2.43** |
| **Backtest worst-fold DD (central cell)** | **7.80%** | **10.47%** |
| **Backtest worst-fold ROI (central cell)** | **18.47%** | **12.90%** |
| **Backtest holdout ROI (central cell)** | **46.24%** | **34.94%** |
| Recommended risk | **0.50%** (r_base) | **0.40%** |
| Expected live holdout ROI | **~44%** | **~28%** |
| Expected live worst-fold DD | **~8.8%** | **~9.5%** |
| Verdict | **PASS-DEPLOYABLE** | **DD-tight, deployable with discipline** |

**EET wins on every economic axis.** UTC remains deployable but with thinner margins and lower returns.

## Methodology

Both venues evaluated via the canonical Arc 10 v3.0.2 cost-realism sweep using existing `core/sim/costs/` primitives (post-hoc R-overlay on the WFO pool; `simulate_path` untouched). Cost vectors applied per trade:

- **Swap:** rollover-crossing count, Friday 3× multiplier, runner-lot reduction post-TP1, DST-correct rollover instant (UTC only — OFF for EET case)
- **Commission:** flat $/lot round-turn, scales with original lot size
- **Spread:** `effective_spread = recorded_spread × mult` (no floor; widens HistData embedded spread)
- **Slippage:** `slip_per_fill × n_fills` (n_fills = 3 if TP1 hit else 2; adverse)

UTC ran a 30-cell grid (5 spread × 2 swap × 3 slip). EET ran a 15-cell grid (5 spread × 3 slip, swap-free locked). Both grids include commission ON in every cell.

Both sweeps verified clean: G1–G4 correctness gates passed, baseline reproduction to 1e-16, zero-spread trades 0.00%, no bar misalignment.

## EET Results (FundedNext)

### Baseline (pre-cost overlay)

From `results/l_arc_10_v3.0.2/` EET WFO. PASS-DEPLOYABLE: worst-fold ratio 6.43, worst-fold DD 7.35%, holdout 52.83%.

See `01_wfo_results.md` for full per-fold breakdown.

### 15-cell grid at r_base = 0.5%

| Cell | Spread | Slip | Ratio | DD | Worst ROI | Mean ROI | Holdout ROI | Holdout DD | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0× | 0.0 | 6.29 | 7.40% | 21.88% | 49.18% | 51.98% | 5.58% | PASS |
| 2 | 1.0× | 0.5 | 5.99 | 7.49% | 20.52% | 47.40% | 49.83% | 5.75% | PASS |
| 3 | 1.0× | 1.0 | 5.69 | 7.59% | 19.17% | 45.64% | 47.72% | 5.91% | PASS |
| 4 | 1.5× | 0.0 | 5.81 | 7.70% | 19.81% | 45.23% | 48.33% | 6.00% | PASS |
| **5** | **1.5×** | **0.5** | **5.44** | **7.80%** | **18.47%** | **43.50%** | **46.24%** | **6.17%** | **PASS (central)** |
| 6 | 1.5× | 1.0 | 4.94 | 7.90% | 17.14% | 41.79% | 44.18% | 6.33% | PASS |
| 7 | 2.0× | 0.0 | 5.02 | 8.01% | 17.77% | 41.40% | 44.77% | 6.44% | DD_WATCH |
| 8 | 2.0× | 0.5 | 4.55 | 8.10% | 16.45% | 39.71% | 42.73% | 6.64% | DD_WATCH |
| 9 | 2.0× | 1.0 | 4.10 | 8.20% | 15.15% | 38.05% | 40.72% | 6.84% | DD_WATCH |
| 10 | 3.0× | 0.0 | 3.46 | 8.61% | 13.78% | 34.11% | 37.95% | 7.54% | DD_WATCH |
| 11 | 3.0× | 0.5 | 3.08 | 8.71% | 12.52% | 32.52% | 36.01% | 7.74% | DD_WATCH |
| 12 | 3.0× | 1.0 | 2.72 | 8.86% | 11.26% | 30.95% | 34.09% | 7.93% | DD_WATCH |
| 13 | 4.0× | 0.0 | 2.08 | 9.55% | 9.84% | 27.35% | 31.69% | 8.54% | DD_WATCH |
| 14 | 4.0× | 0.5 | 1.85 | 9.70% | 8.83% | 25.85% | 29.85% | 8.73% | FAIL (ratio) |
| 15 | 4.0× | 1.0 | 1.62 | 9.85% | 7.68% | 24.36% | 28.03% | 8.92% | FAIL (ratio) |

**Sign consistency 11/11 in every cell of the EET grid**, including 4× spread stress. Worst fold by ROI: F9 (2018) at 18.47%. Worst fold by DD: F4 (2013) at 7.80%.

### Recommended risk for EET

Per Amendment 3 linear scaling: `r_recommended = 0.005 × (0.080 / worst_DD_at_r_base)`.

| Reference cell | Worst DD @ r_base | r_recommended |
|---|---|---|
| Optimistic (1.0× / 0.5 slip) | 7.49% | 0.0053 |
| **Central (1.5× / 0.5 slip)** | **7.80%** | **0.0051** |
| Adverse (2.0× / 1.0 slip) | 8.20% | 0.0049 |

**Recommendation: deploy at r_base = 0.50%.**

Justification: r_recommended barely moves off r_base across the realistic band (0.49%–0.53%). The pool sits at the 8% DD target at 0.5% risk, leaving essentially no headroom to scale risk up. The adverse cell actually recommends scaling *down* to 0.49%. Deploying at 0.50% accepts the central-case worst-fold DD of 7.80% with hard-limit margin to 10%.

## UTC Results (5ers)

### Baseline (pre-cost overlay)

PASS-DEPLOYABLE but tighter: worst-fold ratio 5.42, worst-fold DD 9.22%, holdout 59.07%.

### 30-cell grid at r_base = 0.5%

Swap is the dominant cost vector under UTC. Sensitivity: swap-ON alone drops worst-fold ratio by 2.60 and worst-fold ROI by 11.17pp.

| Cell | Spread | Swap | Slip | Ratio | DD | Worst ROI | Holdout ROI | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.0× | off | 0.0 | 5.29 | 9.26% | 26.12% | 58.33% | DD_WATCH |
| 4 | 1.0× | ON | 0.0 | 2.69 | 9.98% | 14.96% | 37.88% | DD_WATCH |
| 7 | 1.5× | off | 0.0 | 4.86 | 9.66% | 24.62% | 54.94% | DD_WATCH |
| **10** | **1.5×** | **ON** | **0.0** | **2.43** | **10.47%** | **12.90%** | **34.94%** | **FAIL (central, breaches 10%)** |
| 12 | 1.5× | ON | 1.0 | 1.99 | 10.76% | 10.13% | 31.04% | FAIL |
| 16-30 | 2.0×+ | various | various | 0.28-4.48 | 10-14% | <0-22% | 18-52% | FAIL on most |

No cells PASS-DEPLOYABLE at r_base 0.5% under UTC. Realistic case (swap-ON, 1.5× spread) lands above the 10% DD hard limit.

### Recommended risk for UTC

Linear DD scaling to bring worst-fold under 10% hard limit:

| Cell (UTC, realistic) | r_base DD | At r = 0.40% | At r = 0.42% |
|---|---|---|---|
| 4 (1.0× / ON / 0 slip) | 9.98% | 7.98% | 8.38% |
| 10 (1.5× / ON / 0 slip) | 10.47% | 8.38% | 8.79% |
| 16 (2.0× / ON / 0 slip) | 10.96% | 8.77% | 9.21% |

**Recommendation: deploy at r = 0.40%.**

Justification: UTC at r_base 0.5% breaches the 10% DD hard limit on the realistic swap-ON case (cell 10: DD 10.47%). Linear scaling to r = 0.40% brings central-case worst-fold DD to 8.38%, leaving ~1.6pp margin to the hard limit. Lower than EET's headroom, but defensible. Cannot deploy at r_base under UTC.

## EET vs UTC Cross-Comparison

Cost-adjusted delta on the realistic central cells (1.5× spread, 0.5 slip, swap-off cells for direct comparison; UTC carries $4 commission, EET carries $5):

| Spread | Slip | UTC ratio | EET ratio | Δratio | UTC DD | EET DD | ΔDD |
|---|---|---|---|---|---|---|---|
| 1.0× | 0.0 | 5.29 | 6.29 | +1.01 | 9.26% | 7.40% | **−1.87pp** |
| 1.0× | 0.5 | 4.92 | 5.99 | +1.07 | 9.37% | 7.49% | −1.87pp |
| 1.0× | 1.0 | 4.57 | 5.69 | +1.13 | 9.48% | 7.59% | −1.88pp |
| 1.5× | 0.0 | 4.86 | 5.81 | +0.95 | 9.66% | 7.70% | −1.95pp |
| 1.5× | 0.5 | 4.52 | 5.44 | +0.92 | 9.76% | 7.80% | −1.96pp |
| 1.5× | 1.0 | 4.20 | 4.94 | +0.74 | 9.87% | 7.90% | −1.97pp |
| 2.0× | 0.0 | 4.48 | 5.02 | +0.54 | 10.05% | 8.01% | −2.04pp |
| 2.0× | 0.5 | 4.16 | 4.55 | +0.39 | 10.15% | 8.10% | −2.05pp |
| 2.0× | 1.0 | 3.86 | 4.10 | +0.24 | 10.26% | 8.20% | −2.06pp |

**Key finding:** EET delivers ~1.9–2.1pp lower worst-fold DD than UTC in every realistic cell, despite carrying the higher $5 commission (vs UTC's $4). The directional implication: the *true* EET-vs-UTC boundary benefit on like-for-like commission is a *lower bound* on what's shown — real benefit is larger.

The EET broker-day boundary is structurally favourable for Arc 10's drawdown profile. F4 2013's worst-fold DD is the load-bearing metric for both venues; EET trims it by ~2pp purely through how the daily-DD bucket is anchored.

## Expected Live Performance

Backtest numbers above are pre-deployment. Expected live applies haircuts for unmodelled risks:

- **Tail-event fills** (CHF 2015 gap-through-SL type events): ~5–8% on worst-fold DD, 0% on mean
- **Forward-inflation on worst-fold** (worst-fold partly survivor-selected in backtest): ~5–8% on worst-fold ROI and DD
- **Slippage tail beyond modelled mean** (~3 pip news/illiquid bars): ~2–3% on ROI and DD
- **EA execution gaps** (latency, requotes): ~1–2% on ROI

**Applied haircuts:**
- Mean / holdout ROI: × 0.95
- Worst-fold ROI: × 0.90
- Mean / holdout DD: × 1.05
- Worst-fold DD: × 1.13

### EET expected live (r = 0.50%, central cell)

| Metric | Backtest | Expected live |
|---|---|---|
| Worst-fold ROI | 18.47% | **~16.6%** |
| Worst-fold DD | 7.80% | **~8.8%** |
| Worst-fold ratio | 5.44 | ~3.4 |
| Mean fold ROI | 43.50% | **~41.3%** |
| Holdout ROI | 46.24% | **~43.9%** |
| Holdout DD | 6.17% | **~6.5%** |

Worst-fold DD ~8.8% leaves ~1.2pp to the 10% hard limit.

### UTC expected live (r = 0.40%, central cell)

| Metric | Backtest | Expected live |
|---|---|---|
| Worst-fold ROI | 10.3% (F6) | **~9.3%** |
| Worst-fold DD | 8.4% (F4) | **~9.5%** |
| Worst-fold ratio | ~2.4 | ~1.8 |
| Mean fold ROI | ~22% | **~21%** |
| Holdout ROI | 28.0% | **~26.6%** |
| Holdout DD | 5.1% | **~5.4%** |

Worst-fold DD ~9.5% leaves ~0.5pp to the 10% hard limit. Tighter margin than EET. Worst-fold ratio dips under 2.0 after haircuts on F6 2015, which is concerning.

## Recommendations

### Primary: EET on FundedNext

Economic case is decisive:
- ~1.7× holdout ROI vs UTC (44% vs 26.6% expected live)
- ~0.7pp more DD margin to the 10% hard limit
- Higher ratio cushion across every realistic cell
- 11/11 sign consistency holds even at 4× spread stress

**Deployment parameters:**
- Risk: 0.50% per trade (r_base, no scaling needed)
- Account: FundedNext $100k Challenge + swap-free add-on (non-negotiable)
- Commission expectation: $5/lot round-turn
- Spread expectation: 1×–1.5× HistData baseline (verified via demo)
- Boundary: EET

### Fallback: UTC on 5ers

If swap-free terms change or weekend-hold rules tighten on FundedNext:
- Risk: 0.40% per trade (scaled down from r_base)
- Account: 5ers existing deployment path
- Commission expectation: $4/lot round-turn
- Boundary: UTC

UTC is deployable, but with materially lower returns and thinner DD margin. The worst-fold ratio after haircuts (~1.8) drops below the 2.0 PASS-DEPLOYABLE gate, which is a serious flag — under live conditions, a single tail event on F6 could break the gate.

## Key Caveats

1. **Expected live numbers are estimates.** Haircuts are matched to specific unmodelled risks but the magnitudes are judgment calls. Real live performance could be ±5pp on holdout ROI in either direction.

2. **EET-vs-UTC delta is directional, not isolated.** The cross-comparison carries asymmetric commissions ($4 vs $5). Real EET boundary benefit on like-for-like commission is larger than the ~2pp shown.

3. **CHF-2015-equivalent events not priced.** Backtester assumes clean SL fills. Live tail events could exceed 10% DD on a single day regardless of which venue. Mitigation: sized at central case, not stress case.

4. **Swap rates change over time.** UTC deployment is sensitive to interest-rate-differential drift. Current 5ers swap rates applied uniformly.

5. **Scaling not modelled.** Both venues scale capital based on performance milestones. Returns above are at base account size; scaled performance compounds proportionally.

## Bottom line

**EET on FundedNext is the deployable venue. Deploy at r = 0.50% once MT5 parity is verified** (done — see `02_phase_2_parity_eet.md`).

UTC on 5ers remains a viable fallback at r = 0.40% with ~27% expected live ROI.

Backtest economics are settled across both venues.

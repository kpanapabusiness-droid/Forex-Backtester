# KH-24 — Honest Re-Measurement on the Canonical Discovery Gate

> **Date:** 2026-06-06 · **Type:** standalone measurement/validation (NOT a discovery arc — no arc-id, no log row, no core change) · **Engine:** `MultiPairBacktester` (sole gate engine) · **Costs:** FundedNext ON · **Boundary:** EET broker-day (`5ers_eet`)
>
> **VERDICT: KILL.** KH-24 fails the discovery all-folds-positive judge on **both** IS (8/10 folds negative) and OOS (4/6 negative). It is **net-negative in-sample** (mean −6.16%/yr) and **does not beat a matched-fire-rate random-entry null in-sample**. Its only positive aggregate (OOS mean +3.49%/yr) is carried entirely by a single year (2024, +25.6%); strip 2024 and OOS mean is −0.9%/yr. The live "edge" was pre-reset replay inflation, not a real edge — exactly the Arc-10 gate-fidelity defect.

---

## ⚠️ Framing — old KH-24 numbers are VOID

KH-24 ran in the pre-reset era whose replay engine inflated results by skipping pre-+1R-partial stops (the Arc-10 defect; see [docs/ARC_10_GATE_FIDELITY_DEFECT.md](../docs/ARC_10_GATE_FIDELITY_DEFECT.md)). The published lineage (worst-fold ROI **+1.92%**, worst-fold DD **6.37%**, 214 trades) is **not trustworthy**. This re-measurement scores KH-24 only on the verified-honest engine; old numbers are not used as a baseline.

The honest result confirms the inflation directly: on the closest comparable window (the 2021–2026 OOS), the honest worst fold is **−4.75%** (2023), not the replay's +1.92%. The replay turned a negative worst fold positive.

---

## STEP 1 — Confirmed canonical system definition

**KH-24 is unambiguous.** The live MQL5 EA and the Python port agree on every signal constant. No conflicting "live" variant exists.

### Source of truth
| Artifact | Path | Role |
|---|---|---|
| Live MQL5 EA | [EA/KH24_EA.mq5](../EA/KH24_EA.mq5) (internally `KGL_V2` v2.01) | The deployed system — source of truth |
| Python signal | [core/strategies/kh24/signal.py](../core/strategies/kh24/signal.py) | C1–C6, C8, C9 evaluator (bid-OHLC, v3 schema) |
| Strategy assembly | [core/strategies/kh24/kh24.py](../core/strategies/kh24/kh24.py) | SL / trail / risk / exposure wiring |
| H1 CIR filter | [core/strategies/kh24/filters/h1_cir.py](../core/strategies/kh24/filters/h1_cir.py) | T=0.28 close-in-range gate |
| kijun_d1 exit | [core/strategies/kh24/exits/kijun_d1.py](../core/strategies/kh24/exits/kijun_d1.py) | D1 baseline-cross exit |
| SignalModule | [core/strategies/kh24/signal_module.py](../core/strategies/kh24/signal_module.py) | A1-protocol wrapper (used by the gate) |
| A1 adapter | [core/strategies/kh24/a1_adapter.py](../core/strategies/kh24/a1_adapter.py) | `kh24_to_a1()` → faithful A1 config |

> Note: [signals/kb_exhaustion_bar.py](../signals/kb_exhaustion_bar.py) is a *different*, simpler "Phase KC" exhaustion bar (C1–C3 only, has a short side). It is **not** KH-24 and was **not** used. KH-24 is the regime-gated `core/strategies/kh24/` assembly.

### Logic (long-only, 28 FX pairs, H4 primary; D1 + H1 auxiliary)
Entry signal fires on bar N close (entry at bar N+1 open) when **all** hold:
- **C1** close < open (bearish exhaustion bar)
- **C2** |close − open| / ATR(14) ≥ 0.5 (substantial body)
- **C3** (close − low) / (high − low) ≤ 0.24 (closed near the low)
- **C4** close > 4H Kijun(26) (above baseline)
- **C5** close ≤ 4H Kijun + 1.0 × ATR (not over-extended)
- **C6** (close − close[N−10]) / ATR ≤ −0.5 (significant 10-bar drop)
- **C7** *DISABLED* (volume gate — permanently eliminated)
- **C8** prev-D1 close > prev-D1 Kijun(26) (D1 regime up, lag-1)
- **C9** prev-D1 close ≤ prev-D1 Kijun + 1.0 × prev-D1 ATR (D1 not over-extended)
- **H1 CIR gate** (close−low)/(high−low) of last completed H1 bar ≤ 0.28

Exit / risk:
- **SL** = entry − 2.0 × ATR(14) (hard stop, frozen; SL-first / take-the-loss)
- **Trail** activates when close ≥ entry + 2.0 × ATR; trails 1.5 × ATR behind the highest close (bar-close updates only)
- **kijun_d1 exit** when prev-D1 close < prev-D1 Kijun
- **Risk** 1.0% of live (compounding) balance per trade
- **Exposure** per-currency cap = 2, per-pair = 1, no total cap

### Python ⇄ EA agreement — confirmed identical
| Constant | EA (`KH24_EA.mq5`) | Python (`KH24SignalParams` / `KH24Config`) |
|---|---|---|
| Kijun period | `KIJUN_PERIOD=26` | `kijun_period=26` ✓ |
| ATR period | `ATR_PERIOD=14` | `atr_period=14` ✓ |
| Body min | `BODY_SIZE_MIN=0.5` | `long_body_threshold=0.5` ✓ |
| Close-pos max | `CLOSE_POS_MAX=0.24` | `long_close_position_max=0.24` ✓ |
| 4H ATR cap (C5) | `ATR_CAP_4H=1.0` | `c5_distance_cap_atr=1.0` ✓ |
| Counter lookback/depth (C6) | `10 / 0.5` | `c6_depth_bars=10 / c6_depth_threshold=0.5` ✓ |
| D1 ATR cap (C9) | `D1_ATR_CAP=1.0` | `c9_d1_distance_cap_atr=1.0` ✓ |
| SL / trail act / trail dist | `2.0 / 2.0 / 1.5` | `2.0 / 2.0 / 1.5` ✓ |
| Risk / exposure | `RiskPercent=1.0 / ExposureCap=2` | `risk_pct=0.01 / per_currency=2` ✓ |
| H1 CIR threshold | `H1CirThreshold=0.28` | `H1CIRParams.threshold=0.28` ✓ |
| Signal | `C1&C2&C3&C4&C5&C6&C8&C9` (C7 off) | identical ✓ |

**One live-only feature is not in the backtest:** the EA's MT5 economic-calendar **news blackout** (`EnableNewsFilter`, ±3 min around high-impact events). It has no backtest analogue (and FOMC-proximity filtering is on the eliminated list). At H4 a ±3-minute intrabar blackout almost never changes a bar-open entry, so the omission is immaterial. Stated, not silently chosen.

---

## STEP 2 — Port onto the honest engine (UNCHANGED)

The port is the canonical, anchor-verified path — **not** a re-implementation:

```python
from core.strategies.kh24.a1_adapter import kh24_to_a1
from core.strategies.kh24.kh24 import KH24Config
a1_cfg, kh24_signal = kh24_to_a1(KH24Config(), config_id="kh24_canonical")
signal_eval = kh24_signal.evaluate(panels)          # panels = {H4, D1, H1}
runner = ArcFoldRunner(A1Architecture(), signal_eval, panels)
```

`kh24_to_a1(KH24Config())` emits exactly the locked params (SL 2.0, trail 2.0/1.5, risk 1%, exposure None/1/2, H1 CIR 0.28, kijun_d1, C1–C6+C8+C9). The **A1-vs-legacy anchor harness** ([scripts/anchor/check_a1_equivalence.py](../scripts/anchor/check_a1_equivalence.py)) proves `ArcFoldRunner(A1, kh24_to_a1(...))` is byte-equivalent to the legacy `KH24FoldRunner` on the 28-pair HistData universe. So this **is** KH-24, run through the discovery-canonical engine; FundedNext costs are netted at `build_fold_stats_from_run` (the single cost chokepoint), and the SL-first take-the-loss invariant is the engine's.

### Port assumptions (stated explicitly; none chosen for favourability)
1. **Boundary = `5ers_eet`** (EET broker-day) per dispatch. The anchor byte-identity was checked under UTC; EET is the current canonical, lookahead-safe convention and is *more* faithful to the live MT5 broker server time. It changes only D1 bucketing / daily-DD measurement — the trade logic is identical.
2. **SL anchor = signal-bar `close_ask`** as the proxy for the next-bar-open fill (the locked v3-era KH-24 convention the anchor was verified against; sub-pip effect on stop placement vs the EA's post-fill SL correction). The entry itself fills at next-bar open via the driver.
3. **FundedNext costs**: 1.5× spread, $5/lot round-trip, 0.5 pip/fill slippage, **no swaps** (FundedNext convention).
4. **Non-USD-quote sizing simplification** in `LiveBalanceRisk` (balance treated as quote-currency-denominated for JPY/cross pairs — inherited PR-E.1 simplification, minor).

### Data / universe
28 FX pairs, H4 + D1 + H1, real HistData bid+ask, 2010-01-03 → 2026-04-10. Total signal fires (pre-gate): **899**.

---

## STEP 3 — Canonical gate results

**Gate:** full-pool WFO — IS folds (`build_v3_folds`, per-year 2010–2020) + frozen OOS holdout (`build_oos_year_folds`, per-year 2021–present), scored solely by `MultiPairBacktester`, judged `judge_all_folds_positive` (all folds ROI > 0 on IS **and** OOS). One-shot OOS — no tuning. ROI is per-fold = per-year (so ≈ annualised already; full-year folds).

### In-Sample (development window, per-year)
| Year | Trades | ROI | maxDD | Positive? |
|---|---:|---:|---:|:---:|
| 2010 *(empty-IS, memo — excluded from IS judge)* | 33 | −12.94% | 14.83% | ✗ |
| 2011 | 25 | −7.89% | 9.74% | ✗ |
| 2012 | 32 | −2.26% | 9.63% | ✗ |
| 2013 | 20 | **+3.56%** | 6.61% | ✓ |
| 2014 | 37 | −10.10% | 12.60% | ✗ |
| 2015 | 38 | −9.35% | 10.16% | ✗ |
| 2016 | 27 | −13.34% | 15.62% | ✗ |
| 2017 | 29 | −10.00% | 17.68% | ✗ |
| 2018 | 35 | −12.04% | 15.45% | ✗ |
| 2019 | 29 | **+4.38%** | 6.37% | ✓ |
| 2020 | 31 | −4.57% | 13.37% | ✗ |
| **IS judge set (2011–2020)** | **303** | **mean −6.16%** · worst **−13.34%** | worst **17.68%** | **8/10 negative → FAIL** |

### Out-of-Sample (frozen holdout, per-year)
| Year | Trades | ROI | maxDD | Positive? |
|---|---:|---:|---:|:---:|
| 2021 | 46 | **+3.99%** | 11.49% | ✓ |
| 2022 | 43 | −1.97% | 12.35% | ✗ |
| 2023 | 25 | −4.75% | 6.90% | ✗ |
| 2024 | 37 | **+25.56%** | 5.41% | ✓ |
| 2025 | 43 | −1.49% | 9.16% | ✗ |
| 2026 *(partial, 151d)* | 8 | −0.38% | 2.79% | ✗ |
| **OOS judge set (2021–2026)** | **202** | **mean +3.49%** · worst **−4.75%** | worst **12.35%** | **4/6 negative → FAIL** |

> The OOS mean is **entirely carried by 2024 (+25.56%)**. Excluding 2024, OOS mean = **−0.92%/yr**. A single-year outperformance is not a robust edge.

### Discovery judge
| Set | All-folds-positive? |
|---|:---:|
| IS (2011–2020) | **NO** (8/10 negative) |
| OOS (2021–2026) | **NO** (4/6 negative) |
| **Discovery PASS (IS ∧ OOS)** | **NO** |

### Deployment-target characterization (review-time, not a discovery gate)
| Target | IS | OOS | Met? |
|---|---:|---:|:---:|
| Worst-fold annualised > +5% | −13.31% | −4.75% | ✗ / ✗ |
| Mean > +8% | −6.16% | +3.49% | ✗ / ✗ |
| Max DD < 8% | 17.68% (worst) | 12.35% (worst) | ✗ / ✗ |

Fails every deployment target on both windows.

### Fair null — random entry, matched fire-rate, same SL/exit/pairs/costs
Two variants (both via the canonical `null_entry_baseline` + the same `ArcFoldRunner`/A1 config): **canonical** (random entry, SL+trail only) and **same-exit** (random entry + KH-24's `kijun_d1` exit re-attached). Nulls trade more than KH-24 because they are matched on raw fire-rate but skip the H1-CIR gate.

| Window | KH-24 (real) | Null (canonical) | Null (same-exit) |
|---|---:|---:|---:|
| **IS mean ROI** | **−6.16%** | −2.62% | −5.61% |
| IS worst fold | −13.34% | −16.66% | −15.50% |
| IS neg folds | 8/10 | 7/10 | 9/10 |
| IS trades | 303 | 530 | 547 |
| **OOS mean ROI** | **+3.49%** | −5.27% | −1.51% |
| OOS worst fold | −4.75% | −12.70% | −9.09% |
| OOS neg folds | 4/6 | 5/6 | 3/6 |
| OOS trades | 202 | 272 | 284 |

**Reading the null (the key honesty check):**
- **In-sample, KH-24 has NO edge over a coin flip.** Its IS mean (−6.16%) is *worse* than the canonical random null (−2.62%) and ≈ the same-exit null (−5.61%). Entry timing carries no positive information in-sample.
- **Out-of-sample, KH-24 beats the null** (+3.49% vs −1.5%…−5.3%), but: (a) the win is driven by the single 2024 fold, (b) part of the gap is lower cost drag (KH-24 trades ~25–30% less than the matched-fire-rate null because the H1-CIR gate removes entries), and (c) it is still not all-folds-positive. Beats-null-but-fails-the-judge on a single fold is not a deployable edge.

---

## Disposition (§11) — **KILL**

- **PASS** requires all-folds-positive IS ∧ OOS → **No** (fails both).
- **PORTFOLIO** requires *mean-positive net of costs* but not all-folds-positive (archetype: arc 1006, +0.69% IS). KH-24's IS mean is **net-negative (−6.16%)**, so it does not clear the portfolio bar. Per §11: *"you cannot diversify net-negative components positive… a real-but-net-negative signal is KILL, not PORTFOLIO."*
- **KILL** — net-negative in-sample, fails the judge on both windows, and no in-sample edge over random entry. The lone positive aggregate (OOS mean) is single-fold-driven and does not survive the IS evidence.

## Straight verdict

**KH-24 is KILL on the honest engine.** Its deployed "edge" was an artifact of the retired replay engine that skipped pre-partial stops, not a real one — confirming the dispatch's prior and the broader honest-era finding that single-pair directional H4 systems on liquid FX net ≈ cost (here, worse). No core change is warranted; nothing about this measurement suggests a deployable system. **Deployable-system count remains 0.**

This is a measurement, not a deployment candidate, so no §11 independent re-verification is triggered (that path is reserved for a *surprise PASS*).

---

### Reproducibility
Faithful port = `A1Architecture` + `kh24_to_a1(KH24Config())`, anchor-verified ≡ legacy `KH24FoldRunner`. Panels: `build_panel_parallel(PAIRS_28, tf, histdata_root=<histdata_backup>, cache_root=<data/cache>, boundary_convention="5ers_eet")` for H4/D1/H1. Folds: `build_v3_folds().folds` (IS) + `build_oos_year_folds(start_year=2021)` (OOS). Nulls: `build_null_signal_evaluation(real_eval, seed=42)`. Scored via `run_config_over_folds` → `judge_all_folds_positive`; FundedNext costs netted in `build_fold_stats_from_run`. Deterministic (`random_state=42`, `n_jobs=1`, null `seed=42`). Run harness lives in the worktree scratch dir `_kh24_remeasure/` (uncommitted); this report embeds all numbers.

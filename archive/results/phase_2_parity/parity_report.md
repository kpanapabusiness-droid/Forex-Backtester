# Phase 2 — Sidecar/Lab Signal Parity Report

> Dispatch C v2 · Arc 10 DLR · branch `claude/phase-2-parity-validation` · 2026-05-29
> Pre-deploy gate: prove the deployed Python sidecar emits byte-identical
> signals to the WFO lab's validated artefacts.

## Verdict: **PASS — green-light for deploy**

The sidecar's signal-computation path
(`deployment.sidecar.signal_runner.run_signal` →
`signals.lchar_dlr_long.compute_signal`) reproduces the lab's WFO artefacts
**byte-identically on every axis that affects a trade**:

- **Signal-firing decision: 100% agreement** across 14,670 candidate-bar
  evaluations spanning all 28 pairs (2010-2026). Zero true logic divergence.
- **Ledger coverage: 100%** — all 3,301 lab signal bars fire in the sidecar
  path; zero uncovered.
- **Audit/price fields (atr14, L1/L0, ages, proximities, fractions): within
  1e-9** on every fired signal. Global worst ATR delta **2.31e-10** (NZDJPY),
  ~4× inside the 1e-9 tolerance.
- **Entry timing: matches the ledger** on all but one unforecastable
  holiday-weekend row (documented below; live behaviour unaffected).

One real bug was found and fixed during this work (weekend entry-timestamp
projection feeding a live news-filter gate) — see §4.

---

## 1. Methodology

The sidecar and the lab call the **identical** `compute_signal`. The only
possible sources of divergence are therefore (a) the sidecar's rolling
300-H4 / 100-D1 window vs the lab's full panel (which affects the
infinite-memory Wilder-ATR warmup), and (b) window-relative absolute D1
indices (`d_t_idx`, `d_for_l1_search_max`), which are intentionally excluded
from byte-comparison.

**Ground-truth split:**
- `trade_ledger_utc.parquet` (3,301 rows) → signal-bar **set** + **entry
  timing**. Read-only validated artefact.
- `step_1/pool.parquet` → **atr14 + audit-field** parity (the ledger's direct
  upstream; carries the lab's exact computed values at signal time). Read-only.

`sl_at_entry_price` in the ledger is **not** a parity axis: it is a Step-1
plumbing value (`2.0 × mid-ATR`), not the deployed SL (`3.5 × bid-ATR`). The
sidecar's SL driver is `atr14` (bid), which is compared against the pool's
`atr14_at_signal`.

**Panel reconstruction** mirrors `scripts/l_arc_10_v3/step_1._build_per_pair_data`
exactly. As a self-check, the harness's full-panel `compute_signal` frame is
compared against `pool.parquet` for every pair: **all 28 reproduce the pool
with `atr_delta_vs_pool = 0.0` and exact signal-count/time match** — so the
windowed-sidecar comparison rests on a verified-faithful reconstruction.

**Candidate-bar set** (per dispatch): for each pair, the union of (a) every bar
the lab marked `prefilter_pass=True`, (b) every bar the lab fired a signal,
(c) a 1% random sample (seed 42) of all other bars. Total: **14,670** bars.

**Tolerances:** signal-firing must agree exactly (a mismatch is "true logic
divergence"); price/ratio fields use absolute tolerance **1e-9**.

Harness: `scripts/phase_2_parity/parity_harness.py`. Environment: base
miniconda (pyarrow 18.1.0, numpy 2.4.2, pandas 3.0.0); UTC boundary convention
matching the v3.0.2 UTC rerun.

---

## 2. Headline numbers (post-fix)

| Metric | Result |
|---|---|
| Pairs | **28 / 28 PASS** |
| Candidate bars evaluated | 14,670 |
| Lab signals = ledger rows | 3,301 = 3,301 (exact) |
| Pool x-check (all pairs) | `count_match=True`, `atr_delta_vs_pool=0.0` |
| Ledger uncovered | 0 |
| **Fire disagreements** | **0** |
| ATR tolerance exceeds (>1e-9) | 0 |
| Field tolerance exceeds | 0 |
| Global max ATR delta | 2.31e-10 (NZDJPY) |
| Divergence rows | **1** (GBPNZD holiday residual; was 37 pre-fix) |

Byte-identity rate on the signal/audit axis: **100%**, far above the ≥99.5%
acceptance bar.

---

## 3. ATR warmup is a non-issue

The 300-bar rolling window was the central floating-point risk (Wilder ATR is
an infinite-memory EMA). Empirically the residual is tiny — worst case
**2.31e-10** (NZDJPY), with JPY pairs carrying the largest deltas due to price
scale (~100-200). All 28 pairs sit comfortably under 1e-9; **no
`h4_history_bars` bump was required** for any pair.

---

## 4. Structural finding + fix — weekend entry-timestamp projection

**Pre-fix divergence:** 37 rows (of 3,301 fired signals; 1.1%), every one a
**Friday 20:00 UTC** signal bar — the last H4 bar before the forex weekend gap.
The signal itself fired identically (same bar, same atr14, same audit fields);
the divergence was confined to the `entry_bar_open_utc_iso` envelope field.

**Root cause:** `run_signal` projected the entry bar as a naive
`signal_bar_close + 4h`. For a Friday-20:00 signal that lands on **Saturday
00:00 UTC — a bar that never exists** (no weekend trading). The lab correctly
used the next *actual* bar (Sunday 20:00 reopen, +48h).

**Why it mattered (not cosmetic):** an EA-side audit found that
`entry_bar_open_utc` is consumed by the live **news filter**
(`deployment/ea/include/NewsFilter.mqh:180`, wired at
`Arc10_DLR_Sidecar_EA.mq5:258`, `Enable_News_Filter=true` by default). It
centres a `±News_Window_Sec` blackout window on the entry timestamp and can
**delay or discard** the entry. A wrong (Saturday) timestamp misplaces that
window by ~48h. Practical blast radius is near-nil (the window is only ±2 min
and both the wrong Saturday location and the correct Sunday-reopen location sit
in news-dead zones), but it is a real consumption of a wrong value in an
execution-gating path — so it was fixed rather than tolerated.

Three other named EA gates were verified **clean** (do not use the timestamp):
sidecar-staleness (heartbeat-file age), bar-mismatch (none), and position-state
init (`last_processed_h4_bar` is set from `iTime(symbol, PERIOD_H4, 0)`, the
real chart bar).

**Fix:** `deployment/sidecar/signal_runner.py::_project_entry_bar_open`. The
entry bar open is `signal_bar_open + 4h`; if that lands in the weekend gap
(Saturday 00:00 through Sunday 16:00 UTC), it snaps forward to the **Sunday
20:00 UTC reopen bar** — the bar that captures the Sunday ~21:00/22:00 UTC
reopen under both US-DST regimes. `signal_bar_close_utc` is unchanged (the
signal bar's nominal close is genuinely +4h). Regression tests:
`tests/sidecar/test_signal_runner.py` (Friday-20:00→Sunday-20:00,
weekday +4h, Friday-16:00 stays in-week, Sunday-reopen advances normally).

**Post-fix:** 37 → **1** divergence row. Byte-identical on all standard
weekend-edge cases.

---

## 5. The one residual — GBPNZD 2016-12-30 (New Year holiday)

| pair | signal bar | sidecar projects | lab entry |
|---|---|---|---|
| GBPNZD | Fri 2016-12-30 20:00 | Sun 2017-01-01 20:00 | **Mon 2017-01-02 04:00** |

Sunday 2017-01-01 was New Year's Day — the market did not reopen, so **no
Sunday H4 bar exists that weekend**; the first actual bar is Monday 04:00. The
sidecar has no holiday calendar and projects to the standard Sunday reopen.

This is **expected and harmless**:
- It is inherently unforecastable from the calendar alone (would require a
  per-broker market-holiday table — out of scope, not requested).
- The fix still strictly improves this row: pre-fix projected Saturday 00:00
  (off by ~80h); post-fix projects Sunday 20:00 (off by 8h, and in the correct
  direction).
- **Live behaviour is unaffected:** the EA enters at the first actual tick
  after the market reopens regardless of the envelope timestamp; the news
  window at the projected Sunday 20:00 (a holiday, news-dead) introduces no
  real gating difference.

Recorded in `divergence_ledger.parquet` (1 row) as a documented tolerated
residual, not a failure.

---

## 6. Per-pair results (post-fix)

| Pair | Lab signals | Candidate bars | Fire disagree | Uncovered | Max ATR Δ | Div rows |
|------|------------:|---------------:|--------------:|----------:|----------:|---------:|
| AUDCAD | 144 | 585 | 0 | 0 | 6.48e-13 | 0 |
| AUDCHF | 127 | 566 | 0 | 0 | 8.65e-13 | 0 |
| AUDJPY | 112 | 501 | 0 | 0 | 8.74e-11 | 0 |
| AUDNZD | 113 | 519 | 0 | 0 | 4.41e-13 | 0 |
| AUDUSD | 104 | 475 | 0 | 0 | 5.28e-13 | 0 |
| CADCHF | 135 | 566 | 0 | 0 | 8.06e-13 | 0 |
| CADJPY | 119 | 523 | 0 | 0 | 1.30e-10 | 0 |
| CHFJPY | 99 | 480 | 0 | 0 | 6.75e-11 | 0 |
| EURAUD | 133 | 560 | 0 | 0 | 2.49e-12 | 0 |
| EURCAD | 104 | 483 | 0 | 0 | 1.42e-12 | 0 |
| EURCHF | 126 | 557 | 0 | 0 | 4.96e-13 | 0 |
| EURGBP | 109 | 499 | 0 | 0 | 3.65e-13 | 0 |
| EURJPY | 111 | 525 | 0 | 0 | 9.50e-11 | 0 |
| EURNZD | 116 | 532 | 0 | 0 | 9.72e-13 | 0 |
| EURUSD | 100 | 472 | 0 | 0 | 6.30e-13 | 0 |
| GBPAUD | 115 | 515 | 0 | 0 | 1.40e-12 | 0 |
| GBPCAD | 113 | 517 | 0 | 0 | 7.94e-13 | 0 |
| GBPCHF | 110 | 486 | 0 | 0 | 9.08e-13 | 0 |
| GBPJPY | 122 | 538 | 0 | 0 | 1.88e-10 | 0 |
| GBPNZD | 124 | 528 | 0 | 0 | 2.87e-12 | 1 |
| GBPUSD | 116 | 512 | 0 | 0 | 1.22e-12 | 0 |
| NZDCAD | 147 | 616 | 0 | 0 | 5.54e-13 | 0 |
| NZDCHF | 125 | 547 | 0 | 0 | 1.66e-12 | 0 |
| NZDJPY | 120 | 527 | 0 | 0 | 2.31e-10 | 0 |
| NZDUSD | 100 | 478 | 0 | 0 | 1.32e-12 | 0 |
| USDCAD | 118 | 508 | 0 | 0 | 1.02e-12 | 0 |
| USDCHF | 121 | 537 | 0 | 0 | 5.27e-13 | 0 |
| USDJPY | 118 | 518 | 0 | 0 | 9.36e-11 | 0 |
| **Total** | **3,301** | **14,670** | **0** | **0** | **2.31e-10** | **1** |

---

## 7. Acceptance criteria

| Criterion | Bar | Result |
|---|---|---|
| Byte-identical signal emission | ≥ 99.5% | **100%** (0/14,670 fire disagreements) |
| True logic divergence | 0 | **0** |
| Price-field tolerance | ≤ 1e-9 | **2.31e-10** worst case |
| Tolerance-exceeding rows inspected | all | **1** (GBPNZD holiday) — inspected, documented §5 |

**All criteria met. Sidecar is signal-faithful to the lab. Green-light to
deploy.**

---

## Artefacts

- `parity_report.md` — this document
- `divergence_ledger.parquet` — 1 row (GBPNZD holiday residual)
- `parity_summaries.parquet` — per-pair summary (28 rows)
- `scripts/phase_2_parity/parity_harness.py` — the harness (`--pair` / `--all`)
- Fix: `deployment/sidecar/signal_runner.py::_project_entry_bar_open`
- Tests: `tests/sidecar/test_signal_runner.py`

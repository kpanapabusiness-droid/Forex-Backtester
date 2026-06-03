# Phase 2 (EET) — Sidecar/Lab Signal Parity Report

> Arc 10 DLR · FundedNext (`5ers_eet`) convention · branch `claude/phase-2-parity-eet` · 2026-05-29
> Pre-deploy gate: prove the Python sidecar, when driven under the EET broker
> convention, emits byte-identical signals to the locked v3.0.2 EET pool.

## Verdict: **PASS — green-light for FundedNext deploy**

The same sidecar signal path validated for 5ers under `utc`
(`deployment.sidecar.signal_runner.run_signal` →
`signals.lchar_dlr_long.compute_signal`), now run under the FundedNext
`5ers_eet` boundary convention, reproduces the locked v3.0.2 EET artefacts
**byte-identically on every axis that affects a trade**:

- **Signal-firing decision: 100% agreement** across 14,385 candidate-bar
  evaluations spanning all 28 pairs. Zero true logic divergence.
- **Ledger coverage: 100%** — all 3,152 lab signal bars fire in the sidecar
  path; zero uncovered.
- **Audit/price fields (atr14, L1/L0, ages, proximities, fractions): within
  1e-9** on every fired signal. Global worst ATR delta **1.91e-10** (NZDJPY),
  ~5× inside the 1e-9 tolerance.
- **Pool x-check: all 28 pairs** reproduce the locked pool with
  `count_match=True` and `atr_delta_vs_pool=0.0`.
- **Entry timing: matches the EET ledger** on all but four unforecastable
  calendar rows (1 holiday + 3 US/EU DST-mismatch weeks; documented §5). Each
  is byte-identical in signal logic and timing-only; live behaviour unaffected.

No new bug was found in this arc. The four sidecar layers were made
broker-convention-aware (§4) so the *same* code services both 5ers (`utc`) and
FundedNext (`5ers_eet`) from the `boundary_convention` flag.

---

## 1. What changed for the EET convention

Phase 2 (`utc`) already proved the sidecar signal-faithful to the 5ers UTC
rerun. FundedNext runs an **EET/EEST broker server**, so two things differ:

1. **H4/D1 bar boundaries** are anchored on Europe/Athens local
   00/04/08/12/16/20 (stored as true-UTC instants: winter 22/02/06/10/14/18,
   summer 21/01/05/09/13/17). This is the lab's `boundary_convention="5ers_eet"`
   aggregation, byte-identical to the locked v3.0.2 EET pool.
2. **Broker wall-clock → UTC normalisation** in the live fetcher: MT5's
   `copy_rates_from_pos` reports the broker server wall clock; for an EET server
   that must be reinterpreted as Europe/Athens and converted to true UTC.

`compute_signal` itself is convention-agnostic. The only divergence axes are
therefore (a) the rolling-window vs full-panel Wilder-ATR warmup, and (b) the
entry-timestamp projection across the EET-local weekend/DST calendar.

**Ground-truth split (EET):**
- `results/l_arc_10_v3.0.2/step_1/pool.parquet` (locked) → signal-bar **set**,
  **atr14 + audit-field** parity, and **entry timing**. Read-only.
- `trade_ledger_eet.parquet` (3,152 rows × 28 pairs) → a faithful projection of
  the locked pool's `pair / signal_bar_time / entry_time` (built by
  `scripts/phase_2_parity/build_eet_ledger.py`; no recomputation, locked
  artefact untouched).

---

## 2. Headline numbers

| Metric | Result |
|---|---|
| Convention | `5ers_eet` (FundedNext) |
| Pairs | **28 / 28 PASS** |
| Candidate bars evaluated | 14,385 |
| Lab signals = ledger rows | 3,152 = 3,152 (exact) |
| Pool x-check (all pairs) | `count_match=True`, `atr_delta_vs_pool=0.0` |
| Ledger uncovered | 0 |
| **Fire disagreements** | **0** |
| ATR tolerance exceeds (>1e-9) | 0 |
| Field tolerance exceeds | 0 |
| Global max ATR delta | 1.91e-10 (NZDJPY) |
| Divergence rows | **4** (all timing-only advisory residuals, §5) |

Byte-identity rate on the signal/audit axis: **100%**, far above the ≥99.5%
acceptance bar.

---

## 3. ATR warmup is a non-issue under EET

The rolling-window Wilder-ATR residual is just as tiny under EET aggregation as
under UTC — worst case **1.91e-10** (NZDJPY), with JPY pairs carrying the
largest deltas due to price scale. All 28 pairs sit comfortably under 1e-9; no
`h4_history_bars` bump was required for any pair.

---

## 4. The four convention-aware sidecar layers

The dispatch's core deliverable: the sidecar branches on
`boundary_convention` (`utc` | `5ers_eet`) rather than hardcoding UTC. New
central module `deployment/sidecar/boundary.py` owns the convention logic; the
four layers delegate to it.

1. **Entry-bar projection** (`signal_runner._project_entry_bar_open` →
   `boundary.project_entry_bar_open`). EET projects to the next EET-local
   trading-day open, DST resolved by `zoneinfo.ZoneInfo("Europe/Athens")`
   (stdlib, no new dep). Weekend snap is DST-aware: winter the last tradeable
   bar is EET-local Fri 20:00; summer EET-local Sat 00:00 is tradeable; reopen
   is EET-local Mon 00:00 both regimes (Sun 22:00 UTC winter / Sun 21:00 UTC
   summer). UTC path unchanged.
2. **H4-close schedule** (`compute_next_utc_h4_close` → **renamed**
   `compute_next_h4_close` → `boundary.next_h4_close`). EET-anchored vs
   UTC-anchored next-close.
3. **MT5 alignment probe** (`verify_mt5_h4_alignment`). Anchor check against
   the correct grid per convention, **plus** a broker-offset sanity check
   (`_verify_broker_offset`) that compares the observed broker wall-clock offset
   against the expected EET/EEST calendar offset — catching a broker server that
   mishandles EU DST (emits on-grid bars at the wrong absolute offset).
4. **Bar fetcher** (`mt5_data_fetcher._bar_time_to_utc_naive`). UTC passes the
   epoch through; EET localises the broker wall clock to Europe/Athens and
   converts to true UTC — byte-identical to the lab's EET aggregator (anchored
   on the same IANA tz, not a dynamically-probed offset that could be DST
   off-by-one).

`config.py` carries the `boundary_convention` field (default `utc`, validated
against `SUPPORTED_CONVENTIONS`); `sidecar.py` threads it through
`_process_pair` / `main_loop` / `initialize_and_run`.

**Regression tests** exercise *both* paths, including ≥1 DST spring-forward and
≥1 fall-back in the EET path:
- `tests/sidecar/test_boundary.py` — UTC + EET projection (spring-forward
  ground-truth GBPJPY 2019-03-29; fall-back 2024-10-25; summer Sat tradeable /
  winter Fri-last snap; winter pool ground-truth 2013-01-11), next-close
  winter/summer/across-spring, anchor checks, expected offsets, unsupported
  convention raises.
- `tests/sidecar/test_mt5_data_fetcher.py` — UTC pass-through, EET winter
  (Mon 00:00 EET → Sun 22:00 UTC), EET summer (12:00 EEST → 09:00 UTC),
  per-bar DST resolution across spring-forward.
- `tests/sidecar/test_sidecar_loop.py` — EET anchor accept/reject, broker-offset
  winter pass + server-stuck-on-UTC reject.

The pre-existing UTC Phase 2 parity remains byte-identical (GBPJPY UTC
regression re-run: 122 signals, 0 divergence — unchanged).

---

## 5. The four residuals — all timing-only, advisory

Every divergence row is **`category=byte_identical`** with
`lab_fires=sidecar_fires=True` and `in_ledger=True`: the signal logic and all
audit fields match exactly. Only the projected *entry timestamp* differs from
the ledger, because the sidecar projects from the calendar alone and has no
per-broker holiday table nor cross-region DST-mismatch model.

| pair | signal bar (UTC) | sidecar projects | lab entry | Δ | cause |
|---|---|---|---|---:|---|
| EURAUD | Thu 2015-12-24 18:00 | Fri 2015-12-25 22:00 | Mon 2015-12-28 00:00 (22:00 UTC) | +72h late at ledger | **Christmas** market closure |
| AUDJPY | Fri 2024-03-08 18:00 | Sun 2024-03-10 22:00 | Sun 2024-03-10 18:00 | +4h | US/EU DST-mismatch week |
| EURCAD | Fri 2020-03-20 18:00 | Sun 2020-03-22 22:00 | Sun 2020-03-22 18:00 | +4h | US/EU DST-mismatch week |
| EURNZD | Fri 2023-03-17 18:00 | Sun 2023-03-19 22:00 | Sun 2023-03-19 18:00 | +4h | US/EU DST-mismatch week |

**Christmas (EURAUD 2015-12-24):** the market did not reopen on the standard
schedule; the lab's first actual bar is the following Monday EET-local open. The
sidecar, with no holiday calendar, projects the standard next-day open. Same
class as the UTC arc's single New-Year residual.

**US/EU DST-mismatch weeks (the three +4h rows):** in mid-March the US has
sprung forward to EDT while the EU is still on EET; the forex week reopens early
(US-driven Sunday 18:00 UTC) one bar ahead of the EET-local Monday 00:00 open
(Sun 22:00 UTC) the sidecar projects. The signal and all audit fields are
identical; only the entry-bar envelope is one H4 bar early in the ledger.

These are **expected and harmless**:
- Inherently unforecastable from the EET calendar alone (would require a
  per-broker holiday table + cross-region DST model — out of scope, not
  requested).
- **Live behaviour is unaffected:** the EA enters at the first actual tick after
  the market reopens regardless of the envelope timestamp; the news-filter
  window centred on the projected timestamp sits in a news-dead reopen zone in
  every case, so there is no real gating difference.

Recorded in `divergence_ledger_eet.parquet` (4 rows) as documented tolerated
residuals, not failures.

---

## 6. Per-pair results

| Pair | Lab signals | Candidate bars | Fire disagree | Uncovered | Max ATR Δ | Div rows |
|------|------------:|---------------:|--------------:|----------:|----------:|---------:|
| AUDCAD | 131 | 555 | 0 | 0 | 7.58e-13 | 0 |
| AUDCHF | 131 | 559 | 0 | 0 | 8.18e-13 | 0 |
| AUDJPY | 101 | 461 | 0 | 0 | 8.54e-11 | 1 |
| AUDNZD | 107 | 508 | 0 | 0 | 5.91e-13 | 0 |
| AUDUSD | 112 | 489 | 0 | 0 | 4.83e-13 | 0 |
| CADCHF | 133 | 558 | 0 | 0 | 7.30e-13 | 0 |
| CADJPY | 110 | 505 | 0 | 0 | 1.31e-10 | 0 |
| CHFJPY | 117 | 506 | 0 | 0 | 1.06e-10 | 0 |
| EURAUD | 118 | 534 | 0 | 0 | 2.16e-12 | 1 |
| EURCAD | 111 | 513 | 0 | 0 | 1.32e-12 | 1 |
| EURCHF | 124 | 570 | 0 | 0 | 1.31e-12 | 0 |
| EURGBP | 118 | 527 | 0 | 0 | 1.79e-12 | 0 |
| EURJPY | 109 | 497 | 0 | 0 | 1.18e-10 | 0 |
| EURNZD | 113 | 536 | 0 | 0 | 1.43e-12 | 1 |
| EURUSD | 93 | 442 | 0 | 0 | 6.18e-13 | 0 |
| GBPAUD | 107 | 503 | 0 | 0 | 1.08e-12 | 0 |
| GBPCAD | 125 | 551 | 0 | 0 | 2.49e-12 | 0 |
| GBPCHF | 109 | 505 | 0 | 0 | 8.21e-13 | 0 |
| GBPJPY | 100 | 515 | 0 | 0 | 1.42e-10 | 0 |
| GBPNZD | 104 | 505 | 0 | 0 | 8.72e-13 | 0 |
| GBPUSD | 101 | 466 | 0 | 0 | 1.14e-12 | 0 |
| NZDCAD | 133 | 592 | 0 | 0 | 1.06e-12 | 0 |
| NZDCHF | 123 | 548 | 0 | 0 | 1.70e-12 | 0 |
| NZDJPY | 112 | 508 | 0 | 0 | 1.91e-10 | 0 |
| NZDUSD | 91 | 456 | 0 | 0 | 1.20e-12 | 0 |
| USDCAD | 104 | 495 | 0 | 0 | 1.33e-12 | 0 |
| USDCHF | 109 | 491 | 0 | 0 | 5.78e-13 | 0 |
| USDJPY | 106 | 490 | 0 | 0 | 8.24e-11 | 0 |
| **Total** | **3,152** | **14,385** | **0** | **0** | **1.91e-10** | **4** |

---

## 7. Acceptance criteria

| Criterion | Bar | Result |
|---|---|---|
| Byte-identical signal emission | ≥ 99.5% | **100%** (0/14,385 fire disagreements) |
| True logic divergence | 0 | **0** |
| Price-field tolerance | ≤ 1e-9 | **1.91e-10** worst case |
| Pool x-check (all pairs) | count + atr=0 | **28/28** `count_match=True`, `atr_delta=0.0` |
| Tolerance-exceeding rows inspected | all | **4** (timing-only) — inspected, documented §5 |
| UTC Phase 2 parity still byte-identical | unchanged | **PASS** (GBPJPY 122 signals, 0 div) |

**All criteria met. The sidecar is signal-faithful to the locked v3.0.2 EET
pool under the FundedNext `5ers_eet` convention. Green-light to deploy on
FundedNext.**

---

## Artefacts

- `parity_report.md` — this document
- `divergence_ledger_eet.parquet` — 4 rows (timing-only advisory residuals)
- `parity_summaries.parquet` — per-pair summary (28 rows)
- `trade_ledger_eet.parquet` — EET ground-truth ledger (3,152 rows × 28 pairs)
- `scripts/phase_2_parity/build_eet_ledger.py` — pool → EET ledger projection
- `scripts/phase_2_parity/parity_harness.py` — the harness (`--convention 5ers_eet`)
- Convention module: `deployment/sidecar/boundary.py`
- Tests: `tests/sidecar/test_boundary.py`, `test_mt5_data_fetcher.py`,
  `test_sidecar_loop.py`, `test_h4_schedule.py`

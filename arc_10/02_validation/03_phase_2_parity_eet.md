# Phase 2 — EET Parity Validation

> **Verdict:** PASS. Sidecar↔lab signal output byte-identical on EET convention.
> **Source artifact:** `results/phase_2_parity_eet/parity_report.md`
> **Git tag:** `arc-10-eet-parity-validated`

## What this proves

The Python sidecar — when pointed at FundedNext (or EET-aggregated historical data) and run under `5ers_eet` convention — emits signal envelopes byte-identical to the corresponding entries in the locked v3.0.2 EET pool.

This is the FundedNext deployment-readiness gate. Confirms the strategy validated under EET in the WFO is the same strategy that runs live.

## Headline numbers

| Metric | Result |
|---|---|
| Convention | `5ers_eet` (FundedNext) |
| Pairs | 28 / 28 PASS |
| Candidate bars evaluated | 14,385 |
| Lab signals = ledger rows | 3,152 = 3,152 (exact) |
| Pool x-check (all pairs) | `count_match=True`, `atr_delta_vs_pool=0.0` |
| Ledger uncovered | 0 |
| **Fire disagreements** | **0** |
| ATR tolerance exceeds (>1e-9) | 0 |
| Field tolerance exceeds | 0 |
| Global max ATR delta | 1.91e-10 |
| Divergence rows | **4** (all timing-only advisory residuals) |

**Byte-identity rate on the signal/audit axis: 100%**, far above the ≥99.5% acceptance bar.

## What changed for the EET convention

Phase 2 UTC proved the sidecar signal-faithful to the 5ers UTC re-validation. FundedNext runs an EET broker server, so two things differ:

1. **H4/D1 bar boundaries** are anchored on Europe/Athens local 00/04/08/12/16/20 (stored as true-UTC instants: winter 22/02/06/10/14/18, summer 21/01/05/09/13/17). This is the lab's `boundary_convention="5ers_eet"` aggregation.

2. **Broker wall-clock → UTC normalisation** in the live fetcher: MT5's `copy_rates_from_pos` reports the broker server wall clock; for an EET server this must be reinterpreted as Europe/Athens and converted to true UTC.

The signal logic itself (`compute_signal`) is convention-agnostic. Only the H4/D1 panel construction differs.

## The four sidecar layers made broker-convention-aware

The dispatch's core deliverable was making the sidecar configurable rather than UTC-hardcoded:

1. **Entry-bar projection** (`signal_runner._project_entry_bar_open`): EET projects to next EET-local trading-day open via `zoneinfo.ZoneInfo("Europe/Athens")`. DST resolved via IANA tz database. Weekend snap is DST-aware.

2. **H4-close schedule** (`compute_next_h4_close`): wakes on EET-anchored vs UTC-anchored next-close depending on convention.

3. **MT5 alignment probe** (`verify_mt5_h4_alignment`): anchor check against the correct grid per convention. Note: server-clock offset check was REMOVED in PR #230 because 5ers has EET server clock but UTC-anchored bars — server offset and bar anchor are independent attributes. Anchor probe is the authoritative gate.

4. **Bar fetcher** (`mt5_data_fetcher._bar_time_to_utc_naive`): UTC passes the epoch through; EET localises the broker wall clock to Europe/Athens and converts to true UTC.

Central module: `deployment/sidecar/boundary.py` owns the convention logic. Sidecar layers delegate to it.

## The four divergence rows — all timing-only, advisory

Every divergence row has `category=byte_identical` with `lab_fires=sidecar_fires=True` and `in_ledger=True`. Signal logic and audit fields match exactly. Only the projected entry timestamp differs from the ledger, because the sidecar projects from the calendar alone and has no per-broker holiday table nor cross-region DST-mismatch model.

| Pair | Signal bar (UTC) | Sidecar projects | Lab entry | Δ | Cause |
|---|---|---|---|---|---|
| EURAUD | Thu 2015-12-24 18:00 | Fri 2015-12-25 22:00 | Mon 2015-12-28 22:00 | +72h late at ledger | **Christmas** market closure |
| AUDJPY | Fri 2024-03-08 18:00 | Sun 2024-03-10 22:00 | Sun 2024-03-10 18:00 | +4h | US/EU DST-mismatch week |
| EURCAD | Fri 2020-03-20 18:00 | Sun 2020-03-22 22:00 | Sun 2020-03-22 18:00 | +4h | US/EU DST-mismatch week |
| EURNZD | Fri 2023-03-17 18:00 | Sun 2023-03-19 22:00 | Sun 2023-03-19 18:00 | +4h | US/EU DST-mismatch week |

**Christmas (EURAUD 2015-12-24):** the market did not reopen on the standard schedule; the lab's first actual bar is the following Monday EET-local open. The sidecar, with no holiday calendar, projects the standard next-day open. Same class as the UTC arc's single New-Year residual.

**US/EU DST-mismatch weeks (the three +4h rows):** in mid-March the US has sprung forward to EDT while the EU is still on EET; the forex week reopens early (US-driven Sunday 18:00 UTC) one bar ahead of the EET-local Monday 00:00 open (Sun 22:00 UTC). The signal and all audit fields are identical; only the entry-bar envelope is one H4 bar early in the ledger.

**Live behaviour is unaffected:** The EA enters at the first actual tick after the market reopens regardless of the envelope timestamp; the news-filter window centered on the projected timestamp sits in a news-dead reopen zone in every case, so there is no real gating difference.

Recorded in `divergence_ledger_eet.parquet` (4 rows) as documented tolerated residuals, not failures.

## Per-pair results (excerpt)

Full table in `results/phase_2_parity_eet/parity_report.md`. All 28 pairs PASS with zero fire disagreements and zero field tolerance exceeds.

Notable per-pair max ATR deltas:
- NZDJPY: 1.91e-10 (the global max — JPY pairs have largest deltas due to price scale)
- GBPJPY: 1.42e-10
- CADJPY: 1.31e-10
- All others: < 1e-10

## What was found and FIXED during this dispatch

Bug from PR #227 (introduced by the EET parity work):

**Broker UTC-offset sanity check incorrectly conflated server clock with bar anchor convention.** New `_verify_broker_offset` function refused to start on 5ers (which has EET server clock + UTC bar anchors) when running with `convention=utc`.

**Fix:** PR #230 removed `_verify_broker_offset` entirely. The `verify_mt5_h4_alignment` function (which validates bar anchors, not server clocks) is the authoritative gate.

**Tradeoff:** convention selection is now purely the operator's responsibility — pointing UTC config at an EET-anchored broker (or vice versa) would silently corrupt signal computation. Mitigation: each broker's deployment uses a fixed config with a fixed config_hash that the EA verifies on init. Operator error caught at EA init time via hash mismatch.

## Source artifacts

| Artifact | Path |
|---|---|
| Full parity report | `results/phase_2_parity_eet/parity_report.md` |
| Divergence ledger | `results/phase_2_parity_eet/divergence_ledger_eet.parquet` |
| EET pool (lab ground truth) | `results/l_arc_10_v3.0.2/step_1/pool.parquet` |
| EET trade ledger | `results/phase_2_parity_eet/trade_ledger_eet.parquet` |
| Sidecar code (convention-aware) | `deployment/sidecar/boundary.py` |
| Tests (both conventions) | `tests/sidecar/test_boundary.py`, `test_mt5_data_fetcher.py`, `test_sidecar_loop.py`, `test_h4_schedule.py` |

## Cross-check

The UTC parity (`02_phase_2_parity_utc.md`) was re-run after the EET work to verify no regression. UTC GBPJPY regression: 122 signals, 0 divergence — unchanged. Both conventions work, independently and byte-identically.

# Arc 2 redo — Step 1 plumbing summary

Protocol: `L_ARC_PROTOCOL.md` v2.0 §5
Signal:   `TRIAL__mtf_alignment__2_down_mixed__kijun__h_120` (LCHAR_TOPN_REGISTRY.md Entry 2)

## Disposition
**Step 1 disposition: PASS**

| Gate | Result |
|---|---|
| Deterministic (two-run byte-identical) | PASS |
| No lookahead violations | PASS |
| Spread treatment matches SPREAD_SEMANTICS_LOCK | PASS |
| Pool size ≥ 500 | PASS (12262) |

## Period coverage

Per-pair 1H data coverage (raw):

| Pair | First bar | Last bar |
|---|---|---|
| AUD_CAD | 2010-03-22T02:00:00 | 2025-12-31T23:00:00 |
| AUD_CHF | 2010-03-12T01:00:00 | 2025-12-31T23:00:00 |
| AUD_JPY | 2010-03-12T17:00:00 | 2025-12-31T23:00:00 |
| AUD_NZD | 2010-03-11T08:00:00 | 2025-12-31T23:00:00 |
| AUD_USD | 2010-03-12T16:00:00 | 2025-12-31T23:00:00 |
| CAD_CHF | 2010-03-15T03:00:00 | 2025-12-31T23:00:00 |
| CAD_JPY | 2010-03-10T16:00:00 | 2025-12-31T23:00:00 |
| CHF_JPY | 2010-03-11T19:00:00 | 2025-12-31T23:00:00 |
| EUR_AUD | 2010-03-15T02:00:00 | 2025-12-31T23:00:00 |
| EUR_CAD | 2010-03-12T21:00:00 | 2025-12-31T23:00:00 |
| EUR_CHF | 2010-02-23T20:00:00 | 2025-12-31T23:00:00 |
| EUR_GBP | 2010-03-15T03:00:00 | 2025-12-31T23:00:00 |
| EUR_JPY | 2010-03-12T19:00:00 | 2025-12-31T23:00:00 |
| EUR_NZD | 2010-03-10T09:00:00 | 2025-12-31T23:00:00 |
| EUR_USD | 2010-03-12T22:00:00 | 2025-12-31T23:00:00 |
| GBP_AUD | 2010-02-24T11:00:00 | 2025-12-31T23:00:00 |
| GBP_CAD | 2010-03-02T10:00:00 | 2025-12-31T23:00:00 |
| GBP_CHF | 2010-03-11T19:00:00 | 2025-12-31T23:00:00 |
| GBP_JPY | 2010-03-11T22:00:00 | 2025-12-31T23:00:00 |
| GBP_NZD | 2010-05-10T18:00:00 | 2025-12-31T23:00:00 |
| GBP_USD | 2010-03-12T19:00:00 | 2025-12-31T23:00:00 |
| NZD_CAD | 2010-01-15T01:00:00 | 2025-12-31T23:00:00 |
| NZD_CHF | 2010-02-12T18:00:00 | 2025-12-31T23:00:00 |
| NZD_JPY | 2010-03-01T14:00:00 | 2025-12-31T23:00:00 |
| NZD_USD | 2010-03-12T06:00:00 | 2025-12-31T23:00:00 |
| USD_CAD | 2010-03-12T22:00:00 | 2025-12-31T23:00:00 |
| USD_CHF | 2010-03-12T03:00:00 | 2025-12-31T23:00:00 |
| USD_JPY | 2010-03-15T08:00:00 | 2025-12-31T23:00:00 |

Intersection across pairs: 2010-05-10T18:00:00 → 2025-12-31T23:00:00 (15.64 years)

Trade-pool window (signal_time min → exit_time max): 2010-02-10T10:00:00 → 2025-12-19T11:00:00

## Pool size by pair

| Pair | Trades | Note |
|---|---:|---|
| EUR_CHF | 534 |  |
| CAD_CHF | 499 |  |
| NZD_CHF | 487 |  |
| AUD_CHF | 474 |  |
| EUR_AUD | 469 |  |
| AUD_NZD | 467 |  |
| EUR_GBP | 465 |  |
| AUD_USD | 463 |  |
| EUR_CAD | 461 |  |
| GBP_CHF | 460 |  |
| USD_CHF | 453 |  |
| EUR_NZD | 450 |  |
| EUR_USD | 448 |  |
| NZD_CAD | 448 |  |
| NZD_USD | 443 |  |
| GBP_CAD | 442 |  |
| GBP_AUD | 438 |  |
| AUD_CAD | 436 |  |
| GBP_USD | 431 |  |
| AUD_JPY | 418 |  |
| USD_CAD | 406 |  |
| GBP_NZD | 403 |  |
| USD_JPY | 397 |  |
| CAD_JPY | 388 |  |
| EUR_JPY | 377 |  |
| GBP_JPY | 374 |  |
| NZD_JPY | 372 |  |
| CHF_JPY | 359 |  |
| **Total** | **12262** | |

No pairs flagged with n < 30.

## Gate 1 — Determinism (v2.0 §5)

Two-run byte-identical hashes (sha256):

- `trades_all.csv`
  - run 1: `4f89efd42cf5a0d0b96c43f00f738d9615f1d73ce9f827def2792f9141039888`
  - run 2: `4f89efd42cf5a0d0b96c43f00f738d9615f1d73ce9f827def2792f9141039888`
  - match: PASS
- `trades_paths.csv`
  - run 1: `ddb709befb98ace8804003ac45679bcde173108398516477684f260c0b62cf2f`
  - run 2: `ddb709befb98ace8804003ac45679bcde173108398516477684f260c0b62cf2f`
  - match: PASS

**Gate 1: PASS**

## Gate 2 — No lookahead violations

Audit invariants:
1. Signal at bar N close uses only bars ≤ N (kijun_sign computed per-TF from rolling windows ending at N).
2. Entry uses bar N+1 open (`entry_idx = sig_idx + 1`).
3. SL distance uses ATR(14) at bar N (`atr[sig_idx]`), not bar N+1.
4. D1 alignment uses one-day lag (`mr_d1 = floor('D', T_N) − 1`) — runtime asserted at every firing bar via `compute_signal_mask`.
5. 4H alignment uses prior-completed 4H (`mr_4h = floor('4h', T_N) − 1`) — runtime asserted.

**Runtime assertion**: `scripts/arc_2_redo/signal.py::compute_signal_mask` raises `RuntimeError` if any firing bar references a future 4H or D1 bar. The pool build completed without raising, so all firing bars satisfy the invariant by construction.

Spot-check (first 10 trades by trade_id): PASS
- No mismatches found.

**Gate 2: PASS**

## Gate 3 — Spread treatment (SPREAD_SEMANTICS_LOCK)

Spread floor source: `C:\Users\panap\Documents\Forex-Backtester\configs\spread_floors_5ers.yaml` (sha256 `a613b4ce641c8d5218490531770a4924204029dedaa80fb24111beb61bd15547` — match against expected = PASS)
Points-per-pip: 10.0

Conventions verified:
- Entry fill (long) = `open_mid(N+1) + S(N+1)/2`; `spread_pips_used` = max(raw, floor) of bar N+1.
- Stop-out fill (long) = `sl_price − S(k)/2`; `spread_pips_exit` = bar k's spread for intrabar SL trigger.
- Time-exit fill (long) = `open_mid(N+121) − S(N+121)/2`; `spread_pips_exit` = bar N+121's spread.

Spot-check (first 5 trades): re-derived entry spread, exit spread, and both fill prices from raw bar 'spread' column and the floor table → result: PASS
- All 5 spot-checks reproduce the CSV-recorded values exactly.

**Gate 3: PASS**

## Gate 4 — Pool size

- Pool size: **12262**
- Threshold: ≥ 500
- **Gate 4: PASS**

## File sizes and row counts

- `trades_all.csv` — 12262 rows (+1 header), 2,044,036 bytes
- `trades_paths.csv` — 2,955,142 rows (+1 header), 310,307,365 bytes; expected = pool_size × 241 = 2,955,142 (matches)

## Notes / anomalies

- Signals whose 240-bar forward window (bar_offset 240) exceeds the data tail are excluded so every trade has a uniform 241-row path. This is the only filter applied beyond signal definition + execution rules (per v2.0 §5 'no filtering, no analysis').
- Concurrent-per-pair guard: max 1 open position per pair; subsequent signals that fire while a trade is open on the same pair are dropped (matches `CLAUDE.md` L arc config exposure cap).
